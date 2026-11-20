"""
公共基础设施：统一网格、分块栅格 I/O、掩膜、统计、日志。

设计要点
--------
1. 所有阶段产物一律对齐到「基准网格」—— 22313 x 15265, 30 m,
   Krasovsky_1940_Albers, transform = (30, 0, -815998.125, 0, -30, 3876143.0)。
   这是十阶段产物能互相叠加的前提。
2. 全网格 3.4 亿像元，单幅 float32 约 1.3 GB，因此一律**分块**处理，
   块大小按内存预算自适应。
3. 各家 NoData 不统一（-3.4e38 / +3.4e38 / -1.0 / 0），统一在 to_nan() 里归一。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window

# ---------------------------------------------------------------- 路径常量
def _find_pkg_root() -> Path:
    """从 repro/ 向上找复现包根（含 07_绘图工程 与 01_原始数据 的目录）。"""
    here = Path(__file__).resolve().parent          # .../06_文本/repro
    for up in [here] + list(here.parents):
        if (up / "07_绘图工程").exists() and (up / "01_原始数据").exists():
            return up
    return here


_PKG = _find_pkg_root()

#: 旧盘 L:\实验数据 下的子路径 → 复现包内相对路径（按最长前缀优先匹配）。
#: 使复现内核脱离开发机后仍能定位数据（换机兼容）。
_L2J: list[tuple[str, str]] = [
    (r"02_遥感与基础地理\绝对基准栅格_初始层", r"01_原始数据\05_基础地理"),
    (r"02_遥感与基础地理\绝对基准栅格", r"01_原始数据\05_基础地理"),
    (r"02_遥感与基础地理\YRS_县级区划", r"01_原始数据\05_基础地理\县级区划"),
    (r"02_遥感与基础地理\三省行政界限", r"01_原始数据\05_基础地理\三省界限"),
    (r"02_遥感与基础地理\YRS_渔网", r"01_原始数据\05_基础地理\渔网"),
    (r"02_遥感与基础地理\LULC_CLCD", r"03_过程数据\300m工作数据\LULC_CLCD"),
    (r"02_遥感与基础地理\LULC_Carbon", r"03_过程数据\300m工作数据\LULC_Carbon"),
    (r"02_遥感与基础地理\Pre_year", r"03_过程数据\300m工作数据\Pre_year"),
    (r"02_遥感与基础地理\PAWC", r"03_过程数据\300m工作数据"),
    (r"02_遥感与基础地理\Depth_to_bedrock", r"03_过程数据\300m工作数据"),
    (r"02_遥感与基础地理\FVC", r"03_过程数据\300m工作数据\Factor\X7植被覆盖度"),
    (r"02_遥感与基础地理\Soil_HWSD", r"08_L盘补充数据\实验数据\02_遥感与基础地理\Soil_HWSD"),
    (r"02_遥感与基础地理\Soil AP", r"08_L盘补充数据\实验数据\02_遥感与基础地理\Soil AP"),
    (r"03_RUSLE因子\R_factor", r"02_RUSLE因子\R_降雨侵蚀力"),
    (r"03_RUSLE因子\C_factor", r"02_RUSLE因子\C_植被管理"),
    (r"03_RUSLE因子\P_factor", r"02_RUSLE因子\P_水土保持措施"),
    (r"03_RUSLE因子\K_factor", r"02_RUSLE因子\K_土壤可蚀性"),
    (r"03_RUSLE因子\LS_factor", r"02_RUSLE因子\LS_地形"),
    (r"03_RUSLE因子", r"02_RUSLE因子"),
    (r"04_模型输出与结果\Threats Files", r"03_过程数据\300m工作数据\Threats Files"),
    (r"04_模型输出与结果\Eva_year", r"03_过程数据\300m工作数据\Eva_year"),
    (r"04_模型输出与结果\Velocity", r"03_过程数据\300m工作数据\Velocity"),
    (r"04_模型输出与结果", r"08_L盘补充数据\实验数据\04_模型输出与结果"),
    (r"05_分析与统计\Factor_地理探测因子", r"08_L盘补充数据\实验数据\05_分析与统计\Factor_地理探测因子"),
    (r"05_分析与统计\趋势与Hurst", r"08_L盘补充数据\实验数据\05_分析与统计\趋势与Hurst"),
    (r"05_分析与统计", r"08_L盘补充数据\实验数据\05_分析与统计"),
]


class _DataPath:
    """把「旧盘相对路径」解析到「复现包内位置」的可拼接路径。

    用法保持 `C.DATA / "子路径"` 不变，`__truediv__` 内部做 L→J 映射。
    """

    __slots__ = ("_lprefix",)

    def __init__(self, lprefix: str = ""):
        self._lprefix = lprefix.replace("/", "\\").strip("\\")

    def __truediv__(self, other) -> Path:
        rel = str(other).replace("/", "\\").lstrip("\\")
        full = (self._lprefix + "\\" + rel).strip("\\") if self._lprefix else rel
        fl = full.lower()
        for lpref, jpref in _L2J:
            if fl.startswith(lpref.lower()):
                rest = full[len(lpref):].lstrip("\\")
                p = _PKG / jpref.replace("/", "\\")
                return p / rest if rest else p
        # 回退：镜像目录（保留 L:\实验数据\ 结构）
        return _PKG / "08_L盘补充数据" / "实验数据" / full

    def __fspath__(self):
        return str(self)

    def __str__(self):
        return str(_PKG / "08_L盘补充数据" / "实验数据")


DATA = _DataPath("")
OUT_ROOT = Path(os.environ.get("YRS_OUT_ROOT", str(_PKG / "06_文本" / "代码_校正研究" / "out")))

#: 基准栅格（掩膜）—— 网格的一切以它为准
MASK_TIF = DATA / r"02_遥感与基础地理\绝对基准栅格_初始层\mask.tif"

RUSLE = _DataPath(r"03_RUSLE因子")
RESULTS = _DataPath(r"04_模型输出与结果")
ANALYSIS = _DataPath(r"05_分析与统计")
TOOLS = _DataPath(r"06_代码与工具")

#: NoData 候选值（float32 极值常被当成 NoData）
_F32_MAX = float(np.finfo(np.float32).max)

# ---------------------------------------------------------------- 日志


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """同时输出到控制台与 out/logs/<name>.log。"""
    log_dir = log_dir or (OUT_ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:                      # 避免重复添加
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class Timer:
    """with Timer(log, "阶段2 合成 A"): ..."""

    def __init__(self, logger: logging.Logger, label: str):
        self.logger, self.label = logger, label
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info("开始：%s", self.label)
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        self.logger.info("完成：%s  用时 %.1f s", self.label, dt)
        return False


# ---------------------------------------------------------------- 网格


@dataclass
class Grid:
    """基准网格描述。"""

    width: int
    height: int
    transform: tuple
    crs: object
    nodata: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def count(self) -> int:
        return self.width * self.height

    @property
    def pixel_area_m2(self) -> float:
        """像元面积（m²）—— 由 transform 的 x/y 分辨率推出。"""
        return abs(self.transform[0] * self.transform[4])

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "transform": list(self.transform),
            "crs": self.crs.to_string() if self.crs else None,
            "pixel_area_m2": self.pixel_area_m2,
            "cells": self.count,
        }


def base_grid(path: Path | str = MASK_TIF) -> Grid:
    """读取基准网格（默认取 mask.tif）。"""
    with rasterio.open(path) as d:
        return Grid(d.width, d.height, tuple(d.transform)[:6], d.crs, d.nodata)


_BASE: Grid | None = None


def get_base_grid() -> Grid:
    """进程内缓存的基准网格。"""
    global _BASE
    if _BASE is None:
        _BASE = base_grid()
    return _BASE


def meta_for(grid: Grid, dtype: str = "float32", nodata: float | None = None,
             compress: str = "lzw") -> dict:
    """构造与基准网格一致的写出元数据。"""
    if nodata is None:
        nodata = -_F32_MAX if dtype.startswith("float") else 0
    return {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": rasterio.transform.Affine(*grid.transform),
        "nodata": nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }


# ---------------------------------------------------------------- NoData 归一


def nodata_of(ds) -> float | None:
    nd = ds.nodata
    return None if nd is None else float(nd)


def to_nan(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """
    把数组里的 NoData 归一为 NaN。

    兼容盘上出现的四种约定：
      - -3.4028235e38（float32 最小值）
      - +3.4028235e38
      - -1.0（碳储）
      - 0（掩膜 / LULC，此时保持 0 有效需调用方自行处理）
    """
    a = arr.astype(np.float32, copy=False)
    if nodata is not None:
        a = np.where(a == np.float32(nodata), np.nan, a)
    # 兜底：float32 极值一律视为无效
    a = np.where(np.abs(a) >= _F32_MAX * 0.5, np.nan, a)
    return a


def from_nan(arr: np.ndarray, nodata: float, dtype: str = "float32") -> np.ndarray:
    """NaN -> NoData 值，写盘用。"""
    if dtype.startswith("float"):
        return np.where(np.isnan(arr), nodata, arr).astype(dtype)
    # 整数型：四舍五入后写入
    out = np.where(np.isnan(arr), nodata, np.rint(arr))
    return out.astype(dtype)


# ---------------------------------------------------------------- 分块


def block_rows(grid: Grid, block_mb: float = 512.0, n_bands: int = 1,
               dtype_bytes: int = 4) -> Iterator[tuple[int, int]]:
    """
    按内存预算生成 (row_start, row_end) 行块。

    block_mb 为该块**单波段**允许占用的内存（MB）；实际占用约
    rows * width * n_bands * dtype_bytes。
    """
    bytes_per_row = grid.width * n_bands * dtype_bytes
    rows = max(1, int(block_mb * 1024 * 1024 / max(1, bytes_per_row)))
    r = 0
    while r < grid.height:
        yield r, min(r + rows, grid.height)
        r += rows


def read_block(src, row0: int, row1: int, band: int = 1,
               grid: Grid | None = None) -> np.ndarray:
    """读某个行块为 float32（已把 NoData 归为 NaN）。"""
    win = Window(0, row0, src.width, row1 - row0)
    arr = src.read(band, window=win)
    return to_nan(arr, nodata_of(src))


def read_window(path: Path | str, window: Sequence[int] | None = None,
                band: int = 1) -> tuple[np.ndarray, Grid]:
    """
    读入一个窗口为 float32(NaN=无效)。

    window = (x0, y0, x1, y1) 像素坐标；None 表示全幅（慎用，约 1.3 GB）。

    **自动识别窗口产物**：若文件本身的尺寸恰好等于 window 的尺寸，说明它是
    上一个阶段按该窗口写出的独立栅格（自带窗口原点），此时直接读全幅，
    否则再套全局窗口读会越界返回空数组。
    """
    with rasterio.open(path) as src:
        if window is None:
            win = Window(0, 0, src.width, src.height)
        else:
            x0, y0, x1, y1 = window
            ww, hh = x1 - x0, y1 - y0
            if (src.width, src.height) == (ww, hh):
                win = Window(0, 0, src.width, src.height)      # 已是窗口产物
            else:
                win = Window(x0, y0, ww, hh)
        arr = to_nan(src.read(band, window=win), nodata_of(src))
        tr = src.window_transform(win)
        g = Grid(int(win.width), int(win.height), tuple(tr)[:6], src.crs, src.nodata)
    return arr, g


def write_raster(path: Path | str, arr: np.ndarray, grid: Grid,
                 dtype: str = "float32", nodata: float | None = None,
                 compress: str = "lzw") -> Path:
    """按给定网格写出单波段 GeoTIFF（自动填 NoData、建概览）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = meta_for(grid, dtype=dtype, nodata=nodata, compress=compress)
    data = from_nan(arr, m["nodata"], dtype) if dtype.startswith("float") else arr
    with rasterio.open(path, "w", **m) as dst:
        dst.write(data, 1)
    return path


def write_blocks(path: Path | str, grid: Grid,
                 producer: Callable[[int, int], np.ndarray],
                 dtype: str = "float32", nodata: float | None = None,
                 block_mb: float = 512.0, compress: str = "lzw") -> Path:
    """
    分块流式写出：producer(row0, row1) 返回该行块的二维数组。

    适合把「多年序列 -> 单一统计量」这类无法一次性装进内存的运算落盘。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = meta_for(grid, dtype=dtype, nodata=nodata, compress=compress)
    with rasterio.open(path, "w", **m) as dst:
        for r0, r1 in block_rows(grid, block_mb=block_mb):
            blk = producer(r0, r1)
            if dtype.startswith("float"):
                blk = from_nan(blk, m["nodata"], dtype)
            dst.write(blk.astype(dtype), 1, window=Window(0, r0, grid.width, r1 - r0))
    return path


def stack_block(paths: Sequence[Path | str], row0: int, row1: int,
                grid: Grid | None = None) -> np.ndarray:
    """
    把一组同网格栅格的同一行块堆成 (T, rows, width) 的数组。

    这是所有「时序统计」的统一入口。
    """
    out = None
    for i, p in enumerate(paths):
        with rasterio.open(p) as src:
            arr = read_block(src, row0, row1)
        if out is None:
            out = np.empty((len(paths),) + arr.shape, dtype=np.float32)
        out[i] = arr
    if out is None:
        raise ValueError("paths 为空")
    return out


# ---------------------------------------------------------------- 掩膜


def load_mask(window: Sequence[int] | None = None) -> np.ndarray:
    """读研究区掩膜为布尔数组（True = 研究区内）。"""
    with rasterio.open(MASK_TIF) as src:
        if window is None:
            win = Window(0, 0, src.width, src.height)
        else:
            x0, y0, x1, y1 = window
            win = Window(x0, y0, x1 - x0, y1 - y0)
        m = src.read(1, window=win)
    return m > 0


def load_mask_block(row0: int, row1: int) -> np.ndarray:
    """读掩膜的某个行块。"""
    with rasterio.open(MASK_TIF) as src:
        return src.read(1, window=Window(0, row0, src.width, row1 - row0)) > 0


def apply_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """研究区外置 NaN。"""
    return np.where(mask, arr, np.nan)


# ---------------------------------------------------------------- 统计


@dataclass
class RasterStats:
    n_valid: int = 0
    n_total: int = 0
    vmin: float = float("nan")
    vmax: float = float("nan")
    mean: float = float("nan")
    std: float = float("nan")
    p05: float = float("nan")
    p50: float = float("nan")
    p95: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "n_valid": self.n_valid,
            "n_total": self.n_total,
            "valid_frac": (self.n_valid / self.n_total) if self.n_total else 0.0,
            "min": self.vmin, "max": self.vmax, "mean": self.mean,
            "std": self.std, "p05": self.p05, "p50": self.p50, "p95": self.p95,
        }


def stats_of(path: Path | str, band: int = 1, sample: int = 0,
             mask_stats: bool = True) -> RasterStats:
    """
    统计单幅栅格的有效像元统计量。

    sample > 0 时按行抽样（每 sample 行取一行），用于大文件快速探察。
    mask_stats=True 时只统计研究区内像元。
    """
    with rasterio.open(path) as src:
        if src.width != get_base_grid().width:
            # 非基准网格（例如 ERI 的 900 m 面）——全幅统计，不套 30 m 掩膜
            mask_stats = False
        chunks = []
        rows = range(0, src.height, sample) if sample > 0 else None
        if rows is None:
            for r0, r1 in block_rows(Grid(src.width, src.height, tuple(src.transform)[:6], src.crs), 256):
                chunks.append(read_block(src, r0, r1, band))
        else:
            for r in rows:
                chunks.append(read_block(src, r, min(r + 1, src.height), band))
    arr = np.concatenate(chunks, axis=0)
    if mask_stats:
        try:
            m, _ = read_window(MASK_TIF)
            if m.shape == arr.shape:
                arr = apply_mask(arr, m > 0)
        except Exception:
            pass
    v = arr[np.isfinite(arr)]
    st = RasterStats(n_total=int(arr.size), n_valid=int(v.size))
    if v.size:
        st.vmin, st.vmax = float(v.min()), float(v.max())
        st.mean, st.std = float(v.mean()), float(v.std())
        st.p05, st.p50, st.p95 = (float(x) for x in np.percentile(v, [5, 50, 95]))
    return st


# ---------------------------------------------------------------- 输出登记


def ensure_dirs() -> None:
    for s in ["s01", "s02", "s03", "s04", "s05", "s06",
              "s07", "s08", "s09", "s10", "logs"]:
        (OUT_ROOT / s).mkdir(parents=True, exist_ok=True)


def dump_json(obj, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def years(start: int = 2000, end: int = 2020) -> list[int]:
    return list(range(start, end + 1))


def year_tag(y: int) -> str:
    return f"{y:04d}"
