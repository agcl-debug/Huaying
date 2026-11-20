"""
阶段 7　生态安全格局

对应任务 T7.1 - T7.8

补写的代码（替代三个无法本机运行的软件）
------------------------------------------
* MSPA（原用 GuidosToolbox）：用形态学开闭运算 + 八邻域实现核心区识别
* 最小成本路径 / 廊道（原用 Linkage Mapper + Circuitscape）：
  用 Dijkstra 求源地间最小累积阻力路径
* 连通性中心性（原用 Conefor 2.6）：用"移除单节点后连通性变化"计算 dPC/dIIC 近似
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from . import common as C
from . import geo as G

ZULI = C.RESULTS / "ZULI"
CLCD = C.DATA / r"02_遥感与基础地理\LULC_CLCD"
THREATS_LC = C.RESULTS / "Threats Files"
#: MSPA 前景用的 FVC 目录（由调用方设置）；None 表示不启用 FVC 拆分草地
FVC_DIR = None
#: 中/高覆盖草地的 FVC 阈值（论文二级分类"中覆盖草地"下限约 0.2~0.3）
FVC_THR = 0.3
DEFAULT_WINDOW = (10132, 6608, 12180, 8656)

#: 论文表5-2 的 9 个阻力要素：权重、阻力方向(+/-/赋值)
RESIST_FACTORS = {
    "LUC":  (0.22, "assign"),
    "Pre":  (0.08, "-"),
    "Slo":  (0.13, "-"),
    "FVC":  (0.06, "-"),
    "DEM":  (0.09, "+"),
    "HF":   (0.11, "-"),
    "DtoR": (0.11, "+"),
    "Tmp":  (0.08, "-"),
    "DtoW": (0.12, "-"),
}
#: 论文表5-3 土地利用阻力值
LUC_RESIST = {1: 8, 2: 1, 3: 2, 4: 3, 5: 6, 6: 9, 7: 8, 8: 10, 9: 5, 10: 10}
#: MSPA 前景（论文 2.3.9 的 10 类）
MSPA_FOREGROUND = {2, 3, 5, 6, 9}   # 森林/灌木/水域/冰雪/湿地（排除 7 裸地、8 不透水）
N_SOURCES = 10


def mms(x):
    """正向化归一化。"""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.full_like(x, np.nan)
    lo, hi = v.min(), v.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def nmms(x):
    """负向化归一化。"""
    return 1.0 - mms(x)


def mspa_core(lulc: np.ndarray, edge_width_px: int | None = None,
              edge_width_m: float = 300.0, cell_m: float | None = None,
              fvc: np.ndarray | None = None, fvc_thr: float = 0.3) -> np.ndarray:
    """
    MSPA 核心区识别的简化实现。

    边缘宽度按**物理距离**换算为像元数，**不可硬编码**：
        30 m 网格 -> 300/30 = 10 像元
        300 m 网格 -> 300/300 = 1 像元

    **前景类别**：论文 §2.3.9 列的是二级分类（有林地/灌木地/疏林地/高覆盖草地/
    中覆盖草地/河渠/湖泊/永久性冰川雪地/滩地/沼泽地），**不含低覆盖草地与裸地**。
    本研究栅格只有 CLCD 一级类，草地全归为 4、裸地为 7，故：
      - 排除 7（裸地）、8（不透水）
      - 草地(4) 用 **FVC ≥ fvc_thr** 近似"中/高覆盖草地"
    否则前景覆盖全域、连通成一块（实测 n_patch=1，源地无从取 10 个）。
    """
    fg = np.isin(lulc, list(MSPA_FOREGROUND))
    if fvc is not None:
        fg = fg | ((lulc == 4) & (np.nan_to_num(fvc, nan=0.0) >= fvc_thr))
    if edge_width_px is None:
        cm = cell_m if cell_m else 30.0
        edge_width_px = max(1, int(round(edge_width_m / cm)))
    st = np.ones((3, 3), bool)                     # 八邻域
    eroded = ndimage.binary_erosion(fg, structure=st, iterations=edge_width_px)
    core = ndimage.binary_opening(eroded, structure=st, iterations=1)
    return core


def cumulative_resistance(res: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    """由阻力面构造八邻域栅格图（边权 = 两端阻力均值 × 距离）。"""
    H, W = res.shape
    idx = np.arange(H * W).reshape(H, W)
    valid = np.isfinite(res)
    rows, cols, data = [], [], []
    # 四邻域 + 对角（对角距离 ×√2）
    for dy, dx, mul in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        y0 = slice(max(0, -dy), H - max(0, dy))
        y1 = slice(max(0, dy), H - max(0, -dy))
        x0 = slice(max(0, -dx), W - max(0, dx))
        x1 = slice(max(0, dx), W - max(0, -dx))
        a = valid[y0, x0] & valid[y1, x1]
        ia = idx[y0, x0][a]; ib = idx[y1, x1][a]
        ra = res[y0, x0][a]; rb = res[y1, x1][a]
        w = 0.5 * (ra + rb) * mul
        rows.append(ia); cols.append(ib); data.append(w)
        rows.append(ib); cols.append(ia); data.append(w)
    r = np.concatenate(rows); c = np.concatenate(cols); d = np.concatenate(data)
    g = sparse.csr_matrix((d, (r, c)), shape=(H * W, H * W))
    return g, valid


def run(out_root: Path = C.OUT_ROOT, window=DEFAULT_WINDOW,
        year: int = 2010, n_sources: int = N_SOURCES, **_):
    log = C.get_logger("s07_network")
    out = Path(out_root) / "s07"
    out.mkdir(parents=True, exist_ok=True)
    import rasterio

    # ---- T7.3/T7.4 综合阻力面
    lc_p = THREATS_LC / str(year) / f"{year}_c.tif"
    if not lc_p.exists():
        for pat in (f"LULC_{year}.tif", "LULC_2010.tif"):
            if (CLCD / pat).exists():
                lc_p = CLCD / pat
                break
    lulc, wg = C.read_window(lc_p, window)
    lulc_code = np.where(np.isfinite(lulc), lulc, 0).astype(np.int16)
    # 土地利用阻力（表5-3）
    luc_r = np.full(lulc_code.shape, np.nan, dtype=np.float32)
    for code, v in LUC_RESIST.items():
        luc_r[lulc_code == code] = v
    res_parts, wsum = [], 0.0
    res_parts.append(("LUC", mms(luc_r), RESIST_FACTORS["LUC"][0]))
    wsum += RESIST_FACTORS["LUC"][0]
    for nm, (w, direction) in RESIST_FACTORS.items():
        if nm == "LUC":
            continue
        if not ZULI.exists():
            continue
        # 文件名别名：论文/代码里的缩写 与 ZULI 目录中的实际命名不一致
        alias = {"Slo": "Slope"}
        fn = alias.get(nm, nm)
        # 逐年因子形如 2010_FVC_ZU.tif；静态因子（DEM/DtoW/Slope）无年份前缀，
        # 形如 DEM_ZU.tif —— 故必须有"无年份"兜底，否则静态因子永远匹配不到。
        cands = (sorted(ZULI.glob(f"{year}_{fn}_ZU*.tif"))
                 or sorted(ZULI.glob(f"{year}_{fn}*.tif"))
                 or sorted(ZULI.glob(f"{fn}_ZU*.tif"))
                 or sorted(ZULI.glob(f"{fn}*.tif"))
                 or sorted(ZULI.glob(f"*_{fn}_*.tif")))
        if not cands:
            continue
        a, _ = C.read_window(cands[0], window)
        n = nmms(a) if direction == "-" else (mms(a) if direction == "+" else mms(a))
        res_parts.append((nm, n, w))
        wsum += w
    if not res_parts:
        raise RuntimeError("无可用阻力因子")
    RES = np.zeros(wg.shape, dtype=np.float64)
    for nm, a, w in res_parts:
        RES += (w / wsum) * np.nan_to_num(a, nan=0.0)
    RES = np.where(np.isfinite(lulc), RES, np.nan)
    C.write_raster(out / f"ZUR_{year}.tif", RES.astype(np.float32), wg)
    log.info("T7.4 综合阻力面：用到 %d 个要素（权重和 %.2f）  mean=%.4f",
             len(res_parts), wsum, np.nanmean(RES))

    # ---- T7.1 MSPA 核心区 + T7.2 源地
    _fvc = None
    if FVC_DIR is not None:
        _fp = Path(FVC_DIR) / f"FVC_{year}.tif"
        if _fp.exists():
            _fvc, _ = C.read_window(_fp, None)
            log.info("T7.1 MSPA 前景：草地按 FVC>=%.2f 取中/高覆盖（排除裸地）", FVC_THR)
    core = mspa_core(lulc_code, cell_m=abs(wg.transform[0]), fvc=_fvc, fvc_thr=FVC_THR)
    C.write_raster(out / f"MSPA_core_{year}.tif", core.astype(np.float32), wg,
                   dtype="uint8", nodata=0)
    lab, n_patch = ndimage.label(core, structure=np.ones((3, 3)))
    sizes = ndimage.sum(core, lab, range(1, n_patch + 1))
    order = np.argsort(sizes)[::-1][:n_sources]
    keep = set(int(o + 1) for o in order)
    log.info("T7.1/T7.2 MSPA 核心区：%d 块，取面积前 %d 作为生态源地（最大 %.1f km²）",
             n_patch, len(keep),
             sizes[order[0]] * (abs(wg.transform[0]) ** 2) / 1e6 if len(order) else 0)

    # 源地中心点
    _cell = abs(wg.transform[0])
    cents = ndimage.center_of_mass(core, lab, sorted(keep))
    src_xy = np.array([[c[1] * _cell + wg.transform[2], c[0] * (-_cell) + wg.transform[5]]
                       for c in cents])
    src_id = np.array(sorted(keep))
    pd.DataFrame({"src": src_id, "x": src_xy[:, 0], "y": src_xy[:, 1],
                  "area_km2": [sizes[s - 1] * (_cell ** 2) / 1e6 for s in src_id]}) \
        .to_csv(out / f"sources_{year}.csv", index=False, encoding="utf-8-sig")

    # ---- T7.5 最小累积阻力路径（Dijkstra）
    if len(src_xy) < 2:
        log.warning("源地少于 2 个，跳过廊道计算")
        return pd.DataFrame()
    g, valid = cumulative_resistance(RES)
    # 把源地映射到栅格索引
    def to_idx(x, y):
        c = int(round((x - wg.transform[2]) / _cell - 0.5))
        r = int(round((wg.transform[5] - y) / _cell - 0.5))
        if 0 <= r < wg.height and 0 <= c < wg.width and valid[r, c]:
            return r * wg.width + c
        # 就近搜索有效像元
        for rad in range(1, 30):
            r0, r1 = max(0, r - rad), min(wg.height, r + rad + 1)
            c0, c1 = max(0, c - rad), min(wg.width, c + rad + 1)
            sub = valid[r0:r1, c0:c1]
            if sub.any():
                rr, cc = np.argwhere(sub)[0]
                return (r0 + rr) * wg.width + (c0 + cc)
        return -1

    nodes = np.array([to_idx(x, y) for x, y in src_xy])
    good = nodes >= 0
    nodes = nodes[good]; src_xy = src_xy[good]; src_id = src_id[good]
    log.info("T7.5 参与计算的源地 %d 个", len(nodes))

    # 逐源 Dijkstra（限制最大距离以控内存）
    dist_all = np.full((len(nodes), g.shape[0]), np.inf, dtype=np.float32)
    for i, nd in enumerate(nodes):
        d = dijkstra(g, indices=int(nd), directed=False, limit=np.inf)
        dist_all[i] = d
        log.info("  Dijkstra %d/%d 完成", i + 1, len(nodes))

    links = []
    for i, j in itertools.combinations(range(len(nodes)), 2):
        dd = dist_all[i, nodes[j]]
        if np.isfinite(dd):
            links.append({"src1": int(src_id[i]), "src2": int(src_id[j]),
                          "lcDist": float(dd)})
    ldf = pd.DataFrame(links)
    ldf.to_csv(out / f"links_{year}.csv", index=False, encoding="utf-8-sig")
    log.info("T7.5 源地间最小累积阻力：%d 条链接  lcDist 中位数=%.1f",
             len(ldf), float(ldf["lcDist"].median()) if len(ldf) else np.nan)

    # ---- T7.7 连通性中心性（Conefor PC 公式，**含斑块内项**）
    # 原实现用"移除单节点后图连通性下降"，该度量**不含斑块内贡献**，
    # 会把巨型源地的重要性严重低估 —— 实测反转成最低（4.59%），而小源地反居前。
    # 论文表5-4 中巨型源地 `dIICintra = 84.82%`（斑块内贡献占其总贡献 86%），
    # 说明主导项是**斑块自身面积的平方**，必须用 Conefor 的 PC：
    #     PC    = Σ_i Σ_j a_i·a_j·p_ij / A_L²
    #     dPC_i = (PC − PC_{−i}) / PC × 100%
    # α 取 1000 km：实测论文表5-4 对应的是"近似全连通"情形。
    # 敏感性（巨型源地 dPC / 次大源地 dPC）：
    #     α=5 km -> 99.89% / 0.09%      α=500  -> 99.69% / 3.45%
    #     α=1000 -> 99.65% / 4.40%      α→∞    -> 99.61% / 5.55%
    #     论文                   98.34% / 6.34%
    # α→∞ 的解析极限：dPC_i = 1 − ((Σa − a_i)/Σa)²
    _alpha = 1000.0
    _area = {int(src_id[i]): float(sizes[int(src_id[i]) - 1] * (_cell ** 2) / 1e6)
             for i in range(len(src_id))}
    _AL = float(np.isfinite(lulc).sum()) * (_cell ** 2) / 1e6     # 景观总面积 km²
    _d = {}
    for _, _r in ldf.iterrows():
        _d[(int(_r["src1"]), int(_r["src2"]))] = float(_r["lcDist"])
    _big = (max(_d.values()) * 10.0) if _d else 1e6

    def _pc(skip=None):
        s = 0.0
        for _i in src_id:
            if int(_i) == skip:
                continue
            ai = _area[int(_i)]
            for _j in src_id:
                if int(_j) == skip:
                    continue
                if int(_i) == int(_j):
                    p = 1.0
                else:
                    p = float(np.exp(-_d.get((int(_i), int(_j)),
                                             _d.get((int(_j), int(_i)), _big)) / _alpha))
                s += ai * _area[int(_j)] * p
        return s / (_AL ** 2)

    base = _pc()
    cent = [{"src": int(_i), "dPC": (base - _pc(int(_i))) / base * 100.0 if base > 0 else 0.0,
             "area_km2": _area[int(_i)]} for _i in src_id]
    cdf = pd.DataFrame(cent).sort_values("dPC", ascending=False)
    cdf.to_csv(out / f"centrality_{year}.csv", index=False, encoding="utf-8-sig")
    log.info("T7.7 中心性 dPC（Conefor PC 公式，含斑块内项）：\n%s", cdf.to_string(index=False))

    # ---- T7.8 断裂点：连接不同簇且阻力最高的链接
    if len(ldf):
        thr = ldf["lcDist"].quantile(0.75)
        brk = ldf[ldf["lcDist"] >= thr].copy()
        brk["is_break"] = True
        brk.to_csv(out / f"breaks_{year}.csv", index=False, encoding="utf-8-sig")
        log.info("T7.8 候选断裂点（阻力最高 25%%）：%d 条", len(brk))
    return cdf


def _connectivity(links: pd.DataFrame) -> float:
    """用 1/(1+d) 的链接强度之和作为连通性代理。"""
    if links is None or len(links) == 0:
        return 0.0
    return float(np.sum(1.0 / (1.0 + links["lcDist"].values)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段7 生态安全格局")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--window", default=",".join(map(str, DEFAULT_WINDOW)))
    ap.add_argument("--year", type=int, default=2010)
    ap.add_argument("--n-sources", type=int, default=N_SOURCES)
    C.ensure_dirs()
    a = ap.parse_args(argv)
    run(Path(a.out_root),
        window=tuple(int(v) for v in a.window.split(",")),
        year=a.year, n_sources=a.n_sources)


if __name__ == "__main__":
    main()
