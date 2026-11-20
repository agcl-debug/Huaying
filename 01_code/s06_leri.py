"""
阶段 6　景观生态风险

对应任务 T6.1 - T6.8

补写的代码
----------
* 景观指数（PERIM/FRAC/CA/NP/SPLIT/FN/DI/ELDI/ERI）按论文表2-11 公式自行实现，
  替代 Fragstats（本机无该软件）。
* ERI 的克里金插值自行实现（`stats.ordinary_kriging`，球状模型、900 m、12 近邻），
  替代 ArcGIS 自定义工具箱。
* 全局 Moran's I 与局部 LISA 自行实现，**补上盘上 `全局分析.xls`（OLE2 二进制）
  无法读取的缺口**。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage

from . import common as C
from . import geo as G
from . import stats as S

LERI_UNIT_SHP = C.DATA / r"02_遥感与基础地理\YRS_渔网\研究区9km格网.shp"
CLCD = C.DATA / r"02_遥感与基础地理\LULC_CLCD"
DEFAULT_WINDOW = (10132, 6608, 12180, 8656)

#: 论文表2-12 脆弱度分级与归一化值 Vi
VULNERABILITY = {1: 0.07, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.13,
                 6: 0.17, 7: 0.13, 8: 0.03, 9: 0.17, 10: 0.03}
#: 论文表2-11 干扰度权重：破碎度 0.5 / 分离度 0.3 / 分维数 0.2
DI_W = (0.5, 0.3, 0.2)
#: 论文/血缘的 ERI 九级阈值
ERI_BREAKS = [0.002352, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 1.0]
KRIGING_MODEL = {"nugget": 0.00015, "sill": 0.000209 + 0.00015, "range": 688727.5}


def landscape_indices(lulc: np.ndarray, cell_m: float = 30.0,
                      classes=None) -> pd.DataFrame:
    """按论文表2-11 公式计算类型级/景观级指数（全局）。"""
    if classes is None:
        classes = [c for c in np.unique(lulc) if c > 0]
    cell_area = cell_m ** 2
    total_area = float((lulc > 0).sum()) * cell_area
    rows = []
    for c in classes:
        m = lulc == c
        if not m.any():
            continue
        ca = float(m.sum()) * cell_area                       # m²
        lab, npatch = ndimage.label(m, structure=np.ones((3, 3)))
        perim = _perimeter(m, cell_m)
        # FRAC = 2·ln(0.25·P) / ln(A)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = 2.0 * np.log(0.25 * perim) / np.log(ca) if ca > 1 and perim > 0 else np.nan
        frac = float(np.clip(frac, 1.0, 2.0)) if np.isfinite(frac) else 1.0
        # SPLIT（景观分离度）：常用形式 D = 0.5·√(n/A) ；此处取 √(n/A)·A_total/A
        with np.errstate(divide="ignore", invalid="ignore"):
            split = float(np.sqrt(npatch / (ca / 1e6)) * (total_area / 1e6) / (ca / 1e6)) \
                if ca > 0 else np.nan
        # 破碎度 FN = (npatch - 1) / CA(km²)
        fn = float((npatch - 1) / (ca / 1e6)) if ca > 0 else np.nan
        rows.append({"class": int(c), "CA_m2": ca, "CA_km2": ca / 1e6, "NP": int(npatch),
                     "PERIM_m": perim, "FRAC": frac, "SPLIT": split, "FN": fn})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 归一化到 [0,1] 后加权得干扰度
    for col in ("FN", "SPLIT", "FRAC"):
        v = df[col].values.astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        df[col + "_n"] = (v - lo) / (hi - lo) if hi > lo else 0.0
    df["DI"] = (DI_W[0] * df["FN_n"] + DI_W[1] * df["SPLIT_n"] + DI_W[2] * df["FRAC_n"])
    df["Vi"] = df["class"].map(VULNERABILITY).fillna(0.10)
    df["ELDI"] = df["DI"] * df["Vi"]
    return df


def _perimeter(mask: np.ndarray, cell_m: float) -> float:
    """斑块边界总长（m）：与背景相邻的像元边数 × 像元边长。"""
    b = np.zeros_like(mask, dtype=bool)
    b[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    b[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    b[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    b[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    return float(b.sum()) * cell_m


def eri_per_zone(lulc: np.ndarray, zone_labels: np.ndarray,
                 idx: pd.DataFrame) -> pd.DataFrame:
    """ERI_k = Σ_i (A_ki / A_k) · ELDI_i。"""
    codes = sorted(int(c) for c in idx["class"].unique())
    eldi = {int(r["class"]): float(r["ELDI"]) for _, r in idx.iterrows()}
    nz = int(zone_labels.max())
    cmap = {c: i + 1 for i, c in enumerate(codes)}
    comb = np.where((zone_labels > 0) & (lulc > 0), zone_labels * 1000 + lulc, 0)
    vals = np.unique(comb[comb > 0])
    zc = vals // 1000
    cc = vals % 1000
    cnt = np.bincount(comb.ravel(), minlength=int(comb.max()) + 1)[vals]
    ca = pd.DataFrame({"zone": zc.astype(int), "class": cc.astype(int), "n": cnt})
    ca["A"] = ca["n"].astype(float)   # A_ki/A_k 是面积比，像元面积会约掉，直接用像元计数即可
    ca["ELDI"] = ca["class"].map(eldi).fillna(np.nan)
    ca["prod"] = ca["A"] * ca["ELDI"]
    g = ca.groupby("zone").agg(A_total=("A", "sum"), num=("prod", "sum"))
    g["ERI"] = g["num"] / g["A_total"]
    return g.reset_index()


def run(out_root: Path = C.OUT_ROOT, window=DEFAULT_WINDOW,
        year: int = 2010, **_):
    log = C.get_logger("s06_leri")
    out = Path(out_root) / "s06"
    out.mkdir(parents=True, exist_ok=True)
    import rasterio

    # ---- T6.1 评价单元（9 km，1,682 个）
    units = G.read_vector(LERI_UNIT_SHP)
    log.info("T6.1 9 km 评价单元 %d 个", len(units))

    # ---- T6.2 景观指数
    lulc_p = None
    for pat in (f"LULC_{year}.tif", "LULC_2010.tif", "LULC_2000.tif"):
        if (CLCD / pat).exists():
            lulc_p = CLCD / pat
            break
    lulc, wg = C.read_window(lulc_p, window)
    lulc_code = np.where(np.isfinite(lulc), lulc, 0).astype(np.int16)
    idx = landscape_indices(lulc_code, cell_m=abs(wg.transform[0]))
    idx.to_csv(out / f"landscape_indices_{year}.csv", index=False, encoding="utf-8-sig")
    log.info("T6.2 景观指数（%d 类）：\n%s", len(idx),
             idx[["class", "CA_km2", "NP", "FRAC", "FN", "SPLIT", "DI", "Vi", "ELDI"]]
             .to_string(index=False))

    # ---- T6.4 单元 ERI：把单元栅格化到窗口
    # 直接取 read_window 返回的窗口网格 transform。
    # 原写法 Window(*window) 在 window=None（全域）时会 TypeError。
    tr = rasterio.transform.Affine(*wg.transform)
    from rasterio import features
    zone_lab = features.rasterize(
        ((g, i + 1) for i, g in enumerate(units.geometry)),
        out_shape=wg.shape, transform=tr, fill=0, dtype="int32")
    zdf = eri_per_zone(lulc_code, zone_lab, idx)
    zdf.to_csv(out / f"ERI_units_{year}.csv", index=False, encoding="utf-8-sig")
    ev = zdf["ERI"].values
    log.info("T6.4 单元 ERI：n=%d  min=%.4f mean=%.4f max=%.4f",
             ev.size, np.nanmin(ev), np.nanmean(ev), np.nanmax(ev))

    # ---- T6.5 克里金成面（900 m）
    cents = G.points_from_polygons(units)
    pts_all = np.column_stack([cents, np.full(len(cents), np.nan)])
    e_map = dict(zip(zdf["zone"].values, zdf["ERI"].values))
    keep = np.array([i + 1 in e_map and np.isfinite(e_map[i + 1]) for i in range(len(units))])
    sxy = cents[keep]
    sz = np.array([e_map[i + 1] for i in range(len(units)) if keep[i]])
    # 预测网格：900 m，范围取**当前工作网格**的实际范围（不能硬编码 30 m）
    _cell = abs(wg.transform[0])
    _x0, _y0 = wg.transform[2], wg.transform[5]
    _x1 = _x0 + wg.width * _cell
    _y1 = _y0 - wg.height * _cell
    gx = np.arange(_x0 + 450, _x1, 900)
    gy = np.arange(_y0 - 450, _y1, -900)
    GX, GY = np.meshgrid(gx, gy)
    pred_xy = np.column_stack([GX.ravel(), GY.ravel()])
    model = S.fit_variogram(sxy, sz) if len(sz) >= 10 else KRIGING_MODEL
    log.info("T6.5 变差函数拟合：%s", {k: round(v, 6) for k, v in model.items()})
    kz = S.ordinary_kriging(sxy, sz, pred_xy, model, n_neighbors=12)
    ERI_K = kz.reshape(GX.shape)
    kg = C.Grid(ERI_K.shape[1], ERI_K.shape[0],
                (900.0, 0.0, float(gx[0] - 450), 0.0, -900.0, float(gy[0] + 450)),
                rasterio.open(C.MASK_TIF).crs, None)
    C.write_raster(out / f"ERI_{year}_K.tif", ERI_K, kg)
    kv = ERI_K[np.isfinite(ERI_K)]
    log.info("T6.5 ERI 面：%dx%d  mean=%.4f  range=%.4f~%.4f",
             kg.width, kg.height, kv.mean(), kv.min(), kv.max())

    # ---- T6.6 九级分类
    kr = S.classify(ERI_K, ERI_BREAKS)
    C.write_raster(out / f"ERI_{year}_KR.tif", kr.astype(np.float32), kg,
                   dtype="int16", nodata=-128)
    for lv in range(1, 10):
        log.info("  ERI 级别 %d：%d 个像元（%.2f%%）", lv, int((kr == lv).sum()),
                 100.0 * (kr == lv).sum() / max(1, kv.size))

    # ---- T6.7 全局 Moran's I 与 LISA（补上 xls 缺口）
    mask2 = np.isfinite(ERI_K)
    gi = S.morans_i(ERI_K, mask2)
    log.info("T6.7 全局 Moran's I = %.4f  Z=%.2f  P=%.3g  n=%d",
             gi.get("I", np.nan), gi.get("Z", np.nan), gi.get("P", np.nan), gi.get("n", 0))
    li, cls = S.lisa(ERI_K, mask2)
    C.write_raster(out / f"LISA_{year}.tif", li, kg)
    C.write_raster(out / f"LISA_cluster_{year}.tif", cls.astype(np.float32), kg,
                   dtype="int8", nodata=-128)
    C.dump_json({"year": year, "global_moran": gi,
                 "lisa_counts": {int(k): int(v) for k, v in
                                 zip(*np.unique(cls[mask2], return_counts=True))},
                 "论文值": {"Moran_I_mean": 0.547, "annual_decline": 0.0015,
                            "lisa_sig_increase_pct": 20.31}},
                out / f"moran_{year}.json")
    log.info("T6.7 LISA 分类计数：%s",
             {int(k): int(v) for k, v in zip(*np.unique(cls[mask2], return_counts=True))})
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段6 景观生态风险")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--window", default=",".join(map(str, DEFAULT_WINDOW)))
    ap.add_argument("--year", type=int, default=2010)
    C.ensure_dirs()
    a = ap.parse_args(argv)
    run(Path(a.out_root),
        window=tuple(int(v) for v in a.window.split(",")), year=a.year)


if __name__ == "__main__":
    main()
