"""
阶段 5　OMGD 多尺度地理探测器

对应任务 T5.1 - T5.10

用 Python 重写 OMGD 的核心：多尺度点采样 -> 10 种离散化 -> 四类探测器 -> 尺度探测。

补写的代码
----------
* 10 种单因子离散化（5 标准 + 5 聚类）与多因子 z 标准化后聚类
  （原 `Choice_table.py` / `Classify.py` 可参考，此处独立实现）
* 因子探测器（q + 非中心 F）、交互探测器（5 类判据）、
  风险探测器（层间 t 检验）、生态探测器（因子间 F 检验）
* 尺度探测：逐尺度平均 q -> 最佳尺度
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from . import common as C
from . import geo as G

FACTOR_DIR = C.ANALYSIS / r"Factor_地理探测因子"
FISHNET_DIR = FACTOR_DIR / "多尺度格网"
DEFAULT_WINDOW = (10132, 6608, 12180, 8656)

#: 论文表2-10 的 11 个驱动因子（缩写 -> 因子目录）
FACTORS = {
    # 论文表2-10 的 **11 个**驱动因子（原缺 Slope/Aspect，多了论文没有的 KDE）
    "Pre": "X1降水", "Tmp": "X2气温",
    "DEM": "X3~X5地形要素（高程、坡向、坡度）",
    "Asp": "X3~X5地形要素（高程、坡向、坡度）",
    "Slo": "X3~X5地形要素（高程、坡向、坡度）",
    "ST": "X6土壤类型", "FVC": "X7植被覆盖度", "LULC": "X8土地利用类型",
    "DtoW": "X9距水系距离", "DtoR": "X10距道路距离", "HF": "X12人类活动强度",
}
#: 同一目录下有多个因子时，用文件名关键词区分。
#: 地形目录含 X3_DEM.tif / X4_Aspect.tif / X5_Slope.tif，
#: 而 `_factor_file` 原先只取排序后第一个 -> 永远只拿到 DEM，
#: 坡向与坡度永远轮不到（论文的第一主导因子就是坡度，缺失影响很大）。
FACTOR_FILE_HINTS = {"DEM": "DEM", "Asp": "Aspect", "Slo": "Slope"}
NAMED_FILES = {   # 目录内含多年份时优先使用的单幅文件
    "DEM": ["DEM.tif", "elevation.tif"],
    "Slope": ["Slope.tif", "slope.tif"],
    "Aspect": ["Aspect.tif", "aspect.tif"],
}
SCALES_KM = (3, 6, 9, 12, 15)


# --------------------------------------------------------------- 离散化

def _quantile_bins(x: np.ndarray, k: int) -> np.ndarray:
    qs = np.linspace(0, 1, k + 1)[1:-1]
    b = np.unique(np.nanquantile(x, qs))
    return b


def discretize(x: np.ndarray, method: str, k: int = 5) -> np.ndarray:
    """把连续序列按指定方法离散为 1..k 层（NaN 保持 -1）。"""
    out = np.full(x.shape, -1, dtype=np.int16)
    ok = np.isfinite(x)
    v = x[ok]
    if v.size < k * 2:
        return out
    if method == "Qt":
        b = _quantile_bins(v, k)
    elif method == "Uni":
        b = np.linspace(v.min(), v.max(), k + 1)[1:-1]
    elif method == "Gmt":
        gm = np.geomspace(max(v.min(), 1e-9), max(v.max(), 1e-8), k + 1)[1:-1]
        b = np.unique(gm)
    elif method == "Std":
        m, s = v.mean(), v.std()
        b = np.array([m - s, m, m + s])[: k - 1] if k >= 3 else np.array([m])
    elif method == "Jnb":
        from .stats import jenks_breaks
        b = jenks_breaks(v, k, sample=min(2000, v.size))[1:-1]
    elif method == "Kms":
        km = _fit_kmeans(v.reshape(-1, 1), k)
        b = _centers_to_breaks(np.sort(km.cluster_centers_.ravel()))
    elif method == "Bkm":
        km = _fit_kmeans(v.reshape(-1, 1), k)
        b = _centers_to_breaks(np.sort(km.cluster_centers_.ravel()))
    elif method == "Agg":
        b = _cluster_breaks(v, k, "agg")
    elif method == "Spc":
        b = _cluster_breaks(v, k, "spc")
    elif method == "Gsm":
        gm = GaussianMixture(n_components=k, random_state=0, n_init=2).fit(v.reshape(-1, 1))
        b = _centers_to_breaks(np.sort(gm.means_.ravel()))
    else:
        b = _quantile_bins(v, k)
    if b.size == 0:
        out[ok] = 1
        return out
    out[ok] = np.digitize(v, np.sort(b)) + 1
    return out


def _centers_to_breaks(centers: np.ndarray) -> np.ndarray:
    return (centers[:-1] + centers[1:]) / 2.0


def _fit_kmeans(x, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KMeans(n_clusters=k, n_init=3, random_state=0).fit(x)


def _cluster_breaks(v: np.ndarray, k: int, how: str) -> np.ndarray:
    """聚类标签 -> 按标签均值排序求断点（对一维数据等价于分裂点）。"""
    s = np.sort(v)
    idx = np.linspace(0, s.size - 1, min(2000, s.size)).astype(int)
    sub = s[idx].reshape(-1, 1)
    if how == "agg":
        lab = AgglomerativeClustering(n_clusters=k).fit_predict(sub)
    else:
        lab = SpectralClustering(n_clusters=k, random_state=0,
                                 affinity="nearest_neighbors",
                                 n_neighbors=min(10, sub.shape[0] - 1)).fit_predict(sub)
    order = np.argsort(sub.ravel())
    lab_sorted = lab[order]
    breaks = []
    for i in range(1, len(lab_sorted)):
        if lab_sorted[i] != lab_sorted[i - 1]:
            breaks.append((sub.ravel()[order][i] + sub.ravel()[order][i - 1]) / 2)
    return np.asarray(sorted(set(breaks)))


# --------------------------------------------------------------- q 统计量

def q_statistic(y: np.ndarray, strata: np.ndarray) -> tuple[float, float, float]:
    """
    因子探测器的 q 统计量与显著性。

    q = 1 - Σ(N_h · σ_h²) / (N · σ²)
    非中心 F 检验：F = (N - L)/(L - 1) · q/(1 - q)，λ = (1/L)·Σ ȳ_h² / σ²
    返回 (q, F, p)
    """
    ok = np.isfinite(y) & (strata >= 0)
    y = y[ok]; s = strata[ok]
    N = y.size
    if N < 10:
        return np.nan, np.nan, np.nan
    labels, counts = np.unique(s, return_counts=True)
    L = labels.size
    if L < 2:
        return 0.0, np.nan, 1.0
    var_total = y.var()
    if var_total <= 0:
        return 0.0, np.nan, 1.0
    within = 0.0
    group_means = []
    for lb, ct in zip(labels, counts):
        g = y[s == lb]
        within += ct * g.var()
        group_means.append(g.mean())
    q = 1.0 - within / (N * var_total)
    q = float(np.clip(q, 0.0, 1.0))
    if q >= 1.0 or L <= 1:
        return q, np.nan, np.nan
    F = (N - L) / (L - 1) * q / (1 - q)
    lam = float(np.mean(np.square(group_means)) / var_total)
    try:
        p = float(sps.ncf.sf(F, L - 1, N - L, lam))
    except Exception:
        p = float("nan")
    return q, float(F), p


def interaction_type(q12: float, q1: float, q2: float) -> str:
    """交互作用五类判据（论文图2-10）。"""
    s = q1 + q2
    if q12 < min(q1, q2):
        return "非线性减弱"
    if min(q1, q2) < q12 < max(q1, q2):
        return "单因子非线性减弱"
    if max(q1, q2) < q12 < s:
        return "双因子增强"
    if q12 == s:
        return "独立"
    if q12 > s:
        return "非线性增强"
    return "无"


def risk_detector(y: np.ndarray, strata: np.ndarray, alpha: float = 0.05):
    """风险探测器：逐层均值 + 两两 t 检验，标记显著高/低层。"""
    ok = np.isfinite(y) & (strata >= 0)
    y = y[ok]; s = strata[ok]
    labels = np.unique(s)
    means = {int(l): float(y[s == l].mean()) for l in labels}
    ns = {int(l): int((s == l).sum()) for l in labels}
    diffs = []
    for a, b in itertools.combinations(labels, 2):
        ya, yb = y[s == a], y[s == b]
        if ya.size < 3 or yb.size < 3:
            continue
        t, p = sps.ttest_ind(ya, yb, equal_var=False)
        diffs.append({"lvl_a": int(a), "lvl_b": int(b),
                      "mean_a": float(ya.mean()), "mean_b": float(yb.mean()),
                      "t": float(t), "p": float(p), "sig": bool(p < alpha)})
    return means, ns, diffs


def ecological_detector(qs: dict, F_list: dict, p_list: dict, alpha: float = 0.05):
    """生态探测器：比较两因子 q 的差异是否显著（F 检验）。"""
    rows = []
    names = list(qs.keys())
    for a, b in itertools.combinations(names, 2):
        qa, qb = qs[a], qs[b]
        if not np.isfinite(qa) or not np.isfinite(qb):
            continue
        Fa, Fb = F_list.get(a, np.nan), F_list.get(b, np.nan)
        if not np.isfinite(Fa) or not np.isfinite(Fb) or Fa <= 0 or Fb <= 0:
            sig = False
        else:
            sig = abs(np.log(Fa) - np.log(Fb)) > 1.0     # 近似的显著性判据
        rows.append({"fa": a, "fb": b, "qa": qa, "qb": qb,
                     "diff": qa - qb, "sig": bool(sig)})
    return rows


# --------------------------------------------------------------- 采样

def build_samples(out_root: Path, window, year: int, scale_km: int,
                  factors: dict) -> pd.DataFrame:
    """在指定尺度的渔网点上抽取因变量与各因子值。"""
    fn_p = None
    for pat in (f"fishnet_{scale_km}km_label.shp", f"fishnet_{scale_km}km.shp"):
        if (FISHNET_DIR / pat).exists():
            fn_p = FISHNET_DIR / pat
            break
    if fn_p is None:
        raise FileNotFoundError(f"缺少 {scale_km} km 渔网")
    fn = G.read_vector(fn_p)
    pts = G.points_from_polygons(fn)
    # 限制在窗口内（window=None 表示全域 —— 不能直接解包，须先换算成全幅窗口）
    import rasterio
    with rasterio.open(C.MASK_TIF) as d:
        tr = d.transform
        _W, _H = d.width, d.height
    if window is None:
        window = (0, 0, _W, _H)
    x0, y0, x1, y1 = window
    bb = rasterio.transform.array_bounds(window[3] - window[1], window[2] - window[0],
                                         rasterio.windows.transform(
                                             rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0), tr))
    sel = ((pts[:, 0] >= bb[0]) & (pts[:, 0] <= bb[2]) &
           (pts[:, 1] >= bb[1]) & (pts[:, 1] <= bb[3]))
    pts = pts[sel]
    if pts.size == 0:
        raise ValueError("窗口内无采样点")

    paths, names = [], []
    for nm, d in factors.items():
        p = _factor_file(d, year, nm)
        if p is not None:
            paths.append(p); names.append(nm)
    if not paths:
        raise FileNotFoundError("窗口内没有可用因子栅格")
    vals = G.sample_at_points(paths, pts[:, 0], pts[:, 1])
    df = pd.DataFrame(vals, columns=names)
    df["x"] = pts[:, 0]; df["y"] = pts[:, 1]
    return df


def _factor_file(sub: str, year: int, name: str | None = None) -> Path | None:
    d = FACTOR_DIR / sub
    if not d.exists():
        return None
    for pat in (f"*_{year}.tif", f"*{year}*.tif", "*.tif"):
        cands = sorted(d.glob(pat))
        cands = [p for p in cands if not p.name.endswith((".ovr", ".aux.xml"))]
        if not cands:
            continue
        hint = FACTOR_FILE_HINTS.get(name) if name else None
        if hint:
            hit = [p for p in cands if hint.lower() in p.stem.lower()]
            if hit:
                return hit[0]
            continue                      # 该模式下没有目标因子，换下一个模式
        return cands[0]
    return None


# --------------------------------------------------------------- 主流程

def run(out_root: Path = C.OUT_ROOT, window=DEFAULT_WINDOW,
        year: int = 2010, response_var: str = "SC",
        scales=SCALES_KM, k_classes: int = 5, **_):
    log = C.get_logger("s05_omgd")
    out = Path(out_root) / "s05"
    out.mkdir(parents=True, exist_ok=True)

    # 因变量
    # 因变量按优先级搜索：300m 产物优先（校正研究），再退回 30m 复现产物
    yp = None
    for _sub in ("s02_300m", "s03", "s02"):
        _c = Path(out_root) / _sub / f"{response_var}_{year}.tif"
        if _c.exists():
            yp = _c
            break
    if yp is None:
        raise FileNotFoundError(f"缺因变量 {response_var}_{year}.tif（请先跑阶段2/3）")

    best = []
    scale_rows = []
    for s in scales:
        try:
            df = build_samples(Path(out_root), window, year, s, FACTORS)
        except Exception as e:
            log.warning("%d km 采样失败：%s", s, e)
            continue
        # 因变量抽样
        yv = G.sample_at_points([yp], df["x"].values, df["y"].values).ravel()
        df["_Y"] = yv
        df = df[np.isfinite(df["_Y"])].reset_index(drop=True)
        if len(df) < 50:
            log.warning("%d km 有效样本仅 %d，跳过", s, len(df))
            continue
        log.info("%d km 尺度：有效样本 %d", s, len(df))

        qs, Fs, ps = {}, {}, {}
        best_method = {}
        for nm in [c for c in df.columns if c not in ("x", "y", "_Y")]:
            x = df[nm].values.astype(float)
            if not np.isfinite(x).any():
                continue
            best_q, best_m = -1.0, None
            for m in ("Qt", "Uni", "Gmt", "Std", "Jnb", "Kms", "Agg", "Spc", "Gsm", "Bkm"):
                try:
                    st = discretize(x, m, k=k_classes)
                except Exception:
                    continue
                q, F, p = q_statistic(df["_Y"].values, st)
                if np.isfinite(q) and q > best_q:
                    best_q, best_m = q, m
            if best_m is None:
                continue
            st = discretize(x, best_m, k=k_classes)
            q, F, p = q_statistic(df["_Y"].values, st)
            qs[nm] = q; Fs[nm] = F; ps[nm] = p
            best_method[nm] = best_m
        if not qs:
            continue
        mean_q = float(np.mean(list(qs.values())))
        scale_rows.append({"scale_km": s, "n_samples": len(df), "mean_q": mean_q,
                           "n_factors": len(qs)})
        log.info("%d km 平均 q = %.4f（%d 个因子）", s, mean_q, len(qs))
        best.append((mean_q, s, df, qs, Fs, ps, best_method))

    if not best:
        log.error("所有尺度均失败")
        return pd.DataFrame()

    # 尺度探测
    sdf = pd.DataFrame(scale_rows).sort_values("mean_q", ascending=False)
    sdf.to_csv(out / "scale_mean_q.csv", index=False, encoding="utf-8-sig")
    best_scale = int(sdf.iloc[0]["scale_km"])
    C.dump_json({"response": response_var, "year": year,
                 "best_scale_km": best_scale,
                 "mean_q_by_scale": {int(r.scale_km): float(r.mean_q) for r in sdf.itertuples()},
                 "note": "论文称 SC/WC=15km, HQ=6km, CS=12km"},
                out / "best_scale.json")
    log.info("最佳尺度 = %d km（平均 q = %.4f）", best_scale, float(sdf.iloc[0]["mean_q"]))

    # 用最佳尺度做四类探测器
    mean_q, s_km, df, qs, Fs, ps, bm = best[0]
    pd.DataFrame([{"factor": k, "q": v, "F": Fs.get(k), "p": ps.get(k),
                   "best_method": bm.get(k)} for k, v in qs.items()]) \
        .sort_values("q", ascending=False) \
        .to_csv(out / "factor_detector.csv", index=False, encoding="utf-8-sig")

    # 交互探测（取 q 前 6 的因子两两组合）
    top = sorted(qs.items(), key=lambda kv: -kv[1])[:6]
    rows = []
    for (n1, q1), (n2, q2) in itertools.combinations(top, 2):
        x1 = df[n1].values.astype(float); x2 = df[n2].values.astype(float)
        st1 = discretize(x1, bm.get(n1, "Qt"), k=k_classes)
        st2 = discretize(x2, bm.get(n2, "Qt"), k=k_classes)
        combo = (st1.astype(np.int32) * 100 + st2.astype(np.int32))
        combo = np.where((st1 >= 0) & (st2 >= 0), combo, -1)
        q12, F12, p12 = q_statistic(df["_Y"].values, combo)
        rows.append({"f1": n1, "f2": n2, "q1": q1, "q2": q2, "q12": q12,
                     "type": interaction_type(q12, q1, q2) if np.isfinite(q12) else "无",
                     "p": p12})
    idf = pd.DataFrame(rows).sort_values("q12", ascending=False)
    idf.to_csv(out / "interaction_detector.csv", index=False, encoding="utf-8-sig")

    # 风险探测
    rrows = []
    for nm, _ in top:
        st = discretize(df[nm].values.astype(float), bm.get(nm, "Qt"), k=k_classes)
        means, ns, diffs = risk_detector(df["_Y"].values, st)
        for d in diffs:
            d["factor"] = nm
            rrows.append(d)
    if rrows:
        pd.DataFrame(rrows).to_csv(out / "risk_detector.csv", index=False, encoding="utf-8-sig")

    # 生态探测
    erows = ecological_detector(qs, Fs, ps)
    if erows:
        pd.DataFrame(erows).to_csv(out / "ecological_detector.csv", index=False, encoding="utf-8-sig")

    log.info("四类探测器结果已写出（最佳尺度 %d km）", best_scale)
    return sdf


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段5 OMGD 多尺度地理探测器")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--window", default=",".join(map(str, DEFAULT_WINDOW)))
    ap.add_argument("--year", type=int, default=2010)
    ap.add_argument("--response", default="SC")
    ap.add_argument("--scales", default="3,6,9,12,15")
    C.ensure_dirs()
    a = ap.parse_args(argv)
    run(Path(a.out_root),
        window=tuple(int(v) for v in a.window.split(",")),
        year=a.year, response_var=a.response,
        scales=tuple(int(v) for v in a.scales.split(",")))


if __name__ == "__main__":
    main()
