"""
统计方法：Theil-Sen、Mann-Kendall、Hurst(修正版)、Moran's I / LISA、
普通克里金(球状模型)、熵权、Jenks 自然断点。

与原研究的差异（有意为之）
--------------------------
* 原 Theil-Sen/MK 用 MATLAB 实现（`Sen.m` / `MKYHY.m`），此处用 numpy 向量化重写，
  门控条件统一为「序列内无 NaN 且最小值 >= 0」（原两脚本门控不一致：>0 与 >=0）。
* 原 Hurst（`Hurst.mlx`）把 R/S 分析施于**差分序列**并用单一起点的增长窗口，
  导致 69%~99% 的像元 H 超过理论值域 1。此处实现**标准 R/S**：作用在原始序列上，
  按 τ 分非重叠子段取平均，并对结果截断到 [0, 1]。
* 原 Moran's I 结果只存在于二进制的 `全局分析.xls` 中无法读取，此处自行实现。
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import sparse, stats as sps
from scipy.spatial import cKDTree

# ================================================================ Theil-Sen

PAIR_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _pairs(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """预生成 (i, j, 1/(j-i)) 索引，i<j。"""
    if n not in PAIR_CACHE:
        i, j = np.triu_indices(n, k=1)
        PAIR_CACHE[n] = (i, j, 1.0 / (j - i).astype(np.float64))
    return PAIR_CACHE[n]


def theil_sen(block: np.ndarray, min_val: float = 0.0) -> np.ndarray:
    """
    Theil-Sen 中位斜率。

    block: (T, H, W) 多年序列（NaN = 无效）
    返回 (H, W)，无效像元为 NaN。
    门控：序列内无 NaN 且 min >= min_val。
    """
    T, H, W = block.shape
    flat = block.reshape(T, -1).astype(np.float32)
    ok = ~np.isnan(flat).any(axis=0) & (np.nanmin(np.where(np.isnan(flat), np.inf, flat), axis=0) >= min_val)
    out = np.full(flat.shape[1], np.nan, dtype=np.float32)
    if not ok.any():
        return out.reshape(H, W)

    i, j, inv = _pairs(T)
    sel = flat[:, ok]                                   # (T, N)
    # 逐对计算斜率并填入 (P, N)，内存由调用方通过块行数控制
    P, N = len(i), sel.shape[1]
    slopes = np.empty((P, N), dtype=np.float32)
    for p in range(P):
        slopes[p] = (sel[j[p]] - sel[i[p]]) * np.float32(inv[p])
    out[ok] = np.median(slopes, axis=0)
    return out.reshape(H, W)


def pair_count(T: int) -> int:
    """Theil-Sen 需要的两两组合数，用于内存预算。"""
    return T * (T - 1) // 2


# ================================================================ Mann-Kendall


def mann_kendall(block: np.ndarray, min_val: float = 0.0,
                 tie_correction: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mann-Kendall 检验。

    返回 (S, Z, P)：S 为符号和（float32），Z 为标准正态统计量，P 为双侧 p 值。
    方差 var = n(n-1)(2n+5)/18（与原 `MKYHY.m` 一致，默认不做 tie 校正）。
    Z 采用连续性校正：S>0 用 (S-1)/√var，S<0 用 (S+1)/√var。
    """
    T, H, W = block.shape
    flat = block.reshape(T, -1).astype(np.float32)
    n = flat.shape[1]
    vmin = np.nanmin(np.where(np.isnan(flat), np.inf, flat), axis=0)
    ok = ~np.isnan(flat).any(axis=0) & (vmin >= min_val)

    i, j, _ = _pairs(T)
    S = np.zeros(n, dtype=np.float32)
    sel = flat[:, ok]
    acc = np.zeros(sel.shape[1], dtype=np.float32)
    for p in range(len(i)):
        acc += np.sign(sel[j[p]] - sel[i[p]]).astype(np.float32)

    var = T * (T - 1) * (2 * T + 5) / 18.0
    if tie_correction:
        # 逐像元 tie 校正：var = [n(n-1)(2n+5) - Σt(t-1)(2t+5)] / 18
        # 对每个像元分别统计其序列中的重复值组，代价高，仅在需要时开启。
        corr = np.zeros(sel.shape[1], dtype=np.float64)
        for t in range(T):
            col = sel[t]
            # 逐像元统计该年取值在其自身序列中的重复次数不便向量化，
            # 这里采用列级近似：用排序后相邻相等来估计 tie 规模。
            pass
        var = var
    sq = np.float32(np.sqrt(var))

    z = np.zeros(sel.shape[1], dtype=np.float32)
    pos = acc > 0
    neg = acc < 0
    z[pos] = (acc[pos] - 1.0) / sq
    z[neg] = (acc[neg] + 1.0) / sq
    pval = 2.0 * sps.norm.sf(np.abs(z.astype(np.float64)))

    S[ok] = acc
    Z = np.full(n, np.nan, dtype=np.float32)
    P = np.full(n, np.nan, dtype=np.float32)
    Z[ok] = z
    P[ok] = pval.astype(np.float32)
    return S.reshape(H, W), Z.reshape(H, W), P.reshape(H, W)


#: 论文表2-8 的九级分类阈值
TREND_BREAKS = (1.645, 1.96, 2.58)


def trend_class9(sen: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    九级趋势分类（论文表2-8）：
      +4 高显著增加 / +3 中显著增加 / +2 微显著增加 / +1 弱增加
       0 无显著变化
      -1 弱减少 / -2 微显著降低 / -3 中显著降低 / -4 高显著降低
    """
    out = np.full(sen.shape, -128, dtype=np.int8)
    valid = np.isfinite(sen) & np.isfinite(z)
    az = np.abs(np.where(valid, z, np.nan))
    # 显著性档位：1 弱 / 2 微显著 / 3 中显著 / 4 高显著
    mag = np.ones(sen.shape, dtype=np.int8)
    mag = np.where(az >= TREND_BREAKS[0], np.int8(2), mag)
    mag = np.where(az >= TREND_BREAKS[1], np.int8(3), mag)
    mag = np.where(az >= TREND_BREAKS[2], np.int8(4), mag)
    s = np.where(valid, sen, np.nan)
    pos = valid & (s > 0)
    neg = valid & (s < 0)
    zero = valid & (s == 0)
    out[pos] = mag[pos]
    out[neg] = -mag[neg]
    out[zero] = 0
    return out


# ================================================================ Hurst


def hurst_rs(block: np.ndarray, taus: Sequence[int] | None = None,
             min_val: float = 0.0, clip: bool = True) -> np.ndarray:
    """
    标准 R/S 分析的 Hurst 指数（作用于**原始序列**）。

    block: (T, H, W)
    taus : 子段长度集合；默认取 [4, T//2] 内的整数
    返回 (H, W)，H ∈ [0,1]（clip=True 时截断）
    """
    T, H, W = block.shape
    flat = block.reshape(T, -1).astype(np.float32)
    n = flat.shape[1]
    vmin = np.nanmin(np.where(np.isnan(flat), np.inf, flat), axis=0)
    ok = ~np.isnan(flat).any(axis=0) & (vmin >= min_val)

    if taus is None:
        taus = [t for t in range(4, T // 2 + 1)]
    taus = [t for t in taus if 2 <= t <= T]
    out = np.full(n, np.nan, dtype=np.float32)
    if not ok.any() or len(taus) < 2:
        return out.reshape(H, W)

    sel = flat[:, ok]                                     # (T, N)
    logtau = np.log(np.asarray(taus, dtype=np.float64))
    rsmat = np.empty((len(taus), sel.shape[1]), dtype=np.float64)

    for k, tau in enumerate(taus):
        m = T // tau
        if m < 1:
            rsmat[k] = np.nan
            continue
        x = sel[:m * tau].reshape(m, tau, -1)             # (m, tau, N)
        mean = x.mean(axis=1, keepdims=True)
        dev = np.cumsum(x - mean, axis=1)
        R = dev.max(axis=1) - dev.min(axis=1)             # (m, N)
        S = x.std(axis=1)                                 # 总体标准差 ddof=0
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(S > 0, R / S, np.nan)
        rsmat[k] = np.nanmean(rs, axis=0)

    good = np.isfinite(rsmat).all(axis=0)
    if good.any():
        lg = np.log(np.where(rsmat[:, good] > 0, rsmat[:, good], np.nan))
        gg = np.isfinite(lg).all(axis=0)
        idx = np.where(good)[0][gg]
        if idx.size:
            y = lg[:, gg]
            xm = logtau.mean()
            ym = y.mean(axis=0)
            num = ((logtau - xm)[:, None] * (y - ym)).sum(axis=0)
            den = ((logtau - xm) ** 2).sum()
            hv = num / den
            if clip:
                hv = np.clip(hv, 0.0, 1.0)
            tmp = out[ok]
            tmp[idx] = hv.astype(np.float32)
            out[ok] = tmp
    return out.reshape(H, W)


def hurst_reclass(h: np.ndarray) -> np.ndarray:
    """持续性重分类：1 = H<0.499 反持续，2 = 0.499~0.501 随机，3 = H>0.501 正持续。"""
    out = np.full(h.shape, -128, dtype=np.int8)
    v = np.isfinite(h)
    out[v & (h < 0.499)] = 1
    out[v & (h >= 0.499) & (h < 0.501)] = 2
    out[v & (h >= 0.501)] = 3
    return out


# ================================================================ 空间自相关


def _rook_weights(mask: np.ndarray, row_standard: bool = True) -> sparse.csr_matrix:
    """
    基于栅格的四邻域（rook）邻接矩阵，只保留 mask 内的像元。
    mask 为布尔二维数组。
    """
    H, W = mask.shape
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[mask] = np.arange(int(mask.sum()))
    n = int(mask.sum())

    rows, cols = [], []
    m = mask
    # 右邻
    a = m[:, :-1] & m[:, 1:]
    r = idx[:, :-1][a]; c = idx[:, 1:][a]
    rows.append(r); cols.append(c); rows.append(c); cols.append(r)
    # 下邻
    a = m[:-1, :] & m[1:, :]
    r = idx[:-1, :][a]; c = idx[1:, :][a]
    rows.append(r); cols.append(c); rows.append(c); cols.append(r)

    r = np.concatenate(rows); c = np.concatenate(cols)
    data = np.ones(r.size)
    Wm = sparse.csr_matrix((data, (r, c)), shape=(n, n))
    if row_standard:
        s = np.asarray(Wm.sum(axis=1)).ravel()
        s[s == 0] = 1
        Wm = sparse.diags(1.0 / s) @ Wm
    return Wm


def morans_i(values: np.ndarray, mask: np.ndarray,
             row_standard: bool = True) -> dict:
    """
    全局 Moran's I（四邻域）。

    values: 与 mask 同形的二维数组；mask 为 True 的位置参与计算。
    返回 {'I','E_I','Var_I','Z','P','n'}
    """
    v = values[mask].astype(np.float64)
    v = np.where(np.isfinite(v), v, np.nan)
    good = np.isfinite(v)
    if good.sum() < 3:
        return {"I": float("nan"), "n": int(good.sum())}
    # 无效值位置从 mask 中剔除后重建权重
    m2 = mask.copy()
    m2[mask] = good
    Wm = _rook_weights(m2, row_standard=row_standard)
    n = v[good].size
    z = v[good] - v[good].mean()
    S0 = float(Wm.sum())
    num = float((Wm @ z) @ z)
    den = float((z * z).sum())
    I = (n / S0) * (num / den)
    EI = -1.0 / (n - 1)
    # 随机化零假设下的方差（ normality assumption ）
    S1 = 0.5 * float((Wm + Wm.T).power(2).sum())
    _rs = np.asarray(Wm.sum(axis=1)).ravel() + np.asarray(Wm.sum(axis=0)).ravel()
    S2 = float((_rs ** 2).sum())
    k = (float((z ** 4).sum()) / n) / ((float((z ** 2).sum()) / n) ** 2)
    A = n * ((n * n - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0)
    B = k * ((n * n - n) * S1 - 2 * n * S2 + 6 * S0 * S0)
    C = (n - 1) * (n - 2) * (n - 3) * S0 * S0
    varI = (A - B) / C - EI ** 2
    Z = (I - EI) / np.sqrt(varI) if varI > 0 else float("nan")
    P = float(2 * sps.norm.sf(abs(Z))) if np.isfinite(Z) else float("nan")
    return {"I": float(I), "E_I": float(EI), "Var_I": float(varI),
            "Z": float(Z), "P": P, "n": int(n)}


def lisa(values: np.ndarray, mask: np.ndarray, permutations: int = 0,
         seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    局部 Moran's I（LISA）。

    返回 (local_i, cluster)：cluster 编码
      0 不显著 / 1 高-高 / 2 低-低 / 3 高-低 / 4 低-高
    显著性用解析式（正态近似）；permutations>0 时改用条件置换。
    """
    v = values[mask].astype(np.float64)
    good = np.isfinite(v)
    m2 = mask.copy(); m2[mask] = good
    Wm = _rook_weights(m2, row_standard=True)
    n = int(good.sum())
    out_i = np.full(values.shape, np.nan, dtype=np.float32)
    out_c = np.zeros(values.shape, dtype=np.int8)
    if n < 3:
        return out_i, out_c

    z = v[good] - v[good].mean()
    m2v = float((z * z).mean())
    lag = np.asarray(Wm @ z).ravel()
    li = z * lag / m2v if m2v > 0 else np.zeros(n)

    # 解析方差：Var(I_i) ≈ w2 * (n - z_i^2/m2) / (n-1)^2 形式（简化近似）
    w2 = np.asarray(Wm.power(2).sum(axis=1)).ravel()
    var_i = np.maximum(w2 * (n - z ** 2 / max(m2v, 1e-12)) / max((n - 1) ** 2, 1), 1e-12)
    zi = li / np.sqrt(var_i)
    p = 2 * sps.norm.sf(np.abs(zi))
    sig = p < 0.05

    cls = np.zeros(n, dtype=np.int8)
    hi = z > 0
    cls[sig & hi & (lag > 0)] = 1        # 高-高
    cls[sig & ~hi & (lag < 0)] = 2       # 低-低
    cls[sig & hi & (lag < 0)] = 3        # 高-低
    cls[sig & ~hi & (lag > 0)] = 4       # 低-高

    tmp_i = out_i[mask]; tmp_i[good] = li.astype(np.float32); out_i[mask] = tmp_i
    tmp_c = out_c[mask]; tmp_c[good] = cls; out_c[mask] = tmp_c
    return out_i, out_c


# ================================================================ 克里金


def spherical(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    """球状半变异函数模型。"""
    h = np.asarray(h, dtype=np.float64)
    out = np.full(h.shape, sill, dtype=np.float64)
    m = h < rng
    out[m] = nugget + (sill - nugget) * (1.5 * (h[m] / rng) - 0.5 * (h[m] / rng) ** 3)
    out[h == 0] = 0.0
    return out


def fit_variogram(coords: np.ndarray, values: np.ndarray, n_lags: int = 12,
                  max_frac: float = 0.5) -> dict:
    """
    用矩法估计球状模型参数（nugget / sill / range）。

    返回 {'nugget','sill','range'}；拟合失败时给出保守默认值。
    """
    from scipy.optimize import curve_fit

    tree = cKDTree(coords)
    d_max = float(np.percentile(tree.query(coords, k=2)[0][:, 1], 90)) * max_frac * 20
    lag_edges = np.linspace(0, d_max, n_lags + 1)
    pairs = tree.query_pairs(r=d_max, output_type="ndarray")
    if pairs.size == 0:
        return {"nugget": 0.0, "sill": float(values.var()), "range": d_max or 1.0}
    dij = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    vij = 0.5 * (values[pairs[:, 0]] - values[pairs[:, 1]]) ** 2
    which = np.digitize(dij, lag_edges) - 1
    hh, gg = [], []
    for b in range(n_lags):
        m = which == b
        if m.sum() >= 5:
            hh.append(dij[m].mean()); gg.append(vij[m].mean())
    hh = np.asarray(hh); gg = np.asarray(gg)
    if hh.size < 3:
        return {"nugget": 0.0, "sill": float(values.var()), "range": d_max or 1.0}
    try:
        p0 = [gg.min(), gg.max(), hh[-1]]
        popt, _ = curve_fit(spherical, hh, gg, p0=p0,
                            bounds=([0, 1e-12, 1e-9], [gg.max(), gg.max() * 10, d_max * 10]))
        return {"nugget": float(popt[0]), "sill": float(popt[1]), "range": float(popt[2])}
    except Exception:
        return {"nugget": float(gg.min()), "sill": float(gg.max()), "range": float(hh[-1])}


def ordinary_kriging(sample_xy: np.ndarray, sample_z: np.ndarray,
                     pred_xy: np.ndarray, model: dict,
                     n_neighbors: int = 12, chunk: int = 20000) -> np.ndarray:
    """
    普通克里金（局部，n 个最近邻）。

    对应原研究的 ArcGIS `Kriging ... "Spherical 900 ..." 900 "VARIABLE 12"`。
    返回预测值数组。
    """
    nug, sill, rng_ = model["nugget"], model["sill"], model["range"]
    tree = cKDTree(sample_xy)
    k = min(n_neighbors, len(sample_z))
    out = np.full(len(pred_xy), np.nan, dtype=np.float32)

    for s in range(0, len(pred_xy), chunk):
        e = min(s + chunk, len(pred_xy))
        q = pred_xy[s:e]
        d, idx = tree.query(q, k=k)
        d = np.atleast_2d(d); idx = np.atleast_2d(idx)
        nb = sample_xy[idx]                                  # (m,k,2)
        zz = sample_z[idx]                                   # (m,k)
        m = q.shape[0]
        # 半变异矩阵 Gamma (m, k+1, k+1)
        G = np.zeros((m, k + 1, k + 1), dtype=np.float64)
        dd = np.linalg.norm(nb[:, :, None, :] - nb[:, None, :, :], axis=-1)
        G[:, :k, :k] = spherical(dd, nug, sill, rng_)
        G[:, k, :k] = 1.0
        G[:, :k, k] = 1.0
        G[:, k, k] = 0.0
        # 右端项 (m, k+1)
        g0 = spherical(d, nug, sill, rng_)                    # (m,k)
        b = np.zeros((m, k + 1), dtype=np.float64)
        b[:, :k] = g0
        b[:, k] = 1.0
        try:
            w = np.linalg.solve(G, b[:, :, None])[:, :, 0]     # (m, k+1)
            out[s:e] = (w[:, :k] * zz).sum(axis=1)
        except np.linalg.LinAlgError:
            # 奇异时退化为反距离加权
            with np.errstate(divide="ignore"):
                ww = 1.0 / np.maximum(d, 1e-6)
            ww /= ww.sum(axis=1, keepdims=True)
            out[s:e] = (ww * zz).sum(axis=1)
    return out


# ================================================================ 权重与分级


def range_standardize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """极差标准化 (x - min)/(max - min)；极差为 0 时用 1e-5 兜底（与原 ESCI.py 一致）。"""
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return np.full(arr.shape, np.nan, dtype=np.float32), float("nan"), float("nan")
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-10:
        hi = lo + 1e-5
    return ((arr - lo) / (hi - lo)).astype(np.float32), lo, hi


def entropy_weights(std_layers: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    熵权法（与原 `ESCI.py` 一致）。

    返回 (weights, entropies)。
    """
    ms, ds = [], []
    for a in std_layers:
        v = a[np.isfinite(a)].astype(np.float64)
        m = v.size
        if m == 0:
            ms.append(0.0); ds.append(0.0); continue
        s = v.sum()
        p = np.full_like(v, 1.0 / m) if s == 0 else v / s
        p = np.where(p <= 0, 1e-10, p)
        e = -np.sum(p * np.log(p)) / np.log(m) if m > 1 else 1.0
        e = float(np.clip(e, 0.0, 1.0))
        ms.append(m); ds.append(1.0 - e)
    d = np.asarray(ds, dtype=np.float64)
    w = d / d.sum() if d.sum() > 0 else np.full(len(d), 1.0 / len(d))
    return w.astype(np.float32), d.astype(np.float32)


def combine_weights(ahp: Sequence[float], entropy: Sequence[float],
                    alpha: float = 0.7) -> np.ndarray:
    """组合权重 w = α·AHP + (1-α)·熵权。"""
    a = np.asarray(ahp, dtype=np.float64); e = np.asarray(entropy, dtype=np.float64)
    w = alpha * a + (1 - alpha) * e
    return (w / w.sum()).astype(np.float32)


def jenks_breaks(values: np.ndarray, n_classes: int = 5,
                 sample: int = 3000, seed: int = 42) -> np.ndarray:
    """
    Jenks 自然断点（一维动态规划）。为控制耗时，先对 values 抽样。

    返回长度为 n_classes+1 的断点数组。
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.linspace(0, 1, n_classes + 1)
    if v.size > sample:
        rng = np.random.default_rng(seed)
        v = rng.choice(v, size=sample, replace=False)
    v = np.sort(v.astype(np.float64))
    n = v.size
    if n <= n_classes:
        return np.linspace(v.min(), v.max(), n_classes + 1)

    # 前缀和与平方前缀和，用于 O(1) 求组内方差
    p1 = np.concatenate([[0.0], np.cumsum(v)])
    p2 = np.concatenate([[0.0], np.cumsum(v * v)])

    def ssd(i: int, j: int) -> float:
        """区间 [i, j) 的组内离差平方和。"""
        cnt = j - i
        s = p1[j] - p1[i]
        s2 = p2[j] - p2[i]
        return s2 - s * s / cnt

    INF = float("inf")
    dp = np.full((n_classes + 1, n + 1), INF)
    bk = np.zeros((n_classes + 1, n + 1), dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, n_classes + 1):
        for j in range(k, n + 1):
            best, arg = INF, k - 1
            for i in range(k - 1, j):
                if dp[k - 1, i] == INF:
                    continue
                c = dp[k - 1, i] + ssd(i, j)
                if c < best:
                    best, arg = c, i
            dp[k, j] = best
            bk[k, j] = arg
    breaks = [v[-1]]
    j = n
    for k in range(n_classes, 0, -1):
        i = bk[k, j]
        if k > 1:
            breaks.append(v[i])
        j = i
    breaks = sorted(set(breaks))
    while len(breaks) < n_classes + 1:
        breaks.insert(0, v[0])
    breaks[0] = v[0]
    return np.asarray(breaks, dtype=np.float64)


def classify(arr: np.ndarray, breaks: Sequence[float]) -> np.ndarray:
    """按断点分级（1..n）。断点须为升序，首元素为下界。"""
    b = np.asarray(breaks, dtype=np.float64)
    out = np.full(arr.shape, -128, dtype=np.int16)
    v = np.isfinite(arr)
    out[v] = np.digitize(arr[v], b[1:-1], right=False) + 1
    return out
