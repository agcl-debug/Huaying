"""
自检：用已知答案的合成数据验证统计与几何模块是否实现正确。

运行：python -m repro.selftest
"""
from __future__ import annotations

import numpy as np

from . import common as C
from . import geo as G
from . import stats as S


def _ok(name: str, cond: bool, info: str = "") -> bool:
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}  {info}")
    return cond


def test_theil_sen() -> bool:
    print("Theil-Sen")
    H, W, T = 8, 9, 12
    t = np.arange(T, dtype=np.float32)
    slope_true = np.linspace(-1, 3, H * W).reshape(H, W).astype(np.float32)
    blk = slope_true[None] * t[:, None, None] + 5.0        # y = 5 + s*t
    out = S.theil_sen(blk)
    err = np.nanmax(np.abs(out - slope_true))
    return _ok("线性序列斜率还原", err < 1e-4, f"max_err={err:.2e}")


def test_mann_kendall() -> bool:
    print("Mann-Kendall")
    T = 20
    up = np.tile(np.arange(T, dtype=np.float32)[:, None, None], (1, 4, 4))
    S_, Z, P = S.mann_kendall(up)
    exp_S = T * (T - 1) / 2
    a = _ok("单调递增 S = n(n-1)/2", abs(S_[0, 0] - exp_S) < 1e-6, f"S={S_[0,0]:.0f}")
    b = _ok("单调递增 Z > 0 且显著", Z[0, 0] > 2.58 and P[0, 0] < 0.01,
            f"Z={Z[0,0]:.2f} P={P[0,0]:.3g}")
    flat = np.tile(np.full(T, 7.0, np.float32)[:, None, None], (1, 4, 4))
    S2, Z2, _ = S.mann_kendall(flat)
    c = _ok("常量序列 S=0 Z=0", S2[0, 0] == 0 and Z2[0, 0] == 0)
    return a and b and c


def test_trend_class9() -> bool:
    print("趋势九级分类")
    sen = np.array([[1, -1, 1, -1, 0, 1, -1, np.nan]], dtype=np.float32)
    z = np.array([[0.5, 3.0, 2.0, 1.7, 5.0, np.nan, 1.0, 1.0]], dtype=np.float32)
    out = S.trend_class9(sen, z)
    exp = np.array([[1, -4, 3, -2, 0, -128, -1, -128]], dtype=np.int8)
    return _ok("九级编码", (out == exp).all(), f"got={out.ravel().tolist()}")


def test_hurst() -> bool:
    print("Hurst (标准 R/S)")
    rng = np.random.default_rng(0)
    T = 64
    # 白噪声（i.i.d.）-> 理论 H = 0.5（注意：随机游走在 R/S 下理论 H ≈ 0，不能用作 0.5 的参照）
    wn = rng.normal(size=(T, 8, 8)).astype(np.float32)
    wn = wn - wn.min(axis=0, keepdims=True) + 1.0            # 保证非负以通过门控
    h_wn = S.hurst_rs(wn)
    m_wn = float(np.nanmean(h_wn))
    # 强趋势 -> H 应接近 1
    tr = np.tile(np.arange(T, dtype=np.float32)[:, None, None], (1, 8, 8))
    h_tr = S.hurst_rs(tr)
    m_tr = float(np.nanmean(h_tr))
    a = _ok("白噪声 H ≈ 0.5", 0.35 <= m_wn <= 0.70, f"H={m_wn:.3f}")
    b = _ok("强趋势 H → 1", m_tr > 0.85, f"H={m_tr:.3f}")
    c = _ok("值域 [0,1]", np.nanmin(h_wn) >= 0 and np.nanmax(h_tr) <= 1.0,
            f"range=[{np.nanmin(h_wn):.3f},{np.nanmax(h_tr):.3f}]")
    return a and b and c


def test_moran() -> bool:
    print("Moran's I")
    n = 30
    yy, xx = np.mgrid[0:n, 0:n]
    grad = (xx + yy).astype(np.float64)
    mask = np.ones((n, n), bool)
    r1 = S.morans_i(grad, mask)
    chex = ((xx + yy) % 2).astype(np.float64)
    r2 = S.morans_i(chex, mask)
    a = _ok("平滑梯度 I → 1", r1["I"] > 0.85, f"I={r1['I']:.3f}")
    b = _ok("棋盘 I → -1", r2["I"] < -0.7, f"I={r2['I']:.3f}")
    return a and b


def test_lisa() -> bool:
    print("LISA")
    n = 25
    v = np.zeros((n, n))
    c = n // 2
    v[c, c] = 100.0                     # 单个孤高像元：中心应为高-低，四邻应为低-高
    mask = np.ones((n, n), bool)
    li, cls = S.lisa(v, mask)
    center = cls[c, c]
    nb = [cls[c - 1, c], cls[c + 1, c], cls[c, c - 1], cls[c, c + 1]]
    a = _ok("孤高像元中心判为高-低(HL)", center == 3, f"中心={center}")
    b = _ok("四邻判为低-高(LH)", all(x == 4 for x in nb), f"邻居={nb}")
    return a and b


def test_kriging() -> bool:
    print("普通克里金")
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1000, size=(120, 2))
    z = 0.002 * pts[:, 0] + 0.001 * pts[:, 1] + rng.normal(0, 0.05, 120)
    model = S.fit_variogram(pts, z, n_lags=8)
    tgt = rng.uniform(0, 1000, size=(200, 2))
    pred = S.ordinary_kriging(pts, z, tgt, model, n_neighbors=12)
    truth = 0.002 * tgt[:, 0] + 0.001 * tgt[:, 1]
    rmse = float(np.sqrt(np.nanmean((pred - truth) ** 2)))
    return _ok("平面场插值 RMSE 小", rmse < 0.15, f"RMSE={rmse:.4f}  model={ {k: round(v,4) for k,v in model.items()} }")


def test_jenks() -> bool:
    print("Jenks 自然断点")
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(0, 1, 500), rng.normal(20, 1, 500)])
    b = S.jenks_breaks(v, n_classes=2, sample=1000)
    # 断点值取的是"上类的最小值"，应落在两峰之间的空隙里
    n_low = int((v < b[1]).sum())
    ok_break = 10.0 < b[1] < 20.0
    ok_sep = 450 <= n_low <= 550
    return _ok("断点落在双峰空隙且两类各约半数",
               ok_break and ok_sep,
               f"break={b[1]:.2f} n_low={n_low} breaks={np.round(b,2).tolist()}")


def test_entropy_weights() -> bool:
    print("熵权法")
    rng = np.random.default_rng(3)
    a = rng.random(5000).astype(np.float32)          # 高熵 -> 低权重
    b = np.concatenate([np.zeros(4500), np.ones(500)]).astype(np.float32)   # 低熵 -> 高权重
    w, d = S.entropy_weights([a, b])
    return _ok("低熵指标权重更大", w[1] > w[0], f"w={np.round(w,4).tolist()} d={np.round(d,4).tolist()}")


def test_grid_and_io() -> bool:
    print("公共模块 / 栅格 I/O")
    g = C.get_base_grid()
    a = _ok("基准网格 22313x15265", g.width == 22313 and g.height == 15265,
            f"{g.width}x{g.height}")
    b = _ok("像元面积 900 m²", abs(g.pixel_area_m2 - 900.0) < 1e-6, f"{g.pixel_area_m2}")
    # 小窗口读 + 掩膜
    arr, gw = C.read_window(C.RUSLE / r"R_factor\R_2000.tif", (10000, 7000, 10200, 7200))
    c = _ok("窗口读取形状", arr.shape == (200, 200), f"{arr.shape}")
    mk = C.load_mask((10000, 7000, 10200, 7200))
    d = _ok("掩膜与窗口同形", mk.shape == arr.shape, f"{mk.shape}")
    # 写回读一致性
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.gettempdir()) / "repro_selftest.tif"
    C.write_raster(tmp, arr, gw)
    arr2, _ = C.read_window(tmp, (0, 0, gw.width, gw.height))
    e = _ok("写出/读回一致", np.allclose(np.nan_to_num(arr), np.nan_to_num(arr2), atol=1e-3))
    try:
        tmp.unlink()
    except Exception:
        pass
    return a and b and c and d and e


def main() -> int:
    print("=" * 62)
    print("复现工程 · 模块自检")
    print("=" * 62)
    tests = [test_grid_and_io, test_theil_sen, test_mann_kendall, test_trend_class9,
             test_hurst, test_moran, test_lisa, test_kriging, test_jenks,
             test_entropy_weights]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            print(f"  [FAIL] {t.__name__} 抛异常: {type(e).__name__}: {e}")
            results.append(False)
        print()
    n_ok = sum(results)
    print("=" * 62)
    print(f"结果：{n_ok}/{len(results)} 组通过")
    print("=" * 62)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
