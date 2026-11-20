"""
阶段 4　趋势与持续性

对应任务 T4.1 - T4.8

对五套序列逐年计算 Theil-Sen 斜率、Mann-Kendall 的 S/Z/P、九级趋势分类，
以及**修正版 Hurst**（标准 R/S，作用于原始序列并截断到 [0,1]）。

对原研究的三个修正
------------------
1. 门控统一为「无 NaN 且 min >= 0」（原 `Sen.m` 用 >0、`MKYHY.m` 用 >=0，不一致）。
2. Hurst 改用标准 R/S（原版施于差分序列，致 69%~99% 像元 H>1）。
3. 补齐四服务趋势（原盘只有 SC 保留了 .m 脚本）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import stats as S

DEFAULT_WINDOW = (10132, 6608, 12180, 8656)

#: 变量 -> (逐年文件所在目录, 文件名模板)
VAR_DIRS = {
    "SC":  ("s02", "SC_{y}.tif"),
    "SE":  ("s02", "SE_{y}.tif"),
    "WCV": ("s03", "WCV_{y}.tif"),
    "HQ":  ("s03", "HQ_{y}.tif"),
    "CS":  ("s03", "CS_{y}.tif"),
}


def load_stack(out_root: Path, var: str, window, years) -> tuple[np.ndarray, C.Grid | None]:
    sub, tpl = VAR_DIRS[var]
    paths = [out_root / sub / tpl.format(y=y) for y in years]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return np.empty((0,)), None
    arrs, grid = [], None
    for p in paths:
        a, g = C.read_window(p, window)
        arrs.append(a)
        grid = g
    return np.stack(arrs, axis=0), grid


def run(out_root: Path = C.OUT_ROOT, window=DEFAULT_WINDOW,
        years=(2000, 2019), variables=("SC", "SE", "WCV", "HQ", "CS"), **_):
    log = C.get_logger("s04_trend")
    out = Path(out_root) / "s04"
    out.mkdir(parents=True, exist_ok=True)
    ys = list(range(years[0], years[1] + 1))
    summary = []

    for var in variables:
        st, grid = load_stack(Path(out_root), var, window, ys)
        if st.size == 0:
            log.warning("%s 无输入序列，跳过", var)
            continue
        T = st.shape[0]
        if grid is None:
            _, grid = C.read_window(C.MASK_TIF, window)
        log.info("%s：序列 %d 年  形状 %s", var, T, st.shape)

        with C.Timer(log, f"{var} Theil-Sen"):
            sen = S.theil_sen(st)
        with C.Timer(log, f"{var} Mann-Kendall"):
            _, z, p = S.mann_kendall(st)
        trend9 = S.trend_class9(sen, z)
        with C.Timer(log, f"{var} Hurst(标准R/S)"):
            hur = S.hurst_rs(st)
        hurR = S.hurst_reclass(hur)

        C.write_raster(out / f"{var}_Sen.tif", sen, grid)
        C.write_raster(out / f"{var}_MK_Z.tif", z, grid)
        C.write_raster(out / f"{var}_MK_P.tif", p, grid)
        C.write_raster(out / f"{var}_Trend9.tif", trend9, grid, dtype="int16", nodata=-128)
        C.write_raster(out / f"{var}_Hurst.tif", hur, grid)
        C.write_raster(out / f"{var}_HurstR.tif", hurR, grid, dtype="int8", nodata=-128)

        v = np.isfinite(sen)
        az = np.abs(z)
        rec = {
            "var": var, "n_years": T, "n_valid": int(v.sum()),
            "sen_mean": float(np.nanmean(sen)),
            "z_mean": float(np.nanmean(z)),
            "trend_pos_pct": float((trend9 > 0).sum() / max(1, v.sum()) * 100),
            "trend_neg_pct": float((trend9 < 0).sum() / max(1, v.sum()) * 100),
            "sig_pos_165_pct": float((v & (sen > 0) & (az >= 1.645)).sum() / max(1, v.sum()) * 100),
            "sig_pos_196_pct": float((v & (sen > 0) & (az >= 1.96)).sum() / max(1, v.sum()) * 100),
            "hurst_mean": float(np.nanmean(hur)),
            "hurst_gt1_pct": float((hur > 1.0).sum() / max(1, np.isfinite(hur).sum()) * 100),
            "h_reverse_pct": float((hurR == 1).sum() / max(1, (hurR > -128).sum()) * 100),
            "h_random_pct": float((hurR == 2).sum() / max(1, (hurR > -128).sum()) * 100),
            "h_persist_pct": float((hurR == 3).sum() / max(1, (hurR > -128).sum()) * 100),
        }
        summary.append(rec)
        log.info("%s 完成：Sen均值=%.4g  正趋势=%.1f%%  显著正(1.645)=%.1f%%  "
                 "Hurst均值=%.3f  H>1占比=%.1f%%  正持续=%.1f%%",
                 var, rec["sen_mean"], rec["trend_pos_pct"], rec["sig_pos_165_pct"],
                 rec["hurst_mean"], rec["hurst_gt1_pct"], rec["h_persist_pct"])

    df = pd.DataFrame(summary)
    csv = out / "stats_trend.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    log.info("趋势统计写入 %s", csv)

    # 与盘上 FenXi_* 对照（若窗口内存在同名栅格则给出均值对比）
    ref_dir = C.ANALYSIS / "趋势与Hurst"
    ref_rows = []
    for var, sub in (("SC", "FenXi_SC"), ("HQ", "FenXi_HQ"),
                     ("WCV", "Fenxi_WC"), ("CS", "FenXi_CS")):
        d = ref_dir / sub
        if not d.exists():
            continue
        for nm, f in (("Sen", "Sen_Trend.tif"), ("Hurst", "Hurst_Index.tif"),
                      ("MK_Z", "MK_Z_Value.tif")):
            p = d / f
            if not p.exists():
                continue
            try:
                a, _ = C.read_window(p, window)
            except Exception:
                continue
            vv = a[np.isfinite(a)]
            if vv.size:
                ref_rows.append({"var": var, "metric": nm, "ref_mean": float(vv.mean()),
                                 "ref_min": float(vv.min()), "ref_max": float(vv.max())})
    if ref_rows:
        rdf = pd.DataFrame(ref_rows)
        rdf.to_csv(out / "stats_trend_ref.csv", index=False, encoding="utf-8-sig")
        log.info("盘上 FenXi_* 对照：\n%s", rdf.to_string(index=False))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段4 趋势与持续性")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--window", default=",".join(map(str, DEFAULT_WINDOW)))
    ap.add_argument("--years", default="2000,2019")
    ap.add_argument("--vars", default="SC,SE,WCV,HQ,CS")
    C.ensure_dirs()
    a = ap.parse_args(argv)
    run(Path(a.out_root),
        window=tuple(int(v) for v in a.window.split(",")),
        years=tuple(int(v) for v in a.years.split(",")),
        variables=tuple(a.vars.split(",")))


if __name__ == "__main__":
    main()
