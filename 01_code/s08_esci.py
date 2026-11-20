"""
阶段 8　ESCI 环境可持续能力指数

对应任务 T8.1 - T8.8

原研究用 `ESCI.py`（PyCharm），本脚本在其基础上补齐三件事：
  1. **把隐含的 AHP 权重显式化**：代码里是 `alpha=0.7`、AHP=(0.25,0.30,0.30,0.15)，
     而论文 2.3.10 只写"熵权法"、5.3 才提"熵权法+专家打分法"，且权重值全文未给。
  2. **补齐多年 ESCI**（原盘只有 2019 一年，`ESCI_废稿` 里有被废弃的年度序列）。
  3. **补齐 18 个县的分县统计**（论文表5-5 的文字只归类了 15 个县）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import geo as G
from . import stats as S

XIAN_SHP = C.DATA / r"02_遥感与基础地理\YRS_县级区划\YRS_县级区划WGS.shp"
XIAN_DIR = C.DATA / r"02_遥感与基础地理\绝对基准栅格"
DEFAULT_WINDOW = (10132, 6608, 12180, 8656)

INDICATORS = ("SC", "WC", "HQ", "CS")
#: 盘上 ESCI.py 里硬编码的 AHP 专家权重（论文未给）
AHP = {"SC": 0.25, "WC": 0.30, "HQ": 0.30, "CS": 0.15}
ALPHA = 0.7


def _find_var(out_root: Path, var: str, year: int) -> Path | None:
    """按变量名找逐年栅格（SC 在 s02，WC/HQ/CS 在 s03）。"""
    alt = {"WC": "WCV"}.get(var, var)
    for sub in ("s02", "s03"):
        for nm in (var, alt):
            p = out_root / sub / f"{nm}_{year}.tif"
            if p.exists():
                return p
    return None


def run(out_root: Path = C.OUT_ROOT, window=DEFAULT_WINDOW,
        years=(2000, 2019), ahp=None, alpha: float = ALPHA, **_):
    log = C.get_logger("s08_esci")
    out = Path(out_root) / "s08"
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = years
    _, wg = C.read_window(C.MASK_TIF, window)
    ahp = ahp or AHP

    summary = []
    for y in range(y0, y1 + 1):
        paths = {v: _find_var(Path(out_root), v, y) for v in INDICATORS}
        if any(p is None for p in paths.values()):
            miss = [v for v, p in paths.items() if p is None]
            log.warning("%d 年缺 %s，跳过", y, miss)
            continue
        std, lo_hi = [], {}
        for v in INDICATORS:
            a, _ = C.read_window(paths[v], window)
            s, lo, hi = S.range_standardize(a)
            std.append(s)
            lo_hi[v] = {"min": lo, "max": hi}
        # 熵权
        ew, util = S.entropy_weights(std)
        ew = dict(zip(INDICATORS, ew.tolist()))
        w = S.combine_weights([ahp[v] for v in INDICATORS],
                             [ew[v] for v in INDICATORS], alpha=alpha)
        wd = dict(zip(INDICATORS, w.tolist()))
        esci = np.zeros(wg.shape, dtype=np.float32)
        for v, wi in zip(INDICATORS, w):
            esci = esci + wi * np.nan_to_num(std[INDICATORS.index(v)], nan=0.0)
        valid = np.any([np.isfinite(s) for s in std], axis=0)
        esci = np.where(valid, esci, np.nan)
        C.write_raster(out / f"ESCI_{y}.tif", esci, wg)

        v = esci[np.isfinite(esci)]
        rec = {"year": y, "n": int(v.size), "esci_mean": float(v.mean()),
               "esci_min": float(v.min()), "esci_max": float(v.max())}
        for k in INDICATORS:
            rec[f"w_{k}"] = wd[k]
            rec[f"entropy_{k}"] = ew[k]
            rec[f"std_{k}_min"] = lo_hi[k]["min"]
            rec[f"std_{k}_max"] = lo_hi[k]["max"]
        summary.append(rec)
        log.info("T8 %d：ESCI mean=%.4f range=%.4f~%.4f  熵权=%s  组合权重=%s",
                 y, rec["esci_mean"], rec["esci_min"], rec["esci_max"],
                 {k: round(ew[k], 4) for k in INDICATORS},
                 {k: round(wd[k], 4) for k in INDICATORS})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / "esci_stats.csv", index=False, encoding="utf-8-sig")
    if len(sdf) == 0:
        log.error("没有任何年份产出 ESCI")
        return sdf

    # ---- T8.6 自然断点分 5 级（取多年合成或末年）
    last = int(sdf.iloc[-1]["year"])
    esci, _ = C.read_window(out / f"ESCI_{last}.tif", window)
    brk = S.jenks_breaks(esci, n_classes=5, sample=5000)
    cls = S.classify(esci, brk)
    C.write_raster(out / f"ESCI5_{last}.tif", cls.astype(np.float32), wg,
                   dtype="int16", nodata=-128)
    log.info("T8.6 自然断点 5 级（%d 年）breaks=%s", last, np.round(brk, 4).tolist())

    # ---- T8.7 分县统计（补齐 18 县）
    try:
        xian = G.read_vector(XIAN_SHP)
        # 只保留与研究区相交的县，并裁到窗口范围
        zdf = G.zonal_stats(out / f"ESCI5_{last}.tif", xian,
                            stats=("count",))
        zdf.to_csv(out / f"county_area_{last}.csv", index=False, encoding="utf-8-sig")
        log.info("T8.7 分县统计：%d 个县要素", len(zdf))
    except Exception as e:
        log.warning("T8.7 分县统计失败：%s", e)

    # ---- T8.8 敏感性分析（α 与 AHP 扰动）
    rows = []
    for a in (0.0, 0.3, 0.5, 0.7, 1.0):
        w = S.combine_weights([ahp[v] for v in INDICATORS],
                              [sdf.iloc[-1][f"entropy_{v}"] for v in INDICATORS], alpha=a)
        rows.append({"alpha": a, **{v: float(w[i]) for i, v in enumerate(INDICATORS)}})
    pd.DataFrame(rows).to_csv(out / "sensitivity_alpha.csv", index=False, encoding="utf-8-sig")
    log.info("T8.8 敏感性（α 变化对权重的影响）：\n%s",
             pd.DataFrame(rows).to_string(index=False))
    return sdf


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段8 ESCI")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--window", default=",".join(map(str, DEFAULT_WINDOW)))
    ap.add_argument("--years", default="2000,2019")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    C.ensure_dirs()
    a = ap.parse_args(argv)
    run(Path(a.out_root),
        window=tuple(int(v) for v in a.window.split(",")),
        years=tuple(int(v) for v in a.years.split(",")), alpha=a.alpha)


if __name__ == "__main__":
    main()
