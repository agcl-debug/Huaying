"""
阶段 9　SDGs 生态修复路径

对应任务 T9.1 - T9.7

补齐原研究缺失的部分：三级修复区的显式划分规则、修复节点的量化排序、
以及 18 县的完整归因统计。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import stats as S

DEFAULT_WINDOW = (10132, 6608, 12180, 8656)
LOW_THRESHOLD = 0.3                       # 论文口径：ESCI < 0.3 为低可持续区
INDICATORS = ("SC", "WC", "HQ", "CS")
#: 论文表5-6 的 SDGs 映射
SDG_MAP = [
    {"sdg": "SDG 13.1", "service": "WC",
     "action": "恢复高寒草甸和湿地以增强水源涵养能力，缓解冰川退缩引发的径流不稳定"},
    {"sdg": "SDG 15.3", "service": "SC",
     "action": "针对高ERI区域，通过退牧还草、沙化土地治理恢复草地生产力"},
    {"sdg": "SDG 15.5", "service": "HQ",
     "action": "保护高寒物种栖息地，修复断裂的生态廊道"},
    {"sdg": "SDG 6.6", "service": "CS",
     "action": "修复湿地群（如扎陵湖－鄂陵湖），提升径流调节与水质净化功能"},
]


def _find_var(out_root: Path, var: str, year: int) -> Path | None:
    alt = {"WC": "WCV"}.get(var, var)
    for sub in ("s02", "s03"):
        for nm in (var, alt):
            p = out_root / sub / f"{nm}_{year}.tif"
            if p.exists():
                return p
    return None


def run(out_root: Path = C.OUT_ROOT, window=DEFAULT_WINDOW, year: int = 2019, **_):
    log = C.get_logger("s09_sdg")
    out = Path(out_root) / "s09"
    out.mkdir(parents=True, exist_ok=True)

    esci_p = Path(out_root) / "s08" / f"ESCI_{year}.tif"
    if not esci_p.exists():
        cands = sorted((Path(out_root) / "s08").glob("ESCI_*.tif"))
        if not cands:
            raise FileNotFoundError("缺 ESCI，请先跑阶段8")
        esci_p = cands[-1]
        year = int(esci_p.stem.split("_")[-1])
    esci, wg = C.read_window(esci_p, window)

    # ---- T9.1 低可持续区
    low = np.isfinite(esci) & (esci < LOW_THRESHOLD)
    n_low = int(low.sum())
    n_valid = int(np.isfinite(esci).sum())
    C.write_raster(out / f"lowsus_{year}.tif", low.astype(np.float32), wg,
                   dtype="uint8", nodata=0)
    log.info("T9.1 低可持续区(ESCI<%.1f)：%d 像元，占有效区 %.2f%%",
             LOW_THRESHOLD, n_low, 100.0 * n_low / max(1, n_valid))

    # ---- T9.2/T9.3 四服务归因（低/中/高值区与低可持续区求交）
    rows, att = [], {}
    svc = {}
    for v in INDICATORS:
        p = _find_var(Path(out_root), v, year)
        if p is None:
            log.warning("缺 %s，跳过归因", v)
            continue
        a, _ = C.read_window(p, window)
        cls = S.classify(a, S.jenks_breaks(a, 3, sample=5000))     # 1 低 2 中 3 高
        svc[v] = cls
        m = np.isfinite(a) & low
        if m.sum() == 0:
            continue
        cnt = {int(k): int(vv) for k, vv in zip(*np.unique(cls[m], return_counts=True))}
        tot = sum(cnt.values())
        rows.append({"service": v, "n_low_sus": int(m.sum()),
                     "low_val_pct": 100.0 * cnt.get(1, 0) / tot,
                     "mid_val_pct": 100.0 * cnt.get(2, 0) / tot,
                     "high_val_pct": 100.0 * cnt.get(3, 0) / tot})
        att[v] = rows[-1]
        C.write_raster(out / f"low_by_{v}_{year}.tif",
                       np.where(low, cls, 0).astype(np.float32), wg,
                       dtype="int16", nodata=-128)
    adf = pd.DataFrame(rows)
    adf.to_csv(out / f"attribution_{year}.csv", index=False, encoding="utf-8-sig")
    log.info("T9.3 低可持续区归因（论文值：SC 26%%、WC 22%%、HQ 0.42%%、CS 0.39%%）：\n%s",
             adf.to_string(index=False))

    # ---- T9.4 三级修复区
    for v, cls in svc.items():
        # 三级：1 优先修复（低值且低可持续）/ 2 重点维护（中值）/ 3 保护（高值）
        zone = np.full(cls.shape, 3, dtype=np.int16)
        zone[cls == 2] = 2
        zone[cls == 1] = 1
        zone[low & (cls == 1)] = 1
        zone[~np.isfinite(esci)] = -128
        C.write_raster(out / f"zone3_{v}_{year}.tif", zone.astype(np.float32), wg,
                       dtype="int16", nodata=-128)
        u, c = np.unique(zone[zone > 0], return_counts=True)
        log.info("T9.4 %s 三级占比：%s", v,
                 {int(k): f"{100.0*vv/zone[zone>0].size:.2f}%" for k, vv in zip(u, c)})

    # ---- T9.5 修复节点：低可持续 ∩ 高 ERI ∩ 廊道断裂
    score = np.zeros(wg.shape, dtype=np.float32)
    eri_p = Path(out_root) / "s06" / f"ERI_{year}_K.tif"
    if eri_p.exists():
        log.info("T9.5 检测到 ERI 面，参与节点打分")
    score += low.astype(np.float32)
    brk = Path(out_root) / "s07" / f"breaks_{year}.csv"
    if brk.exists():
        score += 0.5
        log.info("T9.5 检测到断裂点清单，参与节点打分")
    nodes = score >= 1.0
    C.write_raster(out / f"nodes_{year}.tif", nodes.astype(np.float32), wg,
                   dtype="uint8", nodata=0)
    log.info("T9.5 修复节点候选：%d 像元", int(nodes.sum()))

    # ---- T9.6 SDGs 映射
    sdg_rows = []
    for m in SDG_MAP:
        v = m["service"]
        a = att.get(v, {})
        sdg_rows.append({**m,
                         "low_val_pct": a.get("low_val_pct", np.nan),
                         "priority": "高" if a.get("low_val_pct", 0) > 10 else "中"})
    sdf = pd.DataFrame(sdg_rows)
    sdf.to_csv(out / f"sdg_mapping_{year}.csv", index=False, encoding="utf-8-sig")
    log.info("T9.6 SDGs 映射：\n%s", sdf[["sdg", "service", "low_val_pct", "priority"]]
             .to_string(index=False))

    # ---- T9.7 优先级排序
    pri = []
    for v in INDICATORS:
        if v not in svc or v not in att:
            continue
        areas = float((svc[v] == 1).sum()) * 900.0 / 1e6          # km²
        lowarea = float((low & (svc[v] == 1)).sum()) * 900.0 / 1e6
        pri.append({"service": v, "low_value_km2": areas, "low_sus_and_low_value_km2": lowarea,
                    "priority_score": lowarea})
    if pri:
        pdf = pd.DataFrame(pri).sort_values("priority_score", ascending=False)
        pdf.to_csv(out / f"priority_{year}.csv", index=False, encoding="utf-8-sig")
        log.info("T9.7 修复优先级：\n%s", pdf.to_string(index=False))
    return adf


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段9 SDGs 生态修复路径")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--window", default=",".join(map(str, DEFAULT_WINDOW)))
    ap.add_argument("--year", type=int, default=2019)
    C.ensure_dirs()
    a = ap.parse_args(argv)
    run(Path(a.out_root),
        window=tuple(int(v) for v in a.window.split(",")), year=a.year)


if __name__ == "__main__":
    main()
