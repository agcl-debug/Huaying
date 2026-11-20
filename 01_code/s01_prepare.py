"""
阶段 1　研究区界定与数据准备

对应任务 T1.1 - T1.7

做四件事：
  1. 读研究区边界并核验坐标系；
  2. 写出 Krasovsky_1940_Albers 的权威定义（取自 mask.tif）与边界投影版；
  3. 核验基准网格（22313 x 15265、30 m、有效像元数）；
  4. 核验多尺度渔网（3/6/9/12/15 km）与 9 km 风险评价单元（1,682）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from . import common as C
from . import geo as G

SRYR_SHP = C.DATA / r"02_遥感与基础地理\绝对基准栅格\FSRYR.shp"
SAN_SHP = C.DATA / r"02_遥感与基础地理\三省行政界限\San.shp"
XIAN_SHP = C.DATA / r"02_遥感与基础地理\YRS_县级区划\YRS_县级区划WGS.shp"
FISHNET_DIR = C.DATA / r"05_分析与统计\Factor_地理探测因子\多尺度格网"
LERI_UNIT_SHP = C.DATA / r"02_遥感与基础地理\YRS_渔网\研究区9km格网.shp"

#: 论文 2.3.7 声明的五档尺度的格网单元数
EXPECT_FISHNET = {3: 34272, 6: 8624, 9: 3825, 12: 2184, 15: 1395}
#: 论文 2.3.8 声明的 9 km 风险评价单元数
EXPECT_LERI_UNITS = 1682


def run(out_root: Path = C.OUT_ROOT, window=None, **_):
    log = C.get_logger("s01_prepare")
    out = Path(out_root) / "s01"
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": 1, "tasks": {}}

    # ---- T1.1 研究区边界 -------------------------------------------------
    with C.Timer(log, "T1.1 读取并核验研究区边界"):
        rep = {}
        for name, p in [("FSRYR", SRYR_SHP), ("San", SAN_SHP), ("县级区划", XIAN_SHP)]:
            if not p.exists():
                rep[name] = {"exists": False, "path": str(p)}
                continue
            g = gpd.read_file(p)
            rep[name] = {
                "exists": True, "path": str(p), "n_features": int(len(g)),
                "crs": g.crs.to_string() if g.crs else None,
                "geom_type": str(sorted(set(g.geom_type))),
                "bounds": [float(v) for v in g.total_bounds],
            }
            log.info("%s: %d 要素, CRS=%s", name, len(g), g.crs.to_string()[:50] if g.crs else None)
        report["tasks"]["T1.1"] = rep

    # ---- T1.2 目标投影 ---------------------------------------------------
    with C.Timer(log, "T1.2 写出 Krasovsky_1940_Albers 定义"):
        crs = G.albers_crs()
        wkt_path = out / "krasovsky_albers.wkt"
        try:
            from pyproj import CRS as PjCRS
            wkt_text = PjCRS.from_wkt(crs.to_wkt()).to_wkt(pretty=True)
        except Exception:
            wkt_text = crs.to_wkt()
        wkt_path.write_text(wkt_text, encoding="utf-8")
        C.dump_json({"proj4": G.KRASOVSKY_ALBERS_PROJ4,
                     "wkt": crs.to_wkt()},
                    out / "krasovsky_albers.json")
        units = None
        try:
            units = crs.linear_units if crs.is_projected else crs.units
        except Exception:
            units = None
        report["tasks"]["T1.2"] = {
            "wkt_file": str(wkt_path),
            "is_projected": bool(crs.is_projected),
            "units": units,
        }
        log.info("CRS = %s  units=%s", crs.to_string()[:70], units)

    # ---- T1.3 边界投影转换 -----------------------------------------------
    with C.Timer(log, "T1.3 边界投影转换(WGS84 -> Albers)"):
        if SRYR_SHP.exists():
            g = G.read_vector(SRYR_SHP, to_albers=True)
            gp = out / "sryr_albers.gpkg"
            G.write_vector(g, gp)
            b = g.total_bounds
            area_km2 = float(g.geometry.area.sum()) / 1e6 if g.crs.is_projected else float("nan")
            report["tasks"]["T1.3"] = {
                "file": str(gp),
                "bounds": [float(v) for v in b],
                "area_km2_sum": area_km2,
                "n": int(len(g)),
            }
            log.info("研究区面积(几何求和) = %.1f km2", area_km2)

    # ---- T1.4 基准网格核验 -----------------------------------------------
    with C.Timer(log, "T1.4 基准掩膜与网格核验"):
        grid = C.get_base_grid()
        with rasterio.open(C.MASK_TIF) as src:
            m = src.read(1)
            nvalid = int((m > 0).sum())
        area_km2 = nvalid * grid.pixel_area_m2 / 1e6
        report["tasks"]["T1.4"] = {
            "mask": str(C.MASK_TIF),
            "grid": grid.to_dict(),
            "n_valid_cells": nvalid,
            "area_km2_from_cells": area_km2,
            "expect_cells": 136205522,
            "match_expect": nvalid == 136205522,
        }
        log.info("网格 %dx%d  有效像元 %d  面积 %.1f km2",
                 grid.width, grid.height, nvalid, area_km2)

    # ---- T1.5 多尺度渔网核验 ---------------------------------------------
    with C.Timer(log, "T1.5 多尺度渔网核验"):
        fish = {}
        if FISHNET_DIR.exists():
            for km, exp in EXPECT_FISHNET.items():
                cand = list(FISHNET_DIR.glob(f"fishnet_{km}km*.shp"))
                # 优先用 _label 版（论文的有效采样点口径）
                labels = [p for p in cand if "_label" in p.name]
                p = labels[0] if labels else (cand[0] if cand else None)
                if p is None:
                    fish[km] = {"exists": False}
                    continue
                n = int(len(gpd.read_file(p)))
                fish[km] = {"path": str(p), "n_features": n,
                            "expect": exp, "match": n == exp}
                log.info("%2d km: %6d 个单元  (论文 %6d)  %s",
                         km, n, exp, "OK" if n == exp else "DIFF")
        report["tasks"]["T1.5"] = fish

    # ---- T1.6 风险评价单元核验 -------------------------------------------
    with C.Timer(log, "T1.6 9km 风险评价单元核验"):
        rep = {"path": str(LERI_UNIT_SHP), "exists": LERI_UNIT_SHP.exists()}
        if LERI_UNIT_SHP.exists():
            g = gpd.read_file(LERI_UNIT_SHP)
            rep.update({"n_features": int(len(g)),
                        "expect": EXPECT_LERI_UNITS,
                        "match": int(len(g)) == EXPECT_LERI_UNITS,
                        "cell_size_m": 9000.0})
            log.info("9km 评价单元 %d 个  (论文 %d)  %s", len(g), EXPECT_LERI_UNITS,
                     "OK" if len(g) == EXPECT_LERI_UNITS else "DIFF")
        report["tasks"]["T1.6"] = rep

    # ---- T1.7 报告 -------------------------------------------------------
    C.dump_json(report, out / "report.json")
    log.info("阶段1 完成，报告写入 %s", out / "report.json")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段1 研究区界定与数据准备")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    C.ensure_dirs()
    a = ap.parse_args(argv)
    run(Path(a.out_root))


if __name__ == "__main__":
    main()
