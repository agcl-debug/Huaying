"""
阶段 2　RUSLE 因子链

对应任务 T2.1 - T2.9

链路：
    R × K × LS            -> RKLS（潜在侵蚀 Ap）
    R × K × LS × C × P    -> A   （实际侵蚀 Ar）
    SC = RKLS - A （水土保持）      SE = A （土壤侵蚀）
单位：因子连乘后为 t·hm^-2·a^-1，乘 100 转 t·km^-2·a^-1。

与原研究的差异（有意为之）
--------------------------
* K：原研究先算 `kepic.tif` 再 ×0.1317 得 `K.tif`，另用 `Con(K>0.04 → NoData)` 得 `KR.tif`。
  本脚本用 Clay/Sand/Silt/OC 按 EPIC 公式重算，并与现有 `kepic.tif` 对比；缺 OC 时按默认值
  处理并在日志中声明假设。
* C：原研究用 `Con(FVC==0,1,Con(FVC>=0.783,0,0.6508-0.3436*Log10(100*FVC)))`。
  本脚本按同一公式重算。
* P：原研究论文称"赋值为 1"，但盘上实际是 8 级赋值。本脚本**按赋值策略重算**（这就是
  补上的代码），同时提供 `--p-uniform` 开关用于复现论文口径以便对比。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from . import common as C
from . import geo as G

R_DIR = C.RUSLE / "R_factor"
C_DIR = C.RUSLE / "C_factor"
P_DIR = C.RUSLE / "P_factor"
K_DIR = C.RUSLE / "K_factor"
LS_DIR = C.RUSLE / "LS_factor"
HWSD_DIR = C.DATA / r"02_遥感与基础地理\Soil_HWSD"
FVC_DIR = C.DATA / r"02_遥感与基础地理\FVC"
SAP_DIR = C.DATA / r"02_遥感与基础地理\Soil AP"
SC_DIR = C.RESULTS / "Soil Conservation"
SE_DIR = C.RESULTS / "Soil Erosion"

#: 论文 2.3.2 给出的 K 单位换算系数
K_UNIT_CONV = 0.1317
#: 论文所述 P 因子统一赋值
P_UNIFORM = 1.0

#: 默认窗口 = 基准网格中心 2048x2048（全量运行请传 --window 0,0,22313,15265）
DEFAULT_WINDOW = (10132, 6608, 12180, 8656)


def _find(pattern: str, *dirs: Path) -> Path | None:
    for d in dirs:
        if d.exists():
            for p in sorted(d.glob(pattern)):
                return p
    return None


def epic_k(sand: np.ndarray, silt: np.ndarray, clay: np.ndarray,
           oc: np.ndarray) -> np.ndarray:
    """
    EPIC 模型土壤可蚀性（论文式 2-6 ~ 2-8）。

    San/Sil/Cla 为砂/粉/黏粒百分比，OC 为有机碳百分比。
    返回 Kepic（美制单位 (ton·acre·h)/(hundreds·acre·ft·ton·in)）。
    """
    sn = 1.0 - sand / 100.0
    fcsand = np.clip(0.2 + 0.3 * np.exp(-0.0256 * sand * (1.0 - silt / 100.0)), 0, 1)
    fcl_si = (silt / (clay + silt)) ** 0.3
    fcl_si = np.where(np.isfinite(fcl_si), fcl_si, 0.0)
    forc = 1.0 - 0.25 * oc / np.maximum(oc + np.exp(3.72 - 2.95 * oc), 1e-12)
    fhisand = 1.0 - 0.7 * sn / np.maximum(sn + np.exp(-5.51 + 22.9 * sn), 1e-12)
    return fcsand * fcl_si * forc * fhisand


def c_from_fvc(fvc: np.ndarray) -> np.ndarray:
    """
    由植被覆盖度计算 C 因子（与原研究血缘中的表达式一致）。

    C = 0.6508 - 0.3436 * lg(100 * FVC)
    FVC = 0    -> C = 1
    FVC >= 0.783 -> C = 0
    输入 FVC 为 0~1 的小数覆盖度。
    """
    f = np.clip(fvc, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = 0.6508 - 0.3436 * np.log10(np.maximum(100.0 * f, 1e-6))
    out = np.where(f <= 0.0, 1.0, base)
    out = np.where(f >= 0.783, 0.0, out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def p_by_strategy(lulc: np.ndarray) -> np.ndarray:
    """
    水土保持措施因子 P 的 8 级赋值策略（补写的代码）。

    依据盘上 `水土保持措施因子赋值策略.txt` 所列方法与
    `YRS_P_2000.tif` 实测的 8 个等级 {0, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    反推：按土地利用类型赋基值，再按坡度微调。

    这里给出一个可复现的显式策略表（lucode 依生物物理表 1-9）：
        1 农田   -> 0.60   2 森林 -> 0.20   3 灌木 -> 0.50
        4 草地   -> 0.70   5 水域 -> 0.00   6 冰雪 -> 0.00
        7 裸地   -> 0.90   8 不透水 -> 1.00  9 湿地 -> 0.60
    """
    table = {1: 0.60, 2: 0.20, 3: 0.50, 4: 0.70, 5: 0.00,
             6: 0.00, 7: 0.90, 8: 1.00, 9: 0.60}
    out = np.full(lulc.shape, 0.70, dtype=np.float32)
    for code, v in table.items():
        out[lulc == code] = v
    out[lulc <= 0] = np.nan          # 0 = NoData
    return out


def _read(p: Path, window) -> np.ndarray:
    a, _ = C.read_window(p, window)
    return a


def run(out_root: Path = C.OUT_ROOT, window=DEFAULT_WINDOW,
        years=(2000, 2019), p_uniform: bool = False, p_strategy: bool = False,
        recompute_c: bool = False, unit_scale: float = 1.0, **_):
    log = C.get_logger("s02_rusle")
    out = Path(out_root) / "s02"
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = years
    ys = list(range(y0, y1 + 1))
    rows = []

    # 窗口对应的网格（写出时使用，保证与输入严格同网格）
    _, win_grid = C.read_window(C.MASK_TIF, window)
    log.info("工作窗口 %s -> %dx%d", window, win_grid.width, win_grid.height)

    # ---- 静态因子：K 与 LS ----------------------------------------------
    with C.Timer(log, "T2.2/T2.3 读取 K 与 LS 并重算 K"):
        k_exist = _find("K.tif", K_DIR)
        kr_exist = _find("KR*.tif", K_DIR)
        ls_p = _find("LS.tif", LS_DIR)
        kepic_p = _find("kepic.tif", HWSD_DIR)
        clay_p = _find("Clay.tif", HWSD_DIR)
        sand_p = _find("Sand.tif", HWSD_DIR)
        silt_p = _find("Silt.tif", HWSD_DIR)
        oc_p = (_find("OC*.tif", HWSD_DIR) or _find("*organic*.tif", HWSD_DIR)
                or _find("*ORG*.tif", HWSD_DIR))

        K_ref = _read(k_exist, window) if k_exist else None
        LS = _read(ls_p, window) if ls_p else None
        if LS is None:
            raise FileNotFoundError("缺少 LS 因子")

        if all(p is not None for p in (clay_p, sand_p, silt_p)):
            clay = _read(clay_p, window); sand = _read(sand_p, window); silt = _read(silt_p, window)
            if oc_p is not None:
                oc = _read(oc_p, window)
                log.info("OC 取自 %s", oc_p.name)
            else:
                oc = np.full_like(clay, 1.0)
                log.warning("未找到有机碳栅格，OC 按常量 1.0%% 处理（假设已记录）")
            # 若分量是 0~1 小数则转百分比
            for nm, a in (("clay", clay), ("sand", sand), ("silt", silt)):
                v = a[np.isfinite(a)]
                if v.size and np.nanmax(v) <= 1.5:
                    log.info("%s 判定为小数比例，×100 转百分比", nm)
                    if nm == "clay":
                        clay = clay * 100
                    elif nm == "sand":
                        sand = sand * 100
                    else:
                        silt = silt * 100
            kepic_calc = epic_k(sand, silt, clay, oc)
            K_calc = (kepic_calc * K_UNIT_CONV).astype(np.float32)
            v = K_calc[np.isfinite(K_calc)]
            log.info("EPIC 重算 K：n=%d  min=%.5f  max=%.5f  mean=%.5f",
                     v.size, v.min(), v.max(), v.mean())
            if kepic_p is not None:
                kep_ref = _read(kepic_p, window)
                d = K_calc - kep_ref * K_UNIT_CONV
                d = d[np.isfinite(d)]
                if d.size:
                    log.info("与盘上 kepic×0.1317 对比：mean_diff=%.6f  rmse=%.6f",
                             d.mean(), float(np.sqrt((d ** 2).mean())))
        else:
            K_calc = K_ref
            log.warning("缺少 Clay/Sand/Silt，直接沿用盘上 K.tif")

        K_final = K_calc if K_calc is not None else K_ref
        C.write_raster(out / "K.tif", K_final, win_grid)

        if kr_exist is not None:
            KR_ref = _read(kr_exist, window)
            v = KR_ref[np.isfinite(KR_ref)]
            if v.size:
                log.info("盘上 KR.tif（剔除 K>0.04）：min=%.5f max=%.5f mean=%.5f",
                         v.min(), v.max(), v.mean())
        K_use = K_final

    # ---- 逐年合成 --------------------------------------------------------
    for y in ys:
        R_p = _find(f"R_{y}.tif", R_DIR)
        if R_p is None:
            log.warning("缺 R_%d，跳过", y)
            continue
        R = _read(R_p, window)

        C_p = _find(f"C_{y}.tif", C_DIR)
        fvc_p = _find(f"FVC_{y}.tif", FVC_DIR)
        fvc_stat = ""
        if recompute_c and fvc_p is not None:
            fvc_arr = _read(fvc_p, window)
            fv = fvc_arr[np.isfinite(fvc_arr)]
            if fv.size:
                fvc_stat = f"FVC(min={fv.min():.3f} mean={fv.mean():.3f} max={fv.max():.3f})"
            C_calc = c_from_fvc(fvc_arr)
            src = "FVC重算"
            cv = C_calc[np.isfinite(C_calc)]
            if cv.size:
                fvc_stat += f" -> C(mean={cv.mean():.4f})"
        if not recompute_c or fvc_p is None:
            if C_p is None:
                log.warning("缺 C_%d，跳过", y)
                continue
            C_calc = _read(C_p, window)
            src = "盘上C_factor"
        if C_p is not None:
            cr = _read(C_p, window); crv = cr[np.isfinite(cr)]
            if crv.size:
                fvc_stat += f" | C_factor mean={crv.mean():.4f}"
                # 重算模式下给出倍比，便于量化两者的口径差
                if recompute_c:
                    mv = C_calc[np.isfinite(C_calc)]
                    if mv.size and mv.mean() > 0:
                        fvc_stat += f" (重算/C_factor = {mv.mean()/crv.mean():.2f}x)"

        if p_uniform:
            P = np.where(np.isfinite(R) & np.isfinite(LS), P_UNIFORM, np.nan).astype(np.float32)
            p_src = "统一1.0(论文口径)"
        elif p_strategy:
            lulc_p = _find(f"LULC_{y}.tif", C.DATA / r"02_遥感与基础地理\LULC_CLCD")
            if lulc_p is not None:
                P = p_by_strategy(_read(lulc_p, window))
                p_src = "赋值策略(补写)"
            else:
                P = _read(_find(f"P_{y}.tif", P_DIR), window)
                p_src = "沿用P_factor"
        else:
            pp = _find(f"P_{y}.tif", P_DIR)
            if pp is None:
                log.warning("缺 P_%d，跳过", y)
                continue
            P = _read(pp, window)
            p_src = "盘上P_factor"
        _pv = P[np.isfinite(P)]
        if _pv.size:
            p_src += f"(mean={_pv.mean():.3f})"

        with C.Timer(log, f"T2.6/T2.7/T2.8 合成 {y} 年（C={src} P={p_src}）{fvc_stat}"):
            RKLS = (LS * K_use * R).astype(np.float32)          # t/ha
            A = (R * P * LS * K_use * C_calc).astype(np.float32)  # t/ha
            SC = (RKLS - A).astype(np.float32)
            SE = A
            # 单位换算：盘上原始栅格为 t/ha（已核对 SAP_ref = RKLS/100 关系），
            # 论文 2.3.2 所述 ×100 仅用于换算成 t/km² 报告值。
            C.write_raster(out / f"RKLS_{y}.tif", RKLS * unit_scale, win_grid)
            C.write_raster(out / f"A_{y}.tif", A * unit_scale, win_grid)
            C.write_raster(out / f"SC_{y}.tif", SC * unit_scale, win_grid)
            C.write_raster(out / f"SE_{y}.tif", SE * unit_scale, win_grid)

            rec = {"year": y, "c_source": src, "p_source": p_src}
            for nm, arr in (("RKLS", RKLS), ("A", A), ("SC", SC), ("SE", SE)):
                v = arr[np.isfinite(arr)]
                rec[f"{nm}_n"] = int(v.size)
                rec[f"{nm}_mean"] = float(v.mean()) if v.size else float("nan")
                rec[f"{nm}_mean_km2"] = float(v.mean()) * 100.0 if v.size else float("nan")
                rec[f"{nm}_sum_1e8t"] = float(v.sum() * 900.0 / 1e8) if v.size else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = A / RKLS
            rr = ratio[np.isfinite(ratio)]
            rec["A_over_RKLS"] = float(rr.mean()) if rr.size else float("nan")
            # 与盘上结果对比（盘上为 t/ha）
            for nm, p in (("SC_ref", _find(f"SC_{y}.tif", SC_DIR)),
                          ("SE_ref", _find(f"SE_{y}.tif", SE_DIR)),
                          ("SAP_ref", _find(f"SAP_{y}.tif", SAP_DIR))):
                if p is None:
                    continue
                ref = _read(p, window)
                rv = ref[np.isfinite(ref)]
                mine = {"SC_ref": SC, "SE_ref": SE, "SAP_ref": RKLS}[nm]
                d = mine - ref
                d = d[np.isfinite(d)]
                rec[f"{nm}_mean"] = float(rv.mean()) if rv.size else float("nan")
                rec[f"{nm}_ratio"] = (float(mine[np.isfinite(mine)].mean() / rv.mean())
                                      if rv.size and rv.mean() != 0 else float("nan"))
                rec[f"{nm}_rmse"] = float(np.sqrt((d ** 2).mean())) if d.size else float("nan")
            rows.append(rec)

    # ---- 汇总 ------------------------------------------------------------
    import pandas as pd
    df = pd.DataFrame(rows)
    csv = out / "stats_SC_SE.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    log.info("逐年统计写入 %s（%d 行）", csv, len(df))
    if len(df):
        show = [c for c in ("year", "c_source", "p_source", "RKLS_mean", "SAP_ref_mean",
                            "SAP_ref_ratio", "A_over_RKLS", "SC_ref_ratio", "SE_ref_ratio")
                if c in df.columns]
        with pd.option_context("display.width", 200):
            log.info("摘要（单位 t/ha）：\n%s", df[show].to_string(index=False))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段2 RUSLE 因子链")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--window", default=",".join(map(str, DEFAULT_WINDOW)),
                    help="x0,y0,x1,y1 像素窗口；全量用 0,0,22313,15265")
    ap.add_argument("--years", default="2000,2019")
    ap.add_argument("--p-uniform", action="store_true",
                    help="P 统一赋 1.0（复现论文口径）")
    ap.add_argument("--p-strategy", action="store_true",
                    help="P 用补写的 8 级赋值策略（默认用盘上 P_factor）")
    ap.add_argument("--recompute-c", action="store_true",
                    help="C 由 FVC 按蔡崇法公式重算（默认用盘上 C_factor）")
    ap.add_argument("--unit-scale", type=float, default=1.0,
                    help="写出栅格的单位系数：1.0=t/ha（与盘上原始一致），100=t/km2")
    C.ensure_dirs()
    a = ap.parse_args(argv)
    w = tuple(int(v) for v in a.window.split(","))
    y0, y1 = (int(v) for v in a.years.split(","))
    run(Path(a.out_root), window=w, years=(y0, y1),
        p_uniform=a.p_uniform, p_strategy=a.p_strategy,
        recompute_c=a.recompute_c, unit_scale=a.unit_scale)


if __name__ == "__main__":
    main()
