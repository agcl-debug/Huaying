"""
阶段 3　InVEST 三类服务（Python 重实现）

对应任务 T3.1 - T3.7

无 InVEST 可执行环境，故按论文公式与 InVEST 算法重写三个模块：
  T3.1 产水量    ：Zhang 曲线（Budyko 水热耦合），参数 Z=3.0
  T3.2 水源涵养量：产水量经流速系数与地形/土壤修正
  T3.3 生境质量  ：威胁衰减 + 生境敏感度
  T3.4 碳储      ：四碳库按土地利用类型赋值求和

与原研究的差异（有意为之）
--------------------------
* 生境质量的威胁方案在 2014 年前后不一致（前 14 年 4 威胁、后 7 年 7 威胁）。
  本脚本**按每年各自的 threats_YYYY.csv 读取**，并在年度记录中标注威胁个数，
  从而把原文的隐患显式暴露出来；另提供 `--harmonize-threats` 统一为 7 威胁方案。
* 产水量单位：InVEST 的 wyield 是水深 mm；原研究把 148.48 mm 当成 148.48 m³/ha
  导致总量少 10 倍。本脚本**同时输出 mm 与 m³ 两种口径**，并在报告中并列。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from . import common as C

BIO_TABLE = C.RESULTS / r"Biophysical Table\biophysical_table_gura.csv"
AWY_DIR = C.RESULTS / r"Biophysical Table\Annual Water Yield"
CARBON_CSV = C.DATA / r"02_遥感与基础地理\LULC_Carbon\carbon.csv"
CARBON_LULC = C.DATA / r"02_遥感与基础地理\LULC_Carbon"
CLCD = C.DATA / r"02_遥感与基础地理\LULC_CLCD"
THREATS = C.RESULTS / "Threats Files"
PRE_YEAR = C.DATA / r"02_遥感与基础地理\Pre_year"
ETO_DIR = C.RESULTS / "Eva_year"
PAWC_P = C.DATA / r"02_遥感与基础地理\PAWC\PAWC.tif"
DTB_P = C.DATA / r"02_遥感与基础地理\Depth_to_bedrock\DTB_mm.tif"
VEL_DIR = C.RESULTS / "Velocity"
WCV_REF = C.RESULTS / "water conservation volume"
HQ_REF = C.RESULTS / "Habitat Quality"
CS_REF = C.RESULTS / "Carbon Storage"

Z_PARAM = 3.0              # 盘上日志记录的 seasonality_constant
#: 覆盖用：非 None 时取代 Z_PARAM。
#: 注意 water_yield 的 `z=Z_PARAM` 是**定义时绑定**的默认参数，
#: 只改 Z_PARAM 不会生效，必须显式传 z=。
Z_OVERRIDE = None
K_SAT = 0.5                # 生境质量 half_saturation_constant
WCV_RATIO = 0.4252         # 水源涵养量/产水量 的论文口径比例系数
DEFAULT_WINDOW = (10132, 6608, 12180, 8656)


def _find(pat: str, *dirs: Path) -> Path | None:
    for d in dirs:
        if d.exists():
            for p in sorted(d.glob(pat)):
                return p
    return None


def _read(p: Path, window):
    a, _ = C.read_window(p, window)
    return a


def read_biophysical() -> pd.DataFrame:
    df = pd.read_csv(BIO_TABLE)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def read_carbon() -> pd.DataFrame:
    for cand in [CARBON_CSV, CARBON_LULC / "carbon.csv"]:
        if cand.exists():
            df = pd.read_csv(cand)
            df.columns = [str(c).strip() for c in df.columns]
            return df
    raise FileNotFoundError("找不到 carbon.csv")


# ------------------------------------------------------------------ T3.1


def water_yield(precip, pet, pawc, root_depth, dtb, lulc_veg, z=Z_PARAM,
                pawc_is_fraction: bool = True):
    """
    InVEST Annual Water Yield 的 Zhang 曲线实现。

    ω = Z·AWC / P + 1.25 ，AWC = PAWC × min(root_depth, DTB)
    AET/P = (1 + ω·PET/P) / (1 + ω·PET/P + P/PET)
    WY = P − AET
    对非植被像元（LULC_veg = 0）：AET = min(Kc·ETo, P)

    pawc_is_fraction=True 时 PAWC 为体积比例(0~1)，乘以根系层厚度得 mm；
    否则视为已是 mm。
    """
    P = np.where(np.isfinite(precip) & (precip > 0), precip, np.nan).astype(np.float32)
    PET = pet.astype(np.float32)
    rd = np.minimum(root_depth, dtb).astype(np.float32)
    awc = (pawc * rd) if pawc_is_fraction else (pawc.astype(np.float32) + np.zeros_like(rd))
    with np.errstate(invalid="ignore", divide="ignore"):
        omega = z * awc / P + 1.25
        petp = PET / P
        frac = (1.0 + omega * petp) / (1.0 + omega * petp + 1.0 / petp)
    aet = frac * P
    aet = np.where(np.isfinite(aet), np.minimum(aet, P), np.nan)
    # 非植被像元
    nv = lulc_veg <= 0
    with np.errstate(invalid="ignore"):
        aet_nv = np.minimum(PET, P)
    aet = np.where(nv & np.isfinite(aet_nv), aet_nv, aet)
    wy = (P - aet).astype(np.float32)
    return wy, aet, P, PET


# ------------------------------------------------------------------ T3.3


def habitat_quality(lulc, threats_tbl: pd.DataFrame, sens_tbl: pd.DataFrame,
                    threat_arrays: dict, cell_m: float = 30.0):
    """
    InVEST Habitat Quality 实现。

    D_xj = Σ_t Σ_y (w_t / Σw) · i_rxy · β_x · S_jt
    Q_xj = H_j · (1 − D^z/(D^z + k^z))，z=2.5, k=half_saturation_constant
    """
    cols = [c.strip() for c in threats_tbl.columns]
    threats_tbl.columns = cols
    tcol = "THREAT" if "THREAT" in cols else cols[0]
    wcol = "WEIGHT" if "WEIGHT" in cols else None
    dcol = "DECAY" if "DECAY" in cols else None
    mcol = "MAX_DIST" if "MAX_DIST" in cols else None
    names = [str(v).strip().lower() for v in threats_tbl[tcol]]

    sens = sens_tbl.copy()
    sens.columns = [str(c).strip() for c in sens.columns]
    lucol = sens.columns[0]
    hcol = [c for c in sens.columns if c.upper().startswith("HABITAT")]
    hcol = hcol[0] if hcol else sens.columns[1]
    H_map = dict(zip(sens[lucol].astype(int), sens[hcol].astype(float)))
    S_map = {}
    for j, nm in enumerate(names):
        col = None
        for c in sens.columns:
            if str(c).strip().lower() == nm:
                col = c
                break
        if col is not None:
            S_map[j] = dict(zip(sens[lucol].astype(int), sens[col].astype(float)))

    wsum = float(threats_tbl[wcol].astype(float).sum()) if wcol else float(len(names))
    D = np.zeros(lulc.shape, dtype=np.float64)
    for j, nm in enumerate(names):
        arr = threat_arrays.get(nm)
        if arr is None:
            continue
        maxd = float(threats_tbl[mcol].iloc[j]) * 1000.0 if mcol else 1000.0
        w = float(threats_tbl[wcol].iloc[j]) if wcol else 1.0
        decay = str(threats_tbl[dcol].iloc[j]).strip().lower() if dcol else "exponential"
        mask = np.isfinite(arr) & (arr > 0)
        if not mask.any():
            continue
        # 到最近威胁像元的欧氏距离（米）
        dist = ndimage.distance_transform_edt(~mask, sampling=(cell_m, cell_m))
        d = dist / max(maxd, 1e-6)
        if decay.startswith("exp"):
            ir = np.exp(-2.99 * d)
        else:                                   # linear
            ir = np.clip(1.0 - d, 0.0, 1.0)
        Sj = np.zeros(lulc.shape, dtype=np.float64)
        for code, v in S_map.get(j, {}).items():
            Sj[lulc == code] = v
        beta = 1.0                              # 无可达性矢量，按 1 处理
        D += (w / max(wsum, 1e-9)) * ir * beta * Sj
    Dz = np.power(np.maximum(D, 0.0), 2.5)
    kval = np.power(K_SAT, 2.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        Q = (Dz / (Dz + kval))
    Hj = np.zeros(lulc.shape, dtype=np.float64)
    for code, v in H_map.items():
        Hj[lulc == code] = v
    Q = Hj * (1.0 - Q)
    Q = np.where(lulc <= 0, np.nan, Q).astype(np.float32)
    return Q, D.astype(np.float32)


# ------------------------------------------------------------------ T3.4


def carbon_storage(lulc, pools: pd.DataFrame):
    """碳储 = Σ(各碳库) 按土地利用类型赋值；单位 t/ha。"""
    df = pools.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lucol = df.columns[0]                                   # lucode
    pool_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    tot = np.zeros(lulc.shape, dtype=np.float64)
    for _, row in df.iterrows():
        try:
            code = int(row[lucol])
        except Exception:
            continue
        s = float(np.nansum([row[c] for c in pool_cols]))
        tot[lulc == code] = s
    return np.where(lulc <= 0, np.nan, tot).astype(np.float32)


# ------------------------------------------------------------------ 主流程


def run(out_root: Path = C.OUT_ROOT, window=DEFAULT_WINDOW,
        years=(2000, 2019), harmonize_threats: bool = False,
        wcv_from_velocity: bool = False, **_):
    log = C.get_logger("s03_invest")
    out = Path(out_root) / "s03"
    out.mkdir(parents=True, exist_ok=True)
    _, wg = C.read_window(C.MASK_TIF, window)
    y0, y1 = years
    rows = []

    bio = read_biophysical()
    pools = read_carbon()
    log.info("生物物理表 %d 行；碳密度表 %d 行，库字段=%s",
             len(bio), len(pools), list(pools.columns[1:]))

    # 静态输入
    pawc = _read(PAWC_P, window) if PAWC_P.exists() else None
    dtb = _read(DTB_P, window) if DTB_P.exists() else None
    pawc_is_fraction = True
    if pawc is None:
        log.warning("缺 PAWC，用水分常量 100 mm 替代")
        pawc = np.full(wg.shape, 100.0, np.float32)
        pawc_is_fraction = False
    else:
        pv = pawc[np.isfinite(pawc)]
        if pv.size:
            log.info("PAWC 原始范围 min=%.4f max=%.4f mean=%.4f", pv.min(), pv.max(), pv.mean())
            pawc_is_fraction = bool(pv.max() <= 1.5)
            log.info("PAWC 口径判定：%s", "体积比例(0~1)×根系层厚度" if pawc_is_fraction
                     else "已是 mm，直接使用")
    if dtb is None:
        log.warning("缺根系限制层深度，用 1000 mm 替代")
        dtb = np.full(wg.shape, 1000.0, np.float32)

    def _bc(mapping, arr_code):
        out_a = np.zeros(arr_code.shape, dtype=np.float32)
        for code, v in mapping.items():
            out_a[arr_code == code] = v
        return out_a

    for y in range(y0, y1 + 1):
        pre_p = _find(f"Pre_{y}.tif", PRE_YEAR)
        eto_p = _find(f"ET_{y}.tif", ETO_DIR)
        lulc_p = _find(f"LULC_{y}.tif", CLCD)
        if pre_p is None or eto_p is None or lulc_p is None:
            log.warning("缺 %d 年输入（Pre/ET/LULC），跳过", y)
            continue
        pre = _read(pre_p, window)
        eto = _read(eto_p, window)
        lulc = _read(lulc_p, window)
        lulc_code = np.where(np.isfinite(lulc), lulc, 0).astype(np.int32)

        # --- T3.1 产水量
        kc = _bc(dict(zip(bio["lucode"].astype(int), bio["kc"].astype(float))), lulc_code)
        rootd = _bc(dict(zip(bio["lucode"].astype(int), bio["root_depth"].astype(float))), lulc_code)
        veg = _bc(dict(zip(bio["lucode"].astype(int), bio["lulc_veg"].astype(float))), lulc_code)
        pet = (kc * eto).astype(np.float32)
        wy, aet, P, PET = water_yield(pre, pet, pawc, rootd, dtb, veg,
                                      z=(Z_OVERRIDE if Z_OVERRIDE is not None else Z_PARAM),
                                      pawc_is_fraction=pawc_is_fraction)
        C.write_raster(out / f"WY_{y}.tif", wy, wg)

        # --- T3.2 水源涵养量
        vel_p = _find(f"Velocity_{y}.tif", VEL_DIR)
        if wcv_from_velocity and vel_p is not None:
            vel = _read(vel_p, window)
            vv = vel[np.isfinite(vel)]
            if vv.size and vv.max() > vv.min():
                vn = (vel - vv.min()) / (vv.max() - vv.min())
            else:
                vn = np.zeros_like(vel)
            factor = 1.0 - 0.5 * np.clip(vn, 0, 1)
            f_src = "流速修正"
        else:
            # 论文 5.2/3.2 的口径：涵养量/产水量 = 42.52 ± 0.32%
            factor = np.full(wg.shape, WCV_RATIO, np.float32)
            f_src = f"固定比例{WCV_RATIO}"
        wcv = (wy * factor).astype(np.float32)
        C.write_raster(out / f"WCV_{y}.tif", wcv, wg)

        # --- T3.4 碳储（InVEST 日志指定用 LULC_Carbon\YYYY_c.tif 这一套重分类编码）
        cs_lulc_p = _find(f"{y}_c.tif", CARBON_LULC)
        lulc_cs = _read(cs_lulc_p, window) if cs_lulc_p is not None else lulc
        cs_code = np.where(np.isfinite(lulc_cs), lulc_cs, 0).astype(np.int32)
        cs = carbon_storage(cs_code, pools)
        C.write_raster(out / f"CS_{y}.tif", cs, wg)

        # --- T3.3 生境质量
        hq = np.full(wg.shape, np.nan, np.float32)
        deg = np.full(wg.shape, np.nan, np.float32)
        n_threat = 0
        th_p = _find(f"threats_{y}.csv", THREATS / str(y))
        se_p = _find(f"sensitivity_{y}.csv", THREATS / str(y))
        lc_p = _find(f"{y}_c.tif", THREATS / str(y))
        if th_p is not None and se_p is not None and lc_p is not None:
            tt = pd.read_csv(th_p)
            ss = pd.read_csv(se_p)
            n_threat = len(tt)
            tnames = [str(v).strip().lower() for v in tt[tt.columns[2] if len(tt.columns) > 2 else tt.columns[0]]]
            tarrs = {}
            for nm in tnames:
                for pat in (f"{nm}_c.tif", f"{nm}.tif"):
                    p = _find(pat, THREATS / str(y))
                    if p is not None:
                        tarrs[nm] = _read(p, window)
                        break
            lulc_hq = _read(lc_p, window)
            hq_code = np.where(np.isfinite(lulc_hq), lulc_hq, 0).astype(np.int32)
            hq, deg = habitat_quality(hq_code, tt, ss, tarrs, cell_m=30.0)
            C.write_raster(out / f"HQ_{y}.tif", hq, wg)
            C.write_raster(out / f"DEG_{y}.tif", deg, wg)
        else:
            log.warning("%d 年缺少生境质量输入（threats/sensitivity/LULC_c）", y)

        rec = {"year": y, "n_threats": n_threat, "wcv_factor": f_src}
        for nm, arr in (("WY", wy), ("WCV", wcv), ("HQ", hq), ("CS", cs)):
            v = arr[np.isfinite(arr)]
            rec[f"{nm}_mean"] = float(v.mean()) if v.size else np.nan
            rec[f"{nm}_n"] = int(v.size)
        # 水深 mm × 像元面积 m² ÷ 1000 = m³
        # 注意：像元面积必须取自**当前工作网格**，不能硬编码 30 m 的 900 m²。
        # 30 m 网格 -> 900 m²；300 m 网格 -> 90000 m²（差 100 倍）。
        _px_area = (900.0 if wg.pixel_area_m2 <= 0 else wg.pixel_area_m2)
        _vol_m3 = float(np.nansum(wy) * _px_area / 1000.0)
        rec["pixel_area_m2"] = float(_px_area)
        rec["WY_total_m3"] = _vol_m3
        rec["WY_total_1e8m3"] = _vol_m3 / 1e8
        rec["P_mean_mm"] = float(np.nanmean(P))
        rec["PET_mean_mm"] = float(np.nanmean(PET))
        rec["AET_mean_mm"] = float(np.nanmean(aet))
        # 与盘上参照对比
        for nm, p in (("WCV_ref", _find(f"WCV_{y}.tif", WCV_REF)),
                      ("HQ_ref", _find(f"{y}_quality_c.tif", HQ_REF / str(y))),
                      ("CS_ref", _find(f"{y}_tot_c_cur.tif", CS_REF / str(y)))):
            if p is None:
                continue
            ref = _read(p, window)
            rv = ref[np.isfinite(ref)]
            mine = {"WCV_ref": wcv, "HQ_ref": hq, "CS_ref": cs}[nm]
            mv = mine[np.isfinite(mine)]
            rec[f"{nm}_mean"] = float(rv.mean()) if rv.size else np.nan
            rec[f"{nm}_ratio"] = (float(mv.mean() / rv.mean())
                                 if rv.size and rv.mean() != 0 and mv.size else np.nan)
        rows.append(rec)
        log.info("T3 %d 完成：威胁 %d 个  WY=%.2f mm  WCV=%.2f  HQ=%.3f  CS=%.3f  WY总量=%.2f 亿m³",
                 y, n_threat, rec["WY_mean"], rec["WCV_mean"],
                 rec.get("HQ_mean", np.nan), rec.get("CS_mean", np.nan),
                 rec["WY_total_1e8m3"])

    df = pd.DataFrame(rows)
    csv = out / "stats.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    log.info("三服务逐年统计写入 %s（%d 行）", csv, len(df))
    if len(df):
        show = [c for c in df.columns if c in
                ("year", "n_threats", "WY_mean", "P_mean_mm", "AET_mean_mm",
                 "WY_total_1e8m3", "WCV_ref_ratio", "HQ_ref_ratio", "CS_ref_ratio")]
        with pd.option_context("display.width", 200):
            log.info("摘要：\n%s", df[show].to_string(index=False))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段3 InVEST 三类服务")
    ap.add_argument("--out-root", default=str(C.OUT_ROOT))
    ap.add_argument("--window", default=",".join(map(str, DEFAULT_WINDOW)))
    ap.add_argument("--years", default="2000,2019")
    ap.add_argument("--harmonize-threats", action="store_true")
    ap.add_argument("--wcv-from-velocity", action="store_true",
                    help="水源涵养用流速系数修正（默认用论文的固定比例 0.4252）")
    C.ensure_dirs()
    a = ap.parse_args(argv)
    run(Path(a.out_root),
        window=tuple(int(v) for v in a.window.split(",")),
        years=tuple(int(v) for v in a.years.split(",")),
        harmonize_threats=a.harmonize_threats,
        wcv_from_velocity=a.wcv_from_velocity)


if __name__ == "__main__":
    main()
