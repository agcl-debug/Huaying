"""
矢量与空间叠加工具：投影、渔网、分区统计、点采样。

坐标系统一取 mask.tif 的 CRS（Krasovsky_1940_Albers，中央经线 105°E，
标准纬线 25°N/47°N），以保证与全部栅格严格一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import box

from . import common as C

#: Krasovsky_1940_Albers 的 proj4 表达（备用；优先用 mask.tif 的 CRS）
KRASOVSKY_ALBERS_PROJ4 = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs"
)


def albers_crs():
    """取权威 CRS —— 直接读 mask.tif，保证与所有栅格一致。"""
    with rasterio.open(C.MASK_TIF) as d:
        return d.crs


def to_albers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """转到基准投影。"""
    if gdf.crs is None:
        raise ValueError("矢量缺少 CRS，无法转换")
    return gdf.to_crs(albers_crs())


def read_vector(path: Path | str, to_albers: bool = True) -> gpd.GeoDataFrame:
    """读矢量，默认转到基准投影。"""
    gdf = gpd.read_file(path)
    if to_albers and gdf.crs is not None:
        gdf = gdf.to_crs(albers_crs())
    return gdf


def write_vector(gdf: gpd.GeoDataFrame, path: Path | str) -> Path:
    """写出矢量（按扩展名选格式；.gpkg 推荐）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG" if path.suffix.lower() == ".gpkg" else None)
    return path


# ---------------------------------------------------------------- 渔网


def fishnet(bounds: Sequence[float], cell: float, crs=None) -> gpd.GeoDataFrame:
    """
    生成规则渔网（矢量单元，含 cell_id 与质心坐标）。

    bounds = (minx, miny, maxx, maxy)
    """
    crs = crs or albers_crs()
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, cell)
    ys = np.arange(miny, maxy, cell)
    geoms, ids, cx, cy = [], [], [], []
    k = 0
    for y in ys:
        for x in xs:
            geoms.append(box(x, y, x + cell, y + cell))
            ids.append(k); cx.append(x + cell / 2); cy.append(y + cell / 2)
            k += 1
    return gpd.GeoDataFrame({"cell_id": ids, "cx": cx, "cy": cy},
                            geometry=geoms, crs=crs)


def fishnet_from_raster(cell: float, mask: bool = True,
                        crs=None) -> gpd.GeoDataFrame:
    """
    按基准栅格范围生成渔网；mask=True 时只保留与研究区相交的单元。
    """
    g = C.get_base_grid()
    with rasterio.open(C.MASK_TIF) as d:
        b = d.bounds
    fn = fishnet((b.left, b.bottom, b.right, b.top), cell, crs=crs or albers_crs())
    if mask:
        m = read_vector_bbox(C.MASK_TIF)
        fn = gpd.sjoin(fn, m, how="inner", predicate="intersects").drop(columns="index_right")
    return fn.reset_index(drop=True)


def read_vector_bbox(mask_tif: Path | str) -> gpd.GeoDataFrame:
    """把研究区掩膜的有效范围转成面（用于快速裁剪渔网）。"""
    with rasterio.open(mask_tif) as d:
        b = d.bounds
    return gpd.GeoDataFrame(geometry=[box(b.left, b.bottom, b.right, b.top)],
                            crs=d.crs)


# ---------------------------------------------------------------- 分区统计


def zonal_stats(raster_path: Path | str, zones: gpd.GeoDataFrame,
                stats: Sequence[str] = ("mean", "sum", "count"),
                all_touched: bool = False, band: int = 1) -> "pd.DataFrame":
    """
    分区统计：把 zones 栅格化为标签图，再用 np.bincount 聚合。

    比逐要素裁剪快几个数量级，适合 1,682 个单元 × 21 年这类规模。
    """
    import pandas as pd

    with rasterio.open(raster_path) as src:
        arr = src.read(band).astype(np.float64)
        nd = src.nodata
        tr = src.transform
        shape = arr.shape
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    arr = np.where(np.abs(arr) >= np.finfo(np.float32).max * 0.5, np.nan, arr)

    labels = features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(zones.geometry)),
        out_shape=shape, transform=tr, fill=0, all_touched=all_touched,
        dtype="int32")

    valid = np.isfinite(arr) & (labels > 0)
    lab = labels[valid]
    val = arr[valid]
    n = len(zones)
    cnt = np.bincount(lab, minlength=n + 1)[1:n + 1].astype(np.float64)
    ssum = np.bincount(lab, weights=val, minlength=n + 1)[1:n + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / np.where(cnt > 0, cnt, np.nan)

    out = {}
    if "count" in stats:
        out["count"] = cnt
    if "sum" in stats:
        out["sum"] = ssum
    if "mean" in stats:
        out["mean"] = mean
    if "max" in stats:
        mx = np.full(n + 1, -np.inf)
        np.maximum.at(mx, lab, val)
        out["max"] = np.where(np.isfinite(mx[1:n + 1]), mx[1:n + 1], np.nan)
    if "min" in stats:
        mn = np.full(n + 1, np.inf)
        np.minimum.at(mn, lab, val)
        out["min"] = np.where(np.isfinite(mn[1:n + 1]), mn[1:n + 1], np.nan)

    res = pd.DataFrame(out)
    for c in zones.columns:
        if c != zones.geometry.name:
            res[c] = zones[c].values
    return res


# ---------------------------------------------------------------- 点采样


def sample_at_points(raster_paths: Sequence[Path | str],
                     xs: np.ndarray, ys: np.ndarray,
                     band: int = 1) -> np.ndarray:
    """
    在给定坐标点抽取多幅栅格的值。

    返回 (N, F) 数组，F = len(raster_paths)。
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    out = np.full((len(xs), len(raster_paths)), np.nan, dtype=np.float32)
    for f, p in enumerate(raster_paths):
        with rasterio.open(p) as src:
            nd = src.nodata
            rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
            rows = np.asarray(rows); cols = np.asarray(cols)
            ok = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
            if not ok.any():
                continue
            r0, r1 = int(rows[ok].min()), int(rows[ok].max()) + 1
            c0, c1 = int(cols[ok].min()), int(cols[ok].max()) + 1
            win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            blk = src.read(band, window=win).astype(np.float32)
            if nd is not None:
                blk = np.where(blk == np.float32(nd), np.nan, blk)
            blk = np.where(np.abs(blk) >= np.finfo(np.float32).max * 0.5, np.nan, blk)
            vals = blk[rows[ok] - r0, cols[ok] - c0]
            tmp = out[ok, f]
            tmp[:] = vals
            out[ok, f] = tmp
    return out


def points_from_polygons(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """取面要素的质心坐标，返回 (N, 2)。"""
    cen = gdf.geometry.centroid
    return np.column_stack([cen.x.values, cen.y.values])


# ---------------------------------------------------------------- 叠加


def overlay_intersection(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """矢量求交（geopandas overlay 封装）。"""
    return gpd.overlay(a, b, how="intersection")


def rasterize_zones(zones: gpd.GeoDataFrame, grid: C.Grid | None = None,
                    all_touched: bool = False) -> np.ndarray:
    """把矢量栅格化到基准网格（值 = 从 1 开始的序号）。"""
    grid = grid or C.get_base_grid()
    with rasterio.open(C.MASK_TIF) as d:
        tr = d.transform
    return features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(zones.geometry)),
        out_shape=(grid.height, grid.width), transform=tr, fill=0,
        all_touched=all_touched, dtype="int32")
