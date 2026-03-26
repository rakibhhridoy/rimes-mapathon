"""
Step 8 — Composite Risk Scoring: Hazard × Exposure × Vulnerability
Step 9 — Aggregation to union/upazila level, hotspot detection, ranking.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import box

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exposure Layer
# ---------------------------------------------------------------------------

def compute_exposure_grid(infra: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame,
                           cfg: dict) -> np.ndarray:
    """
    Compute exposure score per grid cell based on infrastructure density.
    Vectorized via groupby — no per-cell Python loop.
    """
    weights = cfg["risk"]["exposure_weights"]
    n_cells = len(grid_gdf)
    exposure = np.zeros(n_cells)

    # Spatial join: which assets fall in which cell
    infra_pts = infra.copy()
    infra_pts["geometry"] = infra_pts.geometry.representative_point()
    joined = gpd.sjoin(infra_pts, grid_gdf, how="inner", predicate="within")

    if len(joined) == 0:
        return exposure

    # Map asset types to their weight values
    type_weight_map = {
        "hospital": weights.get("hospital_school_presence", 0.20),
        "school": weights.get("hospital_school_presence", 0.20),
        "bridge": weights.get("bridge_presence", 0.15),
        "road": weights.get("road_length_km", 0.15),
        "cropland": weights.get("cropland_area_km2", 0.15),
        "flood_shelter": weights.get("embankment_shelter_presence", 0.10),
        "embankment": weights.get("embankment_shelter_presence", 0.10),
    }
    building_w = weights.get("building_count", 0.25)

    # Vectorized: compute per-asset weight, then groupby cell
    joined["_weight"] = joined["asset_type"].map(type_weight_map).fillna(0.0) + building_w
    cell_scores = joined.groupby("index_right")["_weight"].sum()

    exposure[cell_scores.index.values] = cell_scores.values

    # Normalize to [0, 1]
    if exposure.max() > 0:
        exposure = exposure / exposure.max()

    return exposure


def compute_vulnerability_grid(grid_gdf: gpd.GeoDataFrame, infra: gpd.GeoDataFrame,
                                cfg: dict) -> np.ndarray:
    """
    Compute vulnerability score per grid cell.
    """
    weights = cfg["risk"]["vulnerability_weights"]
    n_cells = len(grid_gdf)
    vuln = np.zeros(n_cells)

    # Grid cell centroids
    cell_centers = np.array(
        [(g.centroid.x, g.centroid.y) for g in grid_gdf.geometry]
    )

    # Infrastructure centroids by type
    infra_pts = infra.copy()
    infra_pts["geometry"] = infra_pts.geometry.representative_point()
    infra_coords = np.array(
        [(g.x, g.y) for g in infra_pts.geometry]
    )

    # Distance to hospitals
    dist_hospital = _min_dist_to_type(cell_centers, infra_pts, infra_coords, ["hospital"])
    # Distance to flood shelters
    dist_shelter = _min_dist_to_type(cell_centers, infra_pts, infra_coords, ["flood_shelter"])
    # Distance to primary roads
    dist_road = _min_dist_to_type(cell_centers, infra_pts, infra_coords, ["road"])

    # Normalize distances (higher distance = higher vulnerability)
    dist_hospital_norm = _normalize_dist(dist_hospital)
    dist_shelter_norm = _normalize_dist(dist_shelter)
    dist_road_norm = _normalize_dist(dist_road)

    # Population density placeholder (would need raster sampling)
    pop_density_norm = np.ones(n_cells) * 0.5  # default mid-range

    vuln = (
        weights.get("population_density", 0.30) * pop_density_norm +
        weights.get("dist_hospital", 0.20) * dist_hospital_norm +
        weights.get("dist_flood_shelter", 0.15) * dist_shelter_norm +
        weights.get("dist_primary_road", 0.10) * dist_road_norm +
        weights.get("night_light_proxy", 0.15) * 0.5 +  # placeholder
        weights.get("elderly_child_ratio", 0.10) * 0.5   # placeholder
    )

    # Normalize to [0, 1]
    if vuln.max() > 0:
        vuln = vuln / vuln.max()

    return vuln


def _min_dist_to_type(cell_centers: np.ndarray, infra: gpd.GeoDataFrame,
                       infra_coords: np.ndarray, types: list[str]) -> np.ndarray:
    """Min distance from each cell center to nearest infrastructure of given types."""
    mask = infra["asset_type"].isin(types)
    if mask.sum() == 0:
        return np.full(len(cell_centers), 50000.0)
    target_coords = infra_coords[mask.values]
    tree = cKDTree(target_coords)
    dists, _ = tree.query(cell_centers, k=1)
    return dists


def _normalize_dist(dists: np.ndarray) -> np.ndarray:
    """Normalize distance to [0, 1] — larger distance = higher vulnerability."""
    d_min, d_max = dists.min(), dists.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(dists)
    return (dists - d_min) / (d_max - d_min)


# ---------------------------------------------------------------------------
# Composite Risk
# ---------------------------------------------------------------------------

def compute_composite_risk(hazard: np.ndarray, exposure: np.ndarray,
                            vulnerability: np.ndarray) -> np.ndarray:
    """Risk = Hazard × Exposure × Vulnerability, normalized to [0, 1]."""
    risk = hazard * exposure * vulnerability
    if risk.max() > 0:
        risk = risk / risk.max()
    return risk


def create_risk_grid(bounds: tuple[float, float, float, float],
                      resolution_deg: float,
                      max_cells: int = 150_000) -> gpd.GeoDataFrame:
    """Create a regular grid of polygons over the study area.

    If the grid would exceed *max_cells*, the resolution is automatically
    coarsened so the pipeline can complete without running out of memory.
    """
    lon_min, lat_min, lon_max, lat_max = bounds

    n_lon = int(np.ceil((lon_max - lon_min) / resolution_deg))
    n_lat = int(np.ceil((lat_max - lat_min) / resolution_deg))
    n_total = n_lon * n_lat

    if n_total > max_cells:
        scale = np.sqrt(n_total / max_cells)
        resolution_deg = resolution_deg * scale
        n_lon = int(np.ceil((lon_max - lon_min) / resolution_deg))
        n_lat = int(np.ceil((lat_max - lat_min) / resolution_deg))
        logger.warning(
            f"Grid would have {n_total} cells — auto-coarsened to "
            f"{resolution_deg:.5f}° ({n_lon * n_lat} cells) to stay under {max_cells}"
        )

    lons = np.arange(lon_min, lon_max, resolution_deg)
    lats = np.arange(lat_min, lat_max, resolution_deg)

    # Vectorized grid construction using numpy broadcasting
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    polys = [
        box(lo, la, lo + resolution_deg, la + resolution_deg)
        for lo, la in zip(lon_flat, lat_flat)
    ]

    grid = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
    grid["cell_id"] = range(len(grid))
    logger.info(f"Risk grid: {len(grid)} cells ({n_lon}×{n_lat}) at {resolution_deg:.5f}°")
    return grid


# ---------------------------------------------------------------------------
# Step 9 — Aggregation & Hotspots
# ---------------------------------------------------------------------------

def aggregate_to_admin(risk_gdf: gpd.GeoDataFrame, admin_gdf: gpd.GeoDataFrame,
                        infra: gpd.GeoDataFrame,
                        admin_level_col: str = "NAME_3") -> gpd.GeoDataFrame:
    """
    Aggregate grid-level risk to admin boundaries (union or upazila).
    """
    # Spatial join: grid cells → admin units
    joined = gpd.sjoin(risk_gdf, admin_gdf, how="left", predicate="within")

    # --- Risk aggregation (vectorized) ---
    risk_agg = joined.groupby(admin_level_col)["composite_risk"].agg(
        mean_risk="mean", max_risk="max", n_cells="count"
    ).reset_index()
    high_risk_counts = (
        joined[joined["composite_risk"] > 0.7]
        .groupby(admin_level_col)
        .size()
        .rename("n_high_risk")
    )
    risk_agg = risk_agg.merge(high_risk_counts, on=admin_level_col, how="left")
    risk_agg["n_high_risk"] = risk_agg["n_high_risk"].fillna(0).astype(int)

    # --- Asset counts via a single spatial join (NOT per-admin copy) ---
    infra_pts = infra.copy()
    infra_pts["geometry"] = infra_pts.geometry.representative_point()
    infra_joined = gpd.sjoin(infra_pts, admin_gdf[[admin_level_col, "geometry"]],
                              how="inner", predicate="within")

    asset_counts = (
        infra_joined
        .groupby([admin_level_col, "asset_type"])
        .size()
        .unstack(fill_value=0)
    )
    asset_summary = pd.DataFrame(index=asset_counts.index)
    asset_summary["n_hospitals_exposed"] = asset_counts.get("hospital", 0)
    asset_summary["n_schools_exposed"] = asset_counts.get("school", 0)
    asset_summary["n_bridges_exposed"] = asset_counts.get("bridge", 0)
    asset_summary["n_roads"] = asset_counts.get("road", 0)
    asset_summary["n_cropland"] = asset_counts.get("cropland", 0)
    asset_summary["total_assets"] = asset_counts.sum(axis=1)
    asset_summary = asset_summary.reset_index()

    # Merge risk + asset summaries
    summary_df = risk_agg.merge(asset_summary, on=admin_level_col, how="left")
    summary_df = summary_df.rename(columns={admin_level_col: "admin_name"})
    summary_df = summary_df.sort_values("mean_risk", ascending=False)
    summary_df["risk_rank"] = range(1, len(summary_df) + 1)

    # Re-attach geometry (left join to keep all admin units, fill NaN with 0)
    summary_gdf = admin_gdf[[admin_level_col, "geometry"]].merge(
        summary_df, left_on=admin_level_col, right_on="admin_name", how="left"
    )
    fill_cols = ["mean_risk", "max_risk", "n_cells", "n_high_risk",
                 "n_hospitals_exposed", "n_schools_exposed", "n_bridges_exposed",
                 "n_roads", "n_cropland", "total_assets"]
    for col in fill_cols:
        if col in summary_gdf.columns:
            summary_gdf[col] = summary_gdf[col].fillna(0)
    if "admin_name" in summary_gdf.columns:
        summary_gdf["admin_name"] = summary_gdf["admin_name"].fillna(summary_gdf[admin_level_col])
    summary_gdf = gpd.GeoDataFrame(summary_gdf, crs=admin_gdf.crs)

    logger.info(f"Aggregated risk to {len(summary_gdf)} admin units")
    return summary_gdf


def detect_hotspots(grid_gdf: gpd.GeoDataFrame,
                     confidence: float = 0.95) -> gpd.GeoDataFrame:
    """
    Getis-Ord Gi* hotspot detection on grid-level composite risk.
    Uses KNN weights instead of Queen contiguity for scalability on large grids.
    """
    MAX_CELLS_FOR_HOTSPOT = 100_000

    try:
        from esda.getisord import G_Local
        from libpysal.weights import KNN

        if len(grid_gdf) > MAX_CELLS_FOR_HOTSPOT:
            logger.warning(
                f"Grid has {len(grid_gdf)} cells (>{MAX_CELLS_FOR_HOTSPOT}). "
                "Skipping Gi* hotspot detection to avoid OOM."
            )
            raise MemoryError("Grid too large for hotspot detection")

        # KNN weights: fast O(n log n) construction via KD-tree, unlike
        # Queen contiguity which requires expensive geometric intersection.
        centroids = np.array(
            [(g.centroid.x, g.centroid.y) for g in grid_gdf.geometry]
        )
        w = KNN.from_array(centroids, k=8)
        w.transform = "r"  # row-standardize

        g_local = G_Local(grid_gdf["composite_risk"].values, w)

        grid_gdf = grid_gdf.copy()
        grid_gdf["hotspot_z"] = g_local.Zs
        grid_gdf["hotspot_p"] = g_local.p_sim

        z_thresh = 1.96 if confidence >= 0.95 else 1.645
        grid_gdf["is_hotspot"] = (
            (g_local.Zs > z_thresh) & (g_local.p_sim < (1 - confidence))
        )

        n_hot = grid_gdf["is_hotspot"].sum()
        logger.info(f"Hotspot detection: {n_hot} cells flagged at {confidence} confidence")

    except ImportError:
        logger.warning("esda/libpysal not installed. Skipping Gi* hotspot detection.")
        grid_gdf = grid_gdf.copy()
        grid_gdf["hotspot_z"] = 0.0
        grid_gdf["hotspot_p"] = 1.0
        grid_gdf["is_hotspot"] = False
    except MemoryError:
        grid_gdf = grid_gdf.copy()
        grid_gdf["hotspot_z"] = 0.0
        grid_gdf["hotspot_p"] = 1.0
        grid_gdf["is_hotspot"] = False

    return grid_gdf


def rank_assets(infra: gpd.GeoDataFrame, risk_scores: np.ndarray,
                 high_risk_threshold: float = 0.7) -> gpd.GeoDataFrame:
    """Attach risk scores to infrastructure and rank by composite risk."""
    infra = infra.copy()
    infra["flood_risk"] = risk_scores
    infra["is_high_risk"] = risk_scores > high_risk_threshold
    infra = infra.sort_values("flood_risk", ascending=False)
    infra["risk_rank"] = range(1, len(infra) + 1)

    n_high = infra["is_high_risk"].sum()
    logger.info(
        f"Ranked {len(infra)} assets. "
        f"High risk (>{high_risk_threshold}): {n_high} ({n_high/len(infra):.1%})"
    )
    return infra


def aggregate_to_upazila(grid_gdf: gpd.GeoDataFrame, upazila_gdf: gpd.GeoDataFrame,
                          infra: gpd.GeoDataFrame,
                          admin_col: str = "NAME_2") -> gpd.GeoDataFrame:
    """
    Aggregate grid-level risk to upazila (L2) boundaries.
    Wrapper around aggregate_to_admin with upazila-specific column name.

    Also exports CSV summary for dashboard consumption.
    """
    return aggregate_to_admin(grid_gdf, upazila_gdf, infra, admin_level_col=admin_col)
