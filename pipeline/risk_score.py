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
    """
    weights = cfg["risk"]["exposure_weights"]
    n_cells = len(grid_gdf)
    exposure = np.zeros(n_cells)

    # Spatial join: which assets fall in which cell
    infra_pts = infra.copy()
    infra_pts["geometry"] = infra_pts.geometry.representative_point()
    joined = gpd.sjoin(infra_pts, grid_gdf, how="left", predicate="within")

    for cell_idx in range(n_cells):
        cell_assets = joined[joined["index_right"] == cell_idx]
        if len(cell_assets) == 0:
            continue

        types = cell_assets["asset_type"].value_counts()
        score = 0.0
        score += types.get("hospital", 0) * weights.get("hospital_school_presence", 0.20)
        score += types.get("school", 0) * weights.get("hospital_school_presence", 0.20)
        score += types.get("bridge", 0) * weights.get("bridge_presence", 0.15)
        score += types.get("road", 0) * weights.get("road_length_km", 0.15)
        score += types.get("cropland", 0) * weights.get("cropland_area_km2", 0.15)
        score += types.get("flood_shelter", 0) * weights.get("embankment_shelter_presence", 0.10)
        score += types.get("embankment", 0) * weights.get("embankment_shelter_presence", 0.10)
        score += len(cell_assets) * weights.get("building_count", 0.25)
        exposure[cell_idx] = score

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
                      resolution_deg: float) -> gpd.GeoDataFrame:
    """Create a regular grid of polygons over the study area."""
    lon_min, lat_min, lon_max, lat_max = bounds

    lons = np.arange(lon_min, lon_max, resolution_deg)
    lats = np.arange(lat_min, lat_max, resolution_deg)

    polys = []
    for lon in lons:
        for lat in lats:
            polys.append(box(lon, lat, lon + resolution_deg, lat + resolution_deg))

    grid = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
    grid["cell_id"] = range(len(grid))
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

    summary = []
    for name, group in joined.groupby(admin_level_col):
        record = {
            "admin_name": name,
            "mean_risk": group["composite_risk"].mean(),
            "max_risk": group["composite_risk"].max(),
            "n_cells": len(group),
            "n_high_risk": (group["composite_risk"] > 0.7).sum(),
        }

        # Count exposed assets in this admin unit
        admin_geom = admin_gdf[admin_gdf[admin_level_col] == name].geometry.unary_union
        if admin_geom is not None:
            infra_pts = infra.copy()
            infra_pts["geometry"] = infra_pts.geometry.representative_point()
            assets_in = infra_pts[infra_pts.within(admin_geom)]
            record["n_hospitals_exposed"] = (assets_in["asset_type"] == "hospital").sum()
            record["n_schools_exposed"] = (assets_in["asset_type"] == "school").sum()
            record["n_bridges_exposed"] = (assets_in["asset_type"] == "bridge").sum()
            record["n_roads"] = (assets_in["asset_type"] == "road").sum()
            record["n_cropland"] = (assets_in["asset_type"] == "cropland").sum()
            record["total_assets"] = len(assets_in)

        summary.append(record)

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("mean_risk", ascending=False)
    summary_df["risk_rank"] = range(1, len(summary_df) + 1)

    # Re-attach geometry
    summary_gdf = admin_gdf[[admin_level_col, "geometry"]].merge(
        summary_df, left_on=admin_level_col, right_on="admin_name"
    )
    summary_gdf = gpd.GeoDataFrame(summary_gdf, crs=admin_gdf.crs)

    logger.info(f"Aggregated risk to {len(summary_gdf)} admin units")
    return summary_gdf


def detect_hotspots(grid_gdf: gpd.GeoDataFrame,
                     confidence: float = 0.95) -> gpd.GeoDataFrame:
    """
    Getis-Ord Gi* hotspot detection on grid-level composite risk.
    """
    try:
        from esda.getisord import G_Local
        from libpysal.weights import Queen

        w = Queen.from_dataframe(grid_gdf)
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
