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

def _cell_centers(grid_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """(N, 2) lon/lat of grid cell centres (grid cells are axis-aligned boxes)."""
    b = grid_gdf.geometry.bounds
    return np.column_stack([(b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2])


def _log_minmax(x: np.ndarray) -> np.ndarray:
    """log1p then min-max to [0, 1]; tames heavy-tailed counts and densities."""
    x = np.log1p(np.clip(x, 0, None))
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def compute_exposure_grid(infra: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame,
                           cfg: dict) -> np.ndarray:
    """
    Exposure per grid cell: type-weighted count of mapped assets in the cell,
    log-scaled to [0, 1]. Cells with no mapped assets have zero exposure.
    """
    weights = cfg["risk"]["exposure_weights"]
    exposure = np.zeros(len(grid_gdf))

    infra_pts = infra.copy()
    infra_pts["geometry"] = infra_pts.geometry.representative_point()
    joined = gpd.sjoin(infra_pts, grid_gdf, how="inner", predicate="within")
    if len(joined) == 0:
        return exposure

    type_weight_map = {
        "hospital": weights.get("hospital_school_presence", 0.20),
        "school": weights.get("hospital_school_presence", 0.20),
        "bridge": weights.get("bridge_presence", 0.15),
        "road": weights.get("road_length_km", 0.15),
        "railway": weights.get("road_length_km", 0.15),
        "cropland": weights.get("cropland_area_km2", 0.15),
        "flood_shelter": weights.get("embankment_shelter_presence", 0.10),
        "embankment": weights.get("embankment_shelter_presence", 0.10),
    }
    default_w = weights.get("other_asset", 0.05)

    joined["_weight"] = joined["asset_type"].map(type_weight_map).fillna(default_w)
    cell_scores = joined.groupby("index_right")["_weight"].sum()
    exposure[cell_scores.index.values] = cell_scores.values
    return _log_minmax(exposure)


def compute_population_exposure_grid(grid_gdf: gpd.GeoDataFrame,
                                     cfg: dict) -> np.ndarray:
    """Population exposure per grid cell, log-scaled to [0, 1].

    The asset-count exposure answers "where is infrastructure at risk"; cities
    dominate it because that is where infrastructure is. This answers the
    different question "where would people be affected", and the two are
    reported side by side rather than one standing in for the other.
    """
    from pipeline.feature_extract import sample_raster_at_points

    pop_path = cfg["data"]["vulnerability"].get("population_path", "")
    if not pop_path or not Path(pop_path).exists():
        logger.warning("Population raster missing; population exposure skipped.")
        return np.zeros(len(grid_gdf))

    centers = _cell_centers(grid_gdf)
    pop = sample_raster_at_points(pop_path, [tuple(c) for c in centers])
    return _log_minmax(pop)


def compute_vulnerability_grid(grid_gdf: gpd.GeoDataFrame, infra: gpd.GeoDataFrame,
                                cfg: dict) -> tuple[np.ndarray, dict]:
    """
    Vulnerability per grid cell from components with real data behind them.

    Components without a data source are left out and the remaining weights
    are renormalised, rather than filled with a constant.

    Returns:
        vuln: (N,) scores in [0, 1]
        used_weights: the renormalised weights actually applied
    """
    from pipeline.feature_extract import project_coords, sample_raster_at_points

    weights = cfg["risk"]["vulnerability_weights"]
    crs = cfg["aoi"]["crs"]

    centers = _cell_centers(grid_gdf)
    centers_m = project_coords(centers, crs)

    infra_pts = infra.copy()
    infra_pts["geometry"] = infra_pts.geometry.representative_point()
    infra_m = project_coords(
        np.column_stack([infra_pts.geometry.x, infra_pts.geometry.y]), crs
    )

    components = {
        "dist_hospital": _normalize_dist(
            _min_dist_to_type(centers_m, infra_pts, infra_m, ["hospital"])),
        "dist_flood_shelter": _normalize_dist(
            _min_dist_to_type(centers_m, infra_pts, infra_m, ["flood_shelter"])),
        "dist_primary_road": _normalize_dist(
            _min_dist_to_type(centers_m, infra_pts, infra_m, ["road"])),
    }

    pop_path = cfg["data"]["vulnerability"].get("population_path", "")
    if pop_path and Path(pop_path).exists():
        pop = sample_raster_at_points(pop_path, [tuple(c) for c in centers])
        components["population_density"] = _log_minmax(pop)
    else:
        logger.warning("Population raster missing; vulnerability excludes it.")

    for name, path in cfg["data"]["vulnerability"].get("extra_rasters", {}).items():
        if Path(path).exists():
            components[name] = _log_minmax(
                sample_raster_at_points(path, [tuple(c) for c in centers]))

    used = {k: weights[k] for k in components if weights.get(k, 0) > 0}
    missing = sorted(set(weights) - set(used))
    if missing:
        logger.warning(f"Vulnerability components without data (excluded): {missing}")
    total = sum(used.values())
    used = {k: v / total for k, v in used.items()}

    vuln = sum(w * components[k] for k, w in used.items())
    logger.info(f"Vulnerability weights applied: {used}")
    return vuln, used


def _min_dist_to_type(cell_centers: np.ndarray, infra: gpd.GeoDataFrame,
                       infra_coords: np.ndarray, types: list[str]) -> np.ndarray:
    """Min distance (metres) from each cell centre to nearest asset of given types."""
    mask = infra["asset_type"].isin(types)
    if mask.sum() == 0:
        return np.full(len(cell_centers), 50000.0)
    tree = cKDTree(infra_coords[mask.values])
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
    """Risk = geometric mean of hazard, exposure and vulnerability, each in [0, 1].

    The geometric mean keeps the multiplicative logic (no exposure, no risk)
    while staying on the same [0, 1] scale as its inputs, so fixed class
    thresholds remain meaningful. Rescaling a raw product by its maximum
    instead pushes almost every cell towards zero.
    """
    return np.cbrt(np.clip(hazard, 0, 1) * np.clip(exposure, 0, 1)
                   * np.clip(vulnerability, 0, 1))


def assign_risk_classes(risk: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Class 1-5 by quantiles of the non-zero composite risk.

    The composite is a geometric mean of three bounded factors, so it rarely
    approaches 1 even where risk is locally worst. Fixed cut-offs would put
    every cell in the lowest class, so cells are ranked against each other —
    the usual practice for susceptibility maps. Class 0 means no mapped
    exposure, which is an absence of data rather than an absence of risk.
    """
    classes = np.zeros(len(risk), dtype=int)
    nonzero = risk > 0
    if nonzero.sum() == 0:
        return classes
    quantiles = np.quantile(risk[nonzero], np.linspace(0, 1, n_classes + 1)[1:-1])
    classes[nonzero] = np.searchsorted(quantiles, risk[nonzero], side="right") + 1
    return classes


def _mean_ignoring_nan(series):
    """Mean that returns NaN for an all-empty group rather than 0."""
    return series.mean()


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
                        admin_level_col: str = "admin_name",
                        high_risk_threshold: float = 0.7) -> gpd.GeoDataFrame:
    """
    Aggregate grid-level risk to administrative units (union, upazila or
    district). `admin_level_col` is the unit's name column; if the boundaries
    carry an `admin_label` column (name plus parent, e.g. "Abdullahpur
    (Gangachara)") it is kept for display, because unit names repeat across
    the country.
    """
    label_col = "admin_label" if "admin_label" in admin_gdf.columns else None
    # Spatial join: grid cells → admin units
    joined = gpd.sjoin(risk_gdf, admin_gdf, how="left", predicate="within")

    # --- Risk aggregation (vectorized) ---
    risk_agg = joined.groupby(admin_level_col)["composite_risk"].agg(
        mean_risk="mean", max_risk="max", n_cells="count"
    ).reset_index()

    # Population-weighted composite, when the grid carries it.
    if "composite_risk_people" in joined.columns:
        people_agg = joined.groupby(admin_level_col)["composite_risk_people"].agg(
            mean_risk_people="mean", max_risk_people="max"
        ).reset_index()
        risk_agg = risk_agg.merge(people_agg, on=admin_level_col, how="left")
    high_risk_counts = (
        joined[joined["composite_risk"] > high_risk_threshold]
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
    geom_cols = [admin_level_col] + ([label_col] if label_col else []) + ["geometry"]
    summary_gdf = admin_gdf[geom_cols].merge(
        summary_df, left_on=admin_level_col, right_on="admin_name", how="left"
    )
    count_cols = ["n_cells", "n_high_risk", "n_hospitals_exposed",
                  "n_schools_exposed", "n_bridges_exposed", "n_roads",
                  "n_cropland", "total_assets"]
    for col in count_cols:
        if col in summary_gdf.columns:
            summary_gdf[col] = summary_gdf[col].fillna(0)

    # A unit with no mapped assets has no exposure to score, which is missing
    # data rather than an absence of risk. Leave its risk NaN and flag it, so
    # the dashboard can say "no mapped assets" instead of showing a low score.
    summary_gdf["has_data"] = summary_gdf.get("total_assets", 0) > 0
    for col in ("mean_risk", "max_risk", "mean_risk_people", "max_risk_people"):
        if col in summary_gdf.columns:
            summary_gdf.loc[~summary_gdf["has_data"], col] = np.nan
    n_no_data = int((~summary_gdf["has_data"]).sum())
    if n_no_data:
        logger.info(
            f"{n_no_data}/{len(summary_gdf)} units have no mapped assets; "
            "their risk is left undefined"
        )
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

    grid_gdf = grid_gdf.copy()
    grid_gdf["hotspot_z"] = 0.0
    grid_gdf["hotspot_p"] = 1.0
    grid_gdf["is_hotspot"] = False

    # Composite risk is zero wherever no assets are mapped, and those cells
    # carry no information for clustering. Restricting Gi* to cells that hold
    # assets both sharpens the statistic and keeps the weights matrix small.
    subset = grid_gdf.index[grid_gdf["composite_risk"] > 0]
    if len(subset) == 0:
        logger.warning("No cells with non-zero risk — skipping hotspot detection.")
        return grid_gdf

    try:
        from esda.getisord import G_Local
        from libpysal.weights import KNN

        if len(subset) > MAX_CELLS_FOR_HOTSPOT:
            logger.warning(
                f"{len(subset)} risk-bearing cells (>{MAX_CELLS_FOR_HOTSPOT}). "
                "Skipping Gi* hotspot detection to avoid OOM."
            )
            raise MemoryError("Too many cells for hotspot detection")

        sub = grid_gdf.loc[subset]
        b = sub.geometry.bounds
        centroids = np.column_stack([(b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2])

        # KNN weights: O(n log n) via KD-tree, unlike Queen contiguity.
        w = KNN.from_array(centroids, k=8)
        w.transform = "r"  # row-standardize

        g_local = G_Local(sub["composite_risk"].values, w)

        z_thresh = 1.96 if confidence >= 0.95 else 1.645
        grid_gdf.loc[subset, "hotspot_z"] = g_local.Zs
        grid_gdf.loc[subset, "hotspot_p"] = g_local.p_sim
        grid_gdf.loc[subset, "is_hotspot"] = (
            (g_local.Zs > z_thresh) & (g_local.p_sim < (1 - confidence))
        )

        n_hot = int(grid_gdf["is_hotspot"].sum())
        logger.info(
            f"Hotspot detection on {len(subset)} risk-bearing cells: "
            f"{n_hot} flagged at {confidence} confidence"
        )

    except ImportError:
        logger.warning("esda/libpysal not installed. Skipping Gi* hotspot detection.")
    except MemoryError:
        pass

    return grid_gdf


def rank_assets(infra: gpd.GeoDataFrame, risk_scores: np.ndarray,
                 high_risk_threshold: float = 0.7,
                 grid_gdf: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Attach model scores to infrastructure and rank them.

    ``flood_risk`` is the GNN probability that the asset sits on flood-prone
    ground. If ``grid_gdf`` is given, the hazard/exposure/vulnerability and
    composite scores of the grid cell containing each asset are attached too.
    """
    infra = infra.copy()
    infra["flood_risk"] = risk_scores
    if grid_gdf is not None:
        cols = ["hazard", "exposure", "vulnerability", "composite_risk",
                "risk_class", "population_exposure", "composite_risk_people"]
        cols = [c for c in cols if c in grid_gdf.columns]
        pts = infra[["geometry"]].copy()
        pts["geometry"] = pts.geometry.representative_point()
        hit = gpd.sjoin(pts, grid_gdf[cols + ["geometry"]], how="left",
                        predicate="within")
        hit = hit[~hit.index.duplicated(keep="first")]
        for c in cols:
            name = "risk_class" if c == "risk_class" else f"cell_{c}"
            infra[name] = hit[c].reindex(infra.index).values
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
                          admin_col: str = "admin_name",
                          high_risk_threshold: float = 0.7) -> gpd.GeoDataFrame:
    """Deprecated alias for aggregate_to_admin, kept for external callers."""
    return aggregate_to_admin(grid_gdf, upazila_gdf, infra, admin_level_col=admin_col,
                              high_risk_threshold=high_risk_threshold)
