"""
Step 3 — Feature Extraction: Sample raster values at infrastructure centroids,
compute distances, build node feature matrix.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_centroids(infra: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute representative point (centroid) for each infrastructure feature."""
    infra = infra.copy()
    infra["centroid"] = infra.geometry.representative_point()
    infra["lon"] = infra["centroid"].x
    infra["lat"] = infra["centroid"].y
    return infra


def sample_raster_at_points(raster_path: str,
                             coords: list[tuple[float, float]],
                             band: int = 1) -> np.ndarray:
    """Sample raster values at given (lon, lat) coordinates."""
    with rasterio.open(raster_path) as src:
        values = np.array([v[0] for v in src.sample(coords, indexes=band)])
    # Replace nodata with 0
    values = np.where(np.isnan(values) | (values < -9000), 0, values)
    return values


def compute_distances_to_features(infra: gpd.GeoDataFrame,
                                   target_types: list[str],
                                   all_coords: np.ndarray) -> np.ndarray:
    """Compute distance from each node to nearest feature of given types."""
    mask = infra["asset_type"].isin(target_types)
    if mask.sum() == 0:
        logger.warning(f"No features of types {target_types} found. Returning max dist.")
        return np.full(len(infra), 50000.0)  # 50km default

    target_coords = all_coords[mask.values]
    tree = cKDTree(target_coords)
    dists, _ = tree.query(all_coords, k=1)
    return dists


def extract_features(cfg: dict, infra: gpd.GeoDataFrame,
                      processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Build node feature matrix X, coordinate array, and labels y.

    Returns:
        X_scaled: (N, D) standardized feature matrix
        coords: (N, 2) array of (lon, lat)
        y: (N,) binary flood labels
        scaler: fitted StandardScaler
    """
    infra = compute_centroids(infra)
    coords = np.column_stack([infra["lon"].values, infra["lat"].values])
    coord_list = list(zip(infra["lon"], infra["lat"]))

    features = {}

    # --- Raster-sampled features ---
    deriv_dir = processed_dir / "dem_derivatives"

    raster_map = {
        "elevation": str(processed_dir / "dem_reprojected.tif"),
        "slope": str(deriv_dir / "slope.tif"),
        "twi": str(deriv_dir / "twi.tif"),
        "hand": str(deriv_dir / "hand.tif"),
        "flow_acc": str(deriv_dir / "flow_accumulation.tif"),
    }

    for name, path in raster_map.items():
        if Path(path).exists():
            features[name] = sample_raster_at_points(path, coord_list)
            logger.info(f"Sampled {name}: min={features[name].min():.2f}, max={features[name].max():.2f}")
        else:
            logger.warning(f"Raster not found: {path}. Using zeros for {name}.")
            features[name] = np.zeros(len(infra))

    # Population density
    pop_path = cfg.get("data", {}).get("vulnerability", {}).get("population_path", "")
    if Path(pop_path).exists():
        features["pop_density"] = sample_raster_at_points(pop_path, coord_list)
    else:
        features["pop_density"] = np.zeros(len(infra))

    # --- Distance-based features ---
    features["dist_hospital"] = compute_distances_to_features(
        infra, ["hospital"], coords
    )
    features["dist_school"] = compute_distances_to_features(
        infra, ["school"], coords
    )
    features["dist_shelter"] = compute_distances_to_features(
        infra, ["flood_shelter"], coords
    )
    features["dist_road"] = compute_distances_to_features(
        infra, ["road"], coords
    )

    # Compute distance to nearest waterway (if available)
    features["dist_water"] = compute_distances_to_features(
        infra, ["irrigation"], coords  # waterways as proxy
    )

    # --- Categorical encoding ---
    asset_codes = infra["asset_type"].astype("category").cat.codes.values.astype(np.float32)
    features["asset_type_code"] = asset_codes

    # --- Assemble feature matrix ---
    feature_names = list(features.keys())
    X = np.column_stack([features[name] for name in feature_names])

    # Handle NaN / inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Labels ---
    flood_label_path = str(processed_dir / "flood_proxy_labels.tif")
    if Path(flood_label_path).exists():
        y = sample_raster_at_points(flood_label_path, coord_list)
        y = (y > 0.5).astype(np.float32)
    else:
        logger.warning("No flood labels found. Using zeros.")
        y = np.zeros(len(infra), dtype=np.float32)

    logger.info(
        f"Feature matrix: {X_scaled.shape}, Labels: {y.shape} "
        f"(positive rate: {y.mean():.2%})"
    )

    # Save to parquet
    feat_df = infra[["asset_type", "name", "priority", "lon", "lat"]].copy()
    for i, name in enumerate(feature_names):
        feat_df[name] = X_scaled[:, i]
    feat_df["flood_label"] = y
    feat_df.to_parquet(str(processed_dir / "node_features.parquet"), index=False)

    return X_scaled, coords, y, scaler
