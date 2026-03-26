"""
Step 7 — Ordinary Kriging: Variogram fitting and spatial interpolation
of GNN flood risk scores to a continuous surface.
"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from pykrige.ok import OrdinaryKriging
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


def fit_and_execute_kriging(coords: np.ndarray, values: np.ndarray,
                             cfg: dict,
                             grid_bounds: tuple[float, float, float, float]
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Fit variogram and execute Ordinary Kriging on a regular grid.

    Args:
        coords: (N, 2) lon/lat of observation points
        values: (N,) GNN risk scores to interpolate
        cfg: pipeline config
        grid_bounds: (lon_min, lat_min, lon_max, lat_max)

    Returns:
        z: (H, W) kriged risk surface
        ss: (H, W) kriging variance
        grid_lon: 1D array of grid longitudes
        grid_lat: 1D array of grid latitudes
        variogram_params: dict of fitted variogram parameters
    """
    krig_cfg = cfg["kriging"]
    variogram_model = krig_cfg["variogram_model"]
    nlags = krig_cfg["nlags"]
    weight = krig_cfg["weight"]
    grid_res = krig_cfg.get("grid_resolution_deg", 0.005)

    lons = coords[:, 0]
    lats = coords[:, 1]
    lon_min, lat_min, lon_max, lat_max = grid_bounds

    # Subsample to avoid OOM — kriging distance matrix scales O(N × M)
    max_points = krig_cfg.get("max_points", 3000)
    if len(values) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(values), size=max_points, replace=False)
        lons = lons[idx]
        lats = lats[idx]
        values = values[idx]
        logger.info(f"Subsampled {len(coords)} → {max_points} points for kriging")

    logger.info(
        f"Fitting {variogram_model} variogram on {len(values)} points, "
        f"nlags={nlags}"
    )

    # Fit Ordinary Kriging
    ok = OrdinaryKriging(
        lons, lats, values,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
        nlags=nlags,
        weight=weight,
    )

    # Extract variogram parameters
    variogram_params = {
        "model": variogram_model,
        "sill": float(ok.variogram_model_parameters[0]),
        "range": float(ok.variogram_model_parameters[1]),
        "nugget": float(ok.variogram_model_parameters[2])
        if len(ok.variogram_model_parameters) > 2 else 0.0,
    }
    logger.info(f"Variogram params: {variogram_params}")

    # Define prediction grid — cap dimensions to avoid OOM
    max_grid_dim = krig_cfg.get("max_grid_dim", 200)
    grid_lon = np.arange(lon_min, lon_max, grid_res)
    grid_lat = np.arange(lat_min, lat_max, grid_res)
    if len(grid_lon) > max_grid_dim or len(grid_lat) > max_grid_dim:
        coarse_res = max((lon_max - lon_min), (lat_max - lat_min)) / max_grid_dim
        grid_lon = np.arange(lon_min, lon_max, coarse_res)
        grid_lat = np.arange(lat_min, lat_max, coarse_res)
        logger.info(f"Coarsened grid to {len(grid_lon)}×{len(grid_lat)} (res={coarse_res:.4f}°) to fit in memory")

    logger.info(
        f"Kriging on {len(grid_lon)}×{len(grid_lat)} grid "
        f"(res={grid_res}°)"
    )

    z, ss = ok.execute("grid", grid_lon, grid_lat)

    # Clip to [0, 1]
    z_data = np.clip(z.data if hasattr(z, 'data') else z, 0, 1)
    ss_data = ss.data if hasattr(ss, 'data') else ss

    logger.info(
        f"Kriged surface: min={z_data.min():.3f}, max={z_data.max():.3f}, "
        f"mean={z_data.mean():.3f}"
    )

    return z_data, ss_data, grid_lon, grid_lat, variogram_params


def hybrid_fusion(kriged_surface: np.ndarray,
                   gnn_scores: np.ndarray,
                   labels: np.ndarray,
                   coords: np.ndarray,
                   grid_lon: np.ndarray,
                   grid_lat: np.ndarray,
                   cfg: dict) -> np.ndarray:
    """
    Hybrid GNN + Kriging: krige the GNN residuals and add to base surface.

    final_risk(x) = kriging_prediction(x) + kriged_gnn_residual(x)

    At node locations, GNN dominates. Between nodes, kriging interpolates.
    """
    # Compute residuals: observed - kriged_at_nodes
    # First, sample kriged surface at node locations
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (grid_lat, grid_lon), kriged_surface,
        method="linear", bounds_error=False, fill_value=0.0
    )

    kriged_at_nodes = interpolator(
        np.column_stack([coords[:, 1], coords[:, 0]])  # (lat, lon)
    )

    residuals = gnn_scores - kriged_at_nodes

    # Krige the residuals (subsample to avoid OOM)
    krig_cfg = cfg["kriging"]
    max_points = krig_cfg.get("max_points", 3000)
    r_lons, r_lats, r_vals = coords[:, 0], coords[:, 1], residuals
    if len(residuals) > max_points:
        rng = np.random.default_rng(43)
        idx = rng.choice(len(residuals), size=max_points, replace=False)
        r_lons, r_lats, r_vals = r_lons[idx], r_lats[idx], r_vals[idx]
    try:
        ok_resid = OrdinaryKriging(
            r_lons, r_lats, r_vals,
            variogram_model=krig_cfg["variogram_model"],
            verbose=False,
            nlags=krig_cfg["nlags"],
            weight=krig_cfg["weight"],
        )
        z_resid, _ = ok_resid.execute("grid", grid_lon, grid_lat)
        z_resid = z_resid.data if hasattr(z_resid, 'data') else z_resid
    except Exception as exc:
        logger.warning(f"Residual kriging failed: {exc}. Using base surface only.")
        z_resid = np.zeros_like(kriged_surface)

    # Fuse
    fused = np.clip(kriged_surface + z_resid, 0, 1)

    logger.info(
        f"Hybrid fusion: base mean={kriged_surface.mean():.3f}, "
        f"residual mean={z_resid.mean():.3f}, "
        f"fused mean={fused.mean():.3f}"
    )

    return fused


def save_kriged_surface(data: np.ndarray, grid_lon: np.ndarray,
                         grid_lat: np.ndarray, output_path: str,
                         description: str = "kriged_surface") -> None:
    """Save kriged surface as GeoTIFF."""
    lon_min, lon_max = grid_lon[0], grid_lon[-1]
    lat_min, lat_max = grid_lat[0], grid_lat[-1]
    height, width = data.shape

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)
        dst.update_tags(description=description)

    logger.info(f"Saved {description} → {output_path}")


def save_variogram_params(params: dict, output_path: str) -> None:
    """Save variogram parameters as JSON."""
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Variogram params → {output_path}")
