"""
Central data loader for pipeline outputs, scoped per region.

Every loader takes a region id and resolves its directories through
dashboard.data.regions, so several study areas coexist without the paths
being hardcoded. When an output is missing the loaders return empty results
or None — they never substitute made-up values.

Heavy imports (geopandas, numpy, pandas, rasterio) are deferred to first use.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import streamlit as st

from dashboard.data.regions import region_config, region_paths

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _import_geo():
    """Lazy import heavy geo libs on first actual data access."""
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    return np, pd, gpd


def raster_paths(region: str) -> dict:
    """Terrain and hazard rasters for a region, whether or not they exist."""
    paths = region_paths(region)
    return {
        "dem": paths["processed"] / "dem_reprojected.tif",
        "slope": paths["processed"] / "dem_derivatives" / "slope.tif",
        "hand": paths["processed"] / "dem_derivatives" / "hand.tif",
        "flood_risk": paths["output"] / "flood_risk_kriged.tif",
        "kriging_variance": paths["output"] / "kriging_variance.tif",
        "landslide": paths["output"] / "landslide_susceptibility.tif",
    }


# ── GeoDataFrames (parquet cache > geojson) ─────────────────────────────────

@st.cache_data(ttl=600)
def load_gdf_fast(region: str, name: str):
    """Load a pipeline GeoDataFrame, preferring the parquet display cache."""
    _, _, gpd = _import_geo()
    paths = region_paths(region)
    parquet_path = paths["cache"] / f"{name}.parquet"
    geojson_path = paths["output"] / f"{name}.geojson"

    if parquet_path.exists():
        gdf = gpd.read_parquet(parquet_path)
    elif geojson_path.exists():
        gdf = gpd.read_file(geojson_path)
    else:
        return gpd.GeoDataFrame()
    gdf.columns = [str(c) for c in gdf.columns]
    return gdf


@st.cache_data(ttl=600)
def load_heatmap_points(region: str) -> list:
    """Pre-computed heatmap points from the display cache."""
    path = region_paths(region)["cache"] / "heatmap_points.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data(ttl=600)
def load_cached_raster_overlay(region: str, raster_name: str) -> dict | None:
    """Pre-rendered raster overlay (reprojected to WGS84 by the cache step)."""
    path = region_paths(region)["cache"] / f"raster_{raster_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_raster_overlay(region: str, raster_name: str) -> dict | None:
    """Overlay for folium ImageOverlay, or None when not pre-rendered.

    Rendering happens in preprocess_cache.py, which reprojects first: an
    ImageOverlay places pixels by lat/lon corners, so a projected raster
    drawn as-is would land in the wrong place.
    """
    return load_cached_raster_overlay(region, raster_name)


# ── Pipeline metadata ────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_pipeline_metadata(region: str) -> dict | None:
    """pipeline_metadata.json for a region, if the pipeline has written it."""
    path = region_paths(region)["output"] / "pipeline_metadata.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load pipeline metadata: {e}")
    return None


@st.cache_data(ttl=300)
def load_validation_metrics(region: str) -> dict | None:
    """Validation against observed floods, if it has been run."""
    path = region_paths(region)["output"] / "validation_metrics.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load validation metrics: {e}")
    return None


@st.cache_data(ttl=300)
def load_hazard_model(region: str) -> dict | None:
    """Terrain hazard model description (features, coefficients, validation)."""
    path = region_paths(region)["output"] / "hazard_model.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load hazard model: {e}")
    return None


@st.cache_data(ttl=300)
def load_landslide_model(region: str) -> dict | None:
    """Fitted landslide model description (features, coefficients, validation)."""
    path = region_paths(region)["output"] / "landslide_model.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load landslide model: {e}")
    return None


@st.cache_data(ttl=300)
def load_landslide_summary(region: str) -> list:
    """Per-upazila landslide susceptibility, if computed."""
    path = region_paths(region)["output"] / "landslide_upazila.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load landslide summary: {e}")
    return []


# ── Kriging variance ─────────────────────────────────────────────────────────

@st.cache_resource
def _load_kriging_raster(region: str):
    """Preload the kriging variance raster once per region."""
    path = region_paths(region)["output"] / "kriging_variance.tif"
    if path.exists():
        try:
            import rasterio
            with rasterio.open(path) as src:
                return {
                    "data": src.read(1),
                    "transform": src.transform,
                    "nodata": src.nodata,
                }
        except Exception as e:
            logger.warning(f"Could not read kriging variance: {e}")
    return None


def _kriging_ci(raster, lat: float, lon: float) -> float | None:
    """95% CI width of the kriged surface at a point, or None if unavailable."""
    np, _, _ = _import_geo()
    if raster is None:
        return None
    import rasterio
    row, col = rasterio.transform.rowcol(raster["transform"], lon, lat)
    data = raster["data"]
    if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
        return None
    val = data[row, col]
    if val == raster["nodata"] or np.isnan(val) or val < 0:
        return None
    return round(float(2 * 1.96 * np.sqrt(val)), 3)


def get_kriging_ci_at_point(region: str, lat: float, lon: float) -> float | None:
    return _kriging_ci(_load_kriging_raster(region), lat, lon)


@st.cache_data(ttl=300)
def get_kriging_ci_batch(region: str, coords: tuple) -> list[float | None]:
    """95% CI widths for a batch of (lat, lon) tuples."""
    raster = _load_kriging_raster(region)
    return [_kriging_ci(raster, lat, lon) for lat, lon in coords]


# ── Mapped shelters ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_mapped_shelters(region: str):
    """Flood shelters mapped in OpenStreetMap, with their model scores.

    OSM records location and name only: capacity and open/closed status are
    unknown and must not be displayed.
    """
    assets = load_gdf_fast(region, "risk_ranked_assets")
    if len(assets) == 0 or "asset_type" not in assets.columns:
        return assets
    shelters = assets[assets["asset_type"] == "flood_shelter"].copy()
    pts = shelters.geometry.representative_point()
    shelters["lat"] = pts.y
    shelters["lon"] = pts.x
    return shelters


# ── Population density ───────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def get_pop_density_points(region: str, bounds: tuple, n_side: int = 60) -> list:
    """Sample WorldPop density on a lattice over `bounds`.

    Returns [(lat, lon, weight)] with weights scaled to [0, 1], or [] if the
    raster is unavailable.
    """
    np, _, _ = _import_geo()
    cfg = region_config(region)
    pop_path = Path(
        cfg.get("data", {}).get("vulnerability", {}).get("population_path", "")
    )
    if not pop_path.is_absolute():
        pop_path = region_paths(region)["raw"].parents[1] / pop_path
    if not pop_path.exists():
        return []
    try:
        import rasterio
        west, south, east, north = bounds
        lons, lats = np.meshgrid(np.linspace(west, east, n_side),
                                 np.linspace(south, north, n_side))
        coords = list(zip(lons.ravel(), lats.ravel()))
        with rasterio.open(pop_path) as src:
            vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = 0
        vals = np.log1p(np.clip(np.nan_to_num(vals), 0, None))
        if vals.max() <= 0:
            return []
        vals /= vals.max()
        return [(lat, lon, float(v)) for (lon, lat), v in zip(coords, vals) if v > 0]
    except Exception as e:
        logger.warning(f"Could not sample WorldPop: {e}")
        return []


# ── AlphaEarth clusters ──────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def get_alphaearth_clusters(region: str) -> dict | None:
    """AlphaEarth cluster GeoJSON, if the optional step has been run."""
    path = region_paths(region)["output"] / "alphaearth_clusters.geojson"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load AlphaEarth clusters: {e}")
    return None
