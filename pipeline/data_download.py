"""
Automatic data download functions for the SGMDI pipeline.

Downloads GADM admin boundaries, SRTM DEM, JRC Global Surface Water,
and WorldPop population density data required by the pipeline.
"""

import gzip
import math
import io
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger("sgmdi.download")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, description: str = "") -> bool:
    """Stream-download a file with progress logging. Returns True on success."""
    label = description or dest.name
    logger.info(f"Downloading {label} from {url}")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1 << 20  # 1 MB
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if pct % 25 == 0:
                        logger.info(f"  {label}: {pct}% ({downloaded}/{total} bytes)")
        logger.info(f"  Saved {label} -> {dest} ({downloaded} bytes)")
        return True
    except Exception as exc:
        logger.warning(f"Failed to download {label}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# 1. GADM admin boundaries
# ---------------------------------------------------------------------------

# Administrative boundaries come from geoBoundaries (gbOpen, CC BY 4.0), which
# may be redistributed — GADM's licence forbids it, so GADM cannot ship in the
# open data archive. geoBoundaries also carries the level this project needs:
#
#   ADM2 = district (64)      ADM3 = upazila (544)      ADM4 = union (5160)
#
# The earlier GADM files were mislabelled: `gadm_union.shp` held upazilas
# (GADM ENGTYPE_3 = "Upazilla") and `gadm_upazila.shp` held districts, so
# every "union" figure the dashboard showed was really an upazila. GADM has no
# union level for Bangladesh at all.
_GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/{level}/"

_ADMIN_LEVELS = {
    "district": "ADM2",
    "upazila": "ADM3",
    "union": "ADM4",
}


def download_admin_boundaries(cfg: dict, output_dir: Path) -> dict:
    """Download geoBoundaries admin levels, clipped to the study area.

    Each output carries a standardised `admin_name` column plus the names of
    its parent units, because geoBoundaries names are not unique on their own
    (Bangladesh has many unions called "Abdullahpur").

    Returns {level_name: path}.
    """
    import geopandas as gpd

    output_dir.mkdir(parents=True, exist_ok=True)
    iso = cfg.get("aoi", {}).get("iso3", "BGD")
    bbox = tuple(cfg["aoi"]["bbox"])

    results = {}
    frames = {}
    for name, level in _ADMIN_LEVELS.items():
        out_path = output_dir / f"{iso.lower()}_{name}.gpkg"
        if out_path.exists():
            logger.info(f"{name} boundaries already exist at {out_path}")
            results[name] = str(out_path)
            frames[name] = gpd.read_file(out_path)
            continue

        # National files are cached and shared between regions: the ADM4 file
        # alone is ~325 MB, and every region would otherwise re-download it.
        cache_dir = Path("data/shared/geoboundaries")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{iso}_{level}.geojson"

        if not cached.exists():
            api_url = _GEOBOUNDARIES_API.format(iso=iso, level=level)
            try:
                meta = requests.get(api_url, timeout=120).json()
                meta = meta[0] if isinstance(meta, list) else meta
                download_url = meta["gjDownloadURL"]
            except Exception as exc:
                logger.warning(f"geoBoundaries API failed for {level}: {exc}")
                results[name] = ""
                continue

            if not _download_file(download_url, cached, f"geoBoundaries {level}"):
                cached.unlink(missing_ok=True)
                results[name] = ""
                continue
        else:
            logger.info(f"Using cached national {level} file: {cached}")

        # Read only what intersects the study area; the bbox filter keeps
        # memory in check on the large national files.
        gdf = gpd.read_file(cached, bbox=bbox)

        gdf = gdf.rename(columns={"shapeName": "admin_name"})
        gdf["admin_level"] = name
        keep = [c for c in ["admin_name", "admin_level", "shapeID", "geometry"]
                if c in gdf.columns]
        gdf = gdf[keep]
        frames[name] = gdf

        gdf.to_file(out_path, driver="GPKG")
        logger.info(f"Saved {len(gdf)} {name} boundaries → {out_path}")
        results[name] = str(out_path)

    _attach_parent_names(frames, results, output_dir, iso)
    return results


def _attach_parent_names(frames: dict, results: dict, output_dir: Path,
                         iso: str) -> None:
    """Label each unit with its parent units, so names are unambiguous.

    geoBoundaries gives no parent field, so parents are found by locating each
    unit's representative point inside the coarser level.
    """
    import geopandas as gpd

    for child, parents in [("union", ["upazila", "district"]),
                           ("upazila", ["district"])]:
        child_gdf = frames.get(child)
        if child_gdf is None or len(child_gdf) == 0:
            continue
        if all(f"{p}_name" in child_gdf.columns for p in parents):
            continue

        points = child_gdf[["geometry"]].copy()
        points["geometry"] = points.geometry.representative_point()

        for parent in parents:
            parent_gdf = frames.get(parent)
            if parent_gdf is None or len(parent_gdf) == 0:
                continue
            joined = gpd.sjoin(
                points, parent_gdf[["admin_name", "geometry"]],
                how="left", predicate="within",
            )
            joined = joined[~joined.index.duplicated(keep="first")]
            child_gdf[f"{parent}_name"] = joined["admin_name"].reindex(
                child_gdf.index).values

        # "Gangachara (Rangpur Sadar)" reads unambiguously; a bare name does not.
        parent_col = f"{parents[0]}_name"
        if parent_col in child_gdf.columns:
            child_gdf["admin_label"] = (
                child_gdf["admin_name"].fillna("unnamed")
                + child_gdf[parent_col].apply(
                    lambda v: f" ({v})" if isinstance(v, str) and v else "")
            )

        out_path = output_dir / f"{iso.lower()}_{child}.gpkg"
        child_gdf.to_file(out_path, driver="GPKG")
        logger.info(f"Labelled {child} boundaries with parent names → {out_path}")
        results[child] = str(out_path)


# ---------------------------------------------------------------------------
# 2. SRTM 30m DEM
# ---------------------------------------------------------------------------

# Tiles covering bbox [88.0, 24.0, 89.9, 26.7]
SHARED_DIR = Path("data/shared")


def _shared_cache(subdir: str, filename: str) -> Path:
    """Path in the cross-region cache for a national or tiled download.

    The same WorldPop country raster and JRC 10-degree tiles serve every
    region, so they are fetched once and clipped per region.
    """
    path = SHARED_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _srtm_tiles_for_bbox(bbox) -> list[tuple[int, int]]:
    """1-degree SRTM tiles covering a bbox [west, south, east, north].

    Derived rather than hardcoded so a different study region works without
    code changes. Tiles are named by their south-west corner.
    """
    west, south, east, north = bbox
    return [
        (lat, lon)
        for lat in range(math.floor(south), math.ceil(north))
        for lon in range(math.floor(west), math.ceil(east))
    ]

_SRTM_URL_TEMPLATE = (
    "https://elevation-tiles-prod.s3.amazonaws.com/skadi/"
    "N{lat:02d}/N{lat:02d}E{lon:03d}.hgt.gz"
)


def download_srtm_dem(cfg: dict, output_dir: Path) -> str:
    """Download SRTM 30m tiles, merge into a single GeoTIFF.

    Returns path to the merged DEM file.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_bounds

    output_dir.mkdir(parents=True, exist_ok=True)
    dem_path = output_dir / "dem_srtm_30m.tif"

    if dem_path.exists():
        logger.info(f"SRTM DEM already exists at {dem_path}, skipping.")
        return str(dem_path)

    tile_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for lat, lon in _srtm_tiles_for_bbox(cfg["aoi"]["bbox"]):
            tile_name = f"N{lat:02d}E{lon:03d}"
            url = _SRTM_URL_TEMPLATE.format(lat=lat, lon=lon)
            gz_path = Path(tmpdir) / f"{tile_name}.hgt.gz"
            hgt_path = Path(tmpdir) / f"{tile_name}.hgt"

            ok = _download_file(url, gz_path, f"SRTM tile {tile_name}")
            if not ok:
                continue

            # Decompress .hgt.gz -> .hgt
            try:
                with gzip.open(gz_path, "rb") as gz_in:
                    with open(hgt_path, "wb") as hgt_out:
                        shutil.copyfileobj(gz_in, hgt_out)
                logger.info(f"  Decompressed {tile_name}.hgt")
            except Exception as exc:
                logger.warning(f"Failed to decompress {tile_name}: {exc}")
                continue

            # Convert .hgt to GeoTIFF
            try:
                tif_path = Path(tmpdir) / f"{tile_name}.tif"
                _hgt_to_geotiff(hgt_path, tif_path, lat, lon)
                tile_paths.append(tif_path)
            except Exception as exc:
                logger.warning(f"Failed to convert {tile_name} to GeoTIFF: {exc}")
                continue

        if not tile_paths:
            logger.warning("No SRTM tiles downloaded successfully.")
            return ""

        # Merge tiles
        try:
            datasets = [rasterio.open(str(p)) for p in tile_paths]
            mosaic, out_transform = merge(datasets)
            for ds in datasets:
                ds.close()

            profile = {
                "driver": "GTiff",
                "dtype": mosaic.dtype,
                "width": mosaic.shape[2],
                "height": mosaic.shape[1],
                "count": 1,
                "crs": "EPSG:4326",
                "transform": out_transform,
                "compress": "deflate",
                "nodata": -32768,
            }
            with rasterio.open(str(dem_path), "w", **profile) as dst:
                dst.write(mosaic)

            logger.info(f"Merged SRTM DEM -> {dem_path}")
            return str(dem_path)
        except Exception as exc:
            logger.warning(f"Failed to merge SRTM tiles: {exc}")
            return ""


def _hgt_to_geotiff(hgt_path: Path, tif_path: Path, lat: int, lon: int):
    """Convert a raw SRTM .hgt file to a GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds

    data = np.fromfile(str(hgt_path), dtype=">i2")
    # SRTM3 = 1201x1201, SRTM1 = 3601x3601
    if data.size == 3601 * 3601:
        size = 3601
    elif data.size == 1201 * 1201:
        size = 1201
    else:
        raise ValueError(f"Unexpected HGT file size: {data.size}")

    data = data.reshape((size, size))
    transform = from_bounds(lon, lat, lon + 1, lat + 1, size, size)

    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "width": size,
        "height": size,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -32768,
    }
    with rasterio.open(str(tif_path), "w", **profile) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------------
# 3. JRC Global Surface Water
# ---------------------------------------------------------------------------

_JRC_URL_TEMPLATE = (
    "https://storage.googleapis.com/global-surface-water/downloads2021/"
    "occurrence/occurrence_{lon}E_{lat}Nv1_4_2021.tif"
)


def _jrc_tiles_for_bbox(bbox) -> list[str]:
    """JRC Global Surface Water tile URLs covering a bbox.

    Tiles are 10 degrees square, named by their north-west corner.
    """
    west, south, east, north = bbox
    lons = range(int(math.floor(west / 10) * 10), int(math.ceil(east / 10) * 10), 10)
    lats = range(int(math.ceil(south / 10) * 10), int(math.ceil(north / 10) * 10) + 10, 10)
    return [
        _JRC_URL_TEMPLATE.format(lon=lon, lat=lat)
        for lat in sorted(set(lats)) for lon in lons
    ]


def download_jrc_water(cfg: dict, output_dir: Path) -> str:
    """Download JRC Global Surface Water occurrence tile and clip to AOI.

    Returns path to the clipped raster.
    """
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds

    output_dir.mkdir(parents=True, exist_ok=True)
    jrc_path = output_dir / "jrc_water_occurrence.tif"

    if jrc_path.exists():
        logger.info(f"JRC water occurrence already exists at {jrc_path}, skipping.")
        return str(jrc_path)

    bbox = cfg.get("aoi", {}).get("bbox", [88.0, 24.0, 89.9, 26.7])

    urls = _jrc_tiles_for_bbox(bbox)
    logger.info(f"JRC tiles covering the study area: {len(urls)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # A study area can straddle two 10-degree tiles (Sylhet and the CHT do),
        # so download each and mosaic before clipping.
        tile_paths = []
        for url in urls:
            name = url.rsplit("/", 1)[-1]
            tile_path = _shared_cache("jrc", name)
            if tile_path.exists():
                logger.info(f"Using cached JRC tile {name}")
                tile_paths.append(tile_path)
            elif _download_file(url, tile_path, f"JRC tile {name}"):
                tile_paths.append(tile_path)
            else:
                tile_path.unlink(missing_ok=True)
        if not tile_paths:
            return ""

        raw_path = Path(tmpdir) / "jrc_raw.tif"
        if len(tile_paths) == 1:
            raw_path = tile_paths[0]
        else:
            from rasterio.merge import merge
            datasets = [rasterio.open(str(t)) for t in tile_paths]
            mosaic, transform = merge(datasets, bounds=tuple(bbox))
            profile = datasets[0].profile.copy()
            profile.update(width=mosaic.shape[2], height=mosaic.shape[1],
                           transform=transform, compress="deflate")
            for ds in datasets:
                ds.close()
            with rasterio.open(str(raw_path), "w", **profile) as dst:
                dst.write(mosaic[0], 1)

        # Clip to AOI bbox
        try:
            with rasterio.open(str(raw_path)) as src:
                window = window_from_bounds(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    transform=src.transform,
                )
                # Ensure window is within raster bounds
                window = window.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height)
                )
                data = src.read(1, window=window)
                transform = src.window_transform(window)

                profile = src.profile.copy()
                profile.update(
                    width=data.shape[1],
                    height=data.shape[0],
                    transform=transform,
                    compress="deflate",
                )
                with rasterio.open(str(jrc_path), "w", **profile) as dst:
                    dst.write(data, 1)

            logger.info(f"Clipped JRC water occurrence -> {jrc_path}")
            return str(jrc_path)
        except Exception as exc:
            logger.warning(f"Failed to clip JRC water occurrence: {exc}")
            # Fall back: just copy the raw file
            try:
                shutil.copy2(str(raw_path), str(jrc_path))
                logger.info(f"Saved raw JRC tile (unclipped) -> {jrc_path}")
                return str(jrc_path)
            except Exception:
                return ""


# ---------------------------------------------------------------------------
# 4. WorldPop population density
# ---------------------------------------------------------------------------

_WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/"
    "Global_2000_2020_Constrained/2020/BSGM/BGD/bgd_ppp_2020_constrained.tif"
)


def download_worldpop(cfg: dict, output_dir: Path) -> str:
    """Download WorldPop constrained population density for Bangladesh.

    Returns path to the downloaded file.
    """
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    pop_path = output_dir / "worldpop_popdens.tif"

    if pop_path.exists():
        logger.info(f"WorldPop data already exists at {pop_path}, skipping.")
        return str(pop_path)

    # One national raster serves every region, so cache it centrally and clip
    # a regional copy from it.
    cached = _shared_cache("worldpop", _WORLDPOP_URL.rsplit("/", 1)[-1])
    if not cached.exists():
        if not _download_file(_WORLDPOP_URL, cached,
                              "WorldPop population density"):
            cached.unlink(missing_ok=True)
            return ""
    else:
        logger.info(f"Using cached WorldPop raster: {cached}")

    bbox = cfg.get("aoi", {}).get("bbox")
    try:
        import rasterio
        from rasterio.windows import from_bounds as window_from_bounds

        with rasterio.open(cached) as src:
            window = window_from_bounds(*bbox, transform=src.transform)
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            data = src.read(1, window=window)
            profile = src.profile.copy()
            profile.update(width=data.shape[1], height=data.shape[0],
                           transform=src.window_transform(window),
                           compress="deflate")
        with rasterio.open(pop_path, "w", **profile) as dst:
            dst.write(data, 1)
        logger.info(f"WorldPop clipped to the study area → {pop_path}")
    except Exception as exc:
        logger.warning(f"Could not clip WorldPop ({exc}); copying the full raster.")
        shutil.copy(cached, pop_path)

    return str(pop_path)


# ---------------------------------------------------------------------------
# 5. GloFAS flood extent (Copernicus CDS API)
# ---------------------------------------------------------------------------

def download_glofas_flood_extent(cfg: dict, output_dir: Path) -> str:
    """Download GloFAS river flood extent from Copernicus Climate Data Store.

    Requires a CDS API key configured at ~/.cdsapirc or via env vars:
        CDSAPI_URL  (default: https://cds.climate.copernicus.eu/api)
        CDSAPI_KEY  (your UID:API-KEY from https://cds.climate.copernicus.eu/user)

    The dataset used is 'cems-glofas-historical' which provides global river
    flood hazard maps at ~0.05° (~5 km) resolution.

    Returns path to the clipped GeoTIFF, or "" on failure.
    """
    import os

    output_dir.mkdir(parents=True, exist_ok=True)
    glofas_path = output_dir / "glofas_flood_extent.tif"

    if glofas_path.exists():
        logger.info(f"GloFAS flood extent already exists at {glofas_path}, skipping.")
        return str(glofas_path)

    # Check for CDS API credentials
    cdsapirc = Path.home() / ".cdsapirc"
    has_env = os.environ.get("CDSAPI_KEY")
    if not cdsapirc.exists() and not has_env:
        logger.warning(
            "GloFAS download requires Copernicus EWDS API credentials.\n"
            "  1. Register at https://ewds.climate.copernicus.eu/\n"
            "  2. Get your personal access token from your EWDS profile\n"
            "  3. Create ~/.cdsapirc with:\n"
            "       url: https://ewds.climate.copernicus.eu/api\n"
            "       key: <YOUR_PERSONAL_ACCESS_TOKEN>\n"
            "  Or set CDSAPI_URL and CDSAPI_KEY environment variables.\n"
            "Skipping GloFAS download."
        )
        return ""

    try:
        import cdsapi
    except ImportError:
        logger.warning(
            "GloFAS download requires the 'cdsapi' package.\n"
            "  Install with: pip install cdsapi\n"
            "Skipping GloFAS download."
        )
        return ""

    bbox = cfg.get("aoi", {}).get("bbox", [88.0, 24.0, 89.9, 26.7])
    # CDS API expects [north, west, south, east]
    area = [bbox[3], bbox[0], bbox[1], bbox[2]]

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_nc = Path(tmpdir) / "glofas_raw.grib"

        try:
            c = cdsapi.Client(
                url=os.environ.get(
                    "CDSAPI_URL", "https://ewds.climate.copernicus.eu/api"
                ),
            )
            c.retrieve(
                "cems-glofas-historical",
                {
                    "system_version": ["version_2_1"],
                    "hydrological_model": ["htessel_lisflood"],
                    "product_type": ["consolidated"],
                    "variable": ["mean_discharge_in_the_last_24_hours"],
                    "hyear": ["2020"],
                    "hmonth": ["july", "august", "september"],
                    "hday": [f"{d:02d}" for d in range(1, 32)],
                    "area": area,
                    "data_format": "grib",
                },
                str(raw_nc),
            )
            logger.info("GloFAS data downloaded from CDS API.")
        except Exception as exc:
            logger.warning(f"Failed to download GloFAS from CDS API: {exc}")
            return ""

        # Convert GRIB → GeoTIFF with max discharge as flood proxy
        try:
            import rasterio
            import xarray as xr

            ds = xr.open_dataset(str(raw_nc), engine="cfgrib")
            # Take max discharge over time as flood extent proxy
            var_name = list(ds.data_vars)[0]
            max_discharge = ds[var_name].max(dim="time")

            # Threshold: flag cells where max discharge exceeds 2yr return
            # period (~top 50th percentile) as flood-prone
            threshold = float(max_discharge.quantile(0.5))
            flood_mask = (max_discharge >= threshold).astype("float32")

            # Save as GeoTIFF
            lats = max_discharge.latitude.values
            lons = max_discharge.longitude.values
            from rasterio.transform import from_bounds

            transform = from_bounds(
                lons.min(), lats.min(), lons.max(), lats.max(),
                len(lons), len(lats),
            )
            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "width": len(lons),
                "height": len(lats),
                "count": 1,
                "crs": "EPSG:4326",
                "transform": transform,
                "compress": "deflate",
                "nodata": -9999,
            }
            with rasterio.open(str(glofas_path), "w", **profile) as dst:
                dst.write(flood_mask.values[np.newaxis, :, :])

            logger.info(f"GloFAS flood extent -> {glofas_path}")
            return str(glofas_path)
        except Exception as exc:
            logger.warning(f"Failed to process GloFAS data: {exc}")
            return ""


# ---------------------------------------------------------------------------
# 6. Sentinel-1 SAR flood extent (Copernicus Dataspace / ASF)
# ---------------------------------------------------------------------------

def download_sentinel1_flood_extent(cfg: dict, output_dir: Path) -> str:
    """Download and process Sentinel-1 SAR imagery to derive flood extent.

    Supports two backends (tried in order):
      1. Copernicus Dataspace Ecosystem (CDSE) via OData API
         Requires env var: CDSE_TOKEN (OAuth access token)
         Register at https://dataspace.copernicus.eu/
      2. ASF (Alaska Satellite Facility) via asf_search
         Requires env vars: EARTHDATA_USER, EARTHDATA_PASS
         Register at https://urs.earthdata.nasa.gov/users/new

    Uses a simple Otsu threshold on VH backscatter to delineate flood extent.

    Returns path to the binary flood extent GeoTIFF, or "" on failure.
    """
    import os

    output_dir.mkdir(parents=True, exist_ok=True)
    sar_path = output_dir / "sentinel1_flood_extent.tif"

    if sar_path.exists():
        logger.info(f"Sentinel-1 flood extent already exists at {sar_path}, skipping.")
        return str(sar_path)

    bbox = cfg.get("aoi", {}).get("bbox", [88.0, 24.0, 89.9, 26.7])

    # --- Try Backend 1: Copernicus Dataspace (CDSE) OData API ---
    cdse_token = os.environ.get("CDSE_TOKEN")
    if cdse_token:
        result = _download_s1_cdse(cfg, output_dir, sar_path, bbox, cdse_token)
        if result:
            return result

    # --- Try Backend 2: ASF via asf_search ---
    earthdata_user = os.environ.get("EARTHDATA_USER")
    earthdata_pass = os.environ.get("EARTHDATA_PASS")
    if earthdata_user and earthdata_pass:
        result = _download_s1_asf(
            cfg, output_dir, sar_path, bbox, earthdata_user, earthdata_pass
        )
        if result:
            return result

    logger.warning(
        "Sentinel-1 SAR download requires credentials for at least one backend:\n"
        "  Option A — Copernicus Dataspace:\n"
        "    1. Register at https://dataspace.copernicus.eu/\n"
        "    2. Set env var CDSE_TOKEN=<your_oauth_access_token>\n"
        "  Option B — ASF / NASA Earthdata:\n"
        "    1. Register at https://urs.earthdata.nasa.gov/users/new\n"
        "    2. Set env vars EARTHDATA_USER and EARTHDATA_PASS\n"
        "Skipping Sentinel-1 download."
    )
    return ""


def _download_s1_cdse(
    cfg: dict, output_dir: Path, sar_path: Path,
    bbox: list, token: str,
) -> str:
    """Download Sentinel-1 GRD product via Copernicus Dataspace OData API."""
    west, south, east, north = bbox

    # Search for IW GRD VH products during monsoon season
    search_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        "$filter=Collection/Name eq 'SENTINEL-1' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
        f"{west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))') "
        "and ContentDate/Start ge 2020-07-01T00:00:00.000Z "
        "and ContentDate/Start le 2020-09-30T23:59:59.999Z "
        "and Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'GRD') "
        "&$orderby=ContentDate/Start desc&$top=1"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(search_url, headers=headers, timeout=60)
        resp.raise_for_status()
        results = resp.json().get("value", [])
        if not results:
            logger.warning("No Sentinel-1 products found on Copernicus Dataspace.")
            return ""

        product_id = results[0]["Id"]
        product_name = results[0].get("Name", product_id)
        logger.info(f"Found Sentinel-1 product: {product_name}")

        # Download the product
        dl_url = (
            f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "s1_product.zip"
            ok = _download_file(dl_url, zip_path, f"Sentinel-1 {product_name}")
            if not ok:
                # Retry with auth header
                resp = requests.get(
                    dl_url, headers=headers, stream=True, timeout=600
                )
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)

            return _process_s1_to_flood_mask(zip_path, sar_path, bbox)

    except Exception as exc:
        logger.warning(f"CDSE Sentinel-1 download failed: {exc}")
        return ""


def _download_s1_asf(
    cfg: dict, output_dir: Path, sar_path: Path,
    bbox: list, user: str, password: str,
) -> str:
    """Download Sentinel-1 GRD product via ASF (Alaska Satellite Facility)."""
    try:
        import asf_search as asf
    except ImportError:
        logger.warning(
            "ASF backend requires 'asf_search' package.\n"
            "  Install with: pip install asf_search\n"
        )
        return ""

    west, south, east, north = bbox

    try:
        results = asf.geo_search(
            platform=[asf.PLATFORM.SENTINEL1],
            intersectsWith=f"POLYGON(({west} {south},{east} {south},"
                           f"{east} {north},{west} {north},{west} {south}))",
            processingLevel=asf.PRODUCT_TYPE.GRD_HD,
            start="2020-07-01",
            end="2020-09-30",
            maxResults=1,
        )
        if not results:
            logger.warning("No Sentinel-1 products found on ASF.")
            return ""

        product = results[0]
        logger.info(f"Found Sentinel-1 product: {product.properties['fileName']}")

        with tempfile.TemporaryDirectory() as tmpdir:
            session = asf.ASFSession().auth_with_creds(user, password)
            product.download(path=tmpdir, session=session)

            # Find the downloaded zip
            zips = list(Path(tmpdir).glob("*.zip"))
            if not zips:
                logger.warning("ASF download produced no zip file.")
                return ""

            return _process_s1_to_flood_mask(zips[0], sar_path, bbox)

    except Exception as exc:
        logger.warning(f"ASF Sentinel-1 download failed: {exc}")
        return ""


def _process_s1_to_flood_mask(
    zip_path: Path, output_path: Path, bbox: list
) -> str:
    """Extract VH band from Sentinel-1 GRD ZIP and derive flood mask via Otsu.

    Returns path to flood extent GeoTIFF or "" on failure.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find VH measurement TIFF inside the SAFE structure
            vh_files = [
                n for n in zf.namelist()
                if "measurement" in n.lower()
                and n.lower().endswith(".tiff")
                and "vh" in n.lower()
            ]
            if not vh_files:
                # Fallback: any .tiff in measurement/
                vh_files = [
                    n for n in zf.namelist()
                    if "measurement" in n.lower() and n.lower().endswith(".tiff")
                ]
            if not vh_files:
                logger.warning("No VH measurement TIFF found in Sentinel-1 product.")
                return ""

            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extract(vh_files[0], tmpdir)
                vh_path = Path(tmpdir) / vh_files[0]

                with rasterio.open(str(vh_path)) as src:
                    # Clip to AOI
                    west, south, east, north = bbox
                    try:
                        window = window_from_bounds(
                            west, south, east, north, transform=src.transform
                        )
                        window = window.intersection(
                            rasterio.windows.Window(0, 0, src.width, src.height)
                        )
                        data = src.read(1, window=window).astype(np.float32)
                        transform = src.window_transform(window)
                    except Exception:
                        # If AOI window fails, read full raster
                        data = src.read(1).astype(np.float32)
                        transform = src.transform

                # Convert to dB if raw amplitude
                valid = data[data > 0]
                if len(valid) == 0:
                    logger.warning("Sentinel-1 VH band has no valid pixels.")
                    return ""

                db = np.full_like(data, np.nan)
                mask = data > 0
                db[mask] = 10.0 * np.log10(data[mask])

                # Otsu thresholding on dB values to separate water/non-water
                flood_mask = _otsu_threshold(db[mask])
                result = np.zeros_like(data, dtype=np.float32)
                result[mask] = flood_mask

                profile = {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "width": data.shape[1],
                    "height": data.shape[0],
                    "count": 1,
                    "crs": "EPSG:4326",
                    "transform": transform,
                    "compress": "deflate",
                    "nodata": -9999,
                }
                with rasterio.open(str(output_path), "w", **profile) as dst:
                    dst.write(result[np.newaxis, :, :])

                logger.info(f"Sentinel-1 flood mask -> {output_path}")
                return str(output_path)

    except Exception as exc:
        logger.warning(f"Failed to process Sentinel-1 product: {exc}")
        return ""


def _otsu_threshold(values: np.ndarray) -> np.ndarray:
    """Simple Otsu thresholding: returns binary array (1=water/flood, 0=land).

    Lower dB values in VH polarization correspond to smoother surfaces (water).
    """
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0:
        return np.zeros_like(values)

    best_thresh = sorted_vals[0]
    best_var = np.inf

    # Test ~256 candidate thresholds for efficiency
    step = max(1, n // 256)
    for i in range(0, n, step):
        t = sorted_vals[i]
        w0 = np.sum(values <= t)
        w1 = n - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = np.mean(values[values <= t])
        m1 = np.mean(values[values > t])
        var = w0 * w1 * (m0 - m1) ** 2
        if var < best_var:
            best_var = var
            best_thresh = t

    # Water = low backscatter (below threshold)
    return (values <= best_thresh).astype(np.float32)


# ---------------------------------------------------------------------------
# Download all
# ---------------------------------------------------------------------------

def download_all(cfg: dict, output_dir: Path) -> dict:
    """Download all required datasets. Returns dict of dataset -> path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    logger.info("--- Downloading GADM admin boundaries ---")
    for level, path in download_admin_boundaries(cfg, output_dir).items():
        results[f"admin_{level}"] = path

    logger.info("--- Downloading SRTM DEM ---")
    results["srtm_dem"] = download_srtm_dem(cfg, output_dir)

    logger.info("--- Downloading JRC Global Surface Water ---")
    results["jrc_water"] = download_jrc_water(cfg, output_dir)

    logger.info("--- Downloading WorldPop population density ---")
    results["worldpop"] = download_worldpop(cfg, output_dir)

    # Optional: GloFAS (requires CDS API key)
    proxy_cfg = cfg.get("data", {}).get("proxy_labels", {})
    if proxy_cfg.get("glofas_return_period"):
        logger.info("--- Downloading GloFAS flood extent ---")
        results["glofas"] = download_glofas_flood_extent(cfg, output_dir)

    # Optional: Sentinel-1 SAR (requires CDSE or Earthdata credentials)
    if proxy_cfg.get("sentinel1_sar"):
        logger.info("--- Downloading Sentinel-1 SAR flood extent ---")
        results["sentinel1"] = download_sentinel1_flood_extent(cfg, output_dir)

    # Summary
    succeeded = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info(f"Download complete: {succeeded}/{total} datasets acquired.")
    return results
