"""
Automatic data download functions for the SGMDI pipeline.

Downloads GADM admin boundaries, SRTM DEM, JRC Global Surface Water,
and WorldPop population density data required by the pipeline.
"""

import gzip
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

_GADM_URLS = {
    "union": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_3.json.zip",
    "upazila": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json.zip",
}


def download_gadm_boundaries(cfg: dict, output_dir: Path) -> tuple:
    """Download GADM Level 2 (upazila) and Level 3 (union) boundaries for
    Rangpur & Rajshahi divisions and save as shapefiles.

    Returns (union_path, upazila_path) as strings.
    """
    import geopandas as gpd

    output_dir.mkdir(parents=True, exist_ok=True)
    divisions = cfg.get("aoi", {}).get("divisions", ["Rangpur", "Rajshahi"])

    union_out = output_dir / "gadm_union.shp"
    upazila_out = output_dir / "gadm_upazila.shp"

    # If both already exist, skip
    if union_out.exists() and upazila_out.exists():
        logger.info("GADM boundaries already exist, skipping download.")
        return str(union_out), str(upazila_out)

    results = {}
    for level_name, url in _GADM_URLS.items():
        out_path = union_out if level_name == "union" else upazila_out
        if out_path.exists():
            logger.info(f"GADM {level_name} already exists at {out_path}, skipping.")
            results[level_name] = str(out_path)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / f"gadm_{level_name}.zip"
            ok = _download_file(url, zip_path, f"GADM {level_name}")
            if not ok:
                results[level_name] = ""
                continue

            # Extract ZIP -> read GeoJSON
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    json_names = [n for n in zf.namelist() if n.endswith(".json")]
                    if not json_names:
                        logger.warning(f"No JSON found in GADM {level_name} ZIP.")
                        results[level_name] = ""
                        continue
                    zf.extract(json_names[0], tmpdir)
                    json_path = Path(tmpdir) / json_names[0]

                gdf = gpd.read_file(str(json_path))

                # Filter to target divisions using NAME_1
                if "NAME_1" in gdf.columns:
                    gdf = gdf[gdf["NAME_1"].isin(divisions)].copy()
                    logger.info(
                        f"Filtered GADM {level_name} to {len(gdf)} features "
                        f"in divisions: {divisions}"
                    )
                else:
                    logger.warning(
                        f"NAME_1 column not found in GADM {level_name}; "
                        "saving all features."
                    )

                gdf.to_file(str(out_path))
                logger.info(f"Saved GADM {level_name} -> {out_path}")
                results[level_name] = str(out_path)
            except Exception as exc:
                logger.warning(f"Error processing GADM {level_name}: {exc}")
                results[level_name] = ""

    return results.get("union", ""), results.get("upazila", "")


# ---------------------------------------------------------------------------
# 2. SRTM 30m DEM
# ---------------------------------------------------------------------------

# Tiles covering bbox [88.0, 24.0, 89.9, 26.7]
_SRTM_TILES = [
    (24, 88), (24, 89),
    (25, 88), (25, 89),
    (26, 88), (26, 89),
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
        for lat, lon in _SRTM_TILES:
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

_JRC_URL = (
    "https://storage.googleapis.com/global-surface-water/downloads2021/"
    "occurrence/occurrence_80E_30Nv1_4_2021.tif"
)


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

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "jrc_raw.tif"
        ok = _download_file(_JRC_URL, raw_path, "JRC Global Surface Water")
        if not ok:
            return ""

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
    output_dir.mkdir(parents=True, exist_ok=True)
    pop_path = output_dir / "worldpop_popdens.tif"

    if pop_path.exists():
        logger.info(f"WorldPop data already exists at {pop_path}, skipping.")
        return str(pop_path)

    ok = _download_file(_WORLDPOP_URL, pop_path, "WorldPop population density")
    if ok:
        return str(pop_path)
    return ""


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
    union_path, upazila_path = download_gadm_boundaries(cfg, output_dir)
    results["gadm_union"] = union_path
    results["gadm_upazila"] = upazila_path

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
