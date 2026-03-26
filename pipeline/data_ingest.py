"""
Step 1 — Data Ingestion: OSM infrastructure, DEM, proxy flood labels, vulnerability layers.
Step 2 — Preprocessing: CRS alignment, clipping, DEM derivatives.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS
from shapely.geometry import box
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Data Ingestion
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_osm_infrastructure(cfg: dict, output_dir: Path) -> gpd.GeoDataFrame:
    """Download OSM infrastructure for Rangpur & Rajshahi divisions.

    Strategy: query by place name (per division) with batched tags to avoid
    Overpass timeouts on huge bounding boxes.
    """
    import time
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure osmnx for longer timeouts and smaller subdivisions
    ox.settings.timeout = 300
    ox.settings.max_query_area_size = 25_000_000_000  # 25B sq m

    divisions = cfg["aoi"].get("divisions", ["Rangpur", "Rajshahi"])
    all_gdfs = []

    # Batch tags by priority into combined dicts (single Overpass query each)
    tag_batches = [
        (
            "lifeline",
            1,
            {
                "amenity": ["hospital", "clinic", "school", "college", "shelter"],
                "man_made": ["bridge", "embankment"],
                "waterway": "dam",
            },
        ),
        (
            "transport",
            1,
            {
                "highway": ["primary", "secondary", "tertiary", "trunk"],
                "railway": "rail",
            },
        ),
        (
            "agriculture",
            2,
            {
                "landuse": ["farmland", "aquaculture"],
                "waterway": ["canal", "ditch"],
                "amenity": "marketplace",
            },
        ),
    ]

    for division in divisions:
        place_name = f"{division} Division, Bangladesh"
        logger.info(f"Fetching OSM data for: {place_name}")

        for batch_name, priority, tags in tag_batches:
            try:
                logger.info(f"  Querying {batch_name} tags for {division}...")
                gdf = ox.features_from_place(place_name, tags=tags)
                gdf["priority"] = priority
                gdf["division"] = division

                # Tag each row with its primary source tag for classification
                gdf["source_tag"] = gdf.apply(
                    lambda row: _detect_source_tag(row, tags), axis=1
                )

                all_gdfs.append(gdf)
                logger.info(
                    f"  {batch_name}: {len(gdf)} features for {division}"
                )
            except Exception as exc:
                logger.warning(
                    f"  {batch_name} failed for {division}: {exc}"
                )

            # Small delay between queries to be polite to Overpass
            time.sleep(5)

    # Also try bridge=yes separately (it's a tag on ways, not amenity)
    for division in divisions:
        place_name = f"{division} Division, Bangladesh"
        try:
            logger.info(f"  Querying bridge=yes for {division}...")
            gdf = ox.features_from_place(place_name, tags={"bridge": "yes"})
            gdf["priority"] = 1
            gdf["division"] = division
            gdf["source_tag"] = "bridge=yes"
            all_gdfs.append(gdf)
            logger.info(f"  bridge=yes: {len(gdf)} features for {division}")
        except Exception as exc:
            logger.warning(f"  bridge=yes failed for {division}: {exc}")
        time.sleep(5)

    if not all_gdfs:
        raise RuntimeError(
            "No OSM features fetched. Check internet connection and try again."
        )

    infra = gpd.GeoDataFrame(
        data=pd.concat(all_gdfs, ignore_index=True),
        crs="EPSG:4326",
    )

    # Assign simplified asset_type
    infra["asset_type"] = infra["source_tag"].apply(_classify_asset)

    # Ensure name column exists
    if "name" not in infra.columns:
        infra["name"] = "unnamed"
    infra["name"] = infra["name"].fillna("unnamed")

    # Deduplicate column names (OSM has case variants like damage_per / damage_Per
    # which collide in case-insensitive SQLite/GPKG). Keep only the first occurrence.
    seen = {}
    drop_cols = []
    for col in infra.columns:
        lower = col.lower()
        if lower in seen:
            drop_cols.append(col)
            logger.warning(f"Dropping duplicate column '{col}' (conflicts with '{seen[lower]}')")
        else:
            seen[lower] = col
    if drop_cols:
        infra = infra.drop(columns=drop_cols)

    # Keep only essential columns to avoid fragmentation and GPKG issues
    keep_cols = [
        "geometry", "name", "asset_type", "source_tag", "priority", "division",
        "amenity", "highway", "bridge", "railway", "waterway", "landuse",
        "man_made", "building",
    ]
    keep_cols = [c for c in keep_cols if c in infra.columns]
    infra = infra[keep_cols].copy()

    out_path = output_dir / "infrastructure_raw.gpkg"
    infra.to_file(out_path, driver="GPKG")
    logger.info(f"Saved {len(infra)} infrastructure features → {out_path}")
    return infra


def _detect_source_tag(row, tags: dict) -> str:
    """Determine which OSM tag matched this feature."""
    for key, values in tags.items():
        if key in row.index and pd.notna(row.get(key)):
            val = row[key]
            if isinstance(values, list):
                if val in values:
                    return f"{key}={val}"
            elif isinstance(values, str):
                if val == values or values is True:
                    return f"{key}={val}"
            else:
                return f"{key}={val}"
    return "other"


def _parse_osm_tag(tag_str: str) -> tuple[str, str]:
    """Parse 'key=value' into (key, value). If no '=', treat as key=True."""
    if "=" in tag_str:
        k, v = tag_str.split("=", 1)
        return k, v
    return tag_str, True


def _classify_asset(tag_str: str) -> str:
    """Map OSM tag string to simplified asset category."""
    mapping = {
        "amenity=hospital": "hospital",
        "amenity=clinic": "hospital",
        "amenity=school": "school",
        "amenity=college": "school",
        "man_made=bridge": "bridge",
        "bridge=yes": "bridge",
        "amenity=shelter": "flood_shelter",
        "man_made=embankment": "embankment",
        "waterway=dam": "embankment",
        "highway=primary": "road",
        "highway=secondary": "road",
        "highway=tertiary": "road",
        "highway=trunk": "road",
        "railway=rail": "railway",
        "amenity=ferry_terminal": "ferry_ghat",
        "landuse=farmland": "cropland",
        "landuse=aquaculture": "fishpond",
        "waterway=canal": "irrigation",
        "waterway=ditch": "irrigation",
        "amenity=marketplace": "market",
    }
    return mapping.get(tag_str, "other")


# ---------------------------------------------------------------------------
# Step 2 — Preprocessing
# ---------------------------------------------------------------------------

def reproject_raster(src_path: str, dst_path: str, dst_crs: str) -> None:
    """Reproject a raster to target CRS."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    logger.info(f"Reprojected {src_path} → {dst_path} ({dst_crs})")


def clip_raster_to_aoi(raster_path: str, aoi_gdf: gpd.GeoDataFrame,
                        output_path: str) -> None:
    """Clip a raster to AOI polygon."""
    with rasterio.open(raster_path) as src:
        aoi_reprojected = aoi_gdf.to_crs(src.crs)
        geoms = [g.__geo_interface__ for g in aoi_reprojected.geometry]
        out_image, out_transform = rio_mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)
    logger.info(f"Clipped {raster_path} → {output_path}")


def compute_dem_derivatives(dem_path: str, output_dir: Path) -> dict[str, str]:
    """Compute slope, TWI, HAND, flow accumulation from DEM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        meta = src.meta.copy()
        transform = src.transform
        nodata = src.nodata or -9999

    meta.update(dtype="float32", nodata=-9999)

    # --- Slope (degrees) ---
    dy, dx = np.gradient(dem, transform[4], transform[0])
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope[dem == nodata] = -9999
    slope_path = str(output_dir / "slope.tif")
    _write_single_band(slope_path, slope.astype(np.float32), meta)
    outputs["slope"] = slope_path

    # --- TWI (Topographic Wetness Index) ---
    # Simplified: TWI = ln(contributing_area / tan(slope_rad))
    slope_rad = np.radians(slope)
    slope_rad[slope_rad < 0.001] = 0.001  # avoid division by zero
    # Approximate contributing area via flow accumulation proxy (simple D8)
    flow_acc = _simple_flow_accumulation(dem, nodata)
    cell_area = abs(transform[0] * transform[4])
    contributing_area = (flow_acc + 1) * cell_area
    twi = np.log(contributing_area / np.tan(slope_rad))
    twi[dem == nodata] = -9999
    twi_path = str(output_dir / "twi.tif")
    _write_single_band(twi_path, twi.astype(np.float32), meta)
    outputs["twi"] = twi_path

    # --- HAND (Height Above Nearest Drainage) ---
    # Simplified: cells with high flow accumulation are drainage; HAND = elev - drainage_elev
    drainage_threshold = np.percentile(flow_acc[flow_acc > 0], 90)
    drainage_mask = flow_acc >= drainage_threshold
    hand = _compute_hand(dem, drainage_mask, nodata)
    hand_path = str(output_dir / "hand.tif")
    _write_single_band(hand_path, hand.astype(np.float32), meta)
    outputs["hand"] = hand_path

    # --- Flow Accumulation ---
    fa_path = str(output_dir / "flow_accumulation.tif")
    flow_acc_out = flow_acc.astype(np.float32)
    flow_acc_out[dem == nodata] = -9999
    _write_single_band(fa_path, flow_acc_out, meta)
    outputs["flow_accumulation"] = fa_path

    logger.info(f"DEM derivatives computed → {output_dir}")
    return outputs


def _write_single_band(path: str, data: np.ndarray, meta: dict) -> None:
    meta_out = meta.copy()
    meta_out["count"] = 1
    with rasterio.open(path, "w", **meta_out) as dst:
        dst.write(data, 1)


def _simple_flow_accumulation(dem: np.ndarray, nodata: float) -> np.ndarray:
    """Simplified D8 flow accumulation (approximate)."""
    rows, cols = dem.shape
    flow_acc = np.zeros_like(dem, dtype=np.float64)
    # D8 direction offsets
    dr = [-1, -1, 0, 1, 1, 1, 0, -1]
    dc = [0, 1, 1, 1, 0, -1, -1, -1]

    # Sort cells by descending elevation
    valid = dem != nodata
    indices = np.argwhere(valid)
    elevations = dem[valid]
    order = np.argsort(-elevations)
    sorted_indices = indices[order]

    for r, c in sorted_indices:
        min_elev = dem[r, c]
        min_dir = -1
        for d in range(8):
            nr, nc = r + dr[d], c + dc[d]
            if 0 <= nr < rows and 0 <= nc < cols and dem[nr, nc] != nodata:
                if dem[nr, nc] < min_elev:
                    min_elev = dem[nr, nc]
                    min_dir = d
        if min_dir >= 0:
            nr, nc = r + dr[min_dir], c + dc[min_dir]
            flow_acc[nr, nc] += flow_acc[r, c] + 1

    return flow_acc


def _compute_hand(dem: np.ndarray, drainage_mask: np.ndarray,
                   nodata: float) -> np.ndarray:
    """Height Above Nearest Drainage — BFS from drainage cells."""
    from scipy.ndimage import distance_transform_edt, label

    rows, cols = dem.shape
    hand = np.full_like(dem, -9999, dtype=np.float64)

    # For each non-drainage cell, find nearest drainage cell elevation
    # Use distance transform to find nearest drainage
    inv_mask = ~drainage_mask
    dist, indices = distance_transform_edt(inv_mask, return_distances=True,
                                            return_indices=True)

    for r in range(rows):
        for c in range(cols):
            if dem[r, c] == nodata:
                continue
            nearest_r, nearest_c = indices[0, r, c], indices[1, r, c]
            if drainage_mask[nearest_r, nearest_c]:
                hand[r, c] = max(0, dem[r, c] - dem[nearest_r, nearest_c])
            else:
                hand[r, c] = 0  # cell is itself drainage or isolated

    return hand


def _resample_to_ref(src_path: str, ref_shape: tuple, ref_transform,
                      ref_crs) -> np.ndarray:
    """Read a raster and resample it to match the reference grid shape."""
    from rasterio.warp import reproject, Resampling

    with rasterio.open(src_path) as src:
        dst = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )
    return dst


def build_ensemble_flood_labels(cfg: dict, dem_derivatives: dict,
                                 output_dir: Path) -> str:
    """Create binary flood proxy labels via majority voting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_cfg = cfg["data"]["proxy_labels"]

    votes = []
    ref_shape = None
    ref_transform = None
    ref_crs = None

    # Source 1: DEM-derived (TWI + HAND) — also sets the reference grid
    if proxy_cfg.get("dem_flood_fill"):
        with rasterio.open(dem_derivatives["twi"]) as src:
            twi = src.read(1)
            meta = src.meta.copy()
            ref_shape = twi.shape
            ref_transform = src.transform
            ref_crs = src.crs
        with rasterio.open(dem_derivatives["hand"]) as src:
            hand = src.read(1)

        twi_thresh = proxy_cfg.get("twi_threshold", 8.0)
        hand_thresh = proxy_cfg.get("hand_threshold_m", 5.0)
        dem_label = ((twi > twi_thresh) & (hand < hand_thresh) &
                     (twi != -9999) & (hand != -9999)).astype(np.float32)
        votes.append(dem_label)
        logger.info("DEM flood-fill proxy label generated")

    # Source 2: JRC Global Surface Water
    jrc_path = output_dir.parent / "raw" / "jrc_water_occurrence.tif"
    if proxy_cfg.get("jrc_global_surface_water") and jrc_path.exists():
        if ref_shape is not None:
            jrc = _resample_to_ref(str(jrc_path), ref_shape, ref_transform, ref_crs)
        else:
            with rasterio.open(str(jrc_path)) as src:
                jrc = src.read(1)
                meta = src.meta.copy()
                ref_shape = jrc.shape
                ref_transform = src.transform
                ref_crs = src.crs
        occ_thresh = proxy_cfg.get("jrc_occurrence_pct", 25)
        jrc_label = (jrc > occ_thresh).astype(np.float32)
        votes.append(jrc_label)
        logger.info("JRC water occurrence proxy label generated")

    # Source 3: GloFAS return period
    glofas_path = output_dir.parent / "raw" / "glofas_flood_extent.tif"
    if proxy_cfg.get("glofas_return_period") and glofas_path.exists():
        if ref_shape is not None:
            glofas = _resample_to_ref(str(glofas_path), ref_shape, ref_transform, ref_crs)
        else:
            with rasterio.open(str(glofas_path)) as src:
                glofas = src.read(1)
                meta = src.meta.copy()
                ref_shape = glofas.shape
                ref_transform = src.transform
                ref_crs = src.crs
        glofas_label = (glofas > 0).astype(np.float32)
        votes.append(glofas_label)
        logger.info("GloFAS return-period proxy label generated")

    # Source 4: Sentinel-1 SAR
    sar_path = output_dir.parent / "raw" / "sentinel1_flood_extent.tif"
    if proxy_cfg.get("sentinel1_sar") and sar_path.exists():
        if ref_shape is not None:
            sar = _resample_to_ref(str(sar_path), ref_shape, ref_transform, ref_crs)
        else:
            with rasterio.open(str(sar_path)) as src:
                sar = src.read(1)
                meta = src.meta.copy()
                ref_shape = sar.shape
                ref_transform = src.transform
                ref_crs = src.crs
        sar_label = (sar > 0).astype(np.float32)
        votes.append(sar_label)
        logger.info("Sentinel-1 SAR proxy label generated")

    if not votes:
        raise RuntimeError("No proxy label sources available. Provide at least DEM.")

    # Majority voting
    vote_stack = np.stack(votes, axis=0)
    n_sources = len(votes)
    threshold = max(2, n_sources // 2 + 1)  # majority
    ensemble = (vote_stack.sum(axis=0) >= threshold).astype(np.float32)

    meta.update(dtype="float32", count=1, nodata=-9999)
    out_path = str(output_dir / "flood_proxy_labels.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(ensemble, 1)

    logger.info(f"Ensemble flood labels (majority {threshold}/{n_sources}) → {out_path}")
    return out_path


def preprocess_all(cfg: dict, raw_dir: Path, processed_dir: Path) -> dict:
    """Run full preprocessing pipeline."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    target_crs = cfg["aoi"]["crs"]

    outputs = {}

    # Reproject DEM
    dem_raw = cfg["data"]["dem"]["path"]
    dem_reproj = str(processed_dir / "dem_reprojected.tif")
    if Path(dem_raw).exists():
        reproject_raster(dem_raw, dem_reproj, target_crs)
        outputs["dem"] = dem_reproj

        # Compute derivatives
        deriv_dir = processed_dir / "dem_derivatives"
        outputs["derivatives"] = compute_dem_derivatives(dem_reproj, deriv_dir)
    else:
        logger.warning(f"DEM not found at {dem_raw}. Skipping DEM derivatives.")

    # Build ensemble flood labels
    if "derivatives" in outputs:
        label_path = build_ensemble_flood_labels(
            cfg, outputs["derivatives"], processed_dir
        )
        outputs["flood_labels"] = label_path

    return outputs
