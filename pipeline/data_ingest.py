"""
Step 1 — Data Ingestion: OSM infrastructure, DEM, proxy flood labels, vulnerability layers.
Step 2 — Preprocessing: CRS alignment, clipping, DEM derivatives.
"""

import logging
import math
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


# The main Overpass endpoint allows two connections per address and refuses
# the rest outright, so failed queries are retried with a growing pause
# rather than sent elsewhere: the public mirrors tested in September 2026
# either hung for the full request timeout or refused connections, and a
# hanging mirror costs more than waiting for a slot. Extra endpoints can be
# listed in data.osm.overpass_endpoints for a deployment that has a reliable
# one.
OVERPASS_ENDPOINTS = ["https://overpass-api.de/api"]
RETRY_PAUSES_S = (15, 30, 60, 120)


TAG_BATCHES = [
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
            "bridge": "yes",
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


def _bbox_tiles(bbox, max_span_deg: float = 0.75):
    """Split a bbox into tiles small enough for Overpass to answer.

    A whole division in one query either times out or is throttled; tiles of
    under a degree come back reliably and can be retried individually.
    """
    west, south, east, north = bbox
    n_lon = max(1, math.ceil((east - west) / max_span_deg))
    n_lat = max(1, math.ceil((north - south) / max_span_deg))
    lon_edges = np.linspace(west, east, n_lon + 1)
    lat_edges = np.linspace(south, north, n_lat + 1)
    return [
        (lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1])
        for i in range(n_lon) for j in range(n_lat)
    ]


def merge_tile_results(frames: list, bbox) -> gpd.GeoDataFrame:
    """Combine per-tile Overpass results into one clipped, de-duplicated frame.

    Tiles overshoot the bounding box slightly, and a long way crossing a tile
    edge is returned by both tiles, so results are clipped to the study area
    and duplicates removed by OSM element type and id. Without the ids (older
    responses) the geometry itself is the key.
    """
    import pandas as pd
    from shapely.geometry import box as shapely_box

    infra = gpd.GeoDataFrame(
        data=pd.concat(frames, ignore_index=True), crs="EPSG:4326",
    )

    before = len(infra)
    infra = infra[infra.geometry.notna() & ~infra.geometry.is_empty]
    infra = infra[infra.intersects(shapely_box(*bbox))]
    logger.info(f"Clipped to the study area: {len(infra)} of {before} features")

    id_cols = [c for c in ("element", "id") if c in infra.columns]
    before = len(infra)
    if id_cols:
        infra = infra.drop_duplicates(subset=id_cols)
    else:
        infra = infra[~infra.geometry.apply(lambda g: g.wkb_hex).duplicated()]
    if len(infra) < before:
        logger.info(f"Removed {before - len(infra)} features returned by more "
                    "than one tile")
    if "id" in infra.columns:
        infra = infra.rename(columns={"id": "osm_id", "element": "osm_type"})
    return infra


def fetch_osm_infrastructure(cfg: dict, output_dir: Path) -> gpd.GeoDataFrame:
    """Download OSM infrastructure for the study area.

    Queries run tile by tile over the configured bounding box, not by place
    name. The bounding box is what the rest of the pipeline uses — the grid,
    the kriging surface, the rasters — so fetching by division name pulled in
    assets far outside the study area (the CHT config covers the hill tracts,
    while "Chittagong Division" reaches the coast and Cox's Bazar) and made
    Overpass time out on the larger divisions.
    """
    import time

    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    ox.settings.timeout = 180                # server-side query budget
    ox.settings.requests_timeout = 90        # give up on a silent connection
    ox.settings.max_query_area_size = 25_000_000_000  # 25B sq m

    bbox = tuple(cfg["aoi"]["bbox"])
    tiles = _bbox_tiles(bbox, cfg["data"]["osm"].get("tile_span_deg", 0.75))
    region_name = cfg["aoi"].get("name", "region")
    logger.info(f"Fetching OSM features for {region_name} in {len(tiles)} tiles")

    endpoints = cfg["data"]["osm"].get("overpass_endpoints") or OVERPASS_ENDPOINTS

    all_gdfs, failures = [], []
    for i, tile in enumerate(tiles, start=1):
        for batch_name, priority, tags in TAG_BATCHES:
            # osmnx 2.x takes (left, bottom, right, top). Each attempt cycles
            # through the endpoints; pauses grow between rounds.
            attempts = [(pause, endpoint)
                        for pause in (0,) + RETRY_PAUSES_S
                        for endpoint in endpoints]
            for attempt, (pause, endpoint) in enumerate(attempts):
                if pause:
                    time.sleep(pause)
                ox.settings.overpass_url = endpoint
                try:
                    gdf = ox.features_from_bbox(bbox=tile, tags=tags)
                    if len(gdf):
                        # osmnx indexes features by (element, id); keep both
                        # as columns so features returned by two adjacent
                        # tiles can be recognised and so the OSM id travels
                        # with the asset for provenance.
                        gdf = gdf.reset_index()
                        gdf["priority"] = priority
                        gdf["source_tag"] = gdf.apply(
                            lambda row: _detect_source_tag(row, tags), axis=1
                        )
                        all_gdfs.append(gdf)
                    logger.info(f"  tile {i}/{len(tiles)} {batch_name}: "
                                f"{len(gdf)} features")
                    break
                except ox._errors.InsufficientResponseError:
                    # Nothing of this kind in this tile — not an error.
                    logger.info(f"  tile {i}/{len(tiles)} {batch_name}: none")
                    break
                except Exception as exc:
                    host = endpoint.split("//")[-1].split("/")[0]
                    logger.warning(f"  tile {i}/{len(tiles)} {batch_name} via "
                                   f"{host} failed: {type(exc).__name__}")
                    if attempt == len(attempts) - 1:
                        failures.append((tile, batch_name, str(exc)))
            time.sleep(2)  # be polite between queries

    if not all_gdfs:
        raise RuntimeError(
            "No OSM features fetched — check the connection and Overpass status."
        )
    if failures:
        logger.warning(f"{len(failures)} tile/tag queries failed after retries; "
                       "coverage may be incomplete")

    infra = merge_tile_results(all_gdfs, bbox)

    infra["asset_type"] = infra["source_tag"].apply(_classify_asset)

    if "name" not in infra.columns:
        infra["name"] = "unnamed"
    infra["name"] = infra["name"].fillna("unnamed")

    # Label each asset with the district it falls in, replacing the old
    # division label that came from the query itself.
    infra["division"] = region_name
    district_path = cfg["data"].get("admin_boundaries", {}).get("district", "")
    if district_path and Path(district_path).exists():
        try:
            districts = gpd.read_file(district_path)[["admin_name", "geometry"]]
            points = infra[["geometry"]].copy()
            points["geometry"] = points.geometry.representative_point()
            joined = gpd.sjoin(points, districts, how="left", predicate="within")
            joined = joined[~joined.index.duplicated(keep="first")]
            infra["division"] = (
                joined["admin_name"].reindex(infra.index).fillna(region_name).values
            )
            logger.info("Labelled assets with their district")
        except Exception as exc:
            logger.warning(f"Could not label assets by district: {exc}")

    # Deduplicate column names (OSM has case variants like damage_per /
    # damage_Per which collide in case-insensitive GPKG).
    seen, drop_cols = {}, []
    for col in infra.columns:
        lower = col.lower()
        if lower in seen:
            drop_cols.append(col)
        else:
            seen[lower] = col
    if drop_cols:
        infra = infra.drop(columns=drop_cols)

    keep_cols = [
        "geometry", "osm_type", "osm_id", "name", "asset_type", "source_tag",
        "priority", "division",
        "amenity", "highway", "bridge", "railway", "waterway", "landuse",
        "man_made", "building",
    ]
    infra = infra[[c for c in keep_cols if c in infra.columns]].copy()

    out_path = output_dir / "infrastructure_raw.gpkg"
    infra.to_file(out_path, driver="GPKG")
    logger.info(f"Saved {len(infra)} infrastructure features → {out_path}")
    logger.info(f"Asset types: {infra['asset_type'].value_counts().to_dict()}")
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


def _d8_receivers(dem: np.ndarray, nodata: float) -> np.ndarray:
    """Index of each cell's steepest-descent neighbour, or -1 where none.

    Vectorised over the eight neighbour directions: comparing eight shifted
    copies of the array costs eight passes, where the per-cell Python loop it
    replaces cost eight operations per cell.
    """
    rows, cols = dem.shape
    valid = dem != nodata

    best_drop = np.zeros_like(dem, dtype=np.float32)
    receiver = np.full(dem.shape, -1, dtype=np.int64)
    flat_index = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)

    for dr, dc in ((-1, 0), (-1, 1), (0, 1), (1, 1),
                   (1, 0), (1, -1), (0, -1), (-1, -1)):
        shifted = np.full_like(dem, nodata)
        shifted_idx = np.full(dem.shape, -1, dtype=np.int64)

        src_rows = slice(max(0, dr), rows + min(0, dr))
        src_cols = slice(max(0, dc), cols + min(0, dc))
        dst_rows = slice(max(0, -dr), rows + min(0, -dr))
        dst_cols = slice(max(0, -dc), cols + min(0, -dc))

        shifted[dst_rows, dst_cols] = dem[src_rows, src_cols]
        shifted_idx[dst_rows, dst_cols] = flat_index[src_rows, src_cols]

        drop = dem - shifted
        better = valid & (shifted != nodata) & (drop > best_drop)
        best_drop = np.where(better, drop, best_drop)
        receiver = np.where(better, shifted_idx, receiver)

    receiver[~valid] = -1
    return receiver


def _accumulate_flow(receiver: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Push one unit of flow from every cell downslope, in elevation order.

    Sequential by nature — a cell's total depends on everything upstream —
    so it is JIT-compiled when numba is available and falls back to plain
    Python otherwise.
    """
    acc = np.zeros(receiver.shape[0], dtype=np.float64)
    for i in order:
        target = receiver[i]
        if target >= 0:
            acc[target] += acc[i] + 1.0
    return acc


try:  # numba turns the accumulation from tens of minutes into seconds
    from numba import njit

    _accumulate_flow = njit(cache=True)(_accumulate_flow)
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_NUMBA = False
    logger.info("numba not installed; flow accumulation will be slower")


def _simple_flow_accumulation(dem: np.ndarray, nodata: float) -> np.ndarray:
    """D8 flow accumulation: cells drained through each cell."""
    receiver = _d8_receivers(dem, nodata)

    valid = (dem != nodata).ravel()
    flat_dem = dem.ravel()
    # Descending elevation: every cell is processed after everything that
    # drains into it.
    order = np.argsort(-flat_dem)
    order = order[valid[order]].astype(np.int64)

    logger.info(f"Flow accumulation over {order.size:,} cells "
                f"({'numba' if _HAS_NUMBA else 'pure Python'})")
    acc = _accumulate_flow(receiver.ravel(), order)
    return acc.reshape(dem.shape)


def _compute_hand(dem: np.ndarray, drainage_mask: np.ndarray,
                   nodata: float) -> np.ndarray:
    """Height Above Nearest Drainage.

    The distance transform already returns, for every cell, the indices of
    the nearest drainage cell; looking up those elevations is a single fancy
    index rather than a per-cell Python loop over tens of millions of cells.
    """
    from scipy.ndimage import distance_transform_edt

    if not drainage_mask.any():
        logger.warning("No drainage cells found; HAND is zero everywhere.")
        hand = np.zeros_like(dem, dtype=np.float64)
        hand[dem == nodata] = -9999.0
        return hand

    # Distances are measured to the nearest zero of the input, so invert the
    # mask to find the nearest drainage cell.
    _, indices = distance_transform_edt(
        ~drainage_mask, return_distances=True, return_indices=True
    )
    nearest_elevation = dem[indices[0], indices[1]]

    hand = np.maximum(0.0, dem.astype(np.float64) - nearest_elevation)
    hand[dem == nodata] = -9999.0
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


def build_observed_flood_labels(cfg: dict, dem_derivatives: dict,
                                output_dir: Path,
                                raw_dir: Path = Path("data/raw")) -> str:
    """Training labels from Sentinel-1 observed flood extents.

    Preferred over the proxy ensemble: the proxy thresholds TWI and HAND,
    which the model also receives as inputs, so a model trained on it can
    only relearn the threshold rule. Observed water is independent evidence.

    A pixel is labelled flooded when it was under water in at least
    `data.labels.min_events` of the mapped events.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_cfg = cfg["data"].get("labels", {})

    freq_path = raw_dir / "s1_flood_frequency.tif"
    if not freq_path.exists():
        raise FileNotFoundError(
            f"{freq_path} missing. Run `python -m pipeline.cli sentinel1` first."
        )

    # Reference grid: the DEM derivatives, so labels align with the features.
    with rasterio.open(dem_derivatives["twi"]) as src:
        ref_shape, ref_transform, ref_crs = src.shape, src.transform, src.crs
        meta = src.meta.copy()

    # Stored as a percentage of events (0-100) to keep the download small.
    freq_pct = _resample_to_ref(str(freq_path), ref_shape, ref_transform, ref_crs)

    n_events = len(cfg.get("sentinel1", {}).get("events", [])) or 1
    min_events = labels_cfg.get("min_events", 1)
    threshold_pct = 100.0 * min_events / n_events - 1e-6

    labels = (freq_pct >= threshold_pct).astype(np.float32)
    positive_rate = float(labels.mean())
    logger.info(
        f"Observed flood labels: {positive_rate:.2%} of pixels flooded in "
        f">= {min_events} of {n_events} events"
    )
    if positive_rate == 0:
        raise RuntimeError(
            "No pixels flagged as flooded — check the Sentinel-1 event windows."
        )

    meta.update(dtype="float32", count=1, nodata=-9999)
    out_path = str(output_dir / "flood_observed_labels.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(labels, 1)
    logger.info(f"Observed flood labels → {out_path}")
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

    # Build training labels. "observed" uses Sentinel-1 flood extents;
    # "proxy" keeps the terrain-threshold ensemble for comparison.
    if "derivatives" in outputs:
        source = cfg["data"].get("labels", {}).get("source", "proxy")
        if source == "observed":
            outputs["flood_labels"] = build_observed_flood_labels(
                cfg, outputs["derivatives"], processed_dir, raw_dir
            )
        else:
            outputs["flood_labels"] = build_ensemble_flood_labels(
                cfg, outputs["derivatives"], processed_dir
            )
        # The proxy ensemble is always written too: the paper compares the
        # two label sets against each other.
        if source == "observed":
            try:
                outputs["proxy_labels"] = build_ensemble_flood_labels(
                    cfg, outputs["derivatives"], processed_dir
                )
            except Exception as exc:
                logger.warning(f"Proxy label comparison unavailable: {exc}")

    return outputs
