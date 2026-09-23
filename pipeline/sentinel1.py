"""
Observed flood extents from Sentinel-1 SAR, via Google Earth Engine.

Why this exists: the pipeline's original training labels came from thresholds
on TWI and HAND, which are also model inputs, so the network could only
relearn the labelling rule. Sentinel-1 sees standing water directly and is
independent of the terrain features, which makes it both an honest training
label and the basis for validation.

Method follows the UN-SPIDER recommended practice for SAR flood mapping:
compare a during-flood image against a dry-season baseline of the same
relative orbit, flag the drop in backscatter that smooth open water produces,
then remove permanent water and terrain that cannot hold water.

Outputs (data/raw/):
    s1_flood_<event>.tif    per-event flood mask (1 = flooded)
    s1_flood_frequency.tif  share of events in which each pixel was flooded
"""

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Sentinel-1 GRD, interferometric wide swath, dual polarisation.
S1_COLLECTION = "COPERNICUS/S1_GRD"
JRC_WATER = "JRC/GSW1_4/GlobalSurfaceWater"
SRTM = "USGS/SRTMGL1_003"


def init_ee(cfg: dict):
    """Initialise Earth Engine with the project from config or environment.

    Earth Engine requires a Cloud project; stored credentials alone are not
    enough. Set `earthengine.project` in config.yaml or EARTHENGINE_PROJECT.
    """
    import ee

    project = (cfg.get("earthengine", {}) or {}).get("project") \
        or os.environ.get("EARTHENGINE_PROJECT")
    if not project:
        raise RuntimeError(
            "No Earth Engine project set. Add\n\n"
            "earthengine:\n  project: your-project-id\n\n"
            "to config.yaml, or export EARTHENGINE_PROJECT=your-project-id. "
            "Find it at https://code.earthengine.google.com (top right)."
        )
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # credentials missing or project not registered
        raise RuntimeError(
            f"Earth Engine init failed for project '{project}': {exc}\n"
            "Run `earthengine authenticate` and check the project is "
            "registered for Earth Engine."
        ) from exc
    logger.info(f"Earth Engine initialised (project: {project})")
    return ee


def _s1_scenes(ee, aoi, start: str, end: str, polarisation: str = "VV"):
    """Sentinel-1 GRD scenes over the AOI in a date range."""
    return (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarisation))
        .select(polarisation)
    )


def flood_mask_for_event(ee, cfg: dict, aoi, event: dict):
    """Flood mask for one event, as an ee.Image of 0/1 (1 = flooded).

    Returns (mask, diagnostics) where diagnostics records scene counts so a
    silent absence of imagery cannot masquerade as an absence of flooding.
    """
    s1_cfg = cfg.get("sentinel1", {})
    pol = s1_cfg.get("polarisation", "VV")
    water_db = s1_cfg.get("water_threshold_db", -16.0)
    drop_db = s1_cfg.get("backscatter_drop_db", -3.0)
    smoothing_m = s1_cfg.get("speckle_filter_m", 50)
    max_slope_deg = s1_cfg.get("max_slope_deg", 5.0)
    permanent_water_pct = s1_cfg.get("permanent_water_pct", 50)
    min_connected_px = s1_cfg.get("min_connected_pixels", 8)

    during = _s1_scenes(ee, aoi, event["start"], event["end"], pol)
    baseline = _s1_scenes(ee, aoi, event["baseline_start"], event["baseline_end"], pol)

    n_during = during.size()
    n_baseline = baseline.size()

    # Speckle is multiplicative and severe in single looks; a focal mean over
    # a few pixels is the standard cheap suppression for threshold mapping.
    def smooth(img):
        return img.focal_mean(smoothing_m, "circle", "meters")

    # Lowest backscatter during the event: water is dark, and taking the
    # minimum catches the flood peak even when scenes span several days.
    during_img = smooth(during.min())
    baseline_img = smooth(baseline.median())

    difference = during_img.subtract(baseline_img)

    # Water is dark in absolute terms AND darker than it was before.
    candidate = during_img.lt(water_db).And(difference.lt(drop_db))

    # Permanent water is not a flood.
    occurrence = ee.Image(JRC_WATER).select("occurrence").unmask(0)
    candidate = candidate.And(occurrence.lt(permanent_water_pct))

    # Radar shadow on slopes mimics water; water does not sit on hillsides.
    slope = ee.Terrain.slope(ee.Image(SRTM))
    candidate = candidate.And(slope.lt(max_slope_deg))

    # Drop speckle-sized specks.
    connected = candidate.selfMask().connectedPixelCount(min_connected_px + 1, True)
    mask = candidate.And(connected.gte(min_connected_px)).unmask(0).rename("flooded")

    return mask.toByte(), {"scenes_during": n_during, "scenes_baseline": n_baseline}


def flood_frequency(ee, cfg: dict, aoi, events: list[dict]):
    """Share of events in which each pixel was flooded, plus per-event masks."""
    masks, diagnostics = [], {}
    for event in events:
        mask, diag = flood_mask_for_event(ee, cfg, aoi, event)
        masks.append(mask)
        diagnostics[event["name"]] = diag
    # All masks share the band name "flooded", so the collection mean is one
    # band: the share of events in which each pixel was flooded. Renaming the
    # masks per event would give a band per event instead.
    stack = ee.ImageCollection(masks)
    frequency = stack.mean().rename("flood_frequency")
    return frequency, masks, diagnostics


def download_image(ee, image, aoi, scale: int, out_path: Path) -> Path:
    """Download an ee.Image over the AOI to a local GeoTIFF.

    Uses getDownloadURL, which caps the request at roughly 50 million pixels,
    so keep `scale` coarse enough for the area: the study bbox needs about
    100 m to stay inside that limit.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = image.getDownloadURL({
        "region": aoi,
        "scale": scale,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })
    logger.info(f"Downloading {out_path.name} at {scale} m …")
    with requests.get(url, stream=True, timeout=1800) as resp:
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
            tmp_path = Path(tmp.name)

    # GEO_TIFF comes back raw, but a multi-band request may arrive zipped.
    if zipfile.is_zipfile(tmp_path):
        with zipfile.ZipFile(tmp_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".tif")]
            if not names:
                raise RuntimeError(f"No GeoTIFF inside archive for {out_path.name}")
            with zf.open(names[0]) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp_path.unlink()
    else:
        shutil.move(str(tmp_path), out_path)

    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved {out_path} ({size_mb:.1f} MB)")
    return out_path


def run_sentinel1_pipeline(cfg: dict, raw_dir: Path = Path("data/raw")) -> dict:
    """Map observed flood extents for the configured events.

    Returns a dict of event name → output path, plus the frequency raster.
    """
    ee = init_ee(cfg)

    s1_cfg = cfg.get("sentinel1", {})
    events = s1_cfg.get("events", [])
    if not events:
        raise ValueError("No flood events configured under `sentinel1.events`.")

    bbox = cfg["aoi"]["bbox"]
    aoi = ee.Geometry.Rectangle(bbox)
    scale = s1_cfg.get("export_scale_m", 100)

    frequency, masks, diagnostics = flood_frequency(ee, cfg, aoi, events)

    # Fail loudly when an event has no imagery: an empty mask would otherwise
    # be indistinguishable from "nothing flooded".
    counts = {name: {k: v.getInfo() for k, v in diag.items()}
              for name, diag in diagnostics.items()}
    for name, diag in counts.items():
        logger.info(f"{name}: {diag['scenes_during']} scenes during, "
                    f"{diag['scenes_baseline']} baseline")
        if diag["scenes_during"] == 0 or diag["scenes_baseline"] == 0:
            raise RuntimeError(
                f"Event '{name}' has no Sentinel-1 coverage "
                f"({diag}); adjust its dates or drop it."
            )

    outputs = {}
    for event, mask in zip(events, masks):
        path = raw_dir / f"s1_flood_{event['name']}.tif"
        if path.exists():
            logger.info(f"{path.name} already present, skipping download")
            outputs[event["name"]] = path
            continue
        outputs[event["name"]] = download_image(ee, mask, aoi, scale, path)

    outputs["frequency"] = download_image(
        ee, frequency.multiply(100).toByte(), aoi, scale,
        raw_dir / "s1_flood_frequency.tif",
    )

    import json
    (raw_dir / "s1_flood_events.json").write_text(json.dumps(
        {"events": events, "scene_counts": counts, "scale_m": scale}, indent=2
    ))
    logger.info(f"Sentinel-1 flood mapping complete: {len(events)} events")
    return outputs
