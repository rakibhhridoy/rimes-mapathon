"""
Sentinel-1 flood masks against an independent flood map.

Why this exists: every label and every validation score in the pipeline rests
on the Sentinel-1 masks, and their accuracy was only argued from other
studies. The Global Flood Database (Tellman et al. 2021) maps the same floods
from MODIS optical imagery with a different sensor, algorithm and team, so
agreement with it is evidence about the masks themselves.

An event opts in by naming its database events in the region configuration:

    sentinel1:
      events:
        - name: "aug2017"
          gfd_ids: [4507, 4508]

The comparison runs on the database's 250 m grid, inside Bangladesh, over
cells that MODIS saw clear at least once and that are not permanent water.
The Sentinel-1 mask is averaged onto that grid and a cell counts as flooded
where at least half of it flooded.

Output (per region): flood_crosscheck.json.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GFD_COLLECTION = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
GFD_SCALE_M = 250
FLOODED_SHARE = 0.5


def _download_gfd(ee, cfg: dict, ids: list[int], out_path: Path) -> Path:
    """The database events as one band: 1 flooded, 2 seen clear, 4 permanent water."""
    from pipeline.sentinel1 import download_image

    west, south, east, north = cfg["aoi"]["bbox"]
    aoi = ee.Geometry.Rectangle([west, south, east, north])
    events = (ee.ImageCollection(GFD_COLLECTION)
              .filter(ee.Filter.inList("id", ids)))
    if events.size().getInfo() != len(ids):
        raise ValueError(f"Global Flood Database has no event among {ids}.")
    flooded = events.select("flooded").max().gt(0)
    clear = events.select("clear_views").max().gt(0)
    permanent = events.select("jrc_perm_water").max().gt(0)
    coded = (flooded.add(clear.multiply(2)).add(permanent.multiply(4))
             .unmask(0).toByte())
    return download_image(ee, coded, aoi, GFD_SCALE_M, out_path)


def _bangladesh_mask(raw_dir: Path, shape, transform, crs) -> np.ndarray:
    import geopandas as gpd
    from rasterio.features import rasterize

    districts = gpd.read_file(str(raw_dir / "bgd_district.gpkg")).to_crs(crs)
    return rasterize(((g, 1) for g in districts.geometry), out_shape=shape,
                     transform=transform, fill=0, dtype="uint8").astype(bool)


def compare_event(s1_path: Path, gfd_path: Path, raw_dir: Path) -> dict:
    """Agreement between one Sentinel-1 mask and the database map of the event."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from sklearn.metrics import roc_auc_score

    from pipeline.validate import _binary_agreement

    with rasterio.open(gfd_path) as src:
        code = src.read(1)
        shape, transform, crs = code.shape, src.transform, src.crs

    # The mask stores dry ground as 0, so it is read as values, not as nodata.
    with rasterio.open(s1_path) as src:
        mask = src.read(1).astype(np.float32)
        share = np.zeros(shape, dtype=np.float32)
        reproject(mask, share, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  src_nodata=None, dst_nodata=None,
                  resampling=Resampling.average)

    reference = (code & 1).astype(bool)
    seen = (code & 2).astype(bool)
    permanent = (code & 4).astype(bool)
    valid = seen & ~permanent & _bangladesh_mask(raw_dir, shape, transform, crs)

    ref, flooded = reference[valid], share[valid] >= FLOODED_SHARE
    result = {
        "n_cells": int(valid.sum()),
        "cell_size_m": GFD_SCALE_M,
        "s1_flooded_share": float(flooded.mean()),
        "gfd_flooded_share": float(ref.mean()),
        "agreement": float((ref == flooded).mean()),
        **_binary_agreement(ref, flooded),
    }
    # Threshold-free: does the flooded share of each cell rank the cells the
    # database calls flooded above the rest?
    if len(np.unique(ref)) == 2:
        result["auc_roc"] = float(roc_auc_score(ref, share[valid]))
    return result


def run_crosscheck(cfg: dict, raw_dir: Path, output_dir: Path) -> dict:
    """Compare every event that names database events, and write the results."""
    from pipeline.sentinel1 import init_ee

    events = [e for e in (cfg.get("sentinel1") or {}).get("events", [])
              if e.get("gfd_ids")]
    if not events:
        raise ValueError("No Sentinel-1 event names gfd_ids in this configuration.")

    ee = init_ee(cfg)
    results = {}
    for event in events:
        gfd_path = raw_dir / f"gfd_{event['name']}.tif"
        if not gfd_path.exists():
            _download_gfd(ee, cfg, [int(i) for i in event["gfd_ids"]], gfd_path)
        results[event["name"]] = {
            "gfd_ids": [int(i) for i in event["gfd_ids"]],
            **compare_event(raw_dir / f"s1_flood_{event['name']}.tif",
                            gfd_path, raw_dir),
        }
        r = results[event["name"]]
        logger.info(f"{event['name']}: POD {r['pod']:.3f}, FAR {r['far']:.3f}, "
                    f"CSI {r['csi']:.3f} over {r['n_cells']:,} cells")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "flood_crosscheck.json"
    path.write_text(json.dumps({"reference": GFD_COLLECTION,
                                "flooded_share_threshold": FLOODED_SHARE,
                                "events": results}, indent=2))
    logger.info(f"Wrote {path}")
    return results
