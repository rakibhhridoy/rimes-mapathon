"""
Build the independent validation regions and run the comparisons with the record.

A validation region is a region of the national partition with every asset
that a study box also holds dropped, so no asset is scored in both. Assets
are matched to the Rangpur-Rajshahi and coastal boxes by OpenStreetMap
identity and to the Sylhet box, whose inventory carries no identifiers, by
location. The national regions themselves are left untouched.

    python scripts/make_validation_region.py                   # every region
    python scripts/make_validation_region.py jamuna_east       # one region
    python scripts/make_validation_region.py --run central     # run on an existing build
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# National region -> independent copy. Each has flood events in at least two
# years, so each later year can be predicted from the ones before it.
REGIONS = {
    "jamuna_east": "jamuna_east_ind",
    "central": "central_ind",
    "chattogram_coast": "chattogram_coast_ind",
    "west_central": "west_central_ind",
}
STUDY_BY_ID = (ROOT / "data" / "raw" / "infrastructure_raw.gpkg",
               ROOT / "data" / "sw_coastal" / "raw" / "infrastructure_raw.gpkg")
STUDY_BY_LOCATION = (ROOT / "data" / "sylhet" / "raw" / "infrastructure_raw.gpkg",)
# Coordinates are compared at 1e-5 degrees, about a metre.
DECIMALS = 5


def _locations(frame):
    import numpy as np

    point = frame.geometry.representative_point()
    return list(zip(np.round(point.x, DECIMALS), np.round(point.y, DECIMALS)))


def build(region: str) -> None:
    import geopandas as gpd

    source, target = ROOT / "data" / region, ROOT / "data" / REGIONS[region]
    # The inputs are linked rather than copied, since the disk cannot hold a
    # second copy of every raster. Only the asset inventory is the region's
    # own; nothing the checks run writes into raw/ or processed/ except the
    # feature table, which is left out so that extraction writes a fresh one.
    own = {"infrastructure_raw.gpkg", "node_features.parquet"}
    for kind in ("raw", "processed"):
        if (target / kind).exists():
            # The SSD's macOS "._" sidecar files vanish with their partners.
            shutil.rmtree(target / kind, onerror=lambda f, path, exc: None
                          if issubclass(exc[0], FileNotFoundError) else exc[1].with_traceback(exc[2]))
        for path in sorted((source / kind).rglob("*")):
            if path.is_dir() or path.name.startswith("._") or path.name in own:
                continue
            link = target / kind / path.relative_to(source / kind)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path)
    (target / "output").mkdir(parents=True, exist_ok=True)

    assets = gpd.read_file(str(source / "raw" / "infrastructure_raw.gpkg"))
    ids, places = set(), set()
    for path in STUDY_BY_ID:
        study = gpd.read_file(str(path), ignore_geometry=True)
        ids |= set(zip(study["osm_type"], study["osm_id"]))
    for path in STUDY_BY_LOCATION:
        places |= set(_locations(gpd.read_file(str(path))))
    by_id = [key in ids for key in zip(assets["osm_type"], assets["osm_id"])]
    by_place = [p in places for p in _locations(assets)]
    shared = [a or b for a, b in zip(by_id, by_place)]
    keep = assets[[not s for s in shared]]
    keep.to_file(str(target / "raw" / "infrastructure_raw.gpkg"), driver="GPKG")
    (target / "output" / "validation_region.json").write_text(json.dumps(
        {"assets_national": len(assets), "assets_kept": len(keep),
         "assets_shared_with_study_boxes": int(sum(shared)),
         "shared_by_osm_id": int(sum(by_id)), "shared_by_location": int(sum(by_place))},
        indent=2))
    print(f"{region}: kept {len(keep):,} of {len(assets):,} assets")


def run(region: str) -> None:
    from pipeline.benchmark import (run_block_size_check, run_past_flooding_feature,
                                    run_persistence_check, run_record_checks,
                                    run_rolling_origin, run_target_variants,
                                    run_temporal_holdout)
    from pipeline.cli import _dir, _infra_path, _load_config

    cfg = _load_config(str(ROOT / "configs" / "validation" / f"{region}.yaml"))
    args = (cfg, _dir(cfg, "processed"), _dir(cfg, "raw"), _dir(cfg, "output"),
            _infra_path(cfg))
    run_rolling_origin(*args)
    # The 2022 cutoff checks need events on both sides of it.
    years = sorted({int(e["start"][:4]) for e in cfg["sentinel1"]["events"]})
    if years[0] <= 2022 < years[-1]:
        for check in (run_temporal_holdout, run_past_flooding_feature, run_persistence_check,
                      run_block_size_check, run_target_variants, run_record_checks):
            check(*args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("regions", nargs="*", default=list(REGIONS),
                        help="national regions to build (default: all)")
    parser.add_argument("--run", action="store_true", help="skip the build step")
    options = parser.parse_args()
    for name in options.regions:
        if not options.run:
            build(name)
        run(name)
