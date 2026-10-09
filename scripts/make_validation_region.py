"""
Build the independent Jamuna east validation region and run the record checks.

The national Jamuna east region shares 1,263 mapped assets with the
Rangpur-Rajshahi study box, which crosses the Jamuna. The validation region
copies the national region's inputs and drops every asset the study box also
holds, so no asset is scored in both, then runs the comparisons with the
flood record that the paper reports for the study regions.

    python scripts/make_validation_region.py           # build and run
    python scripts/make_validation_region.py --run     # run on an existing build
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "data" / "jamuna_east"
TARGET = ROOT / "data" / "jamuna_east_ind"
STUDY_ASSETS = ROOT / "data" / "raw" / "infrastructure_raw.gpkg"
CONFIG = ROOT / "configs" / "validation" / "jamuna_east.yaml"


def build() -> None:
    import geopandas as gpd

    for kind in ("raw", "processed"):
        if (TARGET / kind).exists():
            shutil.rmtree(TARGET / kind)
        shutil.copytree(SOURCE / kind, TARGET / kind)
    # Written by feature extraction for the assets it scores, so not carried over.
    (TARGET / "processed" / "node_features.parquet").unlink(missing_ok=True)
    (TARGET / "output").mkdir(parents=True, exist_ok=True)

    assets = gpd.read_file(str(SOURCE / "raw" / "infrastructure_raw.gpkg"))
    study = gpd.read_file(str(STUDY_ASSETS), ignore_geometry=True)
    shared = set(zip(study["osm_type"], study["osm_id"]))
    keep = assets[[key not in shared for key in zip(assets["osm_type"], assets["osm_id"])]]
    keep.to_file(str(TARGET / "raw" / "infrastructure_raw.gpkg"), driver="GPKG")
    (TARGET / "output" / "validation_region.json").write_text(json.dumps(
        {"assets_national": len(assets), "assets_kept": len(keep),
         "assets_shared_with_study_box": len(assets) - len(keep)}, indent=2))
    print(f"kept {len(keep):,} of {len(assets):,} assets")


def run() -> None:
    from pipeline.benchmark import (run_block_size_check, run_past_flooding_feature,
                                    run_persistence_check, run_record_checks,
                                    run_target_variants, run_temporal_holdout)
    from pipeline.cli import _dir, _infra_path, _load_config

    cfg = _load_config(str(CONFIG))
    args = (cfg, _dir(cfg, "processed"), _dir(cfg, "raw"), _dir(cfg, "output"),
            _infra_path(cfg))
    for check in (run_temporal_holdout, run_past_flooding_feature, run_persistence_check,
                  run_block_size_check, run_target_variants, run_record_checks):
        check(*args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="store_true", help="skip the build step")
    if not parser.parse_args().run:
        build()
    run()
