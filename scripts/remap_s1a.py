"""
Re-map every flood event of the paper's regions from Sentinel-1A alone.

Sentinel-1B failed in December 2021, so the 2017-2020 events were mapped
from two satellites and the 2024 events from one. Each event mask takes the
lowest backscatter over its window, so more scenes can mean more mapped
flooding. These masks give every event the same satellite and revisit, for a
sensitivity check of the temporal tests; the published masks are untouched.

    python scripts/remap_s1a.py            # -> <region raw dir>_s1a/
    python scripts/remap_s1a.py --check    # temporal test on those masks,
                                           #    -> output/sensitivity_s1a/

The check also writes each event's flooded share of the region, from the
published masks and from the Sentinel-1A masks, to event_extents.json.
"""

import copy
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data.regions import REGION_CONFIGS, region_paths  # noqa: E402
from pipeline.cli import _dir, _infra_path, _load_config  # noqa: E402
from pipeline.sentinel1 import run_sentinel1_pipeline  # noqa: E402

PAPER_REGIONS = ("rangpur_rajshahi", "sylhet", "sw_coastal")


def s1a_dir(region: str) -> Path:
    raw = region_paths(region)["raw"]
    return raw.with_name(raw.name + "_s1a")


def flooded_share(cfg: dict, path: Path) -> float:
    """Share of the region's in-country cells that an event mask marks flooded."""
    import numpy as np
    import rasterio

    from pipeline.country import foreign_mask

    with rasterio.open(path) as src:
        flooded = np.nan_to_num(src.read(1)) > 0.5
        home = ~foreign_mask(cfg, flooded.shape, src.transform, src.crs)
    return float(flooded[home].mean())


def check(region: str) -> None:
    import json

    from pipeline.benchmark import run_temporal_holdout

    cfg = _load_config(str(ROOT / REGION_CONFIGS[region][2]))
    out = _dir(cfg, "output") / "sensitivity_s1a"
    run_temporal_holdout(cfg, _dir(cfg, "processed"), s1a_dir(region), out, _infra_path(cfg))
    scenes = {kind: json.loads((d / "s1_flood_events.json").read_text())["scene_counts"]
              for kind, d in (("published", _dir(cfg, "raw")), ("s1a", s1a_dir(region)))}
    extents = {e["name"]: {kind: {"flooded_share": flooded_share(cfg, d / f"s1_flood_{e['name']}.tif"),
                                  "scenes_during": scenes[kind][e["name"]]["scenes_during"]}
                           for kind, d in (("published", _dir(cfg, "raw")),
                                           ("s1a", s1a_dir(region)))}
               for e in cfg["sentinel1"]["events"]}
    (out / "event_extents.json").write_text(json.dumps(extents, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = [a for a in sys.argv[1:] if a != "--check"]
    for region in args or PAPER_REGIONS:
        if "--check" in sys.argv:
            check(region)
            continue
        cfg = copy.deepcopy(_load_config(str(ROOT / REGION_CONFIGS[region][2])))
        cfg.setdefault("sentinel1", {})["platform"] = "A"
        out = s1a_dir(region)
        out.mkdir(parents=True, exist_ok=True)
        run_sentinel1_pipeline(cfg, raw_dir=out)


if __name__ == "__main__":
    main()
