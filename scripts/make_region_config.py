"""
Write a national region's configuration from the partition and its events.

    python scripts/make_region_config.py north_west --events-from config.yaml \\
        --min-events 2 --hazard "Riverine flood" [--no-past-flooding]

The districts come from configs/national/partition.yaml; the bounding box and
map centre are computed from them; the flood events are copied, with their
sources, from an existing configuration whose events were checked for this
area (scripts/check_s1_events.py). Assets come from the national OpenStreetMap
extract. The file is a starting point to read and commit, not to regenerate.
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PBF = "data/shared/osm/bangladesh-2026-09-30.osm.pbf"


def main():
    import geopandas as gpd

    from pipeline.cli import _load_config

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("region")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--events-from", help="an existing config whose events apply")
    group.add_argument("--events-file", help="a YAML list of checked events")
    parser.add_argument("--min-events", type=int, required=True)
    parser.add_argument("--no-past-flooding", action="store_true")
    args = parser.parse_args()

    partition = yaml.safe_load(open(ROOT / "configs/national/partition.yaml"))["flood"]
    spec = partition[args.region]
    country = gpd.read_file(ROOT / "data/shared/geoboundaries/BGD_ADM2.geojson")
    named = country[country["shapeName"].isin(spec["districts"])]
    west, south, east, north = (float(v) for v in named.total_bounds)
    box = [round(west - 0.05, 2), round(south - 0.05, 2), round(east + 0.05, 2), round(north + 0.05, 2)]
    centre = [round((south + north) / 2, 2), round((west + east) / 2, 2)]
    events = (_load_config(str(ROOT / args.events_from))["sentinel1"]["events"] if args.events_from
              else yaml.safe_load(open(ROOT / args.events_file)))
    rid = args.region

    cfg = {
        "extends": "../../config.yaml",
        "paths": {k: f"data/{rid}/{k.split('_')[0]}" for k in ("raw_dir", "processed_dir", "output_dir")},
        "aoi": {"name": rid, "divisions": [], "bbox": box, "districts": spec["districts"],
                "crs": "EPSG:32646", "grid_resolution_m": 500},
        "asset_model": {"past_flooding": not args.no_past_flooding},
        "data": {
            "osm": {"source": "pbf", "pbf_path": PBF},
            "labels": {"min_events": args.min_events},
            "dem": {"path": f"data/{rid}/raw/dem_srtm_30m.tif"},
            "admin_boundaries": {lvl: f"data/{rid}/raw/bgd_{lvl}.gpkg"
                                 for lvl in ("union", "upazila", "district")},
            "vulnerability": {"population_path": f"data/{rid}/raw/worldpop_popdens.tif"},
        },
        "sentinel1": {"events": events},
        "dashboard": {"map_center": centre, "map_zoom": 8},
    }
    header = (f"# {spec['name']}: {', '.join(spec['districts'])}.\n"
              f"# A region of the national partition (configs/national/partition.yaml),\n"
              f"# defined by its districts. Events from {args.events_from or args.events_file}, whose\n"
              f"# windows were checked over these districts with scripts/check_s1_events.py.\n"
              f"# Only the differences from the base config live here.\n")
    out = ROOT / "configs/national" / f"{rid}.yaml"
    out.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"wrote {out.relative_to(ROOT)}: box {box}, {len(events)} events")


if __name__ == "__main__":
    main()
