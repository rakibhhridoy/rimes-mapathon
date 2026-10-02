"""
Check that Sentinel-1 imaged a region during each flood window.

    python scripts/check_s1_events.py configs/national/central.yaml
    python scripts/check_s1_events.py --districts Dhaka Gazipur \\
        --window 2020-07-11 2020-07-31 --window 2024-07-01 2024-07-15

For every window it lists each pass (local time, UTC+6), the orbit direction
and the share of the region the pass covered, then the share covered by all
passes together and by the February-March baseline of the same year. The
region is the union of its districts in the national boundary file, not its
bounding box, so coverage means coverage of the land the model will use.
Needs an Earth Engine project (earthengine.project, or EARTHENGINE_PROJECT).
"""

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def region_geometry(districts: list[str]):
    import ee
    import geopandas as gpd

    country = gpd.read_file(ROOT / "data/shared/geoboundaries/BGD_ADM2.geojson")
    named = country[country["shapeName"].isin(districts)]
    missing = set(districts) - set(named["shapeName"])
    if missing:
        raise SystemExit(f"unknown districts: {sorted(missing)}")
    shape = named.geometry.union_all().simplify(0.005)
    return ee.Geometry(shape.__geo_interface__)


def check(geometry, windows):
    import ee

    area = geometry.area(100).getInfo()

    def passes(start, end):
        return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geometry)
                .filterDate(start, end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))

    def cover(collection):
        if collection.size().getInfo() == 0:
            return 0.0
        return collection.geometry().intersection(geometry, 100).area(100).getInfo() / area

    for name, start, end in windows:
        c = passes(start, end)
        stamps = c.aggregate_array("system:time_start").getInfo()
        days = sorted({dt.datetime.utcfromtimestamp(t / 1000).date() for t in stamps})
        print(f"\n{name}  {start} to {end}")
        for day in days:
            sub = c.filterDate(day.isoformat(), (day + dt.timedelta(days=1)).isoformat())
            local = sorted({(dt.datetime.utcfromtimestamp(t / 1000) + dt.timedelta(hours=6))
                            .strftime("%d %b %H:%M") for t in stamps
                            if dt.datetime.utcfromtimestamp(t / 1000).date() == day})
            orbit = sub.aggregate_first("orbitProperties_pass").getInfo()
            print(f"  {', '.join(local):28s} {orbit:10s} {cover(sub):5.0%}")
        year = start[:4]
        base = passes(f"{year}-02-01", f"{year}-03-31")
        print(f"  all passes {cover(c):.0%}; Feb-Mar {year} baseline {base.size().getInfo()} "
              f"scenes, {cover(base):.0%}")


def main():
    import ee
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", nargs="?")
    parser.add_argument("--districts", nargs="*")
    parser.add_argument("--window", nargs=2, action="append", metavar=("START", "END"))
    args = parser.parse_args()

    from pipeline.cli import _load_config

    cfg = _load_config(args.config) if args.config else _load_config(str(ROOT / "config.yaml"))
    project = os.environ.get("EARTHENGINE_PROJECT") or cfg.get("earthengine", {}).get("project")
    ee.Initialize(project=project)
    districts = args.districts or (cfg.get("aoi") or {}).get("districts")
    if not districts:
        raise SystemExit("name the districts, in the config (aoi.districts) or with --districts")
    windows = ([(f"window {i + 1}", a, b) for i, (a, b) in enumerate(args.window)]
               if args.window else
               [(e["name"], e["start"], e["end"]) for e in cfg["sentinel1"]["events"]])
    check(region_geometry(districts), windows)


if __name__ == "__main__":
    main()
