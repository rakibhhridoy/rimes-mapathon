"""
Build the web application's data: one SQLite file and one tile set per region.

The Streamlit dashboard reads the pipeline's outputs directly and rebuilds the
whole map on every interaction, which ships about 17 MB of HTML each time.
The web application instead reads a database for numbers and vector tiles for
geometry, both built here, once, from the same pipeline outputs.

    python scripts/build_web.py            # every region with results
    python scripts/build_web.py --regions sylhet

Outputs:
    web/data/hazmapper.sqlite      assets, summaries, metrics, search index
    web/data/tiles/<region>.pmtiles  assets, unions and hotspots as tiles
    web/data/overlays/<region>/*.png  pre-rendered raster layers
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.data.regions import REGION_CONFIGS, has_results, region_paths  # noqa: E402

WEB = ROOT / "web" / "data"
DB_PATH = WEB / "hazmapper.sqlite"

# Layers that become vector tiles: source file, zoom range, the fields the
# map actually draws with, and a simplification tolerance in degrees.
# Carrying every column would triple the tiles for nothing: the detail card
# reads the database, not the tile.
TILE_LAYERS = {
    "assets": {
        "source": "risk_ranked_assets.geojson", "zoom": (4, 14),
        "fields": ["name", "asset_type", "division", "flood_risk",
                   "flood_probability", "risk_rank"],
        "simplify": None,
    },
    "unions": {
        "source": "union_risk_summary.geojson", "zoom": (4, 12),
        "fields": ["admin_name", "admin_label", "mean_risk", "max_risk",
                   "mean_risk_people", "n_high_risk", "n_cells"],
        "simplify": 0.0005,          # about 50 m, the display cache's tolerance
    },
    "hotspots": {
        "source": "hotspot_clusters.geojson", "zoom": (4, 12),
        "fields": ["hotspot_z", "composite_risk"], "simplify": 0.0005,
    },
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS regions (
    id TEXT PRIMARY KEY, name TEXT, hazard TEXT, config TEXT,
    centre_lat REAL, centre_lon REAL, zoom INTEGER, bbox TEXT,
    has_assets INTEGER, has_landslide INTEGER
);
CREATE TABLE IF NOT EXISTS assets (
    region TEXT, asset_id INTEGER, name TEXT, asset_type TEXT, division TEXT,
    lat REAL, lon REAL, flood_risk REAL, flood_probability REAL,
    risk_rank INTEGER, is_high_risk INTEGER,
    cell_hazard REAL, cell_exposure REAL, cell_vulnerability REAL,
    cell_composite_risk REAL,
    PRIMARY KEY (region, asset_id)
);
CREATE INDEX IF NOT EXISTS assets_rank ON assets(region, risk_rank);
CREATE INDEX IF NOT EXISTS assets_type ON assets(region, asset_type);
CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts USING fts5(
    name, asset_type, division, region UNINDEXED, asset_id UNINDEXED,
    tokenize = 'unicode61'
);
CREATE TABLE IF NOT EXISTS admin_summary (
    region TEXT, level TEXT, unit_id TEXT, name TEXT, parent TEXT,
    mean_risk REAL, max_risk REAL, mean_risk_people REAL,
    n_assets INTEGER, has_data INTEGER, payload TEXT
);
CREATE INDEX IF NOT EXISTS admin_lookup ON admin_summary(region, level, mean_risk);
CREATE TABLE IF NOT EXISTS metrics (region TEXT, key TEXT, payload TEXT,
    PRIMARY KEY (region, key));
CREATE TABLE IF NOT EXISTS landslide_units (
    region TEXT, unit_id TEXT, name TEXT, label TEXT, district TEXT,
    susceptibility_mean REAL, susceptibility_max REAL, n_pixels INTEGER,
    population INTEGER, n_landslides INTEGER, lat REAL, lon REAL,
    PRIMARY KEY (region, unit_id)
);
CREATE TABLE IF NOT EXISTS landslides (
    region TEXT, landslide_id INTEGER, event_date TEXT, title TEXT,
    category TEXT, trigger TEXT, unit_id TEXT, upazila TEXT, district TEXT,
    lat REAL, lon REAL, susceptibility REAL, unit_susceptibility REAL,
    PRIMARY KEY (region, landslide_id)
);
"""


def _connect() -> sqlite3.Connection:
    WEB.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None      # NaN is not a number worth storing


def load_assets(conn: sqlite3.Connection, region: str, paths: dict) -> int:
    """Asset rows and the search index, from the display cache."""
    import geopandas as gpd

    source = paths["cache"] / "risk_ranked_assets.parquet"
    if not source.exists():
        source = paths["output"] / "risk_ranked_assets.geojson"
        if not source.exists():
            return 0
        frame = gpd.read_file(source)
    else:
        frame = gpd.read_parquet(source)

    if "lon" not in frame.columns or "lat" not in frame.columns:
        points = frame.geometry.representative_point()
        frame = frame.assign(lon=points.x, lat=points.y)

    conn.execute("DELETE FROM assets WHERE region = ?", (region,))
    conn.execute("DELETE FROM assets_fts WHERE region = ?", (region,))

    rows, search = [], []
    for asset_id, row in enumerate(frame.itertuples(index=False)):
        get = lambda key: getattr(row, key, None)      # noqa: E731
        name = str(get("name") or "unnamed")
        asset_type = str(get("asset_type") or "")
        division = str(get("division") or "")
        rows.append((
            region, asset_id, name, asset_type, division,
            _float(get("lat")), _float(get("lon")),
            _float(get("flood_risk")), _float(get("flood_probability")),
            int(get("risk_rank") or 0), int(bool(get("is_high_risk"))),
            _float(get("cell_hazard")), _float(get("cell_exposure")),
            _float(get("cell_vulnerability")), _float(get("cell_composite_risk")),
        ))
        search.append((name, asset_type, division, region, asset_id))

    conn.executemany("INSERT OR REPLACE INTO assets VALUES (" + ",".join("?" * 15) + ")", rows)
    conn.executemany(
        "INSERT INTO assets_fts (name, asset_type, division, region, asset_id) "
        "VALUES (?, ?, ?, ?, ?)", search)
    return len(rows)


def load_admin(conn: sqlite3.Connection, region: str, paths: dict) -> int:
    """Union, upazila and district summaries, kept as rows plus raw payload."""
    import csv

    conn.execute("DELETE FROM admin_summary WHERE region = ?", (region,))
    total = 0
    for level in ("union", "upazila", "district"):
        path = paths["output"] / f"{level}_risk_summary.csv"
        if not path.exists():
            continue
        with open(path) as handle:
            for row in csv.DictReader(handle):
                has_data = str(row.get("has_data", "")).strip().lower() == "true"
                conn.execute(
                    "INSERT INTO admin_summary VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (region, level, row.get("unit_id") or row.get("name"),
                     row.get("name"), row.get("parent") or row.get("label"),
                     _float(row.get("mean_composite_risk")),
                     _float(row.get("max_composite_risk")),
                     _float(row.get("mean_composite_risk_people")),
                     int(_float(row.get("n_assets")) or 0), int(has_data),
                     json.dumps(row)))
                total += 1
    return total


def load_metrics(conn: sqlite3.Connection, region: str, paths: dict) -> int:
    """Everything the panels quote: provenance, validation, models, weights."""
    files = {
        "pipeline_metadata": "pipeline_metadata.json",
        "validation": "validation_metrics.json",
        "hazard_model": "hazard_model.json",
        "landslide_model": "landslide_model.json",
        "benchmark": "benchmark.json",
        "past_model": "past_model_metrics.json",
        "temporal_holdout": "temporal_holdout.json",
        "past_flooding_feature": "past_flooding_feature.json",
        "label_comparison": "label_comparison.json",
        "sensitivity": "sensitivity.json",
        "vulnerability_weights": "vulnerability_weights.json",
        "variogram": "variogram_params.json",
    }
    stored = 0
    for key, name in files.items():
        payload = _read_json(paths["output"] / name)
        if payload is None:
            continue
        conn.execute("INSERT OR REPLACE INTO metrics VALUES (?, ?, ?)",
                     (region, key, json.dumps(payload)))
        stored += 1
    return stored


def _fields_present(source: Path, wanted: list[str]) -> list[str]:
    """The wanted fields the file actually has, so -select cannot fail.

    Matched at the start of a line: ogrinfo's own header carries lines like
    "Layer name: union_risk_summary", which a looser test reads as a field
    called name.
    """
    import re

    result = subprocess.run(["ogrinfo", "-so", "-al", str(source)],
                            capture_output=True, text=True)
    declared = set(re.findall(r"^(\w+): \w+ \(", result.stdout, re.MULTILINE))
    return [field for field in wanted if field in declared]


def build_tiles(region: str, paths: dict) -> float:
    """Vector tiles per layer, as PMTiles.

    One file per layer rather than one per region: the PMTiles driver cannot
    add a layer to a file it has already written. PMTiles is read over HTTP
    range requests, so the map needs no tile server, only nginx.
    """
    out_dir = WEB / "tiles" / region
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0.0
    for layer, spec in TILE_LAYERS.items():
        source = paths["output"] / spec["source"]
        if not source.exists():
            continue
        target = out_dir / f"{layer}.pmtiles"
        if target.exists():
            target.unlink()
        zmin, zmax = spec["zoom"]
        command = ["ogr2ogr", "-f", "PMTiles", str(target), str(source),
                   "-nln", layer,
                   "-dsco", f"MINZOOM={zmin}", "-dsco", f"MAXZOOM={zmax}",
                   "-dsco", "MAX_SIZE=5000000"]
        if spec["fields"]:
            available = _fields_present(source, spec["fields"])
            if available:
                command += ["-select", ",".join(available)]
        if spec["simplify"]:
            command += ["-simplify", str(spec["simplify"])]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{region}/{layer}: {result.stderr.strip()[:400]}")
        written += target.stat().st_size / 1e6
    return round(written, 1)


def _clip_to_country(png: bytes, bounds, cfg: dict) -> bytes:
    """Make the overlay transparent outside the country's land.

    The pre-rendered rasters cover each region's whole bounding box, which
    reaches into India and Myanmar and over the sea. A terrain surface means
    nothing there, so every pixel outside the country polygon configured as
    `aoi.country_boundary` is blanked. Assets use a looser rule
    (pipeline/country.py) that keeps chars and coastal ground just outside the
    polygon; an overlay does not need to.
    """
    import io

    import geopandas as gpd
    import numpy as np
    from PIL import Image
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    boundary = (cfg.get("aoi") or {}).get("country_boundary")
    if not boundary or not (ROOT / boundary).exists():
        return png
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    width, height = image.size
    (south, west), (north, east) = bounds
    country = gpd.read_file(ROOT / boundary).to_crs("EPSG:4326")
    districts = (cfg.get("aoi") or {}).get("districts")
    if districts:
        # a region defined by its districts shows its surfaces only there
        country = country[country["shapeName"].isin(districts)]
    home = country.geometry.union_all()
    inside = rasterize([(home, 1)], out_shape=(height, width),
                       transform=from_bounds(west, south, east, north, width, height),
                       fill=0, dtype="uint8").astype(bool)
    if inside.all():
        return png
    pixels = np.array(image)
    pixels[~inside, 3] = 0
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="PNG", optimize=True)
    return out.getvalue()


def copy_overlays(region: str, paths: dict, cfg: dict | None = None) -> int:
    """The raster layers the pipeline already pre-rendered for the dashboard."""
    out_dir = WEB / "overlays" / region
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in sorted(paths["cache"].glob("raster_*.json")):
        if source.name.startswith("._"):
            continue
        payload = _read_json(source)
        if not payload or "image_base64" not in payload:
            continue
        import base64

        name = source.stem.replace("raster_", "")
        png = base64.b64decode(payload["image_base64"])
        if cfg and payload.get("bounds"):
            png = _clip_to_country(png, payload["bounds"], cfg)
        (out_dir / f"{name}.png").write_bytes(png)
        (out_dir / f"{name}.json").write_text(json.dumps(
            {"bounds": payload.get("bounds"), "name": name}))
        copied += 1
    return copied


def _write_pmtiles(gdf, target: Path, layer: str, zoom: tuple, simplify=None) -> float:
    """One GeoDataFrame as a PMTiles layer, through a temporary GeoJSON."""
    import tempfile

    if target.exists():
        target.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / f"{layer}.geojson"
        gdf.to_file(source, driver="GeoJSON")
        command = ["ogr2ogr", "-f", "PMTiles", str(target), str(source), "-nln", layer,
                   "-dsco", f"MINZOOM={zoom[0]}", "-dsco", f"MAXZOOM={zoom[1]}",
                   "-dsco", "MAX_SIZE=5000000"]
        if simplify:
            command += ["-simplify", str(simplify)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{target.name}: {result.stderr.strip()[:400]}")
    return target.stat().st_size / 1e6


def build_landslide(conn: sqlite3.Connection, region: str, paths: dict) -> dict:
    """Upazila summaries and mapped landslides, as tables and vector tiles.

    Each upazila carries the pipeline's mean and maximum susceptibility and
    population, plus the number of mapped landslides inside it; each landslide
    carries the modelled susceptibility of the ground where it happened, read
    from the surface the map draws.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio

    summary = _read_json(paths["output"] / "landslide_upazila.json")
    boundaries = paths["raw"] / "bgd_upazila.gpkg"
    inventory = paths["raw"] / "coolr_landslides.geojson"
    surface = paths["output"] / "landslide_susceptibility.tif"
    if not (summary and boundaries.exists()):
        return {"units": 0, "landslides": 0}

    units = gpd.read_file(boundaries).to_crs("EPSG:4326")
    figures = {row["admin_label"]: row for row in summary}
    units = units[units["admin_label"].isin(figures)].copy()
    for key in ("susceptibility_mean", "susceptibility_max", "n_pixels", "population"):
        units[key] = units["admin_label"].map(lambda label, k=key: figures[label].get(k))

    points = gpd.read_file(inventory).to_crs("EPSG:4326") if inventory.exists() else None
    if points is not None and len(points):
        points = points.reset_index(drop=True)
        joined = gpd.sjoin(points[["geometry"]], units[["shapeID", "admin_name", "district_name",
                                                       "susceptibility_mean", "geometry"]],
                           how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        points["unit_id"] = joined["shapeID"].values
        points["upazila"] = joined["admin_name"].values
        points["district"] = joined["district_name"].values
        points["unit_susceptibility"] = joined["susceptibility_mean"].values
        points["susceptibility"] = np.nan
        if surface.exists():
            with rasterio.open(surface) as src:
                xy = points.to_crs(src.crs).geometry
                values = np.array([v[0] for v in src.sample(zip(xy.x, xy.y))], dtype=float)
                if src.nodata is not None:
                    values[values == src.nodata] = np.nan
            points["susceptibility"] = values
        dates = pd_dates(points["event_date"])
        units["n_landslides"] = units["shapeID"].map(points["unit_id"].value_counts()).fillna(0).astype(int)
    else:
        units["n_landslides"] = 0

    # a point inside each upazila, so a search can fly to it
    inside = units.geometry.representative_point()
    # A database built before the centre points were added lacks their
    # columns; recreate the table then, and otherwise replace this region's rows.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(landslide_units)")}
    if "lat" not in columns:
        conn.execute("DROP TABLE IF EXISTS landslide_units")
        conn.executescript(SCHEMA)
    conn.execute("DELETE FROM landslide_units WHERE region = ?", (region,))
    conn.executemany(
        "INSERT INTO landslide_units VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(region, r.shapeID, r.admin_name, r.admin_label, r.district_name,
          _float(r.susceptibility_mean), _float(r.susceptibility_max),
          int(r.n_pixels or 0), int(r.population or 0), int(r.n_landslides),
          point.y, point.x)
         for r, point in zip(units.itertuples(), inside)])
    conn.execute("DELETE FROM landslides WHERE region = ?", (region,))
    n_points = 0
    if points is not None and len(points):
        rows = []
        for i, r in enumerate(points.itertuples()):
            # COOLR gives one event id to a whole inventory, so rows are numbered
            rows.append((region, i + 1, dates[i],
                         getattr(r, "event_title", None), getattr(r, "landslide_category", None),
                         getattr(r, "landslide_trigger", None), r.unit_id, r.upazila, r.district,
                         r.geometry.y, r.geometry.x, _float(r.susceptibility),
                         _float(r.unit_susceptibility)))
        conn.executemany("INSERT INTO landslides VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        n_points = len(rows)
        points["landslide_id"] = [row[1] for row in rows]
        points["event_date"] = dates

    out_dir = WEB / "tiles" / region
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_tiles = units.rename(columns={"shapeID": "unit_id", "admin_name": "name",
                                       "district_name": "district"})[
        ["unit_id", "name", "district", "susceptibility_mean", "susceptibility_max",
         "population", "n_landslides", "geometry"]]
    size = _write_pmtiles(unit_tiles, out_dir / "ls_upazilas.pmtiles", "ls_upazilas", (4, 12), 0.0005)
    if n_points:
        point_tiles = points[["landslide_id", "landslide_category", "event_date", "unit_id",
                              "upazila", "district", "susceptibility", "unit_susceptibility",
                              "geometry"]].rename(columns={"landslide_category": "category"})
        size += _write_pmtiles(point_tiles, out_dir / "ls_points.pmtiles", "ls_points", (4, 14))
    conn.commit()
    return {"units": len(units), "landslides": n_points, "tiles_mb": round(size, 1)}


def pd_dates(series) -> list:
    """Event dates as ISO days, whatever form COOLR gave them in."""
    import pandas as pd

    parsed = pd.to_datetime(series, errors="coerce", unit="ms") \
        if pd.api.types.is_numeric_dtype(series) else pd.to_datetime(series, errors="coerce")
    return [d.strftime("%Y-%m-%d") if not pd.isna(d) else None for d in parsed]


def build_region(conn: sqlite3.Connection, region: str) -> dict:
    paths = region_paths(region)
    name, hazard, config = REGION_CONFIGS[region]
    meta = _read_json(paths["output"] / "pipeline_metadata.json") or {}

    from pipeline.cli import _load_config

    cfg = _load_config(str(ROOT / config))
    dashboard_cfg = cfg.get("dashboard", {})
    centre = dashboard_cfg.get("map_center") or [23.7, 90.4]
    landslide = (paths["output"] / "landslide_susceptibility.tif").exists() or \
        (paths["cache"] / "raster_landslide.json").exists()

    counts = {
        "assets": load_assets(conn, region, paths),
        "admin": load_admin(conn, region, paths),
        "metrics": load_metrics(conn, region, paths),
        "overlays": copy_overlays(region, paths, cfg),
    }
    counts["tiles_mb"] = build_tiles(region, paths)
    if landslide:
        counts["landslide"] = build_landslide(conn, region, paths)
        display = _read_json(paths["cache"] / "raster_landslide.json") or {}
        if display:
            # the colour scale the surface was drawn with, for the legend
            conn.execute("INSERT OR REPLACE INTO metrics VALUES (?, ?, ?)",
                         (region, "landslide_display", json.dumps(
                             {"vmin": display.get("vmin"), "vmax": display.get("vmax"),
                              "colormap": "OrRd"})))

    conn.execute(
        "INSERT OR REPLACE INTO regions VALUES (?,?,?,?,?,?,?,?,?,?)",
        (region, name, hazard, config, float(centre[0]), float(centre[1]),
         int(dashboard_cfg.get("map_zoom", 8)),
         json.dumps(cfg.get("aoi", {}).get("bbox")),
         int(counts["assets"] > 0), int(landslide)))
    conn.commit()
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="*", default=None)
    args = parser.parse_args()

    regions = args.regions or [r for r in REGION_CONFIGS if has_results(r)]
    connection = _connect()
    for region_id in regions:
        result = build_region(connection, region_id)
        print(f"{region_id}: {result['assets']:,} assets, {result['admin']:,} admin rows, "
              f"{result['metrics']} metric files, {result['overlays']} overlays, "
              f"{result['tiles_mb']} MB tiles"
              + (f", {result['landslide']['units']} upazilas and "
                 f"{result['landslide']['landslides']:,} landslides"
                 if result.get("landslide") else ""))
    connection.execute("VACUUM")
    connection.close()
    print(f"\nDatabase: {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB)")
