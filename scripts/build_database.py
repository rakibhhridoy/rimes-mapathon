"""
One SQLite database holding everything the pipeline reads and writes.

    python scripts/build_database.py                           # -> dist/hazmapper.sqlite
    python scripts/build_database.py --extract-raster sylhet_dem_srtm_30m dem.tif

The database is SpatiaLite, which is SQLite with spatial metadata, so any
SQLite client reads its tables and QGIS opens the geometry tables as layers.
It is an export: the pipeline still works on files, and this script collects
them into one place for sharing and querying.

Every item is stored in the form a GIS or a query can use:

    vector layers (GeoPackage, GeoJSON)   SpatiaLite geometry tables, one per
                                          region and file, spatially indexed
    tables (CSV, Parquet)                 plain tables
    rasters (GeoTIFF)                     the `rasters` table: one row per raster,
                                          its grid described in columns and the
                                          grid itself as a compressed GeoTIFF
    metrics (JSON)                        the `metrics` table, one row per file,
                                          queryable with SQLite's JSON functions
    arrays (.npy, .npz)                   the `arrays` table, with dtype and shape
    anything else (models, reports)       the `files` table, as bytes

SpatiaLite's own raster support (RasterLite2) is no longer maintained, so a
raster is kept as GeoTIFF bytes, recompressed losslessly, which GDAL and
rasterio read straight from memory and `--extract-raster` writes back out.

Tables are named `<region>_<file>`, with the shared national layers under
`shared_`. The `sources` table records, for every item, the file it came
from, that file's SHA-256 and its licence. GADM files are left out because
GADM's licence forbids redistribution, and the dashboard caches are left out
because they are copies of the outputs.
"""

import argparse
import io
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_archive import _excluded, _licence_for, _sha256  # noqa: E402

VECTOR = {".gpkg", ".geojson"}
TABLE = {".csv", ".parquet"}
RASTER = {".tif", ".tiff"}
ARRAY = {".npy", ".npz"}
METRICS = {".json"}
# Sidecars of formats handled above, or of files that are never included.
SKIP = {".cpg", ".dbf", ".prj", ".shp", ".shx", ".aux", ".xml"}

SHARED_LICENCES = {
    "geoboundaries": "geoBoundaries gbOpen, CC BY 4.0",
    "naturalearth": "Natural Earth, public domain",
    "jrc": "EC JRC Global Surface Water, free with attribution",
    "worldpop": "WorldPop, CC BY 4.0",
}

SCHEMA = """
CREATE TABLE rasters (id INTEGER PRIMARY KEY, name TEXT UNIQUE, region TEXT,
                      epsg INTEGER, crs_wkt TEXT, width INTEGER, height INTEGER,
                      bands INTEGER, dtype TEXT, nodata REAL,
                      transform TEXT, west REAL, south REAL, east REAL, north REAL,
                      geotiff BLOB);
CREATE TABLE metrics (id INTEGER PRIMARY KEY, region TEXT, kind TEXT, name TEXT,
                      json TEXT);
CREATE TABLE arrays (id INTEGER PRIMARY KEY, region TEXT, name TEXT, key TEXT,
                     dtype TEXT, shape TEXT, npy BLOB);
CREATE TABLE files (id INTEGER PRIMARY KEY, region TEXT, path TEXT, name TEXT,
                    bytes BLOB);
CREATE TABLE sources (id INTEGER PRIMARY KEY, table_name TEXT, row_key TEXT,
                      region TEXT, source_path TEXT, sha256 TEXT, licence TEXT);
"""


def collect() -> list[tuple[str, str, Path]]:
    """(region, kind, path) for every file the database should hold."""
    from dashboard.data.regions import REGION_CONFIGS, region_paths

    entries = []
    for region in REGION_CONFIGS:
        paths = region_paths(region)
        for kind in ("raw", "processed", "output"):
            base = paths[kind]
            if base.exists():
                entries += [(region, kind, p) for p in sorted(base.rglob("*"))
                            if p.is_file() and not _excluded(p)]
    shared = ROOT / "data" / "shared"
    if shared.exists():
        entries += [("shared", p.parent.name, p) for p in sorted(shared.rglob("*"))
                    if p.is_file() and not _excluded(p)]
    return entries


def table_name(region: str, path: Path, layer: str | None = None,
               taken: set | None = None, kind: str = "") -> str:
    """A lowercase SQL-safe name, unique within the database.

    Map layers are built first and keep the plain name, so a CSV written
    beside a GeoJSON of the same results becomes `<name>_table`.
    """
    stem = path.stem if not layer or layer == path.stem else f"{path.stem}_{layer}"
    safe = lambda text: re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")
    name = safe(f"{region}_{stem}")
    if taken is not None and name in taken and path.suffix.lower() in TABLE:
        name = safe(f"{region}_{stem}_table")
    if taken is not None and name in taken:
        name = safe(f"{region}_{kind}_{stem}")
    if taken is not None:
        taken.add(name)
    return name


def _order(entry: tuple[str, str, Path]) -> int:
    """Map layers first, so they claim the plain table names."""
    return 0 if entry[2].suffix.lower() in VECTOR else 1


def licence(region: str, kind: str, path: Path) -> str:
    if region == "shared":
        return SHARED_LICENCES.get(kind, "see README")
    return _licence_for(path)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command[:3])} failed: {result.stderr.strip()[:500]}")


def _layers(path: Path) -> list[str]:
    import pyogrio
    return [str(name) for name, _ in pyogrio.list_layers(path)]


def _raster_row(path: Path, name: str, region: str) -> tuple:
    """The raster's grid description and its bytes as a compressed GeoTIFF.

    Recompressed with DEFLATE and a predictor, which is lossless, because some
    pipeline rasters are written uncompressed and would bloat the database.
    """
    import rasterio

    with rasterio.open(path) as src:
        predictor = "3" if src.dtypes[0].startswith("float") else "2"
        meta = (src.crs.to_epsg() if src.crs else None,
                src.crs.to_wkt() if src.crs else None,
                src.width, src.height, src.count, src.dtypes[0], src.nodata,
                json.dumps(list(src.transform)[:6]), *src.bounds)
    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp) / "packed.tif"
        _run(["gdal_translate", "-q", "-of", "GTiff", str(path), str(packed),
              "-co", "COMPRESS=DEFLATE", "-co", f"PREDICTOR={predictor}",
              "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER"])
        return (name, region, *meta, packed.read_bytes())


def build(db: Path, entries: list[tuple[str, str, Path]], log=print) -> dict:
    """Write every entry into the SpatiaLite database at `db`, replacing any old one."""
    import numpy as np

    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    # The spatial metadata comes from GDAL, which creates the database.
    _run(["ogr2ogr", "-f", "SQLite", "-dsco", "SPATIALITE=YES", str(db),
          str(entries_vector_seed()), "-nln", "_seed"])
    _run(["ogrinfo", "-q", str(db), "-sql", "DELLAYER:_seed"])
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.commit()

    taken: set = set()
    counts = {"vector": 0, "table": 0, "raster": 0, "metrics": 0, "arrays": 0, "files": 0}

    def source(table, key, region, rel, sha, lic):
        conn.execute("INSERT INTO sources (table_name, row_key, region, source_path,"
                     " sha256, licence) VALUES (?,?,?,?,?,?)",
                     (table, key, region, rel, sha, lic))

    for region, kind, path in sorted(entries, key=_order):
        suffix = path.suffix.lower()
        if suffix in SKIP:
            continue
        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        sha, lic = _sha256(path), licence(region, kind, path)

        if suffix in VECTOR or suffix in TABLE:
            layers = _layers(path) if suffix in VECTOR else [None]
            for layer in layers:
                name = table_name(region, path, layer, taken, kind)
                conn.commit()            # GDAL writes to the same file next
                command = ["ogr2ogr", "-f", "SQLite", "-update", str(db), str(path)]
                if layer:
                    command.append(layer)
                command += ["-nln", name]
                if suffix in VECTOR:
                    command += ["-lco", "SPATIAL_INDEX=YES", "-lco", "FORMAT=SPATIALITE"]
                if suffix == ".csv":
                    command += ["-oo", "AUTODETECT_TYPE=YES"]
                _run(command)
                source(name, None, region, rel, sha, lic)
                counts["vector" if suffix in VECTOR else "table"] += 1
        elif suffix in RASTER:
            name = table_name(region, path, None, taken, kind)
            conn.execute("INSERT INTO rasters (name, region, epsg, crs_wkt, width, height,"
                         " bands, dtype, nodata, transform, west, south, east, north,"
                         " geotiff) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         _raster_row(path, name, region))
            source("rasters", name, region, rel, sha, lic)
            counts["raster"] += 1
        elif suffix in METRICS:
            text = path.read_text()
            try:
                json.loads(text)
            except ValueError:            # not JSON after all: keep the bytes
                conn.execute("INSERT INTO files (region, path, name, bytes) VALUES (?,?,?,?)",
                             (region, rel, path.name, path.read_bytes()))
                source("files", rel, region, rel, sha, lic)
                counts["files"] += 1
            else:
                conn.execute("INSERT INTO metrics (region, kind, name, json) VALUES (?,?,?,?)",
                             (region, kind, path.stem, text))
                source("metrics", f"{region}/{path.stem}", region, rel, sha, lic)
                counts["metrics"] += 1
        elif suffix in ARRAY:
            loaded = np.load(path, allow_pickle=False)
            items = loaded.items() if suffix == ".npz" else [("", loaded)]
            for key, value in items:
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=False)
                conn.execute("INSERT INTO arrays (region, name, key, dtype, shape, npy)"
                             " VALUES (?,?,?,?,?,?)",
                             (region, path.stem, key, str(value.dtype),
                              json.dumps(list(value.shape)), buffer.getvalue()))
                counts["arrays"] += 1
            source("arrays", f"{region}/{path.stem}", region, rel, sha, lic)
        else:
            conn.execute("INSERT INTO files (region, path, name, bytes) VALUES (?,?,?,?)",
                         (region, rel, path.name, path.read_bytes()))
            source("files", rel, region, rel, sha, lic)
            counts["files"] += 1
        conn.commit()
        log(f"  {rel}")

    conn.close()
    return counts


def entries_vector_seed() -> Path:
    """A one-point GeoJSON that lets GDAL create the SpatiaLite metadata."""
    seed = Path(tempfile.gettempdir()) / "hazmapper_seed.geojson"
    seed.write_text('{"type":"FeatureCollection","features":[{"type":"Feature",'
                    '"properties":{},"geometry":{"type":"Point","coordinates":[0,0]}}]}')
    return seed


def extract_raster(db: Path, name: str, out: Path) -> None:
    """Write one stored raster back out as a GeoTIFF."""
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT geotiff FROM rasters WHERE name = ?", (name,)).fetchone()
    conn.close()
    if row is None:
        raise SystemExit(f"no raster named {name}")
    out.write_bytes(row[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(ROOT / "dist" / "hazmapper.sqlite"))
    parser.add_argument("--extract-raster", nargs=2, metavar=("NAME", "OUT_TIF"),
                        help="write one raster from the database to a GeoTIFF")
    args = parser.parse_args()

    if args.extract_raster:
        extract_raster(Path(args.out), args.extract_raster[0], Path(args.extract_raster[1]))
        return
    entries = collect()
    print(f"Collecting {len(entries)} files into {args.out}")
    counts = build(Path(args.out), entries)
    size = Path(args.out).stat().st_size / 1e9
    print(", ".join(f"{n} {k}" for k, n in counts.items()) + f"; {size:.2f} GB")


if __name__ == "__main__":
    main()
