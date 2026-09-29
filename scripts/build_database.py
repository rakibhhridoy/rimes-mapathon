"""
One SQLite database holding everything the pipeline reads and writes.

    python scripts/build_database.py                 # -> dist/hazmapper.gpkg

The database is a GeoPackage, which is SQLite with a standard layout for
geometry and rasters, so QGIS, ArcGIS, GDAL and any SQLite client open it
directly. It is an export: the pipeline still works on files, and this
script collects them into one place for sharing and querying.

Every item is stored in the form a GIS or a query can use:

    vector layers (GeoPackage, GeoJSON)   geometry tables, one per region and file
    tables (CSV, Parquet)                 attribute tables
    rasters (GeoTIFF)                     raster tables; Float32 and Int16 grids
                                          as GeoPackage gridded coverages
    metrics (JSON)                        the `metrics` table, one row per file,
                                          queryable with SQLite's JSON functions
    arrays (.npy, .npz)                   the `arrays` table, with dtype and shape
    anything else (models, reports)       the `files` table, as bytes

Tables are named `<region>_<file>`, with the shared national layers under
`shared_`. The `sources` table records, for every table and row, the file it
came from, that file's SHA-256 and its licence. GADM files are left out
because GADM's licence forbids redistribution, and the dashboard caches are
left out because they are copies of the outputs.
"""

import argparse
import io
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
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
    """A lowercase SQL-safe name, unique within the database."""
    stem = path.stem if not layer or layer == path.stem else f"{path.stem}_{layer}"
    name = re.sub(r"[^a-z0-9_]+", "_", f"{region}_{stem}".lower()).strip("_")
    if taken is not None and name in taken:
        name = re.sub(r"[^a-z0-9_]+", "_", f"{region}_{kind}_{stem}".lower()).strip("_")
    if taken is not None:
        taken.add(name)
    return name


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


def _register(conn: sqlite3.Connection, table: str, description: str) -> None:
    """List a plain SQLite table in gpkg_contents so GIS software shows it."""
    conn.execute(
        "INSERT OR REPLACE INTO gpkg_contents (table_name, data_type, identifier,"
        " description, last_change) VALUES (?, 'attributes', ?, ?, ?)",
        (table, table, description,
         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")))


def build(db: Path, entries: list[tuple[str, str, Path]], log=print) -> dict:
    """Write every entry into the GeoPackage at `db`, replacing any old one."""
    import numpy as np

    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    taken: set = set()
    sources: list[tuple] = []          # table, row key, region, source, sha256, licence
    metrics, arrays, blobs = [], [], []
    counts = {"vector": 0, "table": 0, "raster": 0, "metrics": 0, "arrays": 0, "files": 0}

    def exists() -> list[str]:
        return ["-update"] if db.exists() else []

    for region, kind, path in entries:
        suffix = path.suffix.lower()
        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        sha, lic = _sha256(path), licence(region, kind, path)
        if suffix in SKIP:
            continue

        if suffix in VECTOR:
            for layer in _layers(path):
                name = table_name(region, path, layer, taken, kind)
                _run(["ogr2ogr", "-f", "GPKG", *exists(), str(db), str(path), layer,
                      "-nln", name, "-lco", "SPATIAL_INDEX=YES"])
                sources.append((name, None, region, rel, sha, lic))
                counts["vector"] += 1
        elif suffix in TABLE:
            name = table_name(region, path, None, taken, kind)
            _run(["ogr2ogr", "-f", "GPKG", *exists(), str(db), str(path),
                  "-nln", name, "-oo", "AUTODETECT_TYPE=YES"]
                 if suffix == ".csv" else
                 ["ogr2ogr", "-f", "GPKG", *exists(), str(db), str(path), "-nln", name])
            sources.append((name, None, region, rel, sha, lic))
            counts["table"] += 1
        elif suffix in RASTER:
            name = table_name(region, path, None, taken, kind)
            _run(["gdal_translate", "-q", "-of", "GPKG", str(path), str(db),
                  "-co", "APPEND_SUBDATASET=YES", "-co", f"RASTER_TABLE={name}",
                  "-co", f"RASTER_DESCRIPTION={rel}"])
            sources.append((name, None, region, rel, sha, lic))
            counts["raster"] += 1
        elif suffix in METRICS:
            text = path.read_text()
            try:
                json.loads(text)
            except ValueError:            # not JSON after all: keep the bytes
                blobs.append((region, rel, path.name, path.read_bytes()))
                sources.append(("files", rel, region, rel, sha, lic))
                counts["files"] += 1
            else:
                metrics.append((region, kind, path.stem, text))
                sources.append(("metrics", f"{region}/{path.stem}", region, rel, sha, lic))
                counts["metrics"] += 1
        elif suffix in ARRAY:
            loaded = np.load(path, allow_pickle=False)
            items = loaded.items() if suffix == ".npz" else [("", loaded)]
            for key, value in items:
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=False)
                arrays.append((region, path.stem, key, str(value.dtype),
                               json.dumps(list(value.shape)), buffer.getvalue()))
                counts["arrays"] += 1
            sources.append(("arrays", f"{region}/{path.stem}", region, rel, sha, lic))
        else:
            blobs.append((region, rel, path.name, path.read_bytes()))
            sources.append(("files", rel, region, rel, sha, lic))
            counts["files"] += 1
        log(f"  {rel}")

    conn = sqlite3.connect(db)
    with conn:
        conn.executescript("""
            CREATE TABLE metrics (id INTEGER PRIMARY KEY, region TEXT, kind TEXT,
                                  name TEXT, json TEXT);
            CREATE TABLE arrays (id INTEGER PRIMARY KEY, region TEXT, name TEXT,
                                 key TEXT, dtype TEXT, shape TEXT, npy BLOB);
            CREATE TABLE files (id INTEGER PRIMARY KEY, region TEXT, path TEXT,
                                name TEXT, bytes BLOB);
            CREATE TABLE sources (id INTEGER PRIMARY KEY, table_name TEXT, row_key TEXT,
                                  region TEXT, source_path TEXT, sha256 TEXT,
                                  licence TEXT);
        """)
        conn.executemany("INSERT INTO metrics (region, kind, name, json) VALUES (?,?,?,?)",
                         metrics)
        conn.executemany("INSERT INTO arrays (region, name, key, dtype, shape, npy)"
                         " VALUES (?,?,?,?,?,?)", arrays)
        conn.executemany("INSERT INTO files (region, path, name, bytes) VALUES (?,?,?,?)",
                         blobs)
        conn.executemany("INSERT INTO sources (table_name, row_key, region, source_path,"
                         " sha256, licence) VALUES (?,?,?,?,?,?)", sources)
        _register(conn, "metrics", "Pipeline metrics, one JSON document per file")
        _register(conn, "arrays", "NumPy arrays saved in .npy format, with dtype and shape")
        _register(conn, "files", "Other pipeline files (models, reports), as bytes")
        _register(conn, "sources", "Source file, SHA-256 and licence of every item")
    conn.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(ROOT / "dist" / "hazmapper.gpkg"))
    args = parser.parse_args()

    entries = collect()
    print(f"Collecting {len(entries)} files into {args.out}")
    counts = build(Path(args.out), entries)
    size = Path(args.out).stat().st_size / 1e9
    print(", ".join(f"{n} {k}" for k, n in counts.items()) + f"; {size:.2f} GB")


if __name__ == "__main__":
    main()
