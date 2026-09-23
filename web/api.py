"""
The web application's API: numbers from SQLite, geometry from vector tiles.

Nothing here computes anything. The pipeline produces the outputs, scripts/
build_web.py turns them into a database and tiles, and this serves them. Any
figure a visitor sees can be traced back to a file the pipeline wrote.

    uvicorn web.api:app --host 127.0.0.1 --port 2030

Responses carry long cache headers because the data only changes when the
pipeline runs again, and the build stamps a version that busts those caches.
"""

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB_PATH = DATA / "hazmapper.sqlite"

# The database changes only when the pipeline and build script run, so its
# modification time is a version that invalidates every cached response.
def _version() -> str:
    return str(int(DB_PATH.stat().st_mtime))


CACHE_HEADERS = {"Cache-Control": "public, max-age=3600, stale-while-revalidate=86400"}
IMMUTABLE_HEADERS = {"Cache-Control": "public, max-age=604800, immutable"}

app = FastAPI(title="Fermium Hazard Mapper", docs_url="/api/docs",
              openapi_url="/api/openapi.json")


def connect() -> sqlite3.Connection:
    """Read-only connection; the API never writes."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def rows(query: str, params: tuple = ()) -> list[dict]:
    with connect() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def one(query: str, params: tuple = ()) -> dict | None:
    result = rows(query, params)
    return result[0] if result else None


def cached(payload) -> JSONResponse:
    return JSONResponse(payload, headers=CACHE_HEADERS)


@app.get("/api/regions")
def list_regions():
    """Every region the build found results for, with its map framing."""
    found = rows("SELECT id, name, hazard, centre_lat, centre_lon, zoom, bbox,"
                 " has_assets, has_landslide FROM regions ORDER BY rowid")
    for region in found:
        region["bbox"] = json.loads(region["bbox"]) if region["bbox"] else None
    return cached({"version": _version(), "regions": found})


@app.get("/api/region/{region}/summary")
def region_summary(region: str):
    """The counters, provenance and validation figures the panels quote."""
    info = one("SELECT * FROM regions WHERE id = ?", (region,))
    if not info:
        raise HTTPException(404, "unknown region")

    metrics = {row["key"]: json.loads(row["payload"])
               for row in rows("SELECT key, payload FROM metrics WHERE region = ?",
                               (region,))}
    counts = one(
        "SELECT COUNT(*) AS assets, SUM(is_high_risk) AS high,"
        " AVG(flood_risk) AS mean_risk FROM assets WHERE region = ?", (region,))
    by_type = {row["asset_type"]: row["n"] for row in rows(
        "SELECT asset_type, COUNT(*) AS n FROM assets WHERE region = ?"
        " GROUP BY asset_type", (region,))}

    return cached({
        "region": dict(info) | {"bbox": json.loads(info["bbox"]) if info["bbox"] else None},
        "counts": {"assets": counts["assets"] or 0,
                   "high_risk": int(counts["high"] or 0),
                   "mean_risk": counts["mean_risk"],
                   "by_type": by_type},
        "metrics": metrics,
    })


@app.get("/api/region/{region}/assets")
def search_assets(region: str, q: str = Query("", max_length=80),
                  limit: int = Query(200, le=2000)):
    """Assets matching a search, or the highest-scoring ones when it is empty.

    Full-text search over name, type and division, so a visitor can look for
    "Kurigram bridge" as readily as a single word.
    """
    if q.strip():
        # FTS5 prefix search, one term at a time, with the query escaped: a
        # visitor's text is data, never syntax.
        terms = " ".join(f'"{term}"*' for term in q.split() if term)
        found = rows(
            "SELECT a.* FROM assets_fts f JOIN assets a"
            "  ON a.region = f.region AND a.asset_id = f.asset_id"
            " WHERE assets_fts MATCH ? AND f.region = ?"
            " ORDER BY a.flood_risk DESC LIMIT ?", (terms, region, limit))
    else:
        found = rows("SELECT * FROM assets WHERE region = ?"
                     " ORDER BY risk_rank LIMIT ?", (region, limit))
    return cached({"count": len(found), "assets": found})


@app.get("/api/region/{region}/asset/{asset_id}")
def asset_detail(region: str, asset_id: int):
    """One asset's own row, as the pipeline wrote it."""
    asset = one("SELECT * FROM assets WHERE region = ? AND asset_id = ?",
                (region, asset_id))
    if not asset:
        raise HTTPException(404, "unknown asset")
    return cached(asset)


@app.get("/api/region/{region}/admin")
def admin_summary(region: str, level: str = Query("union"),
                  limit: int = Query(60, le=500)):
    """Administrative summaries, highest mean risk first."""
    found = rows(
        "SELECT unit_id, name, parent, mean_risk, max_risk, mean_risk_people,"
        "       n_assets, has_data FROM admin_summary"
        " WHERE region = ? AND level = ? AND has_data = 1"
        " ORDER BY mean_risk DESC LIMIT ?", (region, level, limit))
    return cached({"level": level, "units": found})


@app.get("/api/region/{region}/export.csv")
def export_csv(region: str, limit: int = Query(5000, le=100000)):
    """The ranked assets, as the dashboard's download button provided them."""
    import csv
    import io

    found = rows("SELECT risk_rank, name, asset_type, division, lat, lon,"
                 " flood_risk, flood_probability FROM assets"
                 " WHERE region = ? ORDER BY risk_rank LIMIT ?", (region, limit))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(found[0].keys()) if found else [])
    writer.writeheader()
    writer.writerows(found)
    return Response(
        buffer.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{region}_assets.csv"'})


@app.get("/tiles/{region}/{layer}.pmtiles")
def tiles(region: str, layer: str):
    """Vector tiles. The browser reads ranges out of these files directly."""
    path = DATA / "tiles" / region / f"{layer}.pmtiles"
    if not path.is_file() or ".." in region or ".." in layer:
        raise HTTPException(404, "no such tile set")
    return FileResponse(path, media_type="application/octet-stream",
                        headers=IMMUTABLE_HEADERS | {"Accept-Ranges": "bytes"})


@app.get("/overlays/{region}/{name}")
def overlay(region: str, name: str):
    """Raster layers the pipeline pre-rendered, as PNG plus bounds."""
    path = DATA / "overlays" / region / name
    if not path.is_file() or ".." in region or ".." in name:
        raise HTTPException(404, "no such overlay")
    media = "image/png" if path.suffix == ".png" else "application/json"
    return FileResponse(path, media_type=media, headers=IMMUTABLE_HEADERS)


@app.get("/api/health")
def health():
    """Whether the database is present and how many regions it holds."""
    if not DB_PATH.exists():
        raise HTTPException(503, "database not built; run scripts/build_web.py")
    return {"status": "ok", "version": _version(),
            "regions": one("SELECT COUNT(*) AS n FROM regions")["n"]}


# The front end is static files; the API above is mounted first so it wins.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
