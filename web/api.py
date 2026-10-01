"""
The web application's API: numbers from SQLite, geometry from vector tiles.

Nothing here computes anything. The pipeline produces the outputs, scripts/
build_web.py turns them into a database and tiles, and this serves them. Any
figure a visitor sees can be traced back to a file the pipeline wrote.

    uvicorn web.api:app --host 127.0.0.1 --port 2030

Responses carry long cache headers because the data only changes when the
pipeline runs again, and the build stamps a version that busts those caches.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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


# "all" stands for every region at once, the map's default view.
ALL = "all"

# The susceptibility classes the legend names, as score ranges.
CLASSES = {"low": (0.0, 0.30), "moderate": (0.30, 0.50),
           "high": (0.50, 0.70), "very_high": (0.70, 1.01)}


def region_ids(region: str) -> list[str]:
    """The regions a request covers: one, or every flood region for "all".

    "all" is the flood view's combined map, so it takes the regions that have
    scored assets; the landslide region has its own view and endpoints.
    """
    if region == ALL:
        return [row["id"] for row in rows(
            "SELECT id FROM regions WHERE has_assets = 1 ORDER BY rowid")]
    if not one("SELECT 1 AS found FROM regions WHERE id = ?", (region,)):
        raise HTTPException(404, "unknown region")
    return [region]


def placeholders(values: list) -> str:
    return ",".join("?" * len(values))


def overlays_of(region: str) -> list[str]:
    """Raster layers pre-rendered for a region, by name."""
    folder = DATA / "overlays" / region
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.json") if not p.name.startswith("._"))


@app.get("/api/regions")
def list_regions():
    """Every region the build found results for, with its map framing."""
    found = rows("SELECT id, name, hazard, centre_lat, centre_lon, zoom, bbox,"
                 " has_assets, has_landslide FROM regions ORDER BY rowid")
    for region in found:
        region["bbox"] = json.loads(region["bbox"]) if region["bbox"] else None
    return cached({"version": _version(), "regions": found})


def _metrics(region: str) -> dict:
    return {row["key"]: json.loads(row["payload"])
            for row in rows("SELECT key, payload FROM metrics WHERE region = ?", (region,))}


def _region_info(region: str) -> dict:
    info = one("SELECT * FROM regions WHERE id = ?", (region,))
    info["bbox"] = json.loads(info["bbox"]) if info["bbox"] else None
    info["overlays"] = overlays_of(region)
    return info


@app.get("/api/region/{region}/summary")
def region_summary(region: str):
    """The counters, provenance and validation figures the panels quote.

    For "all" the counts cover every region's assets, the metrics come per
    region, and the extent spans every region, the landslide one included.
    """
    ids = region_ids(region)
    marks = placeholders(ids)

    counts = one(
        "SELECT COUNT(*) AS assets, SUM(is_high_risk) AS high,"
        f" AVG(flood_risk) AS mean_risk FROM assets WHERE region IN ({marks})", tuple(ids))
    by_type = {row["asset_type"]: row["n"] for row in rows(
        f"SELECT asset_type, COUNT(*) AS n FROM assets WHERE region IN ({marks})"
        " GROUP BY asset_type", tuple(ids))}
    by_district = rows(
        f"SELECT region, division AS district, COUNT(*) AS n FROM assets"
        f" WHERE region IN ({marks}) GROUP BY region, division ORDER BY region, division",
        tuple(ids))
    # The extent of the mapped assets, so the map can open on every site
    # rather than on the configured bounding box.
    extent = one(f"SELECT MIN(lon) AS west, MIN(lat) AS south, MAX(lon) AS east,"
                 f" MAX(lat) AS north FROM assets WHERE region IN ({marks})", tuple(ids))
    frame = ([extent["west"], extent["south"], extent["east"], extent["north"]]
             if extent and extent["west"] is not None else None)
    infos = [_region_info(r) for r in ids]
    # A region without assets (the landslide one) still belongs in the frame.
    for info in infos:
        if not info["has_assets"] and info["bbox"]:
            w, s_, e, n = info["bbox"]
            frame = [w, s_, e, n] if frame is None else [
                min(frame[0], w), min(frame[1], s_), max(frame[2], e), max(frame[3], n)]

    if region == ALL:
        head = {"id": ALL, "name": "All regions", "hazard": "Flood and landslide",
                "has_assets": int(any(i["has_assets"] for i in infos)),
                "has_landslide": int(any(i["has_landslide"] for i in infos)),
                "bbox": frame, "overlays": []}
        metrics = {}
    else:
        head, metrics = infos[0], _metrics(region)

    return cached({
        "region": head,
        "regions": infos,
        "counts": {"assets": counts["assets"] or 0,
                   "high_risk": int(counts["high"] or 0),
                   "mean_risk": counts["mean_risk"],
                   "by_type": by_type,
                   "by_district": by_district},
        "extent": frame,
        "metrics": metrics,
        "metrics_by_region": {r: _metrics(r) for r in ids},
    })


@app.get("/api/stats")
def filtered_stats(regions: str = Query("", max_length=200),
                   types: str = Query("", max_length=400),
                   districts: str = Query("", max_length=2000),
                   classes: str = Query("", max_length=80),
                   min_score: float = Query(0.0, ge=0, le=1),
                   max_score: float = Query(1.0, ge=0, le=1)):
    """Counts for the assets the map's filters leave visible.

    Every list is comma-separated and an empty one means no restriction, so
    the counters beside the map always describe what the map is showing.
    """
    known = region_ids(ALL)
    chosen = [r for r in regions.split(",") if r] or known
    if any(r not in known for r in chosen):
        raise HTTPException(404, "unknown region")
    where = [f"region IN ({placeholders(chosen)})", "flood_risk >= ?", "flood_risk <= ?"]
    params: list = [*chosen, min_score, max_score]
    for column, text in (("asset_type", types), ("division", districts)):
        values = [v for v in text.split(",") if v]
        if values:
            where.append(f"{column} IN ({placeholders(values)})")
            params += values
    picked = [c for c in classes.split(",") if c]
    if any(c not in CLASSES for c in picked):
        raise HTTPException(400, "unknown class")
    if picked:
        where.append("(" + " OR ".join("(flood_risk >= ? AND flood_risk < ?)" for _ in picked) + ")")
        for c in picked:
            params += CLASSES[c]
    clause = " AND ".join(where)
    counts = one("SELECT COUNT(*) AS assets, SUM(is_high_risk) AS high,"
                 f" AVG(flood_risk) AS mean_risk FROM assets WHERE {clause}", tuple(params))
    by_type = {row["asset_type"]: row["n"] for row in rows(
        f"SELECT asset_type, COUNT(*) AS n FROM assets WHERE {clause} GROUP BY asset_type",
        tuple(params))}
    return cached({"assets": counts["assets"] or 0, "high_risk": int(counts["high"] or 0),
                   "mean_risk": counts["mean_risk"], "by_type": by_type})


@app.get("/api/region/{region}/assets")
def search_assets(region: str, q: str = Query("", max_length=80),
                  limit: int = Query(200, le=2000)):
    """Assets matching a search, or the highest-scoring ones when it is empty.

    Full-text search over name, type and division, so a visitor can look for
    "Kurigram bridge" as readily as a single word.
    """
    ids = region_ids(region)
    marks = placeholders(ids)
    if q.strip():
        # FTS5 prefix search, one term at a time, with the query escaped: a
        # visitor's text is data, never syntax. Quotes inside a term are
        # doubled, as FTS5 string syntax requires.
        terms = " ".join('"' + term.replace('"', '""') + '"*' for term in q.split() if term)
        found = rows(
            "SELECT a.* FROM assets_fts f JOIN assets a"
            "  ON a.region = f.region AND a.asset_id = f.asset_id"
            f" WHERE assets_fts MATCH ? AND f.region IN ({marks})"
            " ORDER BY a.flood_risk DESC LIMIT ?", (terms, *ids, limit))
    else:
        # Ranks are per region, so across regions the score orders the list.
        order = "risk_rank" if len(ids) == 1 else "flood_risk DESC"
        found = rows(f"SELECT * FROM assets WHERE region IN ({marks})"
                     f" ORDER BY {order} LIMIT ?", (*ids, limit))
    return cached({"count": len(found), "assets": found})


@app.get("/api/region/{region}/asset/{asset_id}")
def asset_detail(region: str, asset_id: int):
    """One asset's own row, as the pipeline wrote it."""
    asset = one("SELECT * FROM assets WHERE region = ? AND asset_id = ?",
                (region, asset_id))
    if not asset:
        raise HTTPException(404, "unknown asset")
    return cached(asset)


@app.get("/api/region/{region}/rank/{rank}")
def asset_by_rank(region: str, rank: int):
    """One asset by its rank in its region, which is what a map tile carries."""
    asset = one("SELECT * FROM assets WHERE region = ? AND risk_rank = ?", (region, rank))
    if not asset:
        raise HTTPException(404, "unknown asset")
    return cached(asset)


@app.get("/api/region/{region}/admin")
def admin_summary(region: str, level: str = Query("union"),
                  limit: int = Query(60, le=500)):
    """Administrative summaries, highest mean risk first."""
    ids = region_ids(region)
    found = rows(
        "SELECT region, unit_id, name, parent, mean_risk, max_risk, mean_risk_people,"
        "       n_assets, has_data FROM admin_summary"
        f" WHERE region IN ({placeholders(ids)}) AND level = ? AND has_data = 1"
        " ORDER BY mean_risk DESC LIMIT ?", (*ids, level, limit))
    return cached({"level": level, "units": found})


@app.get("/api/region/{region}/export.csv")
def export_csv(region: str, limit: int = Query(5000, le=100000)):
    """The ranked assets, as the dashboard's download button provided them."""
    import csv
    import io

    ids = region_ids(region)
    order = "risk_rank" if len(ids) == 1 else "flood_risk DESC"
    found = rows("SELECT region, risk_rank, name, asset_type, division, lat, lon,"
                 " flood_risk, flood_probability FROM assets"
                 f" WHERE region IN ({placeholders(ids)}) ORDER BY {order} LIMIT ?",
                 (*ids, limit))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(found[0].keys()) if found else [])
    writer.writeheader()
    writer.writerows(found)
    return Response(
        buffer.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{region}_assets.csv"'})


# ── Landslide view ──────────────────────────────────────────────────────

def landslide_region(region: str) -> str:
    if not one("SELECT 1 AS found FROM regions WHERE id = ? AND has_landslide = 1", (region,)):
        raise HTTPException(404, "no landslide model for this region")
    return region


def _unit_filter(region: str, districts: str, upazilas: str, min_s: float, max_s: float):
    where = ["region = ?", "susceptibility_mean >= ?", "susceptibility_mean <= ?"]
    params: list = [region, min_s, max_s]
    for column, text in (("district", districts), ("unit_id", upazilas)):
        values = [v for v in text.split(",") if v]
        if values:
            where.append(f"{column} IN ({placeholders(values)})")
            params += values
    return " AND ".join(where), params


@app.get("/api/landslide/{region}/summary")
def landslide_summary(region: str):
    """Upazilas, landslide categories and the model, for the landslide view."""
    landslide_region(region)
    info = _region_info(region)
    units = rows("SELECT unit_id, name, district, susceptibility_mean, susceptibility_max,"
                 " population, n_landslides, lat, lon FROM landslide_units WHERE region = ?"
                 " ORDER BY district, name", (region,))
    categories = {row["category"] or "unknown": row["n"] for row in rows(
        "SELECT category, COUNT(*) AS n FROM landslides WHERE region = ? GROUP BY category",
        (region,))}
    extent = one("SELECT MIN(lon) AS west, MIN(lat) AS south, MAX(lon) AS east,"
                 " MAX(lat) AS north FROM landslides WHERE region = ?", (region,))
    metrics = _metrics(region)
    return cached({"region": info, "units": units, "categories": categories,
                   "extent": info["bbox"] or ([extent["west"], extent["south"],
                                               extent["east"], extent["north"]]
                                              if extent and extent["west"] is not None else None),
                   "model": metrics.get("landslide_model", {}),
                   "display": metrics.get("landslide_display", {})})


@app.get("/api/landslide/{region}/stats")
def landslide_stats(region: str,
                    districts: str = Query("", max_length=400),
                    upazilas: str = Query("", max_length=4000),
                    categories: str = Query("", max_length=400),
                    min_score: float = Query(0.0, ge=0, le=1),
                    max_score: float = Query(1.0, ge=0, le=1)):
    """Counts for the upazilas and landslides the filters leave visible.

    Mean susceptibility is weighted by each upazila's area in pixels, so a
    large upazila counts for its size rather than as one unit.
    """
    landslide_region(region)
    clause, params = _unit_filter(region, districts, upazilas, min_score, max_score)
    units = one("SELECT COUNT(*) AS n, SUM(population) AS people,"
                " SUM(susceptibility_mean * n_pixels) / NULLIF(SUM(n_pixels), 0) AS mean"
                f" FROM landslide_units WHERE {clause}", tuple(params))
    picked = [c for c in categories.split(",") if c]
    point_clause = (f"region = ? AND unit_id IN (SELECT unit_id FROM landslide_units"
                    f" WHERE {clause})")
    point_params = [region, *params]
    if picked:
        point_clause += f" AND category IN ({placeholders(picked)})"
        point_params += picked
    points = one(f"SELECT COUNT(*) AS n, AVG(susceptibility) AS at_points FROM landslides"
                 f" WHERE {point_clause}", tuple(point_params))
    top = one(f"SELECT name, district, susceptibility_mean FROM landslide_units WHERE {clause}"
              " ORDER BY susceptibility_mean DESC LIMIT 1", tuple(params))
    return cached({"upazilas": units["n"] or 0, "population": int(units["people"] or 0),
                   "mean_susceptibility": units["mean"], "landslides": points["n"] or 0,
                   "susceptibility_at_landslides": points["at_points"], "highest": top})


@app.get("/api/landslide/{region}/upazilas.csv")
def landslide_csv(region: str):
    """Every upazila's landslide figures, highest mean susceptibility first."""
    import csv
    import io

    landslide_region(region)
    found = rows("SELECT name, district, susceptibility_mean, susceptibility_max,"
                 " population, n_landslides FROM landslide_units WHERE region = ?"
                 " ORDER BY susceptibility_mean DESC", (region,))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(found[0].keys()) if found else [])
    writer.writeheader()
    writer.writerows(found)
    return Response(buffer.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{region}_landslide_upazilas.csv"'})


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


# The page names its script and stylesheet with a hash of their contents, so
# every change gets a URL no cache has seen: Cloudflare keeps .js and .css at
# its edge, and a page served with the previous script would break.
STATIC = ROOT / "static"


def _fingerprint(name: str) -> str:
    return hashlib.sha256((STATIC / name).read_bytes()).hexdigest()[:12]


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index():
    html = (STATIC / "index.html").read_text()
    for name in ("app.js", "app.css"):
        html = html.replace(f'"{name}"', f'"{name}?v={_fingerprint(name)}"')
    # The page itself is always revalidated, so a new fingerprint reaches
    # visitors on their next load.
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


# The front end is static files; the API above is mounted first so it wins.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
