"""
Landslide susceptibility for the Chittagong Hill Tracts, fitted to an
observed inventory.

The earlier version applied a hand-picked logistic curve to slope
(centred on 20 degrees, scale 5) that was never fitted to anything, and when
upazila boundaries were missing it sliced the raster into horizontal bands,
labelled them with real upazila names and invented their populations with a
random number generator. None of that survives here.

Instead: mapped landslide locations come from NASA's Cooperative Open Online
Landslide Repository (COOLR), a logistic regression is fitted to terrain at
those locations against randomly sampled background points, and the fit is
scored on spatially held-out blocks so the reported skill is not inflated by
the clustering of landslides.

Reference:
    Juang, C. S., Stanley, T. A., & Kirschbaum, D. B. (2019). Using citizen
    science to expand the global map of landslides: Introducing the
    Cooperative Open Online Landslide Repository (COOLR). PLOS ONE, 14(7),
    e0218657. https://doi.org/10.1371/journal.pone.0218657
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# COOLR moved to NASA's Earthdata GIS portal; the events layer holds the
# satellite-mapped inventories, the reports layer citizen and media reports.
COOLR_EVENTS = (
    "https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/"
    "COOLR_Events_Points/FeatureServer/0/query"
)
COOLR_PAGE_SIZE = 2000

CITATION = (
    "Juang, C.S., Stanley, T.A., & Kirschbaum, D.B. (2019). Using citizen "
    "science to expand the global map of landslides: Introducing the "
    "Cooperative Open Online Landslide Repository (COOLR). PLOS ONE 14(7): "
    "e0218657. doi:10.1371/journal.pone.0218657"
)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def download_landslide_inventory(cfg: dict, raw_dir: Path) -> Path:
    """Download COOLR landslide points over the study area.

    Returns the path to a GeoJSON of mapped landslide locations.
    """
    import geopandas as gpd
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "coolr_landslides.geojson"
    if out_path.exists():
        logger.info(f"Landslide inventory already present: {out_path}")
        return out_path

    west, south, east, north = cfg["aoi"]["bbox"]
    envelope = json.dumps({
        "xmin": west, "ymin": south, "xmax": east, "ymax": north,
        "spatialReference": {"wkid": 4326},
    })

    features, offset = [], 0
    while True:
        params = {
            "where": "1=1",
            "geometry": envelope,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join([
                "event_id", "event_date", "event_title", "landslide_category",
                "landslide_trigger", "method", "source_name", "source_link",
                "country_name",
            ]),
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": COOLR_PAGE_SIZE,
        }
        resp = requests.get(COOLR_EVENTS, params=params, timeout=300,
                            headers={"User-Agent": "fermium-hazmapper"})
        resp.raise_for_status()
        page = resp.json()
        batch = page.get("features", [])
        features.extend(batch)
        logger.info(f"COOLR: fetched {len(features)} points")
        if len(batch) < COOLR_PAGE_SIZE:
            break
        offset += COOLR_PAGE_SIZE

    if not features:
        raise RuntimeError(
            "COOLR returned no landslide points for this area — a model cannot "
            "be fitted without an inventory."
        )

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf.to_file(out_path, driver="GeoJSON")
    (raw_dir / "coolr_citation.txt").write_text(CITATION + "\n")
    logger.info(f"Landslide inventory: {len(gdf)} points → {out_path}")
    return out_path


def mapped_domain(inventory, buffer_m: float, crs: str):
    """The area an inventory actually covers, as a WGS84 polygon.

    COOLR inventories are mapped campaign by campaign: the CHT set comes from
    one storm and covers part of the region. Sampling background points over
    the whole bounding box would put "never mapped" ground in the negative
    class, and the model would learn the mapping footprint — east versus
    west — instead of terrain. So the negatives are drawn only from the
    convex hull of the inventory, buffered outwards.
    """
    import geopandas as gpd

    hull = gpd.GeoSeries([inventory.unary_union], crs="EPSG:4326")
    hull_m = hull.to_crs(crs).convex_hull.buffer(buffer_m)
    return hull_m.to_crs("EPSG:4326").iloc[0]


def sample_background_points(bbox, n: int, inventory, exclusion_m: float,
                             crs: str, seed: int = 42, domain=None):
    """Random points for the negative class, away from mapped landslides.

    Susceptibility models need locations where landslides were not recorded.
    Points near an inventory point are dropped, because "not recorded" there
    may only mean the mapping stopped at the scarp edge. Draws are confined
    to `domain` when given (see mapped_domain).
    """
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import Point
    from shapely.prepared import prep

    from pipeline.feature_extract import project_coords

    if domain is not None:
        west, south, east, north = domain.bounds
        inside = prep(domain)
    else:
        west, south, east, north = bbox
        inside = None
    rng = np.random.default_rng(seed)

    inv_m = project_coords(
        np.column_stack([inventory.geometry.x, inventory.geometry.y]), crs
    )
    tree = cKDTree(inv_m)

    kept: list[Point] = []
    # Oversample and filter; landslide points are clustered so most draws pass.
    while len(kept) < n:
        draw = max(n * 2, 1000)
        lons = rng.uniform(west, east, draw)
        lats = rng.uniform(south, north, draw)
        pts_m = project_coords(np.column_stack([lons, lats]), crs)
        dist, _ = tree.query(pts_m, k=1)
        for lon, lat, d in zip(lons, lats, dist):
            if d <= exclusion_m:
                continue
            point = Point(lon, lat)
            if inside is not None and not inside.contains(point):
                continue
            kept.append(point)
            if len(kept) >= n:
                break

    logger.info(f"Sampled {len(kept)} background points "
                f"(>{exclusion_m:.0f} m from any mapped landslide)")
    return gpd.GeoDataFrame(geometry=kept, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def terrain_features_at(points, processed_dir: Path, cfg: dict
                        ) -> tuple[np.ndarray, list[str]]:
    """Sample terrain rasters at points; returns (X, feature_names).

    Rasters come from the shared preprocessing step, so the landslide model
    uses the same derivations (and the same CRS-aware sampling) as the flood
    model rather than its own copy.
    """
    from pipeline.feature_extract import sample_raster_at_points

    coords = list(zip(points.geometry.x, points.geometry.y))
    rasters = {
        "elevation": processed_dir / "dem_reprojected.tif",
        "slope": processed_dir / "dem_derivatives" / "slope.tif",
        "twi": processed_dir / "dem_derivatives" / "twi.tif",
        "hand": processed_dir / "dem_derivatives" / "hand.tif",
        "flow_acc": processed_dir / "dem_derivatives" / "flow_accumulation.tif",
    }

    columns, names = [], []
    for name, path in rasters.items():
        if not path.exists():
            logger.warning(f"{path} missing — excluded from the landslide model")
            continue
        columns.append(sample_raster_at_points(str(path), coords))
        names.append(name)

    if not columns:
        raise FileNotFoundError(
            f"No terrain rasters in {processed_dir}. Run `preprocess` first."
        )
    return np.column_stack(columns), names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fit_susceptibility_model(X: np.ndarray, y: np.ndarray, coords_m: np.ndarray,
                             cfg: dict) -> dict:
    """Fit logistic regression with spatially blocked validation.

    Returns a dict with the fitted pipeline, validation metrics and the
    standardised coefficients (which are what a reader can interpret).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from pipeline.graph_build import spatial_block_split

    ls_cfg = cfg.get("landslide", {})
    train_mask, val_mask = spatial_block_split(
        coords_m,
        train_ratio=ls_cfg.get("train_split", 0.8),
        block_size_m=ls_cfg.get("block_size_m", 10_000),
        seed=ls_cfg.get("seed", 42),
    )
    train = train_mask.numpy().astype(bool)
    val = val_mask.numpy().astype(bool)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X[train], y[train])

    scores = model.predict_proba(X[val])[:, 1]
    metrics = {
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "val_positive_rate": float(y[val].mean()),
    }
    if len(np.unique(y[val])) > 1:
        metrics["val_auc_roc"] = float(roc_auc_score(y[val], scores))
        metrics["val_average_precision"] = float(
            average_precision_score(y[val], scores))
    else:
        metrics["note"] = "validation blocks hold a single class"

    logger.info(f"Landslide model (spatial blocks): {metrics}")
    return {"model": model, "metrics": metrics}


def predict_susceptibility_raster(model, processed_dir: Path, output_path: Path,
                                  feature_names: list[str],
                                  rows_per_block: int = 512) -> Path:
    """Apply the fitted model across the region, a horizontal strip at a time.

    The CHT DEM is tens of millions of pixels; stacking every feature band in
    memory and predicting in one call would need several gigabytes, so the
    raster is streamed in blocks of rows.
    """
    import rasterio

    rasters = {
        "elevation": processed_dir / "dem_reprojected.tif",
        "slope": processed_dir / "dem_derivatives" / "slope.tif",
        "twi": processed_dir / "dem_derivatives" / "twi.tif",
        "hand": processed_dir / "dem_derivatives" / "hand.tif",
        "flow_acc": processed_dir / "dem_derivatives" / "flow_accumulation.tif",
    }

    sources = {name: rasterio.open(rasters[name]) for name in feature_names}
    try:
        reference = sources[feature_names[0]]
        profile = reference.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
        height, width = reference.height, reference.width

        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_valid, total_sum = 0, 0.0

        with rasterio.open(output_path, "w", **profile) as dst:
            for row_start in range(0, height, rows_per_block):
                rows = min(rows_per_block, height - row_start)
                window = rasterio.windows.Window(0, row_start, width, rows)

                bands, invalid = [], None
                for name in feature_names:
                    src = sources[name]
                    data = src.read(1, window=window).astype(np.float32)
                    bad = ~np.isfinite(data) | (data < -9000)
                    if src.nodata is not None:
                        bad |= data == src.nodata
                    invalid = bad if invalid is None else (invalid | bad)
                    bands.append(np.nan_to_num(data, nan=0.0))

                stack = np.stack(bands, axis=-1)
                probs = model.predict_proba(
                    stack.reshape(-1, stack.shape[-1])
                )[:, 1].reshape(rows, width).astype(np.float32)
                probs[invalid] = -9999.0

                dst.write(probs, 1, window=window)
                valid = probs[probs >= 0]
                total_valid += valid.size
                total_sum += float(valid.sum())

            dst.update_tags(
                description="Landslide susceptibility, logistic regression "
                            "fitted to the COOLR inventory",
                citation=CITATION,
            )
    finally:
        for src in sources.values():
            src.close()

    mean = total_sum / total_valid if total_valid else float("nan")
    logger.info(f"Susceptibility raster → {output_path} "
                f"(mean {mean:.3f} over {total_valid} valid pixels)")
    return output_path


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_admin(susceptibility_path: Path, admin_path: str,
                       population_path: str | None, output_path: Path) -> list[dict]:
    """Zonal statistics per administrative unit.

    Requires real boundaries: there is no synthetic fallback, because naming
    an arbitrary strip of raster after a real upazila misleads anyone reading
    the table.
    """
    import geopandas as gpd
    from rasterstats import zonal_stats

    if not admin_path or not Path(admin_path).exists():
        raise FileNotFoundError(
            f"Admin boundaries not found at {admin_path}. Run `download` first."
        )

    import rasterio

    admin = gpd.read_file(admin_path)

    # rasterstats does not reproject: zones must be in the raster's CRS or
    # every zone silently comes back empty. The susceptibility raster is in
    # the projected CRS; WorldPop is in WGS84.
    with rasterio.open(susceptibility_path) as src:
        raster_crs = src.crs
    try:
        stats = zonal_stats(admin.to_crs(raster_crs), str(susceptibility_path),
                            stats=["mean", "max", "count"], nodata=-9999.0)
    except (OverflowError, ValueError) as exc:
        # Zones that project to nowhere near the raster overflow the window
        # arithmetic; that is the same failure as no overlap.
        raise RuntimeError(
            "Zonal statistics found no raster pixels in any unit — "
            f"check the CRS of the boundaries and the raster ({exc})."
        ) from exc
    if all((s or {}).get("count", 0) == 0 for s in stats):
        raise RuntimeError(
            "Zonal statistics found no raster pixels in any unit — "
            "check the CRS of the boundaries and the raster."
        )

    pop_stats = None
    if population_path and Path(population_path).exists():
        with rasterio.open(population_path) as src:
            pop_crs = src.crs
        pop_stats = zonal_stats(admin.to_crs(pop_crs), population_path,
                                stats=["sum"], nodata=-99999.0)

    results = []
    for i, (_, row) in enumerate(admin.iterrows()):
        s = stats[i] or {}
        entry = {
            "admin_name": row.get("admin_name"),
            "admin_label": row.get("admin_label", row.get("admin_name")),
            "susceptibility_mean": round(float(s.get("mean") or 0), 4),
            "susceptibility_max": round(float(s.get("max") or 0), 4),
            "n_pixels": int(s.get("count") or 0),
        }
        if pop_stats:
            total = (pop_stats[i] or {}).get("sum")
            entry["population"] = int(total) if total else None
        results.append(entry)

    results.sort(key=lambda r: r["susceptibility_mean"], reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Landslide summary for {len(results)} units → {output_path}")
    return results


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_landslide_pipeline(cfg: dict, raw_dir: Path = Path("data/raw"),
                           processed_dir: Path = Path("data/processed"),
                           output_dir: Path = Path("data/output")) -> str:
    """Inventory → features → fitted model → raster → admin summary."""
    import geopandas as gpd

    from pipeline.feature_extract import project_coords

    output_dir.mkdir(parents=True, exist_ok=True)
    ls_cfg = cfg.get("landslide", {})
    crs = cfg["aoi"]["crs"]

    logger.info("=== Landslide 1/5: inventory ===")
    inventory_path = download_landslide_inventory(cfg, raw_dir)
    inventory = gpd.read_file(inventory_path)

    logger.info("=== Landslide 2/5: background sample ===")
    ratio = ls_cfg.get("background_ratio", 2)
    domain = mapped_domain(
        inventory, buffer_m=ls_cfg.get("domain_buffer_m", 5000), crs=crs
    )
    background = sample_background_points(
        cfg["aoi"]["bbox"], n=len(inventory) * ratio, inventory=inventory,
        exclusion_m=ls_cfg.get("background_exclusion_m", 500), crs=crs,
        seed=ls_cfg.get("seed", 42), domain=domain,
    )

    logger.info("=== Landslide 3/5: terrain features ===")
    X_pos, names = terrain_features_at(inventory, processed_dir, cfg)
    X_neg, _ = terrain_features_at(background, processed_dir, cfg)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    points = np.vstack([
        np.column_stack([inventory.geometry.x, inventory.geometry.y]),
        np.column_stack([background.geometry.x, background.geometry.y]),
    ])
    coords_m = project_coords(points, crs)

    logger.info("=== Landslide 4/5: fit and validate ===")
    fitted = fit_susceptibility_model(X, y, coords_m, cfg)

    coefficients = dict(zip(
        names,
        [round(float(c), 4) for c in
         fitted["model"].named_steps["logisticregression"].coef_[0]],
    ))
    # Inventory dates matter for interpretation: a single-event inventory
    # describes susceptibility to that storm, not to every possible one.
    dates = []
    if "event_date" in inventory.columns:
        import pandas as pd
        parsed = pd.to_datetime(inventory["event_date"], unit="ms", errors="coerce")
        dates = sorted({d.strftime("%Y-%m-%d") for d in parsed.dropna()})

    metadata = {
        "inventory": {
            "source": "NASA COOLR (Events Points)",
            "citation": CITATION,
            "n_landslides": int(len(inventory)),
            "n_background": int(len(background)),
            "event_dates": dates[:20],
            "n_event_dates": len(dates),
            "single_event": len(dates) == 1,
            "background_domain": "convex hull of the inventory, buffered "
                                 f"{ls_cfg.get('domain_buffer_m', 5000)} m",
        },
        "features": names,
        "standardised_coefficients": coefficients,
        "validation": fitted["metrics"],
        "model": "logistic regression, class-weighted, spatially blocked validation",
    }
    (output_dir / "landslide_model.json").write_text(json.dumps(metadata, indent=2))

    logger.info("=== Landslide 5/5: predict and aggregate ===")
    raster_path = predict_susceptibility_raster(
        fitted["model"], processed_dir,
        output_dir / "landslide_susceptibility.tif", names,
    )
    aggregate_to_admin(
        raster_path,
        cfg["data"].get("admin_boundaries", {}).get("upazila", ""),
        cfg["data"].get("vulnerability", {}).get("population_path"),
        output_dir / "landslide_upazila.json",
    )

    logger.info(f"Landslide pipeline complete. Coefficients: {coefficients}")
    return str(output_dir / "landslide_upazila.json")
