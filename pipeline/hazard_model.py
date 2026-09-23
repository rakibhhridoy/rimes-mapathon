"""
Hazard surface from terrain, as an alternative to kriging asset scores.

Kriging interpolates the GNN scores between mapped assets, and how much it
can say between them depends on how spatially structured the scores are. On
the current runs the nugget is about half the full sill in both flood regions
(0.47 in Rangpur and Rajshahi, 0.57 in Sylhet) and the mean Kriging variance
is 54 to 68 per cent of the total variance, so the interpolated surface
smooths towards the regional average: in Rangpur and Rajshahi its standard
deviation across the grid was 0.12 against 0.22 for the terrain surface.

This module instead predicts flood-prone ground from the terrain of each
grid cell, so hazard varies at the resolution of the DEM rather than at the
spacing of mapped infrastructure.

Circularity warning: while training labels come from thresholds on TWI and
HAND (`data.labels.source: proxy`), a terrain model largely reproduces that
rule, and the surface should be read as "ground the ensemble calls
flood-prone", not as an independent prediction. With Sentinel-1 observed
labels the same model becomes a genuine flood-probability surface.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Terrain only. Distances to hospitals or roads describe exposure and
# vulnerability, not how likely the ground is to flood.
TERRAIN_FEATURES = ["elevation", "slope", "twi", "hand", "flow_acc"]


def _raster_paths(processed_dir: Path) -> dict:
    deriv = processed_dir / "dem_derivatives"
    return {
        "elevation": processed_dir / "dem_reprojected.tif",
        "slope": deriv / "slope.tif",
        "twi": deriv / "twi.tif",
        "hand": deriv / "hand.tif",
        "flow_acc": deriv / "flow_accumulation.tif",
    }


def _sample(paths: dict, features: list[str], coords) -> np.ndarray:
    from pipeline.feature_extract import sample_raster_at_points

    return np.column_stack([
        sample_raster_at_points(str(paths[name]), coords) for name in features
    ])


def fit_hazard_model(cfg: dict, processed_dir: Path, output_dir: Path,
                     infra_path: Path) -> dict:
    """Fit a terrain → flood-prone model, validated on held-out blocks."""
    import geopandas as gpd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from pipeline.feature_extract import project_coords, sample_raster_at_points
    from pipeline.graph_build import spatial_block_split

    paths = _raster_paths(processed_dir)
    features = [f for f in TERRAIN_FEATURES if paths[f].exists()]
    if not features:
        raise FileNotFoundError(f"No terrain rasters in {processed_dir}")

    labels_cfg = cfg["data"].get("labels", {})
    label_name = ("flood_observed_labels.tif"
                  if labels_cfg.get("source") == "observed"
                  else "flood_proxy_labels.tif")
    label_path = processed_dir / labels_cfg.get("path", label_name)
    if not label_path.exists():
        raise FileNotFoundError(f"Labels not found: {label_path}")

    # Train where the assets are, so the sample matches the flood model's.
    infra = gpd.read_file(str(infra_path))
    points = infra.geometry.representative_point()
    coords = list(zip(points.x, points.y))

    X = _sample(paths, features, coords)
    y = (sample_raster_at_points(str(label_path), coords) > 0.5).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Labels are single-class; cannot fit a hazard model.")

    coords_m = project_coords(np.column_stack([points.x, points.y]),
                              cfg["aoi"]["crs"])
    train_mask, val_mask = spatial_block_split(
        coords_m,
        train_ratio=cfg["graph"].get("train_split", 0.8),
        block_size_m=cfg["graph"].get("block_size_m", 10_000),
        seed=cfg.get("gnn", {}).get("seed", 42),
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
        "features": features,
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "val_positive_rate": float(y[val].mean()),
        "label_source": labels_cfg.get("source", "proxy"),
    }
    if len(np.unique(y[val])) > 1:
        metrics["val_auc_roc"] = float(roc_auc_score(y[val], scores))
        metrics["val_average_precision"] = float(
            average_precision_score(y[val], scores))
    metrics["standardised_coefficients"] = dict(zip(
        features,
        [round(float(c), 4)
         for c in model.named_steps["logisticregression"].coef_[0]],
    ))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hazard_model.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"Terrain hazard model: {metrics}")
    return {"model": model, "features": features, "metrics": metrics}


def predict_hazard_at(fitted: dict, processed_dir: Path, coords) -> np.ndarray:
    """Flood-prone probability at (lon, lat) coordinates."""
    paths = _raster_paths(processed_dir)
    X = _sample(paths, fitted["features"], list(coords))
    return fitted["model"].predict_proba(X)[:, 1]


def write_hazard_raster(fitted: dict, processed_dir: Path, output_path: Path,
                        rows_per_block: int = 512) -> Path:
    """Render the hazard surface over the DEM grid, streamed in row blocks."""
    import rasterio

    paths = _raster_paths(processed_dir)
    sources = {name: rasterio.open(paths[name]) for name in fitted["features"]}
    try:
        reference = sources[fitted["features"][0]]
        profile = reference.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            for row_start in range(0, reference.height, rows_per_block):
                rows = min(rows_per_block, reference.height - row_start)
                window = rasterio.windows.Window(0, row_start,
                                                 reference.width, rows)
                bands, invalid = [], None
                for name in fitted["features"]:
                    src = sources[name]
                    data = src.read(1, window=window).astype(np.float32)
                    bad = ~np.isfinite(data) | (data < -9000)
                    if src.nodata is not None:
                        bad |= data == src.nodata
                    invalid = bad if invalid is None else (invalid | bad)
                    bands.append(np.nan_to_num(data, nan=0.0))

                stack = np.stack(bands, axis=-1)
                probs = fitted["model"].predict_proba(
                    stack.reshape(-1, stack.shape[-1])
                )[:, 1].reshape(rows, reference.width).astype(np.float32)
                probs[invalid] = -9999.0
                dst.write(probs, 1, window=window)

            dst.update_tags(
                description="Flood hazard from terrain (logistic regression)",
                label_source=fitted["metrics"].get("label_source", "proxy"),
            )
    finally:
        for src in sources.values():
            src.close()

    logger.info(f"Terrain hazard raster → {output_path}")
    return output_path
