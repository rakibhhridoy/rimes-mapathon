"""
Optional environmental covariates: vegetation, soil and long-term rainfall.

The default model sees terrain, population, access distances and asset type.
Flood susceptibility studies often add vegetation, soil and climate layers, so
this module fetches one of each from Earth Engine and tests, on the same
held-out blocks as the benchmark, whether they change the model's skill.
Nothing here alters the default model: the layers enter a model only when
listed under `data.extra_features`, and the ablation writes its own file.

Layers (data/<region>/raw/):
    cov_ndvi_dry.tif     median dry-season (Jan-Mar) MODIS NDVI, 2017-2025, 250 m.
                         Dry season only, because monsoon NDVI drops over standing
                         water and would carry the flood label into the features.
    cov_clay.tif         SoilGrids 2.0 clay content, 0-5 cm, g/kg, 250 m.
    cov_rain_annual.tif  CHIRPS mean annual rainfall 1991-2020, mm, from the 5-day
                         product, exported at its native ~5 km (the daily sum over
                         30 years times out on Earth Engine).
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

COVARIATES = ("ndvi_dry", "clay", "rain_annual")
SCALE_M = 250
# Rainfall is exported near CHIRPS's native 0.05 degrees; finer adds nothing.
SCALES_M = {"ndvi_dry": SCALE_M, "clay": SCALE_M, "rain_annual": 5000}


def _images(ee):
    ndvi = (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate("2017-01-01", "2025-12-31")
            .filter(ee.Filter.calendarRange(1, 3, "month"))
            .select("NDVI").median().multiply(0.0001))
    clay = ee.Image("projects/soilgrids-isric/clay_mean").select("clay_0-5cm_mean")
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")
            .filterDate("1991-01-01", "2021-01-01")
            .select("precipitation").sum().divide(30))
    return {"ndvi_dry": ndvi.toFloat(), "clay": clay.toFloat(), "rain_annual": rain.toFloat()}


def covariate_paths(raw_dir: Path) -> dict:
    return {name: raw_dir / f"cov_{name}.tif" for name in COVARIATES}


def download_covariates(cfg: dict, raw_dir: Path) -> dict:
    """Download the three layers over the region's bounding box."""
    from pipeline.sentinel1 import download_image, init_ee

    ee = init_ee(cfg)
    aoi = ee.Geometry.Rectangle(cfg["aoi"]["bbox"])
    paths = covariate_paths(raw_dir)
    for name, image in _images(ee).items():
        if paths[name].exists():
            logger.info(f"{paths[name].name} already present, skipping download")
            continue
        download_image(ee, image, aoi, SCALES_M[name], paths[name])
    return paths


def run_covariate_ablation(cfg: dict, processed_dir: Path, raw_dir: Path,
                           output_dir: Path, infra_path: Path, seeds=None) -> dict:
    """The default model with and without the covariates, on identical blocks.

    For each block assignment of the benchmark, the default gradient-boosted
    model is fitted twice on the same training blocks, once on the default
    features and once with the three covariates appended, and both are scored
    on the same test blocks. The paired AUC difference is tested with the
    corrected resampled t-test, as elsewhere in the benchmark.

    The same blocks also score each covariate alone and a location control,
    the default features plus projected x/y coordinates. A covariate that gains
    no more than the coordinates mostly tells the model where an asset is,
    which helps between neighbouring held-out blocks without describing the
    ground.
    """
    import geopandas as gpd

    from pipeline.benchmark import (SEEDS, _fit_predict, _metrics, _tabular_models,
                                    corrected_p_value)
    from pipeline.feature_extract import (compute_centroids, extract_features,
                                          project_coords, sample_raster_at_points)
    from pipeline.graph_build import spatial_block_split_three

    seeds = seeds or SEEDS
    infra = gpd.read_file(str(infra_path))
    X, coords, y, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y).astype(int)
    coords = np.asarray(coords)

    points = compute_centroids(infra)
    coord_list = list(zip(points["lon"], points["lat"]))
    extra, coverage = [], {}
    for name, path in covariate_paths(raw_dir).items():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `python -m pipeline.cli covariates` first")
        values = sample_raster_at_points(str(path), coord_list, fill=np.nan)
        coverage[name] = float(np.isfinite(values).mean())
        values = np.where(np.isfinite(values), values, np.nanmedian(values))
        if np.std(values) == 0:
            raise ValueError(f"Covariate {name} is constant over the region's assets")
        extra.append((values - values.mean()) / values.std())
    X_plus = np.column_stack([X] + extra).astype(np.float32)

    coords_m = project_coords(coords, cfg["aoi"]["crs"])
    xy = [(c - c.mean()) / c.std() for c in coords_m.T]
    variants = {f"+{name}": np.column_stack([X, col]).astype(np.float32)
                for name, col in zip(COVARIATES, extra)}
    variants["+xy_coordinates"] = np.column_stack([X] + xy).astype(np.float32)
    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)

    per_seed = []
    for seed in seeds:
        train_mask, _, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        train = train_mask.numpy().astype(bool)
        test = test_mask.numpy().astype(bool)
        if len(np.unique(y[test])) < 2:
            logger.warning(f"seed {seed}: test blocks hold one class, skipped")
            continue
        base = _fit_predict("gradient_boosting", _tabular_models(seed)["gradient_boosting"],
                            X, y, train, test, seed)
        plus = _fit_predict("gradient_boosting", _tabular_models(seed)["gradient_boosting"],
                            X_plus, y, train, test, seed)
        row = {"seed": seed, "n_test": int(test.sum()), "n_train": int(train.sum()),
               "default": _metrics(y[test], base), "with_covariates": _metrics(y[test], plus)}
        row["auc_difference"] = row["with_covariates"]["auc_roc"] - row["default"]["auc_roc"]
        row["variants"] = {}
        for name, Xv in variants.items():
            score = _fit_predict("gradient_boosting",
                                 _tabular_models(seed)["gradient_boosting"],
                                 Xv, y, train, test, seed)
            row["variants"][name] = _metrics(y[test], score)["auc_roc"] - row["default"]["auc_roc"]
        per_seed.append(row)
        logger.info(f"seed {seed}: default {row['default']['auc_roc']:.3f}, "
                    f"with covariates {row['with_covariates']['auc_roc']:.3f}")

    diff = np.array([r["auc_difference"] for r in per_seed])
    summary = {
        "default_auc_mean": float(np.mean([r["default"]["auc_roc"] for r in per_seed])),
        "with_covariates_auc_mean": float(np.mean([r["with_covariates"]["auc_roc"]
                                                   for r in per_seed])),
        "auc_difference_mean": float(diff.mean()),
        "auc_difference_sd": float(diff.std(ddof=1)) if len(diff) > 1 else None,
        "seeds_covariates_ahead": int((diff > 0).sum()),
        "n_seeds": int(len(diff)),
        "p_value": corrected_p_value(diff, [r["n_test"] / r["n_train"] for r in per_seed]),
        "variants": {},
    }
    ratios = [r["n_test"] / r["n_train"] for r in per_seed]
    for name in variants:
        d = np.array([r["variants"][name] for r in per_seed])
        summary["variants"][name] = {
            "auc_difference_mean": float(d.mean()),
            "seeds_ahead": int((d > 0).sum()),
            "p_value": corrected_p_value(d, ratios),
        }
    result = {"covariates": list(COVARIATES), "scale_m": SCALES_M, "coverage": coverage,
              "model": "gradient_boosting", "block_size_m": block_m, "seeds": list(seeds),
              "per_seed": per_seed, "summary": summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "covariate_ablation.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'covariate_ablation.json'}")
    return result
