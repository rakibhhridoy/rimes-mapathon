"""
Validation of the model against observed floods.

Three questions, each answered separately:

1. Does the model rank the places that actually flooded above those that did
   not? Measured at asset locations against Sentinel-1 flood extents, on
   spatially held-out blocks as well as on everything.
2. Does the kriged hazard surface agree with observed flooding across the
   whole grid, not just where assets happen to be?
3. How close is the terrain-threshold proxy label to observed flooding? This
   quantifies what the earlier proxy-trained results were actually learning.

Outputs data/output/validation_metrics.json.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Ranking and calibration metrics for continuous scores."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    out = {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else None,
    }
    if len(np.unique(labels)) < 2:
        out["note"] = "single-class sample; ranking metrics undefined"
        return out

    out["auc_roc"] = float(roc_auc_score(labels, scores))
    out["average_precision"] = float(average_precision_score(labels, scores))
    # Lift over the base rate: how much better than guessing the prevalence.
    out["ap_lift"] = float(out["average_precision"] / labels.mean())
    if scores.min() >= 0 and scores.max() <= 1:
        out["brier"] = float(brier_score_loss(labels, scores))
    return out


def _binary_agreement(reference: np.ndarray, predicted: np.ndarray) -> dict:
    """Categorical scores used in flood mapping: POD, FAR, CSI, kappa.

    `reference` is the observed flood mask, `predicted` the mask being judged.
    """
    from sklearn.metrics import cohen_kappa_score

    ref = reference.astype(bool)
    pred = predicted.astype(bool)
    hits = int((ref & pred).sum())
    misses = int((ref & ~pred).sum())
    false_alarms = int((~ref & pred).sum())
    correct_negatives = int((~ref & ~pred).sum())

    def safe(num, den):
        return float(num / den) if den else None

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        # Probability of detection: share of observed flooding captured.
        "pod": safe(hits, hits + misses),
        # False alarm ratio: share of flagged area that did not flood.
        "far": safe(false_alarms, hits + false_alarms),
        # Critical success index: the usual single summary for flood masks.
        "csi": safe(hits, hits + misses + false_alarms),
        "cohen_kappa": float(cohen_kappa_score(ref.ravel(), pred.ravel()))
        if len(np.unique(ref)) > 1 and len(np.unique(pred)) > 1 else None,
    }


def validate_assets(cfg: dict, output_dir: Path, raw_dir: Path) -> dict:
    """Model scores at asset locations against observed flooding."""
    import geopandas as gpd

    from pipeline.feature_extract import compute_centroids, sample_raster_at_points

    freq_path = raw_dir / "s1_flood_frequency.tif"
    if not freq_path.exists():
        raise FileNotFoundError(
            f"{freq_path} missing. Run `python -m pipeline.cli sentinel1` first."
        )

    scores = np.load(output_dir / "gnn_risk_scores.npy")
    infra = compute_centroids(gpd.read_file(str(raw_dir / "infrastructure_raw.gpkg")))
    coords = list(zip(infra["lon"], infra["lat"]))
    if len(scores) != len(infra):
        raise ValueError(
            f"{len(scores)} scores for {len(infra)} assets — rerun the pipeline."
        )

    freq_pct = sample_raster_at_points(str(freq_path), coords)
    n_events = len(cfg.get("sentinel1", {}).get("events", [])) or 1
    # Same definition as the training labels: flooded in at least
    # `min_events` of the mapped events, not in any one of them.
    min_events = cfg["data"].get("labels", {}).get("min_events", 1)
    threshold_pct = 100.0 * min_events / n_events - 1e-6
    observed = (freq_pct >= threshold_pct).astype(int)

    results = {
        "n_events": n_events,
        "min_events": min_events,
        "observed_flooded_share": float(observed.mean()),
        "all_assets": _metrics(observed, scores),
    }

    # Held-out blocks only: the honest number, since training saw the rest.
    # Newer graphs carry a test_mask that neither training nor calibration
    # touched; older ones only a val_mask.
    graph_path = output_dir / "spatial_graph.pt"
    if graph_path.exists():
        import torch
        graph = torch.load(graph_path, weights_only=False)
        mask = graph.test_mask if "test_mask" in graph else graph.val_mask
        held = mask.numpy().astype(bool)
        if held.shape[0] == len(scores):
            results["held_out_blocks"] = _metrics(observed[held], scores[held])

            # Calibrated probabilities, when the training step produced them:
            # the Brier score is the meaningful number for those.
            prob_path = output_dir / "gnn_flood_probability.npy"
            if prob_path.exists():
                probability = np.load(prob_path)
                if len(probability) == len(scores):
                    results["held_out_blocks_calibrated"] = _metrics(
                        observed[held], probability[held])

    # Per-event: was each flood's footprint ranked highly?
    per_event = {}
    for event in cfg.get("sentinel1", {}).get("events", []):
        path = raw_dir / f"s1_flood_{event['name']}.tif"
        if path.exists():
            mask = (sample_raster_at_points(str(path), coords) > 0).astype(int)
            per_event[event["name"]] = _metrics(mask, scores)
            per_event[event["name"]]["flooded_share"] = float(mask.mean())
    if per_event:
        results["per_event"] = per_event

    return results


def validate_hazard_surface(cfg: dict, output_dir: Path, raw_dir: Path) -> dict:
    """Both hazard surfaces against observed flooding across the grid.

    The Kriged surface interpolates asset scores; the terrain surface is
    evaluated from each cell's own terrain and is the one the composite uses.
    Each is scored on its own grid.
    """
    results = {}
    for key, name in (("kriged", "flood_risk_kriged.tif"),
                      ("terrain", "flood_hazard_terrain.tif")):
        results[key] = _validate_one_surface(cfg, output_dir / name, raw_dir)
    results["used_by_composite"] = cfg.get("risk", {}).get("hazard_source", "kriged")
    return results


def _validate_one_surface(cfg: dict, hazard_path: Path, raw_dir: Path) -> dict:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    freq_path = raw_dir / "s1_flood_frequency.tif"
    if not (hazard_path.exists() and freq_path.exists()):
        return {"note": "hazard surface or flood frequency missing"}

    # Compare on the hazard grid, downsampled where it is finer than about
    # 500 m so a DEM-resolution surface does not mean tens of millions of
    # cells in memory.
    with rasterio.open(hazard_path) as src:
        factor = 1
        pixel_m = abs(src.transform.a) * (1.0 if src.crs.is_projected else 111_000)
        if pixel_m < 400:
            factor = int(round(500 / pixel_m))
        out_shape = (max(1, src.height // factor), max(1, src.width // factor))
        hazard = src.read(1, out_shape=out_shape,
                          resampling=Resampling.average).astype(np.float32)
        ref_transform = src.transform * src.transform.scale(
            src.width / out_shape[1], src.height / out_shape[0])
        ref_crs, ref_shape = src.crs, out_shape
        hazard_nodata = src.nodata

    with rasterio.open(freq_path) as src:
        observed = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=observed,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.average,
        )

    valid = np.isfinite(hazard) & np.isfinite(observed)
    if hazard_nodata is not None:
        valid &= hazard != hazard_nodata
    hazard_v = hazard[valid]
    observed_v = observed[valid]

    # The hazard grid is coarser than the flood map, so `observed` here is a
    # cell's mean flood frequency. A cell counts as flooded when on average it
    # flooded in at least `min_events` events, matching the label definition.
    n_events = len(cfg.get("sentinel1", {}).get("events", [])) or 1
    min_events = cfg["data"].get("labels", {}).get("min_events", 1)
    flooded = (observed_v >= 100.0 * min_events / n_events - 1e-6).astype(int)
    result = {
        "n_cells": int(valid.sum()),
        "min_events": min_events,
        "flooded_cell_share": float(flooded.mean()),
        "ranking": _metrics(flooded, hazard_v),
    }
    if len(np.unique(flooded)) > 1:
        result["spearman"] = float(
            __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
                hazard_v, observed_v
            ).statistic
        )
    return result


def compare_proxy_to_observed(cfg: dict, processed_dir: Path, raw_dir: Path) -> dict:
    """How well the terrain-threshold proxy label matches observed flooding."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    proxy_path = processed_dir / "flood_proxy_labels.tif"
    freq_path = raw_dir / "s1_flood_frequency.tif"
    if not (proxy_path.exists() and freq_path.exists()):
        return {"note": "proxy labels or flood frequency missing"}

    with rasterio.open(proxy_path) as src:
        proxy = src.read(1).astype(np.float32)
        ref_transform, ref_crs, ref_shape = src.transform, src.crs, src.shape

    with rasterio.open(freq_path) as src:
        observed = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=observed,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )

    n_events = len(cfg.get("sentinel1", {}).get("events", [])) or 1
    min_events = cfg["data"].get("labels", {}).get("min_events", 1)
    threshold_pct = 100.0 * min_events / n_events - 1e-6
    valid = np.isfinite(proxy) & (proxy >= 0) & np.isfinite(observed)
    agreement = _binary_agreement(observed[valid] >= threshold_pct, proxy[valid] > 0.5)
    agreement["min_events"] = min_events
    agreement["proxy_positive_share"] = float((proxy[valid] > 0.5).mean())
    agreement["observed_positive_share"] = float((observed[valid] >= threshold_pct).mean())
    return agreement


def run_validation(cfg: dict,
                   output_dir: Path = Path("data/output"),
                   processed_dir: Path = Path("data/processed"),
                   raw_dir: Path = Path("data/raw")) -> dict:
    """Run every validation and write the results."""
    from datetime import datetime, timezone

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_source": cfg["data"].get("labels", {}).get("source", "proxy"),
        "assets_vs_observed": validate_assets(cfg, output_dir, raw_dir),
        "hazard_surface_vs_observed": validate_hazard_surface(cfg, output_dir, raw_dir),
        "proxy_vs_observed": compare_proxy_to_observed(cfg, processed_dir, raw_dir),
    }

    out_path = output_dir / "validation_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Validation metrics → {out_path}")

    held = results["assets_vs_observed"].get("held_out_blocks", {})
    if "auc_roc" in held:
        logger.info(
            f"Held-out blocks vs observed floods: AUC {held['auc_roc']:.3f}, "
            f"AP {held['average_precision']:.3f} "
            f"({held['ap_lift']:.1f}x the base rate)"
        )
    return results
