"""
Like-for-like comparison of the graph model against ordinary tabular models.

Why this exists: the document claimed the graph model works without ever
showing that the graph itself contributes anything. Every model here sees the
same standardised features at the same assets and the same spatial blocks, so
the only difference is the model. Each run is repeated over several block
assignments, because a single split gives a point estimate with no sense of
how much of a gap is noise.

Output (per region): benchmark.json with per-seed and summary metrics.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The first five are the assignments earlier runs used, so those stay comparable.
SEEDS = (42, 7, 13, 21, 99, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17)


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    base = float(y_true.mean())
    out = {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "positive_rate": base,
    }
    out["ap_lift"] = out["average_precision"] / base if base > 0 else None
    return out


def _tabular_models(seed: int):
    """The baselines a reviewer would reach for, all class-balanced."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=1, random_state=seed),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=seed),
    }


def _fit_predict(name, model, X, y, train, test, seed):
    from sklearn.utils.class_weight import compute_sample_weight

    if name == "gradient_boosting":     # no class_weight parameter
        weights = compute_sample_weight("balanced", y[train])
        model.fit(X[train], y[train], sample_weight=weights)
    else:
        model.fit(X[train], y[train])
    return model.predict_proba(X[test])[:, 1]


def run_benchmark(cfg: dict, processed_dir: Path, output_dir: Path,
                  infra_path: Path, seeds=SEEDS) -> dict:
    """Train every model on each block assignment and score the test blocks."""
    import geopandas as gpd
    import torch

    import pandas as pd

    from pipeline.feature_extract import extract_features, project_coords
    from pipeline.gnn_model import train_model
    from pipeline.graph_build import build_spatial_graph, spatial_block_split_three

    infra = gpd.read_file(str(infra_path))
    X, coords, y, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y).astype(int)
    coords = np.asarray(coords)
    # extract_features writes the columns it kept, in order, so the names
    # come back from the parquet rather than being guessed here.
    columns = pd.read_parquet(processed_dir / "node_features.parquet").columns
    skip = {"asset_type", "name", "priority", "lon", "lat", "flood_label"}
    feature_names = [c for c in columns if c not in skip]

    # Blocks are cut in metres, so the split needs projected coordinates.
    coords_m = project_coords(coords, cfg["aoi"]["crs"])

    # Edges do not depend on the split, so the graph is built once and only
    # its masks change from seed to seed.
    graph = build_spatial_graph(X, coords, y, cfg)

    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)

    per_seed = []
    for seed in seeds:
        train_mask, calib_mask, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        train = train_mask.numpy().astype(bool)
        calib = calib_mask.numpy().astype(bool)
        test = test_mask.numpy().astype(bool)
        if len(np.unique(y[test])) < 2:
            logger.warning(f"seed {seed}: test blocks hold one class, skipped")
            continue

        row = {"seed": seed, "n_test": int(test.sum()),
               "test_positive_rate": float(y[test].mean()),
               "models": {}}

        # Graph model on this split, trained exactly as the pipeline trains it.
        graph.train_mask = train_mask
        graph.calib_mask = calib_mask
        graph.val_mask = calib_mask
        graph.test_mask = test_mask
        seeded = dict(cfg)
        seeded["gnn"] = {**cfg.get("gnn", {}), "seed": seed}
        model = train_model(graph, seeded)
        model.eval()
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index).squeeze()
            scores = torch.sigmoid(logits).numpy()
        row["models"]["graph_sage"] = _metrics(y[test], scores[test])

        # The same features without the graph.
        for name, estimator in _tabular_models(seed).items():
            probability = _fit_predict(name, estimator, X, y, train, test, seed)
            row["models"][name] = _metrics(y[test], probability)

        # A model-free reference: rank by wetness index alone.
        if "twi" in feature_names:
            row["models"]["twi_only"] = _metrics(
                y[test], X[test, feature_names.index("twi")])

        per_seed.append(row)
        logger.info(f"seed {seed}: " + ", ".join(
            f"{k} {v['auc_roc']:.3f}" for k, v in row["models"].items()))

    summary = {}
    names = per_seed[0]["models"].keys() if per_seed else []
    for name in names:
        aucs = np.array([r["models"][name]["auc_roc"] for r in per_seed])
        lifts = np.array([r["models"][name]["ap_lift"] for r in per_seed])
        summary[name] = {
            "auc_mean": float(aucs.mean()),
            "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
            "auc_min": float(aucs.min()), "auc_max": float(aucs.max()),
            "ap_lift_mean": float(lifts.mean()),
            "ap_lift_sd": float(lifts.std(ddof=1)) if len(lifts) > 1 else None,
            "n_seeds": int(len(aucs)),
        }

    # Paired difference between the graph model and the best baseline, which
    # is what decides whether the graph earns its place.
    baselines = [n for n in names if n != "graph_sage"]
    if per_seed and baselines:
        best = max(baselines, key=lambda n: summary[n]["auc_mean"])
        diff = np.array([r["models"]["graph_sage"]["auc_roc"]
                         - r["models"][best]["auc_roc"] for r in per_seed])
        paired = {"best_baseline": best, "auc_difference_mean": float(diff.mean()),
                  "auc_difference_sd": float(diff.std(ddof=1)) if len(diff) > 1 else None,
                  "seeds_graph_ahead": int((diff > 0).sum()), "n_seeds": int(len(diff))}
        if len(diff) > 1:
            from scipy import stats

            t_stat, p_value = stats.ttest_rel(
                [r["models"]["graph_sage"]["auc_roc"] for r in per_seed],
                [r["models"][best]["auc_roc"] for r in per_seed])
            paired["t_statistic"] = float(t_stat)
            paired["p_value"] = float(p_value)
        summary["graph_vs_best_baseline"] = paired

    result = {"features": list(feature_names), "n_assets": int(len(y)),
              "block_size_m": block_m, "seeds": list(seeds),
              "per_seed": per_seed, "summary": summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'benchmark.json'}")
    return result

def run_label_comparison(cfg: dict, processed_dir: Path, output_dir: Path,
                         infra_path: Path, seeds=SEEDS) -> dict:
    """Train the configured model on proxy labels and on observed labels.

    Both models are scored against the observed definition on the same test
    blocks, so the comparison isolates the labels. The scores of the first
    assignment are saved for the ROC figure.
    """
    import geopandas as gpd

    from pipeline.asset_model import _fit_tabular, model_type
    from pipeline.feature_extract import (compute_centroids, extract_features,
                                          project_coords, sample_raster_at_points)
    from pipeline.graph_build import spatial_block_split_three

    kind = model_type(cfg)
    if kind == "graph_sage":
        raise ValueError("Label comparison expects a tabular asset model.")

    infra = gpd.read_file(str(infra_path))
    X, coords, y_observed, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y_observed = np.ascontiguousarray(y_observed).astype(int)
    centroids = compute_centroids(gpd.read_file(str(infra_path)))
    point_list = list(zip(centroids["lon"], centroids["lat"]))
    proxy = sample_raster_at_points(
        str(processed_dir / "flood_proxy_labels.tif"), point_list)
    y_proxy = (proxy > 0.5).astype(int)
    coords_m = project_coords(np.asarray(coords), cfg["aoi"]["crs"])

    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)

    per_seed, saved = [], None
    for seed in seeds:
        train_mask, _, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        train = train_mask.numpy().astype(bool)
        test = test_mask.numpy().astype(bool)
        if len(np.unique(y_observed[test])) < 2:
            continue
        row = {"seed": seed, "n_test": int(test.sum()), "labels": {}}
        scores = {}
        for name, labels in (("observed", y_observed), ("proxy", y_proxy)):
            if len(np.unique(labels[train])) < 2:
                continue
            fitted = _fit_tabular(kind, X, labels, train, seed)
            scores[name] = fitted.predict_proba(X[test])[:, 1]
            row["labels"][name] = _metrics(y_observed[test], scores[name])
        per_seed.append(row)
        if saved is None and len(scores) == 2:
            saved = {"y": y_observed[test], "observed_trained": scores["observed"],
                     "proxy_trained": scores["proxy"], "seed": seed}

    summary = {}
    for name in ("observed", "proxy"):
        aucs = np.array([r["labels"][name]["auc_roc"] for r in per_seed
                         if name in r["labels"]])
        if len(aucs):
            summary[name] = {"auc_mean": float(aucs.mean()),
                             "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
                             "n_seeds": int(len(aucs))}
    paired = [r["labels"]["observed"]["auc_roc"] - r["labels"]["proxy"]["auc_roc"]
              for r in per_seed if len(r["labels"]) == 2]
    if paired:
        summary["observed_minus_proxy"] = {
            "mean": float(np.mean(paired)),
            "sd": float(np.std(paired, ddof=1)) if len(paired) > 1 else None,
            "seeds_observed_ahead": int(sum(d > 0 for d in paired)),
            "n_seeds": int(len(paired)),
        }
        if len(paired) > 1:
            from scipy import stats

            summary["observed_minus_proxy"]["p_value"] = float(
                stats.ttest_1samp(paired, 0.0).pvalue)

    result = {"model": kind, "seeds": list(seeds), "per_seed": per_seed,
              "summary": summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_comparison.json").write_text(json.dumps(result, indent=2))
    if saved is not None:
        np.savez(output_dir / "label_comparison_scores.npz", **saved)
    logger.info(f"Wrote {output_dir / 'label_comparison.json'}")
    return result


def run_temporal_holdout(cfg: dict, processed_dir: Path, raw_dir: Path,
                         output_dir: Path, infra_path: Path, seeds=SEEDS,
                         cutoff_year: int = 2022) -> dict:
    """Train on floods up to `cutoff_year` and score against later floods.

    The spatial blocks separate training from testing in space only, since
    labels and reference extents come from the same events. Here the model
    learns from the earlier events and is scored, on the held-out blocks,
    against the extents of the later ones, so the test ground is new in both
    space and time.

    Two references are scored on the same assets. "past_flooding" is the share
    of earlier events in which the ground flooded, which is the map a planner
    would have without any model. "all_events" is the same model trained on
    labels that include the later events, the spatial-only design, and gives
    the ceiling the temporal test is measured against.
    """
    import geopandas as gpd
    from scipy import stats

    from pipeline.asset_model import _fit_tabular, model_type
    from pipeline.feature_extract import (compute_centroids, extract_features,
                                          project_coords, sample_raster_at_points)
    from pipeline.graph_build import spatial_block_split_three

    kind = model_type(cfg)
    if kind == "graph_sage":
        raise ValueError("Temporal hold-out expects a tabular asset model.")

    events = (cfg.get("sentinel1") or {}).get("events", [])
    early = [e for e in events if int(e["start"][:4]) <= cutoff_year]
    late = [e for e in events if int(e["start"][:4]) > cutoff_year]
    if not early or not late:
        raise ValueError(f"Need events on both sides of {cutoff_year}; "
                         f"got {len(early)} before and {len(late)} after.")

    infra = gpd.read_file(str(infra_path))
    X, coords, y_all, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y_all = np.ascontiguousarray(y_all).astype(int)
    centroids = compute_centroids(gpd.read_file(str(infra_path)))
    points = list(zip(centroids["lon"], centroids["lat"]))

    def flooded_count(evts):
        masks = [sample_raster_at_points(str(raw_dir / f"s1_flood_{e['name']}.tif"),
                                         points) for e in evts]
        return np.sum([np.nan_to_num(m) > 0.5 for m in masks], axis=0)

    early_count = flooded_count(early)
    late_count = flooded_count(late)
    # The training threshold follows the region's own rule, capped by the
    # number of earlier events. The later floods count if any of them reached
    # the ground, since there are only one or two of them.
    min_events = min((cfg.get("data", {}).get("labels", {}) or {})
                     .get("min_events", 1), len(early))
    y_early = (early_count >= min_events).astype(int)
    y_late = (late_count >= 1).astype(int)
    past = early_count / len(early)

    coords_m = project_coords(np.asarray(coords), cfg["aoi"]["crs"])
    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)

    per_seed = []
    for seed in seeds:
        train_mask, _, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        train = train_mask.numpy().astype(bool)
        test = test_mask.numpy().astype(bool)
        if (len(np.unique(y_late[test])) < 2
                or len(np.unique(y_early[train])) < 2):
            continue
        scores = {
            "temporal": _fit_tabular(kind, X, y_early, train, seed)
                        .predict_proba(X[test])[:, 1],
            "all_events": _fit_tabular(kind, X, y_all, train, seed)
                          .predict_proba(X[test])[:, 1],
            "past_flooding": past[test],
        }
        per_seed.append({"seed": seed, "n_test": int(test.sum()),
                         "scores": {k: _metrics(y_late[test], v)
                                    for k, v in scores.items()}})

    summary = {}
    for name in ("temporal", "all_events", "past_flooding"):
        aucs = np.array([r["scores"][name]["auc_roc"] for r in per_seed])
        lifts = np.array([r["scores"][name]["ap_lift"] for r in per_seed])
        if len(aucs):
            summary[name] = {
                "auc_mean": float(aucs.mean()),
                "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
                "ap_lift_mean": float(lifts.mean()),
                "n_seeds": int(len(aucs)),
            }
    for ref in ("past_flooding", "all_events"):
        diff = np.array([r["scores"]["temporal"]["auc_roc"]
                         - r["scores"][ref]["auc_roc"] for r in per_seed])
        if len(diff) > 1:
            summary[f"temporal_minus_{ref}"] = {
                "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                "seeds_temporal_ahead": int((diff > 0).sum()),
                "n_seeds": int(len(diff)),
                "p_value": float(stats.ttest_1samp(diff, 0.0).pvalue),
            }

    result = {
        "model": kind, "cutoff_year": cutoff_year, "seeds": list(seeds),
        "train_events": [e["name"] for e in early],
        "test_events": [e["name"] for e in late],
        "train_min_events": int(min_events),
        "train_positive_rate": float(y_early.mean()),
        "test_positive_rate": float(y_late.mean()),
        "per_seed": per_seed, "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temporal_holdout.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'temporal_holdout.json'}")
    return result
