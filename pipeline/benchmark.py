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


def corrected_p_value(diff, test_over_train) -> float | None:
    """Two-sided p-value of a mean paired difference over block assignments.

    The assignments reuse the same assets, so their differences are not
    independent and a plain t-test overstates the evidence. The corrected
    resampled t-test of Nadeau and Bengio (2003) widens the variance by the
    ratio of test to training size, t = mean / sqrt((1/J + n_test/n_train)
    * var), with J - 1 degrees of freedom.
    """
    from scipy import stats

    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return None
    var = diff.var(ddof=1)
    if var == 0:
        return 0.0 if diff.mean() != 0 else 1.0
    ratio = float(np.mean(test_over_train))
    t = diff.mean() / np.sqrt((1.0 / len(diff) + ratio) * var)
    return float(2 * stats.t.sf(abs(t), len(diff) - 1))


def _uncorrected_p_value(diff) -> float | None:
    from scipy import stats

    return float(stats.ttest_1samp(diff, 0.0).pvalue) if len(diff) > 1 else None


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

        row = {"seed": seed, "n_test": int(test.sum()), "n_train": int(train.sum()),
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
            paired["p_value"] = corrected_p_value(
                diff, [r["n_test"] / r["n_train"] for r in per_seed])
            paired["p_value_uncorrected"] = _uncorrected_p_value(diff)
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
        row = {"seed": seed, "n_test": int(test.sum()), "n_train": int(train.sum()),
               "labels": {}}
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
    both = [r for r in per_seed if len(r["labels"]) == 2]
    paired = [r["labels"]["observed"]["auc_roc"] - r["labels"]["proxy"]["auc_roc"]
              for r in both]
    if paired:
        summary["observed_minus_proxy"] = {
            "mean": float(np.mean(paired)),
            "sd": float(np.std(paired, ddof=1)) if len(paired) > 1 else None,
            "seeds_observed_ahead": int(sum(d > 0 for d in paired)),
            "n_seeds": int(len(paired)),
        }
        if len(paired) > 1:
            summary["observed_minus_proxy"]["p_value"] = corrected_p_value(
                paired, [r["n_test"] / r["n_train"] for r in both])
            summary["observed_minus_proxy"]["p_value_uncorrected"] = _uncorrected_p_value(paired)

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

    Three references are scored on the same assets. "past_flooding" is the
    share of earlier events in which the ground flooded, which is the map a
    planner would have without any model. "all_events" is the same model
    trained on labels that include the later events, the spatial-only design,
    and gives the ceiling the temporal test is measured against.
    "proxy_trained" is the same model trained on the terrain-threshold labels.
    In the spatial label comparison the observed-label model is scored against
    its own label definition and the proxy model against another; here both
    are scored against floods neither was trained on, which makes the label
    comparison fair to the proxy.
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
    proxy_path = processed_dir / "flood_proxy_labels.tif"
    y_proxy = ((sample_raster_at_points(str(proxy_path), points) > 0.5).astype(int)
               if proxy_path.exists() else None)

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
        if y_proxy is not None and len(np.unique(y_proxy[train])) == 2:
            scores["proxy_trained"] = (_fit_tabular(kind, X, y_proxy, train, seed)
                                       .predict_proba(X[test])[:, 1])
        per_seed.append({"seed": seed, "n_test": int(test.sum()),
                         "n_train": int(train.sum()),
                         "scores": {k: _metrics(y_late[test], v)
                                    for k, v in scores.items()}})

    summary = {}
    for name in ("temporal", "all_events", "past_flooding", "proxy_trained"):
        rows = [r["scores"][name] for r in per_seed if name in r["scores"]]
        aucs = np.array([m["auc_roc"] for m in rows])
        lifts = np.array([m["ap_lift"] for m in rows])
        if len(aucs):
            summary[name] = {
                "auc_mean": float(aucs.mean()),
                "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
                "ap_lift_mean": float(lifts.mean()),
                "n_seeds": int(len(aucs)),
            }
    for ref in ("past_flooding", "all_events", "proxy_trained"):
        rows = [r for r in per_seed if ref in r["scores"]]
        diff = np.array([r["scores"]["temporal"]["auc_roc"]
                         - r["scores"][ref]["auc_roc"] for r in rows])
        if len(diff) > 1:
            summary[f"temporal_minus_{ref}"] = {
                "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                "seeds_temporal_ahead": int((diff > 0).sum()),
                "n_seeds": int(len(diff)),
                "p_value": corrected_p_value(diff, [r["n_test"] / r["n_train"] for r in rows]),
                "p_value_uncorrected": _uncorrected_p_value(diff),
            }

    result = {
        "model": kind, "cutoff_year": cutoff_year, "seeds": list(seeds),
        "train_events": [e["name"] for e in early],
        "test_events": [e["name"] for e in late],
        "train_min_events": int(min_events),
        "train_positive_rate": float(y_early.mean()),
        "proxy_positive_rate": None if y_proxy is None else float(y_proxy.mean()),
        "test_positive_rate": float(y_late.mean()),
        "per_seed": per_seed, "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temporal_holdout.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'temporal_holdout.json'}")
    return result


def run_persistence_check(cfg: dict, processed_dir: Path, raw_dir: Path,
                          output_dir: Path, infra_path: Path, seeds=SEEDS,
                          cutoff_year: int = 2022) -> dict:
    """Does the record beat the model only because radar repeats its errors?

    The record of earlier flooding and the later floods it is scored against
    come from the same change detection, so ground that radar mistakes for
    water every monsoon, such as wet paddy, would count as flooded in both
    and favour the record. The temporal test is therefore repeated on two
    stricter sets of test assets, with the model and the blocks unchanged:
    "not_always_flooded" leaves out ground flagged in every earlier event,
    where a repeated error would sit, and "outside_recorded_water" leaves out
    ground the JRC record has ever seen under water, which removes seasonal
    wetland. Both remove genuine flood-prone ground as well, so the check is
    conservative: it can only take ground away from the record.
    """
    import geopandas as gpd
    from scipy import stats

    from pipeline.asset_model import _fit_tabular, model_type
    from pipeline.feature_extract import (compute_centroids, extract_features,
                                          project_coords, sample_raster_at_points)
    from pipeline.graph_build import spatial_block_split_three

    kind = model_type(cfg)
    if kind == "graph_sage":
        raise ValueError("Persistence check expects a tabular asset model.")

    events = (cfg.get("sentinel1") or {}).get("events", [])
    early = [e for e in events if int(e["start"][:4]) <= cutoff_year]
    late = [e for e in events if int(e["start"][:4]) > cutoff_year]
    if not early or not late:
        raise ValueError(f"Need events on both sides of {cutoff_year}.")

    infra = gpd.read_file(str(infra_path))
    X, coords, _, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    centroids = compute_centroids(gpd.read_file(str(infra_path)))
    points = list(zip(centroids["lon"], centroids["lat"]))

    def flooded_count(evts):
        masks = [sample_raster_at_points(str(raw_dir / f"s1_flood_{e['name']}.tif"),
                                         points) for e in evts]
        return np.sum([np.nan_to_num(m) > 0.5 for m in masks], axis=0)

    early_count, late_count = flooded_count(early), flooded_count(late)
    # The same labels as the temporal test, so the "all" set reproduces it.
    min_events = min((cfg.get("data", {}).get("labels", {}) or {})
                     .get("min_events", 1), len(early))
    y_early = (early_count >= min_events).astype(int)
    y_late = (late_count >= 1).astype(int)
    past = early_count / len(early)
    occurrence = np.nan_to_num(sample_raster_at_points(
        str(raw_dir / "jrc_water_occurrence.tif"), points))
    subsets = {
        "all": np.ones(len(y_late), dtype=bool),
        "not_always_flooded": early_count < len(early),
        "outside_recorded_water": occurrence <= 0,
    }

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
        model_scores = _fit_tabular(kind, X, y_early, train, seed).predict_proba(X)[:, 1]
        row = {"seed": seed, "n_train": int(train.sum())}
        for name, keep in subsets.items():
            idx = test & keep
            if len(np.unique(y_late[idx])) < 2:
                continue
            row[name] = {"n_test": int(idx.sum()),
                         "positive_rate": float(y_late[idx].mean()),
                         "temporal": _metrics(y_late[idx], model_scores[idx])["auc_roc"],
                         "past_flooding": _metrics(y_late[idx], past[idx])["auc_roc"]}
        per_seed.append(row)

    summary = {}
    for name, keep in subsets.items():
        rows = [dict(r[name], n_train=r["n_train"]) for r in per_seed if name in r]
        if len(rows) < 2:
            continue
        model_auc = np.array([r["temporal"] for r in rows])
        past_auc = np.array([r["past_flooding"] for r in rows])
        diff = model_auc - past_auc
        summary[name] = {
            "share_of_assets": float(keep.mean()),
            "share_of_late_positives": float(y_late[keep].sum() / max(y_late.sum(), 1)),
            "temporal_auc_mean": float(model_auc.mean()),
            "past_flooding_auc_mean": float(past_auc.mean()),
            "temporal_minus_past": {
                "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                "seeds_temporal_ahead": int((diff > 0).sum()),
                "n_seeds": int(len(diff)),
                "p_value": corrected_p_value(diff, [r["n_test"] / r["n_train"] for r in rows]),
                "p_value_uncorrected": _uncorrected_p_value(diff),
            },
        }

    result = {"model": kind, "cutoff_year": cutoff_year, "seeds": list(seeds),
              "train_events": [e["name"] for e in early],
              "test_events": [e["name"] for e in late],
              "per_seed": per_seed, "summary": summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "persistence_check.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'persistence_check.json'}")
    return result


def run_past_flooding_feature(cfg: dict, processed_dir: Path, raw_dir: Path,
                              output_dir: Path, infra_path: Path, seeds=SEEDS,
                              cutoff_year: int = 2022) -> dict:
    """Does the record of earlier flooding improve the model as a feature?

    The feature has to come from floods before the ones the model learns to
    predict, or the model simply copies it. Training therefore predicts the
    last event up to `cutoff_year` from the events before it, and the test
    predicts the later events from all the events up to `cutoff_year`, on
    held-out blocks. The feature is the share of those earlier events in which
    the ground flooded, so it means the same thing in training and test.

    Three rankings are scored against the later floods: the model with the
    feature, the same model without it trained on the same target, and the
    past-flooding share on its own.
    """
    import geopandas as gpd
    from scipy import stats

    from pipeline.asset_model import _fit_tabular, model_type
    from pipeline.feature_extract import (compute_centroids, extract_features,
                                          project_coords, sample_raster_at_points)
    from pipeline.graph_build import spatial_block_split_three

    kind = model_type(cfg)
    if kind == "graph_sage":
        raise ValueError("Past-flooding feature test expects a tabular model.")

    events = sorted((cfg.get("sentinel1") or {}).get("events", []),
                    key=lambda e: e["start"])
    early = [e for e in events if int(e["start"][:4]) <= cutoff_year]
    late = [e for e in events if int(e["start"][:4]) > cutoff_year]
    if len(early) < 2 or not late:
        raise ValueError(f"Need two events up to {cutoff_year} and one after; "
                         f"got {len(early)} and {len(late)}.")
    # The last pre-cutoff year is the training target, every event before it
    # the training history. A year can hold more than one event.
    target_year = int(early[-1]["start"][:4])
    train_target = [e for e in early if int(e["start"][:4]) == target_year]
    train_history = [e for e in early if int(e["start"][:4]) < target_year]
    if not train_history:
        raise ValueError("No events before the training target year.")

    infra = gpd.read_file(str(infra_path))
    X, coords, _, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    centroids = compute_centroids(gpd.read_file(str(infra_path)))
    points = list(zip(centroids["lon"], centroids["lat"]))
    cache = {}

    def flooded(event):
        if event["name"] not in cache:
            m = sample_raster_at_points(
                str(raw_dir / f"s1_flood_{event['name']}.tif"), points)
            cache[event["name"]] = (np.nan_to_num(m) > 0.5).astype(int)
        return cache[event["name"]]

    def share(evts):
        return np.mean([flooded(e) for e in evts], axis=0)

    def any_of(evts):
        return (np.sum([flooded(e) for e in evts], axis=0) >= 1).astype(int)

    y_train, y_test = any_of(train_target), any_of(late)
    past_train, past_test = share(train_history), share(early)
    # Standardise the feature on the training history, as the others are.
    mu, sd = past_train.mean(), past_train.std() or 1.0
    X_train = np.column_stack([X, (past_train - mu) / sd]).astype(np.float32)
    X_test = np.column_stack([X, (past_test - mu) / sd]).astype(np.float32)

    coords_m = project_coords(np.asarray(coords), cfg["aoi"]["crs"])
    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)

    per_seed = []
    for seed in seeds:
        train_mask, _, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        train = train_mask.numpy().astype(bool)
        test = test_mask.numpy().astype(bool)
        if len(np.unique(y_test[test])) < 2 or len(np.unique(y_train[train])) < 2:
            continue
        with_past = _fit_tabular(kind, X_train, y_train, train, seed)
        without = _fit_tabular(kind, X, y_train, train, seed)
        scores = {
            "with_past": with_past.predict_proba(X_test[test])[:, 1],
            "without_past": without.predict_proba(X[test])[:, 1],
            "past_only": past_test[test],
        }
        row = {"seed": seed, "n_test": int(test.sum()), "n_train": int(train.sum()),
               "scores": {k: _metrics(y_test[test], v) for k, v in scores.items()}}
        # Ground no earlier flood reached, where past flooding cannot rank at
        # all: how well the model with the feature ranks it there.
        unseen = past_test[test] == 0
        if len(np.unique(y_test[test][unseen])) == 2:
            from sklearn.metrics import roc_auc_score
            row["unseen"] = {
                "auc_roc": float(roc_auc_score(y_test[test][unseen],
                                               scores["with_past"][unseen])),
                "share_of_assets": float(unseen.mean()),
                "share_of_positives": float(y_test[test][unseen].sum()
                                            / max(y_test[test].sum(), 1)),
            }
        per_seed.append(row)

    summary = {}
    unseen = [r["unseen"] for r in per_seed if "unseen" in r]
    if unseen:
        summary["with_past_on_unseen_ground"] = {
            k: float(np.mean([u[k] for u in unseen])) for k in unseen[0]}
        summary["with_past_on_unseen_ground"]["n_seeds"] = len(unseen)
    for name in ("with_past", "without_past", "past_only"):
        aucs = np.array([r["scores"][name]["auc_roc"] for r in per_seed])
        lifts = np.array([r["scores"][name]["ap_lift"] for r in per_seed])
        if len(aucs):
            summary[name] = {
                "auc_mean": float(aucs.mean()),
                "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
                "ap_lift_mean": float(lifts.mean()),
                "n_seeds": int(len(aucs)),
            }
    for ref in ("past_only", "without_past"):
        diff = np.array([r["scores"]["with_past"]["auc_roc"]
                         - r["scores"][ref]["auc_roc"] for r in per_seed])
        if len(diff) > 1:
            summary[f"with_past_minus_{ref}"] = {
                "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                "seeds_ahead": int((diff > 0).sum()), "n_seeds": int(len(diff)),
                "p_value": corrected_p_value(diff, [r["n_test"] / r["n_train"] for r in per_seed]),
                "p_value_uncorrected": _uncorrected_p_value(diff),
            }

    result = {
        "model": kind, "cutoff_year": cutoff_year, "seeds": list(seeds),
        "train_history": [e["name"] for e in train_history],
        "train_target": [e["name"] for e in train_target],
        "test_history": [e["name"] for e in early],
        "test_target": [e["name"] for e in late],
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "per_seed": per_seed, "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "past_flooding_feature.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'past_flooding_feature.json'}")
    return result


# Features that describe the same thing, permuted together so that one cannot
# stand in for another while it is shuffled.
FEATURE_GROUPS = {
    "terrain": ("elevation", "slope", "twi", "hand", "flow_acc"),
    "access": ("dist_hospital", "dist_school", "dist_shelter", "dist_road",
               "dist_water"),
    "population": ("pop_density",),
    "asset_type": ("asset_type_code",),
}


def run_feature_importance(cfg: dict, processed_dir: Path, output_dir: Path,
                           infra_path: Path, seeds=SEEDS, repeats: int = 5) -> dict:
    """What the default model relies on, measured on the held-out blocks.

    For each block assignment the configured model is fitted on the training
    blocks, and each feature, then each group of related features, is shuffled
    across the test assets; the loss in AUC is its importance. Shuffling a
    feature on its own understates it when a correlated feature carries the
    same information, which is why the groups are also shuffled together.
    """
    import geopandas as gpd
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from pipeline.asset_model import _fit_tabular, model_type
    from pipeline.feature_extract import extract_features, project_coords
    from pipeline.graph_build import spatial_block_split_three

    kind = model_type(cfg)
    if kind == "graph_sage":
        raise ValueError("Feature importance expects a tabular asset model.")

    infra = gpd.read_file(str(infra_path))
    X, coords, y, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y).astype(int)
    columns = pd.read_parquet(processed_dir / "node_features.parquet").columns
    skip = {"asset_type", "name", "priority", "lon", "lat", "flood_label"}
    names = [c for c in columns if c not in skip]
    coords_m = project_coords(np.asarray(coords), cfg["aoi"]["crs"])
    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)

    sets = {f: [names.index(f)] for f in names}
    groups = {g: [names.index(f) for f in fs if f in names]
              for g, fs in FEATURE_GROUPS.items()}
    groups = {g: idx for g, idx in groups.items() if idx}

    per_seed = []
    for seed in seeds:
        train_mask, _, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        train = train_mask.numpy().astype(bool)
        test = test_mask.numpy().astype(bool)
        if len(np.unique(y[test])) < 2:
            continue
        model = _fit_tabular(kind, X, y, train, seed)
        X_test, y_test = X[test], y[test]
        base = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        rng = np.random.default_rng(seed)

        def loss(idx):
            drops = []
            for _ in range(repeats):
                shuffled = X_test.copy()
                order = rng.permutation(len(shuffled))
                shuffled[:, idx] = X_test[order][:, idx]   # one order for the set
                drops.append(base - roc_auc_score(
                    y_test, model.predict_proba(shuffled)[:, 1]))
            return float(np.mean(drops))

        per_seed.append({"seed": seed, "auc": float(base),
                         "features": {f: loss(i) for f, i in sets.items()},
                         "groups": {g: loss(i) for g, i in groups.items()}})
        logger.info(f"seed {seed}: AUC {base:.3f}")

    def summarise(kind_key, keys):
        out = {}
        for k in keys:
            drops = np.array([r[kind_key][k] for r in per_seed])
            out[k] = {"auc_loss_mean": float(drops.mean()),
                      "auc_loss_sd": float(drops.std(ddof=1)) if len(drops) > 1 else None}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["auc_loss_mean"]))

    result = {
        "model": kind, "seeds": list(seeds), "repeats": repeats,
        "groups": {g: list(fs) for g, fs in FEATURE_GROUPS.items()},
        "per_seed": per_seed,
        "summary": {"features": summarise("features", sets),
                    "groups": summarise("groups", groups),
                    "auc_mean": float(np.mean([r["auc"] for r in per_seed]))
                    if per_seed else None,
                    "n_seeds": len(per_seed)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_importance.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'feature_importance.json'}")
    return result


def run_target_variants(cfg: dict, processed_dir: Path, raw_dir: Path,
                        output_dir: Path, infra_path: Path, seeds=SEEDS,
                        cutoff_year: int = 2022) -> dict:
    """Is the record's lead an artefact of the target the model learns?

    The temporal model learns a yes/no label, flooded in at least the
    region's minimum number of earlier events, and is then compared with the
    record itself, the fraction of earlier events in which the ground
    flooded, which keeps more of the information. Two fairer targets are
    tried on the same features and blocks: "any_earlier", flooded in any
    earlier event, which matches how the later floods are counted, and
    "flood_fraction", a regression on the record's own fraction.
    "min_events" is the published target and reproduces the temporal test.
    """
    import geopandas as gpd
    from sklearn.ensemble import HistGradientBoostingRegressor

    from pipeline.asset_model import _fit_tabular, model_type
    from pipeline.feature_extract import (compute_centroids, extract_features,
                                          project_coords, sample_raster_at_points)
    from pipeline.graph_build import spatial_block_split_three

    kind = model_type(cfg)
    if kind != "gradient_boosting":
        raise ValueError("Target variants are set up for the gradient-boosted model.")

    events = (cfg.get("sentinel1") or {}).get("events", [])
    early = [e for e in events if int(e["start"][:4]) <= cutoff_year]
    late = [e for e in events if int(e["start"][:4]) > cutoff_year]
    if not early or not late:
        raise ValueError(f"Need events on both sides of {cutoff_year}.")

    infra = gpd.read_file(str(infra_path))
    X, coords, _, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    centroids = compute_centroids(gpd.read_file(str(infra_path)))
    points = list(zip(centroids["lon"], centroids["lat"]))

    def flooded_count(evts):
        masks = [sample_raster_at_points(str(raw_dir / f"s1_flood_{e['name']}.tif"),
                                         points) for e in evts]
        return np.sum([np.nan_to_num(m) > 0.5 for m in masks], axis=0)

    early_count = flooded_count(early)
    min_events = min((cfg.get("data", {}).get("labels", {}) or {})
                     .get("min_events", 1), len(early))
    y_late = (flooded_count(late) >= 1).astype(int)
    past = early_count / len(early)
    targets = {"min_events": (early_count >= min_events).astype(int),
               "any_earlier": (early_count >= 1).astype(int)}

    coords_m = project_coords(np.asarray(coords), cfg["aoi"]["crs"])
    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)

    per_seed = []
    for seed in seeds:
        train_mask, _, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        train = train_mask.numpy().astype(bool)
        test = test_mask.numpy().astype(bool)
        if len(np.unique(y_late[test])) < 2 or any(
                len(np.unique(y[train])) < 2 for y in targets.values()):
            continue
        scores = {name: _fit_tabular(kind, X, y, train, seed).predict_proba(X[test])[:, 1]
                  for name, y in targets.items()}
        regressor = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                                  random_state=seed)
        scores["flood_fraction"] = regressor.fit(X[train], past[train]).predict(X[test])
        scores["past_flooding"] = past[test]
        per_seed.append({"seed": seed, "n_train": int(train.sum()), "n_test": int(test.sum()),
                         "auc": {k: _metrics(y_late[test], v)["auc_roc"]
                                 for k, v in scores.items()}})

    ratios = [r["n_test"] / r["n_train"] for r in per_seed]
    summary = {"min_events": int(min_events)}
    for name in ("min_events", "any_earlier", "flood_fraction", "past_flooding"):
        aucs = np.array([r["auc"][name] for r in per_seed])
        summary[name] = {"auc_mean": float(aucs.mean()),
                         "auc_sd": float(aucs.std(ddof=1)), "n_seeds": int(len(aucs))}
        if name != "past_flooding":
            diff = aucs - np.array([r["auc"]["past_flooding"] for r in per_seed])
            summary[name]["minus_past"] = {
                "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                "seeds_model_ahead": int((diff > 0).sum()), "n_seeds": int(len(diff)),
                "p_value": corrected_p_value(diff, ratios),
                "p_value_uncorrected": _uncorrected_p_value(diff),
            }

    result = {"model": kind, "cutoff_year": cutoff_year, "seeds": list(seeds),
              "per_seed": per_seed, "summary": summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "target_variants.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'target_variants.json'}")
    return result


def run_block_size_check(cfg: dict, processed_dir: Path, raw_dir: Path,
                         output_dir: Path, infra_path: Path, seeds=SEEDS,
                         block_sizes_m=(10_000, 20_000, 40_000),
                         cutoff_year: int = 2022) -> dict:
    """Do the held-out results hold when the blocks grow?

    Assets within a block's width of a training block still share its terrain,
    so blocks smaller than the range of spatial correlation leave some of the
    advantage a random split gives. The default model's held-out AUC, and the
    temporal test of the model against the record of earlier flooding, are
    therefore repeated with larger blocks on the same seeds.
    """
    import geopandas as gpd

    from pipeline.asset_model import _fit_tabular, model_type
    from pipeline.feature_extract import (compute_centroids, extract_features,
                                          project_coords, sample_raster_at_points)
    from pipeline.graph_build import spatial_block_split_three

    kind = model_type(cfg)
    if kind == "graph_sage":
        raise ValueError("Block-size check expects a tabular asset model.")

    infra = gpd.read_file(str(infra_path))
    X, coords, y_all, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y_all = np.ascontiguousarray(y_all).astype(int)
    coords_m = project_coords(np.asarray(coords), cfg["aoi"]["crs"])
    train_ratio = cfg["graph"].get("train_split", 0.8)

    events = (cfg.get("sentinel1") or {}).get("events", [])
    early = [e for e in events if int(e["start"][:4]) <= cutoff_year]
    late = [e for e in events if int(e["start"][:4]) > cutoff_year]
    temporal = bool(early and late)
    if temporal:
        centroids = compute_centroids(gpd.read_file(str(infra_path)))
        points = list(zip(centroids["lon"], centroids["lat"]))

        def flooded_count(evts):
            masks = [sample_raster_at_points(str(raw_dir / f"s1_flood_{e['name']}.tif"),
                                             points) for e in evts]
            return np.sum([np.nan_to_num(m) > 0.5 for m in masks], axis=0)

        early_count = flooded_count(early)
        min_events = min((cfg.get("data", {}).get("labels", {}) or {})
                         .get("min_events", 1), len(early))
        y_early = (early_count >= min_events).astype(int)
        y_late = (flooded_count(late) >= 1).astype(int)
        past = early_count / len(early)

    sizes = {}
    for block_m in block_sizes_m:
        rows = []
        for seed in seeds:
            train_mask, _, test_mask = spatial_block_split_three(
                coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
            train = train_mask.numpy().astype(bool)
            test = test_mask.numpy().astype(bool)
            if len(np.unique(y_all[test])) < 2 or len(np.unique(y_all[train])) < 2:
                continue
            row = {"seed": seed, "n_train": int(train.sum()), "n_test": int(test.sum()),
                   "spatial_auc": _metrics(y_all[test], _fit_tabular(
                       kind, X, y_all, train, seed).predict_proba(X[test])[:, 1])["auc_roc"]}
            if (temporal and len(np.unique(y_late[test])) == 2
                    and len(np.unique(y_early[train])) == 2):
                row["temporal_auc"] = _metrics(y_late[test], _fit_tabular(
                    kind, X, y_early, train, seed).predict_proba(X[test])[:, 1])["auc_roc"]
                row["past_flooding_auc"] = _metrics(y_late[test], past[test])["auc_roc"]
            rows.append(row)

        spatial = np.array([r["spatial_auc"] for r in rows])
        entry = {"n_seeds": len(rows),
                 "spatial_auc_mean": float(spatial.mean()) if len(rows) else None,
                 "spatial_auc_sd": float(spatial.std(ddof=1)) if len(rows) > 1 else None,
                 "test_share_mean": float(np.mean([r["n_test"] / len(y_all) for r in rows]))
                 if rows else None}
        trows = [r for r in rows if "temporal_auc" in r]
        if len(trows) > 1:
            diff = np.array([r["temporal_auc"] - r["past_flooding_auc"] for r in trows])
            entry["temporal_auc_mean"] = float(np.mean([r["temporal_auc"] for r in trows]))
            entry["past_flooding_auc_mean"] = float(np.mean([r["past_flooding_auc"] for r in trows]))
            entry["temporal_minus_past"] = {
                "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                "seeds_temporal_ahead": int((diff > 0).sum()), "n_seeds": int(len(diff)),
                "p_value": corrected_p_value(diff, [r["n_test"] / r["n_train"] for r in trows]),
                "p_value_uncorrected": _uncorrected_p_value(diff),
            }
        entry["per_seed"] = rows
        sizes[str(block_m)] = entry

    result = {"model": kind, "seeds": list(seeds), "block_sizes_m": list(block_sizes_m),
              "sizes": sizes}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "block_size_check.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'block_size_check.json'}")
    return result


def refresh_p_values(cfg: dict, processed_dir: Path, output_dir: Path,
                     infra_path: Path) -> dict:
    """Recompute the p-values of saved comparisons with the corrected test.

    The scores stay as they are. Only the training-set size of each block
    assignment is needed, and the split is deterministic, so it is rebuilt
    from the coordinates rather than by refitting any model. Each summary
    keeps the plain t-test beside the corrected one as p_value_uncorrected.
    """
    import geopandas as gpd

    from pipeline.feature_extract import extract_features, project_coords
    from pipeline.graph_build import spatial_block_split_three

    infra = gpd.read_file(str(infra_path))
    _, coords, _, _ = extract_features(cfg, infra, processed_dir)
    coords_m = project_coords(np.asarray(coords), cfg["aoi"]["crs"])
    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)
    n_train = {}

    def train_size(seed):
        if seed not in n_train:
            train_mask, _, _ = spatial_block_split_three(
                coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
            n_train[seed] = int(train_mask.numpy().astype(bool).sum())
        return n_train[seed]

    def ratios(rows):
        return [r["n_test"] / train_size(r["seed"]) for r in rows]

    def update(entry, diff, rows):
        entry["p_value"] = corrected_p_value(diff, ratios(rows))
        entry["p_value_uncorrected"] = _uncorrected_p_value(diff)
        entry.pop("t_statistic", None)

    refreshed = {}

    path = output_dir / "benchmark.json"
    if path.exists():
        data = json.loads(path.read_text())
        paired = data["summary"].get("graph_vs_best_baseline")
        if paired:
            rows = data["per_seed"]
            best = paired["best_baseline"]
            diff = np.array([r["models"]["graph_sage"]["auc_roc"]
                             - r["models"][best]["auc_roc"] for r in rows])
            update(paired, diff, rows)
        path.write_text(json.dumps(data, indent=2))
        refreshed["benchmark"] = paired.get("p_value") if paired else None

    path = output_dir / "label_comparison.json"
    if path.exists():
        data = json.loads(path.read_text())
        entry = data["summary"].get("observed_minus_proxy")
        if entry:
            rows = [r for r in data["per_seed"] if len(r["labels"]) == 2]
            diff = np.array([r["labels"]["observed"]["auc_roc"]
                             - r["labels"]["proxy"]["auc_roc"] for r in rows])
            update(entry, diff, rows)
        path.write_text(json.dumps(data, indent=2))
        refreshed["labels"] = entry.get("p_value") if entry else None

    path = output_dir / "temporal_holdout.json"
    if path.exists():
        data = json.loads(path.read_text())
        for ref in ("past_flooding", "all_events", "proxy_trained"):
            entry = data["summary"].get(f"temporal_minus_{ref}")
            if not entry:
                continue
            rows = [r for r in data["per_seed"] if ref in r["scores"]]
            diff = np.array([r["scores"]["temporal"]["auc_roc"]
                             - r["scores"][ref]["auc_roc"] for r in rows])
            update(entry, diff, rows)
            refreshed[f"temporal_minus_{ref}"] = entry["p_value"]
        path.write_text(json.dumps(data, indent=2))

    path = output_dir / "past_flooding_feature.json"
    if path.exists():
        data = json.loads(path.read_text())
        for ref in ("past_only", "without_past"):
            entry = data["summary"].get(f"with_past_minus_{ref}")
            if not entry:
                continue
            rows = data["per_seed"]
            diff = np.array([r["scores"]["with_past"]["auc_roc"]
                             - r["scores"][ref]["auc_roc"] for r in rows])
            update(entry, diff, rows)
            refreshed[f"with_past_minus_{ref}"] = entry["p_value"]
        path.write_text(json.dumps(data, indent=2))

    path = output_dir / "persistence_check.json"
    if path.exists():
        data = json.loads(path.read_text())
        for name, entry in data["summary"].items():
            rows = [dict(r[name], seed=r["seed"]) for r in data["per_seed"] if name in r]
            diff = np.array([r["temporal"] - r["past_flooding"] for r in rows])
            update(entry["temporal_minus_past"], diff, rows)
            refreshed[f"persistence_{name}"] = entry["temporal_minus_past"]["p_value"]
        path.write_text(json.dumps(data, indent=2))

    logger.info(f"Refreshed p-values in {output_dir}: {refreshed}")
    return refreshed


def run_hazard_surface_heldout(cfg: dict, processed_dir: Path, raw_dir: Path,
                               output_dir: Path, infra_path: Path, seeds=SEEDS,
                               cell_m: float = 500.0, rows_per_block: int = 8) -> dict:
    """The terrain hazard surface scored only on ground it was not fitted on.

    The published surface is a logistic regression fitted at the assets of
    the training blocks and then scored across every grid cell, training
    ground included. Here it is refitted on the training blocks of each
    assignment and scored, against observed flooding on the same ~500 m grid
    and with the same flooded-cell rule as the published validation, only on
    cells inside that assignment's test blocks. The whole-grid score of each
    refit is kept beside it for comparison.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from pipeline.country import foreign_mask
    from pipeline.data_ingest import flood_threshold_pct
    from pipeline.feature_extract import project_coords, sample_raster_at_points
    from pipeline.graph_build import spatial_block_split_three
    from pipeline.hazard_model import TERRAIN_FEATURES, _raster_paths, _sample

    paths = _raster_paths(processed_dir)
    features = [f for f in TERRAIN_FEATURES if paths[f].exists()]
    labels_cfg = cfg["data"].get("labels", {})
    label_path = processed_dir / labels_cfg.get("path", "flood_observed_labels.tif")

    infra = gpd.read_file(str(infra_path))
    points = infra.geometry.representative_point()
    coords = list(zip(points.x, points.y))
    X = _sample(paths, features, coords)
    y = (sample_raster_at_points(str(label_path), coords) > 0.5).astype(int)
    ok = np.all(np.isfinite(X), axis=1)
    coords_m = project_coords(np.column_stack([points.x, points.y]), cfg["aoi"]["crs"])
    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)
    asset_blocks = np.floor(coords_m / block_m).astype(np.int64)

    # One logistic model per assignment, written as weights on the raw
    # features so the grid can be scored for every assignment in one pass.
    fits = []
    for seed in seeds:
        train_mask, _, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        train = train_mask.numpy().astype(bool) & ok
        test = test_mask.numpy().astype(bool)
        if len(np.unique(y[train])) < 2:
            continue
        scaler = StandardScaler().fit(X[train])
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            scaler.transform(X[train]), y[train])
        w = model.coef_[0] / scaler.scale_
        b = model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_)
        fits.append({"seed": seed, "w": w, "b": b,
                     "test_blocks": {tuple(t) for t in asset_blocks[test]}})
    W = np.column_stack([f["w"] for f in fits])
    B = np.array([f["b"] for f in fits])

    sources = {name: rasterio.open(paths[name]) for name in features}
    try:
        ref = sources[features[0]]
        factor = max(1, int(round(cell_m / abs(ref.transform.a))))
        out_h, out_w = ref.height // factor, ref.width // factor
        sums = np.zeros((out_h, out_w, len(fits)), dtype=np.float64)
        counts = np.zeros((out_h, out_w), dtype=np.float64)
        for row_start in range(0, out_h * factor, rows_per_block * factor):
            rows = min(rows_per_block * factor, out_h * factor - row_start)
            window = rasterio.windows.Window(0, row_start, out_w * factor, rows)
            bands, invalid = [], None
            for name in features:
                src = sources[name]
                data = src.read(1, window=window).astype(np.float32)
                bad = ~np.isfinite(data) | (data < -9000)
                if src.nodata is not None:
                    bad |= data == src.nodata
                invalid = bad if invalid is None else (invalid | bad)
                bands.append(np.nan_to_num(data, nan=0.0))
            stack = np.stack(bands, axis=-1).reshape(-1, len(features))
            # rows_per_block is in output cells; keep it small, since every
            # assignment is scored at once at full resolution.
            logit = np.clip(stack @ W.astype(np.float32) + B.astype(np.float32), -50, 50)
            probs = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
            probs[invalid.ravel()] = 0.0
            good = (~invalid).astype(np.float64)
            r0 = row_start // factor
            nr = rows // factor
            probs = probs.reshape(nr, factor, out_w, factor, len(fits))
            good = good.reshape(nr, factor, out_w, factor)
            sums[r0:r0 + nr] += probs.sum(axis=(1, 3))
            counts[r0:r0 + nr] += good.sum(axis=(1, 3))
        transform = ref.transform * ref.transform.scale(factor, factor)
        crs = ref.crs
    finally:
        for src in sources.values():
            src.close()

    hazard = sums / np.maximum(counts, 1)[..., None]
    with rasterio.open(raw_dir / "s1_flood_frequency.tif") as src:
        observed = np.zeros((out_h, out_w), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=observed,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.average)
    valid = ((counts >= 0.5 * factor * factor) & np.isfinite(observed)
             & ~foreign_mask(cfg, (out_h, out_w), transform, crs))
    n_events = len(cfg.get("sentinel1", {}).get("events", [])) or 1
    min_events = labels_cfg.get("min_events", 1)
    flooded = observed >= flood_threshold_pct(min_events, n_events)

    # Each cell's block, from its centre in the projected grid.
    cols, rows_ = np.meshgrid(np.arange(out_w) + 0.5, np.arange(out_h) + 0.5)
    xs, ys = rasterio.transform.xy(transform, rows_.ravel() - 0.5, cols.ravel() - 0.5)
    cell_blocks = np.floor(np.column_stack([xs, ys]) / block_m).astype(np.int64)

    per_seed = []
    for i, fit in enumerate(fits):
        in_test = np.array([tuple(c) in fit["test_blocks"] for c in cell_blocks]).reshape(out_h, out_w)
        held = valid & in_test
        row = {"seed": fit["seed"], "n_test_cells": int(held.sum())}
        if len(np.unique(flooded[held])) == 2:
            row["heldout_auc"] = float(roc_auc_score(flooded[held], hazard[..., i][held]))
        if len(np.unique(flooded[valid])) == 2:
            row["whole_grid_auc"] = float(roc_auc_score(flooded[valid], hazard[..., i][valid]))
        per_seed.append(row)

    summary = {}
    for key in ("heldout_auc", "whole_grid_auc"):
        vals = np.array([r[key] for r in per_seed if key in r])
        if len(vals):
            summary[key] = {"mean": float(vals.mean()),
                            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                            "n_seeds": int(len(vals))}
    result = {"features": features, "cell_m": cell_m, "seeds": list(seeds),
              "per_seed": per_seed, "summary": summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hazard_heldout.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'hazard_heldout.json'}")
    return result


# Asset groups a planner would read the two lists for.
PLANNING_GROUPS = {
    "facilities": ("hospital", "school", "flood_shelter"),
    "transport": ("road", "bridge", "railway"),
}


def run_record_checks(cfg: dict, processed_dir: Path, raw_dir: Path,
                      output_dir: Path, infra_path: Path, seeds=SEEDS,
                      cutoff_year: int = 2022, folds: int = 10) -> dict:
    """How long a record has to be, and what the record changes for a planner.

    Record length: the record and the model are both rebuilt from every subset
    of k earlier events, for k from one to all of them, and scored on the
    held-out blocks against the later floods as in the temporal test. The
    model's label follows the region's rule capped at k, so at full length the
    two match the temporal test. A record of one event can only say flooded or
    not, so its ranking is coarse, while the model still ranks every asset.

    Planning lists: every asset is scored once out of fold, by a model trained
    on the other 10 km blocks, so the whole region has held-out scores. The
    record's own list is every asset an earlier flood reached; the model and
    the model given the record as a feature each list the same number of
    assets, their highest scores. Each list is scored by the share of the
    assets the later floods reached that it holds, overall and for facilities
    and transport links.
    """
    from itertools import combinations

    import geopandas as gpd
    from sklearn.metrics import roc_auc_score

    from pipeline.asset_model import _fit_tabular, model_type
    from pipeline.feature_extract import (compute_centroids, extract_features,
                                          project_coords, sample_raster_at_points)
    from pipeline.graph_build import spatial_block_split_three

    kind = model_type(cfg)
    if kind == "graph_sage":
        raise ValueError("Record checks expect a tabular asset model.")
    events = sorted((cfg.get("sentinel1") or {}).get("events", []),
                    key=lambda e: e["start"])
    early = [e for e in events if int(e["start"][:4]) <= cutoff_year]
    late = [e for e in events if int(e["start"][:4]) > cutoff_year]
    if len(early) < 2 or not late:
        raise ValueError(f"Need two events up to {cutoff_year} and one after.")

    infra = gpd.read_file(str(infra_path))
    X, coords, _, _ = extract_features(cfg, infra, processed_dir)
    X = np.ascontiguousarray(X, dtype=np.float32)
    centroids = compute_centroids(gpd.read_file(str(infra_path)))
    points = list(zip(centroids["lon"], centroids["lat"]))
    flooded = {e["name"]: (np.nan_to_num(sample_raster_at_points(
        str(raw_dir / f"s1_flood_{e['name']}.tif"), points)) > 0.5).astype(int)
        for e in events}

    def count(evts):
        return np.sum([flooded[e["name"]] for e in evts], axis=0)

    rule = (cfg.get("data", {}).get("labels", {}) or {}).get("min_events", 1)
    y_late = (count(late) >= 1).astype(int)
    coords_m = project_coords(np.asarray(coords), cfg["aoi"]["crs"])
    block_m = cfg["graph"].get("block_size_m", 10_000)
    train_ratio = cfg["graph"].get("train_split", 0.8)

    # --- Record length ----------------------------------------------------
    splits = []
    for seed in seeds:
        train_mask, _, test_mask = spatial_block_split_three(
            coords_m, train_ratio=train_ratio, block_size_m=block_m, seed=seed)
        test = test_mask.numpy().astype(bool)
        if len(np.unique(y_late[test])) == 2:
            splits.append((seed, train_mask.numpy().astype(bool), test))
    length = []
    for k in range(1, len(early) + 1):
        for subset in combinations(early, k):
            past = count(subset) / k
            y_k = (count(subset) >= min(rule, k)).astype(int)
            rec, mod = [], []
            for seed, train, test in splits:
                rec.append(roc_auc_score(y_late[test], past[test]))
                if len(np.unique(y_k[train])) == 2:
                    model = _fit_tabular(kind, X, y_k, train, seed)
                    mod.append(roc_auc_score(y_late[test],
                                             model.predict_proba(X[test])[:, 1]))
            length.append({"k": k, "events": [e["name"] for e in subset],
                           "record_auc": float(np.mean(rec)),
                           "model_auc": float(np.mean(mod)) if mod else None,
                           "n_seeds": len(rec)})
    by_k = {}
    for k in range(1, len(early) + 1):
        rows = [r for r in length if r["k"] == k]
        models = [r["model_auc"] for r in rows if r["model_auc"] is not None]
        by_k[str(k)] = {"record_auc": float(np.mean([r["record_auc"] for r in rows])),
                        "model_auc": float(np.mean(models)) if models else None,
                        "n_subsets": len(rows)}

    # --- Planning lists ---------------------------------------------------
    block_ids = np.floor(coords_m / block_m).astype(np.int64)
    _, block_of = np.unique(block_ids, axis=0, return_inverse=True)
    block_of = block_of.ravel()
    fold_of_block = np.random.default_rng(42).permutation(block_of.max() + 1) % folds
    fold = fold_of_block[block_of]

    record = count(early) / len(early)
    y_early = (count(early) >= min(rule, len(early))).astype(int)
    # The model with the record as a feature, trained as in the past-flooding
    # test: the last year up to the cutoff from the years before it.
    target_year = int(early[-1]["start"][:4])
    history = [e for e in early if int(e["start"][:4]) < target_year]
    target = [e for e in early if int(e["start"][:4]) == target_year]
    y_target = (count(target) >= 1).astype(int)
    past_train = count(history) / len(history)
    mu, sd = past_train.mean(), past_train.std() or 1.0
    X_hist = np.column_stack([X, (past_train - mu) / sd]).astype(np.float32)
    X_full = np.column_stack([X, (record - mu) / sd]).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_score = np.full(len(X), np.nan)
    combined_score = np.full(len(X), np.nan)
    for f in range(folds):
        test = fold == f
        train = ~test
        if not test.any():
            continue
        model_score[test] = _fit_tabular(kind, X, y_early, train, 42) \
            .predict_proba(X[test])[:, 1]
        combined_score[test] = _fit_tabular(kind, X_hist, y_target, train, 42) \
            .predict_proba(X_full[test])[:, 1]

    np.savez_compressed(output_dir / "record_checks_scores.npz", model=model_score,
                        model_with_record=combined_score, record=record,
                        flooded_later=y_late, fold=fold)
    listed = {"record": record > 0}
    n_listed = int(listed["record"].sum())
    for name, score in (("model", model_score), ("model_with_record", combined_score)):
        top = np.zeros(len(X), dtype=bool)
        top[np.argsort(-score)[:n_listed]] = True
        listed[name] = top

    types = infra["asset_type"].to_numpy()
    groups = {"all": np.ones(len(X), dtype=bool)}
    groups.update({g: np.isin(types, t) for g, t in PLANNING_GROUPS.items()})
    planning = {"n_listed": n_listed, "listed_share": n_listed / len(X), "groups": {}}
    for g, in_group in groups.items():
        hit = (y_late == 1) & in_group
        entry = {"n_assets": int(in_group.sum()), "n_flooded": int(hit.sum())}
        for name, lst in listed.items():
            entry[f"{name}_caught"] = int((lst & hit).sum())
            entry[f"{name}_caught_share"] = float((lst & hit).sum() / max(hit.sum(), 1))
        entry["model_only"] = int((listed["model"] & ~listed["record"] & hit).sum())
        entry["record_only"] = int((listed["record"] & ~listed["model"] & hit).sum())
        planning["groups"][g] = entry
    # Lists of fixed length, as a share of all assets. The record has few
    # distinct values, so its ties are broken at random.
    record_ranked = record + np.random.default_rng(0).random(len(X)) * 1e-6
    later = max(int(y_late.sum()), 1)
    planning["by_list_share"] = {}
    for share in (0.05, 0.10, 0.20, 0.30):
        n = int(round(share * len(X)))
        planning["by_list_share"][f"{share:.2f}"] = {
            name: float(y_late[np.argsort(-score)[:n]].sum() / later)
            for name, score in (("record", record_ranked), ("model", model_score),
                                ("model_with_record", combined_score))}
    # Uncertainty of the out-of-fold differences, by resampling whole 10 km
    # blocks so that neighbouring assets stay together.
    rng = np.random.default_rng(1)
    members = [np.flatnonzero(block_of == b) for b in range(block_of.max() + 1)]
    diffs = {"model_minus_record": [], "model_with_record_minus_record": []}
    for _ in range(1000):
        s = np.concatenate([members[b] for b in rng.integers(0, len(members), len(members))])
        if len(np.unique(y_late[s])) < 2:
            continue
        base = roc_auc_score(y_late[s], record[s])
        diffs["model_minus_record"].append(roc_auc_score(y_late[s], model_score[s]) - base)
        diffs["model_with_record_minus_record"].append(
            roc_auc_score(y_late[s], combined_score[s]) - base)
    planning["block_bootstrap"] = {
        name: {"mean": float(np.mean(v)), "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])],
               "n": len(v)} for name, v in diffs.items()}
    planning["model_auc_out_of_fold"] = float(roc_auc_score(y_late, model_score))
    planning["record_auc_region"] = float(roc_auc_score(y_late, record))
    planning["model_with_record_auc_out_of_fold"] = float(
        roc_auc_score(y_late, combined_score))

    result = {"model": kind, "cutoff_year": cutoff_year, "seeds": list(seeds),
              "early_events": [e["name"] for e in early],
              "late_events": [e["name"] for e in late], "label_rule": rule,
              "record_length": {"subsets": length, "by_k": by_k},
              "planning": planning}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "record_checks.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {output_dir / 'record_checks.json'}")
    return result
