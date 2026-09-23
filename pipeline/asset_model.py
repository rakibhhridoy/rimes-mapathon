"""
The model that scores each asset, chosen by configuration.

The system was built around a graph neural network, but a comparison on
identical features, assets and blocks showed a gradient-boosted tree matching
or beating it in every region (pipeline/benchmark.py). The model is therefore
a setting rather than an assumption, and `asset_model.type` selects it.

Whatever the model, the contract is the same: fit on the training blocks,
calibrate on the calibration blocks, score every asset, and report metrics on
the test blocks that neither step has seen.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

TYPES = ("gradient_boosting", "random_forest", "logistic_regression", "graph_sage")

# Settings of the default model, shared with the chain figure so the diagram
# cannot drift from the code.
BOOST_ITERATIONS = 300
BOOST_LEARNING_RATE = 0.1
FOREST_TREES = 300


def model_type(cfg: dict) -> str:
    chosen = (cfg.get("asset_model") or {}).get("type", "gradient_boosting")
    if chosen not in TYPES:
        raise ValueError(f"asset_model.type must be one of {TYPES}, got '{chosen}'")
    return chosen


def _estimator(kind: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if kind == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=BOOST_ITERATIONS, learning_rate=BOOST_LEARNING_RATE,
            random_state=seed)
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=FOREST_TREES, min_samples_leaf=5, class_weight="balanced",
            n_jobs=1, random_state=seed)
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def _fit_tabular(kind: str, X, y, train, seed: int):
    from sklearn.utils.class_weight import compute_sample_weight

    estimator = _estimator(kind, seed)
    if kind == "gradient_boosting":     # no class_weight parameter of its own
        estimator.fit(X[train], y[train],
                      sample_weight=compute_sample_weight("balanced", y[train]))
    else:
        estimator.fit(X[train], y[train])
    return estimator


def _metrics(y_true, scores, calibrated=None) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)

    base = float(y_true.mean())
    out = {
        "n_test": int(len(y_true)),
        "val_positive_rate": base,
        "val_auc_roc": float(roc_auc_score(y_true, scores)),
        "val_average_precision": float(average_precision_score(y_true, scores)),
        "val_brier": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
        "val_brier_base_rate": float(brier_score_loss(
            y_true, np.full_like(y_true, base, dtype=float))),
    }
    out["val_ap_lift"] = out["val_average_precision"] / base if base else None
    if calibrated is not None:
        out["val_brier_calibrated"] = float(brier_score_loss(y_true, calibrated))
    return out


def fit_asset_model(graph_data, cfg: dict):
    """Fit the configured model and score every asset.

    Returns (scores, probability, metrics, fitted), where `scores` rank assets,
    `probability` is the isotonic-calibrated score, and `fitted` is the model
    object (a torch module for the graph model, an sklearn estimator otherwise).
    """
    kind = model_type(cfg)
    seed = int((cfg.get("gnn") or {}).get("seed", 42))
    y = graph_data.y.numpy().astype(int)
    train = graph_data.train_mask.numpy().astype(bool)
    calib = graph_data.calib_mask.numpy().astype(bool)
    test = graph_data.test_mask.numpy().astype(bool)

    if kind == "graph_sage":
        import torch

        from pipeline.gnn_model import (evaluate_model, extract_embeddings_and_scores,
                                        fit_calibrator, train_model)

        model = train_model(graph_data, cfg)
        calibrator = fit_calibrator(model, graph_data)
        metrics = evaluate_model(model, graph_data, calibrator)
        _, scores = extract_embeddings_and_scores(model, graph_data)
        probability = calibrator.predict(scores) if calibrator is not None else None
        metrics["model_type"] = kind
        return scores, probability, metrics, model

    from sklearn.isotonic import IsotonicRegression

    X = graph_data.x.numpy()
    fitted = _fit_tabular(kind, X, y, train, seed)
    scores = fitted.predict_proba(X)[:, 1]

    calibrator = None
    if calib.sum() and len(np.unique(y[calib])) > 1:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(scores[calib], y[calib])
    probability = calibrator.predict(scores) if calibrator is not None else None

    metrics = _metrics(y[test], scores[test],
                       probability[test] if probability is not None else None)
    metrics.update({
        "model_type": kind,
        "n_train": int(train.sum()),
        "n_calibration": int(calib.sum()),
        "n_val": int(test.sum()),
        "seed": seed,
    })
    logger.info(f"{kind}: test AUC {metrics['val_auc_roc']:.3f}, "
                f"AP {metrics['val_average_precision']:.3f}")
    return scores, probability, metrics, fitted
