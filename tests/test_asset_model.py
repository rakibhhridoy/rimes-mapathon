"""
Tests for the configurable asset model.

The model that scores assets is now a setting, so the contract it must keep is
the same whichever model is chosen: fit on the training blocks, calibrate on
the calibration blocks, and report metrics only from the test blocks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.asset_model import fit_asset_model, model_type


def _toy_graph(n=900, seed=0):
    """A graph whose label depends on one feature, with disjoint masks."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    y = (x[:, 0] + 0.4 * rng.normal(size=n) > 1.0).astype(np.float32)
    masks = {}
    order = rng.permutation(n)
    masks["train"] = np.zeros(n, bool); masks["train"][order[:int(0.7 * n)]] = True
    masks["calib"] = np.zeros(n, bool); masks["calib"][order[int(0.7 * n):int(0.85 * n)]] = True
    masks["test"] = np.zeros(n, bool); masks["test"][order[int(0.85 * n):]] = True
    edge = np.vstack([np.arange(n - 1), np.arange(1, n)])
    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge.astype(np.int64)),
        y=torch.from_numpy(y),
        train_mask=torch.from_numpy(masks["train"]),
        calib_mask=torch.from_numpy(masks["calib"]),
        val_mask=torch.from_numpy(masks["calib"]),
        test_mask=torch.from_numpy(masks["test"]),
    )


def _cfg(kind):
    return {"asset_model": {"type": kind},
            "gnn": {"hidden_channels": 16, "dropout": 0.3, "learning_rate": 0.01,
                    "epochs": 20, "patience": 5, "seed": 42}}


class TestModelType:
    def test_default_is_the_tree_the_comparison_favoured(self):
        assert model_type({}) == "gradient_boosting"

    def test_unknown_type_is_refused(self):
        with pytest.raises(ValueError, match="asset_model.type"):
            model_type({"asset_model": {"type": "wishful_thinking"}})


class TestFitAssetModel:
    @pytest.mark.parametrize("kind", ["gradient_boosting", "random_forest",
                                      "logistic_regression"])
    def test_scores_every_asset_and_learns_the_signal(self, kind):
        graph = _toy_graph()
        scores, probability, metrics, _ = fit_asset_model(graph, _cfg(kind))
        assert len(scores) == graph.num_nodes
        assert len(probability) == graph.num_nodes
        assert metrics["model_type"] == kind
        assert metrics["val_auc_roc"] > 0.8

    def test_metrics_count_only_the_test_blocks(self):
        graph = _toy_graph()
        _, _, metrics, _ = fit_asset_model(graph, _cfg("gradient_boosting"))
        assert metrics["n_test"] == int(graph.test_mask.sum())
        assert metrics["n_train"] == int(graph.train_mask.sum())
        assert metrics["n_test"] != graph.num_nodes

    def test_calibration_improves_on_predicting_the_base_rate(self):
        graph = _toy_graph()
        _, _, metrics, _ = fit_asset_model(graph, _cfg("gradient_boosting"))
        assert metrics["val_brier_calibrated"] <= metrics["val_brier_base_rate"]

    def test_calibrated_probabilities_stay_in_range(self):
        graph = _toy_graph()
        _, probability, _, _ = fit_asset_model(graph, _cfg("gradient_boosting"))
        assert probability.min() >= 0.0 and probability.max() <= 1.0

    def test_graph_model_remains_available(self):
        graph = _toy_graph()
        scores, _, metrics, model = fit_asset_model(graph, _cfg("graph_sage"))
        assert metrics["model_type"] == "graph_sage"
        assert len(scores) == graph.num_nodes
        assert isinstance(model, torch.nn.Module)
