"""
Tests for the three-way spatial split and isotonic calibration.

The class-weighted loss inflates every sigmoid score, so raw scores rank
assets but do not estimate a probability. Calibration must be fitted on
blocks the model never trained on, and the reported metrics must come from
blocks that neither training nor calibration touched.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.graph_build import build_spatial_graph, spatial_block_split_three
from pipeline.gnn_model import evaluate_model, fit_calibrator, train_model


class TestThreeWaySplit:
    def test_masks_partition_every_node(self):
        coords = np.random.default_rng(0).random((600, 2)) * 100_000
        train, calib, test = spatial_block_split_three(coords, 0.8, 10_000)
        total = train.int() + calib.int() + test.int()
        assert (total == 1).all(), "every node in exactly one split"
        assert calib.sum() > 0 and test.sum() > 0

    def test_calibration_and_test_never_share_a_block(self):
        rng = np.random.default_rng(1)
        coords = rng.random((800, 2)) * 100_000
        train, calib, test = spatial_block_split_three(coords, 0.7, 10_000)
        blocks = np.floor(coords / 10_000).astype(int)
        keys = [tuple(b) for b in blocks]
        calib_blocks = {k for k, c in zip(keys, calib.numpy()) if c}
        test_blocks = {k for k, t in zip(keys, test.numpy()) if t}
        assert not (calib_blocks & test_blocks)


class TestCalibration:
    def _graph(self):
        """A small graph whose label depends on one feature, so it is learnable."""
        rng = np.random.default_rng(2)
        n = 1500
        coords = np.column_stack([rng.uniform(88.0, 88.9, n), rng.uniform(25.0, 25.9, n)])
        signal = rng.normal(size=n)
        X = np.column_stack([signal, rng.normal(size=(n, 3))])
        y = (signal + rng.normal(scale=0.5, size=n) > 1.0).astype(np.float32)
        cfg = {"aoi": {"crs": "EPSG:32646"},
               "graph": {"k_neighbors": 4, "max_edge_distance_m": 50_000,
                         "edge_weight": "inverse_distance", "train_split": 0.7,
                         "block_size_m": 10_000},
               "gnn": {"hidden_channels": 16, "dropout": 0.1, "learning_rate": 0.01,
                       "epochs": 60, "patience": 15, "seed": 0}}
        return build_spatial_graph(X, coords, y, cfg), cfg

    def test_graph_carries_calibration_and_test_masks(self):
        graph, _ = self._graph()
        assert "calib_mask" in graph and "test_mask" in graph
        assert not (graph.calib_mask & graph.test_mask).any()
        assert torch.equal(graph.val_mask, graph.calib_mask), \
            "early stopping uses the calibration blocks, not the test blocks"

    def test_calibration_improves_brier_without_changing_ranking(self):
        graph, cfg = self._graph()
        model = train_model(graph, cfg)
        calibrator = fit_calibrator(model, graph)
        assert calibrator is not None

        metrics = evaluate_model(model, graph, calibrator)
        assert metrics["n_calibration"] > 0 and metrics["n_val"] > 0
        assert metrics["val_brier_calibrated"] <= metrics["val_brier"] + 1e-6, \
            "isotonic calibration should not worsen the Brier score"
        assert metrics["val_brier_calibrated"] <= metrics["val_brier_base_rate"] + 0.05

        # Monotone mapping: the ranking is unchanged apart from ties.
        raw = np.linspace(0.01, 0.99, 50)
        mapped = calibrator.predict(raw)
        assert (np.diff(mapped) >= -1e-9).all()

    def test_single_class_calibration_blocks_yield_no_calibrator(self):
        graph, cfg = self._graph()
        graph.y[graph.calib_mask] = 0.0
        model = train_model(graph, cfg)
        assert fit_calibrator(model, graph) is None


class TestThreading:
    """Two OpenMP runtimes are loaded in this environment and PyTorch's
    threaded reductions crashed on tensors above 32,768 elements. The CLI
    pins a single thread before torch is imported."""

    def test_cli_pins_openmp_threads_before_importing_torch(self):
        source = (ROOT / "pipeline" / "cli.py").read_text()
        pin = source.index('os.environ.setdefault("OMP_NUM_THREADS"')
        assert pin < source.index("import click")

    def test_large_boolean_mask_reduces_without_crashing(self):
        # Above torch's 32,768-element grain size, where the crash occurred.
        coords = np.random.default_rng(5).random((40_000, 2)) * 200_000
        train, calib, test = spatial_block_split_three(coords, 0.8, 10_000)
        assert int(train.sum()) + int(calib.sum()) + int(test.sum()) == 40_000
