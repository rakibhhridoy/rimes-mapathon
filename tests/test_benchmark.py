"""
Tests for the baseline comparison and the weight-sensitivity analysis.

The comparison only means something if every model sees the same features,
the same assets and the same held-out blocks, and if the summary reports the
spread across block assignments rather than one lucky split.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.benchmark import _fit_predict, _metrics, _tabular_models
from pipeline.sensitivity import (_top_overlap, _weighted_geometric_mean,
                                  _vulnerability_scenarios)


class TestMetrics:
    def test_ap_lift_is_precision_over_the_base_rate(self):
        y = np.array([0, 0, 0, 1])
        perfect = np.array([0.1, 0.2, 0.3, 0.9])
        out = _metrics(y, perfect)
        assert out["auc_roc"] == 1.0
        # A perfect ranking has average precision 1, against a base rate of
        # 0.25, so the lift is fourfold.
        assert out["ap_lift"] == pytest.approx(4.0)

    def test_random_scores_sit_near_chance(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.2).astype(int)
        out = _metrics(y, rng.random(500))
        assert 0.4 < out["auc_roc"] < 0.6


class TestBaselines:
    def test_every_baseline_learns_a_separable_signal(self):
        """Each model must beat chance on data where one feature decides the
        label, or the comparison would be measuring a broken harness."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(600, 4)).astype(np.float32)
        y = (X[:, 0] + 0.3 * rng.normal(size=600) > 0.8).astype(int)
        train = np.zeros(600, dtype=bool)
        train[:400] = True
        test = ~train
        for name, model in _tabular_models(seed=0).items():
            scores = _fit_predict(name, model, X, y, train, test, seed=0)
            assert _metrics(y[test], scores)["auc_roc"] > 0.8, name

    def test_class_weighting_reaches_the_rare_class(self):
        """With a 2 % positive rate an unweighted model can score everything
        negative; the balanced settings must still rank the positives."""
        rng = np.random.default_rng(2)
        X = rng.normal(size=(1000, 3)).astype(np.float32)
        y = (X[:, 0] > 2.05).astype(int)
        assert 0 < y.mean() < 0.05
        train = np.zeros(1000, dtype=bool)
        train[:700] = True
        test = ~train
        for name, model in _tabular_models(seed=0).items():
            scores = _fit_predict(name, model, X, y, train, test, seed=0)
            assert len(np.unique(scores)) > 1, f"{name} gave one constant score"


class TestSensitivity:
    def test_weighted_geometric_mean_matches_the_plain_one(self):
        a = np.array([0.2, 0.5, 0.9])
        b = np.array([0.4, 0.5, 0.1])
        c = np.array([0.6, 0.5, 0.3])
        equal = _weighted_geometric_mean([a, b, c], [1, 1, 1])
        assert equal == pytest.approx(np.cbrt(a * b * c))

    def test_doubling_a_weight_pulls_towards_that_factor(self):
        a = np.array([0.9, 0.1])
        b = np.array([0.1, 0.9])
        heavy_a = _weighted_geometric_mean([a, b], [2, 1])
        equal = _weighted_geometric_mean([a, b], [1, 1])
        assert heavy_a[0] > equal[0] and heavy_a[1] < equal[1]

    def test_top_overlap_is_one_for_an_unchanged_ranking(self):
        scores = np.linspace(0, 1, 200)
        assert _top_overlap(scores, scores) == 1.0
        assert _top_overlap(scores, -scores) == 0.0

    def test_scenarios_keep_the_components_that_have_data(self):
        cfg = {"risk": {"vulnerability_weights": {
            "population_density": 0.4, "dist_hospital": 0.3,
            "night_light_proxy": 0.0}}}
        scenarios = _vulnerability_scenarios(cfg)
        equal = scenarios["equal_components"]
        assert equal["night_light_proxy"] == 0.0
        assert equal["population_density"] == pytest.approx(0.5)
        assert scenarios["population_only"]["dist_hospital"] == 0.0


class TestTemporalHoldout:
    def test_default_seeds_keep_the_earlier_five_first(self):
        from pipeline.benchmark import SEEDS
        assert SEEDS[:5] == (42, 7, 13, 21, 99)
        assert len(set(SEEDS)) == len(SEEDS) >= 20

    def test_refuses_without_events_after_the_cutoff(self, tmp_path):
        from pipeline.benchmark import run_temporal_holdout
        cfg = {"asset_model": {"type": "gradient_boosting"},
               "sentinel1": {"events": [{"name": "a", "start": "2017-08-12"},
                                        {"name": "b", "start": "2020-07-11"}]}}
        with pytest.raises(ValueError, match="both sides"):
            run_temporal_holdout(cfg, tmp_path, tmp_path, tmp_path, tmp_path / "x.gpkg")

    def test_past_feature_needs_a_history_before_its_training_target(self, tmp_path):
        from pipeline.benchmark import run_past_flooding_feature
        cfg = {"asset_model": {"type": "gradient_boosting"},
               "sentinel1": {"events": [{"name": "a", "start": "2020-07-11"},
                                        {"name": "b", "start": "2024-07-01"}]}}
        with pytest.raises(ValueError, match="two events"):
            run_past_flooding_feature(cfg, tmp_path, tmp_path, tmp_path,
                                      tmp_path / "x.gpkg")
