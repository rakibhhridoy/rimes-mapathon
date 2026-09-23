"""
Tests for the Sentinel-1 label path and the validation metrics.

The Earth Engine calls themselves need credentials and a project, so these
tests cover everything around them: configuration errors surface clearly,
observed labels are built from the frequency raster correctly, and the
metrics behave as claimed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.data_ingest import build_observed_flood_labels
from pipeline.sentinel1 import init_ee
from pipeline.validate import _binary_agreement, _metrics


def _write(path: Path, data: np.ndarray, crs="EPSG:4326", nodata=-9999.0):
    transform = from_origin(88.0, 26.7, 0.01, 0.01)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


class TestEarthEngineConfig:
    def test_missing_project_explains_how_to_set_it(self, monkeypatch):
        monkeypatch.delenv("EARTHENGINE_PROJECT", raising=False)
        with pytest.raises(RuntimeError, match="No Earth Engine project set"):
            init_ee({"earthengine": {"project": ""}})

    def test_environment_variable_is_accepted(self, monkeypatch):
        monkeypatch.setenv("EARTHENGINE_PROJECT", "some-project")
        # Initialisation will fail on credentials, not on a missing project.
        with pytest.raises(RuntimeError) as excinfo:
            init_ee({})
        assert "No Earth Engine project set" not in str(excinfo.value)


class TestObservedLabels:
    """Labels must come from observed water, and the threshold must reflect
    how many events a pixel was flooded in."""

    def _setup(self, tmp_path, freq_pct):
        processed = tmp_path / "processed"
        raw = tmp_path / "raw"
        processed.mkdir()
        raw.mkdir()
        _write(raw / "s1_flood_frequency.tif", freq_pct)
        twi = _write(processed / "twi.tif", np.full(freq_pct.shape, 9.0))
        return processed, raw, {"twi": str(twi)}

    def test_labels_follow_the_frequency_raster(self, tmp_path):
        # Four pixels flooded in 2 of 4 events, four never flooded.
        freq = np.array([[50.0, 50.0, 0.0, 0.0]] * 4)
        processed, raw, deriv = self._setup(tmp_path, freq)
        cfg = {
            "data": {"labels": {"min_events": 1}},
            "sentinel1": {"events": [{"name": f"e{i}"} for i in range(4)]},
        }
        out = build_observed_flood_labels(cfg, deriv, processed, raw)
        with rasterio.open(out) as src:
            labels = src.read(1)
        assert labels[:, :2].all() and not labels[:, 2:].any()

    def test_min_events_raises_the_bar(self, tmp_path):
        # Flooded in 1 of 4 events = 25%; requiring 2 events excludes it.
        freq = np.array([[25.0, 25.0, 75.0, 75.0]] * 4)
        processed, raw, deriv = self._setup(tmp_path, freq)
        cfg = {
            "data": {"labels": {"min_events": 2}},
            "sentinel1": {"events": [{"name": f"e{i}"} for i in range(4)]},
        }
        with rasterio.open(build_observed_flood_labels(cfg, deriv, processed, raw)) as src:
            labels = src.read(1)
        assert not labels[:, :2].any(), "25% should fail a 2-of-4 threshold"
        assert labels[:, 2:].all(), "75% should pass it"

    def test_empty_flood_map_is_an_error_not_a_label_of_zeros(self, tmp_path):
        processed, raw, deriv = self._setup(tmp_path, np.zeros((4, 4)))
        cfg = {"data": {"labels": {"min_events": 1}},
               "sentinel1": {"events": [{"name": "e0"}]}}
        with pytest.raises(RuntimeError, match="No pixels flagged"):
            build_observed_flood_labels(cfg, deriv, processed, raw)

    def test_missing_frequency_raster_names_the_command_to_run(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        (tmp_path / "raw").mkdir()
        twi = _write(processed / "twi.tif", np.full((4, 4), 9.0))
        with pytest.raises(FileNotFoundError, match="sentinel1"):
            build_observed_flood_labels(
                {"data": {"labels": {}}, "sentinel1": {"events": []}},
                {"twi": str(twi)}, processed, tmp_path / "raw",
            )


class TestMetrics:
    def test_perfect_ranking_scores_one(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert _metrics(labels, scores)["auc_roc"] == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert _metrics(labels, scores)["auc_roc"] == pytest.approx(0.0)

    def test_lift_compares_precision_to_the_base_rate(self):
        # 5% positives, ranked perfectly: AP 1.0 is a 20x lift.
        labels = np.concatenate([np.ones(5), np.zeros(95)])
        scores = np.concatenate([np.ones(5), np.zeros(95)])
        assert _metrics(labels, scores)["ap_lift"] == pytest.approx(20.0, rel=0.01)

    def test_single_class_sample_reports_a_note_not_a_number(self):
        out = _metrics(np.zeros(10), np.linspace(0, 1, 10))
        assert "auc_roc" not in out and "note" in out


class TestBinaryAgreement:
    def test_perfect_agreement(self):
        mask = np.array([1, 1, 0, 0])
        out = _binary_agreement(mask, mask)
        assert out["pod"] == 1.0 and out["far"] == 0.0 and out["csi"] == 1.0

    def test_over_prediction_shows_up_as_false_alarms(self):
        observed = np.array([1, 0, 0, 0])
        predicted = np.array([1, 1, 1, 0])
        out = _binary_agreement(observed, predicted)
        assert out["pod"] == 1.0, "the flood was detected"
        assert out["far"] == pytest.approx(2 / 3), "but two thirds was false"
        assert out["csi"] == pytest.approx(1 / 3)

    def test_missed_flood_lowers_pod(self):
        out = _binary_agreement(np.array([1, 1, 1, 0]), np.array([1, 0, 0, 0]))
        assert out["pod"] == pytest.approx(1 / 3)
        assert out["misses"] == 2


class TestLandslideAggregation:
    """Zonal statistics silently return nothing when the zones are in a
    different CRS from the raster — every upazila came back at 0.000."""

    def _susceptibility(self, tmp_path):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        from pipeline.feature_extract import project_coords

        # A projected raster (UTM 46N) with a clear west-to-east gradient.
        x, y = project_coords(np.array([[92.0, 22.5]]), "EPSG:32646")[0]
        path = tmp_path / "susc.tif"
        data = np.tile(np.linspace(0.1, 0.9, 20, dtype=np.float32), (20, 1))
        with rasterio.open(
            path, "w", driver="GTiff", height=20, width=20, count=1,
            dtype="float32", crs="EPSG:32646", nodata=-9999.0,
            transform=from_origin(x - 10_000, y + 10_000, 1000, 1000),
        ) as dst:
            dst.write(data, 1)
        return path, (x, y)

    def test_zones_in_wgs84_are_reprojected_onto_a_utm_raster(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import box

        from pipeline.landslide import aggregate_to_admin

        path, (x, y) = self._susceptibility(tmp_path)
        # Two zones, west and east halves, defined in WGS84.
        utm = gpd.GeoSeries(
            [box(x - 10_000, y - 10_000, x, y + 10_000),
             box(x, y - 10_000, x + 10_000, y + 10_000)],
            crs="EPSG:32646",
        ).to_crs("EPSG:4326")
        admin = gpd.GeoDataFrame(
            {"admin_name": ["west", "east"], "admin_label": ["west", "east"]},
            geometry=utm, crs="EPSG:4326",
        )
        admin_path = tmp_path / "admin.gpkg"
        admin.to_file(admin_path, driver="GPKG")

        rows = aggregate_to_admin(path, str(admin_path), None,
                                  tmp_path / "out.json")
        by_name = {r["admin_name"]: r for r in rows}
        assert by_name["west"]["n_pixels"] > 0
        assert by_name["east"]["susceptibility_mean"] > by_name["west"]["susceptibility_mean"]

    def test_no_overlap_is_an_error_not_a_table_of_zeros(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import box

        from pipeline.landslide import aggregate_to_admin

        path, _ = self._susceptibility(tmp_path)
        far_away = gpd.GeoDataFrame(
            {"admin_name": ["elsewhere"]},
            geometry=[box(0, 0, 1, 1)], crs="EPSG:4326",
        )
        admin_path = tmp_path / "far.gpkg"
        far_away.to_file(admin_path, driver="GPKG")
        with pytest.raises(RuntimeError, match="no raster pixels"):
            aggregate_to_admin(path, str(admin_path), None, tmp_path / "out.json")
