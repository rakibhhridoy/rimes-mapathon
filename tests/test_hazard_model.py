"""
Tests for the terrain hazard surface.

Kriging the GNN scores produced a nearly flat hazard layer where flood-prone
ground is not contiguous (Rangpur/Rajshahi: nugget 0.88 of the sill, kriging
variance equal to the sill). The terrain model exists to vary at DEM
resolution instead, and must refuse to fit rather than invent a surface when
its inputs are missing or degenerate.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.hazard_model import TERRAIN_FEATURES, _raster_paths, fit_hazard_model


def _raster(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs="EPSG:4326",
        transform=from_origin(89.0, 25.1, 0.01, 0.01), nodata=-9999.0,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


class TestFeatureSet:
    def test_only_terrain_is_used(self):
        # Distance to a hospital describes exposure, not how likely the
        # ground is to flood; it must not leak into the hazard layer.
        assert TERRAIN_FEATURES == ["elevation", "slope", "twi", "hand", "flow_acc"]
        for name in TERRAIN_FEATURES:
            assert "dist" not in name

    def test_derivatives_are_looked_up_in_their_subdirectory(self, tmp_path):
        paths = _raster_paths(tmp_path)
        assert paths["elevation"].parent == tmp_path
        assert paths["slope"].parent.name == "dem_derivatives"


class TestFitting:
    def _setup(self, tmp_path, label_values):
        import geopandas as gpd
        from shapely.geometry import Point

        processed = tmp_path / "processed"
        rng = np.random.default_rng(0)
        for name in TERRAIN_FEATURES:
            path = _raster_paths(processed)[name]
            _raster(path, rng.random((10, 10)) * 10)
        _raster(processed / "flood_proxy_labels.tif", label_values)

        pts = [Point(89.0 + 0.01 * i + 0.005, 25.1 - 0.01 * j - 0.005)
               for i in range(10) for j in range(10)]
        infra = tmp_path / "infra.gpkg"
        gpd.GeoDataFrame({"asset_type": ["school"] * len(pts)},
                         geometry=pts, crs="EPSG:4326").to_file(infra, driver="GPKG")

        cfg = {
            "aoi": {"crs": "EPSG:32646"},
            "data": {"labels": {"source": "proxy"}},
            "graph": {"train_split": 0.8, "block_size_m": 1000},
            "gnn": {"seed": 42},
        }
        return cfg, processed, infra

    def test_single_class_labels_are_refused(self, tmp_path):
        cfg, processed, infra = self._setup(tmp_path, np.zeros((10, 10)))
        with pytest.raises(ValueError, match="single-class"):
            fit_hazard_model(cfg, processed, tmp_path / "out", infra)

    def test_missing_labels_name_the_file(self, tmp_path):
        cfg, processed, infra = self._setup(tmp_path, np.ones((10, 10)))
        (processed / "flood_proxy_labels.tif").unlink()
        with pytest.raises(FileNotFoundError, match="Labels not found"):
            fit_hazard_model(cfg, processed, tmp_path / "out", infra)

    def test_fit_reports_features_and_coefficients(self, tmp_path):
        labels = np.zeros((10, 10))
        labels[:5, :] = 1  # a spatially coherent positive class
        cfg, processed, infra = self._setup(tmp_path, labels)

        fitted = fit_hazard_model(cfg, processed, tmp_path / "out", infra)
        metrics = fitted["metrics"]
        assert metrics["features"] == TERRAIN_FEATURES
        assert set(metrics["standardised_coefficients"]) == set(TERRAIN_FEATURES)
        assert metrics["label_source"] == "proxy"
        assert (tmp_path / "out" / "hazard_model.json").exists()

    def test_predictions_are_probabilities(self, tmp_path):
        from pipeline.hazard_model import predict_hazard_at

        labels = np.zeros((10, 10))
        labels[:5, :] = 1
        cfg, processed, infra = self._setup(tmp_path, labels)
        fitted = fit_hazard_model(cfg, processed, tmp_path / "out", infra)

        coords = [(89.02, 25.05), (89.05, 25.02)]
        probs = predict_hazard_at(fitted, processed, coords)
        assert probs.shape == (2,)
        assert ((probs >= 0) & (probs <= 1)).all()
