"""
Tests for region configuration and the two exposure readings.

Region configs inherit from the base config, so a change to the shared model
settings must reach every region; and a region without results must not be
presented as if it had them.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.data.regions import (
    REGION_CONFIGS,
    has_results,
    region_config,
    region_label,
    region_paths,
    region_status,
)
from pipeline.cli import _load_config, _merge
from pipeline.risk_score import compute_population_exposure_grid


class TestConfigInheritance:
    def test_region_config_inherits_shared_model_settings(self):
        base = _load_config(str(ROOT / "config.yaml"))
        sylhet = _load_config(str(ROOT / "configs/sylhet.yaml"))
        assert sylhet["graph"]["k_neighbors"] == base["graph"]["k_neighbors"]
        assert sylhet["risk"]["exposure_weights"] == base["risk"]["exposure_weights"]

    def test_region_config_overrides_what_it_declares(self):
        sylhet = _load_config(str(ROOT / "configs/sylhet.yaml"))
        assert sylhet["aoi"]["name"] == "sylhet"
        assert sylhet["paths"]["output_dir"] == "data/sylhet/output"
        assert sylhet["aoi"]["bbox"] != _load_config(
            str(ROOT / "config.yaml"))["aoi"]["bbox"]

    def test_merge_is_recursive_and_override_wins(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        out = _merge(base, {"a": {"b": 9}})
        assert out == {"a": {"b": 9, "c": 2}, "d": 3}
        assert base["a"]["b"] == 1, "merge must not mutate the base"

    def test_every_registered_region_has_a_config(self):
        for region_id in REGION_CONFIGS:
            cfg = region_config(region_id)
            assert cfg["aoi"]["name"] == region_id
            assert len(cfg["aoi"]["bbox"]) == 4

    def test_landslide_model_is_enabled_only_where_intended(self):
        assert region_config("cht")["landslide"]["enabled"] is True
        assert region_config("sylhet")["landslide"]["enabled"] is False


class TestRegionPaths:
    def test_paths_are_absolute_and_region_scoped(self):
        paths = region_paths("sylhet")
        assert paths["output"].is_absolute()
        assert paths["output"].parts[-2:] == ("sylhet", "output")
        assert paths["cache"].parts[-2:] == ("sylhet", "cache")

    def test_base_region_keeps_the_original_layout(self):
        paths = region_paths("rangpur_rajshahi")
        assert paths["output"].parts[-2:] == ("data", "output")
        assert paths["cache"].parts[-2:] == ("data", "cache")

    def test_availability_follows_the_files_on_disk(self):
        # The base region has been processed in this repo; a region whose
        # directory is empty must report no results.
        assert has_results("rangpur_rajshahi")
        statuses = {r["id"]: r["available"] for r in region_status()}
        assert set(statuses) == set(REGION_CONFIGS)

    def test_label_names_the_hazard(self):
        assert "landslide" in region_label("cht").lower()


class TestPopulationExposure:
    """Infrastructure exposure and population exposure answer different
    questions; the second must not silently fall back to the first."""

    def _grid(self):
        return gpd.GeoDataFrame(
            {"cell_id": [0, 1]},
            geometry=[box(89.0, 25.0, 89.01, 25.01),
                      box(89.01, 25.0, 89.02, 25.01)],
            crs="EPSG:4326",
        )

    def test_missing_population_raster_gives_zero_not_a_guess(self):
        cfg = {"data": {"vulnerability": {"population_path": "does/not/exist.tif"}}}
        out = compute_population_exposure_grid(self._grid(), cfg)
        assert out.shape == (2,)
        assert (out == 0).all()

    def test_real_raster_is_sampled_and_scaled(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin

        path = tmp_path / "pop.tif"
        # Left cell densely populated, right cell empty.
        data = np.array([[1000.0, 0.0]], dtype=np.float32)
        with rasterio.open(
            path, "w", driver="GTiff", height=1, width=2, count=1,
            dtype="float32", crs="EPSG:4326",
            transform=from_origin(89.0, 25.01, 0.01, 0.01), nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        cfg = {"data": {"vulnerability": {"population_path": str(path)}}}
        out = compute_population_exposure_grid(self._grid(), cfg)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.0)


class TestLandslideRegionDetection:
    """A landslide region carries no ranked assets, so the dashboard must
    recognise it from the cached overlay alone. Otherwise a deployment has to
    carry the 250 MB susceptibility raster that nothing reads."""

    def test_cached_overlay_alone_marks_the_region_as_ready(self, tmp_path, monkeypatch):
        from dashboard.data import regions

        paths = {kind: tmp_path / kind for kind in ("raw", "processed", "output", "cache")}
        for path in paths.values():
            path.mkdir()
        monkeypatch.setattr(regions, "region_paths", lambda region: paths)
        assert not regions.has_results("cht")
        (paths["cache"] / "raster_landslide.json").write_text("{}")
        assert regions.has_results("cht")
