"""
Tests for pipeline/crosscheck.py: the Sentinel-1 masks against an independent map.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

from pipeline.crosscheck import compare_event, run_crosscheck  # noqa: E402


def _write(path, array, cell):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(90.0, 24.0, cell, cell)) as dst:
        dst.write(array.astype("uint8"), 1)


@pytest.fixture
def scene(tmp_path):
    # Reference grid: 4 x 4 cells of 0.01 degrees, all seen clear (bit 2).
    code = np.full((4, 4), 2, dtype="uint8")
    code[:2, :] += 1            # top half flooded in the reference
    code[3, 3] += 4             # one permanent-water cell, excluded
    _write(tmp_path / "gfd.tif", code, 0.01)
    # Radar mask at twice the resolution: floods the top-left quarter only.
    mask = np.zeros((8, 8), dtype="uint8")
    mask[:4, :4] = 1
    _write(tmp_path / "s1.tif", mask, 0.005)
    gpd.GeoDataFrame(geometry=[box(90.0, 23.96, 90.04, 24.0)], crs="EPSG:4326"
                     ).to_file(tmp_path / "bgd_district.gpkg")
    return tmp_path


class TestCompareEvent:
    def test_scores_the_overlap_on_the_reference_grid(self, scene):
        r = compare_event(scene / "s1.tif", scene / "gfd.tif", scene)
        assert r["n_cells"] == 15                  # permanent water left out
        assert r["hits"] == 4 and r["misses"] == 4 and r["false_alarms"] == 0
        assert r["pod"] == pytest.approx(0.5)
        assert r["csi"] == pytest.approx(0.5)
        assert r["agreement"] == pytest.approx(11 / 15)

    def test_cells_outside_bangladesh_are_left_out(self, scene):
        gpd.GeoDataFrame(geometry=[box(90.0, 23.98, 90.04, 24.0)], crs="EPSG:4326"
                         ).to_file(scene / "bgd_district.gpkg")
        r = compare_event(scene / "s1.tif", scene / "gfd.tif", scene)
        assert r["n_cells"] == 8


def test_regions_without_a_reference_event_are_refused(tmp_path):
    cfg = {"sentinel1": {"events": [{"name": "jul2019"}]}}
    with pytest.raises(ValueError, match="gfd_ids"):
        run_crosscheck(cfg, tmp_path, tmp_path)
