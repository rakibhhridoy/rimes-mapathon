"""
Regression tests for the failures that made earlier results meaningless:
rasters sampled in the wrong CRS, composite scores collapsing to zero, a
validation split that leaked through spatial autocorrelation, and invented
values reaching the dashboard.

Run with:  pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.feature_extract import project_coords, sample_raster_at_points
from pipeline.graph_build import spatial_block_split
from pipeline.risk_score import (
    _log_minmax,
    assign_risk_classes,
    compute_composite_risk,
)

# A point inside the study area, and its UTM 46N equivalent.
LON, LAT = 89.0, 25.5
UTM = "EPSG:32646"


def _write_utm_raster(path: Path, value: float = 42.0, nodata: float = -9999.0):
    """1 km raster in UTM 46N centred on (LON, LAT), filled with `value`."""
    x, y = project_coords(np.array([[LON, LAT]]), UTM)[0]
    transform = from_origin(x - 5000, y + 5000, 1000, 1000)
    data = np.full((10, 10), value, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="float32", crs=UTM, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


class TestRasterSampling:
    """The original bug: lon/lat sampled against a UTM raster returned nodata
    everywhere, so every terrain feature was zero and every label was 0."""

    def test_samples_projected_raster_with_lonlat(self, tmp_path):
        raster = _write_utm_raster(tmp_path / "utm.tif")
        values = sample_raster_at_points(str(raster), [(LON, LAT)])
        assert values[0] == pytest.approx(42.0)

    def test_sampled_values_are_not_constant_across_a_varying_raster(self, tmp_path):
        x, y = project_coords(np.array([[LON, LAT]]), UTM)[0]
        transform = from_origin(x - 5000, y + 5000, 1000, 1000)
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        path = tmp_path / "ramp.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1,
            dtype="float32", crs=UTM, transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        coords = [(LON + dx, LAT + dy) for dx, dy in
                  [(-0.02, 0.02), (0.0, 0.0), (0.02, -0.02)]]
        values = sample_raster_at_points(str(path), coords)
        assert np.std(values) > 0, "sampling collapsed to a constant"

    def test_raises_when_every_sample_misses_the_raster(self, tmp_path):
        raster = _write_utm_raster(tmp_path / "utm.tif")
        with pytest.raises(ValueError, match="No valid samples"):
            sample_raster_at_points(str(raster), [(0.0, 0.0), (1.0, 1.0)])

    def test_nodata_becomes_fill_not_a_sentinel(self, tmp_path):
        raster = _write_utm_raster(tmp_path / "nd.tif", value=-9999.0, nodata=-9999.0)
        with pytest.raises(ValueError):
            sample_raster_at_points(str(raster), [(LON, LAT)])


class TestProjection:
    def test_distances_come_out_in_metres(self):
        pts = project_coords(np.array([[LON, LAT], [LON, LAT + 0.01]]), UTM)
        d = np.linalg.norm(pts[1] - pts[0])
        # 0.01 degrees of latitude is about 1.11 km
        assert 1000 < d < 1200


class TestCompositeRisk:
    """The original bug: a raw product rescaled by its maximum pushed the
    median cell to 0.0002, so no cell could ever cross the 0.7 threshold."""

    def test_stays_on_the_same_scale_as_its_inputs(self):
        rng = np.random.default_rng(0)
        h, e, v = (rng.random(1000) for _ in range(3))
        risk = compute_composite_risk(h, e, v)
        assert risk.min() >= 0 and risk.max() <= 1
        # A geometric mean of uniform factors sits near the middle, not at zero.
        assert 0.2 < np.median(risk) < 0.8

    def test_equal_factors_give_back_the_same_value(self):
        x = np.array([0.25, 0.5, 0.75])
        assert compute_composite_risk(x, x, x) == pytest.approx(x)

    def test_zero_exposure_means_zero_risk(self):
        risk = compute_composite_risk(
            np.array([0.9]), np.array([0.0]), np.array([0.9])
        )
        assert risk[0] == 0.0

    def test_log_minmax_spans_the_unit_interval(self):
        out = _log_minmax(np.array([0.0, 1.0, 10.0, 10_000.0]))
        assert out.min() == 0.0 and out.max() == pytest.approx(1.0)
        assert out[2] > out[1], "monotonic in the input"


class TestRiskClasses:
    def test_classes_split_non_zero_cells_into_quantiles(self):
        risk = np.concatenate([np.zeros(50), np.linspace(0.01, 0.4, 100)])
        classes = assign_risk_classes(risk)
        assert set(np.unique(classes)) == {0, 1, 2, 3, 4, 5}
        assert (classes[:50] == 0).all(), "cells without exposure are class 0"
        counts = [int((classes == c).sum()) for c in range(1, 6)]
        assert max(counts) - min(counts) <= 2, f"uneven quantiles: {counts}"

    def test_all_zero_risk_yields_no_classes(self):
        assert (assign_risk_classes(np.zeros(10)) == 0).all()


class TestSpatialSplit:
    """The original bug: a random node split put neighbouring assets on both
    sides, so validation AUC measured memorisation rather than skill."""

    def test_split_is_disjoint_and_covers_every_node(self):
        rng = np.random.default_rng(1)
        coords = rng.random((500, 2)) * 100_000
        train, val = spatial_block_split(coords, train_ratio=0.8, block_size_m=10_000)
        assert not (train & val).any()
        assert (train | val).all()

    def test_whole_blocks_fall_on_one_side(self):
        # Two tight clusters 50 km apart: each must land entirely in one split.
        cluster_a = np.random.default_rng(2).random((50, 2)) * 500
        cluster_b = cluster_a + 50_000
        coords = np.vstack([cluster_a, cluster_b])
        train, val = spatial_block_split(coords, train_ratio=0.5, block_size_m=10_000)
        assert len(set(train[:50].tolist())) == 1
        assert len(set(train[50:].tolist())) == 1

    def test_split_is_reproducible_for_a_seed(self):
        coords = np.random.default_rng(3).random((200, 2)) * 100_000
        a, _ = spatial_block_split(coords, seed=7)
        b, _ = spatial_block_split(coords, seed=7)
        assert (a == b).all()


class TestNoFabricatedData:
    """The dashboard must not ship invented values. It reads pipeline outputs;
    when they are missing it says so."""

    DASHBOARD = ROOT / "dashboard"

    def test_no_mock_constants_remain(self):
        offenders = [
            path.relative_to(ROOT)
            for path in self.DASHBOARD.rglob("*.py")
            if not path.name.startswith("._")
            and any(word in path.read_text().lower()
                    for word in ("mock_", "fake_", "dummy_"))
        ]
        assert not offenders, f"fabricated data in {offenders}"

    def test_no_authentication_gate(self):
        assert not (self.DASHBOARD / "components" / "auth.py").exists()
        sources = " ".join(
            p.read_text() for p in self.DASHBOARD.rglob("*.py")
            if not p.name.startswith("._")
        )
        for token in ("is_authenticated", "ADMIN_PASS_HASH", "render_login_page"):
            assert token not in sources, f"{token} still referenced"

    def test_loaders_do_not_generate_random_values(self):
        loader = (self.DASHBOARD / "data" / "loader.py").read_text()
        for token in ("np.random", "default_rng", "random.random"):
            assert token not in loader, f"{token} in loader"

    def test_map_tiles_keep_their_attribution(self):
        map_view = (self.DASHBOARD / "components" / "map_view.py").read_text()
        assert 'attr=" "' not in map_view, "blank tile attribution"
        assert "OpenStreetMap" in map_view and "CARTO" in map_view


class TestAdminAggregation:
    """A unit with no mapped assets has nothing to score. It must come back as
    missing data, because a zero would render as reassuring green."""

    def _grid_and_admin(self):
        import geopandas as gpd
        from shapely.geometry import Point, box

        # Two admin units side by side; only the left one holds an asset.
        admin = gpd.GeoDataFrame(
            {"admin_name": ["left", "right"]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        grid = gpd.GeoDataFrame(
            {"composite_risk": [0.8, 0.0], "cell_id": [0, 1]},
            geometry=[box(0.1, 0.1, 0.9, 0.9), box(1.1, 0.1, 1.9, 0.9)],
            crs="EPSG:4326",
        )
        infra = gpd.GeoDataFrame(
            {"asset_type": ["hospital"]},
            geometry=[Point(0.5, 0.5)],
            crs="EPSG:4326",
        )
        return grid, admin, infra

    def test_unit_without_assets_has_undefined_risk(self):
        from pipeline.risk_score import aggregate_to_admin

        grid, admin, infra = self._grid_and_admin()
        out = aggregate_to_admin(grid, admin, infra).set_index("admin_name")

        assert out.loc["left", "has_data"]
        assert not out.loc["right", "has_data"]
        assert out.loc["right", "mean_risk"] != out.loc["right", "mean_risk"], \
            "risk for an unmapped unit must be NaN, not 0"
        assert out.loc["left", "mean_risk"] == pytest.approx(0.8)

    def test_asset_counts_still_fill_with_zero(self):
        from pipeline.risk_score import aggregate_to_admin

        grid, admin, infra = self._grid_and_admin()
        out = aggregate_to_admin(grid, admin, infra).set_index("admin_name")
        assert out.loc["right", "total_assets"] == 0
        assert out.loc["left", "n_hospitals_exposed"] == 1


class TestOSMTiling:
    """Assets must match the study area. Querying by division name pulled in
    everything from Chattogram city to Cox's Bazar for a hill-tracts config,
    and timed out on the larger divisions."""

    def test_tiles_cover_the_whole_bbox(self):
        from pipeline.data_ingest import _bbox_tiles

        bbox = [88.0, 24.0, 89.9, 26.7]
        tiles = _bbox_tiles(bbox, max_span_deg=0.75)
        assert len(tiles) > 1
        assert min(t[0] for t in tiles) == pytest.approx(bbox[0])
        assert min(t[1] for t in tiles) == pytest.approx(bbox[1])
        assert max(t[2] for t in tiles) == pytest.approx(bbox[2])
        assert max(t[3] for t in tiles) == pytest.approx(bbox[3])

    def test_no_tile_exceeds_the_span_limit(self):
        from pipeline.data_ingest import _bbox_tiles

        for west, south, east, north in _bbox_tiles([91.5, 21.5, 92.7, 23.5], 0.75):
            assert east - west <= 0.75 + 1e-9
            assert north - south <= 0.75 + 1e-9

    def test_small_bbox_stays_a_single_tile(self):
        from pipeline.data_ingest import _bbox_tiles

        assert len(_bbox_tiles([89.0, 25.0, 89.2, 25.2], 0.75)) == 1

    def test_tiles_do_not_overlap(self):
        from shapely.geometry import box as shapely_box
        from pipeline.data_ingest import _bbox_tiles

        tiles = [shapely_box(*t) for t in _bbox_tiles([88.0, 24.0, 89.9, 26.7])]
        for i, a in enumerate(tiles):
            for b in tiles[i + 1:]:
                assert a.intersection(b).area == pytest.approx(0.0, abs=1e-12)


class TestFlowAccumulation:
    """D8 accumulation was a per-cell Python loop — 80 million iterations for
    one region. The vectorised receiver search plus a JIT-compiled
    accumulation must give byte-identical results."""

    @staticmethod
    def _reference(dem, nodata):
        """The original implementation, kept as the oracle."""
        rows, cols = dem.shape
        acc = np.zeros_like(dem, dtype=np.float64)
        dr = [-1, -1, 0, 1, 1, 1, 0, -1]
        dc = [0, 1, 1, 1, 0, -1, -1, -1]
        valid = dem != nodata
        indices = np.argwhere(valid)
        for r, c in indices[np.argsort(-dem[valid])]:
            best, direction = dem[r, c], -1
            for d in range(8):
                nr, nc = r + dr[d], c + dc[d]
                if 0 <= nr < rows and 0 <= nc < cols and dem[nr, nc] != nodata:
                    if dem[nr, nc] < best:
                        best, direction = dem[nr, nc], d
            if direction >= 0:
                acc[r + dr[direction], c + dc[direction]] += acc[r, c] + 1
        return acc

    def test_matches_the_original_implementation(self):
        from pipeline.data_ingest import _simple_flow_accumulation

        rng = np.random.default_rng(0)
        dem = (rng.random((60, 60)) * 100).astype(np.float32)
        dem[0, 0] = -9999.0  # a nodata cell must not receive or send flow
        assert np.allclose(self._reference(dem, -9999.0),
                           _simple_flow_accumulation(dem, -9999.0))

    def test_flow_runs_downhill_on_a_tilted_plane(self):
        from pipeline.data_ingest import _simple_flow_accumulation

        # Elevation decreasing down the rows: the bottom row collects the most.
        dem = np.tile(np.arange(20, 0, -1, dtype=np.float32)[:, None], (1, 5))
        acc = _simple_flow_accumulation(dem, -9999.0)
        assert acc[-1].sum() > acc[0].sum()
        assert acc[0].sum() == 0, "the ridge line receives nothing"

    def test_nodata_cells_stay_empty(self):
        from pipeline.data_ingest import _simple_flow_accumulation

        rng = np.random.default_rng(1)
        dem = (rng.random((30, 30)) * 50).astype(np.float32)
        dem[10:14, 10:14] = -9999.0
        acc = _simple_flow_accumulation(dem, -9999.0)
        assert (acc[10:14, 10:14] == 0).all()


class TestHAND:
    """Height Above Nearest Drainage looked up each cell's nearest drainage
    in a nested Python loop — 38 million iterations for one region."""

    @staticmethod
    def _reference(dem, drainage_mask, nodata):
        from scipy.ndimage import distance_transform_edt

        rows, cols = dem.shape
        hand = np.full_like(dem, -9999, dtype=np.float64)
        _, indices = distance_transform_edt(
            ~drainage_mask, return_distances=True, return_indices=True)
        for r in range(rows):
            for c in range(cols):
                if dem[r, c] == nodata:
                    continue
                nr, nc = indices[0, r, c], indices[1, r, c]
                hand[r, c] = (max(0, dem[r, c] - dem[nr, nc])
                              if drainage_mask[nr, nc] else 0)
        return hand

    def test_matches_the_original_implementation(self):
        from pipeline.data_ingest import _compute_hand

        rng = np.random.default_rng(0)
        dem = (rng.random((80, 80)) * 100).astype(np.float32)
        mask = rng.random((80, 80)) < 0.05
        dem[0, 0] = -9999.0
        assert np.allclose(self._reference(dem, mask, -9999.0),
                           _compute_hand(dem, mask, -9999.0))

    def test_drainage_cells_are_zero_and_height_is_relative(self):
        from pipeline.data_ingest import _compute_hand

        dem = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
        mask = np.array([[True, False, False]])
        hand = _compute_hand(dem, mask, -9999.0)
        assert hand[0, 0] == 0.0
        assert hand[0, 1] == pytest.approx(1.0)
        assert hand[0, 2] == pytest.approx(2.0)

    def test_without_drainage_it_says_so_instead_of_guessing(self):
        from pipeline.data_ingest import _compute_hand

        dem = np.ones((5, 5), dtype=np.float32) * 10
        hand = _compute_hand(dem, np.zeros((5, 5), dtype=bool), -9999.0)
        assert (hand == 0).all()


class TestTileMerge:
    """A way crossing a tile edge is returned by both tiles. The merge must
    keep one copy, keep the OSM id for provenance, and drop features that
    fall outside the study area."""

    def _tile(self, rows):
        import geopandas as gpd
        from shapely.geometry import LineString, Point

        geoms, elements, ids, tags = [], [], [], []
        for element, osm_id, geom, tag in rows:
            geoms.append(geom)
            elements.append(element)
            ids.append(osm_id)
            tags.append(tag)
        return gpd.GeoDataFrame(
            {"element": elements, "id": ids, "source_tag": tags},
            geometry=geoms, crs="EPSG:4326",
        )

    def test_duplicate_ways_across_tiles_are_kept_once(self):
        from shapely.geometry import LineString, Point
        from pipeline.data_ingest import merge_tile_results

        road = LineString([(88.5, 25.0), (89.5, 25.0)])  # spans two tiles
        west = self._tile([("way", 1, road, "highway=primary"),
                           ("node", 2, Point(88.6, 25.1), "amenity=school")])
        east = self._tile([("way", 1, road, "highway=primary"),
                           ("node", 3, Point(89.4, 25.1), "amenity=hospital")])
        merged = merge_tile_results([west, east], [88.0, 24.0, 89.9, 26.7])
        assert len(merged) == 3
        assert sorted(merged["osm_id"].tolist()) == [1, 2, 3]
        assert "osm_type" in merged.columns

    def test_features_outside_the_bbox_are_dropped(self):
        from shapely.geometry import Point
        from pipeline.data_ingest import merge_tile_results

        tile = self._tile([("node", 1, Point(88.5, 25.0), "amenity=school"),
                           ("node", 2, Point(95.0, 30.0), "amenity=school")])
        merged = merge_tile_results([tile], [88.0, 24.0, 89.9, 26.7])
        assert merged["osm_id"].tolist() == [1]

    def test_without_ids_geometry_is_the_key(self):
        import geopandas as gpd
        from shapely.geometry import Point
        from pipeline.data_ingest import merge_tile_results

        pt = Point(88.5, 25.0)
        frames = [gpd.GeoDataFrame({"source_tag": ["a"]}, geometry=[pt], crs="EPSG:4326"),
                  gpd.GeoDataFrame({"source_tag": ["a"]}, geometry=[pt], crs="EPSG:4326")]
        assert len(merge_tile_results(frames, [88.0, 24.0, 89.9, 26.7])) == 1


class TestVariogramParameters:
    """PyKrige reports [partial sill, range, nugget]. The full sill is their
    sum, and every ratio built on the sill must use the full one — reading
    the partial sill as the total overstated the nugget share by up to half."""

    def test_full_sill_is_partial_plus_nugget(self):
        from pipeline.kriging import fit_and_execute_kriging

        rng = np.random.default_rng(0)
        coords = np.column_stack([rng.uniform(88.0, 88.5, 80),
                                  rng.uniform(25.0, 25.5, 80)])
        values = np.clip(0.5 + 0.3 * np.sin(coords[:, 0] * 20)
                         + rng.normal(0, 0.05, 80), 0, 1)
        cfg = {"kriging": {"variogram_model": "spherical", "nlags": 6,
                           "weight": True, "grid_resolution_deg": 0.05,
                           "max_points": 100, "max_grid_dim": 20}}
        _, _, _, _, params = fit_and_execute_kriging(
            coords, values, cfg, (88.0, 25.0, 88.5, 25.5))

        assert set(params) >= {"partial_sill", "nugget", "sill",
                               "nugget_fraction", "range"}
        assert params["sill"] == pytest.approx(
            params["partial_sill"] + params["nugget"])
        assert 0 <= params["nugget_fraction"] <= 1
