"""
Tests for pipeline/country.py: assets across a land border are dropped,
while assets offshore or on the country's side of a disputed line are kept.
"""

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import Point, box

from pipeline.country import _foreign, drop_foreign, foreign_mask


@pytest.fixture
def cfg(tmp_path):
    # Home is x 0-2; the neighbour, x 1.9-4, overlaps it by a sliver that the
    # finer home border must win. Everything above y = 2 is sea.
    home = gpd.GeoDataFrame({"shapeName": ["home"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:4326")
    neighbours = gpd.GeoDataFrame(
        {"ADM0_A3": ["HOM", "NBR"]},
        geometry=[box(0, 0, 2, 2), box(1.9, 0, 4, 2)], crs="EPSG:4326")
    home_path, neighbours_path = tmp_path / "home.geojson", tmp_path / "ne.geojson"
    home.to_file(home_path)
    neighbours.to_file(neighbours_path)
    _foreign.cache_clear()
    return {"aoi": {"bbox": [0, 0, 4, 3], "iso3": "HOM",
                    "country_boundary": str(home_path), "neighbours": str(neighbours_path)}}


def points(*xy):
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in xy], crs="EPSG:4326")


def test_assets_in_a_neighbouring_country_are_dropped(cfg):
    kept = drop_foreign(points((1, 1), (3, 1)), cfg)
    assert [p.x for p in kept.geometry] == [1]


def test_offshore_assets_are_kept(cfg):
    # Outside the home polygon but in no other country, like a char or island.
    kept = drop_foreign(points((1, 2.5)), cfg)
    assert len(kept) == 1


def test_the_home_border_wins_where_the_outlines_overlap(cfg):
    kept = drop_foreign(points((1.95, 1)), cfg)
    assert len(kept) == 1


def test_without_configuration_nothing_is_dropped():
    gdf = points((3, 1))
    assert len(drop_foreign(gdf, {"aoi": {"bbox": [0, 0, 4, 3]}})) == 1


def test_a_configured_but_missing_boundary_stops_the_run(cfg, tmp_path):
    cfg["aoi"]["neighbours"] = str(tmp_path / "absent.geojson")
    with pytest.raises(FileNotFoundError):
        drop_foreign(points((1, 1)), cfg)


def test_raster_mask_marks_only_foreign_cells(cfg):
    # Four columns one degree wide, three rows: only columns 2 and 3 (x 2-4)
    # on land rows are foreign.
    transform = from_bounds(0, 0, 4, 3, 4, 3)
    mask = foreign_mask(cfg, (3, 4), transform, "EPSG:4326")
    expected = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
    assert (mask == expected).all()


def test_land_the_coarse_border_gives_home_is_foreign_beyond_a_margin(cfg, tmp_path):
    # Natural Earth draws home a degree further west than the fine polygon.
    coarse = gpd.GeoDataFrame(
        {"ADM0_A3": ["HOM", "NBR"]},
        geometry=[box(-1, 0, 2, 2), box(1.9, 0, 4, 2)], crs="EPSG:4326")
    path = tmp_path / "coarse.geojson"
    coarse.to_file(path)
    cfg["aoi"].update(neighbours=str(path), bbox=[-1, 0, 4, 3])
    _foreign.cache_clear()
    kept = drop_foreign(points((-0.5, 1), (-0.005, 1)), cfg)
    assert [round(p.x, 3) for p in kept.geometry] == [-0.005]


def test_a_region_named_by_districts_drops_the_rest(tmp_path):
    # Two districts side by side; the region names only the western one.
    country = gpd.GeoDataFrame({"shapeName": ["West", "East"]},
                               geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326")
    world = gpd.GeoDataFrame({"ADM0_A3": ["HOM"]}, geometry=[box(0, 0, 2, 1)], crs="EPSG:4326")
    country.to_file(tmp_path / "adm2.geojson")
    world.to_file(tmp_path / "ne.geojson")
    _foreign.cache_clear()
    cfg = {"aoi": {"bbox": [0, 0, 2, 1], "iso3": "HOM", "districts": ["West"],
                   "country_boundary": str(tmp_path / "adm2.geojson"),
                   "neighbours": str(tmp_path / "ne.geojson")}}
    kept = drop_foreign(points((0.5, 0.5), (1.005, 0.5), (1.5, 0.5)), cfg)
    # the named district and its 1 km margin stay; the other district goes
    assert [round(p.x, 3) for p in kept.geometry] == [0.5, 1.005]


def test_an_unknown_district_name_stops_the_run(tmp_path):
    country = gpd.GeoDataFrame({"shapeName": ["West"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    country.to_file(tmp_path / "adm2.geojson")
    country.rename(columns={"shapeName": "ADM0_A3"}).assign(ADM0_A3="HOM").to_file(tmp_path / "ne.geojson")
    _foreign.cache_clear()
    cfg = {"aoi": {"bbox": [0, 0, 1, 1], "iso3": "HOM", "districts": ["Atlantis"],
                   "country_boundary": str(tmp_path / "adm2.geojson"),
                   "neighbours": str(tmp_path / "ne.geojson")}}
    with pytest.raises(ValueError):
        drop_foreign(points((0.5, 0.5)), cfg)

