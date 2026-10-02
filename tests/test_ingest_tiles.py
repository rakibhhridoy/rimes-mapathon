"""
Tests for the OpenStreetMap fetch: a query too heavy for the server is split
into quarters, and every answer is cached so a rerun fetches only what is
missing.
"""

import geopandas as gpd
from shapely.geometry import Point

import pipeline.data_ingest as ingest


def _answer(tile):
    west, south, east, north = tile
    return gpd.GeoDataFrame({"amenity": ["school"]},
                            geometry=[Point((west + east) / 2, (south + north) / 2)],
                            crs="EPSG:4326")


def test_a_query_that_keeps_failing_is_split_and_cached(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(bbox, tags):
        calls.append(bbox)
        if bbox[2] - bbox[0] > 0.5:          # the whole tile is "too heavy"
            raise ConnectionError("server closed the connection")
        return _answer(bbox)

    monkeypatch.setattr(ingest.ox, "features_from_bbox", fake_fetch)
    monkeypatch.setattr(ingest, "RETRY_PAUSES_S", (0, 0, 0, 0))
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)

    frames, failures = ingest._fetch_query((0, 0, 1, 1), "lifeline", 0, {"amenity": "school"},
                                           ["https://example"], tmp_path, "tile 1/1")
    assert failures == []
    assert len(frames) == 4
    assert len(list(tmp_path.glob("*.pkl"))) == 4

    # A rerun goes straight to the four cached quarters and asks the server nothing.
    calls.clear()
    frames, failures = ingest._fetch_query((0, 0, 1, 1), "lifeline", 0, {"amenity": "school"},
                                           ["https://example"], tmp_path, "tile 1/1")
    assert calls == [] and len(frames) == 4


def test_a_query_that_fails_at_every_size_is_reported(tmp_path, monkeypatch):
    import time

    def always_fails(bbox, tags):
        raise ConnectionError("down")

    monkeypatch.setattr(ingest.ox, "features_from_bbox", always_fails)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    frames, failures = ingest._fetch_query((0, 0, 1, 1), "transport", 1, {"highway": "primary"},
                                           ["https://example"], tmp_path, "tile 1/1")
    assert frames == [] and len(failures) == 16      # four quarters of four quarters


def test_flood_threshold_matches_the_stored_whole_percentages():
    # one flood in three events is stored as 33, two as 66
    assert 33 >= ingest.flood_threshold_pct(1, 3)
    assert 66 >= ingest.flood_threshold_pct(2, 3)
    assert 33 < ingest.flood_threshold_pct(2, 3)
    # where 100 divides evenly the threshold is unchanged
    for m, n, exact in [(1, 4, 25), (2, 4, 50), (2, 5, 40)]:
        assert exact >= ingest.flood_threshold_pct(m, n) > exact - 1
