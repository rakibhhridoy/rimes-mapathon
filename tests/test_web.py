"""
Tests for the web application's data build and API.

The web application exists because the Streamlit dashboard rebuilt and re-sent
about 17 MB of map HTML on every interaction. Its promise is that a visitor's
figures still come from the pipeline, only served instead of recomputed, so
these tests check the path from an output file to an API response.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "web" / "data" / "hazmapper.sqlite"
pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="run scripts/build_web.py to create web/data/hazmapper.sqlite")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from web.api import app

    return TestClient(app)


@pytest.fixture(scope="module")
def connection():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


class TestDatabase:
    def test_every_region_with_assets_has_them_indexed_for_search(self, connection):
        regions = connection.execute(
            "SELECT id FROM regions WHERE has_assets = 1").fetchall()
        assert regions, "no region carries assets"
        for row in regions:
            assets = connection.execute(
                "SELECT COUNT(*) FROM assets WHERE region = ?", (row["id"],)).fetchone()[0]
            indexed = connection.execute(
                "SELECT COUNT(*) FROM assets_fts WHERE region = ?", (row["id"],)).fetchone()[0]
            assert assets == indexed, f"{row['id']}: {assets} assets, {indexed} indexed"

    def test_scores_and_probabilities_stay_in_range(self, connection):
        row = connection.execute(
            "SELECT MIN(flood_risk) AS lo, MAX(flood_risk) AS hi,"
            " MAX(flood_probability) AS phi FROM assets").fetchone()
        assert 0 <= row["lo"] <= row["hi"] <= 1
        # A probability of exactly 1 is the calibration artefact the pipeline
        # corrects; if it reappears the build has picked up stale outputs.
        assert row["phi"] < 1.0

    def test_admin_rows_carry_their_source_payload(self, connection):
        row = connection.execute(
            "SELECT payload FROM admin_summary LIMIT 1").fetchone()
        assert row is not None and json.loads(row["payload"])


class TestAPI:
    def test_health_reports_the_regions_it_has(self, client):
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok" and payload["regions"] >= 1

    def test_summary_matches_the_stored_asset_count(self, client, connection):
        region = connection.execute(
            "SELECT id FROM regions WHERE has_assets = 1 LIMIT 1").fetchone()["id"]
        stored = connection.execute(
            "SELECT COUNT(*) FROM assets WHERE region = ?", (region,)).fetchone()[0]
        payload = client.get(f"/api/region/{region}/summary").json()
        assert payload["counts"]["assets"] == stored

    def test_search_finds_by_type_and_keeps_the_query_as_data(self, client, connection):
        region = connection.execute(
            "SELECT id FROM regions WHERE has_assets = 1 LIMIT 1").fetchone()["id"]
        found = client.get(f"/api/region/{region}/assets", params={"q": "bridge"}).json()
        assert found["count"] > 0
        assert all("bridge" in (a["asset_type"] or "").lower()
                   or "bridge" in (a["name"] or "").lower() for a in found["assets"])
        # FTS syntax in the query must not blow up or inject: it is text.
        hostile = client.get(f"/api/region/{region}/assets", params={"q": 'a" OR "b'})
        assert hostile.status_code == 200

    def test_unknown_region_and_asset_are_refused(self, client):
        assert client.get("/api/region/atlantis/summary").status_code == 404
        assert client.get("/api/region/atlantis/asset/1").status_code == 404

    def test_export_returns_csv_with_a_header(self, client, connection):
        region = connection.execute(
            "SELECT id FROM regions WHERE has_assets = 1 LIMIT 1").fetchone()["id"]
        response = client.get(f"/api/region/{region}/export.csv", params={"limit": 5})
        assert response.status_code == 200
        assert response.text.splitlines()[0].startswith("region,risk_rank,name")

    def test_tiles_are_served_and_traversal_is_refused(self, client, connection):
        region = connection.execute(
            "SELECT id FROM regions WHERE has_assets = 1 LIMIT 1").fetchone()["id"]
        assert client.get(f"/tiles/{region}/assets.pmtiles").status_code == 200
        assert client.get("/tiles/..%2F..%2Fetc/assets.pmtiles").status_code == 404
        assert client.get(f"/tiles/{region}/nonsense.pmtiles").status_code == 404

    def test_responses_are_cacheable(self, client):
        """Caching is the point: a visitor's second view should not recompute."""
        assert "max-age" in client.get("/api/regions").headers.get("cache-control", "")

    def test_all_view_covers_every_region(self, client, connection):
        stored = connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        payload = client.get("/api/region/all/summary").json()
        assert payload["counts"]["assets"] == stored
        # "all" is the flood view: every region with scored assets, no other
        assert {r["id"] for r in payload["regions"]} == {
            row["id"] for row in connection.execute("SELECT id FROM regions WHERE has_assets = 1")}
        # districts add up to the assets, and the frame spans every region
        assert sum(d["n"] for d in payload["counts"]["by_district"]) == stored
        west, south, east, north = payload["extent"]
        assert west < east and south < north

    def test_filtered_counts_match_the_database(self, client, connection):
        region = connection.execute(
            "SELECT id FROM regions WHERE has_assets = 1 LIMIT 1").fetchone()["id"]
        expected = connection.execute(
            "SELECT COUNT(*) FROM assets WHERE region = ? AND asset_type = 'school'"
            " AND flood_risk >= 0.7", (region,)).fetchone()[0]
        payload = client.get("/api/stats", params={
            "regions": region, "types": "school", "classes": "very_high"}).json()
        assert payload["assets"] == expected
        unfiltered = client.get("/api/stats").json()
        assert unfiltered["assets"] == connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]

    def test_filters_refuse_unknown_values(self, client):
        assert client.get("/api/stats", params={"classes": "extreme"}).status_code == 400
        assert client.get("/api/stats", params={"regions": "atlantis"}).status_code == 404

    def test_a_map_tile_rank_finds_its_asset(self, client, connection):
        row = connection.execute(
            "SELECT region, risk_rank, asset_id FROM assets LIMIT 1").fetchone()
        asset = client.get(f"/api/region/{row['region']}/rank/{row['risk_rank']}").json()
        assert asset["asset_id"] == row["asset_id"]
        assert client.get(f"/api/region/{row['region']}/rank/0").status_code == 404

    def test_page_names_its_script_by_content_and_is_revalidated(self, client):
        response = client.get("/")
        assert response.headers.get("cache-control") == "no-cache"
        assert 'src="app.js?v=' in response.text
        assert 'href="app.css?v=' in response.text


class TestLandslideAPI:
    @pytest.fixture
    def region(self, connection):
        row = connection.execute("SELECT id FROM regions WHERE has_landslide = 1 LIMIT 1").fetchone()
        if row is None:
            pytest.skip("no landslide region in this build")
        return row["id"]

    def test_summary_lists_every_upazila(self, client, connection, region):
        stored = connection.execute(
            "SELECT COUNT(*) FROM landslide_units WHERE region = ?", (region,)).fetchone()[0]
        payload = client.get(f"/api/landslide/{region}/summary").json()
        assert len(payload["units"]) == stored > 0
        assert payload["display"]["vmin"] < payload["display"]["vmax"]

    def test_counts_follow_the_district_filter(self, client, connection, region):
        district = connection.execute(
            "SELECT district FROM landslide_units WHERE region = ? LIMIT 1", (region,)).fetchone()[0]
        expected = connection.execute(
            "SELECT COUNT(*), SUM(population), SUM(n_landslides) FROM landslide_units"
            " WHERE region = ? AND district = ?", (region, district)).fetchone()
        payload = client.get(f"/api/landslide/{region}/stats", params={"districts": district}).json()
        assert (payload["upazilas"], payload["population"], payload["landslides"]) == tuple(expected)

    def test_every_landslide_counts_once(self, client, connection, region):
        mapped = connection.execute(
            "SELECT COUNT(*) FROM landslides WHERE region = ?", (region,)).fetchone()[0]
        assert client.get(f"/api/landslide/{region}/stats").json()["landslides"] == mapped

    def test_a_flood_region_has_no_landslide_view(self, client, connection):
        flood = connection.execute(
            "SELECT id FROM regions WHERE has_landslide = 0 LIMIT 1").fetchone()
        if flood:
            assert client.get(f"/api/landslide/{flood['id']}/summary").status_code == 404



class TestAreas:
    """Divisions, districts, upazilas and unions, and their figures."""

    def test_flood_levels_add_up(self, connection):
        # every asset sits in one union, so each level counts every asset once
        total = connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        for level in ("division", "district", "upazila", "union"):
            summed = connection.execute(
                "SELECT SUM(n_assets) FROM admin_units WHERE hazard = 'flood' AND level = ?",
                (level,)).fetchone()[0]
            assert summed == total, f"{level}: {summed} of {total} assets"

    def test_every_unit_knows_its_parent(self, connection):
        orphans = connection.execute(
            "SELECT COUNT(*) FROM admin_units WHERE level != 'division' AND parent IS NULL"
            " AND NOT (hazard = 'landslide' AND level = 'district')").fetchone()[0]
        assert orphans == 0

    def test_share_high_matches_the_assets(self, connection):
        row = connection.execute(
            "SELECT id, n_assets, n_high, share_high FROM admin_units"
            " WHERE hazard = 'flood' AND level = 'district' AND n_assets > 0 LIMIT 1").fetchone()
        high = connection.execute(
            "SELECT COUNT(*) FROM assets WHERE adm_district = ? AND flood_risk >= 0.5",
            (row["id"],)).fetchone()[0]
        assert high == row["n_high"]
        assert row["share_high"] == pytest.approx(high / row["n_assets"])

    def test_pickers_list_children_of_a_parent(self, client):
        divisions = client.get("/api/areas/flood?levels=division").json()["areas"]
        assert len(divisions) == 8
        upazila = client.get("/api/areas/flood?levels=upazila").json()["areas"][0]
        unions = client.get(f"/api/areas/flood?levels=union&parent={upazila['id']}").json()["areas"]
        assert unions and all(u["upazila"] == upazila["id"] for u in unions)

    def test_area_filter_narrows_the_counters(self, client):
        district = client.get("/api/area/flood/district/Kurigram").json()
        stats = client.get("/api/stats?area=district:Kurigram").json()
        assert stats["assets"] == district["n_assets"]
        assert client.get("/api/stats").json()["assets"] > stats["assets"]

    def test_area_filter_is_validated(self, client):
        assert client.get("/api/stats?area=province:Kurigram").status_code == 400
        assert client.get("/api/stats?area=district").status_code == 400
        assert client.get("/api/area/flood/district/Nowhere").status_code == 404
        assert client.get("/api/areas/flood?levels=province").status_code == 400
