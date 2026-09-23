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
        assert response.text.splitlines()[0].startswith("risk_rank,name")

    def test_tiles_are_served_and_traversal_is_refused(self, client, connection):
        region = connection.execute(
            "SELECT id FROM regions WHERE has_assets = 1 LIMIT 1").fetchone()["id"]
        assert client.get(f"/tiles/{region}/assets.pmtiles").status_code == 200
        assert client.get("/tiles/..%2F..%2Fetc/assets.pmtiles").status_code == 404
        assert client.get(f"/tiles/{region}/nonsense.pmtiles").status_code == 404

    def test_responses_are_cacheable(self, client):
        """Caching is the point: a visitor's second view should not recompute."""
        assert "max-age" in client.get("/api/regions").headers.get("cache-control", "")
