"""
Tests for scripts/build_database.py: every kind of pipeline file lands in
the GeoPackage in a form a GIS or a query can read back.
"""

import io
import json
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_database import build, table_name  # noqa: E402


@pytest.fixture
def sources(tmp_path):
    region = tmp_path / "region" / "output"
    region.mkdir(parents=True)
    gpd.GeoDataFrame({"name": ["a", "b"], "flood_risk": [0.1, 0.9]},
                     geometry=[Point(90, 24), Point(90.1, 24.1)],
                     crs="EPSG:4326").to_file(region / "assets.geojson")
    pd.DataFrame({"unit": ["x"], "mean_risk": [0.4]}).to_csv(region / "summary.csv", index=False)
    (region / "benchmark.json").write_text(json.dumps({"auc": 0.8}))
    np.save(region / "scores.npy", np.array([0.1, 0.9], dtype=np.float32))
    (region / "model.joblib").write_bytes(b"not really a model")
    with rasterio.open(region / "hazard.tif", "w", driver="GTiff", width=4, height=3,
                       count=1, dtype="float32", crs="EPSG:32646",
                       transform=from_bounds(0, 0, 400, 300, 4, 3)) as dst:
        dst.write(np.arange(12, dtype=np.float32).reshape(1, 3, 4))
    return [("sylhet", "output", p) for p in sorted(region.iterdir())]


@pytest.fixture
def db(tmp_path, sources):
    path = tmp_path / "all.gpkg"
    build(path, sources, log=lambda *_: None)
    return path


def test_vector_layers_keep_their_geometry_and_attributes(db):
    assets = gpd.read_file(db, layer="sylhet_assets")
    assert list(assets["name"]) == ["a", "b"]
    assert assets.crs.to_epsg() == 4326


def test_float_rasters_keep_their_values(db):
    with rasterio.open(f"GPKG:{db}:sylhet_hazard") as src:
        assert src.read(1)[2, 3] == pytest.approx(11.0)
        assert src.crs.to_epsg() == 32646


def test_tables_metrics_and_arrays_are_queryable(db):
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT mean_risk FROM sylhet_summary").fetchone()[0] == pytest.approx(0.4)
    assert conn.execute("SELECT json_extract(json, '$.auc') FROM metrics"
                        " WHERE name = 'benchmark'").fetchone()[0] == pytest.approx(0.8)
    blob, shape = conn.execute("SELECT npy, shape FROM arrays WHERE name = 'scores'").fetchone()
    assert np.load(io.BytesIO(blob)).tolist() == pytest.approx([0.1, 0.9])
    assert json.loads(shape) == [2]


def test_every_item_records_its_source_checksum_and_licence(db, sources):
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT source_path, sha256, licence FROM sources").fetchall()
    assert len(rows) == len(sources)
    assert all(len(sha) == 64 and licence for _, sha, licence in rows)


def test_attribute_tables_are_listed_for_gis_software(db):
    conn = sqlite3.connect(db)
    listed = {row[0] for row in conn.execute("SELECT table_name FROM gpkg_contents")}
    assert {"metrics", "arrays", "files", "sources", "sylhet_assets", "sylhet_hazard"} <= listed


def test_names_are_sql_safe_and_unique():
    taken = set()
    first = table_name("sw_coastal", Path("s1-flood 2024.tif"), taken=taken, kind="raw")
    second = table_name("sw_coastal", Path("s1-flood 2024.tif"), taken=taken, kind="processed")
    assert first == "sw_coastal_s1_flood_2024"
    assert second != first
