# Changelog

## v3.0.0 (2026-10-02)

National coverage and a new public front end.

**Pipeline**
- National partition of Bangladesh into eight flood regions by flood regime
  (`configs/national/partition.yaml`), covering all 61 lowland districts, plus
  the Chittagong Hill Tracts landslide region; 114,569 mapped assets.
- Regions can be defined by a list of districts (`aoi.districts`); assets,
  grid cells and validation outside them are dropped.
- Every region is kept to Bangladesh: assets and grid cells across a land
  border are dropped (`pipeline/country.py`), with a `clip` stage for asset
  files fetched earlier.
- OpenStreetMap assets can be read from a national Geofabrik extract,
  converted once into an indexed file (`data.osm.source: pbf`); the Overpass
  fetch is cached, resumable, and splits a query the server cannot answer.
- Sentinel-1 downloads that Earth Engine refuses are split into quarters.
- Flood threshold fixed to match the whole percentages the frequency layer
  stores (with three events the one-event rule had acted as a two-event rule).
- Feature importance is computed by the `benchmark` stage; the temporal test
  is skipped, and says so, when a region's events do not span the cut-off.
- Event coverage check (`scripts/check_s1_events.py`) and region
  configuration generator (`scripts/make_region_config.py`).

**Web application** (`web/`, replaces Streamlit as the public site)
- FastAPI + SQLite + PMTiles + MapLibre GL; built by `scripts/build_web.py`.
- Separate flood and landslide views; an "All" view of every flood region.
- Division, district, upazila and union drill-down chosen by zoom, with
  cascading pickers and area cards (`scripts/admin_areas.py`).
- Official DDM MRVA return-period scenario maps drawn live from UNOSAT.
- Opacity sliders, shareable links that reopen the same view, and PNG export.
- White relief basemap; filters on the left, results on the right.

**Data**
- One SpatiaLite database of all outputs (`scripts/build_database.py`).

**Tests**: 159 (from 132).

## v2.0.0 (2026-09)

Audited and corrected release: seventeen defects of the first release fixed
and guarded by tests, gradient boosting as the default model, radar labels,
spatial-block validation over repeated assignments, and the three validation
regions plus the landslide region. Data archived at
https://doi.org/10.5281/zenodo.22978729.
