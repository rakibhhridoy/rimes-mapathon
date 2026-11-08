# Changelog

## v3.1.0 (2026-11-08)

More held-out validation and a reproduction demo; the default model is
unchanged.

**New held-out tests** (each on the benchmark's block assignments)
- `threshold-metrics`: precision, recall and F1 beside the AUC, with the
  threshold chosen where F1 peaks on calibration blocks, for every model and
  for the wetness index alone.
- `covariates` and `covariate-ablation`: optional dry-season MODIS NDVI,
  SoilGrids topsoil clay and CHIRPS mean annual rainfall, tested against the
  default features and against asset coordinates alone. Optional features
  enter a model through `data.extra_features` (`pipeline/covariates.py`).
- `prediction-spread`: per-asset spread of held-out probability over 100
  block assignments, and how often each asset's flag holds under the
  calibration threshold and under a fixed flagged share.
- `rolling-origin` and `record-checks`: each flood year predicted from the
  years before it and scored against the record of earlier flooding, with
  the JRC hazard map and Molchan scores; record length against skill.
- Every comparison over block assignments uses the corrected resampled
  t-test (Nadeau and Bengio 2003); `refresh-pvalues` updates stored results.
- Block-size check (20 and 40 km) and held-out scores for the hazard surface.

**Other**
- Validation-region configurations (`configs/validation/`) and
  `scripts/make_validation_region.py`; `scripts/remap_s1a.py` remaps events
  with Sentinel-1A only.
- Sentinel-1 docstring corrected: composites span relative orbits.
- `scripts/demo.py` reproduces a validation region from the data archive
  without Earth Engine and checks it against the archived outputs; downloads
  resume after a stall.
- The download stage fetches the country-border files the clip needs.
- Continuous integration on GitHub Actions; `requirements-lock.txt` records
  the tested versions; the NumPy < 2 pin is lifted.
- Tests that need pipeline outputs skip on a fresh clone instead of failing.
- Raster sampling marks points outside the raster as missing by its bounds.

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
