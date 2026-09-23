# SGMDI — Smart Geospatial Mapping & Disaster Impact Intelligence

Flood risk pipeline and public dashboard for **Rangpur & Rajshahi Divisions, Bangladesh**. It combines OpenStreetMap infrastructure, a gradient-boosted tree trained on flood extents observed by Sentinel-1, a terrain hazard surface, and a composite hazard–exposure–vulnerability score on a ~500 m grid.

**Live dashboard:** <https://fermium.systems/hazmapper> — open to everyone, no sign-in.

> **This is a research prototype, not a warning service.** Scores describe
> relative susceptibility under typical monsoon conditions; they are not
> forecasts of any specific flood. For official warnings use
> [FFWC](https://www.ffwc.gov.bd) and [BMD](https://www.bmd.gov.bd); in an
> emergency dial 999.

Everything the dashboard displays comes from pipeline outputs in `data/output`.
Where an output is missing, the dashboard says so rather than showing a
placeholder — see `tests/test_pipeline.py::TestNoFabricatedData`.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/rakibhhridoy/rimes-mapathon.git
cd rimes-mapathon
```

### 2. Download the data

The full data folder is archived on Zenodo:

> **DOI**: [10.5281/zenodo.19233968](https://doi.org/10.5281/zenodo.19233968)

Download and extract the `data/` folder into the project root so the structure looks like:

```
rimes-mapathon/
├── config.yaml
├── pipeline/
├── dashboard/
├── data/
│   ├── raw/                  # Source datasets (~75 MB)
│   │   ├── dem_srtm_30m.tif
│   │   ├── jrc_water_occurrence.tif
│   │   ├── worldpop_popdens.tif
│   │   ├── infrastructure_raw.gpkg
│   │   ├── gadm_union.*
│   │   └── gadm_upazila.*
│   ├── processed/            # DEM derivatives, flood labels (~1.6 GB)
│   └── output/               # Pipeline results (~101 MB)
│       ├── risk_ranked_assets.geojson
│       ├── union_risk_summary.geojson
│       ├── risk_grid.geojson
│       ├── flood_risk_kriged.tif
│       ├── situation_report.pdf
│       └── ...
└── docs/
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`numpy` is pinned below 2.0 because the torch 2.2 wheels are built against
numpy 1.x. On platforms with newer torch wheels the pin can be relaxed.

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py -- --config config.yaml
```

No pipeline execution needed — the Zenodo archive includes all pre-computed
outputs. Server options (loopback bind, CSRF, no uploads) live in
`.streamlit/config.toml`; `apps.conf` and `hazmapper.service` are the nginx and
systemd units used for the public deployment, including per-IP rate limits and
memory caps.

### Deploying an update to the public server

```bash
rsync -a --exclude data --exclude .git ./ webuser@server:/var/www/hazmapper/
rsync -a data/output data/cache webuser@server:/var/www/hazmapper/data/
rsync -a data/sylhet/output data/sylhet/cache webuser@server:/var/www/hazmapper/data/sylhet/
rsync -a data/cht/output data/cht/cache webuser@server:/var/www/hazmapper/data/cht/
ssh webuser@server 'cd /var/www/hazmapper && pip install -r requirements.txt && sudo systemctl restart hazmapper'
```

The dashboard needs only each region's `output/` and `cache/` directories, not
the raw or processed data, and it discovers regions from `configs/` and the
files present. `hazmapper.service` and `apps.conf` carry the systemd and nginx
settings.

### 5. Run the tests

```bash
pytest -q
```

## Re-running the Pipeline (optional)

If you want to regenerate results from scratch instead of using pre-computed outputs:

After any re-run, rebuild the display caches the dashboard reads:

```bash
python preprocess_cache.py
```

```bash
# Full pipeline (all steps)
python -m pipeline.cli run

# Or individual steps
python -m pipeline.cli download      # Step 0: Download GADM, SRTM, JRC, WorldPop
python -m pipeline.cli ingest        # Step 1: OSM infrastructure
python -m pipeline.cli preprocess    # Step 2: Reproject, clip, DEM derivatives
python -m pipeline.cli features      # Step 3: Feature extraction
python -m pipeline.cli graph         # Step 4: Spatial k-NN graph
python -m pipeline.cli train         # Step 5-6: Train GNN + embeddings
python -m pipeline.cli krige         # Step 7: Kriging interpolation
python -m pipeline.cli risk          # Step 8-9: Composite risk, aggregation, ranking
python -m pipeline.cli metadata      # Pipeline confidence metrics
python -m pipeline.cli landslide     # CHT landslide susceptibility (optional)
```

## Study Area

- **Divisions**: Rangpur and Rajshahi, Bangladesh
- **Bounding Box**: 88.0°E – 89.9°E, 24.0°N – 26.7°N
- **CRS**: EPSG:32646 (UTM Zone 46N)
- **Grid Resolution**: ~500m

## Pipeline Architecture

| Step | Module | Description |
|------|--------|-------------|
| 0 | `data_download` | Download GADM, SRTM DEM, JRC water, WorldPop |
| 1 | `data_ingest` | Fetch OSM infrastructure via Overpass API |
| 2 | `data_ingest` | Reproject, clip, compute DEM derivatives, flood labels |
| 3 | `feature_extract` | Extract raster + spatial features at infrastructure points |
| 4 | `graph_build` | Build spatial k-NN graph (k=6) |
| 5-6 | `asset_model` | Fit the asset model (`asset_model.type`, default gradient boosting; GraphSAGE optional), calibrate, score assets |
| 7 | `kriging` | Ordinary Kriging interpolation of flood risk surface |
| 8-9 | `risk_score` | Composite risk (geometric mean of hazard, exposure, vulnerability), hotspots, ranking |

### Administrative levels

Boundaries come from **geoBoundaries gbOpen** (CC BY 4.0, redistributable):
ADM2 districts, ADM3 upazilas and ADM4 unions. Within the study area that is
21 districts, 148 upazilas and 1,339 unions, and the pipeline writes a summary
for each level.

This corrects a mislabelling: the earlier GADM files named `gadm_union.shp`
and `gadm_upazila.shp` actually held **upazilas** and **districts**
(GADM's own `ENGTYPE_3` reads "Upazilla"), so every figure previously labelled
"union" was an upazila. GADM has no union level for Bangladesh, and its licence
forbids redistribution, so it cannot ship in the open data archive either.
Unit names repeat across the country, so each unit also carries an
`admin_label` naming its parent — "Abdullahpur (Gangachara)".

### Two readings of exposure

Every cell carries two composite scores, because "where is infrastructure at
risk" and "where would people be affected" are different questions:

| Column | Exposure measured by | Tends to rank highest |
|---|---|---|
| `composite_risk` | type-weighted count of mapped assets | urban municipalities |
| `composite_risk_people` | WorldPop population density | crowded floodplain settlements |

Both appear on the union and upazila summaries (`mean_risk`,
`mean_risk_people`) and in the dashboard cards. Neither is a substitute for
the other, and the population figures come from WorldPop's *constrained*
product, which holds values only where buildings were detected — scattered
rural settlement can be missed.

### Running another region

Region configs inherit the base config with `extends:` and override only what
differs, so the shared model settings stay in one place:

```bash
python -m pipeline.cli -c configs/sylhet.yaml run
python preprocess_cache.py -c configs/sylhet.yaml
```

Each region writes to its own `data/<region>/` tree (`paths:` in the config),
and the dashboard shows a region selector once more than one has results.
DEM and surface-water tiles are derived from the bounding box, and national
datasets (WorldPop, geoBoundaries, JRC tiles) are cached once in
`data/shared/` and clipped per region.

Available regions: `config.yaml` (Rangpur & Rajshahi), `configs/sylhet.yaml`,
`configs/sw_coastal.yaml`, `configs/cht.yaml`.

### OpenStreetMap ingestion

Assets are fetched **by bounding box, in tiles of 0.75°**, not by place name.
Querying `"<Division> Division, Bangladesh"` pulled in assets far outside the
study area — the CHT config covers the hill tracts, while Chittagong Division
reaches the coast and Cox's Bazar — and timed out on the larger divisions.
Results are clipped to the bounding box and each asset is labelled with the
district it falls in.

The main Overpass endpoint allows two connections per address and refuses the
rest outright, so failed queries are retried with a growing pause. Public
mirrors hung or refused when tested, so none is listed by default;
`data.osm.overpass_endpoints` accepts extra endpoints for deployments that
have a reliable one.

### Landslide susceptibility (Chittagong Hill Tracts)

`configs/cht.yaml` enables a landslide model that is **fitted to observed
landslides** rather than assumed:

```bash
python -m pipeline.cli -c configs/cht.yaml download
python -m pipeline.cli -c configs/cht.yaml ingest
python -m pipeline.cli -c configs/cht.yaml preprocess
python -m pipeline.cli -c configs/cht.yaml landslide
```

Mapped landslide locations come from NASA's Cooperative Open Online Landslide
Repository (COOLR); a class-weighted logistic regression is fitted to terrain
at those locations against randomly sampled background points at least 500 m
away, and validated on held-out 10 km blocks. The fitted coefficients,
inventory size and validation AUC are written to
`data/cht/output/landslide_model.json`.

This replaces a hand-picked logistic curve on slope that had never been fitted
to anything, and an "upazila" fallback that sliced the raster into horizontal
bands, labelled them with real upazila names and invented their populations
with a random number generator.

> Juang, C.S., Stanley, T.A., & Kirschbaum, D.B. (2019). Using citizen science
> to expand the global map of landslides: Introducing the Cooperative Open
> Online Landslide Repository (COOLR). *PLOS ONE* 14(7): e0218657.

### Validating against observed floods

The terrain-threshold labels are circular — TWI and HAND are model inputs as
well as the basis of the labels. Sentinel-1 sees standing water directly:

```bash
export EARTHENGINE_PROJECT=your-ee-project   # or set earthengine.project
python -m pipeline.cli sentinel1             # map flood extents for each event
# set data.labels.source: observed in config.yaml, then rerun the pipeline
python -m pipeline.cli validate              # AUC, POD/FAR/CSI vs observed
```

`validate` reports three things separately: how well model scores rank
locations that actually flooded, how well the kriged surface agrees with
observed flooding across the grid, and how closely the old proxy labels match
observed flooding. Results land in `data/output/validation_metrics.json`.

### Modelling notes

- **Sampling** — rasters are sampled in their own CRS and distances computed in
  metres. Sampling a UTM raster with lon/lat silently returns nodata for every
  point, which previously left all terrain features constant and all labels zero.
  The pipeline now aborts if a feature comes out constant or a label column has
  a single class.
- **Labels are observed floods.** Training labels come from Sentinel-1 radar
  extents of past floods (five events for Rangpur & Rajshahi, four for Sylhet,
  four cyclones on the coast), with a pixel labelled flood-prone where it
  flooded in at least two events (one on the coast, where surge floods drain
  between satellite passes). The earlier terrain-threshold proxy is a poor map
  of flooding (probability of detection 0.14 against observed floods in
  Rangpur & Rajshahi), but a model trained on it still ranks well: on the same
  test blocks it reaches AUC 0.81–0.85 against observed floods, against
  0.84–0.87 for the observed-label models on the same single split. Both
  figures come from one block assignment; across five assignments the graph
  model averages 0.795–0.849 by region with a spread of 0.02–0.04, which is
  as large as the label gain itself. The larger gain is in being able to
  validate against real floods at all.
- **The graph earns nothing.** `python -m pipeline.cli benchmark` refits the
  graph network and ordinary tabular models on the same features, assets and
  blocks, over five block assignments. Gradient boosting beats the graph
  network by 0.030 AUC in Rangpur & Rajshahi (p = 0.004, ahead on 5 of 5
  splits) and a random forest by 0.027 in Sylhet (p = 0.04); on the coast the
  models are indistinguishable. Gradient boosting is therefore the default
  (`asset_model.type`), with GraphSAGE kept as an option. The same command
  also compares training labels: observed Sentinel-1 extents beat terrain
  thresholds by 0.137 AUC in Rangpur & Rajshahi, 0.071 in Sylhet and 0.031 on
  the coast. Results land in `<output>/benchmark.json` and
  `<output>/label_comparison.json`.
- **Weights are stress-tested.** `python -m pipeline.cli sensitivity`
  recomputes the composite under other weightings. The cell ranking survives
  (Spearman 0.91–0.99 for exposure and vulnerability weights, 0.96–0.98 under
  random ±50 % perturbation), but about a fifth of the top decile changes
  places, and dropping the access distances from vulnerability moves it a lot
  (0.42–0.59). Results land in `<output>/sensitivity.json`.
- **Validation** — whole 10 km blocks are held out (`graph.block_size_m`), split
  between calibration and test. Neighbouring assets share terrain, so a random
  node split measures memorisation rather than skill. Metrics land in
  `data/output/gnn_metrics.json` and `validation_metrics.json`.
- **Two numbers per asset.** `flood_risk` is the raw class-weighted score and
  ranks assets; `flood_probability` is that score passed through an isotonic
  calibration fitted on the calibration blocks. It is an approximate
  probability that the ground floods in at least `min_events` mapped events.
  Flood prevalence differs severalfold between blocks (4 % in the Rangpur
  calibration blocks, 14 % in its test blocks), so the probability is only a
  region-wide approximation and underestimates the most flood-prone areas;
  the calibrated Brier score beats the base rate only narrowly. Never read
  the raw score as a likelihood.
- **Composite risk** is the geometric mean of the three factors, so it keeps the
  0–1 scale of its inputs. Because it is bounded by the weakest factor it rarely
  approaches 1, so cells are also assigned relative classes 1–5 by quantile
  (`risk_class`; class 0 means no mapped exposure).
- **Vulnerability** uses only components with data behind them — population
  density and distances to hospitals, shelters and roads. Components listed in
  `config.yaml` without a raster are excluded and the remaining weights
  renormalised; the applied weights are written to
  `data/output/vulnerability_weights.json`.

### Known limitations

- Radar cannot tell inundated paddy from a flood, and Sentinel-1 passes only
  every 6–12 days, so short-lived surge flooding on the coast is
  under-recorded. The two-event label threshold limits the first problem where
  the data allow it.
- Composite risk is defined only where OpenStreetMap has mapped assets (7% of
  grid cells). Sparse mapping reads as low exposure, not as low risk.
- Composite risk rises with the density of mapped assets, so urban
  municipalities rank highest. That is the intended reading for *infrastructure*
  risk, but it is not a measure of risk to people.
- Areas with no assets mapped in OpenStreetMap (250 of 1,339 unions) get no
  score at all rather than a zero, and are drawn grey on the map.

## Data Sources

| Dataset | Source | Resolution |
|---------|--------|------------|
| Infrastructure | OpenStreetMap (Overpass API) | Vector |
| DEM | SRTM 30m | 30m |
| Water occurrence | JRC Global Surface Water | 30m |
| Population | WorldPop | 100m |
| Admin boundaries | geoBoundaries gbOpen (ADM2 district, ADM3 upazila, ADM4 union) | Vector |

## Outputs

- **GeoJSON**: Ranked assets, union/upazila risk summaries, risk grid, hotspot clusters
- **CSV**: Asset rankings, admin-level summaries
- **GeoTIFF**: Kriged flood risk surface, kriging variance
- **PDF**: Situation report with top at-risk unions and assets
- **Dashboard**: Interactive Streamlit app with map, analytics, and export

## Documentation

See the `docs/` folder for detailed documentation:

- [Project Overview](docs/00_PROJECT_OVERVIEW.md)
- [Data Pipeline](docs/01_DATA_PIPELINE.md)
- [GNN & Kriging Model](docs/02_GNN_KRIGING_MODEL.md)
- [Risk Assessment](docs/03_RISK_ASSESSMENT.md)
- [Dashboard](docs/04_DASHBOARD.md)
- [Ten-Step Procedure](docs/05_TEN_STEP_PROCEDURE.md)

## Citing this work

If you use this code or its outputs, please cite the software (MIT licence, see
`LICENSE`) and the input data archive:

> Hasan, M. R., Zubyer, S., & Arabi, F. Z. (2026). *Fermium Hazard Mapper /
> SGMDI: flood risk to infrastructure in Rangpur and Rajshahi, Bangladesh.*
> Zenodo. https://doi.org/10.5281/zenodo.19233968

## Licence

Source code: MIT (see `LICENSE`). Input datasets keep their own licences —
OpenStreetMap (ODbL), WorldPop (CC BY 4.0), JRC Global Surface Water, SRTM
(public domain) and GADM (non-commercial, no redistribution). Attribution for
map tiles and data is shown in the dashboard.

Originally developed for the RIMES Mapathon (ResilienceAI track) by Team
Fermium — Shoumik Zubyer, Md Rakib Hasan and Fazla Zawadul Arabi.
