# 05 — 10-Step Pipeline Procedure

## Complete Pipeline Flow

```
Step 1          Step 2           Step 3            Step 4
┌──────────┐   ┌──────────┐    ┌──────────┐     ┌──────────┐
│ Ingest   │──▶│Preprocess│───▶│ Extract  │────▶│ Build    │
│ OSM+DEM+ │   │ CRS/clip │    │ Features │     │ Spatial  │
│ Proxy    │   │ resample │    │ at nodes │     │ Graph    │
└──────────┘   └──────────┘    └──────────┘     └────┬─────┘
                                                      │
Step 7          Step 6           Step 5               │
┌──────────┐   ┌──────────┐    ┌──────────┐     ┌────▼─────┐
│ Ordinary │◀──│ Extract  │◀───│ Train    │◀────│ Ensemble │
│ Kriging  │   │ Embed +  │    │ GraphSAGE│     │ Label    │
│ Surface  │   │ Risk Prob│    │ Model    │     │ Fusion   │
└────┬─────┘   └──────────┘    └──────────┘     └──────────┘
     │
     │          Step 8           Step 9            Step 10
     │         ┌──────────┐    ┌──────────┐     ┌──────────┐
     └────────▶│Composite │───▶│ Rank +   │────▶│Dashboard │
               │ Risk =   │    │ Aggregate│     │ Cards +  │
               │ H × E × V│    │ Hotspots │     │ Map +    │
               └──────────┘    └──────────┘     │ Sidebar  │
                                                └──────────┘
```

---

## Step 1 — Data Ingestion

**Script**: `pipeline/data_ingest.py`

**Inputs**: config.yaml (AOI bbox, OSM tags)

**Actions**:
- Query OSM Overpass API for Rangpur + Rajshahi divisions
  - Priority 1: hospitals, schools, bridges, flood shelters, embankments, roads, railways, ferry ghats
  - Priority 2: cropland, fishponds, irrigation canals, market roads
- Download SRTM 30m DEM tiles covering AOI
- Download JRC Global Surface Water occurrence layer
- Download GloFAS flood return-period extents (if available)
- Acquire Sentinel-1 SAR scenes for known flood events (optional)
- Download WorldPop population density raster
- Download GADM admin boundaries (Level 2 upazila + Level 3 union)

**Outputs**: `data/raw/` directory with all source files

**CLI**: `python -m pipeline.cli ingest --config config.yaml`

---

## Step 2 — Preprocessing & CRS Alignment

**Script**: `pipeline/data_ingest.py` (preprocessing functions)

**Actions**:
- Reproject all layers to EPSG:32646 (UTM 46N)
- Clip all layers to Rangpur + Rajshahi division boundary
- Resample rasters to 500m common grid
- Compute DEM derivatives:
  - Slope (degrees)
  - TWI (Topographic Wetness Index)
  - HAND (Height Above Nearest Drainage)
  - Flow accumulation
- Rasterize vector hazard zones to match grid
- Clean and validate geometries

**Outputs**: `data/processed/` with aligned rasters and clean vectors

**CLI**: `python -m pipeline.cli preprocess --config config.yaml`

---

## Step 3 — Feature Extraction at Infrastructure Nodes

**Script**: `pipeline/feature_extract.py`

**Actions**:
- Compute centroids for all infrastructure features
- Sample raster values at each centroid:
  - Elevation, slope, TWI, HAND, flow accumulation
  - NDVI (from Sentinel-2 if available)
  - Population density
- Compute derived features:
  - Distance to nearest river/waterway
  - Distance to nearest hospital
  - Distance to nearest school/flood shelter
  - Distance to nearest primary road
- Encode asset type (one-hot or ordinal)
- Assemble node feature matrix `X` of shape `(N_nodes, D_features)`
- Standardize features (StandardScaler)

**Outputs**: `data/processed/node_features.parquet`

**CLI**: `python -m pipeline.cli features --config config.yaml`

---

## Step 4 — Ensemble Flood Label Fusion

**Script**: `pipeline/feature_extract.py` (label section)

**Actions**:
- Generate binary flood mask from each source:
  1. DEM-derived: TWI > threshold AND HAND < threshold
  2. JRC: water occurrence > 25% (historically flooded)
  3. GloFAS: inside 20-year return-period extent
  4. SAR: flood detected in backscatter change (if available)
- Majority voting: label = 1 if ≥ 2/4 sources agree (≥ 2/3 if SAR unavailable)
- Sample ensemble label at each infrastructure centroid
- Store as target variable `y`

**Outputs**: `data/processed/flood_proxy_labels.tif`, labels in `node_features.parquet`

---

## Step 5 — Spatial Graph Construction

**Script**: `pipeline/graph_build.py`

**Actions**:
- Build k-NN graph (k=6) from infrastructure centroid coordinates
- Use `scipy.spatial.cKDTree` for efficient neighbor search
- Cap edges at `max_edge_distance_m=5000`
- Compute edge weights: `w_ij = 1 / distance(i, j)`
- Construct PyTorch Geometric `Data` object:
  - `x`: node features (N, D)
  - `edge_index`: (2, E) edge list
  - `edge_attr`: (E, 1) edge weights
  - `y`: (N,) flood labels
- Create train/val masks (80/20 stratified split)

**Outputs**: `data/processed/spatial_graph.pt`

**CLI**: `python -m pipeline.cli graph --config config.yaml`

---

## Step 6 — Train GraphSAGE Model

**Script**: `pipeline/gnn_model.py`

**Actions**:
- Load spatial graph from Step 5
- Initialize FloodGNN (2-layer GraphSAGE, 64 hidden, dropout 0.3)
- Train with:
  - Adam optimizer, lr=1e-3
  - BCEWithLogitsLoss with pos_weight for class imbalance
  - Early stopping (patience=20) on validation loss
  - Up to 200 epochs
- Log training metrics (loss, AUC-ROC per epoch)
- Save best model checkpoint

**Outputs**: `data/output/gnn_model.pt`, training log

**CLI**: `python -m pipeline.cli train --config config.yaml`

---

## Step 7 — Extract Embeddings & Risk Scores

**Script**: `pipeline/gnn_model.py` (inference section)

**Actions**:
- Load trained model
- Forward pass in eval mode:
  - Extract 64-dim node embeddings from penultimate layer
  - Compute flood risk probability via sigmoid on logits
- Compute residuals: `residual = observed_label - gnn_probability`
- Attach `gnn_risk_score` and `embedding` to infrastructure GeoDataFrame

**Outputs**: `data/output/node_embeddings.npy`, `gnn_risk_scores.csv`

---

## Step 8 — Ordinary Kriging Interpolation

**Script**: `pipeline/kriging.py`

**Actions**:
- Fit empirical semivariogram on GNN risk scores at node locations
- Fit spherical theoretical variogram (pykrige)
- Define prediction grid covering study area (500m resolution)
- Execute Ordinary Kriging → continuous flood risk surface
- Also krige kriging variance → uncertainty surface
- Hybrid fusion: `final_risk = kriging_surface + gnn_residual_correction` (at node locations, GNN dominates; between nodes, kriging interpolates)
- Export as GeoTIFF

**Outputs**:
- `data/output/flood_risk_kriged.tif`
- `data/output/kriging_variance.tif`
- `data/output/variogram_params.json`

**CLI**: `python -m pipeline.cli krige --config config.yaml`

---

## Step 9 — Composite Risk Scoring & Ranking

**Script**: `pipeline/risk_score.py`

**Actions**:
- Compute Exposure layer:
  - Building density, road length, bridge/hospital/school presence per cell
  - Weighted sum → normalize [0,1]
- Compute Vulnerability layer:
  - Population density, dist_hospital, dist_shelter, dist_road
  - Weighted sum → normalize [0,1]
- Composite: `Risk = Hazard × Exposure × Vulnerability`
- Aggregate to union level (mean, max, count of high-risk assets)
- Aggregate to upazila level
- Run Getis-Ord Gi* hotspot detection
- Rank unions by composite risk

**Outputs**:
- `data/output/composite_risk.tif`
- `data/output/union_risk_summary.geojson`
- `data/output/upazila_risk_summary.geojson`
- `data/output/risk_ranked_assets.csv`
- `data/output/risk_ranked_assets.geojson`
- `data/output/hotspot_clusters.geojson`

**CLI**: `python -m pipeline.cli risk --config config.yaml`

---

## Step 10 — Dashboard & Export

**Script**: `dashboard/app.py`

**Actions**:
- Load all output files from Step 9
- Render three-panel Streamlit dashboard:
  1. **Union risk cards** — scrollable carousel sorted by risk
  2. **Interactive Folium map** — toggleable layers (risk, exposure, vulnerability, hotspots, admin boundaries)
  3. **Sidebar analytics** — ranked table, histograms, bar charts, scatter plots
- Enable filtering by division, upazila, union, risk threshold, asset type
- Enable export (CSV, GeoJSON, PDF report)

**CLI**: `streamlit run dashboard/app.py -- --config config.yaml`

---

## Full Pipeline — Single Command

```bash
# Run all steps sequentially
python -m pipeline.cli run --config config.yaml

# Or run individual steps
python -m pipeline.cli ingest --config config.yaml
python -m pipeline.cli preprocess --config config.yaml
python -m pipeline.cli features --config config.yaml
python -m pipeline.cli graph --config config.yaml
python -m pipeline.cli train --config config.yaml
python -m pipeline.cli krige --config config.yaml
python -m pipeline.cli risk --config config.yaml

# Launch dashboard
streamlit run dashboard/app.py -- --config config.yaml
```
