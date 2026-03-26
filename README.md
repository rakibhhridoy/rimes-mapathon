# SGMDI — Smart Geospatial Mapping & Disaster Impact Intelligence

Flood risk intelligence pipeline and interactive dashboard for **Rangpur & Rajshahi Divisions, Bangladesh**. Combines OSM infrastructure mapping, GNN-based risk scoring, Kriging interpolation, and composite hazard-exposure-vulnerability assessment.

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

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py -- --config config.yaml
```

No pipeline execution needed — the Zenodo archive includes all pre-computed outputs.

## Re-running the Pipeline (optional)

If you want to regenerate results from scratch instead of using pre-computed outputs:

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
| 5-6 | `gnn_model` | Train 2-layer GraphSAGE, extract embeddings + risk scores |
| 7 | `kriging` | Ordinary Kriging interpolation of flood risk surface |
| 8-9 | `risk_score` | Composite risk (Hazard x Exposure x Vulnerability), hotspots, ranking |

## Data Sources

| Dataset | Source | Resolution |
|---------|--------|------------|
| Infrastructure | OpenStreetMap (Overpass API) | Vector |
| DEM | SRTM 30m | 30m |
| Water occurrence | JRC Global Surface Water | 30m |
| Population | WorldPop | 100m |
| Admin boundaries | GADM v4.1 (L2 upazila, L3 union) | Vector |

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

## License

This project was developed for the RIMES Mapathon competition.
