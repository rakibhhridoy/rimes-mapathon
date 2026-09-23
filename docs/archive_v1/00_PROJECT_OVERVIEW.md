# Smart Geospatial Mapping & Disaster Impact Intelligence (SGMDI)

## Problem Statement

Rangpur and Rajshahi divisions in Bangladesh lack accurate, up-to-date geospatial data of critical community resources. During floods, responders struggle to assess which infrastructure (cropland, buildings, bridges, roads) is at risk and where to deploy resources first.

## Objectives

1. Map critical infrastructure exposure to flood hazard
2. Identify the most vulnerable communities using spatial data
3. Propose automated, reproducible risk assessment pipelines

## Study Area

- **Divisions**: Rangpur and Rajshahi, Bangladesh
- **CRS**: EPSG:32646 (UTM Zone 46N)
- **Grid Resolution**: 500m cells

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SGMDI System                         │
├──────────────┬──────────────┬───────────────────────────┤
│  Data Layer  │  Model Layer │  Output Layer             │
├──────────────┼──────────────┼───────────────────────────┤
│ OSM Overpass │ Spatial Graph│ Union-level Risk Cards    │
│ SRTM DEM    │ Construction │ Interactive Folium Map    │
│ JRC GSW     │              │ Sidebar Analytics Panel   │
│ Sentinel-1  │ Hybrid GNN + │                           │
│ GloFAS      │ Kriging      │ GeoTIFF / GeoJSON Export  │
│ Admin Bounds │              │ Ranked CSV Reports        │
└──────────────┴──────────────┴───────────────────────────┘
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hazard focus | Flood only | Primary disaster risk in Rangpur/Rajshahi |
| GNN architecture | Hybrid GNN (GraphSAGE) + Kriging | Kriging provides spatial prior; GNN corrects residuals |
| Ground truth | Ensemble proxy labels | DEM-derived + JRC/GloFAS + Sentinel-1 SAR fused via majority voting |
| Infra priority | Lifeline + Transport (primary), Agriculture (secondary) | Emergency response first, food security second |
| Vulnerability | Composite weighting | Population + critical infra + livelihood layers combined |
| Dashboard | Multi-view Streamlit | Risk cards + interactive map + sidebar analytics |
| Deployment | Local + GitHub | No VPS; clone and run locally |

## Repository Structure

```
mapathon/
├── config.yaml                  # Pipeline configuration
├── requirements.txt             # Python dependencies
├── docs/
│   ├── 00_PROJECT_OVERVIEW.md   # This file
│   ├── 01_DATA_PIPELINE.md      # Data acquisition & preprocessing
│   ├── 02_GNN_KRIGING_MODEL.md  # Model architecture & training
│   ├── 03_RISK_ASSESSMENT.md    # Risk scoring & aggregation
│   └── 04_DASHBOARD.md          # Streamlit dashboard spec
├── pipeline/
│   ├── __init__.py
│   ├── data_ingest.py           # Step 1-2: OSM + raster loading
│   ├── feature_extract.py       # Step 3: Sample hazard values at centroids
│   ├── graph_build.py           # Step 4: Spatial graph construction
│   ├── gnn_model.py             # Step 5-6: GraphSAGE training + embeddings
│   ├── kriging.py               # Step 7: Variogram fitting + Ordinary Kriging
│   ├── risk_score.py            # Step 8: Composite risk computation
│   ├── export.py                # Step 9: GeoTIFF + GeoJSON + CSV export
│   └── cli.py                   # CLI entry point (click)
├── dashboard/
│   ├── app.py                   # Streamlit main app
│   ├── components/
│   │   ├── risk_cards.py        # Union-level risk card widgets
│   │   ├── map_view.py          # Folium interactive map
│   │   └── sidebar.py           # Analytics sidebar with filters
│   └── assets/
├── data/
│   ├── raw/                     # Downloaded OSM, DEM, satellite data
│   ├── processed/               # Feature matrices, graph objects
│   └── output/                  # Risk rasters, GeoJSON, CSV reports
└── tests/
```
