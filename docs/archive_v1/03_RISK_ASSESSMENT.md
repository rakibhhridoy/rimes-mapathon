# 03 — Risk Assessment & Community Ranking

## Risk Framework

```
Risk = Hazard × Exposure × Vulnerability
```

Each component is normalized to [0, 1] before multiplication.

## Component Definitions

### Hazard (H) — Flood intensity

Source: Hybrid GNN + Kriging output (`flood_risk_kriged.tif`)

Represents the spatial probability / intensity of flood occurrence at each grid cell. Derived from:
- Ensemble proxy labels (DEM + JRC + GloFAS + SAR)
- GNN-learned spatial patterns
- Kriging-interpolated continuous surface

### Exposure (E) — What's in harm's way

Quantifies the density and value of infrastructure within each grid cell or admin unit.

| Layer | Weight | Source |
|-------|--------|--------|
| Building count per cell | 0.25 | OSM |
| Road length (km) per cell | 0.15 | OSM |
| Bridge presence (binary) | 0.15 | OSM |
| Cropland area (km²) per cell | 0.15 | OSM |
| Hospital/school presence | 0.20 | OSM |
| Embankment/shelter presence | 0.10 | OSM |

Composite Exposure = weighted sum, min-max normalized to [0, 1].

### Vulnerability (V) — Who's most at risk

Composite of population exposure, critical infrastructure access, and livelihood factors.

| Layer | Weight | Source |
|-------|--------|--------|
| Population density | 0.30 | WorldPop |
| Distance to nearest hospital | 0.20 | OSM-computed |
| Distance to nearest flood shelter | 0.15 | OSM-computed |
| Distance to nearest primary road | 0.10 | OSM-computed |
| Poverty proxy (night light intensity) | 0.15 | VIIRS |
| Elderly/child population ratio | 0.10 | Census (if available) |

Composite Vulnerability = weighted sum, min-max normalized to [0, 1].

## Aggregation Levels

### Grid-cell level (500m × 500m)
- Raw Risk = H × E × V at each cell
- Output: `composite_risk.tif`

### Union level (GADM Level 3)
- Mean and max risk per union
- Count of high-risk infrastructure per union
- Union risk rank (1 = highest risk)
- Output: `union_risk_summary.geojson`

### Upazila level (GADM Level 2)
- Aggregated from union-level scores
- Used for district-level dashboards
- Output: `upazila_risk_summary.geojson`

## Hotspot Detection

**Getis-Ord Gi* statistic** identifies statistically significant clusters of high-risk cells:

```python
from esda.getisord import G_Local
from libpysal.weights import Queen

w = Queen.from_dataframe(grid_gdf)
g_local = G_Local(grid_gdf["composite_risk"], w)
grid_gdf["hotspot_z"] = g_local.Zs
grid_gdf["hotspot_p"] = g_local.p_sim
grid_gdf["is_hotspot"] = (g_local.Zs > 1.96) & (g_local.p_sim < 0.05)
```

## Ranked Output for Responders

### Per-asset ranking
```
risk_ranked_assets.csv
├── asset_id
├── asset_type (hospital | school | bridge | road | cropland | ...)
├── name
├── union_name
├── upazila_name
├── flood_risk_score     (0-1)
├── exposure_score       (0-1)
├── vulnerability_score  (0-1)
├── composite_risk       (0-1)
├── risk_rank            (1 = highest)
├── latitude
└── longitude
```

### Per-union ranking
```
union_risk_summary.csv
├── union_name
├── upazila_name
├── division_name
├── mean_risk
├── max_risk
├── n_high_risk_assets    (composite_risk > 0.7)
├── n_hospitals_exposed
├── n_schools_exposed
├── n_bridges_exposed
├── cropland_exposed_km2
├── population_exposed
├── risk_rank
└── geometry (polygon)
```

## CLI Usage

```bash
python -m pipeline.cli risk --config config.yaml
python -m pipeline.cli rank --config config.yaml --level union
python -m pipeline.cli hotspot --config config.yaml --confidence 0.95
```
