# 01 — Data Acquisition & Preprocessing

## Data Sources

### 1. OpenStreetMap (via Overpass API / osmnx)

**Priority 1 — Lifeline Infrastructure**
- Hospitals, clinics (`amenity=hospital`, `amenity=clinic`)
- Schools (`amenity=school`, `amenity=college`)
- Bridges (`man_made=bridge`, `bridge=yes`)
- Flood shelters (`amenity=shelter`, `shelter_type=flood`)
- Embankments (`man_made=embankment`, `waterway=dam`)

**Priority 1 — Transport Network**
- Roads (`highway=*` — primary, secondary, tertiary, trunk)
- Railways (`railway=rail`)
- Ferry ghats (`amenity=ferry_terminal`)
- Key intersections and connectivity nodes

**Priority 2 — Agriculture**
- Cropland (`landuse=farmland`)
- Fishponds (`landuse=aquaculture`)
- Irrigation canals (`waterway=canal`, `waterway=ditch`)
- Market access roads (roads connecting to `amenity=marketplace`)

### 2. DEM / Terrain (SRTM 30m)

- Source: USGS EarthExplorer or OpenTopography
- Products derived:
  - **Elevation** (raw DEM)
  - **Slope** (degrees) via `richdem` or `whitebox`
  - **Aspect** (flow direction)
  - **TWI** — Topographic Wetness Index = ln(upstream_area / tan(slope))
  - **Flow Accumulation** — proxy for drainage channels
  - **Hand** — Height Above Nearest Drainage

### 3. Proxy Flood Labels (Ensemble — No Ground Truth)

Since no field-collected flood observations are available, we build **ensemble pseudo-labels** from three independent sources:

| Source | What it provides | Resolution |
|--------|-----------------|------------|
| **DEM-derived flood-fill** | Binary inundation mask from TWI + HAND thresholds | 30m |
| **JRC Global Surface Water** | Historical water occurrence frequency (1984–2021) | 30m |
| **GloFAS / Fathom** | Return-period flood extent (5yr, 20yr, 100yr) | ~90m–1km |
| **Sentinel-1 SAR** | Event-specific flood extent from backscatter change detection | 10m |

**Fusion strategy**: Majority voting across sources. A grid cell is labeled "flood-exposed" if ≥ 2 out of 4 sources agree.

### 4. Socioeconomic / Vulnerability Layers

- Population density: WorldPop (100m)
- Admin boundaries: GADM Level 3 (union) + Level 2 (upazila)
- Distance to nearest hospital/school: computed from OSM

## Preprocessing Steps

```
1. Reproject all layers → EPSG:32646 (UTM 46N)
2. Clip to Rangpur + Rajshahi division boundary
3. Resample rasters to common 500m grid (or 100m for fine analysis)
4. Rasterize vector hazard zones to match grid
5. Compute DEM derivatives (slope, TWI, HAND, flow accumulation)
6. Extract centroids for all vector infrastructure features
7. Sample all raster values at infrastructure centroids
8. Build ensemble flood label via majority voting
9. Store processed data as GeoParquet + GeoTIFF
```

## File Outputs

| File | Format | Description |
|------|--------|-------------|
| `infrastructure_rangpur_rajshahi.gpkg` | GeoPackage | All OSM infrastructure with type labels |
| `dem_derivatives.tif` | Multi-band GeoTIFF | Bands: elevation, slope, TWI, HAND, flow_acc |
| `flood_proxy_labels.tif` | GeoTIFF | Ensemble binary flood mask |
| `vulnerability_layers.tif` | Multi-band GeoTIFF | Bands: pop_density, dist_hospital, dist_school |
| `node_features.parquet` | GeoParquet | Per-infrastructure feature table ready for graph |

## CLI Usage

```bash
python -m pipeline.cli ingest --config config.yaml --aoi rangpur_rajshahi
python -m pipeline.cli preprocess --config config.yaml
```
