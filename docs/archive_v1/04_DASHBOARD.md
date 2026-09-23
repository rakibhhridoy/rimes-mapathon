# 04 — Streamlit Dashboard Specification

## Layout

Three integrated views in a single Streamlit app:

```
┌──────────────────────────────────────────────────────────────────┐
│  SGMDI — Smart Geospatial Mapping & Disaster Impact Intelligence │
│  [Rangpur ▼] [Rajshahi ▼]  [Union filter ▼]  [Risk ≥ 0.5 ━━●━] │
├──────────────────────────────────────┬───────────────────────────┤
│                                      │                           │
│         INTERACTIVE MAP              │    SIDEBAR ANALYTICS      │
│         (Folium / Deck.gl)           │                           │
│                                      │  ┌─────────────────────┐ │
│    ┌─────────────────────────────┐   │  │ Top 10 At-Risk      │ │
│    │  ● Hospital (risk: 0.87)    │   │  │ Unions              │ │
│    │  ▲ Bridge (risk: 0.92)      │   │  │ 1. Kurigram Sadar   │ │
│    │  ■ School (risk: 0.74)      │   │  │ 2. Chilmari         │ │
│    │                              │   │  │ 3. Rowmari          │ │
│    │  [Flood Risk] [Exposure]     │   │  │ ...                 │ │
│    │  [Vulnerability] [Hotspots]  │   │  └─────────────────────┘ │
│    └─────────────────────────────┘   │                           │
│                                      │  ┌─────────────────────┐ │
│                                      │  │ Risk Distribution   │ │
│                                      │  │ (histogram/boxplot) │ │
│                                      │  └─────────────────────┘ │
│                                      │                           │
│                                      │  ┌─────────────────────┐ │
│                                      │  │ Exposed Assets      │ │
│                                      │  │ by Category (bar)   │ │
│                                      │  └─────────────────────┘ │
├──────────────────────────────────────┴───────────────────────────┤
│                     UNION-LEVEL RISK CARDS                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Kurigram     │ │ Chilmari     │ │ Rowmari      │  ◀ ● ● ▶   │
│  │ Sadar Union  │ │ Union        │ │ Union        │             │
│  │              │ │              │ │              │             │
│  │ Risk: 0.92   │ │ Risk: 0.88   │ │ Risk: 0.85   │             │
│  │ ████████░░   │ │ ████████░░   │ │ ████████░░   │             │
│  │              │ │              │ │              │             │
│  │ Top exposed: │ │ Top exposed: │ │ Top exposed: │             │
│  │ • Hospital×2 │ │ • Bridge×3   │ │ • School×4   │             │
│  │ • Bridge×1   │ │ • Road 12km  │ │ • Cropland   │             │
│  │ • Cropland   │ │ • School×2   │ │   85 ha      │             │
│  │   120 ha     │ │              │ │ • Hospital×1 │             │
│  │              │ │              │ │              │             │
│  │ [View Detail]│ │ [View Detail]│ │ [View Detail]│             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

## View 1 — Union-Level Risk Cards

Each card displays:
- **Union name** and parent upazila/division
- **Composite risk score** (0–1) with color-coded progress bar
  - 🔴 ≥ 0.7 Critical | 🟠 0.5–0.7 High | 🟡 0.3–0.5 Moderate | 🟢 < 0.3 Low
- **Top 5 exposed assets** by category with counts
- **Population exposed** estimate
- **Recommended actions** (auto-generated from risk profile):
  - "Evacuate low-lying areas near Teesta river"
  - "Pre-position boats at Chilmari ghat"
  - "Reinforce embankment at km 12.4"
- **[View Detail]** button → zooms map to union, shows all assets

Scrollable horizontal carousel, sorted by risk (highest first).

## View 2 — Interactive Map

Built with `streamlit-folium`:

**Base layers** (toggle):
- OpenStreetMap
- Satellite (Esri World Imagery)
- Terrain (Stamen)

**Overlay layers** (toggle):
- Flood risk heatmap (kriged surface as raster overlay)
- Exposure density (infrastructure count per cell)
- Vulnerability index (color-coded cells)
- Hotspot clusters (Gi* significant zones, red polygons)
- Admin boundaries (union / upazila outlines)

**Markers**:
- Color-coded by risk: red (critical) → green (low)
- Icon by type: 🏥 hospital, 🏫 school, 🌉 bridge, 🛣️ road, 🌾 cropland
- Click → popup with asset details, risk breakdown, and nearest shelter

**Map interactions**:
- Click union polygon → filters sidebar + shows risk card
- Draw tool → select custom AOI for on-the-fly statistics

## View 3 — Sidebar Analytics

**Filters** (top):
- Division dropdown (Rangpur / Rajshahi)
- District/Upazila dropdown (cascading)
- Union multi-select
- Risk threshold slider (0.0 – 1.0)
- Asset type checkboxes

**Charts**:
1. **Ranked union table** — sortable by risk, population, exposed assets
2. **Risk distribution** — histogram of composite risk across selected area
3. **Exposed assets by category** — stacked bar chart (hospital, school, bridge, road, cropland)
4. **Risk vs. vulnerability scatter** — each point is a union; size = population

**Export buttons**:
- Download filtered data as CSV
- Download current map view as GeoJSON
- Generate PDF situation report

## Data Flow

```
data/output/
├── flood_risk_kriged.tif      → Map heatmap layer
├── composite_risk.tif         → Map + cards
├── union_risk_summary.geojson → Cards + sidebar table
├── risk_ranked_assets.geojson → Map markers
├── top50_risk_assets.geojson  → Quick-view layer
└── risk_ranked_assets.csv     → Sidebar table + export
```

## CLI Usage

```bash
# Run dashboard locally
streamlit run dashboard/app.py -- --config config.yaml

# Generate static PDF report without dashboard
python -m pipeline.cli report --config config.yaml --format pdf --output reports/
```

## Deployment (Local / GitHub)

1. Clone the repository
2. `pip install -r requirements.txt`
3. Run pipeline: `python -m pipeline.cli run --config config.yaml`
4. Launch dashboard: `streamlit run dashboard/app.py`
5. Open browser at `http://localhost:8501`

No VPS required. All processing and visualization runs locally.
