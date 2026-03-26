# ── Fermium Hazard Mapper · Data Constants ──────────────────────────────────
# Ported from Fermium_Hazard_Mapper_Stack.jsx

PIPELINE_STAGES = [
    {
        "id": 1, "title": "Ingestion & ETL", "desc": "Automated data acquisition",
        "items": [
            "GEE API → Sentinel-1/2, Landsat-9 STAC pull",
            "SRTM/Copernicus 30m DEM via OpenTopography",
            "OSM Overpass API: roads, bridges, health, schools",
            "BBS CSV shapefile ingest (upazila-level census)",
            "AlphaEarth embedding API (256×256m tiles)",
            "BWDB/BMD rainfall gauge telemetry (REST)",
            "Output: COG GeoTIFF + GeoJSON tiles @ 10–30m",
        ],
    },
    {
        "id": 2, "title": "Preprocessing", "desc": "Harmonisation & normalisation",
        "items": [
            "Reproject → EPSG:32646 (UTM Zone 46N)",
            "Co-register all rasters to 10m Sentinel-2 grid",
            "Speckle filter SAR (Lee 5×5 kernel)",
            "Cloud masking (Sentinel-2 QA60 + s2cloudless)",
            "Terrain analysis: GRASS r.slope.aspect.curvature",
            "Tidal correction DEM for coastal zones (Satkhira)",
            "Min-max normalise each feature band [0,1]",
        ],
    },
    {
        "id": 3, "title": "Feature Engineering", "desc": "Index computation & embeddings",
        "items": [
            "NDWI, NDVI, MNDWI, EVI per scene",
            "HAND index (Height Above Nearest Drainage)",
            "SAR backscatter: pre-event composite",
            "Slope / aspect / curvature / TWI rasters",
            "AlphaEarth: tile embed → UMAP → k-means (k=8)",
            "SoVI: PCA on 20 BBS census variables",
            "Proximity buffers: rivers 1km, coast 5km, roads 200m",
        ],
    },
    {
        "id": 4, "title": "Risk Modelling", "desc": "GNN + Kriging + AHP fusion",
        "items": [
            "Ordinary/Universal Kriging → continuous hazard surface",
            "Infrastructure graph: PyG GraphSAGE (3-layer GAT)",
            "AHP weight matrix → flood/landslide composite",
            "GBM + RF ensemble (12 conditioning factors, CHT)",
            "Uncertainty: Kriging variance + MC dropout",
            "CMIP6 SSP245/585 scenario projections",
            "Output: per-asset risk score + confidence interval",
        ],
    },
    {
        "id": 5, "title": "Vulnerability Integration", "desc": "5-class composite CVI",
        "items": [
            "Domain weights: Demo 30%, Econ 25%, Infra 20%",
            "Housing 15%, Agri/Livelihood 10%",
            "CVI = Exposure x Sensitivity / Adaptive Capacity",
            "K-means clustering → 5 vulnerability classes",
            "Housing: kacha % + proximity to hazard zone",
            "Agri: saline crop exposure + irrigation gap index",
            "Upazila-level choropleth + sub-unit drill-down",
        ],
    },
    {
        "id": 6, "title": "Multi-Hazard Fusion", "desc": "Weighted overlay + uncertainty",
        "items": [
            "Fuse flood / landslide / coastal into unified score",
            "Infrastructure cascading risk (GNN propagation)",
            "Mapping confidence: 3-tier (High/Medium/Low)",
            "Alert threshold matrix (susceptibility x rainfall)",
            "Rolling 5-day rainfall trigger (71-282mm, CHT)",
            "Forecast horizon: 3-day (BGD Met Office API)",
            "Output: Vector tiles + uncertainty rasters",
        ],
    },
    {
        "id": 7, "title": "Delivery & Dashboard", "desc": "NOAA-style responder interface",
        "items": [
            "Streamlit dashboard + Folium map engine",
            "COG tiles served via TiTiler (FastAPI)",
            "Real-time pipeline status indicator",
            "SMS/email dispatch: Twilio + SendGrid",
            "Snakemake DAG + DVC versioning (reproducible)",
            "Docker Compose deploy (GEE Colab + Cloud Run)",
            "WCAG 2.1 AA, Bengali/English i18n",
        ],
    },
]

DATA_SOURCES = {
    "NW Rangpur (Fluvial)": {
        "center": [25.74, 89.25],
        "zoom": 9,
        "hazard": "Fluvial Flood — Teesta/Brahmaputra basin",
        "color": "#1a6b8a",
        "accent": "#3cb8de",
        "alert_level": "ORANGE",
        "risk_score": 0.72,
        "sources": [
            ("DEM/Elevation", "SRTM 30m + Copernicus GLO-30", "Flood inundation depth, HAND index"),
            ("Slope / Drainage", "SRTM-derived GRASS GIS", "Drainage density, flow accumulation"),
            ("Distance to Teesta", "OSM waterways + BWDB gauges", "Proximity risk buffer (0-5km)"),
            ("LULC / NDVI", "Sentinel-2 (10m, 5-day revisit)", "Crop type, vegetation, bare soil"),
            ("Rainfall", "CHIRPS 5km + BMD telemetry", "24h/72h antecedent rainfall trigger"),
            ("Soil Permeability", "HWSD v2.0 / FAO-UNESCO", "Infiltration capacity, runoff coeff"),
            ("SAR Inundation", "Sentinel-1 GRD VV/VH", "Flood extent mapping (-15dB thresh)"),
            ("Population", "BBS 2022 + WorldPop 100m", "Exposed population per flood zone"),
        ],
        "mock_assets": [
            (25.74, 89.25, "Rangpur Hospital", "hospital", 0.71),
            (25.78, 89.30, "Teesta Bridge Km42", "bridge", 0.89),
            (25.70, 89.20, "Gangachara School", "school", 0.54),
            (25.76, 89.27, "Taraganj Road", "road", 0.62),
            (25.72, 89.22, "Char Settlement A", "shelter", 0.91),
        ],
    },
    "NE Sylhet (Flash Flood)": {
        "center": [24.89, 91.87],
        "zoom": 9,
        "hazard": "Flash Flood — Surma-Meghna basin haor system",
        "color": "#1a6b52",
        "accent": "#3cdea0",
        "alert_level": "RED",
        "risk_score": 0.85,
        "sources": [
            ("DEM / Haor Bathymetry", "SRTM + IWM Bangladesh DEM", "Haor inundation extent, water depth"),
            ("Distance to Surma", "OSM + BWDB gauge network", "Flash flood proximity (0-3km)"),
            ("SAR Change Detection", "Sentinel-1 GRD bi-temporal", "Pre/post-event flood delineation"),
            ("Boro Rice Calendar", "BBS agricultural census 2023", "Early flash flood crop loss risk"),
            ("NDWI / MNDWI", "Sentinel-2 B3/B8/B11", "Open water fraction per tile"),
            ("AlphaEarth Embeddings", "Google AlphaEarth API", "Haor land cover similarity search"),
            ("Road / Bridge Density", "OSM + LGED road database", "Isolation risk (cut-off communities)"),
            ("Household Income", "BBS HIES 2022 (<BDT10k)", "Economic vulnerability weight"),
        ],
        "mock_assets": [
            (24.89, 91.87, "Sylhet MAG Hospital", "hospital", 0.66),
            (24.94, 91.92, "Surma Bridge", "bridge", 0.83),
            (24.85, 91.82, "Haor Fringe School", "school", 0.77),
            (24.91, 91.90, "Hakaluki Haor Settlement", "shelter", 0.95),
            (24.87, 91.85, "Sunamganj Road A", "road", 0.58),
        ],
    },
    "SW Coastal (Satkhira/Patuakhali)": {
        "center": [22.10, 89.40],
        "zoom": 9,
        "hazard": "Coastal Inundation — cyclone / tidal surge / SLR",
        "color": "#6b4a1a",
        "accent": "#dea03c",
        "alert_level": "YELLOW",
        "risk_score": 0.64,
        "sources": [
            ("Coastal DEM (tidal)", "CoastalDEM v2 + SRTM-corrected", "Storm surge inundation, SLR +0.5m"),
            ("Tidal Flats", "Sentinel-2 MNDWI time series", "Intertidal zone delineation"),
            ("Distance to Coast", "OSM coastline + BIWTA data", "Surge exposure buffer (0-10km)"),
            ("Mangrove Cover", "Global Mangrove Watch (JAXA)", "Natural buffer attenuation factor"),
            ("Saline Crop Types", "BBS 2023 + Sentinel-2 NDVI", "Shrimp/saline rice inundation loss"),
            ("SAR Cyclone Inundation", "Sentinel-1 + Sentinel-3 OLCI", "Post-cyclone flood mapping"),
            ("Sea-level Gauge", "BIWTA tide gauge (Mongla/Khepupara)", "Real-time surge threshold trigger"),
            ("Embankment (polders)", "BWDB polder GIS layer", "Breach risk x exposure population"),
        ],
        "mock_assets": [
            (22.10, 89.40, "Satkhira Upazila Hospital", "hospital", 0.60),
            (22.05, 89.35, "Polder-22 Embankment", "bridge", 0.78),
            (22.15, 89.45, "Shyamnagar School", "school", 0.55),
            (22.08, 89.38, "Sundarbans Buffer Settlement", "shelter", 0.82),
            (22.12, 89.42, "Kaliganj Road", "road", 0.47),
        ],
    },
}

LANDSLIDE_DATA = {
    "center": [22.35, 91.80],
    "zoom": 10,
    "hazard": "Rainfall-Induced Landslide — Chittagong Hill Tracts",
    "alert_level": "RED",
    "risk_score": 0.88,
    "rainfall_5day_mm": 187,
    "sources": [
        ("Slope / Aspect / Curvature", "SRTM 30m GRASS r.slope.aspect", "Primary susceptibility factor (AUC 0.93)"),
        ("Geology / Lithology", "GSB 1:250k + USGS GMNA", "Rock type, fault proximity"),
        ("Distance to Streams", "OSM + Sentinel-2 derived", "Undercutting / fluvial erosion risk"),
        ("Distance to Roads", "OSM + RHD BD shapefile", "Hill cut / road-induced instability"),
        ("Landslide Inventory", "Roy et al. 2025 (170 pts) + Rahman et al. 2025 (730 pts)", "Training / validation dataset"),
        ("Rainfall Threshold", "BMD 1960-2025 + CHIRPS", "5-day antecedent: 71-282mm trigger"),
        ("LULC / NDVI", "Sentinel-2 10m time series", "Vegetation root reinforcement proxy"),
        ("Soil Texture / Permeability", "HWSD v2 + SRDI Bangladesh", "Pore-water pressure accumulation"),
    ],
    "mock_assets": [
        (22.35, 91.80, "Rangamati Sadar Hospital", "hospital", 0.82),
        (22.40, 91.85, "Kaptai Road Km18 Cut Slope", "road", 0.95),
        (22.30, 91.75, "Khagrachari School", "school", 0.71),
        (22.38, 91.82, "Hillside Settlement B", "shelter", 0.88),
        (22.33, 91.78, "Chandraghona Bridge", "bridge", 0.76),
    ],
    "conditioning_factors": [
        ("Slope", 0.22, "Primary driver — steep terrain (>30 deg) in CHT"),
        ("Rainfall (5-day)", 0.18, "Trigger threshold: 71-282mm antecedent"),
        ("Aspect", 0.10, "South-facing slopes have lower shear resistance"),
        ("Curvature", 0.09, "Concave profile concentrates pore-water pressure"),
        ("Elevation", 0.08, "Mid-slope positions most susceptible (400-800m)"),
        ("LULC", 0.07, "Deforested areas have 3x higher susceptibility"),
        ("Soil Texture", 0.07, "Sandy clay loam — rapid saturation"),
        ("Geology", 0.06, "Tertiary shale / mudstone — low shear strength"),
        ("Distance to Stream", 0.05, "Lateral undercutting within 50m"),
        ("Distance to Road", 0.04, "Hill-cut instability within 200m"),
        ("NDVI", 0.04, "Low NDVI (<0.3) amplifies susceptibility 1.8x"),
    ],
}

VULNERABILITY_CLASSES = [
    {
        "cls": "Class 1", "label": "Very Low", "color": "#1a6b52", "text_color": "#3cdea0",
        "housing": "Pucca RCC, >5m elevation, >1km from flood body",
        "agri": "Irrigated HYV Boro, no saline exposure, paved access road",
        "infra": "Pucca school/health facility, on-grid power, tarmac road access",
        "pop": "Upazila HQ clusters, income >BDT 25k, low flood frequency",
        "cvi_range": "0.00-0.20",
    },
    {
        "cls": "Class 2", "label": "Low", "color": "#3a7a3a", "text_color": "#3cdea0",
        "housing": "Semi-pucca/brick, 2-5m elevation, 500m-1km from river",
        "agri": "Mixed Aman/Boro, marginal irrigation, kutcha road",
        "infra": "Brick school, occasional power outage, seasonal road flood",
        "pop": "Union HQ, income BDT 15-25k, 1-in-10yr flood exposure",
        "cvi_range": "0.21-0.40",
    },
    {
        "cls": "Class 3", "label": "Moderate", "color": "#8a6a1a", "text_color": "#dea03c",
        "housing": "Kutcha/tin-roof, 1-2m elevation, 200-500m from river",
        "agri": "Rain-fed Aman, partial salinity, bridge-dependent access",
        "infra": "Bamboo/tin school, unreliable power, road floods 1-3mo/yr",
        "pop": "Para/village level, income BDT 10-15k, 1-in-5yr flood",
        "cvi_range": "0.41-0.60",
    },
    {
        "cls": "Class 4", "label": "High", "color": "#8a3a1a", "text_color": "#de7a3c",
        "housing": "Mud/jute-stick wall, <1m elevation, 50-200m from river",
        "agri": "Saline intrusion on Boro, crop loss >30%, isolated by flood",
        "infra": "No permanent school/clinic, off-grid, road impassable >3mo",
        "pop": "Char/floodplain belt, income <BDT 10k, 1-in-2yr flood",
        "cvi_range": "0.61-0.80",
    },
    {
        "cls": "Class 5", "label": "Very High", "color": "#8a1a1a", "text_color": "#de3c3c",
        "housing": "Polythene/thatch, below river bankfull, <50m from water",
        "agri": "100% saline/tidal inundation, no viable crop, fishing-only",
        "infra": "Zero permanent infra, no power, completely isolated by flood",
        "pop": "Tidal char / haor fringe / CHT foothill, income <BDT 5k",
        "cvi_range": "0.81-1.00",
    },
]

LITERATURE = [
    {
        "cite": "Rahman et al. (2025)",
        "title": "Remote sensing and GIS-driven landslide susceptibility mapping using RF and MaxEnt — CHT Bangladesh",
        "journal": "Discover Sustainability (Springer)",
        "doi": "10.1007/s43621-025-02084-x",
        "relevance": "RF (AUC=0.93) on 15 factors, 730 inventory pts. 79% of CHT classified high-very high susceptibility. PRIMARY model for Fermium landslide layer.",
        "factors": ["elevation", "slope", "rainfall", "soil texture", "LULC", "geology", "distance to stream"],
    },
    {
        "cite": "Roy et al. (2025)",
        "title": "Landslide susceptibility assessment in Chittagong division using RS, GIS, and machine learning",
        "journal": "Discover Geoscience 3:221",
        "doi": "10.1007/s44288-025-00337-w",
        "relevance": "GBM + RF on 12 conditioning factors (170 inventory pts), AUC 0.83. Benchmark for Chattogram Division.",
        "factors": ["elevation", "slope", "rainfall", "LULC", "aspect", "curvature", "drainage proximity"],
    },
    {
        "cite": "Hasan et al. (2025)",
        "title": "Predictive landslide susceptibility modeling: RF, BRT, KNN in Khagrachari district",
        "journal": "Env. Science & Pollution Research 32, 31204-31221",
        "doi": "10.1007/s11356-024-34949-5",
        "relevance": "15 conditioning factors, 127 inventory pts. RF outperforms KNN/BRT. Khagrachari district benchmark.",
        "factors": ["slope", "aspect", "curvature", "NDVI", "geology", "soil texture", "road proximity", "stream proximity"],
    },
    {
        "cite": "Islam et al. (2025)",
        "title": "Predicting urban landslides leveraging hybrid ML model and CMIP6 projections — Chattogram CDA",
        "journal": "Urban Climate (ScienceDirect)",
        "doi": "10.1016/j.uclim.2025.02XXX",
        "relevance": "LR-bNB hybrid >90% accuracy; CMIP6 SSP126/585 to 2100. 12% area very high risk under current climate.",
        "factors": ["topographic", "hydrological", "soil", "geological", "CMIP6 rainfall projections"],
    },
    {
        "cite": "Ullah & Tuhin (2026)",
        "title": "Geostatistical & Geospatial Modelling of Landslide Susceptibility in Rangamati, CHT",
        "journal": "Dhaka Univ. J. Earth & Env. Sci. 14(2), 193-209",
        "doi": "10.3329/dujees.v14i2.87606",
        "relevance": "Kriging-based susceptibility for Rangamati — directly validates Fermium's variogram module.",
        "factors": ["geostatistical interpolation", "variogram", "Rangamati CHT", "terrain factors"],
    },
    {
        "cite": "Ahmed & Rahman (2018/updated 2025)",
        "title": "Dynamic Web-GIS Based Landslide EWS for Chittagong Metropolitan Area",
        "journal": "ISPRS Int. J. Geo-Inf. 7(12):485",
        "doi": "10.3390/ijgi7120485",
        "relevance": "ANN susceptibility + rainfall threshold matrix (5-day 71-282mm). Blueprint for Fermium EWS alert logic.",
        "factors": ["soil permeability", "geology", "slope", "aspect", "distance to stream", "fault line", "hill cut", "road cut"],
    },
]

AHP_FLOOD_WEIGHTS = [
    ("DEM/Elevation", 0.25, "Highest flood correlation"),
    ("Distance to river", 0.20, "Teesta / Surma proximity"),
    ("HAND index", 0.18, "Drainage proximity rank"),
    ("SAR inundation", 0.15, "Observed flood extent"),
    ("Rainfall intensity", 0.12, "CHIRPS 5-day accumulation"),
    ("LULC / soil permeability", 0.10, "Runoff amplifier factor"),
]

MOCK_UPAZILA_RISK = {
    "NW Rangpur (Fluvial)": [
        {"upazila": "Gangachara", "cvi": 0.82, "flood_risk": 0.87, "pop_exposed": 145000, "class": 5},
        {"upazila": "Taraganj", "cvi": 0.71, "flood_risk": 0.74, "pop_exposed": 98000, "class": 4},
        {"upazila": "Kaunia", "cvi": 0.65, "flood_risk": 0.69, "pop_exposed": 112000, "class": 4},
        {"upazila": "Rangpur Sadar", "cvi": 0.38, "flood_risk": 0.42, "pop_exposed": 280000, "class": 2},
        {"upazila": "Pirganj", "cvi": 0.55, "flood_risk": 0.58, "pop_exposed": 175000, "class": 3},
    ],
    "NE Sylhet (Flash Flood)": [
        {"upazila": "Sunamganj Sadar", "cvi": 0.88, "flood_risk": 0.92, "pop_exposed": 67000, "class": 5},
        {"upazila": "Derai (Haor Belt)", "cvi": 0.91, "flood_risk": 0.94, "pop_exposed": 54000, "class": 5},
        {"upazila": "Sylhet Sadar", "cvi": 0.35, "flood_risk": 0.48, "pop_exposed": 430000, "class": 2},
        {"upazila": "Beanibazar", "cvi": 0.59, "flood_risk": 0.62, "pop_exposed": 88000, "class": 3},
        {"upazila": "Jaintiapur", "cvi": 0.72, "flood_risk": 0.76, "pop_exposed": 72000, "class": 4},
    ],
    "SW Coastal (Satkhira/Patuakhali)": [
        {"upazila": "Shyamnagar", "cvi": 0.85, "flood_risk": 0.88, "pop_exposed": 92000, "class": 5},
        {"upazila": "Kaliganj", "cvi": 0.76, "flood_risk": 0.79, "pop_exposed": 78000, "class": 4},
        {"upazila": "Satkhira Sadar", "cvi": 0.48, "flood_risk": 0.52, "pop_exposed": 185000, "class": 3},
        {"upazila": "Galachipa (Patuakhali)", "cvi": 0.81, "flood_risk": 0.84, "pop_exposed": 61000, "class": 5},
        {"upazila": "Dashmina", "cvi": 0.69, "flood_risk": 0.72, "pop_exposed": 55000, "class": 4},
    ],
}

MOCK_LANDSLIDE_UPAZILA = [
    {"upazila": "Rangamati Sadar", "susceptibility": 0.88, "pop_exposed": 45000, "class": 5},
    {"upazila": "Khagrachari Sadar", "susceptibility": 0.82, "pop_exposed": 38000, "class": 5},
    {"upazila": "Kaptai", "susceptibility": 0.76, "pop_exposed": 29000, "class": 4},
    {"upazila": "Rangunia (Chattogram)", "susceptibility": 0.71, "pop_exposed": 95000, "class": 4},
    {"upazila": "Bandarban Sadar", "susceptibility": 0.79, "pop_exposed": 32000, "class": 4},
]

EMERGENCY_SHELTERS = [
    {"name": "Rangpur Central Shelter", "region": "NW Rangpur", "lat": 25.74, "lon": 89.25, "capacity": 2500, "status": "OPEN", "cvi_class": 3},
    {"name": "Sylhet Cyclone Shelter A", "region": "NE Sylhet", "lat": 24.89, "lon": 91.87, "capacity": 1800, "status": "OPEN", "cvi_class": 5},
    {"name": "Shyamnagar Polder Shelter", "region": "SW Coastal", "lat": 22.10, "lon": 89.40, "capacity": 1200, "status": "AT CAPACITY", "cvi_class": 5},
    {"name": "Satkhira Upazila Complex", "region": "SW Coastal", "lat": 22.72, "lon": 89.07, "capacity": 3000, "status": "OPEN", "cvi_class": 4},
    {"name": "Rangamati Relief Camp", "region": "CHT", "lat": 22.35, "lon": 91.80, "capacity": 900, "status": "OPEN", "cvi_class": 5},
]

MOBILE_FINANCING = [
    {"program": "bKash Emergency Transfer", "amount_bdt": 5000, "target": "Class 4-5, active flood zone", "status": "ACTIVE", "disbursed": 12400},
    {"program": "Nagad Livelihood Grant", "amount_bdt": 8000, "target": "Farmers with >50% crop loss", "status": "ACTIVE", "disbursed": 6800},
    {"program": "PKSF Micro-Credit Emergency", "amount_bdt": 15000, "target": "Char & haor fringe households", "status": "ACTIVE", "disbursed": 3200},
    {"program": "World Food Programme e-Voucher", "amount_bdt": 3000, "target": "Class 5 coastal inundation zone", "status": "ACTIVE", "disbursed": 9100},
    {"program": "GoB Kash Sahayata (Landslide)", "amount_bdt": 10000, "target": "CHT affected families", "status": "PENDING", "disbursed": 0},
]
