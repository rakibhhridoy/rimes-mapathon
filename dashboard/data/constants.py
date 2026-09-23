# ── Fermium Hazard Mapper · Static reference content ─────────────────────────
# Only descriptive text lives here. Every number shown on a map, chart or
# table must come from pipeline outputs in data/output — never from this file.

# Datasets the pipeline actually ingests (pipeline/data_download.py,
# pipeline/data_ingest.py). Keep in sync when sources change.
DATA_SOURCES_USED = [
    {
        "name": "OpenStreetMap",
        "use": "Infrastructure assets (roads, bridges, schools, hospitals, "
               "shelters, embankments, cropland)",
        "licence": "ODbL 1.0",
        "attribution": "© OpenStreetMap contributors",
        "url": "https://www.openstreetmap.org/copyright",
    },
    {
        "name": "SRTM 1 arc-second DEM (NASA / USGS)",
        "use": "Elevation, slope, TWI, HAND, flow accumulation",
        "licence": "Public domain",
        "attribution": "NASA / USGS",
        "url": "https://www.earthdata.nasa.gov/data/instruments/srtm",
    },
    {
        "name": "Sentinel-1 GRD (Copernicus, via Google Earth Engine)",
        "use": "Observed flood extents for training labels and validation",
        "licence": "Copernicus open data",
        "attribution": "Contains modified Copernicus Sentinel data",
        "url": "https://sentinel.esa.int/web/sentinel/missions/sentinel-1",
    },
    {
        "name": "JRC Global Surface Water (Pekel et al. 2016)",
        "use": "Permanent water mask for the flood mapping",
        "licence": "Free with attribution (Copernicus)",
        "attribution": "EC JRC / Google",
        "url": "https://global-surface-water.appspot.com/",
    },
    {
        "name": "WorldPop population density",
        "use": "Population component of vulnerability",
        "licence": "CC BY 4.0",
        "attribution": "WorldPop, University of Southampton",
        "url": "https://www.worldpop.org/",
    },
    {
        "name": "geoBoundaries (gbOpen)",
        "use": "Union, upazila and district aggregation",
        "licence": "CC BY 4.0",
        "attribution": "geoBoundaries, William & Mary geoLab",
        "url": "https://www.geoboundaries.org/",
    },
]

# Authoritative sources the public should use for warnings and help.
# This dashboard is not a warning service.
OFFICIAL_RESOURCES = [
    {
        "name": "Flood Forecasting and Warning Centre (FFWC), BWDB",
        "detail": "Official river-level forecasts and flood warnings",
        "url": "https://www.ffwc.gov.bd",
    },
    {
        "name": "Bangladesh Meteorological Department (BMD)",
        "detail": "Weather, rainfall and cyclone warnings",
        "url": "https://www.bmd.gov.bd",
    },
    {
        # ddm.gov.bd serves a certificate issued for another domain
        # (checked 2026-09-22), so browsers warn on it. Linking the national
        # portal instead; restore the direct link once the certificate is fixed.
        "name": "Department of Disaster Management (DDM)",
        "detail": "Shelters, relief and disaster response — via the national portal",
        "url": "https://bangladesh.gov.bd",
    },
    {
        "name": "Disaster early-warning voice service — dial 1090",
        "detail": "Toll-free recorded weather and flood warnings",
        "url": None,
    },
    {
        "name": "National Emergency Service — dial 999",
        "detail": "Police, fire service and ambulance",
        "url": None,
    },
]

# Plain-language description of what the pipeline does (pipeline/cli.py `run`).
METHOD_STEPS = [
    ("Infrastructure", "Download mapped assets from OpenStreetMap for the study area."),
    ("Terrain", "Derive slope, topographic wetness (TWI), height above nearest "
                "drainage (HAND) and flow accumulation from the SRTM 30 m DEM."),
    ("Observed floods", "Map standing water from Sentinel-1 radar during past "
                        "floods (2017-2024), against a dry-season baseline, with "
                        "permanent water and steep ground removed. Ground that "
                        "flooded in at least two events is the training label."),
    ("Graph model", "Link each asset to its nearest neighbours and train a two-layer "
                    "GraphSAGE network to predict whether an asset sits on "
                    "flood-prone ground. Validation holds out whole 10 km blocks."),
    ("Interpolation", "Krige the model scores into a continuous hazard surface "
                      "with a variance map."),
    ("Risk", "Combine hazard, exposure and vulnerability (population, distance "
             "to hospitals, shelters and roads) as a geometric mean on a ~500 m "
             "grid. Exposure is measured twice: by the infrastructure standing "
             "there, and by the people living there."),
    ("Aggregation", "Summarise by union, upazila and district, and detect "
                    "statistically significant hotspots (Getis-Ord Gi*)."),
]

KNOWN_LIMITATIONS = [
    "The hazard surface is interpolated from scores at mapped assets, and "
    "between them its confidence interval covers most of the 0-1 range. Read "
    "it as informative near mapped infrastructure and weak far from it.",
    "Two scores are reported per area. The infrastructure score rises where "
    "more is built, so cities lead it; the population score rises where more "
    "people live. Neither is a measure of the other.",
    "Population comes from WorldPop's constrained product, which carries values "
    "only where buildings were detected, so scattered rural settlement can be "
    "missed entirely.",
    "Flood labels come from Sentinel-1 radar, which cannot tell inundated "
    "paddy from a flood, and which passes only every 6-12 days, so short-lived "
    "surge flooding on the coast is under-recorded.",
    "OpenStreetMap coverage is uneven: places with few mapped assets show low "
    "exposure even where many people live.",
    "Vulnerability uses population density and access distances only; income, "
    "housing and age structure are not yet included.",
    "Scores describe relative susceptibility under typical monsoon conditions. "
    "They are not forecasts of any specific flood event.",
]
