"""
Bangladesh's administrative hierarchy for the website: division, district,
upazila and union, with each hazard's figures summed up through it.

geoBoundaries publishes districts (ADM2), upazilas (ADM3) and unions (ADM4)
as separate files without parent links, so each unit is placed in its parent
by a point inside it. Divisions are not published at all and are dissolved
from their districts. The map draws one level at a time, chosen by zoom, so
every level carries its own figures:

    flood      share of assets in the High or Very high class (score >= 0.5),
               mean score, asset counts and population, from every flood
               region's assets
    landslide  mean and highest susceptibility, population and mapped
               landslides, from the Hill Tracts surface

Nothing here models anything; it adds up what the pipeline wrote.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES = ROOT / "data" / "shared" / "geoboundaries"
WORLDPOP = ROOT / "data" / "shared" / "worldpop" / "bgd_ppp_2020_constrained.tif"

LEVELS = ["division", "district", "upazila", "union"]

# The eight divisions and their districts, under geoBoundaries' district names.
DIVISIONS = {
    "Barishal": ["Barguna", "Barisal", "Bhola", "Jhalokati", "Patuakhali", "Pirojpur"],
    "Chattogram": ["Bandarban", "Brahamanbaria", "Chandpur", "Chittagong", "Comilla",
                   "Cox's Bazar", "Feni", "Khagrachhari", "Lakshmipur", "Noakhali", "Rangamati"],
    "Dhaka": ["Dhaka", "Faridpur", "Gazipur", "Gopalganj", "Kishoreganj", "Madaripur",
              "Manikganj", "Munshiganj", "Narayanganj", "Narsingdi", "Rajbari", "Shariatpur",
              "Tangail"],
    "Khulna": ["Bagerhat", "Chuadanga", "Jessore", "Jhenaidah", "Khulna", "Kushtia", "Magura",
               "Meherpur", "Narail", "Satkhira"],
    "Mymensingh": ["Jamalpur", "Mymensingh", "Netrakona", "Sherpur"],
    "Rajshahi": ["Bogra", "Joypurhat", "Naogaon", "Natore", "Nawabganj", "Pabna", "Rajshahi",
                 "Sirajganj"],
    "Rangpur": ["Dinajpur", "Gaibandha", "Kurigram", "Lalmonirhat", "Nilphamari", "Panchagarh",
                "Rangpur", "Thakurgaon"],
    "Sylhet": ["Habiganj", "Maulvibazar", "Sunamganj", "Sylhet"],
}
DISTRICT_DIVISION = {d: div for div, ds in DIVISIONS.items() for d in ds}

# An asset on a char or the shore can fall just outside every union polygon;
# within this distance it goes to the nearest one.
NEAREST_M = 2000
METRIC_CRS = "EPSG:3106"        # Gulshan 303 / Bangladesh TM


def _parent(child, parent, column: str):
    """The parent each child unit lies in, by a point inside the child."""
    import geopandas as gpd

    points = child[["geometry"]].copy()
    points["geometry"] = child.representative_point()
    hit = gpd.sjoin(points, parent[[column, "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    missing = hit[column].isna()
    if missing.any():
        near = gpd.sjoin_nearest(points.loc[missing[missing].index].to_crs(METRIC_CRS),
                                 parent[[column, "geometry"]].to_crs(METRIC_CRS), how="left")
        near = near[~near.index.duplicated(keep="first")]
        hit.loc[near.index, column] = near[column]
    return hit[column].reindex(child.index).values


def hierarchy() -> dict:
    """One GeoDataFrame per level, each unit knowing every unit above it.

    Columns: id, name, division, district, upazila (the upazila's id), union
    (the union's id), geometry. Districts are identified by name, which is
    unique; upazilas and unions by their geoBoundaries shapeID, since names
    repeat across the country.
    """
    import geopandas as gpd

    districts = gpd.read_file(BOUNDARIES / "BGD_ADM2.geojson").to_crs("EPSG:4326")
    upazilas = gpd.read_file(BOUNDARIES / "BGD_ADM3.geojson").to_crs("EPSG:4326")
    unions = gpd.read_file(BOUNDARIES / "BGD_ADM4.geojson").to_crs("EPSG:4326")

    districts["district"] = districts["shapeName"]
    unknown = set(districts["district"]) - set(DISTRICT_DIVISION)
    if unknown:
        raise ValueError(f"districts without a division: {sorted(unknown)}")
    districts["division"] = districts["district"].map(DISTRICT_DIVISION)
    districts["id"] = districts["district"]

    upazilas["district"] = _parent(upazilas, districts, "district")
    upazilas["division"] = upazilas["district"].map(DISTRICT_DIVISION)
    upazilas["upazila"] = upazilas["id"] = upazilas["shapeID"]

    unions["upazila"] = _parent(unions, upazilas, "upazila")
    lookup = upazilas.set_index("upazila")
    unions["district"] = unions["upazila"].map(lookup["district"])
    unions["division"] = unions["upazila"].map(lookup["division"])
    unions["union"] = unions["id"] = unions["shapeID"]

    divisions = districts.dissolve(by="division", as_index=False)[["division", "geometry"]]
    divisions["id"] = divisions["division"]

    for frame, names in ((divisions, "division"), (districts, "shapeName"),
                         (upazilas, "shapeName"), (unions, "shapeName")):
        frame["name"] = frame[names]
    cols = {"division": ["id", "name", "division"],
            "district": ["id", "name", "division", "district"],
            "upazila": ["id", "name", "division", "district", "upazila"],
            "union": ["id", "name", "division", "district", "upazila", "union"]}
    out = {}
    for level, frame in zip(LEVELS, (divisions, districts, upazilas, unions)):
        frame = frame[cols[level] + ["geometry"]].copy()
        frame["level"] = level
        out[level] = frame.reset_index(drop=True)
    return out


def assign(points, levels: dict) -> pd.DataFrame:
    """The union, upazila, district and division each point falls in.

    `points` is a GeoDataFrame of points in EPSG:4326. Points beyond
    NEAREST_M of every union keep no area.
    """
    import geopandas as gpd

    unions = levels["union"]
    keep = ["union", "upazila", "district", "division"]
    hit = gpd.sjoin(points[["geometry"]], unions[keep + ["geometry"]], how="left",
                    predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")][keep]
    missing = hit["union"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(points.loc[missing[missing].index, ["geometry"]].to_crs(METRIC_CRS),
                                 unions[keep + ["geometry"]].to_crs(METRIC_CRS),
                                 how="left", max_distance=NEAREST_M)
        near = near[~near.index.duplicated(keep="first")]
        hit.loc[near.index, keep] = near[keep].values
    return hit.reindex(points.index)


def _zonal(unions, raster: Path, nodata_floor=None) -> pd.DataFrame:
    """Per-union sum, count, and maximum of a raster's valid pixels.

    The unions are burnt into the raster's own grid once, so every pixel
    counts towards exactly one union, and the sums add up exactly through
    the levels above.
    """
    import rasterio
    from rasterio import features

    with rasterio.open(raster) as src:
        values = src.read(1, masked=True)
        shapes = unions.to_crs(src.crs)
        burnt = features.rasterize(
            ((geom, i + 1) for i, geom in enumerate(shapes.geometry)),
            out_shape=src.shape, transform=src.transform, fill=0, dtype="int32")
    data = values.filled(np.nan).astype("float64")
    valid = ~np.isnan(data) & (burnt > 0)
    if nodata_floor is not None:
        valid &= data > nodata_floor
    ids = burnt[valid]
    vals = data[valid]
    n = len(unions) + 1
    total = np.bincount(ids, weights=vals, minlength=n)[1:]
    count = np.bincount(ids, minlength=n)[1:]
    peak = np.full(n, np.nan)
    np.fmax.at(peak, ids, vals)
    return pd.DataFrame({"sum": total, "count": count, "max": peak[1:]}, index=unions.index)


def population(levels: dict) -> pd.Series:
    """WorldPop 2020 people per union."""
    if not WORLDPOP.exists():
        return pd.Series(0.0, index=levels["union"].index)
    return _zonal(levels["union"], WORLDPOP, nodata_floor=-1)["sum"]


def _bbox(frame) -> pd.DataFrame:
    b = frame.geometry.bounds
    inside = frame.geometry.representative_point()
    return pd.DataFrame({"west": b.minx, "south": b.miny, "east": b.maxx, "north": b.maxy,
                         "lon": inside.x, "lat": inside.y}, index=frame.index)


def flood_areas(levels: dict, assets: pd.DataFrame, people: pd.Series) -> dict:
    """Flood figures for every level, from every flood region's assets.

    `assets` has one row per asset with region, asset_type, flood_risk and the
    four area columns from `assign`. A unit with no assets keeps its outline
    and no figures, so the map shows it as unscored rather than safe.
    """
    a = assets.copy()
    a["high"] = (a["flood_risk"] >= 0.5).astype(int)
    for kind in ("hospital", "school", "bridge", "flood_shelter", "cropland"):
        a[f"n_{kind}"] = (a["asset_type"] == kind).astype(int)
    union_people = levels["union"].assign(people=people.values)

    out = {}
    for level in LEVELS:
        units = levels[level].copy()
        grouped = a.dropna(subset=[level]).groupby(level)
        stats = grouped.agg(
            n_assets=("flood_risk", "size"), n_high=("high", "sum"),
            mean_score=("flood_risk", "mean"),
            n_hospital=("n_hospital", "sum"), n_school=("n_school", "sum"),
            n_bridge=("n_bridge", "sum"), n_shelter=("n_flood_shelter", "sum"),
            n_cropland=("n_cropland", "sum"))
        # the flood region holding most of the unit's assets; a division can
        # span several, so it lists them all
        by_region = a.dropna(subset=[level]).groupby([level, "region"]).size()
        main = by_region.sort_values(ascending=False).reset_index().drop_duplicates(level)
        stats["region"] = main.set_index(level)["region"]
        stats["regions"] = by_region.reset_index().groupby(level)["region"].agg(
            lambda s: ",".join(sorted(s)))
        units = units.join(stats, on="id")
        units["share_high"] = units["n_high"] / units["n_assets"]
        units["population"] = units["id"].map(union_people.groupby(level)["people"].sum()).values
        out[level] = units.join(_bbox(units))
    return out


def landslide_areas(levels: dict, surface: Path, upazila_figures: list, points,
                    people: pd.Series) -> dict:
    """Landslide figures for the districts, upazilas and unions the surface covers.

    Upazilas keep the pipeline's own figures, so the map agrees with the
    rankings table; unions are read from the surface here, and districts are
    the pixel-weighted sum of their upazilas.
    """
    unions = levels["union"]
    zonal = _zonal(unions, surface, nodata_floor=-1)
    covered = zonal["count"] > 0
    u = unions[covered].copy()
    u["susceptibility_mean"] = (zonal["sum"] / zonal["count"])[covered].values
    u["susceptibility_max"] = zonal["max"][covered].values
    u["n_pixels"] = zonal["count"][covered].values
    u["population"] = people[covered].values

    upazilas = levels["upazila"]
    figures = {row["admin_label"]: row for row in upazila_figures}
    labels = upazilas["name"] + " (" + upazilas["district"] + ")"
    z = upazilas[labels.isin(figures)].copy()
    z_labels = labels[labels.isin(figures)]
    for key in ("susceptibility_mean", "susceptibility_max", "n_pixels", "population"):
        z[key] = [figures[label].get(key) for label in z_labels]
    u = u[u["upazila"].isin(z["id"])]

    d = z.groupby("district").apply(lambda g: pd.Series({
        "susceptibility_mean": np.average(g["susceptibility_mean"], weights=g["n_pixels"]),
        "susceptibility_max": g["susceptibility_max"].max(),
        "n_pixels": g["n_pixels"].sum(), "population": g["population"].sum()}),
        include_groups=False)
    districts = levels["district"]
    d = districts[districts["id"].isin(d.index)].join(d, on="id")

    if points is not None and len(points):
        where = assign(points, levels)
        for frame, level in ((u, "union"), (z, "upazila"), (d, "district")):
            frame["n_landslides"] = frame["id"].map(where[level].value_counts()).fillna(0).astype(int)
    else:
        for frame in (u, z, d):
            frame["n_landslides"] = 0
    return {level: frame.join(_bbox(frame)) for level, frame in
            (("district", d), ("upazila", z), ("union", u))}
