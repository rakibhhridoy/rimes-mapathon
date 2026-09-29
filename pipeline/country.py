"""
Keep the study to one country.

The regions are bounding boxes, and on Bangladesh's land borders a box takes
in ground across the border: two fifths of the mapped assets in the Rangpur
and Rajshahi box were in West Bengal, and a quarter of the Sylhet box's in
Meghalaya, Assam and Tripura. Assets, grid cells and grid-level scores there
describe another country's infrastructure and are dropped.

The rule removes what lies inside a neighbouring country, or on land more
than about 1 km outside the country's own polygon, rather than keeping only
what lies inside that polygon. Administrative polygons stop
at the coastline and at some river channels, so "inside Bangladesh" would also
drop embankments, bridges and ponds on chars and coastal islands, which belong
to the study. Neighbours come from Natural Earth, whose borders are coarser
than geoBoundaries', so the country's own polygon is subtracted from them and
the finer border wins wherever the two disagree, and the 1 km margin catches
the slivers the coarse border hands to the home country.

Configured under `aoi`:

    country_boundary: data/shared/geoboundaries/BGD_ADM2.geojson
    neighbours: data/shared/naturalearth/ne_10m_admin_0_countries.geojson

Without both keys nothing is dropped, so a region elsewhere keeps working.
"""

import logging
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np

logger = logging.getLogger("sgmdi.country")


@lru_cache(maxsize=8)
def _foreign(country_path: str, neighbours_path: str, iso3: str, bbox: tuple):
    from shapely.geometry import box

    area = box(*bbox).buffer(0.5)
    country = gpd.read_file(country_path).to_crs("EPSG:4326")
    home = country.geometry.union_all()
    # About 1 km, in degrees: chars and coastal ground just outside the
    # administrative polygon stay in the study.
    margin = home.buffer(0.01)
    world = gpd.read_file(neighbours_path).to_crs("EPSG:4326")
    world = world[world.intersects(area)]
    code = "ADM0_A3" if "ADM0_A3" in world.columns else "ISO_A3"
    others = world[world[code] != iso3]
    # Natural Earth's border is coarse enough to leave slivers that it gives
    # to the home country and geoBoundaries does not, so land well outside the
    # home polygon counts as foreign whichever country Natural Earth names.
    land = world.geometry.union_all().intersection(area)
    foreign = land.difference(margin)
    if not others.empty:
        foreign = foreign.union(others.geometry.union_all().intersection(area).difference(home))
    return foreign


def foreign_geometry(cfg: dict):
    """Neighbouring countries' ground near the region, or None if unconfigured."""
    aoi = cfg.get("aoi") or {}
    country_path, neighbours_path = aoi.get("country_boundary"), aoi.get("neighbours")
    if not (country_path and neighbours_path):
        return None
    for path in (country_path, neighbours_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} is configured as a country boundary but is missing; "
                "without it assets across the border would be kept silently")
    return _foreign(str(country_path), str(neighbours_path),
                    aoi.get("iso3", "BGD"), tuple(aoi["bbox"]))


def drop_foreign(gdf: gpd.GeoDataFrame, cfg: dict, what: str = "features") -> gpd.GeoDataFrame:
    """Rows whose representative point is not in a neighbouring country."""
    foreign = foreign_geometry(cfg)
    if foreign is None or gdf.empty:
        return gdf
    points = gdf.to_crs("EPSG:4326").geometry.representative_point()
    inside = points.within(foreign).to_numpy()
    if inside.any():
        logger.info(f"Dropped {int(inside.sum())} of {len(gdf)} {what} "
                    "that lie in a neighbouring country")
    return gdf[~inside]


def foreign_mask(cfg: dict, shape, transform, crs) -> np.ndarray:
    """Boolean raster, True where a cell centre lies in a neighbouring country."""
    from rasterio.features import rasterize

    foreign = foreign_geometry(cfg)
    if foreign is None or foreign.is_empty:
        return np.zeros(shape, dtype=bool)
    geometry = gpd.GeoSeries([foreign], crs="EPSG:4326").to_crs(crs).iloc[0]
    return rasterize([(geometry, 1)], out_shape=shape, transform=transform,
                     fill=0, dtype="uint8").astype(bool)
