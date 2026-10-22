"""
Published flood maps scored as rivals to the model and the flood record.

Why this exists: the record of earlier flooding outranked the terrain model on
later floods, but agencies plan with hydrodynamic hazard maps rather than with
susceptibility models, and a radar record has to be built before it can be
used. Two published maps cover both cases on the same assets:

- the JRC global river flood hazard maps (Baugh et al. 2024, GloFAS, 3 arc
  seconds), simulated with LISFLOOD-FP for return periods of 10 to 500 years.
  An asset scores the annual exceedance probability of the most frequent
  flood that reaches it, 1/10 for the 10-year map down to 1/500, and 0 when
  no map does. Ties are broken by the 500-year depth.
- JRC Global Surface Water occurrence (Pekel et al. 2016), the share of
  Landsat observations from 1984 to 2021 that saw water, which is a record of
  flooding available anywhere without processing radar.

The hazard maps are read from data/shared/jrc_hazard, where fetch.sh cuts the
Bangladesh window from the JRC Data Store tiles (CEMS-GLOFAS/flood_hazard,
version 2.1.2).
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HAZARD_DIR = ROOT / "data" / "shared" / "jrc_hazard"
RETURN_PERIODS = (10, 20, 50, 75, 100, 200, 500)
# The Global Surface Water occurrence layer ends in 2021, so it is an earlier
# record only for floods after that year.
OCCURRENCE_LAST_YEAR = 2021


def _sample_tiles(rp: int, points) -> np.ndarray:
    """Depth in metres at each (lon, lat), 0 outside the map."""
    import rasterio

    depth = np.zeros(len(points))
    lon = np.array([p[0] for p in points])
    lat = np.array([p[1] for p in points])
    for path in sorted(HAZARD_DIR.glob(f"bgd_RP{rp}_depth.tif")):
        with rasterio.open(path) as src:
            b = src.bounds
            inside = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top)
            if not inside.any():
                continue
            values = np.array([v[0] for v in src.sample(
                list(zip(lon[inside], lat[inside])))], dtype=float)
            if src.nodata is not None:
                values[values == src.nodata] = 0.0
            depth[inside] = np.nan_to_num(values, nan=0.0)
    return np.clip(depth, 0.0, None)


def hazard_map_scores(points) -> np.ndarray | None:
    """Annual exceedance probability of the first flood that reaches each asset."""
    if not all((HAZARD_DIR / f"bgd_RP{rp}_depth.tif").exists() for rp in RETURN_PERIODS):
        return None
    score = np.zeros(len(points))
    for rp in reversed(RETURN_PERIODS):           # most frequent last, so it wins
        score[_sample_tiles(rp, points) > 0] = 1.0 / rp
    tie_break = _sample_tiles(RETURN_PERIODS[-1], points)
    return score + 1e-6 * tie_break / max(tie_break.max(), 1e-9)


def occurrence_scores(raw_dir: Path, points) -> np.ndarray | None:
    from pipeline.feature_extract import sample_raster_at_points

    path = raw_dir / "jrc_water_occurrence.tif"
    if not path.exists():
        return None
    return np.nan_to_num(sample_raster_at_points(str(path), points))
