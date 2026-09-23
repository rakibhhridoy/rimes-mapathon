"""
Pre-build fast binary caches from GeoJSON/GeoTIFF files.
Run after a pipeline run, once per region:
    python preprocess_cache.py                        # base region
    python preprocess_cache.py -c configs/sylhet.yaml # another region

Converts:
  - 71MB risk_grid.geojson      → ~8MB  risk_grid.parquet
  - 18MB risk_ranked_assets     → ~2MB  risk_ranked_assets.parquet
  - 584KB union_risk_summary    → ~60KB union_risk_summary.parquet
  - hotspot_clusters.geojson    → hotspot_clusters.parquet
  - Raster overlays (GeoTIFF)   → pre-rendered PNG base64 JSON
"""

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.cli import _load_config

# Set from the config in main(); the dashboard resolves the same paths via
# dashboard/data/regions.py, so both sides stay in step.
OUTPUT_DIR = Path("data/output")
PROCESSED_DIR = Path("data/processed")
CACHE_DIR = Path("data/cache")


def _set_dirs(config_path: str) -> str:
    """Point the module at one region's directories."""
    global OUTPUT_DIR, PROCESSED_DIR, CACHE_DIR
    cfg = _load_config(config_path)
    paths = cfg.get("paths", {}) or {}
    OUTPUT_DIR = Path(paths.get("output_dir", "data/output"))
    PROCESSED_DIR = Path(paths.get("processed_dir", "data/processed"))
    CACHE_DIR = OUTPUT_DIR.parent / "cache"
    return cfg.get("aoi", {}).get("name", "region")


def convert_geojson_to_parquet():
    """Convert large GeoJSON files to GeoParquet for 5-10x faster loading."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        "risk_ranked_assets.geojson",
        "union_risk_summary.geojson",
        "district_risk_summary.geojson",
        "hotspot_clusters.geojson",
        "risk_grid.geojson",
        "upazila_risk_summary.geojson",
        "top50_risk_assets.geojson",
    ]

    for fname in files:
        src = OUTPUT_DIR / fname
        dst = CACHE_DIR / fname.replace(".geojson", ".parquet")
        if not src.exists():
            print(f"  SKIP {fname} (not found)")
            continue

        t0 = time.time()
        print(f"  Converting {fname} ({src.stat().st_size / 1024 / 1024:.1f}MB)...", end=" ", flush=True)
        gdf = gpd.read_file(src)

        # Simplify display geometry. The browser redraws these polygons on
        # every pan, and 1,339 unions at full precision is ~36 MB; ~50 m
        # tolerance is invisible at the zoom levels the map allows.
        if "risk_grid" in fname:
            gdf.geometry = gdf.geometry.simplify(tolerance=0.001, preserve_topology=True)
        elif any(level in fname for level in ("union", "upazila", "district")):
            gdf.geometry = gdf.geometry.simplify(tolerance=0.0005, preserve_topology=True)

        gdf.to_parquet(dst)
        dt = time.time() - t0
        print(f"→ {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f}MB) [{dt:.1f}s]")


def prerender_raster_overlays():
    """Pre-render raster overlays as base64 PNG for folium ImageOverlay.

    Rasters are reprojected to WGS84 first: ImageOverlay places an image by
    lat/lon corner bounds, so a UTM raster rendered as-is lands in the wrong
    place. Large rasters are downsampled — the overlay is a picture, and the
    browser cannot use more than a couple of thousand pixels a side.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    raster_map = {
        "dem": PROCESSED_DIR / "dem_reprojected.tif",
        "slope": PROCESSED_DIR / "dem_derivatives" / "slope.tif",
        "hand": PROCESSED_DIR / "dem_derivatives" / "hand.tif",
        "flood_risk": OUTPUT_DIR / "flood_risk_kriged.tif",
        "kriging_variance": OUTPUT_DIR / "kriging_variance.tif",
        "landslide": OUTPUT_DIR / "landslide_susceptibility.tif",
    }

    cmap_map = {
        "dem": "terrain",
        "slope": "YlOrRd",
        "hand": "Blues_r",
        "flood_risk": "RdYlGn_r",
        "kriging_variance": "Purples",
        "landslide": "OrRd",
    }

    max_dim = 2000

    for name, path in raster_map.items():
        dst = CACHE_DIR / f"raster_{name}.json"
        if not path.exists():
            print(f"  SKIP raster {name} (not found)")
            continue

        t0 = time.time()
        print(f"  Pre-rendering {name}...", end=" ", flush=True)

        try:
            import base64
            from io import BytesIO

            import rasterio
            from matplotlib import colormaps
            from PIL import Image
            from rasterio.enums import Resampling
            from rasterio.warp import calculate_default_transform, reproject

            with rasterio.open(path) as src:
                dst_crs = "EPSG:4326"
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                scale = max(width / max_dim, height / max_dim, 1)
                width, height = int(width / scale), int(height / scale)
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds,
                    dst_width=width, dst_height=height,
                )

                data = np.full((height, width), np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )

            west, north = transform * (0, 0)
            east, south = transform * (width, height)

            valid = data[~np.isnan(data)]
            if len(valid) == 0:
                print("SKIP (no valid data)")
                continue

            vmin, vmax = np.percentile(valid, [2, 98])
            if vmax - vmin < 1e-6:
                print("SKIP (no range)")
                continue

            norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            rgba = colormaps[cmap_map[name]](np.nan_to_num(norm))
            rgba[np.isnan(data)] = [0, 0, 0, 0]
            rgba[:, :, 3] *= 0.6

            img = Image.fromarray((rgba * 255).astype(np.uint8))
            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)

            result = {
                "image_base64": base64.b64encode(buf.getvalue()).decode(),
                "bounds": [[south, west], [north, east]],
                "name": name,
                "vmin": float(vmin),
                "vmax": float(vmax),
            }
            with open(dst, "w") as f:
                json.dump(result, f)

            print(f"→ {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f}MB) "
                  f"[{time.time() - t0:.1f}s]")
        except Exception as e:
            print(f"ERROR: {e}")


def precompute_heatmap_data():
    """Extract heatmap points from risk_grid so we don't iterate 71MB at runtime."""
    src = OUTPUT_DIR / "risk_grid.geojson"
    dst = CACHE_DIR / "heatmap_points.json"
    if not src.exists():
        print("  SKIP heatmap (risk_grid not found)")
        return

    t0 = time.time()
    print("  Extracting heatmap points...", end=" ", flush=True)

    gdf = gpd.read_file(src)
    if "composite_risk" not in gdf.columns:
        print("SKIP (no composite_risk column)")
        return

    centroids = gdf.geometry.centroid
    risks = gdf["composite_risk"].values
    mask = risks > 0.1

    points = [
        [round(float(centroids.iloc[i].y), 5),
         round(float(centroids.iloc[i].x), 5),
         round(float(risks[i]), 3)]
        for i in range(len(gdf)) if mask[i]
    ]

    with open(dst, "w") as f:
        json.dump(points, f)

    dt = time.time() - t0
    print(f"→ {dst.name} ({len(points)} points) [{dt:.1f}s]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="Region config (default: config.yaml)")
    args = parser.parse_args()
    region_name = _set_dirs(args.config)

    print(f"=== Fermium HazMapper — Pre-processing Cache ({region_name}) ===\n")
    print(f"  output:  {OUTPUT_DIR}\n  cache:   {CACHE_DIR}\n")

    print("[1/3] Converting GeoJSON → GeoParquet:")
    convert_geojson_to_parquet()

    print("\n[2/3] Pre-rendering raster overlays:")
    prerender_raster_overlays()

    print("\n[3/3] Pre-computing heatmap data:")
    precompute_heatmap_data()

    print(f"\nDone! Cache files in {CACHE_DIR}/")
