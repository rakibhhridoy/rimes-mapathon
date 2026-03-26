"""
AlphaEarth integration — download Google satellite embeddings via Earth Engine,
cluster with UMAP+KMeans, and export as GeoJSON for map overlay.

Requires: earthengine-api, umap-learn, scikit-learn
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Earth Engine ImageCollection for satellite embeddings
EE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def check_ee_auth() -> bool:
    """Check if Earth Engine is authenticated and initialized."""
    try:
        import ee
        ee.Initialize()
        return True
    except Exception:
        try:
            import ee
            ee.Authenticate()
            ee.Initialize()
            return True
        except Exception as e:
            logger.warning(f"Earth Engine not available: {e}")
            return False


def download_embeddings(bbox: list, year: int, output_dir: str,
                        scale: int = 256) -> dict | None:
    """
    Download AlphaEarth satellite embeddings for the given bbox and year.

    Args:
        bbox: [west, south, east, north]
        year: Year for annual embeddings
        output_dir: Directory to save outputs
        scale: Resolution in meters (default 256m)

    Returns:
        dict with 'coords' (N,2), 'embeddings' (N,64) arrays and paths,
        or None on failure.
    """
    try:
        import ee
        ee.Initialize()
    except Exception as e:
        logger.error(f"Earth Engine init failed: {e}")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region = ee.Geometry.Rectangle(bbox)

    # Get the annual embedding image for the given year
    collection = ee.ImageCollection(EE_COLLECTION)
    image = collection.filter(
        ee.Filter.calendarRange(year, year, "year")
    ).first()

    if image is None:
        logger.error(f"No AlphaEarth image found for year {year}")
        return None

    # Get band names (should be 64 embedding dimensions)
    band_names = image.bandNames().getInfo()
    n_bands = len(band_names)
    logger.info(f"AlphaEarth: {n_bands} bands for year {year}")

    # Sample the image at regular grid points within bbox
    # Use sampleRectangle for small areas, or reduceRegion for large areas
    try:
        # Try sampleRectangle (works for smaller regions)
        sample = image.sampleRectangle(region=region, defaultValue=0)
        arrays = {}
        for band in band_names:
            arr = np.array(sample.get(band).getInfo())
            arrays[band] = arr

        # Stack into (H, W, 64) then reshape to (N, 64)
        h, w = arrays[band_names[0]].shape
        embeddings_3d = np.stack([arrays[b] for b in band_names], axis=-1)
        embeddings = embeddings_3d.reshape(-1, n_bands)

        # Generate coordinate grid
        west, south, east, north = bbox
        lons = np.linspace(west, east, w)
        lats = np.linspace(north, south, h)  # north to south
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        coords = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=-1)

    except Exception as e:
        logger.warning(f"sampleRectangle failed ({e}), using sample() instead")
        # Fallback: sample at points
        points = image.sample(
            region=region,
            scale=scale,
            numPixels=5000,
            geometries=True,
        )
        features = points.getInfo()["features"]
        if not features:
            logger.error("No sample points returned")
            return None

        embeddings = []
        coords = []
        for f in features:
            props = f["properties"]
            geom = f["geometry"]["coordinates"]
            vec = [props.get(b, 0) for b in band_names]
            embeddings.append(vec)
            coords.append(geom)

        embeddings = np.array(embeddings)
        coords = np.array(coords)

    # Filter out zero/invalid embeddings
    valid_mask = np.abs(embeddings).sum(axis=1) > 0
    embeddings = embeddings[valid_mask]
    coords = coords[valid_mask]

    logger.info(f"Downloaded {len(embeddings)} valid embedding points")

    # Save
    emb_path = output_dir / "alphaearth_embeddings.npy"
    coord_path = output_dir / "alphaearth_coords.npy"
    np.save(emb_path, embeddings)
    np.save(coord_path, coords)

    return {
        "embeddings": embeddings,
        "coords": coords,
        "emb_path": str(emb_path),
        "coord_path": str(coord_path),
        "n_points": len(embeddings),
        "n_dims": n_bands,
    }


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 8,
                       umap_dims: int = 2) -> dict:
    """
    Reduce dimensionality with UMAP and cluster with KMeans.

    Returns dict with 'labels', 'centroids_2d', 'umap_coords'.
    """
    from sklearn.cluster import KMeans
    from umap import UMAP

    logger.info(f"UMAP reduction: {embeddings.shape[1]}d → {umap_dims}d")
    reducer = UMAP(n_components=umap_dims, random_state=42, n_neighbors=15)
    umap_coords = reducer.fit_transform(embeddings)

    logger.info(f"KMeans clustering: k={n_clusters}")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(umap_coords)

    # Compute centroids in UMAP space
    centroids_2d = km.cluster_centers_

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  Cluster {u}: {c} points")

    return {
        "labels": labels,
        "centroids_2d": centroids_2d,
        "umap_coords": umap_coords,
    }


def export_clusters_geojson(coords: np.ndarray, labels: np.ndarray,
                            output_path: str,
                            embeddings: np.ndarray = None) -> str:
    """
    Export clustered points as GeoJSON for map overlay.

    Args:
        coords: (N, 2) array of [lon, lat]
        labels: (N,) cluster labels
        output_path: Path for output GeoJSON
        embeddings: optional (N, D) embeddings for properties

    Returns:
        Path to written GeoJSON file.
    """
    # Cluster colors
    cluster_colors = [
        "#3cb8de", "#3cdea0", "#dea03c", "#de7a3c",
        "#de3c78", "#a03cde", "#88b860", "#de3c3c",
        "#38bdf8", "#22c55e", "#f59e0b", "#ef4444",
    ]

    features = []
    for i in range(len(coords)):
        props = {
            "cluster": int(labels[i]),
            "color": cluster_colors[int(labels[i]) % len(cluster_colors)],
        }
        if embeddings is not None:
            # Store first 3 PCA-like components for tooltip
            props["emb_norm"] = float(np.linalg.norm(embeddings[i]))

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(coords[i, 0]), float(coords[i, 1])],
            },
            "properties": props,
        }
        features.append(feature)

    # Also add cluster centroids as separate features
    n_clusters = len(set(labels))
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue
        centroid_lon = float(coords[mask, 0].mean())
        centroid_lat = float(coords[mask, 1].mean())
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [centroid_lon, centroid_lat],
            },
            "properties": {
                "cluster": c,
                "is_centroid": True,
                "color": cluster_colors[c % len(cluster_colors)],
                "n_points": int(mask.sum()),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    logger.info(f"AlphaEarth clusters exported → {output_path} "
                f"({len(features)} features, {n_clusters} clusters)")
    return str(output_path)


def run_alphaearth_pipeline(cfg: dict) -> str | None:
    """
    Full AlphaEarth pipeline: download → cluster → export.

    Returns path to output GeoJSON, or None on failure.
    """
    if not check_ee_auth():
        logger.error("Earth Engine authentication required. "
                     "Run ee.Authenticate() interactively first.")
        return None

    bbox = cfg["aoi"]["bbox"]
    output_dir = Path(cfg.get("alphaearth", {}).get(
        "output_dir", "data/output"))
    n_clusters = cfg.get("alphaearth", {}).get("n_clusters", 8)
    year = cfg.get("alphaearth", {}).get("year", 2024)
    scale = cfg.get("alphaearth", {}).get("scale", 256)

    # Step 1: Download
    logger.info("=== AlphaEarth Step 1: Download embeddings ===")
    result = download_embeddings(bbox, year, str(output_dir), scale=scale)
    if result is None:
        return None

    # Step 2: Cluster
    logger.info("=== AlphaEarth Step 2: UMAP + KMeans clustering ===")
    cluster_result = cluster_embeddings(result["embeddings"], n_clusters=n_clusters)

    # Step 3: Export
    logger.info("=== AlphaEarth Step 3: Export GeoJSON ===")
    geojson_path = output_dir / "alphaearth_clusters.geojson"
    export_clusters_geojson(
        result["coords"],
        cluster_result["labels"],
        str(geojson_path),
        embeddings=result["embeddings"],
    )

    # Also save UMAP coordinates for visualization
    umap_path = output_dir / "alphaearth_umap.npy"
    np.save(umap_path, cluster_result["umap_coords"])

    logger.info(f"AlphaEarth pipeline complete: {result['n_points']} points, "
                f"{n_clusters} clusters")
    return str(geojson_path)
