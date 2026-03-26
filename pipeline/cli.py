"""
CLI entry point for the SGMDI pipeline.
Provides step-by-step commands and a full `run` command.
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgmdi")


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class ConfigGroup(click.Group):
    """Allow --config/-c to appear before or after the subcommand."""
    def parse_args(self, ctx, args):
        # If -c/--config appears after the subcommand, move it before
        reordered = list(args)
        for flag in ("-c", "--config"):
            if flag in reordered:
                idx = reordered.index(flag)
                if idx + 1 < len(reordered):
                    val = reordered.pop(idx + 1)
                    opt = reordered.pop(idx)
                    reordered = [opt, val] + reordered
        return super().parse_args(ctx, reordered)


@click.group(cls=ConfigGroup)
@click.option("--config", "-c", default="config.yaml", help="Path to config YAML")
@click.pass_context
def cli(ctx, config):
    """SGMDI — Smart Geospatial Mapping & Disaster Impact Intelligence Pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)
    ctx.obj["config_path"] = config


# ---------------------------------------------------------------------------
# Step 0 — Download
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def download(ctx):
    """Step 0: Download GADM boundaries, SRTM DEM, JRC water, WorldPop data."""
    from pipeline.data_download import download_all

    cfg = ctx.obj["config"]
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 0: Data Download ===")
    results = download_all(cfg, raw_dir)
    succeeded = sum(1 for v in results.values() if v)
    click.echo(f"Downloaded {succeeded}/{len(results)} datasets.")


# ---------------------------------------------------------------------------
# Step 1 — Ingest
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def ingest(ctx):
    """Step 1: Download data (if needed) then ingest OSM infrastructure."""
    from pipeline.data_ingest import fetch_osm_infrastructure

    cfg = ctx.obj["config"]
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Ensure required datasets are downloaded before ingestion
    ctx.invoke(download)

    logger.info("=== Step 1: Data Ingestion ===")
    infra = fetch_osm_infrastructure(cfg, raw_dir)
    click.echo(f"Fetched {len(infra)} infrastructure features.")


# ---------------------------------------------------------------------------
# Step 2 — Preprocess
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def preprocess(ctx):
    """Step 2: Reproject, clip, compute DEM derivatives, build flood labels."""
    from pipeline.data_ingest import preprocess_all

    cfg = ctx.obj["config"]
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    logger.info("=== Step 2: Preprocessing ===")
    outputs = preprocess_all(cfg, raw_dir, processed_dir)
    click.echo(f"Preprocessing complete. Outputs: {list(outputs.keys())}")


# ---------------------------------------------------------------------------
# Step 3 — Feature extraction
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def features(ctx):
    """Step 3: Extract features at infrastructure centroids."""
    import geopandas as gpd
    from pipeline.feature_extract import extract_features

    cfg = ctx.obj["config"]
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")

    infra_path = raw_dir / "infrastructure_raw.gpkg"
    if not infra_path.exists():
        click.echo("Error: Run 'ingest' first.", err=True)
        sys.exit(1)

    infra = gpd.read_file(str(infra_path))
    logger.info("=== Step 3: Feature Extraction ===")
    X, coords, y, scaler = extract_features(cfg, infra, processed_dir)
    click.echo(f"Features: {X.shape}, Labels positive rate: {y.mean():.2%}")


# ---------------------------------------------------------------------------
# Step 4 — Graph construction
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def graph(ctx):
    """Step 4: Build spatial k-NN graph."""
    import geopandas as gpd
    import numpy as np
    from pipeline.feature_extract import extract_features
    from pipeline.graph_build import build_spatial_graph, save_graph

    cfg = ctx.obj["config"]
    processed_dir = Path("data/processed")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    infra = gpd.read_file(str(Path("data/raw/infrastructure_raw.gpkg")))
    X, coords, y, _ = extract_features(cfg, infra, processed_dir)

    logger.info("=== Step 4: Graph Construction ===")
    graph_data = build_spatial_graph(X, coords, y, cfg)
    save_graph(graph_data, str(output_dir / "spatial_graph.pt"))
    click.echo(
        f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges"
    )


# ---------------------------------------------------------------------------
# Step 5-6 — Train GNN
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def train(ctx):
    """Step 5-6: Train GraphSAGE and extract embeddings."""
    import numpy as np
    from pipeline.graph_build import load_graph
    from pipeline.gnn_model import (
        train_model, extract_embeddings_and_scores, save_model,
    )

    cfg = ctx.obj["config"]
    output_dir = Path("data/output")

    graph_path = output_dir / "spatial_graph.pt"
    if not graph_path.exists():
        click.echo("Error: Run 'graph' first.", err=True)
        sys.exit(1)

    graph_data = load_graph(str(graph_path))

    logger.info("=== Step 5: Training GraphSAGE ===")
    model = train_model(graph_data, cfg)
    save_model(model, str(output_dir / "gnn_model.pt"))

    logger.info("=== Step 6: Extracting Embeddings ===")
    embeddings, risk_scores = extract_embeddings_and_scores(model, graph_data)
    np.save(str(output_dir / "node_embeddings.npy"), embeddings)
    np.save(str(output_dir / "gnn_risk_scores.npy"), risk_scores)

    click.echo(
        f"Model trained. Risk scores: mean={risk_scores.mean():.3f}, "
        f"std={risk_scores.std():.3f}"
    )


# ---------------------------------------------------------------------------
# Step 7 — Kriging
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def krige(ctx):
    """Step 7: Ordinary Kriging interpolation of flood risk."""
    import geopandas as gpd
    import numpy as np
    from pipeline.feature_extract import compute_centroids
    from pipeline.kriging import (
        fit_and_execute_kriging, hybrid_fusion,
        save_kriged_surface, save_variogram_params,
    )

    cfg = ctx.obj["config"]
    output_dir = Path("data/output")

    risk_scores = np.load(str(output_dir / "gnn_risk_scores.npy"))
    infra = gpd.read_file(str(Path("data/raw/infrastructure_raw.gpkg")))
    infra = compute_centroids(infra)
    coords = np.column_stack([infra["lon"].values, infra["lat"].values])

    bounds = tuple(cfg["aoi"]["bbox"])

    logger.info("=== Step 7: Kriging ===")
    z, ss, grid_lon, grid_lat, vparams = fit_and_execute_kriging(
        coords, risk_scores, cfg, bounds
    )

    # Hybrid fusion
    z_fused = hybrid_fusion(z, risk_scores, risk_scores, coords, grid_lon, grid_lat, cfg)

    save_kriged_surface(z_fused, grid_lon, grid_lat,
                         str(output_dir / "flood_risk_kriged.tif"),
                         "Hybrid GNN+Kriging flood risk")
    save_kriged_surface(ss, grid_lon, grid_lat,
                         str(output_dir / "kriging_variance.tif"),
                         "Kriging variance (uncertainty)")
    save_variogram_params(vparams, str(output_dir / "variogram_params.json"))

    click.echo(f"Kriged surface: {z_fused.shape}, mean risk: {z_fused.mean():.3f}")


# ---------------------------------------------------------------------------
# Step 8-9 — Risk scoring & ranking
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def risk(ctx):
    """Step 8-9: Composite risk, aggregation, hotspots, ranking."""
    import geopandas as gpd
    import numpy as np
    from pipeline.risk_score import (
        create_risk_grid, compute_exposure_grid, compute_vulnerability_grid,
        compute_composite_risk, aggregate_to_admin, detect_hotspots, rank_assets,
    )
    from pipeline.export import (
        export_ranked_csv, export_geojson, export_union_summary,
        export_hotspots, generate_pdf_report,
    )
    from pipeline.feature_extract import compute_centroids

    cfg = ctx.obj["config"]
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    infra = gpd.read_file(str(Path("data/raw/infrastructure_raw.gpkg")))
    infra = compute_centroids(infra)
    risk_scores = np.load(str(output_dir / "gnn_risk_scores.npy"))

    bounds = tuple(cfg["aoi"]["bbox"])
    grid_res = cfg["kriging"].get("grid_resolution_deg", 0.005)

    logger.info("=== Step 8: Composite Risk ===")
    grid_gdf = create_risk_grid(bounds, grid_res)

    # Sample kriged hazard at grid cells
    hazard = _sample_raster_at_grid(
        str(output_dir / "flood_risk_kriged.tif"), grid_gdf
    )
    exposure = compute_exposure_grid(infra, grid_gdf, cfg)
    vulnerability = compute_vulnerability_grid(grid_gdf, infra, cfg)

    grid_gdf["hazard"] = hazard
    grid_gdf["exposure"] = exposure
    grid_gdf["vulnerability"] = vulnerability
    grid_gdf["composite_risk"] = compute_composite_risk(hazard, exposure, vulnerability)

    logger.info("=== Step 9: Aggregation & Ranking ===")

    # Hotspots
    grid_gdf = detect_hotspots(
        grid_gdf, cfg["risk"].get("hotspot_confidence", 0.95)
    )

    # Rank individual assets
    ranked_infra = rank_assets(
        infra, risk_scores, cfg["risk"].get("high_risk_threshold", 0.7)
    )

    # Aggregate to union level (if boundaries available)
    union_path = cfg["data"]["vulnerability"].get("admin_boundaries_l3", "")
    union_gdf = None
    if Path(union_path).exists():
        admin_gdf = gpd.read_file(union_path)
        union_gdf = aggregate_to_admin(grid_gdf, admin_gdf, infra)
        export_union_summary(union_gdf, output_dir)

    # Export everything
    export_ranked_csv(ranked_infra, str(output_dir / "risk_ranked_assets.csv"))
    export_geojson(ranked_infra, str(output_dir / "risk_ranked_assets.geojson"))
    export_geojson(ranked_infra, str(output_dir / "top50_risk_assets.geojson"),
                    max_features=50)
    export_hotspots(grid_gdf, str(output_dir / "hotspot_clusters.geojson"))
    export_geojson(grid_gdf, str(output_dir / "risk_grid.geojson"))

    # PDF report
    generate_pdf_report(
        union_gdf, ranked_infra,
        str(output_dir / "situation_report.pdf")
    )

    # Upazila aggregation (L2)
    upazila_path = cfg["data"]["vulnerability"].get("admin_boundaries_l2", "")
    if Path(upazila_path).exists():
        from pipeline.risk_score import aggregate_to_upazila
        upazila_gdf = gpd.read_file(upazila_path)
        upazila_summary = aggregate_to_upazila(grid_gdf, upazila_gdf, infra)
        # Export GeoJSON
        export_geojson(upazila_summary, str(output_dir / "upazila_risk_summary.geojson"))
        # Export CSV for dashboard loader
        cols = [c for c in upazila_summary.columns if c != "geometry"]
        upazila_summary[cols].to_csv(
            str(output_dir / "upazila_risk_summary.csv"), index=False
        )
        logger.info("Upazila risk summary exported")

    click.echo("Risk assessment complete. Outputs in data/output/")


# ---------------------------------------------------------------------------
# Metadata — pipeline confidence metrics
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def metadata(ctx):
    """Export pipeline confidence metadata (kriging var, GNN CI, IQR, density)."""
    from pipeline.metadata import compute_confidence_metadata, export_metadata

    cfg = ctx.obj["config"]
    output_dir = Path("data/output")
    processed_dir = Path("data/processed")

    logger.info("=== Pipeline Metadata ===")
    meta = compute_confidence_metadata(cfg, str(output_dir), str(processed_dir))
    export_metadata(meta, str(output_dir / "pipeline_metadata.json"))
    click.echo(f"Metadata exported: {output_dir / 'pipeline_metadata.json'}")


# ---------------------------------------------------------------------------
# Landslide — CHT slope-based susceptibility
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def landslide(ctx):
    """Compute landslide susceptibility for CHT region."""
    from pipeline.landslide import run_landslide_pipeline

    cfg = ctx.obj["config"]
    logger.info("=== Landslide Susceptibility ===")
    result = run_landslide_pipeline(cfg)
    if result:
        click.echo(f"Landslide pipeline complete: {result}")
    else:
        click.echo("Landslide pipeline failed.", err=True)


# ---------------------------------------------------------------------------
# AlphaEarth — satellite embedding clusters
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def alphaearth(ctx):
    """Download AlphaEarth embeddings, cluster, and export GeoJSON."""
    from pipeline.alphaearth import run_alphaearth_pipeline

    cfg = ctx.obj["config"]
    logger.info("=== AlphaEarth Integration ===")
    result = run_alphaearth_pipeline(cfg)
    if result:
        click.echo(f"AlphaEarth clusters exported: {result}")
    else:
        click.echo("AlphaEarth pipeline failed (EE auth required).", err=True)


def _sample_raster_at_grid(raster_path: str, grid_gdf) -> np.ndarray:
    """Sample raster value at each grid cell centroid."""
    import rasterio

    centroids = [(g.centroid.x, g.centroid.y) for g in grid_gdf.geometry]
    if not Path(raster_path).exists():
        logger.warning(f"Raster not found: {raster_path}. Using zeros.")
        return np.zeros(len(grid_gdf))

    with rasterio.open(raster_path) as src:
        values = np.array([v[0] for v in src.sample(centroids)])
    values = np.nan_to_num(values, nan=0.0)
    return np.clip(values, 0, 1)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def run(ctx):
    """Run the full 10-step pipeline end-to-end."""
    logger.info("=" * 60)
    logger.info("SGMDI — Full Pipeline Execution")
    logger.info("=" * 60)

    ctx.invoke(download)
    ctx.invoke(ingest)
    ctx.invoke(preprocess)
    ctx.invoke(features)
    ctx.invoke(graph)
    ctx.invoke(train)
    ctx.invoke(krige)
    ctx.invoke(risk)
    ctx.invoke(metadata)
    ctx.invoke(landslide)

    click.echo("\nPipeline complete! Launch dashboard with:")
    click.echo("  streamlit run dashboard/app.py -- --config config.yaml")


if __name__ == "__main__":
    cli()
