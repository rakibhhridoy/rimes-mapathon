"""
CLI entry point for the SGMDI pipeline.
Provides step-by-step commands and a full `run` command.
"""

import logging
import os
import sys
from pathlib import Path

# Two OpenMP runtimes are loaded in this environment (PyTorch carries its own
# alongside the one numba and scikit-learn use), and PyTorch's threaded
# reductions crash once a tensor exceeds its 32,768-element grain size. One
# thread costs little here — the graphs have tens of thousands of nodes — and
# the setting must precede the first import of torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import click
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgmdi")


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_config(config_path: str) -> dict:
    """Load a config, following an `extends:` chain.

    A region config states only what differs from the base, so the shared
    model settings stay in one place:

        extends: ../config.yaml
        aoi: {name: sylhet, bbox: [...]}
    """
    path = Path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        cfg = _merge(_load_config(str(parent_path)), cfg)
    return cfg


def _dir(cfg: dict, kind: str) -> Path:
    """Directory for raw / processed / output data.

    Defaults keep the single-region layout (data/raw, data/processed,
    data/output); a region config overrides them under `paths:` so several
    study areas can live side by side.
    """
    default = f"data/{kind}"
    path = Path((cfg.get("paths", {}) or {}).get(f"{kind}_dir", default))
    return path


def _infra_path(cfg: dict) -> Path:
    return _dir(cfg, "raw") / "infrastructure_raw.gpkg"


def _susceptibility_path(output_dir: Path) -> Path:
    """Scores without the flood record, for anything scored against it."""
    path = output_dir / "susceptibility_scores.npy"
    return path if path.exists() else output_dir / "gnn_risk_scores.npy"


def cfg_has_landslide(cfg: dict) -> bool:
    """True when the region config asks for the landslide model."""
    return bool((cfg.get("landslide") or {}).get("enabled", False))


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
    raw_dir = _dir(cfg, "raw")
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
    raw_dir = _dir(cfg, "raw")
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
    raw_dir = _dir(cfg, "raw")
    processed_dir = _dir(cfg, "processed")

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
    processed_dir = _dir(cfg, "processed")
    raw_dir = _dir(cfg, "raw")

    infra_path = _infra_path(cfg)
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
    processed_dir = _dir(cfg, "processed")
    output_dir = _dir(cfg, "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    infra = gpd.read_file(str(_infra_path(cfg)))
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
    """Step 5-6: Fit the asset model (see asset_model.type) and score assets."""
    import json

    import numpy as np

    from pipeline.asset_model import fit_asset_model, model_type
    from pipeline.graph_build import load_graph

    cfg = ctx.obj["config"]
    output_dir = _dir(cfg, "output")

    graph_path = output_dir / "spatial_graph.pt"
    if not graph_path.exists():
        click.echo("Error: Run 'graph' first.", err=True)
        sys.exit(1)

    graph_data = load_graph(str(graph_path))
    kind = model_type(cfg)

    logger.info(f"=== Step 5: Training the asset model ({kind}) ===")
    risk_scores, probability, metrics, fitted = fit_asset_model(graph_data, cfg)
    with open(output_dir / "gnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if kind == "graph_sage":
        from pipeline.gnn_model import extract_embeddings_and_scores, save_model

        save_model(fitted, str(output_dir / "gnn_model.pt"))
        embeddings, _ = extract_embeddings_and_scores(fitted, graph_data)
        np.save(str(output_dir / "node_embeddings.npy"), embeddings)
    else:
        import joblib

        joblib.dump(fitted, str(output_dir / "asset_model.joblib"))

    # Validation and Kriging read these, since they carry no flood record and
    # so can be scored against the observed extents without circularity.
    np.save(str(output_dir / "susceptibility_scores.npy"), risk_scores)
    if probability is not None:
        np.save(str(output_dir / "susceptibility_probability.npy"), probability)

    # With asset_model.past_flooding the map shows a model that also knows
    # where earlier floods reached (see past_flooding_inputs). Its own metrics
    # come from the held-out blocks of a setup that predicts the latest event
    # year from the ones before it.
    labels_cfg = (cfg.get("data", {}).get("labels", {}) or {})
    if ((cfg.get("asset_model") or {}).get("past_flooding")
            and labels_cfg.get("source") == "observed"):
        import joblib

        from pipeline.asset_model import fit_past_flooding_model, past_flooding_inputs

        history, full, target, info = past_flooding_inputs(
            cfg, _dir(cfg, "raw"), _infra_path(cfg))
        risk_scores, probability, past_metrics, past_model = fit_past_flooding_model(
            graph_data, cfg, history, full, target)
        past_metrics.update(info)
        with open(output_dir / "past_model_metrics.json", "w") as f:
            json.dump(past_metrics, f, indent=2)
        joblib.dump(past_model, str(output_dir / "past_model.joblib"))
    else:
        (output_dir / "past_model_metrics.json").unlink(missing_ok=True)

    # The file names are historical: the map, rankings and popups read them,
    # and they hold the scores of whichever model the map shows.
    np.save(str(output_dir / "gnn_risk_scores.npy"), risk_scores)
    if probability is not None:
        np.save(str(output_dir / "gnn_flood_probability.npy"), probability)
        logger.info(f"Calibrated flood probability: mean={probability.mean():.3f}")

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
    output_dir = _dir(cfg, "output")

    risk_scores = np.load(str(_susceptibility_path(output_dir)))
    infra = gpd.read_file(str(_infra_path(cfg)))
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
    import json
    import geopandas as gpd
    import numpy as np
    from pipeline.risk_score import (
        create_risk_grid, compute_exposure_grid, compute_vulnerability_grid,
        compute_population_exposure_grid, compute_composite_risk,
        aggregate_to_admin, detect_hotspots, rank_assets, assign_risk_classes,
    )
    from pipeline.export import (
        export_ranked_csv, export_geojson,
        export_hotspots, generate_pdf_report,
    )
    from pipeline.feature_extract import compute_centroids

    cfg = ctx.obj["config"]
    output_dir = _dir(cfg, "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    infra = gpd.read_file(str(_infra_path(cfg)))
    infra = compute_centroids(infra)
    risk_scores = np.load(str(output_dir / "gnn_risk_scores.npy"))

    bounds = tuple(cfg["aoi"]["bbox"])
    grid_res = cfg["kriging"].get("grid_resolution_deg", 0.005)

    logger.info("=== Step 8: Composite Risk ===")
    processed_dir = _dir(cfg, "processed")
    grid_gdf = create_risk_grid(bounds, grid_res)

    # Hazard per grid cell. Two surfaces are available: the kriged GNN
    # scores, and a terrain model evaluated at each cell. Which one the
    # composite uses is a config choice, and both are kept on the grid so the
    # difference can be inspected.
    from pipeline.hazard_model import (
        fit_hazard_model, predict_hazard_at, write_hazard_raster,
    )

    hazard_kriged = _sample_raster_at_grid(
        str(output_dir / "flood_risk_kriged.tif"), grid_gdf
    )
    grid_gdf["hazard_kriged"] = hazard_kriged

    hazard_source = cfg["risk"].get("hazard_source", "kriged")
    hazard_terrain = None
    try:
        fitted = fit_hazard_model(cfg, processed_dir, output_dir, _infra_path(cfg))
        b = grid_gdf.geometry.bounds
        centres = list(zip((b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2))
        hazard_terrain = predict_hazard_at(fitted, processed_dir, centres)
        grid_gdf["hazard_terrain"] = hazard_terrain
        write_hazard_raster(fitted, processed_dir,
                            output_dir / "flood_hazard_terrain.tif")
    except Exception as exc:
        logger.warning(f"Terrain hazard model unavailable: {exc}")
        if hazard_source == "terrain_model":
            logger.warning("Falling back to the kriged hazard surface.")
            hazard_source = "kriged"

    hazard = hazard_terrain if hazard_source == "terrain_model" else hazard_kriged
    logger.info(f"Composite risk uses the '{hazard_source}' hazard surface "
                f"(mean {np.mean(hazard):.3f}, std {np.std(hazard):.3f})")
    exposure = compute_exposure_grid(infra, grid_gdf, cfg)
    vulnerability, vuln_weights = compute_vulnerability_grid(grid_gdf, infra, cfg)
    with open(output_dir / "vulnerability_weights.json", "w") as f:
        json.dump(vuln_weights, f, indent=2)

    grid_gdf["hazard"] = hazard
    grid_gdf["exposure"] = exposure
    grid_gdf["vulnerability"] = vulnerability
    grid_gdf["composite_risk"] = compute_composite_risk(hazard, exposure, vulnerability)
    grid_gdf["risk_class"] = assign_risk_classes(grid_gdf["composite_risk"].values)

    # Second reading of the same hazard: who is exposed, rather than what.
    population_exposure = compute_population_exposure_grid(grid_gdf, cfg)
    grid_gdf["population_exposure"] = population_exposure
    grid_gdf["composite_risk_people"] = compute_composite_risk(
        hazard, population_exposure, vulnerability
    )
    grid_gdf["risk_class_people"] = assign_risk_classes(
        grid_gdf["composite_risk_people"].values
    )

    logger.info("=== Step 9: Aggregation & Ranking ===")

    # Hotspots
    grid_gdf = detect_hotspots(
        grid_gdf, cfg["risk"].get("hotspot_confidence", 0.95)
    )

    # Rank individual assets
    threshold = cfg["risk"].get("high_risk_threshold", 0.7)
    ranked_infra = rank_assets(infra, risk_scores, threshold, grid_gdf=grid_gdf)
    prob_path = output_dir / "gnn_flood_probability.npy"
    if prob_path.exists():
        probability = np.load(prob_path)
        if len(probability) == len(ranked_infra):
            # rank_assets sorted the frame; align by original index.
            ranked_infra["flood_probability"] = probability[ranked_infra.index.values]

    # Aggregate to each administrative level that has boundaries on disk.
    # geoBoundaries ADM4/ADM3/ADM2 = union / upazila / district.
    admin_paths = cfg["data"].get("admin_boundaries", {})
    summaries = {}
    for level in ("union", "upazila", "district"):
        path = admin_paths.get(level, "")
        if not path or not Path(path).exists():
            logger.warning(f"No {level} boundaries at {path} — skipping.")
            continue
        admin_gdf = gpd.read_file(path)
        summary = aggregate_to_admin(grid_gdf, admin_gdf, infra,
                                     high_risk_threshold=threshold)
        summaries[level] = summary

        export_geojson(summary, str(output_dir / f"{level}_risk_summary.geojson"))
        cols = [c for c in summary.columns if c != "geometry"]
        summary[cols].to_csv(
            str(output_dir / f"{level}_risk_summary.csv"), index=False
        )
        logger.info(f"{level.title()} risk summary exported ({len(summary)} units)")

    # Export everything else
    export_ranked_csv(ranked_infra, str(output_dir / "risk_ranked_assets.csv"))
    export_geojson(ranked_infra, str(output_dir / "risk_ranked_assets.geojson"))
    export_geojson(ranked_infra, str(output_dir / "top50_risk_assets.geojson"),
                    max_features=50)
    export_hotspots(grid_gdf, str(output_dir / "hotspot_clusters.geojson"))
    export_geojson(grid_gdf, str(output_dir / "risk_grid.geojson"))

    generate_pdf_report(
        summaries.get("upazila", summaries.get("union")), ranked_infra,
        str(output_dir / "situation_report.pdf")
    )

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
    output_dir = _dir(cfg, "output")
    processed_dir = _dir(cfg, "processed")

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
    result = run_landslide_pipeline(
        cfg, _dir(cfg, "raw"), _dir(cfg, "processed"), _dir(cfg, "output")
    )
    click.echo(f"Landslide pipeline complete: {result}")


# ---------------------------------------------------------------------------
# Sentinel-1 — observed flood extents
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def sentinel1(ctx):
    """Map observed flood extents from Sentinel-1 SAR (needs Earth Engine)."""
    from pipeline.sentinel1 import run_sentinel1_pipeline

    cfg = ctx.obj["config"]
    logger.info("=== Sentinel-1 Observed Flood Extents ===")
    outputs = run_sentinel1_pipeline(cfg, _dir(cfg, "raw"))
    click.echo(f"Mapped {len(outputs) - 1} events → data/raw/")
    click.echo("Set data.labels.source: observed in config.yaml, then rerun "
               "`preprocess`, `features`, `graph`, `train`, `krige`, `risk`.")


# ---------------------------------------------------------------------------
# Validation — model vs observed floods
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def validate(ctx):
    """Validate model scores against Sentinel-1 observed flood extents."""
    from pipeline.validate import run_validation

    cfg = ctx.obj["config"]
    logger.info("=== Validation Against Observed Floods ===")
    results = run_validation(cfg, _dir(cfg, "output"),
                             _dir(cfg, "processed"), _dir(cfg, "raw"))
    assets = results["assets_vs_observed"]
    held = assets.get("held_out_blocks") or assets["all_assets"]
    click.echo(f"Observed flooded share: {assets['observed_flooded_share']:.2%}")
    if "auc_roc" in held:
        click.echo(f"AUC: {held['auc_roc']:.3f} | AP: {held['average_precision']:.3f}")
    click.echo("Full metrics: data/output/validation_metrics.json")


@cli.command()
@click.pass_context
def crosscheck(ctx):
    """Check Sentinel-1 flood masks against the Global Flood Database."""
    from pipeline.crosscheck import run_crosscheck

    cfg = ctx.obj["config"]
    logger.info("=== Sentinel-1 masks against the Global Flood Database ===")
    for name, r in run_crosscheck(cfg, _dir(cfg, "raw"), _dir(cfg, "output")).items():
        click.echo(f"{name}: POD {r['pod']:.3f} | FAR {r['far']:.3f} | "
                   f"CSI {r['csi']:.3f} | {r['n_cells']:,} cells")


# ---------------------------------------------------------------------------
# Benchmark — the graph model against ordinary tabular models
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--seeds", default=None,
              help="Comma-separated block-split seeds (default: twenty seeds).")
@click.pass_context
def benchmark(ctx, seeds):
    """Compare the graph model with tabular baselines over repeated splits."""
    from pipeline.benchmark import (SEEDS, run_benchmark, run_label_comparison,
                                    run_past_flooding_feature, run_temporal_holdout)

    cfg = ctx.obj["config"]
    chosen = tuple(int(s) for s in seeds.split(",")) if seeds else SEEDS
    logger.info("=== Benchmark: graph model against tabular baselines ===")
    results = run_benchmark(cfg, _dir(cfg, "processed"), _dir(cfg, "output"),
                            _infra_path(cfg), seeds=chosen)
    labels = run_label_comparison(cfg, _dir(cfg, "processed"), _dir(cfg, "output"),
                                  _infra_path(cfg), seeds=chosen)
    if (cfg.get("data", {}).get("labels", {}) or {}).get("source") == "observed":
        temporal = run_temporal_holdout(cfg, _dir(cfg, "processed"), _dir(cfg, "raw"),
                                        _dir(cfg, "output"), _infra_path(cfg),
                                        seeds=chosen)
        past = run_past_flooding_feature(cfg, _dir(cfg, "processed"), _dir(cfg, "raw"),
                                         _dir(cfg, "output"), _infra_path(cfg),
                                         seeds=chosen)
        w = past["summary"].get("with_past") or {}
        if w:
            click.echo(f"with past flooding as a feature: AUC {w['auc_mean']:.3f}")
        t = temporal["summary"].get("temporal") or {}
        if t:
            click.echo(f"trained to {temporal['cutoff_year']}, scored on "
                       f"{', '.join(temporal['test_events'])}: AUC {t['auc_mean']:.3f}")
    gap = labels["summary"].get("observed_minus_proxy") or {}
    if gap:
        click.echo(f"observed minus proxy labels: {gap['mean']:+.3f} AUC "
                   f"(ahead in {gap['seeds_observed_ahead']} of "
                   f"{gap['n_seeds']} splits)")
    for name, stats in results["summary"].items():
        if name == "graph_vs_best_baseline":
            click.echo(f"graph minus {stats['best_baseline']}: "
                       f"{stats['auc_difference_mean']:+.3f} AUC "
                       f"(ahead in {stats['seeds_graph_ahead']} of "
                       f"{stats['n_seeds']} splits)")
        else:
            sd = stats["auc_sd"]
            spread = f" ± {sd:.3f}" if sd is not None else ""
            click.echo(f"{name:22s} AUC {stats['auc_mean']:.3f}{spread}")
    click.echo("Full metrics: <output>/benchmark.json")


# ---------------------------------------------------------------------------
# Sensitivity — how much the composite depends on its chosen weights
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--draws", default=20, show_default=True,
              help="Random weight perturbations to draw.")
@click.pass_context
def sensitivity(ctx, draws):
    """Recompute composite risk under other weights and report rank movement."""
    from pipeline.sensitivity import run_sensitivity

    cfg = ctx.obj["config"]
    logger.info("=== Weight sensitivity of the composite risk ===")
    results = run_sensitivity(cfg, _dir(cfg, "output"), _infra_path(cfg),
                              n_draws=draws)
    for name, stats in results["scenarios"].items():
        click.echo(f"{name:34s} rho {stats['spearman']:.3f}  "
                   f"top decile kept {stats['top_decile_overlap']:.2f}")
    random_draws = results["random_perturbation"]
    click.echo(f"random weights ({draws} draws): rho "
               f"{random_draws['spearman_mean']:.3f} "
               f"(min {random_draws['spearman_min']:.3f})")
    click.echo("Full metrics: <output>/sensitivity.json")


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

    b = grid_gdf.geometry.bounds
    centroids = list(zip((b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2))
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

    # Validation needs Sentinel-1 extents; skip cleanly when absent.
    if (_dir(ctx.obj["config"], "raw") / "s1_flood_frequency.tif").exists():
        ctx.invoke(validate)
    else:
        logger.warning(
            "No Sentinel-1 flood extents — skipping validation. "
            "Run `python -m pipeline.cli sentinel1` to enable it."
        )

    if cfg_has_landslide(ctx.obj["config"]):
        ctx.invoke(landslide)
    else:
        logger.info("Region has no landslide section — skipping that model.")

    click.echo("\nPipeline complete! Launch dashboard with:")
    click.echo("  streamlit run dashboard/app.py -- --config config.yaml")


if __name__ == "__main__":
    cli()
