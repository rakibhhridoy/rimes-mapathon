"""
Pipeline metadata — compute real confidence metrics from pipeline outputs
and export to pipeline_metadata.json for the dashboard.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def mc_dropout_inference(model_path: str, graph_path: str,
                         in_dim: int, hidden_dim: int = 64,
                         dropout: float = 0.3,
                         n_forward: int = 30) -> dict:
    """
    Run MC-Dropout inference on the trained GNN model.

    Loads the model in training mode (dropout active), runs N forward passes,
    and computes per-node mean, std, and 90% CI width.

    Returns dict with 'mean', 'std', 'ci90_width' (scalar: mean across nodes).
    """
    import torch
    from pipeline.gnn_model import FloodGNN

    graph_data = torch.load(graph_path, weights_only=False)
    model = FloodGNN(in_dim, hidden_dim, dropout)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Enable dropout for MC inference
    model.train()

    all_preds = []
    with torch.no_grad():
        for i in range(n_forward):
            logits = model(graph_data.x, graph_data.edge_index).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
            all_preds.append(probs)

    all_preds = np.stack(all_preds, axis=0)  # (n_forward, N)
    node_mean = all_preds.mean(axis=0)
    node_std = all_preds.std(axis=0)
    # 90% CI width = 2 * 1.645 * std
    ci90_width = 2 * 1.645 * node_std

    result = {
        "mean_risk": float(node_mean.mean()),
        "std_risk": float(node_std.mean()),
        "ci90_width_mean": float(ci90_width.mean()),
        "ci90_width_median": float(np.median(ci90_width)),
        "n_forward_passes": n_forward,
    }
    logger.info(f"MC-Dropout: mean CI90 width = {result['ci90_width_mean']:.4f} "
                f"over {n_forward} passes")
    return result


def compute_confidence_metadata(cfg: dict, output_dir: str,
                                processed_dir: str) -> dict:
    """
    Compute all confidence metrics from existing pipeline outputs.

    Metrics:
    - kriging_variance_mean: mean of kriging variance surface (if available)
    - gnn_ci90_width: mean 90% CI width from MC-Dropout
    - ensemble_iqr: IQR of GNN risk scores
    - data_density_pct: % of grid cells with >= 1 infrastructure node
    - variogram_params: from variogram_params.json (if available)
    """
    output_dir = Path(output_dir)
    processed_dir = Path(processed_dir)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0.0",
    }

    # --- Kriging variance ---
    kriging_path = output_dir / "kriging_variance.tif"
    if kriging_path.exists():
        try:
            import rasterio
            with rasterio.open(kriging_path) as src:
                variance = src.read(1)
                valid = variance[variance != src.nodata] if src.nodata else variance.ravel()
                metadata["kriging_variance_mean"] = float(np.mean(valid))
                metadata["kriging_variance_median"] = float(np.median(valid))
                metadata["kriging_variance_max"] = float(np.max(valid))
            logger.info(f"Kriging variance: mean={metadata['kriging_variance_mean']:.4f}")
        except Exception as e:
            logger.warning(f"Could not read kriging variance: {e}")
            metadata["kriging_variance_mean"] = None
    else:
        logger.info("Kriging variance raster not found — will use GNN score variance as proxy")
        # Use variance of GNN scores as proxy
        scores_path = output_dir / "gnn_risk_scores.npy"
        if scores_path.exists():
            scores = np.load(scores_path)
            metadata["kriging_variance_mean"] = float(np.var(scores))
        else:
            metadata["kriging_variance_mean"] = None

    # --- GNN MC-Dropout CI ---
    model_path = output_dir / "gnn_model.pt"
    graph_path = output_dir / "spatial_graph.pt"
    if model_path.exists() and graph_path.exists():
        try:
            import torch
            graph_data = torch.load(graph_path, weights_only=False)
            in_dim = graph_data.x.shape[1]
            gnn_cfg = cfg.get("gnn", {})

            mc_result = mc_dropout_inference(
                str(model_path), str(graph_path),
                in_dim=in_dim,
                hidden_dim=gnn_cfg.get("hidden_channels", 64),
                dropout=gnn_cfg.get("dropout", 0.3),
                n_forward=30,
            )
            metadata["gnn_ci90_width"] = mc_result["ci90_width_mean"]
            metadata["gnn_mc_dropout"] = mc_result
        except Exception as e:
            logger.warning(f"MC-Dropout inference failed: {e}")
            metadata["gnn_ci90_width"] = None
    else:
        logger.warning("GNN model or graph not found — skipping MC-Dropout")
        metadata["gnn_ci90_width"] = None

    # --- Ensemble IQR of risk scores ---
    scores_path = output_dir / "gnn_risk_scores.npy"
    if scores_path.exists():
        scores = np.load(scores_path)
        q25, q75 = np.percentile(scores, [25, 75])
        metadata["ensemble_iqr"] = float(q75 - q25)
        metadata["risk_score_stats"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "q25": float(q25),
            "median": float(np.median(scores)),
            "q75": float(q75),
            "n_assets": int(len(scores)),
        }
        logger.info(f"Risk scores: IQR={metadata['ensemble_iqr']:.4f}, "
                     f"n={len(scores)}")
    else:
        metadata["ensemble_iqr"] = None
        metadata["risk_score_stats"] = None

    # --- Data density ---
    features_path = processed_dir / "node_features.parquet"
    if features_path.exists():
        try:
            import pandas as pd
            nf = pd.read_parquet(features_path)
            n_nodes = len(nf)
            # Estimate grid coverage: count unique grid cells occupied
            # Use 500m grid resolution from config
            grid_res = cfg.get("aoi", {}).get("grid_resolution_m", 500)
            bbox = cfg.get("aoi", {}).get("bbox", [88.0, 24.0, 89.9, 26.7])
            # Approximate degrees per grid cell
            deg_per_cell = grid_res / 111000  # rough conversion
            n_lon_cells = int((bbox[2] - bbox[0]) / deg_per_cell)
            n_lat_cells = int((bbox[3] - bbox[1]) / deg_per_cell)
            total_cells = n_lon_cells * n_lat_cells

            if "longitude" in nf.columns and "latitude" in nf.columns:
                lon_idx = ((nf["longitude"] - bbox[0]) / deg_per_cell).astype(int)
                lat_idx = ((nf["latitude"] - bbox[1]) / deg_per_cell).astype(int)
                occupied = len(set(zip(lon_idx.tolist(), lat_idx.tolist())))
                density = (occupied / total_cells * 100) if total_cells > 0 else 0
            else:
                # Fallback: ratio of nodes to grid cells
                density = (n_nodes / total_cells * 100) if total_cells > 0 else 0

            metadata["data_density_pct"] = round(min(density, 100.0), 1)
            metadata["n_infrastructure_nodes"] = n_nodes
            metadata["grid_cells_total"] = total_cells
            logger.info(f"Data density: {metadata['data_density_pct']}% "
                         f"({n_nodes} nodes, {total_cells} cells)")
        except Exception as e:
            logger.warning(f"Could not compute data density: {e}")
            metadata["data_density_pct"] = None
    else:
        metadata["data_density_pct"] = None

    # --- Variogram params ---
    vario_path = output_dir / "variogram_params.json"
    if vario_path.exists():
        try:
            with open(vario_path) as f:
                metadata["variogram_params"] = json.load(f)
            logger.info("Loaded variogram parameters")
        except Exception as e:
            logger.warning(f"Could not load variogram params: {e}")
            metadata["variogram_params"] = None
    else:
        metadata["variogram_params"] = None

    return metadata


def export_metadata(metadata: dict, output_path: str) -> None:
    """Write metadata dict to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Pipeline metadata exported → {output_path}")
