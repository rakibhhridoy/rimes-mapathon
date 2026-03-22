"""
Step 4 — Spatial Graph Construction: k-NN graph with inverse-distance weights,
PyTorch Geometric Data object.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def build_spatial_graph(X: np.ndarray, coords: np.ndarray, y: np.ndarray,
                         cfg: dict) -> Data:
    """
    Build a k-NN spatial graph from infrastructure coordinates.

    Args:
        X: (N, D) standardized feature matrix
        coords: (N, 2) array of (lon, lat) or projected coordinates
        y: (N,) binary flood labels
        cfg: pipeline config dict

    Returns:
        PyTorch Geometric Data object
    """
    k = cfg["graph"]["k_neighbors"]
    max_dist = cfg["graph"]["max_edge_distance_m"]
    edge_weight_type = cfg["graph"]["edge_weight"]

    N = len(coords)
    logger.info(f"Building k-NN graph: {N} nodes, k={k}")

    # Build k-NN tree
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)  # +1 for self

    src_nodes = []
    dst_nodes = []
    edge_weights = []

    for i in range(N):
        for j_idx in range(1, k + 1):  # skip self at index 0
            j = idxs[i, j_idx]
            d = dists[i, j_idx]

            # Skip edges beyond max distance
            if d > max_dist:
                continue

            src_nodes.append(i)
            dst_nodes.append(j)

            if edge_weight_type == "inverse_distance":
                w = 1.0 / max(d, 1e-6)
            else:
                w = 1.0
            edge_weights.append(w)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    node_features = torch.tensor(X, dtype=torch.float)
    labels = torch.tensor(y, dtype=torch.float)

    # Train/val split masks
    train_ratio = cfg["graph"].get("train_split", 0.8)
    n = len(y)
    perm = torch.randperm(n)
    split = int(train_ratio * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:split]] = True
    val_mask = ~train_mask

    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
    )

    logger.info(
        f"Graph built: {graph_data.num_nodes} nodes, "
        f"{graph_data.num_edges} edges, "
        f"avg degree: {graph_data.num_edges / graph_data.num_nodes:.1f}"
    )

    return graph_data


def save_graph(graph_data: Data, output_path: str) -> None:
    """Save graph to disk."""
    torch.save(graph_data, output_path)
    logger.info(f"Graph saved → {output_path}")


def load_graph(path: str) -> Data:
    """Load graph from disk."""
    return torch.load(path, weights_only=False)
