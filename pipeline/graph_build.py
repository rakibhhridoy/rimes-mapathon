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
        coords: (N, 2) array of (lon, lat); projected to cfg["aoi"]["crs"]
            so that max_edge_distance_m and edge weights are in metres
        y: (N,) binary flood labels
        cfg: pipeline config dict

    Returns:
        PyTorch Geometric Data object
    """
    from pipeline.feature_extract import project_coords

    k = cfg["graph"]["k_neighbors"]
    max_dist = cfg["graph"]["max_edge_distance_m"]
    edge_weight_type = cfg["graph"]["edge_weight"]

    coords = project_coords(np.asarray(coords), cfg["aoi"]["crs"])
    N = len(coords)
    logger.info(f"Building k-NN graph: {N} nodes, k={k}")

    # Build k-NN tree
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)  # +1 for self

    # Vectorised edge construction: drop the self-match column, then keep
    # neighbour pairs within max_dist. Building tensors from Python lists of
    # numpy scalars is slow and crashes some torch builds.
    neigh_idx = idxs[:, 1:k + 1]
    neigh_dist = dists[:, 1:k + 1]
    src = np.repeat(np.arange(N), neigh_idx.shape[1])
    dst = neigh_idx.ravel()
    dist = neigh_dist.ravel()

    keep = np.isfinite(dist) & (dist <= max_dist)
    src, dst, dist = src[keep], dst[keep], dist[keep]
    dropped = int((~keep).sum())
    if dropped:
        logger.info(f"Dropped {dropped} neighbour pairs beyond {max_dist} m")

    if edge_weight_type == "inverse_distance":
        weights = 1.0 / np.maximum(dist, 1e-6)
    else:
        weights = np.ones_like(dist)

    edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))
    edge_attr = torch.from_numpy(weights.astype(np.float32)).unsqueeze(1)
    node_features = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    labels = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    train_mask, calib_mask, test_mask = spatial_block_split_three(
        coords,
        train_ratio=cfg["graph"].get("train_split", 0.8),
        block_size_m=cfg["graph"].get("block_size_m", 10_000),
        seed=cfg.get("gnn", {}).get("seed", 42),
    )
    # val_mask keeps its old meaning for early stopping: the calibration
    # blocks. Reported metrics come from test_mask, which neither training
    # nor calibration has seen.
    val_mask = calib_mask

    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        calib_mask=calib_mask,
        test_mask=test_mask,
    )

    logger.info(
        f"Graph built: {graph_data.num_nodes} nodes, "
        f"{graph_data.num_edges} edges, "
        f"avg degree: {graph_data.num_edges / graph_data.num_nodes:.1f}"
    )

    return graph_data


def spatial_block_split(coords_m: np.ndarray, train_ratio: float = 0.8,
                        block_size_m: float = 10_000, seed: int = 42
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign whole square blocks of nodes to train or validation.

    A random node split leaks information through spatial autocorrelation:
    a validation node's neighbours sit in the training set, inflating AUC.
    Holding out entire blocks keeps validation nodes spatially separate.
    """
    block_ids = np.floor(coords_m / block_size_m).astype(np.int64)
    _, block_of_node = np.unique(block_ids, axis=0, return_inverse=True)
    block_of_node = block_of_node.ravel()
    n_blocks = block_of_node.max() + 1

    rng = np.random.default_rng(seed)
    val_blocks = rng.random(n_blocks) >= train_ratio
    val = torch.tensor(np.ascontiguousarray(val_blocks[block_of_node], dtype=bool))

    logger.info(
        f"Spatial block split: {n_blocks} blocks of {block_size_m / 1000:.0f} km, "
        f"{int((~val).sum())} train / {int(val.sum())} val nodes"
    )
    return ~val, val


def spatial_block_split_three(coords_m: np.ndarray, train_ratio: float = 0.8,
                              block_size_m: float = 10_000, seed: int = 42
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train / calibration / test masks by whole spatial block.

    The held-out share is halved between calibration and test so that the
    probabilities can be calibrated on blocks the model never trained on,
    while the reported scores come from blocks that neither training nor
    calibration touched.
    """
    block_ids = np.floor(coords_m / block_size_m).astype(np.int64)
    _, block_of_node = np.unique(block_ids, axis=0, return_inverse=True)
    block_of_node = block_of_node.ravel()
    n_blocks = block_of_node.max() + 1

    rng = np.random.default_rng(seed)
    draw = rng.random(n_blocks)
    held_out = draw >= train_ratio
    # Split the held-out blocks in half by a second draw, so the choice of
    # which held-out block calibrates and which tests is itself random.
    second = rng.random(n_blocks) < 0.5
    calib_blocks = held_out & second
    test_blocks = held_out & ~second

    # torch.tensor copies; from_numpy over a temporary boolean array crashed
    # this torch build on the first reduction.
    train = torch.tensor(np.ascontiguousarray(~held_out[block_of_node], dtype=bool))
    calib = torch.tensor(np.ascontiguousarray(calib_blocks[block_of_node], dtype=bool))
    test = torch.tensor(np.ascontiguousarray(test_blocks[block_of_node], dtype=bool))

    logger.info(
        f"Spatial block split: {n_blocks} blocks of {block_size_m / 1000:.0f} km, "
        f"{int(train.sum())} train / {int(calib.sum())} calibration / "
        f"{int(test.sum())} test nodes"
    )
    return train, calib, test


def save_graph(graph_data: Data, output_path: str) -> None:
    """Save graph to disk."""
    torch.save(graph_data, output_path)
    logger.info(f"Graph saved → {output_path}")


def load_graph(path: str) -> Data:
    """Load graph from disk."""
    return torch.load(path, weights_only=False)
