"""
Step 5 — Train GraphSAGE model for flood exposure prediction.
Step 6 — Extract node embeddings and GNN risk scores.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class FloodGNN(torch.nn.Module):
    """
    2-layer GraphSAGE for flood exposure classification.
    Provides both classification (sigmoid) and embeddings.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (N, 1)."""
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.classifier(h)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return 64-dim node embeddings from penultimate layer."""
        h = F.relu(self.conv1(x, edge_index))
        return F.relu(self.conv2(h, edge_index))


# ---------------------------------------------------------------------------
# Step 5 — Training
# ---------------------------------------------------------------------------

def train_model(graph_data: Data, cfg: dict) -> FloodGNN:
    """Train the FloodGNN model with early stopping."""
    gnn_cfg = cfg["gnn"]

    # Set seed
    torch.manual_seed(gnn_cfg.get("seed", 42))
    np.random.seed(gnn_cfg.get("seed", 42))

    in_dim = graph_data.x.shape[1]
    hidden_dim = gnn_cfg["hidden_channels"]
    dropout = gnn_cfg["dropout"]
    lr = gnn_cfg["learning_rate"]
    epochs = gnn_cfg["epochs"]
    patience = gnn_cfg["patience"]

    model = FloodGNN(in_dim, hidden_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Handle class imbalance
    pos_count = graph_data.y[graph_data.train_mask].sum()
    neg_count = graph_data.train_mask.sum() - pos_count
    if pos_count > 0:
        pos_weight = torch.tensor([neg_count / pos_count])
    else:
        pos_weight = torch.tensor([1.0])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info(
        f"Training FloodGNN: {in_dim}→{hidden_dim}→1, "
        f"pos_weight={pos_weight.item():.2f}, epochs={epochs}"
    )

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(graph_data.x, graph_data.edge_index).squeeze()

        train_loss = criterion(
            logits[graph_data.train_mask],
            graph_data.y[graph_data.train_mask]
        )
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(graph_data.x, graph_data.edge_index).squeeze()
            val_loss = criterion(
                val_logits[graph_data.val_mask],
                graph_data.y[graph_data.val_mask]
            ).item()

            # AUC-ROC
            val_probs = torch.sigmoid(val_logits[graph_data.val_mask]).numpy()
            val_labels = graph_data.y[graph_data.val_mask].numpy()

        if epoch % 20 == 0:
            auc = _compute_auc(val_labels, val_probs)
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss.item():.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val AUC: {auc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return model


def _compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC-ROC. Falls back to 0.5 if single class."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return 0.5


def _test_mask(graph_data: Data):
    """Blocks used for reporting: test_mask when present, else val_mask."""
    return graph_data.test_mask if "test_mask" in graph_data else graph_data.val_mask


def fit_calibrator(model: FloodGNN, graph_data: Data):
    """Isotonic calibration of the sigmoid scores on the calibration blocks.

    The class-weighted loss inflates every score, so the raw sigmoid is a
    ranking, not a probability. Isotonic regression maps it onto the observed
    frequency of flooding on blocks the model did not train on, and is
    monotone, so rankings are unchanged.
    """
    from sklearn.isotonic import IsotonicRegression

    calib = graph_data.calib_mask if "calib_mask" in graph_data else None
    if calib is None or int(calib.sum()) == 0:
        return None
    model.eval()
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index).squeeze(-1)
    scores = torch.sigmoid(logits[calib]).numpy()
    labels = graph_data.y[calib].numpy()
    if len(np.unique(labels)) < 2:
        logger.warning("Calibration blocks hold a single class; no calibration.")
        return None
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(scores, labels)
    logger.info(f"Isotonic calibrator fitted on {int(calib.sum())} nodes")
    return calibrator


def evaluate_model(model: FloodGNN, graph_data: Data, calibrator=None) -> dict:
    """Metrics on the test blocks, raw and (if available) calibrated.

    The key `val_*` is kept for compatibility with earlier output files; it
    now refers to the test blocks.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss

    model.eval()
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index).squeeze(-1)
    mask = _test_mask(graph_data)
    probs = torch.sigmoid(logits[mask]).numpy()
    labels = graph_data.y[mask].numpy()

    metrics = {
        "n_train": int(graph_data.train_mask.sum()),
        "n_calibration": int(graph_data.calib_mask.sum()) if "calib_mask" in graph_data else 0,
        "n_val": int(mask.sum()),
        "val_positive_rate": float(labels.mean()),
        "val_auc_roc": _compute_auc(labels, probs),
        "val_brier": float(brier_score_loss(labels, probs)),
    }
    if len(np.unique(labels)) > 1:
        metrics["val_average_precision"] = float(average_precision_score(labels, probs))
    if calibrator is not None:
        calibrated = calibrator.predict(probs)
        metrics["val_brier_calibrated"] = float(brier_score_loss(labels, calibrated))
        metrics["val_brier_base_rate"] = float(
            brier_score_loss(labels, np.full_like(probs, labels.mean())))
    logger.info(f"Test blocks: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Step 6 — Inference: Embeddings + Risk Scores
# ---------------------------------------------------------------------------

def extract_embeddings_and_scores(model: FloodGNN,
                                   graph_data: Data) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract node embeddings and raw flood scores.

    Returns:
        embeddings: (N, 64) node embeddings
        risk_scores: (N,) sigmoid scores in [0, 1] — a ranking, not a
            calibrated probability (see fit_calibrator)
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.embed(graph_data.x, graph_data.edge_index).numpy()
        logits = model(graph_data.x, graph_data.edge_index).squeeze(-1).numpy()
        logits = np.atleast_1d(logits)  # ensure 1D even for single node
        risk_scores = 1 / (1 + np.exp(-logits))  # sigmoid

    logger.info(
        f"Embeddings: {embeddings.shape}, "
        f"Risk scores: mean={risk_scores.mean():.3f}, "
        f"std={risk_scores.std():.3f}"
    )
    return embeddings, risk_scores


def save_model(model: FloodGNN, path: str) -> None:
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved → {path}")


def load_model(path: str, in_dim: int, hidden_dim: int = 64,
                dropout: float = 0.3) -> FloodGNN:
    model = FloodGNN(in_dim, hidden_dim, dropout)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
