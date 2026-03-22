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


# ---------------------------------------------------------------------------
# Step 6 — Inference: Embeddings + Risk Scores
# ---------------------------------------------------------------------------

def extract_embeddings_and_scores(model: FloodGNN,
                                   graph_data: Data) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract node embeddings and flood risk probabilities.

    Returns:
        embeddings: (N, 64) node embeddings
        risk_scores: (N,) flood risk probability [0, 1]
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
