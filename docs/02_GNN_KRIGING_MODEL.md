# 02 — Hybrid GNN + Kriging Model

## Architecture Overview

```
                    ┌──────────────────┐
                    │  Infrastructure   │
                    │  Point Cloud      │
                    │  (N nodes)        │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  k-NN Spatial     │
                    │  Graph (k=6)      │
                    │  Edge weights:    │
                    │  1/distance       │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼─────────┐        ┌─────────▼────────┐
     │  Ordinary Kriging │        │  GraphSAGE GNN   │
     │  (Spatial Prior)  │        │  (Residual Corr.) │
     │                   │        │                   │
     │  • Empirical      │        │  • 2-layer SAGE   │
     │    semivariogram  │        │  • 64 hidden dim  │
     │  • Spherical fit  │        │  • Dropout 0.3    │
     │  • Grid predict   │        │  • BCE loss       │
     └────────┬──────────┘        └─────────┬────────┘
              │                             │
              │    z_kriging                │  z_gnn_residual
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Final Risk =     │
                    │  z_kriging +      │
                    │  z_gnn_residual   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Risk = Hazard ×  │
                    │  Exposure ×       │
                    │  Vulnerability    │
                    └──────────────────┘
```

## Step-by-Step Model Pipeline

### Step 4 — Graph Construction

- **Nodes**: Infrastructure asset centroids (buildings, bridges, roads, cropland)
- **Edges**: k-Nearest Neighbors with k=6 using `scipy.spatial.cKDTree`
- **Edge weights**: Inverse Euclidean distance (capped at `max_edge_distance_m=5000`)
- **Node features** (per asset):

| Feature | Source | Description |
|---------|--------|-------------|
| `flood_proxy` | Ensemble label | Binary/continuous flood exposure from majority voting |
| `elevation` | SRTM DEM | Meters above sea level |
| `slope` | DEM-derived | Degrees |
| `twi` | DEM-derived | Topographic Wetness Index |
| `hand` | DEM-derived | Height Above Nearest Drainage (m) |
| `ndvi` | Sentinel-2 | Vegetation index |
| `pop_density` | WorldPop | People per km² |
| `dist_hospital` | OSM-computed | Distance to nearest hospital (m) |
| `dist_school` | OSM-computed | Distance to nearest school (m) |
| `asset_type` | OSM category | One-hot or ordinal encoded |

### Step 5 — GraphSAGE Training

```python
class FloodGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.classifier(h)

    def embed(self, x, edge_index):
        """Return 64-dim node embeddings for kriging input."""
        h = F.relu(self.conv1(x, edge_index))
        return F.relu(self.conv2(h, edge_index))
```

**Training config**:
- Optimizer: Adam, lr=1e-3
- Loss: BCEWithLogitsLoss (handles class imbalance with pos_weight)
- Split: 80/20 stratified random
- Early stopping: patience=20 on validation loss
- Epochs: up to 200

**Target variable**: Ensemble flood proxy label (binary: exposed / not exposed)

### Step 6 — Node Embeddings + GNN Risk Scores

After training:
1. Extract 64-dim embeddings from `model.embed()`
2. Compute flood risk probability via sigmoid on classifier logits
3. Compute **kriging residual** = observed_label − gnn_prediction
4. These residuals become the target for the kriging step

### Step 7 — Ordinary Kriging

**Why kriging after GNN?**
The GNN captures local graph-structure patterns but can't extrapolate smoothly to unobserved grid cells. Kriging fills the spatial gaps with uncertainty estimates.

**Procedure**:
1. Compute empirical semivariogram from GNN risk scores at node locations
2. Fit spherical variogram model (nugget, sill, range)
3. Execute Ordinary Kriging on a regular grid covering the study area
4. Output: continuous flood risk surface + kriging variance (uncertainty)

```python
ok = OrdinaryKriging(
    lons, lats, gnn_risk_scores,
    variogram_model="spherical",
    nlags=15,
    weight=True
)
z_risk, ss_risk = ok.execute("grid", grid_lon, grid_lat)
# z_risk  → interpolated flood risk surface
# ss_risk → kriging variance (confidence map)
```

**Hybrid fusion**:
```
final_risk(x) = kriging_prediction(x) + gnn_residual_correction(x)
```

At node locations, the GNN prediction dominates. Between nodes, kriging smoothly interpolates with quantified uncertainty.

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| AUC-ROC | Binary classification of flood-exposed vs. safe nodes |
| Brier Score | Calibration of predicted probabilities |
| Semivariogram RMSE | Quality of variogram model fit |
| Cross-validated MAE | Leave-one-out kriging prediction accuracy |
| Spatial autocorrelation (Moran's I) | Check residuals are spatially random (good model) |

## Outputs

| File | Format | Description |
|------|--------|-------------|
| `gnn_model.pt` | PyTorch checkpoint | Trained GraphSAGE weights |
| `node_embeddings.npy` | NumPy array | 64-dim embeddings per node |
| `flood_risk_kriged.tif` | GeoTIFF | Interpolated flood risk surface |
| `kriging_variance.tif` | GeoTIFF | Prediction uncertainty surface |
| `variogram_params.json` | JSON | Fitted variogram parameters |
