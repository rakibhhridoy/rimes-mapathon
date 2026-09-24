"""
Build the article figures from pipeline outputs.

Every figure reads the files the pipeline wrote; nothing is typed in. Run
after the pipeline and before compiling the document:

    python docs/figures/make_figures.py

Palette: categorical slots 1-2 of the reference palette (blue #2a78d6,
orange #eb6834, validated for CVD and contrast) and its single-hue blue ramp
for magnitudes. The proxy-trained series is also dashed, so the pair reads in
greyscale print.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.cli import _load_config  # noqa: E402
from pipeline.feature_extract import compute_centroids, sample_raster_at_points  # noqa: E402

OUT = Path(__file__).resolve().parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

COL_W = 3.42     # single column, inches (86.8 mm)
FULL_W = 6.93    # both columns, inches (176 mm)

REGIONS = {
    "rangpur_rajshahi": ("Rangpur & Rajshahi", "config.yaml"),
    "sylhet": ("Sylhet", "configs/sylhet.yaml"),
    "sw_coastal": ("South-west coast", "configs/sw_coastal.yaml"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "grid.color": GRID,
    "grid.linewidth": 0.5, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def _cfg(region):
    return _load_config(str(ROOT / REGIONS[region][1]))


def _paths(cfg):
    p = cfg["paths"]
    return ROOT / p["raw_dir"], ROOT / p["processed_dir"], ROOT / p["output_dir"]


def _districts():
    path = ROOT / "data/shared/geoboundaries/BGD_ADM2.geojson"
    return gpd.read_file(path).simplify(0.005)


def _read_wgs84(path, max_dim=900):
    """Read a raster reprojected to WGS84 at modest resolution for plotting."""
    with rasterio.open(path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        scale = max(w / max_dim, h / max_dim, 1)
        w, h = int(w / scale), int(h / scale)
        transform, w, h = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds,
            dst_width=w, dst_height=h)
        data = np.full((h, w), np.nan, dtype=np.float32)
        reproject(rasterio.band(src, 1), data, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform,
                  dst_crs="EPSG:4326", dst_nodata=np.nan,
                  resampling=Resampling.average)
    west, north = transform * (0, 0)
    east, south = transform * (w, h)
    return data, (west, east, south, north)


# ── Figure 1: study area ────────────────────────────────────────────────────
def fig_study_area():
    """Bangladesh with each study region tinted inside the country: blue
    steps for the three flood regions, orange for the landslide region,
    neighbouring countries pale and the sea pale blue."""
    districts = _districts()
    countries = gpd.read_file(ROOT / "data/shared/naturalearth/ne_10m_admin_0_countries.geojson")
    from shapely.geometry import box as _box
    # Only the frame is drawn, so clip and simplify to keep the PDF small.
    neighbours = gpd.clip(countries[countries["ADM0_A3"] != "BGD"],
                          _box(87.5, 20.0, 93.5, 27.2)).simplify(0.005)
    country = gpd.GeoSeries([districts.buffer(0).union_all()], crs=districts.crs)

    SEA, LAND, NEIGHBOUR = "#EEF4FA", "#ECEBE7", "#F8F7F4"
    regions = {
        "rangpur_rajshahi": ("Rangpur &\nRajshahi", "flood", "#C3D9EF"),
        "sylhet": ("Sylhet", "flash flood", "#86B0DD"),
        "sw_coastal": ("South-west\ncoast", "surge", "#A6CEE3"),
        "cht": ("Chittagong\nHill Tracts", "landslide", "#F4A582"),
    }
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 1.22))
    ax.set_facecolor(SEA)
    neighbours.plot(ax=ax, facecolor=NEIGHBOUR, edgecolor="#c9c7c0", linewidth=0.4, zorder=1)
    districts.plot(ax=ax, facecolor=LAND, edgecolor="none", zorder=2)
    from shapely.geometry import box
    for rid, (name, hazard, colour) in regions.items():
        path = "configs/cht.yaml" if rid == "cht" else REGIONS[rid][1]
        w, s_, e, n = _load_config(str(ROOT / path))["aoi"]["bbox"]
        tint = gpd.clip(country, box(w, s_, e, n))
        tint.plot(ax=ax, facecolor=colour, edgecolor="none", zorder=3)
        ax.add_patch(Rectangle((w, s_), e - w, n - s_, fill=False,
                               edgecolor=INK, linewidth=0.9, zorder=5))
        ax.text(w + 0.07, n - 0.07, f"{name}\n({hazard})", ha="left", va="top",
                fontsize=6.5, color=INK, linespacing=1.05, zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    districts.boundary.plot(ax=ax, color="#9a988f", linewidth=0.25, zorder=4)
    country.boundary.plot(ax=ax, color="#6f6d66", linewidth=0.5, zorder=4)
    ax.text(90.35, 21.0, "Bay of Bengal", ha="center", fontsize=6.5,
            color="#4F7FB5", style="italic", zorder=6)
    ax.text(88.25, 23.2, "INDIA", ha="center", fontsize=6, color="#8f8d86", zorder=6)
    ax.set_xlim(87.9, 93.0)
    ax.set_ylim(20.5, 26.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, zorder=0, color="#dde6ef")
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_study_area.pdf")
    plt.close(fig)


# ── Figure 3: observed flood frequency ──────────────────────────────────────
def fig_flood_frequency():
    """Rangpur full height on the left, Sylhet and the coast stacked right,
    each panel at its true aspect so the three share one map scale."""
    districts = _districts()
    boxes = {r: _cfg(r)["aoi"]["bbox"] for r in REGIONS}
    aspect = {r: (b[2] - b[0]) / (b[3] - b[1]) for r, b in boxes.items()}
    # Right column width w; its panels' heights sum to the left panel height.
    right_h = [1 / aspect["sylhet"], 1 / aspect["sw_coastal"]]
    height = sum(right_h)
    left_w = aspect["rangpur_rajshahi"] * height
    fig = plt.figure(figsize=(FULL_W * 0.86, FULL_W * 0.86 * height / (left_w + 1) * 1.02))
    gs = fig.add_gridspec(2, 2, width_ratios=[left_w, 1], height_ratios=right_h,
                          wspace=0.16, hspace=0.3, left=0.07, right=0.99,
                          top=0.95, bottom=0.16)
    axes = {"rangpur_rajshahi": fig.add_subplot(gs[:, 0]),
            "sylhet": fig.add_subplot(gs[0, 1]),
            "sw_coastal": fig.add_subplot(gs[1, 1])}
    image = None
    for region, ax in axes.items():
        cfg = _cfg(region)
        raw, _, _ = _paths(cfg)
        data, extent = _read_wgs84(raw / "s1_flood_frequency.tif")
        data = np.where(data > 0, data, np.nan)   # never-flooded ground left blank
        districts.plot(ax=ax, facecolor="#f7f6f3", edgecolor="none")
        image = ax.imshow(data, extent=extent, cmap=SEQ, vmin=0, vmax=100,
                          interpolation="nearest", zorder=2)
        districts.boundary.plot(ax=ax, color="#8f8d86", linewidth=0.3, zorder=3)
        w, s_, e, n = boxes[region]
        ax.set_xlim(w, e); ax.set_ylim(s_, n); ax.set_aspect("equal")
        n_events = len(cfg["sentinel1"]["events"])
        ax.set_title(f"{REGIONS[region][0]} ({n_events} events)", color=INK)
        ax.tick_params(length=2)
    cax = fig.add_axes([0.25, 0.075, 0.5, 0.02])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_label("Share of mapped events in which the ground flooded (%)")
    cbar.outline.set_linewidth(0.4)
    fig.savefig(OUT / "fig_flood_frequency.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ── Figure 4: hazard surfaces ───────────────────────────────────────────────
def fig_hazard_surfaces():
    cfg = _cfg("rangpur_rajshahi")
    _, _, out = _paths(cfg)
    districts = _districts()
    panels = [("Kriged asset scores", out / "flood_risk_kriged.tif"),
              ("Terrain model", out / "flood_hazard_terrain.tif")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.78, 3.0),
                             gridspec_kw={"wspace": 0.08})
    image = None
    for ax, (title, path) in zip(axes, panels):
        data, extent = _read_wgs84(path)
        image = ax.imshow(data, extent=extent, cmap=SEQ, vmin=0, vmax=1,
                          interpolation="nearest")
        districts.boundary.plot(ax=ax, color="white", linewidth=0.3)
        w, s, e, n = cfg["aoi"]["bbox"]
        ax.set_xlim(w, e); ax.set_ylim(s, n); ax.set_aspect("equal")
        ax.set_title(title, color=INK)
        ax.tick_params(length=2)
    axes[1].set_yticklabels([])
    cbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.015, aspect=25)
    cbar.set_label("Flood hazard score")
    cbar.outline.set_linewidth(0.4)
    fig.savefig(OUT / "fig_hazard_surfaces.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ── Scores and labels on the test blocks ────────────────────────────────────
def _label_comparison(region):
    """Observed labels and the scores of the two label sources, from the
    pipeline's own label comparison (python -m pipeline.cli benchmark)."""
    _, _, out = _paths(_cfg(region))
    data = np.load(out / "label_comparison_scores.npz")
    return data["y"], data["observed_trained"], data["proxy_trained"]


def _calibration_data(region):
    """Raw scores, calibrated probabilities and labels on the test blocks."""
    import torch

    cfg = _cfg(region)
    raw, _, out = _paths(cfg)
    graph = torch.load(out / "spatial_graph.pt", weights_only=False)
    test = (graph.test_mask if "test_mask" in graph else graph.val_mask).numpy().astype(bool)
    y = graph.y.numpy().astype(int)[test]
    # The model without the flood record: the map's scores may carry it, and
    # this figure scores them against observed flooding.
    def _first(*names):
        return next(out / n for n in names if (out / n).exists())
    scores = np.load(_first("susceptibility_scores.npy", "gnn_risk_scores.npy"))[test]
    probability = np.load(_first("susceptibility_probability.npy",
                                 "gnn_flood_probability.npy"))[test]
    return y, scores, probability


# ── Figure 5: ROC curves ────────────────────────────────────────────────────
def fig_roc():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45),
                             gridspec_kw={"wspace": 0.28})
    summary = {}
    for ax, region in zip(axes, REGIONS):
        from sklearn.metrics import roc_auc_score

        y, obs_scores, proxy_scores = _label_comparison(region)
        a_obs, a_proxy = roc_auc_score(y, obs_scores), roc_auc_score(y, proxy_scores)
        summary[region] = {"observed": a_obs, "proxy": a_proxy, "n": int(len(y)),
                           "positives": int(y.sum())}
        ax.plot([0, 1], [0, 1], color="#9a988f", linewidth=0.7,
                linestyle=(0, (2, 2)), zorder=1)
        fpr, tpr, _ = roc_curve(y, proxy_scores)
        ax.plot(fpr, tpr, color=ORANGE, linewidth=1.5, linestyle=(0, (4, 1.6)),
                label="Trained on terrain proxy labels", zorder=2)
        fpr, tpr, _ = roc_curve(y, obs_scores)
        ax.plot(fpr, tpr, color=BLUE, linewidth=1.6,
                label="Trained on observed Sentinel-1 floods", zorder=3)
        ax.text(0.97, 0.05, f"AUC observed {a_obs:.3f}\nAUC proxy {a_proxy:.3f}",
                ha="right", va="bottom", fontsize=7, color=INK,
                transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.set_aspect("equal")
        ax.set_title(f"{REGIONS[region][0]} (n = {len(y):,})", color=INK)
        ax.set_xlabel("False positive rate")
        ax.grid(True, zorder=0)
        ax.tick_params(length=2)
    axes[0].set_ylabel("True positive rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2,
               frameon=False, handlelength=2.6, bbox_to_anchor=(0.5, 1.03))
    fig.savefig(OUT / "fig_roc.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return summary


# ── Figure 6: calibration ───────────────────────────────────────────────────
def fig_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45),
                             gridspec_kw={"wspace": 0.28})
    for ax, region in zip(axes, REGIONS):
        y, raw_scores, probability = _calibration_data(region)

        def reliability(p, bins=10):
            edges = np.quantile(p, np.linspace(0, 1, bins + 1))
            edges[-1] += 1e-9
            idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
            xs, ys = [], []
            for b in range(bins):
                m = idx == b
                if m.sum() >= 20:
                    xs.append(p[m].mean()); ys.append(y[m].mean())
            return np.array(xs), np.array(ys)

        ax.plot([0, 1], [0, 1], color="#9a988f", linewidth=0.7,
                linestyle=(0, (2, 2)), zorder=1)
        x, obs = reliability(raw_scores)
        ax.plot(x, obs, color=ORANGE, linewidth=1.5, linestyle=(0, (4, 1.6)),
                marker="o", markersize=3.2, markeredgecolor="white",
                markeredgewidth=0.6, label="Raw score", zorder=2)
        x, obs = reliability(probability)
        ax.plot(x, obs, color=BLUE, linewidth=1.6, marker="o", markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.6,
                label="Calibrated probability", zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(REGIONS[region][0], color=INK)
        ax.set_xlabel("Predicted probability (test-block deciles)")
        ax.grid(True, zorder=0)
        ax.tick_params(length=2)
    axes[0].set_ylabel("Observed share flooded")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2,
               frameon=False, handlelength=2.6, bbox_to_anchor=(0.5, 1.03))
    fig.savefig(OUT / "fig_calibration.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ── Figure 2 inputs: counts for the processing-chain diagram ────────────────
def chain_counts():
    """Counts shown in the source boxes of the D3 chain diagram, read from
    the pipeline outputs and written to chain_d3/counts.json."""
    import csv

    assets = events = unions = 0
    for region in REGIONS:
        cfg = _cfg(region)
        raw, _, out = _paths(cfg)
        meta = json.loads((out / "pipeline_metadata.json").read_text())
        assets += int(meta["risk_score_stats"]["n_assets"])
        events += len(cfg["sentinel1"]["events"])
        with open(out / "union_risk_summary.csv") as f:
            unions += sum(1 for _ in csv.DictReader(f))
    cht = _load_config(str(ROOT / "configs/cht.yaml"))
    _, _, cht_out = _paths(cht)
    landslide = json.loads((cht_out / "landslide_model.json").read_text())

    def metres(path):
        with rasterio.open(path) as src:
            res = abs(src.res[0])
            if src.crs.is_geographic:          # spacing along a meridian
                res *= 111_320
        return int(round(res / 10.0) * 10)

    from pipeline.asset_model import (BOOST_ITERATIONS, BOOST_LEARNING_RATE,
                                      model_type)

    rr = _cfg("rangpur_rajshahi")
    raw, _, _ = _paths(rr)
    graph, gnn = rr["graph"], rr["gnn"]
    counts = {
        "regions": len(REGIONS),
        "assets": assets,
        "events": events,
        "unions": unions,
        "landslides": int(landslide["inventory"]["n_landslides"]),
        "srtm_m": metres(raw / "dem_srtm_30m.tif"),
        "jrc_m": metres(raw / "jrc_water_occurrence.tif"),
        "worldpop_m": metres(raw / "worldpop_popdens.tif"),
        "grid_m": int(rr["aoi"]["grid_resolution_m"]),
        "k": int(graph["k_neighbors"]),
        "max_edge_km": graph["max_edge_distance_m"] / 1000,
        "block_km": graph["block_size_m"] / 1000,
        "train_pct": int(round(100 * graph.get("train_split", 0.8))),
        "hidden": int(gnn["hidden_channels"]),
        "dropout": gnn["dropout"],
        "model": model_type(rr),
        "boost_iterations": BOOST_ITERATIONS,
        "boost_lr": BOOST_LEARNING_RATE,
    }
    path = OUT / "chain_d3" / "counts.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(counts, indent=2))
    return counts


# ── Figure 2: processing chain (D3, after the GENESIS architecture figure) ──
def fig_chain():
    """Render chain_d3/chain.js headlessly and convert the SVG to PDF.
    Needs node (with `npm install` run once in chain_d3) and rsvg-convert."""
    import subprocess

    chain_counts()
    d3_dir = OUT / "chain_d3"
    subprocess.run(["node", "render.mjs"], cwd=d3_dir, check=True)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT / "fig_chain.pdf"),
                    str(d3_dir / "chain.svg")], check=True)


# ── Figure 7: the graph model against tabular baselines ────────────────────
def fig_benchmark():
    """Per-region AUC of every model over the same repeated block splits.
    One dot per split, with the mean marked, so the spread is visible."""
    order = [("graph_sage", "GraphSAGE"), ("gradient_boosting", "Gradient boosting"),
             ("random_forest", "Random forest"), ("logistic_regression", "Logistic regression"),
             ("twi_only", "Wetness index alone")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.35),
                             gridspec_kw={"wspace": 0.12})
    for ax, region in zip(axes, REGIONS):
        _, _, out = _paths(_cfg(region))
        bench = json.loads((out / "benchmark.json").read_text())
        for row, (key, label) in enumerate(order):
            aucs = [r["models"][key]["auc_roc"] for r in bench["per_seed"]]
            colour = BLUE if key == "graph_sage" else ORANGE
            y = len(order) - 1 - row
            ax.scatter(aucs, [y] * len(aucs), s=13, facecolor="white",
                       edgecolor=colour, linewidth=0.9, zorder=3)
            ax.plot([np.mean(aucs)], [y], marker="|", markersize=11,
                    markeredgewidth=1.8, color=colour, zorder=4)
        ax.axvline(0.5, color="#9a988f", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([label for _, label in order][::-1])
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("AUC on held-out test blocks")
        ax.set_title(REGIONS[region][0], color=INK)
        ax.grid(True, axis="x", zorder=0)
        ax.tick_params(length=2)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.savefig(OUT / "fig_benchmark.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    fig_chain(); print("fig_chain.pdf")
    fig_study_area(); print("fig_study_area.pdf")
    fig_flood_frequency(); print("fig_flood_frequency.pdf")
    fig_hazard_surfaces(); print("fig_hazard_surfaces.pdf")
    summary = fig_roc(); print("fig_roc.pdf", json.dumps(
        {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
         for k, v in summary.items()}))
    fig_calibration(); print("fig_calibration.pdf")
    fig_benchmark(); print("fig_benchmark.pdf")
