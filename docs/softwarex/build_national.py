"""
Write the SoftwareX manuscript's national-partition table from pipeline outputs.

Like docs/build_numbers.py for the other documents, nothing here is typed by
hand: each row reads a region's validation, benchmark and label-comparison
files, so rerunning a region and this script updates the table.

    python docs/softwarex/build_national.py     # -> docs/softwarex/national.tex,
                                                #    docs/figures/fig_national.pdf
                                                #    and the chain figure's counts

The chain figure's national counts (docs/figures/chain_d3/counts_national.json)
are written here too; render them with `node render.mjs --national` and
`rsvg-convert -f pdf -o ../fig_chain_national.pdf chain_national.svg`.
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.data.regions import REGION_CONFIGS, region_paths  # noqa: E402

OUT = Path(__file__).with_name("national.tex")
PARTITION = ROOT / "configs" / "national" / "partition.yaml"
CHAIN = ROOT / "docs" / "figures" / "chain_d3"
UNIONS = ROOT / "data" / "shared" / "geoboundaries" / "BGD_ADM4.geojson"
DISTRICTS = ROOT / "data" / "shared" / "geoboundaries" / "BGD_ADM2.geojson"
FIGURE = ROOT / "docs" / "figures" / "fig_national.pdf"

# Okabe-Ito colours, which stay distinct for colour-blind readers, one per
# flood region; the landslide region is grey and hatched.
REGION_COLOURS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                  "#0072B2", "#D55E00", "#CC79A7", "#44AA99"]


def event_windows(partition: dict) -> int:
    """Distinct Sentinel-1 event windows over every region of the partition."""
    from pipeline.cli import _load_config

    windows = set()
    for region in partition:
        _, _, config = REGION_CONFIGS[region]
        for event in _load_config(str(ROOT / config))["sentinel1"]["events"]:
            windows.add((event["start"], event["end"]))
    return len(windows)


def write_chain_counts(partition: dict, assets: int) -> None:
    """The chain figure's counts for the national partition."""
    counts = json.loads((CHAIN / "counts.json").read_text())
    n_windows = event_windows(partition)
    counts.update({
        "regions": len(partition), "assets": assets, "events": n_windows,
        "events_label": f"{n_windows} event windows",
        "unions": len(_json(UNIONS)["features"]),
    })
    (CHAIN / "counts_national.json").write_text(json.dumps(counts, indent=2) + "\n")


def _json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def region_row(region: str, name: str) -> dict:
    out = region_paths(region)["output"]
    validation = _json(out / "validation_metrics.json")
    benchmark = _json(out / "benchmark.json")["summary"]
    labels = _json(out / "label_comparison.json")["summary"]["observed_minus_proxy"]
    assets = len(_json(out / "risk_ranked_assets.geojson")["features"])
    meta = _json(out / "pipeline_metadata.json")
    boosting = benchmark["gradient_boosting"]
    return {
        "name": name, "assets": assets,
        "rate": meta["label_positive_rate"],
        "auc": boosting["auc_mean"], "sd": boosting["auc_sd"],
        "observed": validation["assets_vs_observed"]["held_out_blocks"]["auc_roc"],
        "gain": labels["mean"], "ahead": labels["seeds_observed_ahead"], "seeds": labels["n_seeds"],
        "graph": benchmark["graph_vs_best_baseline"]["auc_difference_mean"],
    }


def make_figure(partition: dict, landslide: list[str], rows: list[dict]) -> None:
    """(a) the partition on a map; (b) each region's AUC on held-out blocks and
    against observed floods; (c) the AUC won by training on radar labels."""
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # LaTeX sets the text, so the figure reads in the article's own typeface
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    districts = gpd.read_file(DISTRICTS).to_crs("EPSG:32646")
    region_of = {d: r for r, spec in partition.items() for d in spec["districts"]}
    region_of.update({d: "cht" for d in landslide})
    districts["region"] = districts["shapeName"].map(region_of)
    regions = districts.dissolve(by="region").reset_index()

    order = list(partition)
    colour = dict(zip(order, REGION_COLOURS))
    # Drawn at the article's text width (390 pt), so it prints unscaled and its
    # labels keep their size; panels are placed by hand because the region
    # names need the gap between map and chart.
    fig = plt.figure(figsize=(390 / 72, 3.7))
    ax_map = fig.add_axes([0.0, 0.05, 0.25, 0.84])
    ax_auc = fig.add_axes([0.555, 0.21, 0.255, 0.69])
    ax_gain = fig.add_axes([0.845, 0.21, 0.135, 0.69], sharey=ax_auc)

    for _, unit in regions.iterrows():
        if unit["region"] == "cht":
            gpd.GeoSeries([unit.geometry]).plot(ax=ax_map, facecolor="#d9d9d9", edgecolor="#8c8c8c",
                                                 hatch="////", linewidth=0.3)
        else:
            gpd.GeoSeries([unit.geometry]).plot(ax=ax_map, facecolor=colour[unit["region"]],
                                                 edgecolor="white", linewidth=0.5)
    districts.boundary.plot(ax=ax_map, color="white", linewidth=0.15)
    for i, region in enumerate(order, start=1):
        point = regions.loc[regions["region"] == region].geometry.iloc[0].representative_point()
        ax_map.annotate(rf"\textbf{{{i}}}", (point.x, point.y), ha="center", va="center", fontsize=7.5,
                        bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="none", alpha=0.9))
    ax_map.set_axis_off()
    ax_map.set_aspect("equal")
    ax_map.legend(handles=[Patch(facecolor="#d9d9d9", edgecolor="#8c8c8c", hatch="////",
                                 label="Hill Tracts (landslide)")],
                  loc="lower left", frameon=False, fontsize=7, handlelength=1.2)
    # the map keeps its aspect and shrinks, so its title is set level with b and c
    fig.text(0.005, 0.9 + 6 / (3.7 * 72), r"\textbf{a}\enspace National partition",
             fontsize=8.5, va="bottom")

    y = list(range(len(rows)))[::-1]
    labels = [rf"{i}\enspace {r['name']} ({r['assets']:,})" for i, r in enumerate(rows, start=1)]
    for yi, r, region in zip(y, rows, order):
        ax_auc.errorbar(r["auc"], yi, xerr=r["sd"], fmt="o", color=colour[region], ms=4.5,
                        mec="black", mew=0.4, ecolor="black", elinewidth=0.7, capsize=2)
        ax_auc.plot(r["observed"], yi, marker="D", ms=4, mfc="white", mec="black", mew=0.7, ls="none")
    ax_auc.axvline(0.5, color="#999999", lw=0.6, ls=":")
    ax_auc.set_xlim(0.45, 1.0)
    ax_auc.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax_auc.set_yticks(y, labels)
    ax_auc.set_xlabel("AUC")
    ax_auc.grid(axis="x", color="#e5e5e5", lw=0.5)
    ax_auc.set_axisbelow(True)
    ax_auc.legend(handles=[
        plt.Line2D([], [], marker="o", color="black", mfc="#bbbbbb", ms=4.5, lw=0.7,
                   label=r"held-out blocks, mean $\pm$ s.d."),
        plt.Line2D([], [], marker="D", color="black", mfc="white", ms=4, ls="none",
                   label="against observed floods")],
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=6.5, frameon=False)
    ax_auc.set_title(r"\textbf{b}\enspace Skill on unseen ground", loc="left", fontsize=8.5)

    for yi, r, region in zip(y, rows, order):
        ax_gain.barh(yi, r["gain"], color=colour[region], edgecolor="black", lw=0.4, height=0.6)
        ax_gain.text(r["gain"] + 0.008, yi, f"{r['ahead']}/{r['seeds']}", va="center", fontsize=6.5)
    ax_gain.axvline(0, color="black", lw=0.6)
    ax_gain.set_xlim(0, 0.27)
    ax_gain.set_xticks([0, 0.1, 0.2])
    ax_gain.set_xlabel(r"AUC gain")
    ax_gain.tick_params(axis="y", left=False, labelleft=False)
    ax_gain.grid(axis="x", color="#e5e5e5", lw=0.5)
    ax_gain.set_axisbelow(True)
    ax_gain.set_title(r"\textbf{c}\enspace Radar labels", loc="left", fontsize=8.5)
    for ax in (ax_auc, ax_gain):
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIGURE, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{FIGURE}")


def main() -> None:
    config = yaml.safe_load(PARTITION.read_text())
    partition = config["flood"]
    rows = [region_row(region, spec["name"]) for region, spec in partition.items()]
    n_districts = sum(len(spec["districts"]) for spec in partition.values())
    total = sum(r["assets"] for r in rows)
    # how far the graph network trailed the best model, as positive gaps
    gaps = [-r["graph"] for r in rows]

    lines = [
        "% Written by docs/softwarex/build_national.py from the pipeline outputs; do not edit.",
        f"\\newcommand{{\\NatRegions}}{{{len(rows)}}}",
        f"\\newcommand{{\\NatDistricts}}{{{n_districts}}}",
        f"\\newcommand{{\\NatAssets}}{{{total:,}}}",
        f"\\newcommand{{\\NatGraphGapMin}}{{{min(gaps):.3f}}}",
        f"\\newcommand{{\\NatGraphGapMax}}{{{max(gaps):.3f}}}",
        f"\\newcommand{{\\NatAUCMin}}{{{min(r['auc'] for r in rows):.3f}}}",
        f"\\newcommand{{\\NatAUCMax}}{{{max(r['auc'] for r in rows):.3f}}}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    write_chain_counts(partition, total)
    make_figure(partition, config["landslide"]["cht"]["districts"], rows)
    print(f"{OUT}: {len(rows)} regions, {n_districts} districts, {total:,} assets")


if __name__ == "__main__":
    main()
