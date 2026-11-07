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
TABLE = Path(__file__).with_name("national_table.tex")
PARTITION = ROOT / "configs" / "national" / "partition.yaml"
CHAIN = ROOT / "docs" / "figures" / "chain_d3"
UNIONS = ROOT / "data" / "shared" / "geoboundaries" / "BGD_ADM4.geojson"
DISTRICTS = ROOT / "data" / "shared" / "geoboundaries" / "BGD_ADM2.geojson"
FIGURE = ROOT / "docs" / "figures" / "fig_national.pdf"
SPREAD_FIGURE = ROOT / "docs" / "figures" / "fig_spread.pdf"

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


def event_years(partition: dict) -> tuple[int, int]:
    """First and last year of the Sentinel-1 events behind the labels."""
    from pipeline.cli import _load_config

    years = [int(event["start"][:4])
             for region in partition
             for event in _load_config(str(ROOT / REGION_CONFIGS[region][2]))["sentinel1"]["events"]]
    return min(years), max(years)


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
        "key": region, "name": name, "assets": assets,
        "rate": meta["label_positive_rate"],
        "auc": boosting["auc_mean"], "sd": boosting["auc_sd"],
        "observed": validation["assets_vs_observed"]["held_out_blocks"]["auc_roc"],
        "gain": labels["mean"], "ahead": labels["seeds_observed_ahead"], "seeds": labels["n_seeds"],
        "graph": benchmark["graph_vs_best_baseline"]["auc_difference_mean"],
        **threshold_fields(out),
    }


def threshold_fields(out: Path) -> dict:
    """Precision, recall and F1 at the calibration-block threshold, and the
    covariate test, when those stages have been run for the region."""
    fields = {}
    path = out / "threshold_metrics.json"
    if path.exists():
        t = _json(path)
        s = t["summary"]
        fields |= {
            "share": sum(r["test_positive_rate"] for r in t["per_seed"]) / len(t["per_seed"]),
            "precision": s["gradient_boosting"]["precision"]["mean"],
            "recall": s["gradient_boosting"]["recall"]["mean"],
            "f1": s["gradient_boosting"]["f1"]["mean"], "f1_sd": s["gradient_boosting"]["f1"]["sd"],
            "twi_auc": s["twi_only"]["auc_roc"]["mean"], "twi_f1": s["twi_only"]["f1"]["mean"],
            "rf_auc": s["random_forest"]["auc_roc"]["mean"],
        }
    path = out / "prediction_spread.json"
    if path.exists():
        p = _json(path)
        fields |= {"spread_splits": p["n_splits"], "spread_min": p["min_predictions"],
                   **{f"spread_{k}": v for k, v in p["summary"].items()}}
    path = out / "covariate_ablation.json"
    if path.exists():
        c = _json(path)["summary"]
        fields |= {"cov_gain": c["auc_difference_mean"], "cov_p": c["p_value"],
                   "cov_ahead": c["seeds_covariates_ahead"], "cov_seeds": c["n_seeds"]}
        if "variants" in c:
            fields["xy_gain"] = c["variants"]["+xy_coordinates"]["auc_difference_mean"]
            for name in ("ndvi_dry", "clay", "rain_annual"):
                fields[f"{name}_gain"] = c["variants"][f"+{name}"]["auc_difference_mean"]
    return fields


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def make_spread_figure(rows: list[dict]) -> None:
    """(a) per-asset spread of held-out probability over the block assignments;
    (b) how often an asset's flag holds when the training blocks change."""
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    districts = gpd.read_file(DISTRICTS)
    # about 200 m, invisible at print size; the full outlines make an 8 MB file
    districts["geometry"] = districts.geometry.simplify(0.002)
    fig = plt.figure(figsize=(390 / 72, 3.5))
    ax_map = fig.add_axes([0.0, 0.04, 0.33, 0.86])
    ax_bar = fig.add_axes([0.62, 0.2, 0.36, 0.7])

    districts.plot(ax=ax_map, facecolor="#f2f2f2", edgecolor="#bdbdbd", linewidth=0.2)
    lon, lat, sd = [], [], []
    for r in rows:
        d = np.load(region_paths(r["key"])["output"] / "prediction_spread.npz")
        ok = np.isfinite(d["sd"])
        lon.append(d["lon"][ok]); lat.append(d["lat"][ok]); sd.append(d["sd"][ok])
    lon, lat, sd = map(np.concatenate, (lon, lat, sd))
    order = np.argsort(sd)              # widest spreads drawn last, on top
    points = ax_map.scatter(lon[order], lat[order], c=sd[order], s=0.15, cmap="magma_r",
                            vmin=0, vmax=0.3, linewidths=0, rasterized=True)
    ax_map.set_axis_off()
    ax_map.set_aspect("equal")
    bar = fig.colorbar(points, ax=ax_map, orientation="horizontal", fraction=0.04, pad=0.0,
                       shrink=0.8, extend="max")
    bar.set_label("s.d. of held-out probability", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    fig.text(0.005, 0.9 + 6 / (3.5 * 72), r"\textbf{a}\enspace Spread of each asset's score",
             fontsize=8.5, va="bottom")

    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        t, f = 100 * r["spread_share_flag_consistent"], 100 * r["spread_fixed_share_consistent"]
        ax_bar.plot([t, f], [yi, yi], color="#bdbdbd", lw=1.2, zorder=1)
        ax_bar.plot(t, yi, "o", ms=4.5, mfc="white", mec="black", mew=0.7, zorder=2)
        ax_bar.plot(f, yi, "o", ms=4.5, color="#3b0f70", mec="black", mew=0.4, zorder=3)
    ax_bar.set_yticks(y, [rf"{i}\enspace {r['name']}" for i, r in enumerate(rows, start=1)])
    ax_bar.set_xlim(35, 100)
    ax_bar.set_xlabel(r"assets whose flag never changes (\%)")
    ax_bar.grid(axis="x", color="#e5e5e5", lw=0.5)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.legend(handles=[
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="black", ms=4.5,
                   label="threshold from calibration blocks"),
        plt.Line2D([], [], marker="o", ls="none", color="#3b0f70", mec="black", ms=4.5,
                   label="fixed share of assets")],
        loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2, fontsize=6.5, frameon=False)
    ax_bar.set_title(r"\textbf{b}\enspace Stability of the flag", loc="left", fontsize=8.5)

    fig.savefig(SPREAD_FIGURE, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)
    print(f"{SPREAD_FIGURE}")


def write_table(rows: list[dict]) -> None:
    """Held-out skill of the default model by region, beside its baselines."""
    def f(v, d=3):
        return "--" if v is None else f"{v:.{d}f}"

    def signed(v):
        return "--" if v is None else f"{v:+.3f}"

    lines = [
        "% Written by docs/softwarex/build_national.py from the pipeline outputs; do not edit.",
        "\\begin{tabular}{@{}lrcccccccrr@{}}",
        "\\toprule",
        " & & \\multicolumn{4}{c}{Default model (gradient boosting)} & RF & "
        "\\multicolumn{2}{c}{Wetness index} & \\multicolumn{2}{c}{$\\Delta$AUC when added} \\\\",
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-9}\\cmidrule(l){10-11}",
        "Region & Floods (\\%) & AUC & Precision & Recall & F1 & AUC & AUC & F1 & "
        "Covariates & Coordinates \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i} {r['name']} & {100 * r['share']:.1f} & {f(r['auc'])} $\\pm$ {f(r['sd'])} & "
            f"{f(r['precision'])} & {f(r['recall'])} & {f(r['f1'])} $\\pm$ {f(r['f1_sd'])} & "
            f"{f(r['rf_auc'])} & {f(r['twi_auc'])} & {f(r['twi_f1'])} & "
            f"{signed(r.get('cov_gain'))} & {signed(r.get('xy_gain'))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    TABLE.write_text("\n".join(lines) + "\n")


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
    import re
    import subprocess

    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                               cwd=ROOT, capture_output=True, text=True).stdout
    tests = int(re.search(r"(\d+) tests? collected", collected).group(1))
    lines.append(f"\\newcommand{{\\NatTests}}{{{tests}}}")
    first, last = event_years(partition)
    lines += [f"\\newcommand{{\\NatEventWindows}}{{{event_windows(partition)}}}",
              f"\\newcommand{{\\NatEventFirst}}{{{first}}}",
              f"\\newcommand{{\\NatEventLast}}{{{last}}}"]
    if all("f1" in r for r in rows):
        rf_lead = [r["rf_auc"] - r["auc"] for r in rows]
        lines += [
            f"\\newcommand{{\\NatFOneMin}}{{{min(r['f1'] for r in rows):.2f}}}",
            f"\\newcommand{{\\NatFOneMax}}{{{max(r['f1'] for r in rows):.2f}}}",
            f"\\newcommand{{\\NatTWIFOneMin}}{{{min(r['twi_f1'] for r in rows):.2f}}}",
            f"\\newcommand{{\\NatTWIFOneMax}}{{{max(r['twi_f1'] for r in rows):.2f}}}",
            f"\\newcommand{{\\NatTWIAUCMin}}{{{min(r['twi_auc'] for r in rows):.2f}}}",
            f"\\newcommand{{\\NatTWIAUCMax}}{{{max(r['twi_auc'] for r in rows):.2f}}}",
            f"\\newcommand{{\\NatShareMin}}{{{100 * min(r['share'] for r in rows):.0f}}}",
            f"\\newcommand{{\\NatShareMax}}{{{100 * max(r['share'] for r in rows):.0f}}}",
            f"\\newcommand{{\\NatRFLeadMin}}{{{min(rf_lead):.3f}}}",
            f"\\newcommand{{\\NatRFLeadMax}}{{{max(rf_lead):.3f}}}",
            f"\\newcommand{{\\NatRFAhead}}{{{sum(d > 0 for d in rf_lead)}}}",
            f"\\newcommand{{\\NatRFWithinSD}}{{{sum(abs(d) < r['sd'] for d, r in zip(rf_lead, rows))}}}",
        ]
    if all("xy_gain" in r for r in rows):
        lines += [
            f"\\newcommand{{\\NatCovGainMin}}{{{min(r['cov_gain'] for r in rows):.3f}}}",
            f"\\newcommand{{\\NatCovGainMax}}{{{max(r['cov_gain'] for r in rows):.3f}}}",
            f"\\newcommand{{\\NatCovSignificant}}{{{sum(r['cov_p'] < 0.05 for r in rows)}}}",
            f"\\newcommand{{\\NatXYGainMin}}{{{min(r['xy_gain'] for r in rows):.3f}}}",
            f"\\newcommand{{\\NatXYGainMax}}{{{max(r['xy_gain'] for r in rows):.3f}}}",
            f"\\newcommand{{\\NatCovBeatsXY}}{{{sum(r['cov_gain'] > r['xy_gain'] for r in rows)}}}",
            # coordinates' gain as a share of the three layers' gain, over all regions
            f"\\newcommand{{\\NatXYSharePct}}{{{100 * sum(r['xy_gain'] for r in rows) / sum(r['cov_gain'] for r in rows):.0f}}}",
            f"\\newcommand{{\\NatNDVILargest}}{{{sum(max(('ndvi_dry', 'clay', 'rain_annual'), key=lambda k: r[k + '_gain']) == 'ndvi_dry' for r in rows)}}}",
            f"\\newcommand{{\\NatRainLargest}}{{{sum(max(('ndvi_dry', 'clay', 'rain_annual'), key=lambda k: r[k + '_gain']) == 'rain_annual' for r in rows)}}}",
            f"\\newcommand{{\\NatClayGainMax}}{{{max(r['clay_gain'] for r in rows):.3f}}}",
            f"\\newcommand{{\\NatClayNegative}}{{{sum(r['clay_gain'] < 0 for r in rows)}}}",
        ]
        use = [r["name"] for r in rows if r["cov_p"] < 0.05 and r["cov_gain"] > r["xy_gain"]]
        lines += [f"\\newcommand{{\\NatCovUseCount}}{{{len(use)}}}",
                  f"\\newcommand{{\\NatCovUseRegions}}{{{_join(use)}}}"]
        write_table(rows)
    if all("twi_auc" in r for r in rows):
        # per-region skill for the comparison with published studies
        for r in rows:
            k = r["key"].replace("_", "")
            lines += [f"\\newcommand{{\\NatAUC{k}}}{{{r['auc']:.3f}}}",
                      f"\\newcommand{{\\NatSD{k}}}{{{r['sd']:.3f}}}",
                      f"\\newcommand{{\\NatTWIAUC{k}}}{{{r['twi_auc']:.2f}}}"]
    if all("spread_splits" in r for r in rows):
        def pct(key, fn):
            return f"{100 * fn(r[key] for r in rows):.0f}"
        lines += [
            f"\\newcommand{{\\NatSpreadSplits}}{{{rows[0]['spread_splits']}}}",
            f"\\newcommand{{\\NatSpreadMinPred}}{{{rows[0]['spread_min']}}}",
            f"\\newcommand{{\\NatSpreadScoredMin}}{{{pct('spread_share_scored', min)}}}",
            f"\\newcommand{{\\NatSpreadConsistentMin}}{{{pct('spread_share_flag_consistent', min)}}}",
            f"\\newcommand{{\\NatSpreadConsistentMax}}{{{pct('spread_share_flag_consistent', max)}}}",
            f"\\newcommand{{\\NatSpreadAlwaysMin}}{{{pct('spread_share_always_given_ever', min)}}}",
            f"\\newcommand{{\\NatSpreadAlwaysMax}}{{{pct('spread_share_always_given_ever', max)}}}",
            f"\\newcommand{{\\NatSpreadTopSDMin}}{{{min(r['spread_sd_median_top_decile'] for r in rows):.2f}}}",
            f"\\newcommand{{\\NatSpreadTopSDMax}}{{{max(r['spread_sd_median_top_decile'] for r in rows):.2f}}}",
            f"\\newcommand{{\\NatSpreadMedianPred}}{{{min(r['spread_median_predictions'] for r in rows):.0f}--{max(r['spread_median_predictions'] for r in rows):.0f}}}",
        ]
        if all("spread_fixed_share_consistent" in r for r in rows):
            lines += [
                f"\\newcommand{{\\NatSpreadFixedConsistentMin}}{{{pct('spread_fixed_share_consistent', min)}}}",
                f"\\newcommand{{\\NatSpreadFixedConsistentMax}}{{{pct('spread_fixed_share_consistent', max)}}}",
                f"\\newcommand{{\\NatSpreadFixedAlwaysMin}}{{{pct('spread_fixed_share_always_given_ever', min)}}}",
                f"\\newcommand{{\\NatSpreadFixedAlwaysMax}}{{{pct('spread_fixed_share_always_given_ever', max)}}}",
            ]
        make_spread_figure(rows)
    OUT.write_text("\n".join(lines) + "\n")
    write_chain_counts(partition, total)
    make_figure(partition, config["landslide"]["cht"]["districts"], rows)
    print(f"{OUT}: {len(rows)} regions, {n_districts} districts, {total:,} assets")


if __name__ == "__main__":
    main()
