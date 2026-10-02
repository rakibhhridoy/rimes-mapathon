"""
Write the SoftwareX manuscript's national-partition table from pipeline outputs.

Like docs/build_numbers.py for the other documents, nothing here is typed by
hand: each row reads a region's validation, benchmark and label-comparison
files, so rerunning a region and this script updates the table.

    python docs/softwarex/build_national.py     # -> docs/softwarex/national.tex
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


def main() -> None:
    partition = yaml.safe_load(PARTITION.read_text())["flood"]
    rows = [region_row(region, spec["name"]) for region, spec in partition.items()]
    n_districts = sum(len(spec["districts"]) for spec in partition.values())
    total = sum(r["assets"] for r in rows)
    # how far the graph network trailed the best model, as positive gaps
    gaps = [-r["graph"] for r in rows]

    body = "\n".join(
        f"{r['name']} & {r['assets']:,} & {100 * r['rate']:.1f} & "
        f"{r['auc']:.3f} $\\pm$ {r['sd']:.3f} & {r['observed']:.3f} & "
        f"{r['gain']:+.3f} ({r['ahead']}/{r['seeds']}) \\\\" for r in rows)
    lines = [
        "% Written by docs/softwarex/build_national.py from the pipeline outputs; do not edit.",
        f"\\newcommand{{\\NatRegions}}{{{len(rows)}}}",
        f"\\newcommand{{\\NatDistricts}}{{{n_districts}}}",
        f"\\newcommand{{\\NatAssets}}{{{total:,}}}",
        f"\\newcommand{{\\NatGraphGapMin}}{{{min(gaps):.3f}}}",
        f"\\newcommand{{\\NatGraphGapMax}}{{{max(gaps):.3f}}}",
        f"\\newcommand{{\\NatTableRows}}{{{body}}}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    write_chain_counts(partition, total)
    print(f"{OUT}: {len(rows)} regions, {n_districts} districts, {total:,} assets")


if __name__ == "__main__":
    main()
