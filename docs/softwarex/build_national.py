"""
Write the SoftwareX manuscript's national-partition table from pipeline outputs.

Like docs/build_numbers.py for the other documents, nothing here is typed by
hand: each row reads a region's validation, benchmark and label-comparison
files, so rerunning a region and this script updates the table.

    python docs/softwarex/build_national.py     # -> docs/softwarex/national.tex
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.data.regions import region_paths  # noqa: E402

OUT = Path(__file__).with_name("national.tex")
PARTITION = ROOT / "configs" / "national" / "partition.yaml"


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
    print(f"{OUT}: {len(rows)} regions, {n_districts} districts, {total:,} assets")


if __name__ == "__main__":
    main()
