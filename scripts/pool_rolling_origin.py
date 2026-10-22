"""
Pool the rolling-origin tests of every region and relate the record's lead
to how much of each flood fell on ground no earlier flood had reached.

Each region-year test in rolling_origin.json gives the mean AUC of the record
of earlier flooding and of the model over the block assignments, and the
share of that year's flooded assets standing on new ground. If the record
wins where floods return to the same ground, its lead should fall as that
share rises. The relation is summarised by Spearman's rank correlation, an
ordinary least-squares line with its zero crossing, and intervals from
resampling the tests.

    python scripts/pool_rolling_origin.py     # writes data/pooled/rolling_origin_pooled.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Region -> (role, output directory). The study regions keep their paper
# boxes; the validation regions hold no asset a study box holds.
REGIONS = {
    "rangpur_rajshahi": ("study", "data/output"),
    "sylhet": ("study", "data/sylhet/output"),
    "sw_coastal": ("study", "data/sw_coastal/output"),
    "jamuna_east_ind": ("validation", "data/jamuna_east_ind/output"),
    "central_ind": ("validation", "data/central_ind/output"),
    "chattogram_coast_ind": ("validation", "data/chattogram_coast_ind/output"),
    "west_central_ind": ("validation", "data/west_central_ind/output"),
}
OUT = ROOT / "data" / "pooled" / "rolling_origin_pooled.json"
P_LEVEL = 0.05


def collect() -> list[dict]:
    rows = []
    for region, (role, out) in REGIONS.items():
        path = ROOT / out / "rolling_origin.json"
        if not path.exists():
            print(f"missing: {path}")
            continue
        for t in json.loads(path.read_text())["tests"]:
            s = t["summary"]
            if "model" not in s or "record" not in s:
                continue
            diff = s.get("model_minus_record", {})
            row = {"region": region, "role": role, "test_year": t["test_year"],
                   "n_earlier_years": t["n_earlier_years"], "n_seeds": t["n_seeds"],
                   "new_ground_share": t["new_ground_share"],
                   "test_positive_rate": t["test_positive_rate"],
                   "record_auc": s["record"]["auc_roc_mean"],
                   "model_auc": s["model"]["auc_roc_mean"],
                   "record_minus_model": s["record"]["auc_roc_mean"] - s["model"]["auc_roc_mean"],
                   "p_value": diff.get("p_value"),
                   "skill_over_record": t["skill_over_record"],
                   "record_ass": s["record"].get("area_skill_score_mean"),
                   "model_ass": s["model"].get("area_skill_score_mean")}
            for name in ("model_with_record", "hazard_map", "water_occurrence"):
                if name in s:
                    row[f"{name}_auc"] = s[name]["auc_roc_mean"]
                    row[f"{name}_ass"] = s[name].get("area_skill_score_mean")
                    gap = s.get(f"{name}_minus_record", {})
                    row[f"{name}_minus_record"] = gap.get("mean")
                    row[f"{name}_minus_record_p"] = gap.get("p_value")
            row["verdict"] = ("none" if row["p_value"] is None or row["p_value"] >= P_LEVEL
                              else "record" if row["record_minus_model"] > 0 else "model")
            rows.append(row)
    return rows


def relation(rows: list[dict], draws: int = 5000) -> dict:
    from scipy import stats

    x = np.array([r["new_ground_share"] for r in rows])
    y = np.array([r["record_minus_model"] for r in rows])
    rho, p = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(draws):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) > 2:
            boot.append(np.polyfit(x[i], y[i], 1))
    boot = np.array(boot)
    crossing = -boot[:, 1] / boot[:, 0]
    crossing = crossing[(boot[:, 0] < 0) & np.isfinite(crossing)]
    return {"n_tests": len(rows), "spearman_rho": float(rho), "spearman_p": float(p),
            "slope": float(slope), "intercept": float(intercept),
            "slope_ci95": [float(v) for v in np.percentile(boot[:, 0], [2.5, 97.5])],
            "zero_crossing": float(-intercept / slope) if slope < 0 else None,
            "zero_crossing_ci95": ([float(v) for v in np.percentile(crossing, [2.5, 97.5])]
                                   if len(crossing) > 100 else None)}


def main() -> None:
    rows = collect()
    by_role = {role: [r for r in rows if r["role"] == role] for role in ("study", "validation")}
    result = {"tests": rows, "all": relation(rows),
              "verdicts": {v: sum(r["verdict"] == v for r in rows)
                           for v in ("record", "model", "none")}}
    for role, subset in by_role.items():
        if len(subset) >= 4:
            result[role] = relation(subset)
    mapped = [r for r in rows if "hazard_map_auc" in r]
    if mapped:
        result["hazard_map"] = {
            "n_tests": len(mapped),
            "record_ahead": sum(r["hazard_map_minus_record"] < 0 for r in mapped),
            "record_ahead_significant": sum(r["hazard_map_minus_record"] < 0
                                            and (r["hazard_map_minus_record_p"] or 1) < P_LEVEL
                                            for r in mapped),
            "model_ahead_of_map": sum(r["model_auc"] > r["hazard_map_auc"] for r in mapped),
            "auc_mean": float(np.mean([r["hazard_map_auc"] for r in mapped])),
            "auc_min": float(min(r["hazard_map_auc"] for r in mapped)),
            "auc_max": float(max(r["hazard_map_auc"] for r in mapped))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    for r in rows:
        print(f"{r['region']:22s} {r['test_year']}  new ground {r['new_ground_share']:.2f}  "
              f"record {r['record_auc']:.3f}  model {r['model_auc']:.3f}  "
              f"map {r.get('hazard_map_auc', float('nan')):.3f}  {r['verdict']}")
    a = result["all"]
    print(f"{a['n_tests']} tests: Spearman rho {a['spearman_rho']:.2f} (p={a['spearman_p']:.3f}), "
          f"zero crossing {a['zero_crossing']}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
