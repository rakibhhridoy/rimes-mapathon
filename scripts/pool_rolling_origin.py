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


def _spearman(x, y) -> dict:
    from scipy import stats

    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p)}


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


def record_length(rows: list[dict]) -> dict:
    """New ground falls as the record lengthens, so the record's lead could
    follow the number of earlier years instead, or the size of the flood.
    Partial rank correlations separate each from new ground."""
    from scipy import stats

    lead = stats.rankdata([r["record_minus_model"] for r in rows])
    ground = stats.rankdata([r["new_ground_share"] for r in rows])
    years = stats.rankdata([r["n_earlier_years"] for r in rows])
    # A small flood may fall only on ground that always floods, so the share
    # of assets the flood reached is held fixed as well.
    size = stats.rankdata([r["test_positive_rate"] for r in rows])

    def residual(y, x):
        design = np.column_stack([np.ones(len(x)), x])
        return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]

    out = {}
    for name, a, b, given in (("lead_new_ground_given_years", lead, ground, years),
                              ("lead_years_given_new_ground", lead, years, ground),
                              ("lead_new_ground_given_size", lead, ground, size),
                              ("lead_size_given_new_ground", lead, size, ground)):
        r, p = stats.pearsonr(residual(a, given), residual(b, given))
        out[name] = {"rho": float(r), "p": float(p)}
    out["new_ground_vs_size"] = _spearman(ground, size)
    return out


def by_region(rows: list[dict], draws: int = 5000) -> dict:
    """Tests in one region share its assets and terrain, so they are not
    independent. The relation is checked with each region left out in turn,
    with whole regions resampled, and within each region on its own."""
    from scipy import stats

    regions = sorted({r["region"] for r in rows})
    left_out = []
    for region in regions:
        kept = [r for r in rows if r["region"] != region]
        rho, p = stats.spearmanr([r["new_ground_share"] for r in kept],
                                 [r["record_minus_model"] for r in kept])
        left_out.append({"region": region, "rho": float(rho), "p": float(p)})
    rng = np.random.default_rng(0)
    rhos, crossings = [], []
    for _ in range(draws):
        picked = [r for region in rng.choice(regions, len(regions))
                  for r in rows if r["region"] == region]
        x = np.array([r["new_ground_share"] for r in picked])
        y = np.array([r["record_minus_model"] for r in picked])
        if len(np.unique(x)) < 3:
            continue
        rhos.append(stats.spearmanr(x, y)[0])
        slope, intercept = np.polyfit(x, y, 1)
        if slope < 0:
            crossings.append(-intercept / slope)
    # Within a region the lead should fall from its test with the least new
    # ground to its test with the most.
    within = []
    for region in regions:
        own = sorted((r for r in rows if r["region"] == region),
                     key=lambda r: r["new_ground_share"])
        if len(own) >= 2:
            within.append(own[-1]["record_minus_model"] < own[0]["record_minus_model"])
    return {"left_out": left_out,
            "left_out_rho_range": [min(r["rho"] for r in left_out),
                                   max(r["rho"] for r in left_out)],
            "left_out_p_max": max(r["p"] for r in left_out),
            "resampled_rho_ci95": [float(v) for v in np.percentile(rhos, [2.5, 97.5])],
            "resampled_rho_negative_share": float(np.mean(np.array(rhos) < 0)),
            "resampled_crossing_ci95": [float(v) for v in np.percentile(crossings, [2.5, 97.5])],
            "regions_with_several_tests": len(within),
            "regions_lead_falls": int(sum(within))}


def main() -> None:
    rows = collect()
    by_role = {role: [r for r in rows if r["role"] == role] for role in ("study", "validation")}
    result = {"tests": rows, "all": relation(rows), "record_length": record_length(rows),
              "by_region": by_region(rows),
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
            "map_ahead_significant": sum(r["hazard_map_minus_record"] > 0
                                         and (r["hazard_map_minus_record_p"] or 1) < P_LEVEL
                                         for r in mapped),
            "model_ahead_of_map": sum(r["model_auc"] > r["hazard_map_auc"] for r in mapped),
            # Does the map, like the model, gain on the record where floods reach new ground?
            "lead_over_map_vs_new_ground": _spearman(
                [r["new_ground_share"] for r in mapped],
                [-r["hazard_map_minus_record"] for r in mapped]),
            "auc_mean": float(np.mean([r["hazard_map_auc"] for r in mapped])),
            "auc_min": float(min(r["hazard_map_auc"] for r in mapped)),
            "auc_max": float(max(r["hazard_map_auc"] for r in mapped))}
    both = [r for r in rows if "model_with_record_auc" in r]
    if both:
        result["model_with_record"] = {
            "n_tests": len(both),
            "ahead_of_both": sum(r["model_with_record_auc"] > max(r["record_auc"], r["model_auc"])
                                 for r in both),
            "regions_ahead_of_both": sorted({r["region"] for r in both
                                             if r["model_with_record_auc"]
                                             > max(r["record_auc"], r["model_auc"])})}
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
