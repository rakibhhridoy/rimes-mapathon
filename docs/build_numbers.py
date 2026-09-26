"""
Extract every reported figure from pipeline outputs into LaTeX macros.

The technical document and the paper must never hardcode a number that the
pipeline computes: the earlier draft quoted an AUC of 0.93 and a "79% of the
CHT at high susceptibility" that came from other people's papers, and stale
figures are how a document ends up describing a model that no longer exists.

Every macro is written from a file in data/*/output, and any figure without a
source becomes \\todo{...} so it is visible in the rendered PDF instead of
silently wrong.

Usage:
    python docs/build_numbers.py                  # all regions
    python docs/build_numbers.py --out docs/numbers.tex
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.data.regions import REGION_CONFIGS, region_paths  # noqa: E402

# LaTeX macro names must be letters only, so region ids map to camel case.
REGION_MACRO = {
    "rangpur_rajshahi": "RR",
    "sylhet": "Sylhet",
    "sw_coastal": "Coastal",
    "cht": "CHT",
}


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _macro(name: str, value) -> str:
    r"""One \newcommand line; missing values render as a visible \todo."""
    if value is None:
        body = r"\todo{missing}"
    elif isinstance(value, float):
        body = f"{value:.3f}".rstrip("0").rstrip(".")
    elif isinstance(value, int):
        body = f"{value:,}"
    else:
        body = str(value)
    return rf"\newcommand{{\{name}}}{{{body}}}"


def _nugget_fraction(variogram: dict):
    if not variogram or variogram.get("nugget") is None:
        return None
    if variogram.get("nugget_fraction") is not None:
        return variogram["nugget_fraction"]
    partial = variogram.get("partial_sill", variogram.get("sill"))
    if partial is None:
        return None
    total = partial + variogram["nugget"]
    return variogram["nugget"] / total if total > 0 else None


def _p(value):
    """A p-value with its relation, for use as $p\\Macro{}$: "=0.04" or "<0.001"."""
    if value is None:
        return None
    value = float(value)
    return "<0.001" if value < 0.001 else "=" + f"{value:.3f}".rstrip("0").rstrip(".")


def _gap(value):
    """The size of a signed difference, for sentences that name its direction."""
    return None if value is None else abs(float(value))


def _pct(value):
    return None if value is None else round(100 * float(value), 1)


def _mapped_cell_share(output_dir: Path):
    """Share of grid cells holding any mapped asset, where composite risk is defined."""
    path = output_dir / "risk_grid.geojson"
    if not path.exists():
        return None
    try:
        import pyogrio
        grid = pyogrio.read_dataframe(str(path), columns=["exposure"],
                                      read_geometry=False)
    except Exception:
        return None
    return float((grid["exposure"] > 0).mean()) if len(grid) else None


def region_numbers(region_id: str) -> list[str]:
    """Macros for one region, named e.g. \\RRvalAUC."""
    prefix = REGION_MACRO[region_id]
    paths = region_paths(region_id)
    lines = [f"% ---- {region_id} ----"]

    meta = _read_json(paths["output"] / "pipeline_metadata.json") or {}
    stats = meta.get("risk_score_stats") or {}
    validation = meta.get("gnn_validation") or {}
    weights = meta.get("vulnerability_weights") or {}
    variogram = meta.get("variogram_params") or {}

    lines += [
        _macro(f"{prefix}Assets", stats.get("n_assets")),
        _macro(f"{prefix}MappedCellShare", _pct(_mapped_cell_share(paths["output"]))),
        _macro(f"{prefix}LabelRate", _pct(meta.get("label_positive_rate"))),
        _macro(f"{prefix}ValAUC", validation.get("val_auc_roc")),
        _macro(f"{prefix}ValAP", validation.get("val_average_precision")),
        _macro(f"{prefix}ValAPLift", validation.get("val_ap_lift")),
        _macro(f"{prefix}ValBrier", validation.get("val_brier")),
        _macro(f"{prefix}ValBrierCalibrated", validation.get("val_brier_calibrated")),
        _macro(f"{prefix}ValBrierBaseRate", validation.get("val_brier_base_rate")),
        _macro(f"{prefix}NCalibration", validation.get("n_calibration")),
        _macro(f"{prefix}NTrain", validation.get("n_train")),
        _macro(f"{prefix}NVal", validation.get("n_val")),
        _macro(f"{prefix}ScoreMean", stats.get("mean")),
        _macro(f"{prefix}ScoreMedian", stats.get("median")),
        _macro(f"{prefix}KrigingVar", meta.get("kriging_variance_mean")),
        _macro(f"{prefix}KrigingSillRatio",
               meta.get("kriging_variance_sill_ratio")),
        _macro(f"{prefix}VariogramRange", variogram.get("range")),
        # Nugget as a fraction of the full sill. Older variogram files stored
        # the partial sill as "sill"; rebuild them from partial_sill when the
        # newer key is present, otherwise treat "sill" as partial.
        _macro(f"{prefix}NuggetSillRatio", _nugget_fraction(variogram)),
        _macro(f"{prefix}Processed", (meta.get("generated_at") or "")[:10] or None),
        _macro(f"{prefix}NVulnComponents", len(weights) or None),
    ]

    # Observed-flood validation, when Sentinel-1 extents have been mapped.
    observed = _read_json(paths["output"] / "validation_metrics.json") or {}
    assets = observed.get("assets_vs_observed") or {}
    held = assets.get("held_out_blocks") or {}
    proxy = observed.get("proxy_vs_observed") or {}
    lines += [
        _macro(f"{prefix}ObsEvents", assets.get("n_events")),
        _macro(f"{prefix}ObsFloodedShare", _pct(assets.get("observed_flooded_share"))),
        _macro(f"{prefix}ObsAUC", held.get("auc_roc")),
        _macro(f"{prefix}ObsAPLift", held.get("ap_lift")),
        _macro(f"{prefix}ObsBrierCalibrated",
               (observed.get("assets_vs_observed") or {}).get(
                   "held_out_blocks_calibrated", {}).get("brier")),
        _macro(f"{prefix}ProxyPOD", proxy.get("pod")),
        _macro(f"{prefix}ProxyFAR", proxy.get("far")),
        _macro(f"{prefix}ProxyCSI", proxy.get("csi")),
        _macro(f"{prefix}ProxyKappa", proxy.get("cohen_kappa")),
        _macro(f"{prefix}ProxyShare", _pct(proxy.get("proxy_positive_share"))),
        _macro(f"{prefix}ObservedShare", _pct(proxy.get("observed_positive_share"))),
        _macro(f"{prefix}MinEvents", assets.get("min_events")),
    ]

    # Hazard surfaces against observed flooding, across the whole grid.
    surfaces = observed.get("hazard_surface_vs_observed") or {}
    for key in ("kriged", "terrain"):
        ranking = (surfaces.get(key) or {}).get("ranking") or {}
        lines.append(_macro(f"{prefix}Surface{key.title()}ObsAUC", ranking.get("auc_roc")))

    # The same validation for the model trained on proxy labels, kept from
    # before the switch to observed labels: the comparison the paper turns on.
    before = _read_json(paths["output"] / "validation_metrics_proxy_trained.json") or {}
    before_held = (before.get("assets_vs_observed") or {}).get("held_out_blocks") or {}
    lines += [
        _macro(f"{prefix}ProxyTrainedObsAUC", before_held.get("auc_roc")),
        _macro(f"{prefix}ProxyTrainedObsAPLift", before_held.get("ap_lift")),
    ]

    # Label prevalence in the calibration and test blocks. They can differ
    # severalfold, which limits how far the calibrated probabilities transfer.
    calib_rate = test_rate = None
    graph_path = paths["output"] / "spatial_graph.pt"
    if graph_path.exists():
        import torch

        graph = torch.load(graph_path, weights_only=False)
        y = graph.y.numpy()
        if "calib_mask" in graph and "test_mask" in graph:
            calib_rate = float(y[graph.calib_mask.numpy().astype(bool)].mean())
            test_rate = float(y[graph.test_mask.numpy().astype(bool)].mean())
    lines += [
        _macro(f"{prefix}CalibBlockRate", _pct(calib_rate)),
        _macro(f"{prefix}TestBlockRate", _pct(test_rate)),
    ]

    # Spread of each hazard surface across the grid: how much spatial detail
    # it carries. Read at reduced resolution; the standard deviation of a
    # 30 m surface is stable under an 8x decimation.
    spread = {}
    for key, name in (("Kriged", "flood_risk_kriged.tif"),
                      ("Terrain", "flood_hazard_terrain.tif")):
        path = paths["output"] / name
        spread[key] = None
        if path.exists():
            import rasterio

            with rasterio.open(path) as src:
                step = max(1, min(src.width, src.height) // 1500)
                data = src.read(1, masked=True, out_shape=(
                    src.height // step, src.width // step))
            spread[key] = round(float(data.std()), 2)
    lines += [
        _macro(f"{prefix}KrigedGridSD", spread["Kriged"]),
        _macro(f"{prefix}TerrainGridSD", spread["Terrain"]),
    ]

    # Terrain hazard model (the alternative to kriging).
    hazard = _read_json(paths["output"] / "hazard_model.json") or {}
    lines += [
        _macro(f"{prefix}HazardAUC", hazard.get("val_auc_roc")),
        _macro(f"{prefix}HazardAP", hazard.get("val_average_precision")),
    ]
    for feature, coefficient in (
            hazard.get("standardised_coefficients") or {}).items():
        lines.append(_macro(f"{prefix}HazCoef{feature.title().replace('_', '')}",
                            coefficient))

    # Landslide regions report a fitted model instead of a flood model.
    landslide = _read_json(paths["output"] / "landslide_model.json") or {}
    inventory = landslide.get("inventory") or {}
    ls_val = landslide.get("validation") or {}
    if inventory or ls_val:
        lines += [
            _macro(f"{prefix}Landslides", inventory.get("n_landslides")),
            _macro(f"{prefix}Background", inventory.get("n_background")),
            _macro(f"{prefix}LSAUC", ls_val.get("val_auc_roc")),
            _macro(f"{prefix}LSAP", ls_val.get("val_average_precision")),
            _macro(f"{prefix}LSPositiveRate", _pct(ls_val.get("val_positive_rate"))),
        ]
        for feature, coefficient in (
                landslide.get("standardised_coefficients") or {}).items():
            lines.append(_macro(f"{prefix}Coef{feature.title().replace('_', '')}",
                                coefficient))

    # Benchmark: the graph model against tabular baselines over repeated
    # block assignments. Point estimates from one split hid both the spread
    # and the fact that the baselines win.
    bench = _read_json(paths["output"] / "benchmark.json") or {}
    bsum = bench.get("summary") or {}
    NAMES = {"graph_sage": "Graph", "logistic_regression": "Logistic",
             "random_forest": "Forest", "gradient_boosting": "Boosting",
             "twi_only": "Twi"}
    for key, macro in NAMES.items():
        stats = bsum.get(key) or {}
        lines += [
            _macro(f"{prefix}Bench{macro}AUC", stats.get("auc_mean")),
            _macro(f"{prefix}Bench{macro}SD", stats.get("auc_sd")),
            _macro(f"{prefix}Bench{macro}Lift",
                   round(stats["ap_lift_mean"], 2) if stats.get("ap_lift_mean") else None),
        ]
    paired = bsum.get("graph_vs_best_baseline") or {}
    best = paired.get("best_baseline")
    lines += [
        _macro(f"{prefix}BenchBest", {"random_forest": "random forest",
                                      "gradient_boosting": "gradient boosting",
                                      "logistic_regression": "logistic regression"}.get(best, best)),
        _macro(f"{prefix}BenchDiff", paired.get("auc_difference_mean")),
        _macro(f"{prefix}BenchGap",
               abs(paired["auc_difference_mean"]) if paired.get("auc_difference_mean") else None),
        _macro(f"{prefix}BenchDiffP", _p(paired.get("p_value"))),
        _macro(f"{prefix}BenchAhead", paired.get("seeds_graph_ahead")),
        _macro(f"{prefix}BenchSeeds", paired.get("n_seeds")),
    ]

    # Label comparison: the deployed model trained on proxy labels against
    # the same model trained on observed extents, over the same assignments.
    labels = _read_json(paths["output"] / "label_comparison.json") or {}
    lsum = labels.get("summary") or {}
    gap = lsum.get("observed_minus_proxy") or {}
    lines += [
        _macro(f"{prefix}LabelObsAUC", (lsum.get("observed") or {}).get("auc_mean")),
        _macro(f"{prefix}LabelObsSD", (lsum.get("observed") or {}).get("auc_sd")),
        _macro(f"{prefix}LabelProxyAUC", (lsum.get("proxy") or {}).get("auc_mean")),
        _macro(f"{prefix}LabelProxySD", (lsum.get("proxy") or {}).get("auc_sd")),
        _macro(f"{prefix}LabelGap", gap.get("mean")),
        _macro(f"{prefix}LabelGapP", _p(gap.get("p_value"))),
        _macro(f"{prefix}LabelAhead", gap.get("seeds_observed_ahead")),
    ]

    # Temporal hold-out: trained on the earlier events, scored against the
    # later ones on held-out blocks, beside the past-flooding reference.
    temporal = _read_json(paths["output"] / "temporal_holdout.json") or {}
    tsum = temporal.get("summary") or {}
    tpast = tsum.get("temporal_minus_past_flooding") or {}
    tall = tsum.get("temporal_minus_all_events") or {}
    tproxy = tsum.get("temporal_minus_proxy_trained") or {}
    lines += [
        _macro(f"{prefix}TempAUC", (tsum.get("temporal") or {}).get("auc_mean")),
        _macro(f"{prefix}TempSD", (tsum.get("temporal") or {}).get("auc_sd")),
        _macro(f"{prefix}TempLift", (tsum.get("temporal") or {}).get("ap_lift_mean")),
        _macro(f"{prefix}TempAllAUC", (tsum.get("all_events") or {}).get("auc_mean")),
        _macro(f"{prefix}TempPastAUC", (tsum.get("past_flooding") or {}).get("auc_mean")),
        _macro(f"{prefix}TempPastSD", (tsum.get("past_flooding") or {}).get("auc_sd")),
        _macro(f"{prefix}TempPastLift", (tsum.get("past_flooding") or {}).get("ap_lift_mean")),
        _macro(f"{prefix}TempMinusPast", tpast.get("mean")),
        _macro(f"{prefix}TempMinusPastP", _p(tpast.get("p_value"))),
        _macro(f"{prefix}TempAheadPast", tpast.get("seeds_temporal_ahead")),
        _macro(f"{prefix}TempMinusAll", tall.get("mean")),
        _macro(f"{prefix}TempMinusAllP", _p(tall.get("p_value"))),
        _macro(f"{prefix}TempPastGap", _gap(tpast.get("mean"))),
        _macro(f"{prefix}TempAllGap", _gap(tall.get("mean"))),
        _macro(f"{prefix}TempSeeds", (tsum.get("temporal") or {}).get("n_seeds")),
        _macro(f"{prefix}TempTestRate", _pct(temporal.get("test_positive_rate"))),
        # The label comparison on floods neither label source was trained on.
        _macro(f"{prefix}TempProxyAUC", (tsum.get("proxy_trained") or {}).get("auc_mean")),
        _macro(f"{prefix}TempProxySD", (tsum.get("proxy_trained") or {}).get("auc_sd")),
        _macro(f"{prefix}TempProxyGap", tproxy.get("mean")),
        _macro(f"{prefix}TempProxyGapP", _p(tproxy.get("p_value"))),
        _macro(f"{prefix}TempProxyAhead", tproxy.get("seeds_temporal_ahead")),
    ]

    # Sentinel-1 masks against the Global Flood Database, one event per region.
    cross = (_read_json(paths["output"] / "flood_crosscheck.json") or {}).get("events") or {}
    xc = next(iter(cross.values()), {})
    lines += [
        _macro(f"{prefix}XcCells", xc.get("n_cells")),
        _macro(f"{prefix}XcSOneShare", _pct(xc.get("s1_flooded_share"))),
        _macro(f"{prefix}XcGfdShare", _pct(xc.get("gfd_flooded_share"))),
        _macro(f"{prefix}XcPOD", xc.get("pod")),
        _macro(f"{prefix}XcFAR", xc.get("far")),
        _macro(f"{prefix}XcCSI", xc.get("csi")),
        _macro(f"{prefix}XcKappa", xc.get("cohen_kappa")),
        _macro(f"{prefix}XcAgree", _pct(xc.get("agreement"))),
        _macro(f"{prefix}XcAUC", xc.get("auc_roc")),
    ]

    # Past flooding as a feature, trained on the latest pre-cutoff event.
    pastf = _read_json(paths["output"] / "past_flooding_feature.json") or {}
    psum = pastf.get("summary") or {}
    pvp = psum.get("with_past_minus_past_only") or {}
    pvw = psum.get("with_past_minus_without_past") or {}
    lines += [
        _macro(f"{prefix}PastFeatAUC", (psum.get("with_past") or {}).get("auc_mean")),
        _macro(f"{prefix}PastFeatSD", (psum.get("with_past") or {}).get("auc_sd")),
        _macro(f"{prefix}PastFeatWithoutAUC", (psum.get("without_past") or {}).get("auc_mean")),
        _macro(f"{prefix}PastFeatOnlyAUC", (psum.get("past_only") or {}).get("auc_mean")),
        _macro(f"{prefix}PastFeatOverPast", pvp.get("mean")),
        _macro(f"{prefix}PastFeatOverPastP", _p(pvp.get("p_value"))),
        _macro(f"{prefix}PastFeatOverPastAhead", pvp.get("seeds_ahead")),
        _macro(f"{prefix}PastFeatOverWithout", pvw.get("mean")),
        _macro(f"{prefix}PastFeatOverWithoutP", _p(pvw.get("p_value"))),
        _macro(f"{prefix}PastFeatUnseenAUC", (psum.get("with_past_on_unseen_ground") or {}).get("auc_roc")),
        _macro(f"{prefix}PastFeatUnseenShare", _pct((psum.get("with_past_on_unseen_ground") or {}).get("share_of_assets"))),
        _macro(f"{prefix}PastFeatUnseenPos", _pct((psum.get("with_past_on_unseen_ground") or {}).get("share_of_positives"))),
    ]

    # Weight sensitivity of the composite risk.
    sens = _read_json(paths["output"] / "sensitivity.json") or {}
    scenarios = sens.get("scenarios") or {}
    random_draws = sens.get("random_perturbation") or {}
    def _scenario(name, field):
        return (scenarios.get(name) or {}).get(field)
    lines += [
        _macro(f"{prefix}SensEqualTypesRho", _scenario("exposure:equal_types", "spearman")),
        _macro(f"{prefix}SensPopOnlyRho", _scenario("vulnerability:population_only", "spearman")),
        _macro(f"{prefix}SensExposureDoubleRho", _scenario("factors:exposure_double", "spearman")),
        _macro(f"{prefix}SensExposureDoubleTop", _scenario("factors:exposure_double", "top_decile_overlap")),
        _macro(f"{prefix}SensRandomRho", random_draws.get("spearman_mean")),
        _macro(f"{prefix}SensRandomRhoMin", random_draws.get("spearman_min")),
        _macro(f"{prefix}SensRandomTop", random_draws.get("top_decile_overlap_mean")),
        _macro(f"{prefix}SensDraws", random_draws.get("n_draws")),
    ]

    # Administrative counts, straight from the exported summaries.
    for level in ("union", "upazila", "district"):
        csv_path = paths["output"] / f"{level}_risk_summary.csv"
        count = scored = None
        if csv_path.exists():
            import csv

            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            count = len(rows)
            scored = sum(1 for r in rows
                         if (r.get("has_data") or "").strip().lower() == "true")
        lines.append(_macro(f"{prefix}N{level.title()}s", count))
        lines.append(_macro(f"{prefix}N{level.title()}sScored", scored))

    return lines


def build(out_path: Path) -> Path:
    lines = [
        "% Generated by docs/build_numbers.py — do not edit by hand.",
        f"% Built {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "%",
        "% Any figure the pipeline did not produce renders as \\todo{missing},",
        "% so a gap is visible in the PDF rather than filled from memory.",
        r"\providecommand{\todo}[1]{\textbf{[TODO: #1]}}",
        "",
    ]
    for region_id in REGION_CONFIGS:
        lines += region_numbers(region_id) + [""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    missing = sum(1 for line in lines if r"\todo{missing}" in line)
    defined = sum(1 for line in lines if line.startswith(r"\newcommand"))
    print(f"Wrote {out_path} — {defined} macros, {missing} still missing")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "docs" / "numbers.tex"))
    args = parser.parse_args()
    build(Path(args.out))
