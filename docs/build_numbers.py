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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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

# Regions run only to validate the comparison with the flood record, which do
# not appear on the website: prefix and config.
VALIDATION_REGIONS = {
    "jamuna_east_ind": ("Jamuna", "configs/validation/jamuna_east.yaml"),
}


def _validation_paths(config_file: str) -> dict:
    from pipeline.cli import _load_config

    paths = _load_config(str(ROOT / config_file)).get("paths", {}) or {}
    return {kind: ROOT / paths[f"{kind}_dir"] for kind in ("raw", "processed", "output")}


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


class _Fixed(float):
    """A float that prints with a set number of decimals, such as a percentage."""

    def __new__(cls, value, digits):
        obj = super().__new__(cls, value)
        obj.digits = digits
        return obj


def _macro(name: str, value) -> str:
    r"""One \newcommand line; missing values render as a visible \todo.

    Floats keep a fixed number of decimals, trailing zeros included, so that
    figures set side by side read alike: scores to three decimals,
    percentages to one, and lifts over the base rate to one.
    """
    if value is None:
        body = r"\todo{missing}"
    elif isinstance(value, _Fixed):
        body = f"{value:.{value.digits}f}"
    elif isinstance(value, float):
        digits = 1 if name.endswith("Lift") else 3
        body = f"{value:.{digits}f}"
    elif isinstance(value, int):
        body = f"{value:,}"
    else:
        body = str(value)
    if isinstance(value, float) and body.startswith("-"):
        # A true minus sign, which works in text and in mathematics alike.
        body = r"\ensuremath{-}" + body[1:]
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
    return "<0.001" if value < 0.001 else "=" + f"{value:.3f}"


def _gap(value):
    """The size of a signed difference, for sentences that name its direction."""
    return None if value is None else abs(float(value))


def _pct(value):
    return None if value is None else _Fixed(100 * float(value), 1)


def _record_skill(rows, model, record):
    """Skill over the record, (AUC_model - AUC_record) / (1 - AUC_record).

    Like a forecast skill score against persistence, it is the share of the
    record's remaining ranking error that the model removes: zero when the
    model only matches the record, negative when it falls behind. It is taken
    from the mean AUCs over the block assignments, since a ratio per
    assignment swings without bound where the record is nearly perfect.
    """
    pairs = [(r["scores"][model]["auc_roc"], r["scores"][record]["auc_roc"])
             for r in rows if model in r["scores"] and record in r["scores"]]
    if not pairs:
        return None
    m, b = np.mean(pairs, axis=0)
    return float((m - b) / (1 - b)) if b < 1 else None


def _raw_events_path(paths: dict) -> Path:
    return paths["raw"] / "s1_flood_events.json"


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


def region_numbers(region_id: str, prefix: str | None = None,
                   paths: dict | None = None) -> list[str]:
    """Macros for one region, named e.g. \\RRvalAUC."""
    prefix = prefix or REGION_MACRO[region_id]
    paths = paths or region_paths(region_id)
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
        # PyKrige fits on longitude and latitude, so the range is in degrees;
        # at 24 degrees N one degree spans about 102 km east-west and 111 km
        # north-south, and 106 km is taken between them, to the nearest km.
        _macro(f"{prefix}VariogramRangeKm",
               int(round(variogram["range"] * 106)) if variogram.get("range") else None),
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
            spread[key] = float(data.std())
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
                   stats["ap_lift_mean"] if stats.get("ap_lift_mean") else None),
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

    # Permutation importance of the default model on the held-out blocks.
    imp = (_read_json(paths["output"] / "feature_importance.json") or {}).get("summary") or {}
    ifeat, igroup = imp.get("features") or {}, imp.get("groups") or {}

    def _loss(table, key):
        return (table.get(key) or {}).get("auc_loss_mean")

    other_terrain = [_loss(ifeat, k) for k in ("slope", "twi", "hand", "flow_acc")]
    other_terrain = [v for v in other_terrain if v is not None]
    lines += [
        _macro(f"{prefix}ImpElevation", _loss(ifeat, "elevation")),
        _macro(f"{prefix}ImpOtherTerrainMax", max(other_terrain) if other_terrain else None),
        _macro(f"{prefix}ImpTerrain", _loss(igroup, "terrain")),
        _macro(f"{prefix}ImpAccess", _loss(igroup, "access")),
        _macro(f"{prefix}ImpPopulation", _loss(igroup, "population")),
        _macro(f"{prefix}ImpAssetType", _loss(igroup, "asset_type")),
    ]

    # The temporal test on stricter test sets, against repeated radar errors.
    persist = (_read_json(paths["output"] / "persistence_check.json") or {}).get("summary") or {}
    for key, tag in (("not_always_flooded", "Always"), ("outside_recorded_water", "Water")):
        sub = persist.get(key) or {}
        diff = sub.get("temporal_minus_past") or {}
        lines += [
            _macro(f"{prefix}Persist{tag}AssetShare", _pct(sub.get("share_of_assets"))),
            _macro(f"{prefix}Persist{tag}PosShare", _pct(sub.get("share_of_late_positives"))),
            _macro(f"{prefix}Persist{tag}ModelAUC", sub.get("temporal_auc_mean")),
            _macro(f"{prefix}Persist{tag}RecordAUC", sub.get("past_flooding_auc_mean")),
            _macro(f"{prefix}Persist{tag}Gap", _gap(diff.get("mean"))),
            _macro(f"{prefix}Persist{tag}P", _p(diff.get("p_value"))),
            _macro(f"{prefix}Persist{tag}Ahead", diff.get("seeds_temporal_ahead")),
            _macro(f"{prefix}Persist{tag}RecordAhead",
                   diff["n_seeds"] - diff["seeds_temporal_ahead"] if diff else None),
        ]

    # Held-out results with larger blocks, on the same seeds.
    blocks = (_read_json(paths["output"] / "block_size_check.json") or {}).get("sizes") or {}
    for size in ("10000", "20000", "40000"):
        entry = blocks.get(size) or {}
        diff = entry.get("temporal_minus_past") or {}
        # LaTeX command names cannot hold digits, so the width is spelt out.
        tag = "Block" + {"10000": "Ten", "20000": "Twenty", "40000": "Forty"}[size]
        lines += [
            _macro(f"{prefix}{tag}SpatialAUC", entry.get("spatial_auc_mean")),
            _macro(f"{prefix}{tag}Seeds", entry.get("n_seeds")),
            _macro(f"{prefix}{tag}ModelAUC", entry.get("temporal_auc_mean")),
            _macro(f"{prefix}{tag}RecordAUC", entry.get("past_flooding_auc_mean")),
            _macro(f"{prefix}{tag}TempGap", _gap(diff.get("mean"))),
            _macro(f"{prefix}{tag}TempP", _p(diff.get("p_value"))),
            _macro(f"{prefix}{tag}TempAhead", diff.get("seeds_temporal_ahead")),
            _macro(f"{prefix}{tag}TempSeeds", diff.get("n_seeds")),
        ]

    # The temporal test with the model trained on fairer targets.
    tv = (_read_json(paths["output"] / "target_variants.json") or {}).get("summary") or {}
    for key, tag in (("any_earlier", "Any"), ("flood_fraction", "Frac")):
        sub = tv.get(key) or {}
        diff = sub.get("minus_past") or {}
        lines += [
            _macro(f"{prefix}Target{tag}AUC", sub.get("auc_mean")),
            _macro(f"{prefix}Target{tag}Gap", _gap(diff.get("mean"))),
            _macro(f"{prefix}Target{tag}P", _p(diff.get("p_value"))),
            _macro(f"{prefix}Target{tag}Ahead", diff.get("seeds_model_ahead")),
        ]

    # The temporal test on masks re-mapped from Sentinel-1A alone, and the
    # flooded share of each event (scripts/remap_s1a.py).
    sens = paths["output"] / "sensitivity_s1a"
    single = (_read_json(sens / "temporal_holdout.json") or {}).get("summary") or {}
    sdiff = single.get("temporal_minus_past_flooding") or {}
    lines += [
        _macro(f"{prefix}SingleSatModelAUC", (single.get("temporal") or {}).get("auc_mean")),
        _macro(f"{prefix}SingleSatRecordAUC", (single.get("past_flooding") or {}).get("auc_mean")),
        _macro(f"{prefix}SingleSatGap", _gap(sdiff.get("mean"))),
        _macro(f"{prefix}SingleSatP", _p(sdiff.get("p_value"))),
    ]
    extents = _read_json(sens / "event_extents.json") or {}
    cfg_events = {e["name"]: int(e["start"][:4]) for e in
                  (_read_json(_raw_events_path(paths)) or {}).get("events", [])}
    early = [v["published"]["flooded_share"] for k, v in extents.items() if cfg_events.get(k, 9999) <= 2022]
    late = [v["published"]["flooded_share"] for k, v in extents.items() if cfg_events.get(k, 0) > 2022]
    shrink = [1 - v["s1a"]["flooded_share"] / v["published"]["flooded_share"]
              for v in extents.values() if v["published"]["flooded_share"] > 0]
    lines += [
        _macro(f"{prefix}EarlyExtentMin", _pct(min(early)) if early else None),
        _macro(f"{prefix}EarlyExtentMax", _pct(max(early)) if early else None),
        _macro(f"{prefix}LateExtentMin", _pct(min(late)) if late else None),
        _macro(f"{prefix}LateExtentMax", _pct(max(late)) if late else None),
        _macro(f"{prefix}SingleSatShrinkMax", _pct(max(shrink)) if shrink else None),
    ]

    # Where the published single assignment (the default seed) ranks among
    # the repeated ones, by the AUC of the observed-label model.
    lc_rows = (_read_json(paths["output"] / "label_comparison.json") or {}).get("per_seed") or []
    aucs = {r["seed"]: r["labels"]["observed"]["auc_roc"] for r in lc_rows if "observed" in r["labels"]}
    default_seed = 42
    lines.append(_macro(f"{prefix}DefaultSeedRank",
                        1 + sum(a > aucs[default_seed] for a in aucs.values())
                        if default_seed in aucs else None))

    # The terrain hazard surface scored only on held-out blocks.
    hz = (_read_json(paths["output"] / "hazard_heldout.json") or {}).get("summary") or {}
    lines += [
        _macro(f"{prefix}HazardHeldoutAUC", (hz.get("heldout_auc") or {}).get("mean")),
        _macro(f"{prefix}HazardHeldoutSD", (hz.get("heldout_auc") or {}).get("sd")),
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

    # A validation region's assets, after those shared with a study box are dropped.
    vreg = _read_json(paths["output"] / "validation_region.json")
    if vreg:
        lines += [_macro(f"{prefix}SharedAssets", vreg.get("assets_shared_with_study_box")),
                  _macro(f"{prefix}IndAssets", vreg.get("assets_kept"))]

    # Skill over the record, and the record checks (length and planning lists).
    temporal_rows = (_read_json(paths["output"] / "temporal_holdout.json") or {}).get("per_seed") or []
    lines += [
        _macro(f"{prefix}SkillModel", _record_skill(temporal_rows, "temporal", "past_flooding")),
        _macro(f"{prefix}SkillPastFeat", _record_skill(pastf.get("per_seed") or [], "with_past", "past_only")),
    ]
    rc = _read_json(paths["output"] / "record_checks.json") or {}
    by_k = (rc.get("record_length") or {}).get("by_k") or {}
    for k, word in ((1, "One"), (2, "Two"), (3, "Three")):
        row = by_k.get(str(k))
        if row is None:             # Sylhet has two events before the cutoff
            continue
        lines += [
            _macro(f"{prefix}RecordLen{word}Rec", row.get("record_auc")),
            _macro(f"{prefix}RecordLen{word}Model", row.get("model_auc")),
        ]
    plan = rc.get("planning") or {}
    lines.append(_macro(f"{prefix}PlanListShare", _pct(plan.get("listed_share"))))
    boot = plan.get("block_bootstrap") or {}
    gap = boot.get("model_minus_record") or {}
    both = boot.get("model_with_record_minus_record") or {}
    lines += [
        _macro(f"{prefix}OofModelAUC", plan.get("model_auc_out_of_fold")),
        _macro(f"{prefix}OofRecordAUC", plan.get("record_auc_region")),
        _macro(f"{prefix}OofBothAUC", plan.get("model_with_record_auc_out_of_fold")),
        _macro(f"{prefix}OofGapLow", (gap.get("ci95") or [None, None])[0]),
        _macro(f"{prefix}OofGapHigh", (gap.get("ci95") or [None, None])[1]),
        _macro(f"{prefix}OofBothGapLow", (both.get("ci95") or [None, None])[0]),
        _macro(f"{prefix}OofBothGapHigh", (both.get("ci95") or [None, None])[1]),
    ]
    curve = plan.get("by_list_share") or {}
    for share, word in (("0.05", "Five"), ("0.10", "Ten"), ("0.20", "Twenty"), ("0.30", "Thirty")):
        row = curve.get(share) or {}
        lines += [
            _macro(f"{prefix}List{word}Rec", _pct(row.get("record"))),
            _macro(f"{prefix}List{word}Model", _pct(row.get("model"))),
            _macro(f"{prefix}List{word}Both", _pct(row.get("model_with_record"))),
        ]
    for g, word in (("all", "All"), ("facilities", "Facil"), ("transport", "Trans")):
        e = (plan.get("groups") or {}).get(g) or {}
        lines += [
            _macro(f"{prefix}Plan{word}Flooded", e.get("n_flooded")),
            _macro(f"{prefix}Plan{word}Record", _pct(e.get("record_caught_share"))),
            _macro(f"{prefix}Plan{word}Model", _pct(e.get("model_caught_share"))),
            _macro(f"{prefix}Plan{word}Both", _pct(e.get("model_with_record_caught_share"))),
            _macro(f"{prefix}Plan{word}ModelOnly", e.get("model_only")),
            _macro(f"{prefix}Plan{word}RecordOnly", e.get("record_only")),
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


def _shared_p_values(lines: list[str]) -> list[str]:
    r"""p-values of one comparison across the three flood regions, in a single
    macro: "all $p<0.001$" when every region gives the same value, otherwise
    the three in region order."""
    defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", "\n".join(lines), re.M))
    out = []
    for stem in ("LabelGapP", "TempProxyGapP"):
        values = [defined.get(f"{prefix}{stem}") for prefix in ("RR", "Sylhet", "Coastal")]
        if any(v is None or "todo" in v for v in values):
            out.append(_macro(f"All{stem}", None))
        elif len(set(values)) == 1:
            out.append(_macro(f"All{stem}", f"all $p{values[0]}$"))
        else:
            out.append(_macro(f"All{stem}", f"$p{values[0]}$, $p{values[1]}$ and $p{values[2]}$"))
    return out


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
    # The papers describe the regions that have a macro prefix; regions added
    # since (docs/national/) appear on the website, not in these documents.
    for region_id in (r for r in REGION_CONFIGS if r in REGION_MACRO):
        lines += region_numbers(region_id) + [""]
    for region_id, (prefix, config_file) in VALIDATION_REGIONS.items():
        if (ROOT / config_file).exists():
            lines += region_numbers(region_id, prefix, _validation_paths(config_file)) + [""]
    lines += _shared_p_values(lines) + [""]

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
