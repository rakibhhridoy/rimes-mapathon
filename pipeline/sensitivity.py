"""
How much the composite risk depends on weights that were chosen, not measured.

The exposure weights per asset type, the vulnerability component weights and
the equal treatment of hazard, exposure and vulnerability in the geometric
mean are all judgements. This module recomputes the composite under other
defensible choices and reports how far the ranking moves, because a ranking
that survives its own weights can be acted on and one that does not cannot.

Output (per region): sensitivity.json.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def _top_overlap(reference: np.ndarray, other: np.ndarray, share=0.10) -> float:
    """Share of the reference's top cells that the variant also ranks top."""
    k = max(1, int(round(share * len(reference))))
    ref_top = set(np.argsort(-reference)[:k])
    other_top = set(np.argsort(-other)[:k])
    return len(ref_top & other_top) / k


def _weighted_geometric_mean(factors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    total = float(sum(weights))
    out = np.ones_like(factors[0], dtype=float)
    for factor, weight in zip(factors, weights):
        out *= np.power(np.clip(factor, 0.0, None), weight / total)
    return out


def _exposure_scenarios(cfg: dict) -> dict:
    published = dict(cfg["risk"]["exposure_weights"])
    equal = {k: 0.15 for k in published}
    lifeline = dict(published)
    lifeline.update({"hospital_school_presence": 0.40,
                     "embankment_shelter_presence": 0.25})
    return {"published": published, "equal_types": equal, "lifeline_heavy": lifeline}


def _vulnerability_scenarios(cfg: dict) -> dict:
    published = dict(cfg["risk"]["vulnerability_weights"])
    applied = [k for k, v in published.items() if v and v > 0]
    equal = {k: (1.0 / len(applied) if k in applied else 0.0) for k in published}
    population = {k: (1.0 if k == "population_density" else 0.0) for k in published}
    return {"published": published, "equal_components": equal,
            "population_only": population}


def run_sensitivity(cfg: dict, output_dir: Path, infra_path: Path,
                    n_draws: int = 20, seed: int = 0) -> dict:
    """Recompute the composite under other weights and report rank movement."""
    import geopandas as gpd

    from pipeline.risk_score import (compute_exposure_grid,
                                     compute_vulnerability_grid)

    grid = gpd.read_file(output_dir / "risk_grid.geojson")
    infra = gpd.read_file(str(infra_path))
    hazard = grid["hazard"].to_numpy(dtype=float)
    published_risk = grid["composite_risk"].to_numpy(dtype=float)

    # Cells with no mapped exposure carry no composite, so they would dominate
    # any rank correlation with ties. Only scored cells are compared.
    scored = published_risk > 0
    logger.info(f"{scored.sum():,} scored cells of {len(grid):,}")

    def composite(exposure_weights, vulnerability_weights, exponents=(1, 1, 1)):
        local = json.loads(json.dumps(cfg))          # deep copy of plain data
        local["risk"]["exposure_weights"] = exposure_weights
        local["risk"]["vulnerability_weights"] = vulnerability_weights
        exposure = compute_exposure_grid(infra, grid, local)
        vulnerability, _ = compute_vulnerability_grid(grid, infra, local)
        return _weighted_geometric_mean(
            [hazard, np.asarray(exposure), np.asarray(vulnerability)],
            list(exponents))

    exposure_sets = _exposure_scenarios(cfg)
    vulnerability_sets = _vulnerability_scenarios(cfg)
    reference = composite(exposure_sets["published"], vulnerability_sets["published"])

    scenarios = {}
    for name, weights in exposure_sets.items():
        if name == "published":
            continue
        scenarios[f"exposure:{name}"] = composite(weights, vulnerability_sets["published"])
    for name, weights in vulnerability_sets.items():
        if name == "published":
            continue
        scenarios[f"vulnerability:{name}"] = composite(exposure_sets["published"], weights)
    for name, exponents in {"hazard_double": (2, 1, 1), "exposure_double": (1, 2, 1),
                            "vulnerability_double": (1, 1, 2)}.items():
        scenarios[f"factors:{name}"] = composite(
            exposure_sets["published"], vulnerability_sets["published"], exponents)

    results = {"n_cells": int(len(grid)), "n_scored_cells": int(scored.sum()),
               "scenarios": {}}
    for name, variant in scenarios.items():
        results["scenarios"][name] = {
            "spearman": _spearman(reference[scored], variant[scored]),
            "top_decile_overlap": _top_overlap(reference[scored], variant[scored]),
        }
        logger.info(f"{name}: rho {results['scenarios'][name]['spearman']:.3f}, "
                    f"top-decile overlap "
                    f"{results['scenarios'][name]['top_decile_overlap']:.2f}")

    # Random perturbation of every chosen weight, which asks the blunter
    # question: does the ranking hold when none of the weights is trusted?
    rng = np.random.default_rng(seed)
    rhos, overlaps = [], []
    for _ in range(n_draws):
        jitter_e = {k: float(v * rng.uniform(0.5, 1.5))
                    for k, v in exposure_sets["published"].items()}
        jitter_v = {k: float(v * rng.uniform(0.5, 1.5))
                    for k, v in vulnerability_sets["published"].items()}
        variant = composite(jitter_e, jitter_v)
        rhos.append(_spearman(reference[scored], variant[scored]))
        overlaps.append(_top_overlap(reference[scored], variant[scored]))
    results["random_perturbation"] = {
        "n_draws": n_draws, "weight_range": "uniform 0.5 to 1.5 of each weight",
        "spearman_mean": float(np.mean(rhos)), "spearman_min": float(np.min(rhos)),
        "top_decile_overlap_mean": float(np.mean(overlaps)),
        "top_decile_overlap_min": float(np.min(overlaps)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sensitivity.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {output_dir / 'sensitivity.json'}")
    return results
