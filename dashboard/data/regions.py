"""
Region registry for the dashboard.

A region is available when its pipeline has produced ranked assets. Regions
without results are listed as in preparation rather than hidden, so the map's
coverage is honest about what has and has not been processed.
"""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Region id → (display name, hazard, config file). The config carries the
# bounding box, map centre and data paths, so nothing is duplicated here.
REGION_CONFIGS = {
    "rangpur_rajshahi": ("Rangpur & Rajshahi", "Riverine flood", "config.yaml"),
    "sylhet": ("Sylhet", "Flash flood", "configs/sylhet.yaml"),
    "sw_coastal": ("South-west coast", "Coastal & tidal flooding",
                   "configs/sw_coastal.yaml"),
    "cht": ("Chittagong Hill Tracts", "Rainfall-triggered landslide",
            "configs/cht.yaml"),
}


def _load_config(path: Path) -> dict:
    """Load a config, following its `extends:` chain (same rules as the CLI)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = _load_config((path.parent / parent).resolve())
        cfg = _merge(base, cfg)
    return cfg


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=8)
def region_config(region_id: str) -> dict:
    """Merged config for one region."""
    _, _, config_file = REGION_CONFIGS[region_id]
    return _load_config(ROOT / config_file)


@lru_cache(maxsize=8)
def region_paths(region_id: str) -> dict:
    """Absolute raw / processed / output / cache directories for a region."""
    cfg = region_config(region_id)
    paths = cfg.get("paths", {}) or {}
    out = {
        kind: ROOT / paths.get(f"{kind}_dir", f"data/{kind}")
        for kind in ("raw", "processed", "output")
    }
    # Display caches sit beside the outputs they are built from.
    out["cache"] = out["output"].parent / "cache"
    return out


def has_results(region_id: str) -> bool:
    """True when the region's pipeline has produced ranked assets."""
    paths = region_paths(region_id)
    return any([
        (paths["cache"] / "risk_ranked_assets.parquet").exists(),
        (paths["output"] / "risk_ranked_assets.geojson").exists(),
        (paths["output"] / "landslide_susceptibility.tif").exists(),
    ])


def available_regions() -> list[str]:
    """Region ids with results, in registry order."""
    return [rid for rid in REGION_CONFIGS if has_results(rid)]


def region_status() -> list[dict]:
    """Every region with its display name, hazard and availability."""
    return [
        {
            "id": rid,
            "name": name,
            "hazard": hazard,
            "available": has_results(rid),
        }
        for rid, (name, hazard, _) in REGION_CONFIGS.items()
    ]


def region_label(region_id: str) -> str:
    name, hazard, _ = REGION_CONFIGS[region_id]
    return f"{name} — {hazard}"
