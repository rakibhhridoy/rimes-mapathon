"""
Assemble the data archive for Zenodo.

One compressed archive per region holding the inputs the pipeline ingests
and the outputs it produces, plus a manifest with SHA-256 checksums and the
licence of every input. The GADM files that the March 2026 archive carried
are excluded: GADM forbids redistribution, and the pipeline no longer uses
them.

Usage:
    python scripts/make_archive.py                 # all regions with results
    python scripts/make_archive.py --regions cht   # one region
    python scripts/make_archive.py --regions jamuna_east_validation
    python scripts/make_archive.py --regions rolling_origin   # rolling-origin outputs
    python scripts/make_archive.py --out dist/zenodo
"""

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.data.regions import REGION_CONFIGS, has_results, region_paths  # noqa: E402

# Anything matching these is never archived.
EXCLUDE_PATTERNS = ("gadm", "._", ".DS_Store")

LICENCES = {
    "infrastructure_raw.gpkg": "OpenStreetMap contributors, ODbL 1.0",
    "dem_srtm_30m.tif": "NASA SRTM, public domain",
    "jrc_water_occurrence.tif": "EC JRC Global Surface Water, free with attribution",
    "worldpop_popdens.tif": "WorldPop, CC BY 4.0",
    "bgd_union.gpkg": "geoBoundaries gbOpen, CC BY 4.0",
    "bgd_upazila.gpkg": "geoBoundaries gbOpen, CC BY 4.0",
    "bgd_district.gpkg": "geoBoundaries gbOpen, CC BY 4.0",
    "coolr_landslides.geojson": "NASA COOLR, open (Juang et al. 2019)",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(path: Path) -> bool:
    return any(pattern in path.name for pattern in EXCLUDE_PATTERNS)


def _licence_for(path: Path) -> str:
    for name, licence in LICENCES.items():
        if path.name == name:
            return licence
    if path.name.startswith("s1_flood"):
        return "Contains modified Copernicus Sentinel data, processed with Google Earth Engine"
    if "output" in path.parts:
        return "Pipeline output, CC BY 4.0"
    return "see README"


# Regions run only to validate the paper's comparison with the flood record,
# which are not on the website: id, name, hazard and config.
VALIDATION_REGIONS = {
    "jamuna_east_validation": ("Jamuna east (validation)", "Riverine & flash flood",
                               "configs/validation/jamuna_east.yaml"),
    "central_validation": ("Central (validation)", "Riverine flood",
                           "configs/validation/central.yaml"),
    "chattogram_coast_validation": ("Chattogram coast (validation)", "Flash flood & surge",
                                    "configs/validation/chattogram_coast.yaml"),
    "west_central_validation": ("West-central (validation)", "Waterlogging",
                                "configs/validation/west_central.yaml"),
}
# National administrative layers, identical in every region and already in the
# study-region archives, are left out of the validation archives.
SHARED_LAYERS = ("bgd_union.gpkg", "bgd_upazila.gpkg", "bgd_district.gpkg")


def _paths_from_config(config_file: str) -> dict:
    from pipeline.cli import _load_config

    paths = _load_config(str(ROOT / config_file)).get("paths", {}) or {}
    return {kind: ROOT / paths[f"{kind}_dir"] for kind in ("raw", "processed", "output")}


def archive_region(region_id: str, out_dir: Path) -> Path:
    if region_id in VALIDATION_REGIONS:
        name, hazard, config_file = VALIDATION_REGIONS[region_id]
        paths = _paths_from_config(config_file)
    else:
        paths = region_paths(region_id)
        name, hazard, _ = REGION_CONFIGS[region_id]
    # raw_s1a holds the events re-mapped from Sentinel-1A alone
    # (scripts/remap_s1a.py), for the single-satellite check of the paper.
    paths = dict(paths, raw_s1a=paths["raw"].with_name(paths["raw"].name + "_s1a"))
    files = []
    for kind in ("raw", "raw_s1a", "output"):
        base = paths[kind]
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or _excluded(path):
                continue
            if region_id in VALIDATION_REGIONS and path.name in SHARED_LAYERS:
                continue
            files.append((kind, path))

    manifest = {
        "region": region_id,
        "name": name,
        "hazard": hazard,
        "built": date.today().isoformat(),
        "code": "https://github.com/rakibhhridoy/rimes-mapathon",
        "files": [],
    }
    if region_id in VALIDATION_REGIONS:
        manifest["not_included"] = (f"{', '.join(SHARED_LAYERS)}: national layers, identical "
                                    "to those in the study-region archives")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"hazmapper_{region_id}.tar.gz"

    with tarfile.open(archive_path, "w:gz", dereference=True) as tar:
        for kind, path in files:
            arcname = f"{region_id}/{kind}/{path.relative_to(paths[kind])}"
            tar.add(path, arcname=arcname)
            manifest["files"].append({
                "path": arcname,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "licence": _licence_for(path),
            })
        manifest_path = out_dir / f"hazmapper_{region_id}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        tar.add(manifest_path, arcname=f"{region_id}/MANIFEST.json")

    size_mb = archive_path.stat().st_size / 1e6
    print(f"{region_id}: {len(files)} files → {archive_path} ({size_mb:.0f} MB)")
    return archive_path


def archive_rolling_origin(out_dir: Path) -> Path:
    """The rolling-origin outputs of the study regions, which their v5 archives
    predate, and the pooled summary across all regions."""
    sources = [ROOT / "data" / "pooled" / "rolling_origin_pooled.json"]
    sources += [region_paths(r)["output"] / "rolling_origin.json"
                for r in ("rangpur_rajshahi", "sylhet", "sw_coastal")]
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "hazmapper_rolling_origin.tar.gz"
    manifest = {"name": "Rolling-origin tests of the record against the model",
                "built": date.today().isoformat(),
                "code": "https://github.com/rakibhhridoy/rimes-mapathon", "files": []}
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sources:
            region = "pooled" if path.parent.name == "pooled" else path.parent.parent.name
            region = "rangpur_rajshahi" if region == "data" else region
            arcname = f"rolling_origin/{region}/{path.name}"
            tar.add(path, arcname=arcname)
            manifest["files"].append({"path": arcname, "bytes": path.stat().st_size,
                                      "sha256": _sha256(path),
                                      "licence": "Pipeline output, CC BY 4.0"})
        manifest_path = out_dir / "hazmapper_rolling_origin_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        tar.add(manifest_path, arcname="rolling_origin/MANIFEST.json")
    print(f"rolling_origin: {len(sources)} files → {archive_path}")
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--out", default=str(ROOT / "dist" / "zenodo"))
    args = parser.parse_args()

    regions = args.regions or [r for r in REGION_CONFIGS if has_results(r)]
    for region in regions:
        if region == "rolling_origin":
            archive_rolling_origin(Path(args.out))
        else:
            archive_region(region, Path(args.out))
    print("Upload the archives and manifests to Zenodo as a new version of the "
          "data record (scripts/zenodo_upload.py --new-version), and cite the DOI it assigns.")
