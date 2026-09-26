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


def archive_region(region_id: str, out_dir: Path) -> Path:
    paths = region_paths(region_id)
    name, hazard, _ = REGION_CONFIGS[region_id]
    files = []
    for kind in ("raw", "output"):
        base = paths[kind]
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not _excluded(path):
                files.append((kind, path))

    manifest = {
        "region": region_id,
        "name": name,
        "hazard": hazard,
        "built": date.today().isoformat(),
        "code": "https://github.com/rakibhhridoy/rimes-mapathon",
        "files": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"hazmapper_{region_id}.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--out", default=str(ROOT / "dist" / "zenodo"))
    args = parser.parse_args()

    regions = args.regions or [r for r in REGION_CONFIGS if has_results(r)]
    for region in regions:
        archive_region(region, Path(args.out))
    print("Upload the archives and manifests to Zenodo as a new version of "
          "10.5281/zenodo.22978729, and cite the DOI it assigns.")
