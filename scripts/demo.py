"""
Reproduce one region from the published data archive, without Earth Engine.

Downloads the region's archive from the Zenodo data record, checks it
against Zenodo's checksum, unpacks the inputs where the pipeline reads them
and the archived outputs beside them as a reference, runs the pipeline from
terrain preprocessing to validation, and compares the new results with the
archived ones. The Sentinel-1 flood masks are in the archive, so no Earth
Engine account is needed; the two country-border files are fetched from
geoBoundaries and Natural Earth if missing.

    python scripts/demo.py                    # Sylhet, about 15 minutes
    python scripts/demo.py --region sw_coastal
    python scripts/demo.py --skip-download    # archive already unpacked

The comparison passes when every figure agrees with the archive to within
--tolerance (default 0.005 in AUC).
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.data.regions import REGION_CONFIGS, region_paths  # noqa: E402

RECORD = "23108572"           # data archive, version 3
STAGES = ["preprocess", "features", "graph", "train", "krige", "risk", "metadata", "validate"]
CHUNK = 1 << 20
RETRIES = 8


def download(region: str, record: str, target: Path) -> Path:
    """The region's archive from the Zenodo record, checked against its MD5."""
    meta = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
    meta.raise_for_status()
    name = f"hazmapper_{region}.tar.gz"
    entry = next((f for f in meta.json()["files"] if f["key"] == name), None)
    if entry is None:
        sys.exit(f"{name} is not in Zenodo record {record}")
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    expected = entry["checksum"].removeprefix("md5:")
    if path.exists() and _md5(path) == expected:
        print(f"{name} already downloaded")
        return path
    print(f"downloading {name} ({entry['size'] / 1e6:,.0f} MB)", flush=True)
    # Zenodo connections can stall on large files, so a broken download
    # resumes from the bytes already on disk instead of starting over.
    for attempt in range(1, RETRIES + 1):
        have = path.stat().st_size if path.exists() else 0
        if have >= entry["size"]:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(entry["links"]["self"], stream=True, headers=headers,
                              timeout=(60, 120)) as r:
                r.raise_for_status()
                # a server that ignores the range sends the whole file again
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(path, mode) as f:
                    for block in r.iter_content(CHUNK):
                        f.write(block)
        except requests.RequestException as error:
            print(f"  attempt {attempt} stopped at {path.stat().st_size / 1e6:,.0f} MB "
                  f"({type(error).__name__}); resuming", flush=True)
            time.sleep(10 * attempt)
    if _md5(path) != expected:
        path.unlink()
        sys.exit(f"{name}: checksum does not match Zenodo's; download again")
    return path


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack(region: str, archive: Path, reference: Path) -> None:
    """Inputs to the region's raw directory; archived outputs to `reference`."""
    paths = region_paths(region)
    targets = {"raw": paths["raw"], "output": reference}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts      # region/kind/...
            if len(parts) < 3 or parts[1] not in targets or not member.isfile():
                continue
            out = targets[parts[1]].joinpath(*parts[2:])
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"inputs in {paths['raw']}, archived outputs in {reference}")


def figures(output: Path) -> dict:
    """The headline figures of a region's outputs."""
    validation = json.loads((output / "validation_metrics.json").read_text())
    meta = json.loads((output / "pipeline_metadata.json").read_text())
    assets = len(json.loads((output / "risk_ranked_assets.geojson").read_text())["features"])
    held = validation["assets_vs_observed"]["held_out_blocks"]
    return {
        "assets": assets,
        "flood-prone share": meta.get("label_positive_rate"),
        "validation AUC": (meta.get("gnn_validation") or {}).get("val_auc_roc"),
        "AUC vs observed floods": held.get("auc_roc"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", default="sylhet",
                        choices=["rangpur_rajshahi", "sylhet", "sw_coastal"])
    parser.add_argument("--record", default=RECORD)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()

    from pipeline.cli import _load_config
    from pipeline.country import ensure_country_files

    config = REGION_CONFIGS[args.region][2]
    reference = ROOT / "data" / "reference" / args.region
    if not args.skip_download:
        archive = download(args.region, args.record, ROOT / "dist" / "demo")
        unpack(args.region, archive, reference)
    ensure_country_files(_load_config(str(ROOT / config)))

    for stage in STAGES:
        print(f"== {stage}", flush=True)
        subprocess.run([sys.executable, "-m", "pipeline.cli", "-c", config, stage],
                       cwd=ROOT, check=True)

    new, old = figures(region_paths(args.region)["output"]), figures(reference)
    print(f"\n{'':26}{'this run':>12}{'archive':>12}")
    failed = []
    for key in new:
        a, b = new[key], old[key]
        print(f"{key:26}{a:>12,.3f}{b:>12,.3f}" if isinstance(a, float) else f"{key:26}{a:>12,}{b:>12,}")
        close = a == b if isinstance(a, int) else (a is not None and b is not None
                                                   and abs(a - b) <= args.tolerance)
        if not close:
            failed.append(key)
    if failed:
        sys.exit(f"\nDiffers from the archive: {', '.join(failed)}")
    print("\nReproduced: every figure agrees with the archive.")


if __name__ == "__main__":
    main()
