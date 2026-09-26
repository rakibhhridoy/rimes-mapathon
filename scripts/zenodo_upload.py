"""
Upload the data archive to a Zenodo draft through the API.

The browser upload tends to fail on files of several hundred megabytes, so
this script puts each file into the draft's bucket separately, retries a
failed file, and checks every upload against the local MD5 checksum. A file
already in the draft with the right checksum is skipped, so after a failure
the script can simply be run again.

It uploads what `scripts/make_archive.py` writes: the four region archives
and their manifests, manifests first and then the archives from smallest to
largest. It never publishes: review the draft in the browser and publish
there.

The access token is read from `.zenodo_token` in the repository root (git
ignores it). Create one at zenodo.org under Applications → Personal access
tokens, with the `deposit:write` scope, and paste it into that file.

Usage:
    python scripts/zenodo_upload.py                     # list your drafts
    python scripts/zenodo_upload.py --draft 12345678    # upload to that draft
    python scripts/zenodo_upload.py --draft 12345678 --remove-others
    python scripts/zenodo_upload.py --new-version 19233968 --remove-others

The draft ID is the number in the browser address while you edit the draft
(zenodo.org/uploads/12345678). `--new-version` instead opens a new-version
draft of a published record (or reuses the one already open) and uploads to
it. `--remove-others` deletes files in the draft
that are not part of this archive, such as the March 2026 files a new version
carries over or a partly uploaded zip.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = ROOT / ".zenodo_token"
API = "https://zenodo.org/api"

# The connect timeout also bounds each socket write, so it has to allow for
# the connection stalling for minutes during a large upload.
RETRIES = 5
CHUNK = 1024 * 1024


def read_token():
    if not TOKEN_FILE.exists():
        sys.exit(f"No token file at {TOKEN_FILE}.")
    lines = [line.strip() for line in TOKEN_FILE.read_text().splitlines()]
    tokens = [line for line in lines if line and not line.startswith("#")]
    if not tokens:
        sys.exit(f"Paste your Zenodo access token into {TOKEN_FILE}.")
    return tokens[0]


def archive_files(dist):
    """The manifests, then the archives from smallest to largest."""
    manifests = sorted(dist.glob("hazmapper_*_manifest.json"))
    archives = sorted(dist.glob("hazmapper_*.tar.gz"), key=lambda p: p.stat().st_size)
    if not archives:
        sys.exit(f"No archives in {dist}: run scripts/make_archive.py first.")
    return manifests + archives


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class Progress:
    """A file wrapper that reports how much requests has read from it."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.size = path.stat().st_size
        self.sent = 0
        self.name = path.name

    def __len__(self):
        return self.size

    def read(self, n=-1):
        block = self.f.read(n)
        self.sent += len(block)
        pct = 100 * self.sent / self.size if self.size else 100
        print(f"\r  {self.name}: {self.sent / 1e6:,.0f} of {self.size / 1e6:,.0f} MB "
              f"({pct:.0f}%)", end="", flush=True)
        return block

    def close(self):
        self.f.close()


def list_drafts(session):
    r = session.get(f"{API}/deposit/depositions", params={"status": "draft"})
    r.raise_for_status()
    drafts = r.json()
    if not drafts:
        print("You have no drafts. Create the new version in the browser "
              "(\"New version\" on the record page), then run this again.")
        return
    print("Your drafts:")
    for d in drafts:
        title = d.get("title") or d.get("metadata", {}).get("title") or "(no title)"
        print(f"  {d['id']}  {title}")
    print("Run again with --draft <id>.")


def new_version_draft(session, record):
    """The ID of the new-version draft of a published record, created if needed."""
    r = session.post(f"{API}/deposit/depositions/{record}/actions/newversion")
    if r.status_code in (401, 403):
        sys.exit("Zenodo refused the token: check it has the deposit:write scope.")
    r.raise_for_status()
    draft_id = int(r.json()["links"]["latest_draft"].rstrip("/").rsplit("/", 1)[-1])
    if draft_id == record:
        sys.exit(f"Zenodo did not open a new version of {record}.")
    print(f"new-version draft of {record}: {draft_id} "
          f"(https://zenodo.org/uploads/{draft_id})")
    return draft_id


def upload(session, bucket, path, local_md5):
    for attempt in range(1, RETRIES + 1):
        body = Progress(path)
        try:
            r = session.put(f"{bucket}/{path.name}", data=body,
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=(300, 3600))
            print()
            r.raise_for_status()
            remote = r.json().get("checksum", "").removeprefix("md5:")
            if remote == local_md5:
                return True
            print(f"  checksum mismatch (Zenodo {remote}, local {local_md5})")
        except requests.RequestException as e:
            print(f"\n  attempt {attempt} failed: {e}")
        finally:
            body.close()
        if attempt < RETRIES:
            wait = 30 * attempt
            print(f"  retrying in {wait} s")
            time.sleep(wait)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=int, help="ID of the Zenodo draft")
    parser.add_argument("--new-version", type=int, metavar="RECORD",
                        help="open a new-version draft of this published record")
    parser.add_argument("--dist", default=str(ROOT / "dist" / "zenodo"))
    parser.add_argument("--remove-others", action="store_true",
                        help="delete draft files that are not part of this archive")
    args = parser.parse_args()

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {read_token()}"

    if args.new_version is not None:
        args.draft = new_version_draft(session, args.new_version)

    if args.draft is None:
        list_drafts(session)
        return

    r = session.get(f"{API}/deposit/depositions/{args.draft}")
    if r.status_code in (401, 403):
        sys.exit("Zenodo refused the token: check it has the deposit:write scope.")
    r.raise_for_status()
    draft = r.json()
    if draft.get("submitted") and draft.get("state") == "done":
        sys.exit(f"{args.draft} is already published; give the ID of the draft.")
    bucket = draft["links"]["bucket"]
    remote = {f["filename"]: f for f in draft.get("files", [])}

    files = archive_files(Path(args.dist))
    wanted = {p.name for p in files}

    others = [f for name, f in remote.items() if name not in wanted]
    for f in others:
        if args.remove_others:
            session.delete(f"{API}/deposit/depositions/{args.draft}/files/{f['id']}"
                           ).raise_for_status()
            print(f"removed {f['filename']}")
        else:
            print(f"not part of this archive, left in the draft: {f['filename']} "
                  "(use --remove-others to delete)")

    failed = []
    for path in files:
        local_md5 = md5(path)
        if remote.get(path.name, {}).get("checksum") == local_md5:
            print(f"already uploaded: {path.name}")
            continue
        print(f"uploading {path.name}")
        if not upload(session, bucket, path, local_md5):
            failed.append(path.name)

    if failed:
        print("\nNot uploaded: " + ", ".join(failed))
        print("Run the same command again to retry them; finished files are skipped.")
        sys.exit(1)
    print(f"\nAll {len(files)} files are in draft {args.draft} with matching checksums.")
    print("Review the draft in the browser and publish it there, then cite the "
          "version DOI it assigns.")


if __name__ == "__main__":
    main()
