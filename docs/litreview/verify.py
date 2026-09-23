"""
Resolve DOIs against Crossref and OpenAlex and cache what they say.

Nothing enters the literature review unless its DOI resolves here: the
earlier dashboard literature panel carried a placeholder DOI, and a review
built from memory is how such things happen. Abstracts are reconstructed from
OpenAlex's inverted index so a claim about a paper can be checked against the
paper's own summary.

Usage:
    python verify.py 10.1038/nature20584 10.1371/journal.pone.0218657
    python verify.py --search "flood susceptibility Bangladesh machine learning" --n 15
"""

import json
import sys
import time
import urllib.parse
from pathlib import Path

import requests

CACHE = Path(__file__).with_name("verified.json")
HEADERS = {"User-Agent": "fermium-hazmapper-litreview (mailto:rakibhhridoy.py@gmail.com)"}


def _load():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _abstract(inv):
    if not inv:
        return None
    words = {}
    for word, positions in inv.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def verify(doi: str) -> dict:
    doi = doi.lower().strip()
    cache = _load()
    if doi in cache:
        return cache[doi]
    rec = {"doi": doi, "crossref": False}
    r = requests.get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}",
                     headers=HEADERS, timeout=60)
    if r.status_code == 200:
        m = r.json()["message"]
        rec.update({
            "crossref": True,
            "title": (m.get("title") or [""])[0],
            "journal": (m.get("container-title") or [""])[0],
            "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "authors": [f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                        for a in m.get("author", [])],
            "volume": m.get("volume"), "issue": m.get("issue"),
            "pages": m.get("page") or m.get("article-number"),
            "type": m.get("type"),
        })
    o = requests.get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}",
                     headers=HEADERS, timeout=60)
    if o.status_code == 200:
        w = o.json()
        rec["abstract"] = _abstract(w.get("abstract_inverted_index"))
        rec["cited_by"] = w.get("cited_by_count")
        if not rec.get("title"):
            rec["title"] = w.get("title")
    cache[doi] = rec
    CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    time.sleep(0.2)
    return rec


def search(query: str, n: int = 15, since: int = 2015) -> list:
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(query)
           + f"&filter=from_publication_date:{since}-01-01,has_doi:true"
           + f"&sort=relevance_score:desc&per-page={n}")
    r = requests.get(url, headers=HEADERS, timeout=60).json()
    out = []
    for w in r.get("results", []):
        out.append({
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "cited_by": w.get("cited_by_count"),
        })
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--search":
        q = args[1]
        n = int(args[3]) if len(args) > 3 and args[2] == "--n" else 15
        for w in search(q, n):
            print(f"{w['year']} | {w['cited_by']:>5} cit | {w['doi']} | {w['title'][:110]} | {w['venue']}")
    else:
        for d in args:
            rec = verify(d)
            ok = "OK " if rec["crossref"] else "NOT FOUND"
            print(f"{ok} {d} | {rec.get('year')} | {str(rec.get('title'))[:90]} | {rec.get('journal')}")


def fill_abstract(doi: str) -> str | None:
    """Fetch an abstract from Semantic Scholar when OpenAlex has none."""
    doi = doi.lower()
    cache = _load()
    rec = cache.get(doi) or verify(doi)
    if rec.get("abstract"):
        return rec["abstract"]
    r = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                     params={"fields": "abstract,tldr"}, headers=HEADERS, timeout=60)
    if r.status_code == 200:
        j = r.json()
        text = j.get("abstract") or ((j.get("tldr") or {}).get("text"))
        if text:
            rec["abstract"] = text
            rec["abstract_source"] = "semanticscholar" + ("" if j.get("abstract") else " (tldr)")
            cache = _load(); cache[doi] = rec
            CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    time.sleep(1.2)
    return rec.get("abstract")
