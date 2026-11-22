#!/usr/bin/env python3
"""Refresh data/aa_snapshot.json: the Artificial Analysis Intelligence Index for small open models.

Two routes:
  * With an API key (AA_API_KEY env var): the documented free endpoint
    https://artificialanalysis.ai/api/v2/language/models/free  (attribution required).
  * Without a key: pass saved copies of the public pages, for a manual one-off refresh:
        python3 scripts/aa_snapshot.py --html small.html --html models.html
    (small = https://artificialanalysis.ai/models/open-source/small,
     models = https://artificialanalysis.ai/models)

The daily Action only uses the API route, and only when the AA_API_KEY secret is set. Otherwise the
committed snapshot is kept, and the page shows its date.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "aa_snapshot.json")
API = "https://artificialanalysis.ai/api/v2/language/models/free"
MAX_B = 48


def flight_models(path):
    """Model records embedded in a saved Artificial Analysis page (Next.js flight data)."""
    t = open(path, encoding="utf-8").read()
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', t, flags=re.S)
    s = "".join(json.loads('"' + c + '"') for c in chunks)
    found = {}
    for key in ('"initialModels":[', '"models":['):
        pos = 0
        while True:
            i = s.find(key, pos)
            if i < 0:
                break
            j = i + len(key) - 1
            depth, k = 0, j
            for k in range(j, len(s)):
                if s[k] == "[":
                    depth += 1
                elif s[k] == "]":
                    depth -= 1
                    if depth == 0:
                        break
            try:
                for m in json.loads(s[j:k + 1]):
                    if isinstance(m, dict) and m.get("slug") and "intelligenceIndex" in m:
                        found[m["slug"]] = m
            except ValueError:
                pass
            pos = k
    return list(found.values())


def hf_repo(url):
    m = re.search(r"huggingface\.co/([^/\s]+/[^/\s?#]+)", url or "")
    return m.group(1) if m else None


def normalise_page(models):
    rows = []
    for m in models:
        idx = m.get("intelligenceIndex")
        if idx is None or idx <= 0:
            continue
        creator = m.get("creator")
        rows.append({
            "name": re.sub(r"\s*\((?:[^)]*)\)\s*$", "", m["name"]).strip(),
            "variant": m["name"],
            "org": creator.get("name") if isinstance(creator, dict) else "",
            "date": m.get("releaseDate"),
            "params_b": m.get("parameters"),
            "active_b": m.get("inferenceParametersActiveBillions"),
            "index": round(idx, 1),
            "hf_repo": hf_repo(m.get("modelWeightsSourceUrl")),
            "url": "https://artificialanalysis.ai" + (m.get("url") or f"/models/{m['slug']}"),
        })
    return rows


def normalise_api(payload):
    rows = []
    for m in payload.get("data", payload if isinstance(payload, list) else []):
        ev = m.get("evaluations") or {}
        idx = ev.get("artificial_analysis_intelligence_index") or m.get("intelligence_index")
        if idx is None or idx <= 0:
            continue
        creator = m.get("model_creator") or {}
        rows.append({
            "name": m.get("name"), "variant": m.get("name"), "org": creator.get("name", ""),
            "date": m.get("release_date"), "params_b": m.get("parameters") or m.get("total_parameters_billions"),
            "active_b": m.get("active_parameters_billions"), "index": round(float(idx), 1),
            "hf_repo": hf_repo(m.get("model_weights_url") or m.get("huggingface_url")),
            "url": f"https://artificialanalysis.ai/models/{m.get('slug', '')}",
        })
    return rows


def build(rows, source):
    frontier = max(rows, key=lambda r: r["index"]) if rows else None
    best = {}
    for r in rows:
        if not r["params_b"] or r["params_b"] > MAX_B or not r["hf_repo"]:
            continue
        k = r["hf_repo"].lower()
        if k not in best or r["index"] > best[k]["index"]:
            best[k] = r
    return {
        "snapshot_date": dt.date.today().isoformat(),
        "source": source,
        "frontier": {"name": frontier["variant"], "index": frontier["index"], "url": frontier["url"]} if frontier else None,
        "models": sorted(best.values(), key=lambda r: -r["index"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="append", default=[], help="saved Artificial Analysis page")
    a = ap.parse_args()
    if a.html:
        rows = []
        for p in a.html:
            rows += normalise_page(flight_models(p))
        snap = build(rows, "Artificial Analysis website (manual snapshot)")
    else:
        key = os.environ.get("AA_API_KEY")
        if not key:
            print("No AA_API_KEY; keeping the existing snapshot.")
            return
        req = urllib.request.Request(API, headers={"x-api-key": key, "User-Agent": "tiny-lmarena-updater (no contact)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            rows = normalise_api(json.load(r))
        if not rows:
            sys.exit("API returned no usable rows; keeping the existing snapshot.")
        snap = build(rows, "Artificial Analysis API (free tier)")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(snap['models'])} small models; frontier {snap['frontier']}")


if __name__ == "__main__":
    main()
