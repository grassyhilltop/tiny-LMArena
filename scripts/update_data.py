#!/usr/bin/env python3
"""Fetch Epoch Capabilities Index (ECI) scores, keep the models that run on one 32 GB GPU,
fit the trend, and write data/leaderboard.json (plus a CSV and a daily history log).

Standard library only, so the daily GitHub Action needs no installs.

    python3 scripts/update_data.py            # fetch live data
    python3 scripts/update_data.py --offline  # reuse data/raw/* from the last run

Sources (all public):
  * Epoch AI, ECI scores and model metadata (CC BY 4.0): https://epoch.ai/data/eci_scores.csv,
    https://epoch.ai/data/benchmarked_models.csv
  * Hugging Face model API: parameter counts (safetensors), downloads and likes
  * Artificial Analysis Intelligence Index: data/aa_snapshot.json (see scripts/aa_snapshot.py)
"""
import argparse
import csv
import datetime as dt
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config")
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")

EPOCH_ECI = "https://epoch.ai/data/eci_scores.csv"
EPOCH_MODELS = "https://epoch.ai/data/benchmarked_models.csv"
HF_API = "https://huggingface.co/api/models/{repo}?expand%5B%5D=safetensors&expand%5B%5D=downloads&expand%5B%5D=likes"
UA = "tiny-lmarena-updater (no contact)"
GPU_TIERS = [8, 12, 16, 24, 32]


# ---------------------------------------------------------------- fetching

def fetch(url, timeout=60, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccup: back off and retry
            last = e
            time.sleep(2 * (i + 1))
    raise last


def load_csv_text(text):
    return list(csv.DictReader(io.StringIO(text)))


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- small helpers

def to_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def parse_date(s):
    return dt.date.fromisoformat(s[:10])


def year_frac(d):
    """Decimal year, e.g. 2026-07-02 -> 2026.5."""
    start = dt.date(d.year, 1, 1)
    days = (dt.date(d.year + 1, 1, 1) - start).days
    return d.year + (d - start).days / days


def from_year_frac(y):
    yr = int(math.floor(y))
    start = dt.date(yr, 1, 1)
    days = (dt.date(yr + 1, 1, 1) - start).days
    return start + dt.timedelta(days=round((y - yr) * days))


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# Two-sided 90% t quantiles (0.95 one-sided) for small samples.
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833,
       10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740,
       18: 1.734, 19: 1.729, 20: 1.725, 22: 1.717, 24: 1.711, 26: 1.706, 28: 1.701, 30: 1.697,
       40: 1.684, 60: 1.671, 120: 1.658}


def t_quantile(df):
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    for k in keys:
        if k >= df:
            return T95[k]
    return 1.645


def ols(points):
    """Least squares y = a + b x. Returns dict with the pieces needed for a prediction band."""
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xm = sum(xs) / n
    ym = sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ym - b * xm
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - ym) ** 2 for y in ys)
    s = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    return {"a": a, "b": b, "n": n, "xm": xm, "sxx": sxx, "s": s,
            "r2": 1 - sse / sst if sst else 1.0, "t": t_quantile(n - 2)}


def pred_half_width(fit, x):
    return fit["t"] * fit["s"] * math.sqrt(1 + 1 / fit["n"] + (x - fit["xm"]) ** 2 / fit["sxx"])


def crossing(fit, value):
    """Date the trend line reaches value, plus the range where the prediction band does."""
    b = fit["b"]
    if b <= 0:
        return None
    x0 = (value - fit["a"]) / b
    # early: upper band reaches value; late: lower band reaches value. Scan in 1-day steps.
    def first_x(sign):
        lo = x0 - 15
        step = 1 / 365.25
        x = lo
        while x < x0 + 15:
            yv = fit["a"] + b * x + sign * pred_half_width(fit, x)
            if yv >= value:
                return x
            x += step
        return None
    early, late = first_x(+1), first_x(-1)
    return {"date": from_year_frac(x0).isoformat(),
            "early": from_year_frac(early).isoformat() if early else None,
            "late": from_year_frac(late).isoformat() if late else None}


# ---------------------------------------------------------------- main build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="reuse data/raw instead of fetching")
    ap.add_argument("--today", help="override today's date (YYYY-MM-DD), for testing")
    args = ap.parse_args()

    site = read_json(os.path.join(CONFIG, "site.json"), {})
    today = parse_date(args.today) if args.today else dt.datetime.now(dt.timezone.utc).date()
    os.makedirs(RAW, exist_ok=True)

    # 1. Epoch data -------------------------------------------------------------
    raw_eci = os.path.join(RAW, "eci_scores.csv")
    raw_models = os.path.join(RAW, "benchmarked_models.csv")
    if not args.offline:
        for url, path in ((EPOCH_ECI, raw_eci), (EPOCH_MODELS, raw_models)):
            body = fetch(url)
            if len(body) < 1000:
                sys.exit(f"Epoch download looks wrong ({url}, {len(body)} bytes); keeping old data")
            with open(path, "wb") as f:
                f.write(body)
    eci_rows = load_csv_text(open(raw_eci, encoding="utf-8").read())
    model_rows = load_csv_text(open(raw_models, encoding="utf-8").read())
    for col in ("Model", "eci", "eci_ci_low", "eci_ci_high", "date", "Accessibility group"):
        if col not in eci_rows[0]:
            sys.exit(f"eci_scores.csv has changed format (missing column {col!r})")

    epoch_params, epoch_links, epoch_src = {}, {}, {}
    for r in model_rows:
        first = next((t for t in re.split(r"[\s,]+", r.get("Link") or "") if t.startswith("http")), None)
        if first and r["Model"] not in epoch_src:
            epoch_src[r["Model"]] = first.strip()
        p = to_float(r.get("Parameters"))
        if p:
            epoch_params[r["Model"]] = max(p / 1e9, epoch_params.get(r["Model"], 0))
        for tok in re.split(r"[\s,]+", r.get("Link") or ""):
            if "huggingface.co/" in tok and r["Model"] not in epoch_links:
                epoch_links[r["Model"]] = tok.strip().rstrip("/")

    overrides = {r["epoch_model"]: r for r in csv.DictReader(open(os.path.join(CONFIG, "models.csv"), encoding="utf-8"))}

    # 2. Hugging Face metadata (cached, so a flaky API never blanks the page) ------
    hf_cache_path = os.path.join(DATA, "hf_cache.json")
    hf_cache = read_json(hf_cache_path, {})

    def hf_info(repo):
        if not repo:
            return {}
        if args.offline:
            return hf_cache.get(repo, {})
        try:
            d = json.loads(fetch(HF_API.format(repo=urllib.parse.quote(repo, safe="/")), timeout=30, tries=2))
            info = {"id": d.get("id", repo),
                    "params_b": round(d["safetensors"]["total"] / 1e9, 2) if d.get("safetensors", {}).get("total") else None,
                    "downloads": d.get("downloads"), "likes": d.get("likes"),
                    "fetched": today.isoformat()}
            hf_cache[repo] = info
            return info
        except Exception as e:
            print(f"  HF lookup failed for {repo}: {e}", file=sys.stderr)
            return hf_cache.get(repo, {})

    # 3. Classify every Epoch model ---------------------------------------------
    rule = site["slm_rule"]
    max_params = rule["vram_gb"] * rule["weights_share_of_vram"] / rule["bytes_per_param_4bit"]  # 48B
    anth_max = site.get("anthropic_class_max_b", 31)

    all_models, slms, unclassified = [], [], []
    for r in eci_rows:
        eci = to_float(r["eci"])
        if eci is None or not r["date"]:
            continue
        name = r["Model"]
        ov = overrides.get(name, {})
        d = parse_date(r["date"])
        group = r["Accessibility group"]
        if ov.get("open_weights") == "yes":
            open_w = True
        elif ov.get("open_weights") == "no":
            open_w = False
        else:
            open_w = group == "Open weights"
        rec = {"model": name, "date": d.isoformat(), "eci": eci,
               "lo": to_float(r["eci_ci_low"]), "hi": to_float(r["eci_ci_high"]),
               "org": r.get("Organization", ""), "open": open_w,
               "access": r.get("Model accessibility", "")}
        all_models.append(rec)
        if not open_w:
            continue
        params, psrc = to_float(ov.get("params_b")), "curated"
        hf_repo = ov.get("hf_repo") or ""
        if not hf_repo and name in epoch_links:
            m = re.search(r"huggingface\.co/([^/\s]+/[^/\s?#]+)", epoch_links[name])
            hf_repo = m.group(1) if m else ""
        if params is None and name in epoch_params:
            params, psrc = epoch_params[name], "Epoch AI"
        info = {}
        # Look up HF for every plausible SLM (for links and downloads) and for unknown sizes.
        if hf_repo and (params is None or params <= max_params):
            info = hf_info(hf_repo)
        if params is None and info.get("params_b"):
            params, psrc = info["params_b"], "Hugging Face safetensors"
        if params is None:
            unclassified.append({"model": name, "date": rec["date"], "eci": eci, "org": rec["org"]})
            continue
        rec["params_b"] = round(params, 2)
        if params > max_params:
            continue
        active = to_float(ov.get("active_b"))
        if active is None:
            m = re.search(r"[-\s]A(\d+(?:\.\d+)?)B\b", name)
            active = float(m.group(1)) if m else None
        disp = ov.get("display_name") or r.get("Display name") or name
        slm = dict(rec)
        slm.update({
            "id": slugify(disp), "name": disp, "params_b": round(params, 1), "params_source": psrc,
            "active_b": active, "vram4_gb": round(params * rule["bytes_per_param_4bit"], 1),
            "min_gpu_gb": next((g for g in GPU_TIERS if params <= g * rule["weights_share_of_vram"] / rule["bytes_per_param_4bit"]), None),
            "anthropic_class": params <= anth_max,
            "hf_repo": info.get("id") if info.get("id") else None,
            "source_url": epoch_src.get(name),
            "downloads": info.get("downloads"), "likes": info.get("likes"),
            "note": ov.get("note") or "",
        })
        slms.append(slm)

    write_json(hf_cache_path, hf_cache)
    all_models.sort(key=lambda m: (m["date"], -m["eci"]))
    slms.sort(key=lambda m: -m["eci"])
    for i, m in enumerate(slms):
        m["rank"] = i + 1
    # Rank (UB), as on LMArena: 1 + number of models that are better with no overlap in intervals.
    for m in slms:
        m["rank_ub"] = 1 + sum(1 for o in slms if o["lo"] is not None and m["hi"] is not None and o["lo"] > m["hi"])

    # 4. Best-at-release flags, per SLM definition --------------------------------
    defs = {"fit32": lambda m: True, "anthropic": lambda m: m["anthropic_class"]}
    records = {}
    for key, keep in defs.items():
        best = -1e9
        recs = []
        for m in sorted([m for m in slms if keep(m)], key=lambda m: (m["date"], -m["eci"])):
            if m["eci"] > best:
                best = m["eci"]
                recs.append(m["id"])
        records[key] = recs
    for m in slms:
        m["record"] = {k: (m["id"] in v) for k, v in records.items()}

    # 5. Frontier (best ECI of any model, open or closed, to date) -----------------
    frontier_steps, best = [], -1e9
    for m in all_models:
        if m["eci"] > best:
            best = m["eci"]
            frontier_steps.append({"date": m["date"], "eci": m["eci"], "model": m["model"], "org": m["org"]})

    def frontier_at(d):
        v = None
        for s in frontier_steps:
            if s["date"] <= d.isoformat():
                v = s
        return v

    def frontier_first_reached(value):
        for s in frontier_steps:
            if s["eci"] >= value:
                return s
        return None

    # 6. Thresholds ----------------------------------------------------------------
    by_model = {m["model"]: m for m in all_models}
    thresholds = []
    for t in site["thresholds"]:
        th = {k: t[k] for k in t if k != "kind"}
        if t["kind"] == "model":
            ref = by_model.get(t["model"])
            if not ref:
                print(f"  threshold model {t['model']} missing from Epoch data", file=sys.stderr)
                continue
            th.update({"value": ref["eci"], "lo": ref["lo"], "hi": ref["hi"], "ref_model": ref["model"], "ref_date": ref["date"]})
        elif t["kind"] == "frontier_ago":
            ref = frontier_at(today - dt.timedelta(days=t["days"]))
            th.update({"value": ref["eci"], "ref_model": ref["model"], "ref_date": ref["date"],
                       "as_of": (today - dt.timedelta(days=t["days"])).isoformat()})
        elif t["kind"] == "frontier_now":
            ref = frontier_steps[-1]
            th.update({"value": ref["eci"], "ref_model": ref["model"], "ref_date": ref["date"]})
        th["ref_params_b"] = round(epoch_params[th["ref_model"]], 0) if th.get("ref_model") in epoch_params else None
        thresholds.append(th)

    # 7. Trend fits and crossing dates, for each definition x window ---------------
    windows = {}
    for wk, spec in site["fit"]["windows"].items():
        start = today + dt.timedelta(days=int(spec)) if spec.startswith("-") else parse_date(spec)
        windows[wk] = start
    fits = {}
    for dk in defs:
        rec_models = [m for m in slms if m["record"][dk]]
        best_now = max((m for m in slms if defs[dk](m)), key=lambda m: m["eci"])
        for wk, start in windows.items():
            pts = [(year_frac(parse_date(m["date"])), m["eci"]) for m in rec_models if m["date"] >= start.isoformat()]
            if len(pts) < 3:
                continue
            f = ols(pts)
            today_x = year_frac(today)
            out = {"def": dk, "window": wk, "start": start.isoformat(), "points": len(pts),
                   "slope_per_year": round(f["b"], 2), "r2": round(f["r2"], 3),
                   "a": f["a"], "b": f["b"], "n": f["n"], "xm": f["xm"], "sxx": f["sxx"], "s": f["s"], "t": f["t"],
                   "trend_today": round(f["a"] + f["b"] * today_x, 2),
                   "best": {"id": best_now["id"], "name": best_now["name"], "eci": best_now["eci"], "date": best_now["date"]},
                   "thresholds": {}}
            for th in thresholds:
                crossed = next((m for m in sorted(slms, key=lambda m: m["date"]) if defs[dk](m) and m["eci"] >= th["value"]), None)
                c = crossing(f, th["value"])
                out["thresholds"][th["key"]] = {
                    "gap": round(th["value"] - best_now["eci"], 2),
                    "trend": c,
                    "crossed_by": {"name": crossed["name"], "date": crossed["date"], "eci": crossed["eci"]} if crossed else None,
                }
            fits[f"{dk}|{wk}"] = out

    # 8. Lag behind the frontier -----------------------------------------------------
    lag = {}
    for dk in defs:
        pts = []
        for m in sorted([m for m in slms if m["record"][dk]], key=lambda m: m["date"]):
            fr = frontier_first_reached(m["eci"])
            if fr:
                pts.append({"id": m["id"], "name": m["name"], "date": m["date"], "eci": m["eci"],
                            "frontier_model": fr["model"], "frontier_date": fr["date"],
                            "lag_months": round((parse_date(m["date"]) - parse_date(fr["date"])).days / 30.44, 1)})
        best_now = max((m for m in slms if defs[dk](m)), key=lambda m: m["eci"])
        fr = frontier_first_reached(best_now["eci"])
        lag[dk] = {"records": pts,
                   "now": {"name": best_now["name"], "eci": best_now["eci"], "frontier_model": fr["model"],
                           "frontier_date": fr["date"],
                           "months": round((today - parse_date(fr["date"])).days / 30.44, 1)} if fr else None}
    # monthly best-to-date series for frontier and SLMs (for the gap chart)
    monthly = []
    d = dt.date(2023, 1, 1)
    while d <= today:
        end = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
        end = min(end, today)
        f = frontier_at(end)
        row = {"date": end.isoformat(), "frontier": f["eci"] if f else None}
        for dk in defs:
            cand = [m["eci"] for m in slms if defs[dk](m) and m["date"] <= end.isoformat()]
            row[dk] = max(cand) if cand else None
        monthly.append(row)
        d = end + dt.timedelta(days=1)
    frontier_pts = [(year_frac(parse_date(s["date"])), s["eci"]) for s in frontier_steps if s["date"] >= "2023-07-01"]
    frontier_fit = ols(frontier_pts) if len(frontier_pts) >= 3 else None

    # 9. Artificial Analysis cross-check and the watchlist ---------------------------
    aa = read_json(os.path.join(DATA, "aa_snapshot.json"), {})
    slm_repos = {(m["hf_repo"] or "").lower(): m for m in slms if m["hf_repo"]}
    aa_rows = []
    for x in aa.get("models", []):
        repo = (x.get("hf_repo") or "").lower()
        match = slm_repos.get(repo)
        aa_rows.append(dict(x, slm_id=match["id"] if match else None, eci=match["eci"] if match else None))
    watch = []
    for w in csv.DictReader(open(os.path.join(CONFIG, "watchlist.csv"), encoding="utf-8")):
        if (w["hf_repo"] or "").lower() in slm_repos:
            continue  # it has an ECI score now
        watch.append({"name": w["name"], "org": w["org"], "hf_repo": w["hf_repo"], "date": w["release_date"],
                      "params_b": to_float(w["params_b"]), "active_b": to_float(w["active_b"]), "note": w["note"],
                      "aa_index": None})
    for x in aa_rows:
        if x["slm_id"] or not x.get("params_b") or x["params_b"] > max_params:
            continue
        if any((w["hf_repo"] or "").lower() == (x.get("hf_repo") or "").lower() for w in watch):
            continue
        watch.append({"name": x["name"], "org": x.get("org", ""), "hf_repo": x.get("hf_repo"), "date": x.get("date"),
                      "params_b": x["params_b"], "active_b": x.get("active_b"), "note": "",
                      "aa_index": x.get("index")})
    for w in watch:
        hit = next((x for x in aa_rows if (x.get("hf_repo") or "").lower() == (w["hf_repo"] or "").lower()), None)
        if hit and w["aa_index"] is None:
            w["aa_index"] = hit.get("index")
    watch.sort(key=lambda w: w["date"] or "", reverse=True)

    # 10. Assemble -------------------------------------------------------------------
    eci_updated = None
    if not args.offline:
        eci_updated = today.isoformat()
    prev = read_json(os.path.join(DATA, "leaderboard.json"), {})
    out = {
        "generated": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "today": today.isoformat(),
        "epoch_fetched": eci_updated or prev.get("epoch_fetched"),
        "rule": {"max_params_b": max_params, "vram_gb": rule["vram_gb"], "anthropic_max_b": anth_max,
                 "bytes_per_param_4bit": rule["bytes_per_param_4bit"], "weights_share": rule["weights_share_of_vram"]},
        "counts": {"eci_models": len(all_models), "open_models": sum(1 for m in all_models if m["open"]), "slms": len(slms)},
        "slms": slms,
        "frontier_steps": frontier_steps,
        "frontier_fit": {"slope_per_year": round(frontier_fit["b"], 2), "r2": round(frontier_fit["r2"], 3),
                         "a": frontier_fit["a"], "b": frontier_fit["b"]} if frontier_fit else None,
        "thresholds": thresholds,
        "fits": fits,
        "lag": lag,
        "monthly": monthly,
        "aa": {"snapshot_date": aa.get("snapshot_date"), "source": aa.get("source"),
               "frontier": aa.get("frontier"), "models": aa_rows},
        "watchlist": watch,
        "unclassified": unclassified,
    }

    # 11. Events for alerts (compare with the previous run) ---------------------------
    events = []
    if prev.get("slms"):
        p_top = prev["slms"][0]["name"]
        if slms[0]["name"] != p_top:
            events.append({"type": "new_leader", "title": f"New #1 SLM: {slms[0]['name']} (ECI {slms[0]['eci']:.1f})",
                           "detail": f"{slms[0]['name']} replaces {p_top} at the top of the single-GPU leaderboard."})
        p_top3 = {m["name"] for m in prev["slms"][:3]}
        for m in slms[:3]:
            if m["name"] not in p_top3 and m["name"] != slms[0]["name"]:
                events.append({"type": "top3", "title": f"{m['name']} enters the top 3 (ECI {m['eci']:.1f})",
                               "detail": f"Ranked #{m['rank']} among {len(slms)} single-GPU models."})
        p_names = {m["name"] for m in prev["slms"]}
        for m in slms:
            if m["name"] not in p_names:
                events.append({"type": "new_model", "title": f"New SLM scored: {m['name']} (ECI {m['eci']:.1f}, #{m['rank']})",
                               "detail": f"{m['params_b']}B parameters, released {m['date']}."})
        p_fit = (prev.get("fits") or {}).get("fit32|since2023", {}).get("thresholds", {})
        n_fit = fits.get("fit32|since2023", {}).get("thresholds", {})
        for th in thresholds:
            was = (p_fit.get(th["key"]) or {}).get("crossed_by")
            now = (n_fit.get(th["key"]) or {}).get("crossed_by")
            if now and not was:
                events.append({"type": "threshold", "key": th["key"],
                               "title": f"Threshold crossed: an SLM reached the {th['short']} ({th['value']:.1f} ECI)",
                               "detail": f"{now['name']} scores {now['eci']:.1f} on the ECI, at or above {th.get('ref_model', '')}."})
    out["events"] = events

    write_json(os.path.join(DATA, "leaderboard.json"), out)

    # flat CSV for people who want the table
    with open(os.path.join(DATA, "slm_leaderboard.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "rank_ub", "model", "eci", "eci_ci_low", "eci_ci_high", "release_date", "organization",
                    "params_b", "active_b", "vram_4bit_gb", "smallest_gpu_gb", "hugging_face", "params_source"])
        for m in slms:
            w.writerow([m["rank"], m["rank_ub"], m["name"], m["eci"], m["lo"], m["hi"], m["date"], m["org"],
                        m["params_b"], m["active_b"] or "", m["vram4_gb"], m["min_gpu_gb"],
                        f"https://huggingface.co/{m['hf_repo']}" if m["hf_repo"] else "", m["params_source"]])

    # daily history of the top 10 (one block per date)
    hist = os.path.join(DATA, "history.csv")
    have = set()
    if os.path.exists(hist):
        with open(hist, encoding="utf-8") as f:
            have = {r["date"] for r in csv.DictReader(f)}
    if today.isoformat() not in have:
        new = not os.path.exists(hist)
        with open(hist, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "rank", "model", "eci", "eci_ci_low", "eci_ci_high"])
            for m in slms[:10]:
                w.writerow([today.isoformat(), m["rank"], m["name"], m["eci"], m["lo"], m["hi"]])

    if events:
        write_json(os.path.join(DATA, "events_pending.json"), events)
        log = read_json(os.path.join(DATA, "events_log.json"), [])
        log.extend(dict(e, date=today.isoformat()) for e in events)
        write_json(os.path.join(DATA, "events_log.json"), log)

    f = fits.get("fit32|since2023")
    print(f"SLMs with ECI: {len(slms)} of {len(all_models)} models. #1 {slms[0]['name']} {slms[0]['eci']}")
    if f:
        print(f"Trend {f['slope_per_year']} ECI/yr (R2 {f['r2']}, {f['points']} pts)")
        for k, v in f["thresholds"].items():
            print(f"  {k}: gap {v['gap']}, trend {v['trend']}, crossed {v['crossed_by']}")
    if unclassified:
        print("Open models with unknown size (add to config/models.csv):", ", ".join(u["model"] for u in unclassified))
    print(f"Events: {len(events)}")


if __name__ == "__main__":
    main()
