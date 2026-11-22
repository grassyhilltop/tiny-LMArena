#!/usr/bin/env python3
"""Announce events found by update_data.py (data/events_pending.json).

Each event becomes a GitHub release (watchers who chose Watch -> Custom -> Releases get an email, and
the releases.atom feed updates). If the BUTTONDOWN_API_KEY secret is set and newsletter.send_alerts
is true in config/site.json, the same text is also sent to the Buttondown mailing list.

    python3 scripts/send_alerts.py           # announce pending events, if any
    python3 scripts/send_alerts.py --test    # announce a clearly labelled test event
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PENDING = os.path.join(ROOT, "data", "events_pending.json")


def body_text(events, site, test):
    lines = []
    if test:
        lines += ["This is a test of the alert pipeline. Nothing has changed on the leaderboard.", ""]
    for e in events:
        lines += [f"## {e['title']}", e.get("detail", ""), ""]
    lines += [f"Live leaderboard and forecast: {site['site_url']}",
              "Scores: Epoch Capabilities Index (Epoch AI, CC BY 4.0)."]
    return "\n".join(lines)


def github_release(title, body, test):
    tag = "alert-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + ("-test" if test else "")
    cmd = ["gh", "release", "create", tag, "--title", title, "--notes", body]
    if test:
        cmd.append("--prerelease")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("release:", r.stdout.strip() or r.stderr.strip())


def buttondown(subject, body, test, site):
    key = os.environ.get("BUTTONDOWN_API_KEY")
    nl = site.get("newsletter", {})
    if not key or not nl.get("send_alerts"):
        print("Buttondown: not configured, skipped")
        return
    payload = {"subject": subject, "body": body, "status": "draft" if test else nl.get("email_status", "about_to_send")}
    req = urllib.request.Request("https://api.buttondown.com/v1/emails", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Token {key}", "Content-Type": "application/json",
                                          "X-Buttondown-Live-Dangerously": "true"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print("Buttondown:", r.status, "(draft)" if test else "")
    except Exception as e:  # never fail the daily job over email
        print("Buttondown error:", e, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    site = json.load(open(os.path.join(ROOT, "config", "site.json"), encoding="utf-8"))
    if a.test:
        events = [{"title": f"Test alert from {site['site_name']}", "detail": "If you can read this, alerts work."}]
    elif os.path.exists(PENDING):
        events = json.load(open(PENDING, encoding="utf-8"))
    else:
        print("No events.")
        return
    if not events:
        return
    subject = ("[Test] " if a.test else "") + (events[0]["title"] if len(events) == 1 else f"{events[0]['title']} (+{len(events) - 1} more)")
    body = body_text(events, site, a.test)
    github_release(subject, body, a.test)
    buttondown(subject, body, a.test, site)
    if not a.test and os.path.exists(PENDING):
        os.remove(PENDING)


if __name__ == "__main__":
    main()
