#!/usr/bin/env python3
"""Build index.html (GitHub Pages) and dist/artifact.html (a snapshot fragment) from
src/template.html, config/site.json and data/leaderboard.json. The data is embedded in the page,
so it renders with no network requests besides fonts and the chart library."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    site = json.load(open(os.path.join(ROOT, "config", "site.json"), encoding="utf-8"))
    data = json.load(open(os.path.join(ROOT, "data", "leaderboard.json"), encoding="utf-8"))
    tpl = open(os.path.join(ROOT, "src", "template.html"), encoding="utf-8").read()

    # Only what the page needs; drop bulky fields.
    data["site"] = {k: site[k] for k in ("site_name", "repo", "site_url", "author", "affiliation", "newsletter")}
    data.pop("unclassified", None)
    data.pop("events", None)
    # Escape every "<" (and > and &) so no third-party string can end or confuse the <script> block.
    blob = (json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))

    tpl = tpl.replace("__SITE_NAME__", site["site_name"]).replace("/*__DATA__*/", blob)
    head, body = tpl.split("<!--BODY-->", 1)
    head = head.replace("<!--HEAD-->", "").strip()

    url = site["site_url"]
    og = (f'<meta property="og:title" content="{site["site_name"]}: single-GPU language model leaderboard">\n'
          f'<meta property="og:description" content="The most capable open models that run on one 32 GB GPU, ranked by the Epoch Capabilities Index, with forecasts of when they reach key capability thresholds.">\n'
          f'<meta property="og:url" content="{url}">\n<meta property="og:type" content="website">\n'
          f'<meta name="twitter:card" content="summary">\n<link rel="canonical" href="{url}">')
    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + head + "\n" + og + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")
    with open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)

    # Fragment for a claude.ai artifact snapshot (the artifact host adds the document skeleton).
    frag_head = head.replace("<title>", '<title>', 1)
    frag_head = frag_head.replace("</title>", '</title>\n<meta name="claude:viewport-fit-sides" content="cover">', 1)
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    with open(os.path.join(ROOT, "dist", "artifact.html"), "w", encoding="utf-8") as f:
        f.write(frag_head + "\n" + body.strip() + "\n")
    print(f"index.html {len(page) // 1024} KB")


if __name__ == "__main__":
    main()
