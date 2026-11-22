# Notes for agents working on this site

## Charts (owner's standing rule)

* Every chart has a clear title at the top that says what it shows, so a screenshot explains itself, and a legend
  that names every series, band and threshold. The title must look like a title; any subtitle or TL;DR line under it
  is visibly smaller and lower in emphasis.
* Labels must not overlap each other, the data points, or the busy area where the trend meets a threshold. Move
  secondary text (crossing dates, "Today", ranges) to quieter parts of the chart, or leave it to the tooltip.
* The landing view must show the top 3, the key numbers and the whole forecast chart above the fold on a laptop
  screen (about 1440 x 860). Check with a screenshot before publishing.

## How the site is built

* `scripts/update_data.py` (daily, GitHub Action) fetches Epoch's ECI data and writes `data/leaderboard.json`;
  `scripts/build_page.py` embeds it into `index.html` from `src/template.html`. Edit the template, never `index.html`.
* The page refits the trend in the browser for each filter (region, model class, window). That code mirrors the
  statistics in `update_data.py` (best-at-release records, OLS, 90% prediction interval, crossing by quadratic);
  keep the two in step.
* Repositories `tiny-LMArena` and `SLMArena` hold the same code; only `config/site.json` and the README title differ.
* Never send the owner's name, email or GitHub handle to external services (User-Agent stays neutral).
