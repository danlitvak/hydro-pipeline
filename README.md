# hydro-pipeline

Data pipeline for BC river monitoring: ingests live water level and discharge
data for 8 Water Survey of Canada stations from the ECCC MSC GeoMet API,
stores it in SQLite, computes rolling statistics with z-score anomaly flags
and t-based confidence intervals, and renders a self-contained HTML report —
Plotly time series per station plus a folium map colour-coded by current
anomaly status.

![report screenshot](docs/screenshot.png)
*(screenshot placeholder — open [`report.html`](report.html) for the live artifact)*

## Architecture

```
stations.toml            ECCC GeoMet OGC API (api.weather.gc.ca)
     │                     hydrometric-realtime  (5-min, ~30-day retention)
     │                     hydrometric-daily-mean (validated, long lag)
     ▼                              │
python -m hydro fetch  ◄────────────┘        paged, retried, User-Agent,
     │                                       polite request pacing
     ▼
data/hydro.db :: raw_readings                idempotent upsert on
     │                                       (station, timestamp, parameter, source)
     ▼
python -m hydro analyze
     │   clean.py    coerce → normalize units → align timestamps →
     │               daily aggregation (local calendar days) →
     │               merge sources (validated wins) → fill short gaps
     │   analyze.py  7-day rolling mean/std → t-based 95% CI →
     │               z-score vs previous-day baseline → |z| > 2 flags
     ▼
data/hydro.db :: clean_daily, daily_stats
     │
     ▼
python -m hydro report  ──►  report.html     (Plotly inlined — works offline;
                                              folium map, light + dark)
```

Every transform is a pure function on pandas objects; all statistics and
cleaning rules are covered by pytest with deterministic synthetic fixtures
(the test suite never touches the network).

## Stations

All 8 station IDs were verified against the live API before being locked into
`stations.toml` — each returns current realtime level *and* discharge data.

| WSC ID  | Station                                     | Regime notes                    |
| ------- | ------------------------------------------- | ------------------------------- |
| 08MF005 | Fraser River at Hope                        | large basin, snowmelt freshet   |
| 08GA010 | Capilano River above Intake                 | coastal, rain-driven            |
| 08HD035 | Campbell River near Campbell River Cableway | Vancouver Island, regulated     |
| 08GA022 | Squamish River near Brackendale             | coast mountains, rain + melt    |
| 08MG005 | Lillooet River near Pemberton               | glacier-fed                     |
| 08LF051 | Thompson River near Spences Bridge          | large interior tributary        |
| 08MH001 | Chilliwack River at Vedder Crossing         | flashy, rain-driven             |
| 08NM050 | Okanagan River at Penticton                 | regulated, dry interior         |

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev]"

python -m hydro fetch      # ingest the configured window into data/hydro.db
python -m hydro analyze    # clean + compute rolling stats and anomaly flags
python -m hydro report     # render report.html

python -m pytest           # run the test suite (offline)
```

Useful flags: `--config` (alternate TOML), `--db` (alternate SQLite path),
`fetch --window-days N`, `report --out PATH`.

## Methodology

**Cleaning.** Provisional realtime readings arrive every 5–15 minutes with
UTC timestamps. They are averaged into daily means per *local* calendar day
(America/Vancouver — a 06:00 UTC sample belongs to the previous BC day), and
a day with fewer than `min_daily_samples` readings is treated as missing
rather than reported as a falsely-precise "mean". Units are normalized to
metres and m³/s through an explicit conversion table; an unrecognized unit
raises instead of passing through unscaled. When ECCC's *validated* daily
means overlap the window, they take precedence over provisional aggregates.
Interior gaps up to 2 days are linearly interpolated (and labelled `filled`);
longer outages stay missing — interpolating across a week would fabricate a
hydrograph, whereas missing days just widen the confidence interval via a
smaller n.

**Rolling statistics.** For each day, the trailing 7 days form a small sample
from the river's "current regime": sample mean x̄ₜ and sample standard
deviation sₜ (ddof = 1). Windows with fewer than 4 observations are not
reported.

**Confidence intervals (STATS 302 reasoning).** With n ≤ 7 and unknown
population variance, (x̄ − μ)/(s/√n) is approximately Student-t with n−1
degrees of freedom, so the shaded band is

> x̄ₜ ± t₀.₉₇₅,ₙ₋₁ · sₜ/√n

using the t quantile rather than z = 1.96 — at n = 7 the multiplier is 2.447,
about 25 % wider, which is material at these sample sizes. Missing days
shrink n and honestly widen the band. Caveat, stated rather than hidden:
consecutive daily river observations are positively autocorrelated, so the
effective sample size is below n and the band is somewhat narrower than a
fully honest interval; it should be read as descriptive uncertainty of the
weekly running mean, not as inference about independent draws.

**Anomaly detection.** Each day is scored against the *previous* day's
rolling baseline:

> zₜ = (xₜ − x̄ₜ₋₁) / sₜ₋₁

Excluding today from its own baseline matters — a spike included in its own
window drags the mean toward itself and inflates σ, muting exactly the signal
being tested for. Under an approximately normal baseline P(|Z| > 2) ≈ 4.6 %,
so `|z| > 2` flags roughly the most surprising ~5 % of days. Because river
series are autocorrelated and non-stationary (freshets, storms, dam
operations), this is a *screening rule* with a deliberate threshold, not a
calibrated hypothesis test — the report presents flags as "worth a look",
with the z value attached. A constant week (σ = 0) yields no z-score at all
rather than division noise.

## Data source quirks worth knowing

* `hydrometric-realtime` retains only ~30 days, at 5-minute (sometimes
  15-minute) resolution, and the data is **provisional** — values can be
  revised. Upserts on `(station, timestamp, parameter, source)` make
  re-fetches idempotent and let revisions overwrite in place.
* `hydrometric-daily-mean` is **validated** but published with a lag of a
  year or more. The pipeline requests the full window from both collections;
  whenever validated data eventually overlaps, it silently wins during
  cleaning.
* Consequence: a single fetch yields ~30 days of usable history, and the
  90-day window fills up across repeated runs because SQLite accumulates
  beyond the API's retention (the weekly CI run carries the database forward
  via the actions cache).
* Some stations publish only water level (e.g. tidal reaches); every station
  in `stations.toml` was chosen to provide both parameters.

## CI

* **tests** — pytest on every push / PR.
* **weekly-report** — Mondays: restore cached SQLite → `fetch` → `analyze` →
  `report` → commit the refreshed `report.html`. Public data, zero secrets.

## Layout

```
hydro/            package (config, api, db, ingest, clean, analyze, report, cli)
stations.toml     verified stations + API/pipeline/analysis settings
tests/            offline unit tests with synthetic fixtures
report.html       generated artifact (committed, refreshed weekly)
data/hydro.db     local SQLite (gitignored)
```

Data: Environment and Climate Change Canada, MSC GeoMet OGC API
(`api.weather.gc.ca`). Provisional hydrometric data subject to revision.
