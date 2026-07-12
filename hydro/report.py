"""Report generation: one self-contained HTML file.

Contents:

* a summary table (latest reading, z-score and status per station),
* a folium map of stations colour-coded by current anomaly status
  (embedded as an ``iframe srcdoc``; light and dark tile variants),
* per-station Plotly time series — daily values, 7-day rolling mean with a
  95 % t-based confidence band, and anomaly markers.

plotly.js is inlined so the charts work offline; map tiles and webfonts
degrade gracefully without a network. Styling follows the house style:
monochrome zinc, square corners, JetBrains Mono headings, light + dark both
supported (the toggle restyles the Plotly figures in place).
"""

from __future__ import annotations

import html as html_mod
import json
from datetime import datetime, timezone
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from hydro.config import Config, Station

PARAM_LABELS = {"discharge": "discharge (m3/s)", "level": "water level (m)"}
PARAM_ORDER = ["discharge", "level"]

ZINC = {
    "band": "rgba(113,113,122,0.16)",
    "daily": "#71717a",
    "rollmean_light": "#18181b",
    "grid_light": "rgba(113,113,122,0.16)",
    "font_light": "#71717a",
    "red": "#dc2626",
}


def _latest_status(stats: pd.DataFrame) -> pd.DataFrame:
    """Per station: latest observed date, per-parameter latest value/z/flag."""
    rows = []
    for station_id, grp in stats.groupby("station_id"):
        obs = grp[grp["value"].notna()]
        if obs.empty:
            rows.append({"station_id": station_id, "date": None})
            continue
        latest_date = obs["date"].max()
        row: dict = {"station_id": station_id, "date": latest_date, "anomaly": False}
        for param in PARAM_ORDER:
            p = obs[(obs["parameter"] == param) & (obs["date"] == latest_date)]
            if p.empty:
                p = obs[obs["parameter"] == param].tail(1)
            if p.empty:
                continue
            r = p.iloc[-1]
            row[f"{param}_value"] = r["value"]
            row[f"{param}_z"] = r["zscore"]
            row["anomaly"] = row["anomaly"] or bool(r["is_anomaly"])
        rows.append(row)
    return pd.DataFrame(rows)


def _make_station_figure(grp: pd.DataFrame, station: Station) -> go.Figure:
    """Two stacked panels (discharge, level) with CI band + anomaly markers."""
    params = [p for p in PARAM_ORDER if not grp[grp["parameter"] == p]["value"].isna().all()]
    fig = make_subplots(
        rows=len(params), cols=1, shared_xaxes=True, vertical_spacing=0.10
    )

    for i, param in enumerate(params, start=1):
        p = grp[grp["parameter"] == param].sort_values("date")
        x = pd.to_datetime(p["date"])
        show_legend = i == 1

        # CI band (upper edge first, then lower filling up to it)
        fig.add_trace(
            go.Scatter(
                x=x, y=p["ci_hi"], mode="lines",
                line={"width": 0, "color": "rgba(0,0,0,0)"},
                hoverinfo="skip", showlegend=False, legendgroup="ci",
            ),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=p["ci_lo"], mode="lines", fill="tonexty",
                fillcolor=ZINC["band"],
                line={"width": 0, "color": "rgba(0,0,0,0)"},
                hoverinfo="skip", name="95% CI (rolling mean)",
                legendgroup="ci", showlegend=show_legend,
            ),
            row=i, col=1,
        )
        # Daily observations
        fig.add_trace(
            go.Scatter(
                x=x, y=p["value"], mode="lines+markers",
                line={"width": 1.1, "color": ZINC["daily"]},
                marker={"size": 3, "color": ZINC["daily"]},
                name="daily value", legendgroup="daily", showlegend=show_legend,
                hovertemplate="%{x|%b %d} - %{y:.3~f}<extra>daily</extra>",
            ),
            row=i, col=1,
        )
        # Rolling mean (meta tag lets the theme toggle recolour it)
        fig.add_trace(
            go.Scatter(
                x=x, y=p["roll_mean"], mode="lines",
                line={"width": 2.2, "color": ZINC["rollmean_light"]},
                meta="rollmean", name="7-day mean", legendgroup="mean",
                showlegend=show_legend,
                hovertemplate="%{x|%b %d} - %{y:.3~f}<extra>7-day mean</extra>",
            ),
            row=i, col=1,
        )
        # Anomalies
        anom = p[p["is_anomaly"] == 1]
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(anom["date"]), y=anom["value"], mode="markers",
                marker={"size": 9, "color": ZINC["red"], "symbol": "circle-open",
                        "line": {"width": 2, "color": ZINC["red"]}},
                name="anomaly (|z| > 2)", legendgroup="anom", showlegend=show_legend,
                customdata=anom["zscore"],
                hovertemplate="%{x|%b %d} - %{y:.3~f} (z = %{customdata:.2f})<extra>anomaly</extra>",
            ),
            row=i, col=1,
        )
        fig.update_yaxes(title_text=PARAM_LABELS[param], row=i, col=1)

    fig.update_layout(
        height=260 * len(params) + 90,
        margin={"l": 64, "r": 16, "t": 8, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "size": 12,
              "color": ZINC["font_light"]},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.06, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(gridcolor=ZINC["grid_light"], zeroline=False, linecolor=ZINC["grid_light"])
    fig.update_yaxes(gridcolor=ZINC["grid_light"], zeroline=False, linecolor=ZINC["grid_light"])
    return fig


def _folium_map(cfg: Config, status: pd.DataFrame, dark: bool) -> str:
    """Render the station map to a standalone HTML string."""
    lats = [s.lat for s in cfg.stations]
    lons = [s.lon for s in cfg.stations]
    fmap = folium.Map(
        location=[sum(lats) / len(lats), sum(lons) / len(lons)],
        zoom_start=6,
        tiles="cartodbdark_matter" if dark else "cartodbpositron",
        attr=None,
    )
    by_id = {r["station_id"]: r for _, r in status.iterrows()} if not status.empty else {}
    for s in cfg.stations:
        st = by_id.get(s.id, {})
        anomalous = bool(st.get("anomaly", False))
        colour = ZINC["red"] if anomalous else ("#a1a1aa" if dark else "#52525b")
        bits = [f"<b>{html_mod.escape(s.name)}</b>", f"{s.id}"]
        for param in PARAM_ORDER:
            v = st.get(f"{param}_value")
            z = st.get(f"{param}_z")
            if v is not None and not pd.isna(v):
                unit = "m3/s" if param == "discharge" else "m"
                ztxt = f", z = {z:+.2f}" if z is not None and not pd.isna(z) else ""
                bits.append(f"{param}: {v:,.2f} {unit}{ztxt}")
        bits.append("status: " + ("ANOMALY" if anomalous else "normal"))
        folium.CircleMarker(
            location=[s.lat, s.lon],
            radius=8,
            color=colour,
            weight=2,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85 if anomalous else 0.55,
            tooltip="<br>".join(bits),
        ).add_to(fmap)
    return fmap.get_root().render()


def _fmt(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:,.{digits}f}"


def _summary_table(cfg: Config, status: pd.DataFrame) -> str:
    by_id = {r["station_id"]: r for _, r in status.iterrows()} if not status.empty else {}
    rows = []
    for s in cfg.stations:
        st = by_id.get(s.id, {})
        anomalous = bool(st.get("anomaly", False))
        badge = (
            '<span class="badge badge-anom">anomaly</span>'
            if anomalous
            else ('<span class="badge">normal</span>' if st.get("date") else '<span class="badge">no data</span>')
        )
        zs = [st.get("discharge_z"), st.get("level_z")]
        zs = [z for z in zs if z is not None and not pd.isna(z)]
        max_z = max(zs, key=abs) if zs else None
        rows.append(
            "<tr>"
            f'<td><a href="#st-{s.id}">{html_mod.escape(s.name)}</a></td>'
            f'<td class="mono">{s.id}</td>'
            f'<td>{st.get("date") or "-"}</td>'
            f'<td class="num">{_fmt(st.get("discharge_value"))}</td>'
            f'<td class="num">{_fmt(st.get("level_value"), 3)}</td>'
            f'<td class="num">{_fmt(max_z)}</td>'
            f"<td>{badge}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr><th>station</th><th>id</th><th>latest day</th>'
        "<th>discharge (m3/s)</th><th>level (m)</th><th>max |z|</th><th>status</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


CSS = """
:root {
  --bg: #fafafa; --fg: #18181b; --muted: #71717a; --border: #e4e4e7;
  --card: #ffffff; --destructive: #dc2626;
}
:root[data-theme="dark"] {
  --bg: #09090b; --fg: #f4f4f5; --muted: #a1a1aa; --border: #27272a;
  --card: #111113; --destructive: #ef4444;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--fg);
  font-family: Inter, system-ui, -apple-system, sans-serif;
  font-size: 15px; line-height: 1.55;
  transition: background .2s ease, color .2s ease;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 24px 80px; }
h1, h2, h3 { font-family: "JetBrains Mono", ui-monospace, monospace; font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 48px 0 12px; border-top: 1px solid var(--border); padding-top: 28px; }
.mono { font-family: "JetBrains Mono", ui-monospace, monospace; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 32px; }
.meta .sep { margin: 0 8px; opacity: .5; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
th { font-family: "JetBrains Mono", ui-monospace, monospace; font-weight: 500; font-size: 12px;
     color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
td.num { font-variant-numeric: tabular-nums; }
a { color: var(--fg); text-decoration: underline; text-decoration-color: var(--muted); text-underline-offset: 3px; }
.badge { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 11px; text-transform: uppercase;
  letter-spacing: .05em; border: 1px solid var(--border); padding: 2px 8px; color: var(--muted); }
.badge-anom { color: var(--destructive); border-color: var(--destructive); }
.card { border: 1px solid var(--border); background: var(--card); padding: 16px; margin: 16px 0; }
.station-sub { color: var(--muted); font-size: 13px; margin: -6px 0 10px; }
iframe.map { width: 100%; height: 520px; border: 1px solid var(--border); display: block; background: var(--card); }
:root[data-theme="dark"] .map-light { display: none; }
:root:not([data-theme="dark"]) .map-dark { display: none; }
.toggle { position: fixed; top: 16px; right: 16px; z-index: 10;
  font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12px;
  background: color-mix(in oklab, var(--card) 50%, transparent);
  backdrop-filter: blur(6px);
  color: var(--fg); border: 1px solid var(--border); padding: 6px 12px; cursor: pointer; }
.toggle:hover { border-color: var(--muted); }
.foot { color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--border); margin-top: 56px; padding-top: 20px; }
.foot p { margin: 6px 0; }
@media (prefers-reduced-motion: reduce) { body { transition: none; } }
"""

THEME_JS = """
(function () {
  var KEY = "hydro-theme";
  function systemDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function current() {
    var saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    return saved || (systemDark() ? "dark" : "light");
  }
  function restylePlots(dark) {
    if (typeof Plotly === "undefined") return;
    var font = dark ? "#a1a1aa" : "#71717a";
    var grid = dark ? "rgba(161,161,170,0.14)" : "rgba(113,113,122,0.16)";
    var mean = dark ? "#f4f4f5" : "#18181b";
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      var re = { "font.color": font };
      Object.keys(gd.layout || {}).forEach(function (k) {
        if (/^[xy]axis\\d*$/.test(k)) {
          re[k + ".gridcolor"] = grid;
          re[k + ".linecolor"] = grid;
        }
      });
      Plotly.relayout(gd, re);
      (gd.data || []).forEach(function (tr, i) {
        if (tr.meta === "rollmean") Plotly.restyle(gd, { "line.color": mean }, [i]);
      });
    });
  }
  function apply(mode, persist) {
    if (mode === "dark") document.documentElement.setAttribute("data-theme", "dark");
    else document.documentElement.removeAttribute("data-theme");
    if (persist) { try { localStorage.setItem(KEY, mode); } catch (e) {} }
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.textContent = mode === "dark" ? "light" : "dark";
    restylePlots(mode === "dark");
  }
  // This script sits at the end of <body>, so the DOM (and every Plotly
  // figure's init script) has already run: apply immediately.
  apply(current(), false);
  var btn = document.getElementById("theme-toggle");
  if (btn) btn.addEventListener("click", function () {
    apply(current() === "dark" ? "light" : "dark", true);
  });
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      var saved = null;
      try { saved = localStorage.getItem(KEY); } catch (e) {}
      if (!saved) apply(current(), false);
    });
  }
})();
"""


def build_report(
    cfg: Config,
    stats: pd.DataFrame,
    out_path: str | Path = "report.html",
    generated_at: datetime | None = None,
) -> Path:
    """Assemble the full HTML report and write it to ``out_path``."""
    generated_at = generated_at or datetime.now(timezone.utc)
    status = _latest_status(stats)

    n_anom = int(stats["is_anomaly"].sum()) if not stats.empty else 0
    n_obs = int(stats["value"].notna().sum()) if not stats.empty else 0
    dates = stats.loc[stats["value"].notna(), "date"] if not stats.empty else pd.Series(dtype=str)
    span = f"{dates.min()} to {dates.max()}" if not dates.empty else "no data"

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">")
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append("<title>BC Hydrometric Monitor</title>")
    parts.append(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600'
        '&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
    )
    parts.append(f"<style>{CSS}</style>")
    parts.append(f"<script>{get_plotlyjs()}</script>")
    parts.append("</head><body>")
    parts.append('<button class="toggle" id="theme-toggle" aria-label="toggle theme">dark</button>')
    parts.append('<div class="wrap">')

    parts.append("<h1>BC HYDROMETRIC MONITOR</h1>")
    parts.append(
        '<p class="meta">'
        f"{len(cfg.stations)} stations<span class=\"sep\">/</span>"
        f"{n_obs} station-days<span class=\"sep\">/</span>"
        f"{n_anom} anomalies flagged<span class=\"sep\">/</span>"
        f"window {span}<span class=\"sep\">/</span>"
        f"generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}<span class=\"sep\">/</span>"
        "source: ECCC MSC GeoMet hydrometric API"
        "</p>"
    )

    parts.append("<h2>SUMMARY</h2>")
    parts.append(_summary_table(cfg, status))

    parts.append("<h2>STATION MAP</h2>")
    for variant, dark in (("map-light", False), ("map-dark", True)):
        map_html = _folium_map(cfg, status, dark=dark)
        parts.append(
            f'<iframe class="map {variant}" loading="lazy" title="station map" '
            f'srcdoc="{html_mod.escape(map_html, quote=True)}"></iframe>'
        )
    parts.append(
        '<p class="meta">stations red when the latest observation of either parameter '
        "deviates more than 2 sigma from its trailing 7-day baseline (tiles need a network connection)</p>"
    )

    for s in cfg.stations:
        grp = stats[stats["station_id"] == s.id]
        parts.append(f'<h2 id="st-{s.id}">{html_mod.escape(s.name.upper())}</h2>')
        st = status[status["station_id"] == s.id] if not status.empty else pd.DataFrame()
        latest = st.iloc[0]["date"] if not st.empty and st.iloc[0].get("date") else None
        parts.append(
            f'<p class="station-sub mono">{s.id} - {html_mod.escape(s.river)} - '
            f'{s.lat:.4f}, {s.lon:.4f}' + (f" - latest {latest}" if latest else "") + "</p>"
        )
        if grp.empty or grp["value"].isna().all():
            parts.append('<p class="meta">no observations in the current window.</p>')
            continue
        fig = _make_station_figure(grp, s)
        parts.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"displayModeBar": False, "responsive": True},
            )
        )

    parts.append('<div class="foot">')
    parts.append(
        "<p>METHOD - Daily series: 5/15-minute provisional readings averaged per local "
        "calendar day (days with too few samples dropped); validated daily means take "
        "precedence when published; interior gaps up to 2 days linearly interpolated.</p>"
        "<p>Rolling mean and sample std over a trailing 7-day window. The shaded band is a "
        "t-based 95% confidence interval for the window mean (t quantile with n-1 degrees "
        "of freedom - not z, since n is at most 7). Anomalies: |z| &gt; 2 against the "
        "previous day's baseline, so a spike is not absorbed into its own yardstick. "
        "Consecutive days are autocorrelated, so treat both as screening statistics.</p>"
        "<p>Data: Environment and Climate Change Canada, MSC GeoMet OGC API "
        '(<span class="mono">api.weather.gc.ca</span>) - hydrometric-realtime and '
        "hydrometric-daily-mean collections. Provisional data subject to revision.</p>"
    )
    parts.append("</div></div>")
    parts.append(f"<script>{THEME_JS}</script>")
    parts.append("</body></html>")

    out = Path(out_path)
    out.write_text("".join(parts), encoding="utf-8")
    return out
