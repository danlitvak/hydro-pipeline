"""Report smoke tests: rendered offline from synthetic statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydro.report import build_report


def _stats_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=10, freq="D").strftime("%Y-%m-%d")
    rows = []
    for station in ["08XX001", "08XX002"]:
        for param in ["level", "discharge"]:
            for i, d in enumerate(dates):
                value = 100.0 + i if param == "discharge" else 2.0 + i / 10
                rows.append(
                    {
                        "station_id": station,
                        "date": d,
                        "parameter": param,
                        "value": value,
                        "roll_mean": value - 0.5,
                        "roll_std": 1.0,
                        "n_obs": min(i + 1, 7),
                        "ci_lo": value - 1.5,
                        "ci_hi": value + 0.5,
                        "zscore": 2.5 if (i == 9 and station == "08XX001") else 0.1,
                        "is_anomaly": 1 if (i == 9 and station == "08XX001") else 0,
                    }
                )
    return pd.DataFrame(rows)


def test_build_report_produces_selfcontained_html(tmp_path, tiny_config):
    out = build_report(tiny_config, _stats_frame(), out_path=tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")

    assert html.startswith("<!DOCTYPE html>")
    assert "Test River at Testville" in html
    assert "MOCK CREEK NEAR MOCKTON" in html
    # plotly is inlined (offline-capable), figures and both map variants present
    assert "plotly" in html.lower()
    assert html.count("js-plotly-plot") >= 1
    assert 'class="map map-light"' in html
    assert 'class="map map-dark"' in html
    # the anomalous station is badged
    assert "badge-anom" in html


def test_build_report_handles_station_without_data(tmp_path, tiny_config):
    stats = _stats_frame()
    stats = stats[stats["station_id"] != "08XX002"]  # second station has nothing
    out = build_report(tiny_config, stats, out_path=tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    assert "no observations in the current window" in html


def test_build_report_all_nan_station(tmp_path, tiny_config):
    stats = _stats_frame()
    stats.loc[stats["station_id"] == "08XX002", "value"] = np.nan
    out = build_report(tiny_config, stats, out_path=tmp_path / "report.html")
    assert out.exists()
