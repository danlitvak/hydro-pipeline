"""Unit tests for the statistics layer — deterministic, hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hydro.analyze import (
    analyze,
    analyze_series,
    ci_bounds,
    flag_anomalies,
    rolling_stats,
    t_multiplier,
    zscores,
)


def _series(values, start="2026-06-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def _daily_frame(values, station="08XX001", parameter="level", start="2026-06-01"):
    dates = pd.date_range(start, periods=len(values), freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "station_id": station,
            "date": dates,
            "parameter": parameter,
            "value": values,
            "n_samples": 288,
            "source": "realtime",
        }
    )


# --- rolling statistics ------------------------------------------------------

def test_rolling_mean_and_std_match_hand_computation():
    s = _series(range(1, 11))  # 1..10
    rs = rolling_stats(s, window=7, min_periods=4)
    # day 7 window = 1..7: mean 4, sample var = 28/6
    assert rs["roll_mean"].iloc[6] == pytest.approx(4.0)
    assert rs["roll_std"].iloc[6] == pytest.approx(math.sqrt(28 / 6))
    assert rs["n_obs"].iloc[6] == 7
    # day 10 window = 4..10: mean 7
    assert rs["roll_mean"].iloc[9] == pytest.approx(7.0)


def test_rolling_stats_respect_min_periods():
    s = _series(range(1, 11))
    rs = rolling_stats(s, window=7, min_periods=4)
    assert np.isnan(rs["roll_mean"].iloc[2])   # only 3 obs
    assert np.isnan(rs["n_obs"].iloc[2])
    assert rs["roll_mean"].iloc[3] == pytest.approx(2.5)  # first 4 obs


def test_rolling_stats_count_ignores_missing_days():
    s = _series([1, 2, np.nan, 4, 5, 6, 7, 8])
    rs = rolling_stats(s, window=7, min_periods=4)
    # day 7 window has one NaN -> 6 observations
    assert rs["n_obs"].iloc[6] == 6
    assert rs["roll_mean"].iloc[6] == pytest.approx((1 + 2 + 4 + 5 + 6 + 7) / 6)


# --- confidence intervals -------------------------------------------------------

def test_t_multiplier_matches_tables():
    # classic table values: t_{0.975, 6} = 2.447, t_{0.975, 3} = 3.182
    assert t_multiplier(np.array([7]), 0.95)[0] == pytest.approx(2.4469, abs=1e-3)
    assert t_multiplier(np.array([4]), 0.95)[0] == pytest.approx(3.1824, abs=1e-3)


def test_t_multiplier_undefined_below_two_observations():
    out = t_multiplier(np.array([1, 0]), 0.95)
    assert np.isnan(out).all()


def test_ci_bounds_hand_computed():
    mean = pd.Series([10.0])
    std = pd.Series([1.0])
    n = pd.Series([7.0])
    lo, hi = ci_bounds(mean, std, n, ci_level=0.95)
    half = 2.446912 * 1.0 / math.sqrt(7)
    assert lo.iloc[0] == pytest.approx(10 - half, abs=1e-4)
    assert hi.iloc[0] == pytest.approx(10 + half, abs=1e-4)


def test_ci_widens_when_observations_drop_out():
    # same mean/std, fewer obs -> wider interval (bigger t, bigger sqrt(n) penalty)
    mean = pd.Series([10.0, 10.0])
    std = pd.Series([1.0, 1.0])
    n = pd.Series([7.0, 4.0])
    lo, hi = ci_bounds(mean, std, n, ci_level=0.95)
    assert (hi.iloc[1] - lo.iloc[1]) > (hi.iloc[0] - lo.iloc[0])


# --- z-scores and anomaly flags ---------------------------------------------------

def test_zscore_uses_previous_day_baseline():
    # alternating 9/11 for 7 days, then a 25.0 spike
    base = [9, 11, 9, 11, 9, 11, 9]
    s = _series(base + [25.0])
    rs = rolling_stats(s, window=7, min_periods=4)
    z = zscores(s, rs["roll_mean"], rs["roll_std"])

    mean7 = sum(base) / 7                      # 9.857142...
    var7 = sum((x - mean7) ** 2 for x in base) / 6
    expected = (25.0 - mean7) / math.sqrt(var7)
    assert z.iloc[7] == pytest.approx(expected, abs=1e-6)
    assert abs(z.iloc[7]) > 10  # the spike is not absorbed into its own baseline


def test_constant_series_yields_no_zscores_or_flags():
    s = _series([5.0] * 10)
    rs = rolling_stats(s, window=7, min_periods=4)
    z = zscores(s, rs["roll_mean"], rs["roll_std"])
    flags = flag_anomalies(z, threshold=2.0)
    assert z.isna().all()          # sigma == 0 -> no yardstick, not division noise
    assert flags.sum() == 0


def test_flag_threshold_is_strict_and_two_sided():
    z = pd.Series([2.0, 2.05, -2.05, np.nan, 0.0])
    flags = flag_anomalies(z, threshold=2.0)
    assert flags.tolist() == [0, 1, 1, 0, 0]


# --- driver -------------------------------------------------------------------------

def test_analyze_series_reindexes_calendar_gaps():
    df = _daily_frame([1.0, 2.0, 3.0])
    df = df[df["date"] != "2026-06-02"]  # drop the middle day from the *input*
    out = analyze_series(df, window=7, min_periods=2)
    assert out["date"].tolist() == ["2026-06-01", "2026-06-02", "2026-06-03"]
    assert np.isnan(out.loc[out["date"] == "2026-06-02", "value"]).all()
    assert out.loc[out["date"] == "2026-06-02", "is_anomaly"].iloc[0] == 0


def test_analyze_groups_by_station_and_parameter():
    a = _daily_frame([1, 2, 3, 4, 5, 6, 7], station="08XX001", parameter="level")
    b = _daily_frame([10, 20, 30, 40, 50, 60, 70], station="08XX002", parameter="discharge")
    out = analyze(pd.concat([a, b], ignore_index=True), window=7, min_periods=4)
    assert set(out["station_id"]) == {"08XX001", "08XX002"}
    got_a = out[out["station_id"] == "08XX001"]
    assert got_a["roll_mean"].iloc[-1] == pytest.approx(4.0)
    got_b = out[out["station_id"] == "08XX002"]
    assert got_b["roll_mean"].iloc[-1] == pytest.approx(40.0)


def test_analyze_flags_spike_in_realistic_frame():
    values = [100, 104, 98, 102, 99, 103, 101, 100, 102, 180.0]  # jump at the end
    out = analyze(_daily_frame(values), window=7, min_periods=4, z_threshold=2.0)
    assert out["is_anomaly"].iloc[-1] == 1
    assert out["is_anomaly"].iloc[:-1].sum() == 0
    assert out["zscore"].iloc[-1] > 2


def test_analyze_empty_input():
    out = analyze(pd.DataFrame(columns=["station_id", "date", "parameter", "value"]))
    assert out.empty
