"""Unit tests for the cleaning rules — synthetic frames only, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydro.clean import (
    aggregate_daily,
    align_timestamps,
    clean,
    coerce_numeric,
    daily_mean_rows,
    fill_small_gaps,
    merge_sources,
    normalize_units,
)
from tests.conftest import make_raw

TZ = "America/Vancouver"


# --- coercion -------------------------------------------------------------

def test_coerce_numeric_turns_garbage_into_nan():
    df = make_raw(
        [
            {"ts_utc": "2026-06-15T00:00:00Z", "value": "1.5"},
            {"ts_utc": "2026-06-15T00:05:00Z", "value": None},
            {"ts_utc": "2026-06-15T00:10:00Z", "value": "not-a-number"},
            {"ts_utc": "2026-06-15T00:15:00Z", "value": 2},
        ]
    )
    out = coerce_numeric(df)
    assert out["value"].tolist()[0] == pytest.approx(1.5)
    assert np.isnan(out["value"].iloc[1])
    assert np.isnan(out["value"].iloc[2])
    assert out["value"].iloc[3] == pytest.approx(2.0)
    assert out["value"].dtype == np.float64


# --- unit normalization ----------------------------------------------------

def test_normalize_units_converts_cm_level_to_metres():
    df = make_raw([{"ts_utc": "2026-06-15T00:00:00Z", "value": 150.0, "unit": "cm"}])
    out = normalize_units(df)
    assert out["value"].iloc[0] == pytest.approx(1.5)
    assert out["unit"].iloc[0] == "m"


def test_normalize_units_converts_litres_per_second_discharge():
    df = make_raw(
        [{"ts_utc": "2026-06-15T00:00:00Z", "value": 2500.0,
          "unit": "L/s", "parameter": "discharge"}]
    )
    out = normalize_units(df)
    assert out["value"].iloc[0] == pytest.approx(2.5)
    assert out["unit"].iloc[0] == "m3/s"


def test_normalize_units_leaves_canonical_untouched():
    df = make_raw(
        [
            {"ts_utc": "2026-06-15T00:00:00Z", "value": 3.25, "unit": "m"},
            {"ts_utc": "2026-06-15T00:00:00Z", "value": 810.0,
             "unit": "m3/s", "parameter": "discharge"},
        ]
    )
    out = normalize_units(df)
    assert out["value"].tolist() == pytest.approx([3.25, 810.0])


def test_normalize_units_rejects_unknown_unit():
    df = make_raw([{"ts_utc": "2026-06-15T00:00:00Z", "value": 1.0, "unit": "fathoms"}])
    with pytest.raises(ValueError, match="fathoms"):
        normalize_units(df)


# --- timestamp alignment ----------------------------------------------------

def test_align_timestamps_parses_utc_and_sorts():
    df = make_raw(
        [
            {"ts_utc": "2026-06-15T00:10:00Z", "value": 2.0},
            {"ts_utc": "2026-06-15T00:00:00Z", "value": 1.0},
        ]
    )
    out = align_timestamps(df)
    assert str(out["ts"].dt.tz) == "UTC"
    assert out["ts"].is_monotonic_increasing
    assert out["value"].tolist() == [1.0, 2.0]


def test_align_timestamps_keeps_latest_fetch_for_duplicates():
    df = make_raw(
        [
            {"ts_utc": "2026-06-15T00:00:00Z", "value": 1.0, "fetched_at": "2026-06-15T01:00:00Z"},
            {"ts_utc": "2026-06-15T00:00:00Z", "value": 9.9, "fetched_at": "2026-06-16T01:00:00Z"},
        ]
    )
    out = align_timestamps(df)
    assert len(out) == 1
    assert out["value"].iloc[0] == pytest.approx(9.9)


# --- daily aggregation -------------------------------------------------------

def test_aggregate_daily_full_day_mean(realtime_day):
    df = align_timestamps(realtime_day)
    out = aggregate_daily(df, tz=TZ, min_samples=24)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["date"] == "2026-06-15"
    assert row["value"] == pytest.approx(2.0)
    assert row["n_samples"] == 288
    assert row["source"] == "realtime"


def test_aggregate_daily_drops_sparse_days():
    ts = pd.date_range("2026-06-15 08:00", periods=10, freq="5min", tz="UTC")
    df = align_timestamps(
        make_raw([{"ts_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": 5.0} for t in ts])
    )
    out = aggregate_daily(df, tz=TZ, min_samples=24)
    assert out.empty


def test_aggregate_daily_buckets_by_local_calendar_day():
    # 06:00 UTC on Jan 2 is 22:00 PST on Jan 1 — must land in the Jan 1 bucket.
    df = align_timestamps(make_raw([{"ts_utc": "2026-01-02T06:00:00Z", "value": 7.0}]))
    out = aggregate_daily(df, tz=TZ, min_samples=1)
    assert out["date"].tolist() == ["2026-01-01"]


# --- validated daily means -----------------------------------------------------

def test_daily_mean_rows_take_date_verbatim():
    # Stored at midnight UTC; the published DATE must not be timezone-shifted.
    df = align_timestamps(
        make_raw([{"ts_utc": "2024-05-01T00:00:00Z", "value": 123.0,
                   "source": "daily-mean", "parameter": "discharge", "unit": "m3/s"}])
    )
    out = daily_mean_rows(df)
    assert out["date"].tolist() == ["2024-05-01"]
    assert out["source"].tolist() == ["daily-mean"]


def test_merge_sources_validated_wins():
    validated = pd.DataFrame(
        [{"station_id": "08XX001", "date": "2026-06-15", "parameter": "level",
          "value": 1.11, "n_samples": 1, "source": "daily-mean"}]
    )
    provisional = pd.DataFrame(
        [
            {"station_id": "08XX001", "date": "2026-06-15", "parameter": "level",
             "value": 9.99, "n_samples": 288, "source": "realtime"},
            {"station_id": "08XX001", "date": "2026-06-16", "parameter": "level",
             "value": 2.22, "n_samples": 288, "source": "realtime"},
        ]
    )
    out = merge_sources(validated, provisional)
    assert len(out) == 2
    day1 = out[out["date"] == "2026-06-15"].iloc[0]
    assert day1["value"] == pytest.approx(1.11)
    assert day1["source"] == "daily-mean"


# --- gap filling ---------------------------------------------------------------

def _daily_series(values):
    idx = pd.date_range("2026-06-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_fill_small_gaps_interpolates_short_gap():
    s = _daily_series([2.0, 2.0, np.nan, np.nan, 8.0, 8.0])
    filled, mask = fill_small_gaps(s, max_gap_days=2)
    assert filled.iloc[2] == pytest.approx(4.0)
    assert filled.iloc[3] == pytest.approx(6.0)
    assert mask.tolist() == [False, False, True, True, False, False]


def test_fill_small_gaps_leaves_long_gap_missing():
    s = _daily_series([2.0, np.nan, np.nan, np.nan, 8.0])
    filled, mask = fill_small_gaps(s, max_gap_days=2)
    assert filled.isna().sum() == 3
    assert not mask.any()


def test_fill_small_gaps_never_extrapolates_edges():
    s = _daily_series([np.nan, 3.0, 4.0, np.nan])
    filled, mask = fill_small_gaps(s, max_gap_days=2)
    assert np.isnan(filled.iloc[0])
    assert np.isnan(filled.iloc[-1])
    assert not mask.any()


# --- full stage ------------------------------------------------------------------

def test_clean_end_to_end_prefers_validated_and_fills_gaps():
    rows = []
    # three full realtime days (June 15, 16, 18 — June 17 missing entirely)
    for day, val in [("2026-06-15", 2.0), ("2026-06-16", 4.0), ("2026-06-18", 8.0)]:
        ts = pd.date_range(f"{day} 07:00", periods=288, freq="5min", tz="UTC")
        rows += [{"ts_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": val} for t in ts]
    df = make_raw(rows)
    # validated daily mean also published for June 15 with a revised value
    df = pd.concat(
        [df, make_raw([{"ts_utc": "2026-06-15T00:00:00Z", "value": 2.5,
                        "source": "daily-mean"}])],
        ignore_index=True,
    )

    out = clean(df, tz=TZ, min_daily_samples=24, max_gap_days=2)
    out = out.set_index("date")

    assert out.loc["2026-06-15", "value"] == pytest.approx(2.5)      # validated wins
    assert out.loc["2026-06-15", "source"] == "daily-mean"
    assert out.loc["2026-06-16", "value"] == pytest.approx(4.0)
    assert out.loc["2026-06-17", "value"] == pytest.approx(6.0)      # interpolated
    assert out.loc["2026-06-17", "source"] == "filled"
    assert out.loc["2026-06-18", "value"] == pytest.approx(8.0)


def test_clean_empty_frame_is_fine():
    out = clean(pd.DataFrame())
    assert out.empty


def test_clean_realtime_only_no_validated_rows():
    """Regression: with zero daily-mean rows (the normal case for a recent
    window — validated data lags by over a year) the empty 'validated' frame
    must not degrade the value column to object dtype."""
    rows = []
    for day, val in [("2026-06-15", 2.0), ("2026-06-16", 4.0), ("2026-06-18", 8.0)]:
        ts = pd.date_range(f"{day} 07:00", periods=288, freq="5min", tz="UTC")
        rows += [{"ts_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": val} for t in ts]
    out = clean(make_raw(rows), tz=TZ, min_daily_samples=24, max_gap_days=2)
    assert out["value"].dtype == np.float64
    assert len(out) == 4  # 3 observed days + 1 interpolated
    assert out.set_index("date").loc["2026-06-17", "source"] == "filled"
