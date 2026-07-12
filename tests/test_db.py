"""Storage tests: idempotent upserts and round-trips through SQLite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydro import db as dbm


def _reading(ts="2026-06-15T00:00:00Z", value=1.0, **kw) -> dbm.RawReading:
    defaults = dict(
        station_id="08XX001",
        ts_utc=ts,
        parameter="level",
        source="realtime",
        value=value,
        unit="m",
        symbol=None,
        fetched_at="2026-07-01T00:00:00Z",
    )
    defaults.update(kw)
    return dbm.RawReading(**defaults)


def test_upsert_is_idempotent(tmp_path):
    conn = dbm.connect(tmp_path / "t.db")
    rows = [_reading(f"2026-06-15T00:{m:02d}:00Z", float(m)) for m in range(5)]
    dbm.upsert_raw(conn, rows)
    dbm.upsert_raw(conn, rows)  # identical second run
    raw = dbm.read_raw(conn)
    assert len(raw) == 5
    conn.close()


def test_upsert_overwrites_revised_values(tmp_path):
    conn = dbm.connect(tmp_path / "t.db")
    dbm.upsert_raw(conn, [_reading(value=1.0)])
    dbm.upsert_raw(conn, [_reading(value=2.5, fetched_at="2026-07-02T00:00:00Z")])
    raw = dbm.read_raw(conn)
    assert len(raw) == 1
    assert raw["value"].iloc[0] == 2.5
    assert raw["fetched_at"].iloc[0] == "2026-07-02T00:00:00Z"
    conn.close()


def test_same_timestamp_different_source_coexists(tmp_path):
    conn = dbm.connect(tmp_path / "t.db")
    dbm.upsert_raw(conn, [
        _reading(value=1.0, source="realtime"),
        _reading(value=1.1, source="daily-mean"),
    ])
    assert len(dbm.read_raw(conn)) == 2
    conn.close()


def test_stats_roundtrip_preserves_nan_as_null(tmp_path):
    conn = dbm.connect(tmp_path / "t.db")
    df = pd.DataFrame(
        {
            "station_id": ["08XX001", "08XX001"],
            "date": ["2026-06-15", "2026-06-16"],
            "parameter": ["level", "level"],
            "value": [1.5, np.nan],
            "roll_mean": [1.4, np.nan],
            "roll_std": [0.1, np.nan],
            "n_obs": [7.0, np.nan],
            "ci_lo": [1.3, np.nan],
            "ci_hi": [1.5, np.nan],
            "zscore": [0.5, np.nan],
            "is_anomaly": [0, 0],
        }
    )
    dbm.replace_daily_stats(conn, df)
    out = dbm.read_daily_stats(conn)
    assert len(out) == 2
    assert out["value"].iloc[0] == 1.5
    assert np.isnan(out["value"].iloc[1])
    assert np.isnan(out["zscore"].iloc[1])
    conn.close()


def test_replace_clean_daily_replaces_not_appends(tmp_path):
    conn = dbm.connect(tmp_path / "t.db")
    df1 = pd.DataFrame(
        [{"station_id": "08XX001", "date": "2026-06-15", "parameter": "level",
          "value": 1.0, "n_samples": 288, "source": "realtime"}]
    )
    df2 = pd.DataFrame(
        [{"station_id": "08XX001", "date": "2026-06-16", "parameter": "level",
          "value": 2.0, "n_samples": 288, "source": "realtime"}]
    )
    dbm.replace_clean_daily(conn, df1)
    dbm.replace_clean_daily(conn, df2)
    out = dbm.read_clean_daily(conn)
    assert out["date"].tolist() == ["2026-06-16"]
    conn.close()
