"""SQLite storage.

Three tables, one per pipeline stage:

* ``raw_readings`` — readings exactly as returned by the API (one row per
  station / timestamp / parameter / source). Idempotent upserts: re-fetching
  the same window never duplicates rows, and revised values overwrite.
* ``clean_daily``  — one value per station / local calendar day / parameter
  after cleaning (aggregation, source merging, small-gap interpolation).
* ``daily_stats``  — rolling statistics, confidence intervals and anomaly
  flags computed from ``clean_daily``.

The ``analyze`` stage rebuilds ``clean_daily`` and ``daily_stats`` from raw on
every run, so raw data remains the only stateful layer.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# pandas hands back numpy scalars; sqlite3 only accepts Python natives
# (np.float64 subclasses float, but np.int64 does not subclass int).
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.float32, float)

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_readings (
    station_id TEXT NOT NULL,
    ts_utc     TEXT NOT NULL,   -- ISO-8601 UTC timestamp
    parameter  TEXT NOT NULL,   -- 'level' | 'discharge'
    source     TEXT NOT NULL,   -- 'realtime' | 'daily-mean'
    value      REAL,
    unit       TEXT NOT NULL,   -- unit as reported ('m', 'm3/s')
    symbol     TEXT,            -- ECCC qualifier (ice, estimated, ...)
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (station_id, ts_utc, parameter, source)
);
CREATE INDEX IF NOT EXISTS idx_raw_station_param
    ON raw_readings (station_id, parameter, ts_utc);

CREATE TABLE IF NOT EXISTS clean_daily (
    station_id TEXT NOT NULL,
    date       TEXT NOT NULL,   -- local calendar date (YYYY-MM-DD)
    parameter  TEXT NOT NULL,
    value      REAL,
    n_samples  INTEGER NOT NULL,
    source     TEXT NOT NULL,   -- 'realtime' | 'daily-mean' | 'filled'
    PRIMARY KEY (station_id, date, parameter)
);

CREATE TABLE IF NOT EXISTS daily_stats (
    station_id TEXT NOT NULL,
    date       TEXT NOT NULL,
    parameter  TEXT NOT NULL,
    value      REAL,
    roll_mean  REAL,
    roll_std   REAL,
    n_obs      INTEGER,
    ci_lo      REAL,
    ci_hi      REAL,
    zscore     REAL,
    is_anomaly INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (station_id, date, parameter)
);
"""


@dataclass(frozen=True)
class RawReading:
    station_id: str
    ts_utc: str
    parameter: str
    source: str
    value: float | None
    unit: str
    symbol: str | None
    fetched_at: str


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the pipeline database."""
    p = Path(path)
    if p.parent and str(p.parent) not in (".", ""):
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert_raw(conn: sqlite3.Connection, rows: Iterable[RawReading]) -> int:
    """Insert-or-update raw readings. Returns the number of rows written.

    The primary key (station_id, ts_utc, parameter, source) makes re-runs
    idempotent: the same reading fetched twice results in one row, and a
    revised value for an existing timestamp overwrites the stale one.
    """
    rows = list(rows)
    conn.executemany(
        """
        INSERT INTO raw_readings
            (station_id, ts_utc, parameter, source, value, unit, symbol, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (station_id, ts_utc, parameter, source) DO UPDATE SET
            value = excluded.value,
            unit = excluded.unit,
            symbol = excluded.symbol,
            fetched_at = excluded.fetched_at
        """,
        [
            (r.station_id, r.ts_utc, r.parameter, r.source, r.value, r.unit, r.symbol, r.fetched_at)
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def read_raw(conn: sqlite3.Connection, station_id: str | None = None) -> pd.DataFrame:
    query = "SELECT station_id, ts_utc, parameter, source, value, unit, symbol, fetched_at FROM raw_readings"
    params: tuple = ()
    if station_id is not None:
        query += " WHERE station_id = ?"
        params = (station_id,)
    return pd.read_sql_query(query, conn, params=params)


def replace_clean_daily(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Replace the clean_daily table with the given frame."""
    conn.execute("DELETE FROM clean_daily")
    cols = ["station_id", "date", "parameter", "value", "n_samples", "source"]
    conn.executemany(
        "INSERT INTO clean_daily (station_id, date, parameter, value, n_samples, source)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        df[cols].itertuples(index=False, name=None),
    )
    conn.commit()
    return len(df)


def replace_daily_stats(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Replace the daily_stats table with the given frame."""
    conn.execute("DELETE FROM daily_stats")
    cols = ["station_id", "date", "parameter", "value", "roll_mean", "roll_std",
            "n_obs", "ci_lo", "ci_hi", "zscore", "is_anomaly"]
    frame = df[cols].astype(object).where(df[cols].notna(), None)
    conn.executemany(
        "INSERT INTO daily_stats (station_id, date, parameter, value, roll_mean, roll_std,"
        " n_obs, ci_lo, ci_hi, zscore, is_anomaly) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        frame.itertuples(index=False, name=None),
    )
    conn.commit()
    return len(df)


def read_daily_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT station_id, date, parameter, value, roll_mean, roll_std, n_obs,"
        " ci_lo, ci_hi, zscore, is_anomaly FROM daily_stats ORDER BY station_id, parameter, date",
        conn,
    )


def read_clean_daily(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT station_id, date, parameter, value, n_samples, source"
        " FROM clean_daily ORDER BY station_id, parameter, date",
        conn,
    )


def raw_counts(conn: sqlite3.Connection) -> pd.DataFrame:
    """Row counts per station and source (used for CLI logging)."""
    return pd.read_sql_query(
        "SELECT station_id, source, COUNT(*) AS rows FROM raw_readings"
        " GROUP BY station_id, source ORDER BY station_id, source",
        conn,
    )
