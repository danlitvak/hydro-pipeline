"""Cleaning: raw API readings -> one tidy value per station, day, parameter.

Every rule is a pure function on pandas objects (no I/O, no globals), so each
one is unit-testable with synthetic frames:

1. :func:`coerce_numeric`     — force values to floats; garbage becomes NaN.
2. :func:`normalize_units`    — convert to canonical units (m, m3/s);
   unknown units fail loudly rather than silently corrupting data.
3. :func:`align_timestamps`   — parse ISO-8601 into tz-aware UTC, sort, and
   drop duplicate readings (keeping the most recently fetched).
4. :func:`aggregate_daily`    — collapse 5/15-minute realtime samples into a
   daily mean per *local* calendar day (America/Vancouver by default), with
   a minimum-sample threshold so a nearly-empty day cannot masquerade as a
   trustworthy daily mean.
5. :func:`merge_sources`      — validated daily means beat provisional
   realtime aggregates for the same day.
6. :func:`fill_small_gaps`    — linearly interpolate interior gaps up to
   ``max_gap_days`` long; longer outages stay NaN (interpolating across a
   week of missing data would invent hydrology that never happened).

:func:`clean` composes 1-6 into the full pipeline stage.
"""

from __future__ import annotations

import pandas as pd

#: canonical unit per parameter
CANONICAL_UNITS = {"level": "m", "discharge": "m3/s"}

#: multiplicative conversion factors to the canonical unit. The ECCC API
#: reports canonical units already, but the pipeline refuses to assume it:
#: any reading tagged with a convertible unit is converted, and an
#: unrecognised unit raises instead of passing through unscaled.
UNIT_FACTORS: dict[tuple[str, str], float] = {
    ("m", "m"): 1.0,
    ("cm", "m"): 0.01,
    ("mm", "m"): 0.001,
    ("ft", "m"): 0.3048,
    ("m3/s", "m3/s"): 1.0,
    ("l/s", "m3/s"): 0.001,
    ("cfs", "m3/s"): 0.028316846592,
}

CLEAN_COLUMNS = ["station_id", "date", "parameter", "value", "n_samples", "source"]

_CLEAN_DTYPES = {
    "station_id": "str", "date": "str", "parameter": "str",
    "value": "float64", "n_samples": "int64", "source": "str",
}


def empty_clean_frame() -> pd.DataFrame:
    """An empty clean_daily frame with correct dtypes (a bare
    ``DataFrame(columns=...)`` is all-object and would poison later concats)."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _CLEAN_DTYPES.items()})


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``value`` coerced to float64; unparseable -> NaN."""
    out = df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every reading to the canonical unit for its parameter.

    Raises ``ValueError`` for a unit with no known conversion — a silent
    pass-through of, say, cubic feet per second labelled as m3/s would be a
    factor-of-35 error in every downstream statistic.
    """
    out = df.copy()
    units = out["unit"].astype(str).str.lower().str.strip()
    params = out["parameter"].astype(str)

    factors = pd.Series(1.0, index=out.index)
    for (src_unit, canon), factor in UNIT_FACTORS.items():
        mask = (units == src_unit) & (params.map(CANONICAL_UNITS) == canon)
        factors[mask] = factor

    known = pd.Series(False, index=out.index)
    for (src_unit, canon) in UNIT_FACTORS:
        known |= (units == src_unit) & (params.map(CANONICAL_UNITS) == canon)
    if (~known).any():
        bad = sorted(set(zip(params[~known], units[~known])))
        raise ValueError(f"no unit conversion known for (parameter, unit) pairs: {bad}")

    out["value"] = out["value"] * factors
    out["unit"] = params.map(CANONICAL_UNITS)
    return out


def align_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``ts_utc`` into a tz-aware UTC ``ts`` column, sort, de-duplicate.

    If the same (station, timestamp, parameter, source) appears more than
    once — e.g. overlapping fetches where the provider revised a value — the
    row with the latest ``fetched_at`` wins.
    """
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts_utc"], utc=True, format="ISO8601")
    sort_cols = ["station_id", "parameter", "source", "ts"]
    if "fetched_at" in out.columns:
        out = out.sort_values(sort_cols + ["fetched_at"])
    else:
        out = out.sort_values(sort_cols)
    out = out.drop_duplicates(subset=["station_id", "ts", "parameter", "source"], keep="last")
    return out.reset_index(drop=True)


def aggregate_daily(
    df: pd.DataFrame,
    tz: str = "America/Vancouver",
    min_samples: int = 24,
) -> pd.DataFrame:
    """Collapse high-frequency realtime readings into local-day means.

    Readings are bucketed by calendar day *in the station's local timezone*
    (a 5-minute sample at 06:00 UTC belongs to the previous local day in BC),
    then averaged. Days with fewer than ``min_samples`` readings are dropped:
    a daily "mean" built from a couple of samples of a river with a strong
    diurnal cycle (snowmelt afternoons, dam release schedules) is biased, and
    it is better to record the day as missing than as falsely precise.
    """
    if df.empty:
        return empty_clean_frame()
    out = df.copy()
    out["date"] = out["ts"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
    grouped = (
        out.dropna(subset=["value"])
        .groupby(["station_id", "date", "parameter"], as_index=False)
        .agg(value=("value", "mean"), n_samples=("value", "size"))
    )
    grouped = grouped[grouped["n_samples"] >= min_samples].copy()
    grouped["source"] = "realtime"
    return grouped[CLEAN_COLUMNS].reset_index(drop=True)


def daily_mean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape validated daily-mean readings into the clean_daily schema.

    Published daily means already represent one calendar day; their date is
    taken verbatim (they are stored at midnight UTC purely so raw timestamps
    are uniform), *not* shifted into local time.
    """
    if df.empty:
        return empty_clean_frame()
    out = df.dropna(subset=["value"]).copy()
    out["date"] = out["ts"].dt.strftime("%Y-%m-%d")
    out["n_samples"] = 1
    out["source"] = "daily-mean"
    return out[CLEAN_COLUMNS].reset_index(drop=True)


def merge_sources(validated: pd.DataFrame, provisional: pd.DataFrame) -> pd.DataFrame:
    """Combine sources; validated daily means win over realtime aggregates."""
    frames = [f for f in (validated, provisional) if not f.empty]
    if not frames:
        return empty_clean_frame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["station_id", "date", "parameter"], keep="first")
    return merged.sort_values(["station_id", "parameter", "date"]).reset_index(drop=True)


def fill_small_gaps(series: pd.Series, max_gap_days: int = 2) -> tuple[pd.Series, pd.Series]:
    """Interpolate interior NaN runs of length <= ``max_gap_days``.

    Takes a daily series with a contiguous ``DatetimeIndex`` and returns
    ``(filled_series, filled_mask)``. Only *interior* gaps are filled (no
    extrapolation past the first/last observation), and only short ones:
    linear interpolation is a defensible estimate across a one- or two-day
    telemetry dropout, but across a longer outage it would fabricate a
    hydrograph, so longer runs are left missing for the statistics layer to
    handle honestly (they simply reduce the rolling-window sample count).
    """
    isna = series.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("size")
    fillable = isna & (run_len <= max_gap_days)

    interpolated = series.interpolate(method="linear", limit_area="inside")
    filled = series.where(~fillable, interpolated)
    filled_mask = fillable & filled.notna()
    return filled, filled_mask


def clean(
    raw: pd.DataFrame,
    tz: str = "America/Vancouver",
    min_daily_samples: int = 24,
    max_gap_days: int = 2,
) -> pd.DataFrame:
    """Full cleaning stage: raw readings frame -> clean_daily frame."""
    if raw.empty:
        return empty_clean_frame()

    df = coerce_numeric(raw)
    df = normalize_units(df)
    df = align_timestamps(df)

    provisional = aggregate_daily(
        df[df["source"] == "realtime"], tz=tz, min_samples=min_daily_samples
    )
    validated = daily_mean_rows(df[df["source"] == "daily-mean"])
    merged = merge_sources(validated, provisional)

    # Re-index each station/parameter series onto its full daily range and
    # patch short interior gaps.
    pieces: list[pd.DataFrame] = []
    for (station_id, parameter), grp in merged.groupby(["station_id", "parameter"]):
        idx = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        s = grp.set_index(pd.to_datetime(grp["date"]))["value"].reindex(idx).astype("float64")
        n = grp.set_index(pd.to_datetime(grp["date"]))["n_samples"].reindex(idx).fillna(0).astype(int)
        src = grp.set_index(pd.to_datetime(grp["date"]))["source"].reindex(idx)

        filled, mask = fill_small_gaps(s, max_gap_days=max_gap_days)
        src = src.where(~mask, "filled")

        piece = pd.DataFrame(
            {
                "station_id": station_id,
                "date": idx.strftime("%Y-%m-%d"),
                "parameter": parameter,
                "value": filled.to_numpy(),
                "n_samples": n.to_numpy(),
                "source": src.fillna("missing").to_numpy(),
            }
        )
        # Long gaps remain NaN; keep those rows out of clean_daily (the
        # statistics layer re-indexes again, so absence == missing).
        piece = piece[piece["value"].notna()]
        pieces.append(piece)

    if not pieces:
        return empty_clean_frame()
    return pd.concat(pieces, ignore_index=True)[CLEAN_COLUMNS]
