"""Shared synthetic fixtures. No test in this suite touches the network."""

from __future__ import annotations

import pandas as pd
import pytest

from hydro.config import AnalysisSettings, ApiSettings, Config, PipelineSettings, Station


@pytest.fixture()
def tiny_config() -> Config:
    return Config(
        stations=(
            Station(id="08XX001", name="Test River at Testville", river="Test River",
                    lat=49.0, lon=-122.0),
            Station(id="08XX002", name="Mock Creek near Mockton", river="Mock Creek",
                    lat=50.0, lon=-123.0),
        ),
        api=ApiSettings(),
        pipeline=PipelineSettings(min_daily_samples=2),
        analysis=AnalysisSettings(),
    )


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """Build a raw_readings-shaped frame with sensible defaults per row."""
    defaults = {
        "station_id": "08XX001",
        "parameter": "level",
        "source": "realtime",
        "unit": "m",
        "symbol": None,
        "fetched_at": "2026-07-01T00:00:00Z",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


@pytest.fixture()
def realtime_day() -> pd.DataFrame:
    """One full local day (2026-06-15, PDT = UTC-7) of 5-minute level data."""
    ts = pd.date_range("2026-06-15 07:00", periods=288, freq="5min", tz="UTC")
    return make_raw([{"ts_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": 2.0} for t in ts])
