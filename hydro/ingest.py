"""Ingestion: pull a time window of readings per station into raw SQLite.

Two collections are queried for every station:

* ``hydrometric-realtime`` for provisional high-frequency data. The API only
  retains ~30 days, so a 90-day request simply returns what exists.
* ``hydrometric-daily-mean`` for validated daily means. These are published
  with a long lag, so for a recent window this usually returns nothing — but
  when the window does overlap published data, validated values flow in and
  (during cleaning) take precedence over provisional ones.

Because upserts are idempotent, running ``fetch`` on a schedule accumulates a
history in SQLite that outlives the API's realtime retention window.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from hydro.api import EcccClient
from hydro.config import Config, Station
from hydro.db import RawReading

log = logging.getLogger(__name__)

# Canonical units per WSC/ECCC hydrometric documentation: water level is
# metres, discharge is cubic metres per second.
PARAM_FIELDS = {
    "level": ("LEVEL", "LEVEL_SYMBOL_EN", "m"),
    "discharge": ("DISCHARGE", "DISCHARGE_SYMBOL_EN", "m3/s"),
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _features_to_rows(
    features: list[dict],
    station: Station,
    source: str,
    ts_key: str,
    parameters: tuple[str, ...],
    fetched_at: str,
) -> list[RawReading]:
    """Flatten GeoJSON features into one RawReading per parameter."""
    rows: list[RawReading] = []
    for feat in features:
        props = feat.get("properties", {})
        ts = props.get(ts_key)
        if not ts:
            continue
        if source == "daily-mean":
            # DATE is a bare calendar date; store as midnight UTC so every
            # ts_utc in raw_readings is a full ISO-8601 instant.
            ts = f"{ts}T00:00:00Z"
        for param in parameters:
            value_key, symbol_key, unit = PARAM_FIELDS[param]
            value = props.get(value_key)
            if value is None:
                continue
            rows.append(
                RawReading(
                    station_id=station.id,
                    ts_utc=ts,
                    parameter=param,
                    source=source,
                    value=float(value),
                    unit=unit,
                    symbol=props.get(symbol_key) or None,
                    fetched_at=fetched_at,
                )
            )
    return rows


def fetch_station(client: EcccClient, cfg: Config, station: Station) -> list[RawReading]:
    """Fetch the configured window for one station from both collections."""
    end = _now_utc()
    start = end - timedelta(days=cfg.pipeline.window_days)
    interval = (
        f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    fetched_at = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = cfg.pipeline.parameters

    realtime = list(
        client.iter_items(
            "hydrometric-realtime",
            {"STATION_NUMBER": station.id, "datetime": interval},
        )
    )
    rows = _features_to_rows(realtime, station, "realtime", "DATETIME", params, fetched_at)

    daily_interval = f"{start.date()}/{end.date()}"
    daily = list(
        client.iter_items(
            "hydrometric-daily-mean",
            {"STATION_NUMBER": station.id, "datetime": daily_interval},
        )
    )
    rows += _features_to_rows(daily, station, "daily-mean", "DATE", params, fetched_at)

    log.info(
        "%s (%s): %d realtime features, %d daily-mean features -> %d raw rows",
        station.id, station.name, len(realtime), len(daily), len(rows),
    )
    return rows
