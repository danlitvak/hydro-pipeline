"""Configuration loading.

``stations.toml`` is the single source of truth for the station list, API
settings and analysis parameters. Stations are recorded there only after being
verified against the live API (see the comment header in the file).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = "stations.toml"


@dataclass(frozen=True)
class Station:
    id: str
    name: str
    river: str
    lat: float
    lon: float


@dataclass(frozen=True)
class ApiSettings:
    base_url: str = "https://api.weather.gc.ca"
    user_agent: str = "hydro-pipeline/0.1"
    page_limit: int = 10000
    request_delay_s: float = 0.5
    timeout_s: float = 60.0
    max_retries: int = 3


@dataclass(frozen=True)
class PipelineSettings:
    window_days: int = 90
    parameters: tuple[str, ...] = ("level", "discharge")
    timezone: str = "America/Vancouver"
    min_daily_samples: int = 24
    max_gap_days: int = 2


@dataclass(frozen=True)
class AnalysisSettings:
    rolling_window_days: int = 7
    min_periods: int = 4
    z_threshold: float = 2.0
    ci_level: float = 0.95


@dataclass(frozen=True)
class Config:
    stations: tuple[Station, ...]
    api: ApiSettings = field(default_factory=ApiSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)

    def station(self, station_id: str) -> Station:
        for s in self.stations:
            if s.id == station_id:
                return s
        raise KeyError(f"station {station_id!r} not in config")


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Parse a TOML config file into a :class:`Config`."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    stations = tuple(
        Station(
            id=s["id"],
            name=s["name"],
            river=s.get("river", s["name"]),
            lat=float(s["lat"]),
            lon=float(s["lon"]),
        )
        for s in raw.get("stations", [])
    )
    if not stations:
        raise ValueError(f"no [[stations]] entries found in {path}")

    api = ApiSettings(**raw.get("api", {}))
    pipe_raw = dict(raw.get("pipeline", {}))
    if "parameters" in pipe_raw:
        pipe_raw["parameters"] = tuple(pipe_raw["parameters"])
    pipeline = PipelineSettings(**pipe_raw)
    analysis = AnalysisSettings(**raw.get("analysis", {}))

    return Config(stations=stations, api=api, pipeline=pipeline, analysis=analysis)
