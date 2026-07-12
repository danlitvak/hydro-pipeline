"""Config parsing tests, including the repo's real stations.toml."""

from __future__ import annotations

from pathlib import Path

import pytest

from hydro.config import load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "stations.toml"


def test_repo_config_parses():
    cfg = load_config(REPO_CONFIG)
    assert len(cfg.stations) == 8
    ids = [s.id for s in cfg.stations]
    assert "08MF005" in ids  # Fraser River at Hope
    assert "08GA010" in ids  # Capilano
    assert "08HD035" in ids  # Campbell
    assert cfg.pipeline.parameters == ("level", "discharge")
    assert cfg.analysis.z_threshold == 2.0
    assert cfg.api.base_url.startswith("https://api.weather.gc.ca")


def test_station_lookup():
    cfg = load_config(REPO_CONFIG)
    fraser = cfg.station("08MF005")
    assert "Fraser" in fraser.name
    assert 48 < fraser.lat < 51
    assert -126 < fraser.lon < -119
    with pytest.raises(KeyError):
        cfg.station("00XX000")


def test_config_without_stations_rejected(tmp_path):
    p = tmp_path / "empty.toml"
    p.write_text("[api]\nbase_url = 'https://example.test'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stations"):
        load_config(p)
