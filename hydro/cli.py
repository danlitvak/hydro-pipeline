"""Command-line interface: ``python -m hydro fetch|analyze|report``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from hydro import __version__
from hydro.config import DEFAULT_CONFIG, load_config

log = logging.getLogger("hydro")

DEFAULT_DB = "data/hydro.db"
DEFAULT_REPORT = "report.html"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to stations.toml")
    p.add_argument("--db", default=DEFAULT_DB, help="path to the SQLite database")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hydro",
        description="BC hydrometric data pipeline (ECCC GeoMet API -> SQLite -> stats -> HTML report)",
    )
    parser.add_argument("--version", action="version", version=f"hydro {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="ingest the configured window into raw SQLite")
    _add_common(p_fetch)
    p_fetch.add_argument(
        "--window-days", type=int, default=None,
        help="override [pipeline].window_days from the config",
    )

    p_analyze = sub.add_parser("analyze", help="clean raw data and compute rolling statistics")
    _add_common(p_analyze)

    p_report = sub.add_parser("report", help="render the HTML report from computed statistics")
    _add_common(p_report)
    p_report.add_argument("--out", default=DEFAULT_REPORT, help="output HTML path")

    return parser


def cmd_fetch(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from hydro import db as dbm
    from hydro.api import EcccClient
    from hydro.ingest import fetch_station

    cfg = load_config(args.config)
    if args.window_days:
        cfg = replace(cfg, pipeline=replace(cfg.pipeline, window_days=args.window_days))

    client = EcccClient(cfg.api)
    conn = dbm.connect(args.db)
    total = 0
    try:
        for station in cfg.stations:
            rows = fetch_station(client, cfg, station)
            written = dbm.upsert_raw(conn, rows)
            total += written
            time.sleep(cfg.api.request_delay_s)
        counts = dbm.raw_counts(conn)
        log.info("fetch complete: %d rows upserted this run", total)
        for _, r in counts.iterrows():
            log.info("  %s %-10s %6d rows in db", r["station_id"], r["source"], r["rows"])
    finally:
        conn.close()
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from hydro import db as dbm
    from hydro.analyze import analyze
    from hydro.clean import clean

    cfg = load_config(args.config)
    conn = dbm.connect(args.db)
    try:
        raw = dbm.read_raw(conn)
        if raw.empty:
            log.error("no raw data in %s - run `python -m hydro fetch` first", args.db)
            return 1
        cleaned = clean(
            raw,
            tz=cfg.pipeline.timezone,
            min_daily_samples=cfg.pipeline.min_daily_samples,
            max_gap_days=cfg.pipeline.max_gap_days,
        )
        n_clean = dbm.replace_clean_daily(conn, cleaned)
        stats = analyze(
            cleaned,
            window=cfg.analysis.rolling_window_days,
            min_periods=cfg.analysis.min_periods,
            z_threshold=cfg.analysis.z_threshold,
            ci_level=cfg.analysis.ci_level,
        )
        n_stats = dbm.replace_daily_stats(conn, stats)
        n_anom = int(stats["is_anomaly"].sum()) if not stats.empty else 0
        log.info(
            "analyze complete: %d raw rows -> %d clean station-days -> %d stat rows, %d anomalies flagged",
            len(raw), n_clean, n_stats, n_anom,
        )
    finally:
        conn.close()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from hydro import db as dbm
    from hydro.report import build_report

    cfg = load_config(args.config)
    conn = dbm.connect(args.db)
    try:
        stats = dbm.read_daily_stats(conn)
    finally:
        conn.close()
    if stats.empty:
        log.error("no statistics in %s - run `python -m hydro analyze` first", args.db)
        return 1
    out = build_report(cfg, stats, out_path=args.out)
    size_kb = Path(out).stat().st_size / 1024
    log.info("report written: %s (%.0f KiB)", out, size_kb)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    handlers = {"fetch": cmd_fetch, "analyze": cmd_analyze, "report": cmd_report}
    return handlers[args.command](args)
