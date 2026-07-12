"""Minimal client for the ECCC GeoMet OGC API (api.weather.gc.ca).

The hydrometric collections used here:

* ``hydrometric-realtime``   — provisional 5- or 15-minute readings, rolling
  retention of roughly the last 30 days. Timestamps are UTC (``DATETIME``).
* ``hydrometric-daily-mean`` — validated daily means (``DATE``), published
  with a long lag (typically a year or more behind today).
* ``hydrometric-stations``   — station metadata (used during exploration to
  verify station IDs; the verified results live in ``stations.toml``).

The client is deliberately polite: it identifies itself with a User-Agent,
sleeps between requests, retries transient failures with backoff, and pages
with the documented ``limit``/``offset`` parameters.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import requests

from hydro.config import ApiSettings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EcccClient:
    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers["User-Agent"] = settings.user_agent
        self.session.headers["Accept"] = "application/geo+json, application/json"

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with bounded retries and exponential backoff."""
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.settings.timeout_s)
                if resp.status_code in RETRYABLE_STATUS:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in RETRYABLE_STATUS:
                    raise  # 4xx other than 429: our request is wrong, do not hammer
                last_exc = exc
                if attempt < self.settings.max_retries:
                    log.warning("request failed (%s), retry %d/%d in %.0fs",
                                exc, attempt, self.settings.max_retries, delay)
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"GET {url} failed after {self.settings.max_retries} attempts") from last_exc

    def iter_items(self, collection: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield GeoJSON features from a collection, transparently paging."""
        url = f"{self.settings.base_url}/collections/{collection}/items"
        offset = 0
        while True:
            page_params = {
                **params,
                "f": "json",
                "limit": self.settings.page_limit,
                "offset": offset,
            }
            data = self._get(url, page_params)
            features = data.get("features", [])
            yield from features
            returned = data.get("numberReturned", len(features))
            if returned < self.settings.page_limit or not features:
                return
            offset += returned
            time.sleep(self.settings.request_delay_s)
