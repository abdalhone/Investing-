"""HTTP helpers with polite rate limiting for free public data sources."""

from __future__ import annotations

import os
import threading
import time

import requests

# SEC asks automated clients to identify themselves ("Sample Company Name AdminContact@<domain>")
# and to stay under 10 requests/second. Set SEC_USER_AGENT to your own name + email.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "aifirm-research admin@example.com")

_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _throttle(host: str, min_interval: float) -> None:
    with _lock:
        now = time.monotonic()
        wait = _last_call.get(host, 0.0) + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def get(url: str, *, host_key: str, min_interval: float = 0.15, headers: dict | None = None,
        timeout: float = 30.0, retries: int = 3) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        _throttle(host_key, min_interval)
        try:
            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} for {url}", response=resp)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # network or retryable HTTP error
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise
            time.sleep(1.5 * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def sec_get_json(url: str) -> dict:
    return get(url, host_key="sec", min_interval=0.15, headers={"User-Agent": SEC_USER_AGENT}).json()
