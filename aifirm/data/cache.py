"""Tiny disk cache for HTTP JSON/CSV payloads and pandas objects.

Everything the data desk downloads is cached so that (a) repeated runs are cheap and
fast, (b) backtests are reproducible, and (c) we respect free-tier rate limits
(SEC fair-access policy: <= 10 requests/second, Yahoo throttling).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable


def default_dir() -> Path:
    """Where the cache lives unless a directory is given: ``$AIFIRM_CACHE_DIR``, else
    ``$XDG_CACHE_HOME/aifirm``, else ``~/.cache/aifirm``. Never the working directory: a
    scheduled run started from another directory must reuse the same downloads and the
    same cached LLM answers (a cold cache refetches data and pays for every call again).
    Read when a cache is created, so setting the variable later in a process works."""
    env = os.environ.get("AIFIRM_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "aifirm"


_MISSING = object()


class DiskCache:
    def __init__(self, root: Path | str | None = None, enabled: bool = True):
        self.root = Path(root) if root else default_dir()
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str, ext: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}.{ext}"

    def get_or_set(
        self,
        namespace: str,
        key: str,
        producer: Callable[[], Any],
        ttl_seconds: float | None = None,
        kind: str = "pickle",
        ttl_for: Callable[[Any], float | None] | None = None,
    ) -> Any:
        """Return the cached value for (namespace, key) or call ``producer`` and store it.

        ``ttl_seconds=None`` means cache forever (appropriate for immutable history).
        ``ttl_for``, if given, is called with the cached (or freshly produced) value and
        returns the TTL that applies to *that* value, overriding ``ttl_seconds``. Use it when
        a result may be incomplete (e.g. a throttled download): a complete result can be
        kept forever while a partial one expires quickly and is fetched again.
        """
        if not self.enabled:
            return producer()
        ext = "json" if kind == "json" else "pkl"
        path = self._path(namespace, key, ext)
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if ttl_for is not None or ttl_seconds is None or age < ttl_seconds:
                cached: Any = _MISSING
                try:
                    if kind == "json":
                        cached = json.loads(path.read_text())
                    else:
                        with path.open("rb") as fh:
                            cached = pickle.load(fh)
                except Exception:  # corrupted cache entry -> refetch
                    path.unlink(missing_ok=True)
                if cached is not _MISSING:
                    ttl = ttl_for(cached) if ttl_for is not None else ttl_seconds
                    if ttl is None or age < ttl:
                        return cached
        value = producer()
        tmp = path.with_suffix(path.suffix + ".tmp")
        if kind == "json":
            tmp.write_text(json.dumps(value))
        else:
            with tmp.open("wb") as fh:
                pickle.dump(value, fh)
        tmp.replace(path)
        return value


_default_cache: DiskCache | None = None


def default_cache() -> DiskCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = DiskCache()
    return _default_cache


def set_default_cache(cache: DiskCache) -> None:
    global _default_cache
    _default_cache = cache
