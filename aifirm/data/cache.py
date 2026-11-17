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

DEFAULT_DIR = Path(os.environ.get("AIFIRM_CACHE_DIR", Path.cwd() / ".cache" / "aifirm"))


class DiskCache:
    def __init__(self, root: Path | str | None = None, enabled: bool = True):
        self.root = Path(root) if root else DEFAULT_DIR
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
    ) -> Any:
        """Return the cached value for (namespace, key) or call ``producer`` and store it.

        ``ttl_seconds=None`` means cache forever (appropriate for immutable history).
        """
        if not self.enabled:
            return producer()
        ext = "json" if kind == "json" else "pkl"
        path = self._path(namespace, key, ext)
        if path.exists():
            fresh = ttl_seconds is None or (time.time() - path.stat().st_mtime) < ttl_seconds
            if fresh:
                try:
                    if kind == "json":
                        return json.loads(path.read_text())
                    with path.open("rb") as fh:
                        return pickle.load(fh)
                except Exception:  # corrupted cache entry -> refetch
                    path.unlink(missing_ok=True)
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
