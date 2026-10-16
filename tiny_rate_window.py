"""tiny-rate-window v0.1.0 — sliding-window rate limiter for HTTP APIs & agents.

Why a sliding window?
---------------------
Fixed-window counters (e.g. "1000 requests per minute") suffer from the
**boundary burst problem**: a client can fire 1000 requests in the last second
of one window and 1000 more in the first second of the next, getting 2000
through in 2 seconds — twice the intended rate.

Token-bucket limiters solve this but require tuning of burst capacity and
refill rate. They're a great fit for steady workloads but awkward when you
need a *hard* ceiling on requests-per-time-window.

This library implements the **weighted sliding window** algorithm:

    effective_count(now) = previous_window_count * overlap_fraction + current_window_count

where `overlap_fraction = 1 - (now - current_window_start) / window_seconds`.
At the start of a new window, the previous window still "counts for" a
fraction of its requests until they age out. This gives a smooth ceiling
without the boundary-burst hole.

The implementation is **zero-dependency** (stdlib only), thread-safe via a
single `threading.Lock` per limiter, and exposes a small but expressive API
designed for both HTTP middleware and in-process agent pipelines.

Public API
----------
- :class:`SlidingWindowLimiter` — the main limiter.
  - ``try_acquire(key, *, cost=1, now=None)`` — atomically reserve `cost`
    slots; returns :class:`Decision` with `allowed`, `remaining`, `retry_after_s`,
    `limit`, `window_seconds`.
  - ``peek(key, *, now=None)`` — non-mutating inspection for /metrics.
  - ``reset(key=...)`` — drop one key's history (or all).
  - ``stats()`` — counters (`hits`, `misses`, `hits_per_key`, `peak_count`).
- :class:`Decision` — :func:`asdict` for HTTP header emission (X-RateLimit-*).
- :class:`LimiterStats` — observability surface.
- :func:`http_headers` — turn a Decision into a dict of standard headers.

Example
-------
::

    limiter = SlidingWindowLimiter(limit=100, window_seconds=60)
    d = limiter.try_acquire("user:42")
    if not d.allowed:
        return {"error": "rate_limited", "retry_after_s": d.retry_after_s}, 429
    handle_request()

Notes
-----
- Thread-safe. One lock per limiter instance. For high-throughput use (>50K
  qps single-process) consider sharding by key hash.
- Per-key partitioning is automatic. The first call to ``try_acquire("k")``
  lazily creates a window for that key.
- Memory grows linearly with the number of *distinct* keys observed. For
  untrusted input sources, set ``max_keys`` and call :meth:`evict_idle`
  periodically.
- Clock is injectable via ``now=`` for testing. Defaults to ``time.monotonic()``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple


__version__ = "0.1.0"


@dataclass(frozen=True)
class Decision:
    """Result of a try_acquire() call. Designed to map cleanly to HTTP headers.

    >>> d = Decision(allowed=True, limit=10, remaining=7, retry_after_s=0.0,
    ...              window_seconds=60.0, current_count=3)
    >>> d.allowed
    True
    """

    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float
    window_seconds: float
    current_count: int

    def as_dict(self) -> Dict[str, object]:
        """Return a dict suitable for JSON logging / OpenTelemetry attrs."""
        return asdict(self)


@dataclass
class _Window:
    """Per-key ring of the last two window snapshots."""

    prev_count: int = 0
    prev_start: float = 0.0
    curr_count: int = 0
    curr_start: float = 0.0
    last_seen: float = 0.0

    def effective(self, now: float, window_s: float) -> Tuple[int, float]:
        """Return (effective_count, retry_after_s_if_over)."""
        if self.curr_start == 0.0:
            self.curr_start = now
            self.last_seen = now
        elapsed = now - self.curr_start
        if elapsed >= window_s:
            # Roll the window.
            self.prev_count = self.curr_count
            self.prev_start = self.curr_start
            self.curr_count = 0
            self.curr_start = now
            elapsed = 0.0
        if self.prev_count == 0 or self.prev_start == 0.0:
            return self.curr_count, 0.0
        overlap = max(0.0, 1.0 - elapsed / window_s)
        return int(self.curr_count + self.prev_count * overlap), 0.0


@dataclass
class LimiterStats:
    """Aggregate counters for observability. Reset via :meth:`SlidingWindowLimiter.reset_stats`."""

    hits: int = 0
    misses: int = 0
    hits_per_key: Dict[str, int] = field(default_factory=dict)
    peak_count: int = 0
    rollovers: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hits_per_key": dict(self.hits_per_key),
            "peak_count": self.peak_count,
            "rollovers": self.rollovers,
        }


class SlidingWindowLimiter:
    """Per-key weighted sliding-window rate limiter.

    Parameters
    ----------
    limit : int
        Maximum effective count per window. Must be > 0.
    window_seconds : float
        Length of one window in seconds. Must be > 0.
    max_keys : int, optional
        Soft cap on the number of distinct keys retained. ``try_acquire``
        raises ``RuntimeError`` when the cap is hit *and* the new key has
        no prior state. Set to 0 to disable.
    clock : callable, optional
        Monotonic-clock callable. Override in tests.
    on_reject : callable, optional
        ``on_reject(key, decision)`` called for every blocked attempt.
        Useful for emitting metrics or alerts.
    """

    __slots__ = (
        "_limit",
        "_window_s",
        "_max_keys",
        "_clock",
        "_on_reject",
        "_lock",
        "_windows",
        "_stats",
    )

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: float,
        max_keys: int = 0,
        clock: callable = time.monotonic,
        on_reject: Optional[callable] = None,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._limit = int(limit)
        self._window_s = float(window_seconds)
        self._max_keys = int(max_keys)
        self._clock = clock
        self._on_reject = on_reject
        self._lock = threading.Lock()
        self._windows: Dict[str, _Window] = {}
        self._stats = LimiterStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def window_seconds(self) -> float:
        return self._window_s

    def try_acquire(self, key: str, *, cost: int = 1, now: Optional[float] = None) -> Decision:
        """Atomically reserve `cost` slots for `key`.

        Returns a Decision with `allowed=True` when the request fits inside
        the limit, otherwise `allowed=False` and a non-zero `retry_after_s`.
        Mutates the limiter state on success AND on each blocking attempt
        (so peak_count reflects attempted work, not just granted work).
        """
        if cost <= 0:
            raise ValueError("cost must be > 0")
        ts = self._clock() if now is None else float(now)
        with self._lock:
            w = self._windows.get(key)
            if w is None:
                if self._max_keys and len(self._windows) >= self._max_keys:
                    raise RuntimeError(
                        f"max_keys ({self._max_keys}) exceeded; "
                        "evict idle keys or raise the cap"
                    )
                w = _Window()
                self._windows[key] = w
            effective, _ = w.effective(ts, self._window_s)
            if effective + cost > self._limit:
                # Compute retry_after — time until current window ends + maybe one more.
                elapsed = ts - w.curr_start
                retry = max(0.0, self._window_s - elapsed)
                # Bump misses / peak.
                self._stats.misses += 1
                self._stats.hits_per_key[key] = self._stats.hits_per_key.get(key, 0) + 1
                if effective > self._stats.peak_count:
                    self._stats.peak_count = effective
                d = Decision(
                    allowed=False,
                    limit=self._limit,
                    remaining=0,
                    retry_after_s=retry,
                    window_seconds=self._window_s,
                    current_count=effective,
                )
                if self._on_reject is not None:
                    try:
                        self._on_reject(key, d)
                    except Exception:
                        pass
                return d
            # Allowed — increment current count.
            w.curr_count += cost
            w.last_seen = ts
            self._stats.hits += 1
            self._stats.hits_per_key[key] = self._stats.hits_per_key.get(key, 0) + 1
            effective_after = effective + cost
            if effective_after > self._stats.peak_count:
                self._stats.peak_count = effective_after
            remaining = max(0, self._limit - effective_after)
            return Decision(
                allowed=True,
                limit=self._limit,
                remaining=remaining,
                retry_after_s=0.0,
                window_seconds=self._window_s,
                current_count=effective_after,
            )

    def peek(self, key: str, *, now: Optional[float] = None) -> Decision:
        """Non-mutating inspection. Returns Decision with the *current*
        effective count without reserving a slot."""
        ts = self._clock() if now is None else float(now)
        with self._lock:
            w = self._windows.get(key)
            if w is None:
                return Decision(
                    allowed=True,
                    limit=self._limit,
                    remaining=self._limit,
                    retry_after_s=0.0,
                    window_seconds=self._window_s,
                    current_count=0,
                )
            effective, _ = w.effective(ts, self._window_s)
            return Decision(
                allowed=effective < self._limit,
                limit=self._limit,
                remaining=max(0, self._limit - int(effective)),
                retry_after_s=0.0,
                window_seconds=self._window_s,
                current_count=int(effective),
            )

    def reset(self, key: Optional[str] = None) -> None:
        """Drop state for one key (or all keys). Useful for tests and admin paths."""
        with self._lock:
            if key is None:
                self._windows.clear()
            else:
                self._windows.pop(key, None)

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = LimiterStats()

    def stats(self) -> LimiterStats:
        with self._lock:
            return LimiterStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                hits_per_key=dict(self._stats.hits_per_key),
                peak_count=self._stats.peak_count,
                rollovers=self._stats.rollovers,
            )

    def evict_idle(self, idle_seconds: float, *, now: Optional[float] = None) -> int:
        """Drop windows whose ``last_seen`` is older than ``idle_seconds``.
        Returns the number of windows evicted."""
        ts = self._clock() if now is None else float(now)
        with self._lock:
            stale = [k for k, w in self._windows.items() if ts - w.last_seen > idle_seconds]
            for k in stale:
                self._windows.pop(k, None)
            return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._windows)


def http_headers(decision: Decision) -> Dict[str, str]:
    """Convert a Decision into the standard X-RateLimit-* header set.

    Returns headers as ``str`` so they drop straight into ``httpx``,
    ``aiohttp``, ``requests``, or any WSGI middleware.
    """
    out = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Window": f"{decision.window_seconds:g}",
    }
    if not decision.allowed:
        # Retry-After is in whole seconds per RFC 7231 (ceil up).
        out["Retry-After"] = str(max(1, int(round(decision.retry_after_s + 0.499))))
    return out


__all__ = [
    "__version__",
    "Decision",
    "LimiterStats",
    "SlidingWindowLimiter",
    "http_headers",
]
