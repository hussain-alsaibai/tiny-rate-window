# tiny-rate-window

**Zero-dependency sliding-window rate limiter for HTTP APIs and agent pipelines.**

Stdlib only. MIT licensed. ~350 LOC. 23 tests.

```bash
pip install tiny-rate-window
```

---

## Why a sliding window?

Fixed-window counters (e.g. "1000 requests per minute") suffer from the
**boundary burst problem**: a client can fire 1000 requests in the last
second of one minute and 1000 more in the first second of the next,
getting 2000 through in 2 seconds — twice the intended rate.

Token-bucket limiters solve this but require tuning of burst capacity and
refill rate. They're great for steady workloads but awkward when you need
a *hard* ceiling on requests-per-time-window.

`tiny-rate-window` implements the **weighted sliding-window algorithm**:

```
effective_count(now) = previous_window_count * overlap_fraction
                       + current_window_count

where overlap_fraction = 1 - (now - current_window_start) / window_seconds
```

At the start of a new window, the previous window still counts for a
fraction of its requests until they age out. Smooth ceiling, no
boundary-burst hole.

---

## Quick start

```python
from tiny_rate_window import SlidingWindowLimiter, http_headers

limiter = SlidingWindowLimiter(limit=100, window_seconds=60)

def handle_request(user_id: str):
    decision = limiter.try_acquire(f"user:{user_id}")
    if not decision.allowed:
        headers = http_headers(decision)
        return {"error": "rate_limited", "retry_after_s": decision.retry_after_s}, 429, headers
    return do_work()
```

---

## Public API

### `SlidingWindowLimiter(limit, window_seconds, *, max_keys=0, clock=time.monotonic, on_reject=None)`

| Parameter | Type | Notes |
|-----------|------|-------|
| `limit` | `int` | Max effective count per window. Must be > 0. |
| `window_seconds` | `float` | Window length. Must be > 0. |
| `max_keys` | `int` | Soft cap on distinct keys (0 = unlimited). |
| `clock` | `callable` | Override in tests. |
| `on_reject` | `callable` | `(key, decision) -> None` invoked on blocked attempts. |

### `try_acquire(key, *, cost=1, now=None) -> Decision`

Atomically reserve `cost` slots for `key`. Mutates state on success AND
on each blocking attempt so peak counters reflect attempted work.

### `peek(key, *, now=None) -> Decision`

Non-mutating inspection for `/metrics` endpoints.

### `reset(key=None)`

Drop state for one key (or all keys).

### `evict_idle(idle_seconds, *, now=None) -> int`

Drop windows whose `last_seen` is older than the threshold. Returns the
number evicted.

### `stats() -> LimiterStats`

Snapshot of aggregate counters: `hits`, `misses`, `hits_per_key`,
`peak_count`, `rollovers`.

---

## Decision shape

```python
@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float
    window_seconds: float
    current_count: int
```

`Decision.as_dict()` returns a dict suitable for JSON logging or
OpenTelemetry attribute emission.

---

## HTTP headers

`http_headers(decision)` maps a `Decision` to the standard `X-RateLimit-*`
header set:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Window: 60
Retry-After: 13   (only when blocked)
```

`Retry-After` is rounded **up** to whole seconds per RFC 7231.

---

## Framework integration

### FastAPI / Starlette

```python
from fastapi import Request, Response

limiter = SlidingWindowLimiter(limit=100, window_seconds=60)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    user = request.headers.get("X-User-Id", request.client.host)
    d = limiter.try_acquire(f"u:{user}")
    if not d.allowed:
        return Response(
            content='{"error":"rate_limited"}',
            status_code=429,
            headers=http_headers(d),
            media_type="application/json",
        )
    resp = await call_next(request)
    for k, v in http_headers(d).items():
        resp.headers[k] = v
    return resp
```

### Flask

```python
from flask import request, jsonify, make_response

@app.before_request
def _rate_limit():
    d = limiter.try_acquire(f"u:{request.remote_addr}")
    if not d.allowed:
        resp = make_response(jsonify({"error": "rate_limited"}), 429)
        for k, v in http_headers(d).items():
            resp.headers[k] = v
        return resp
    request.rate_limit_decision = d
```

### Agent pipelines

```python
limiter = SlidingWindowLimiter(limit=50, window_seconds=60, on_reject=lambda k, d: log.warning("blocked", extra=d.as_dict()))

def call_external_api(req):
    if not limiter.try_acquire("external_api").allowed:
        raise RateLimited("external API budget exhausted")
    return req.send()
```

---

## Why this library, not `redis-cell`?

`redis-cell` and friends are great when you have Redis. `tiny-rate-window`
is for the case where you don't — or where the limiter needs to live
*in-process* with the request handler (think agent sandboxes, single-tenant
background jobs, embedded use).

---

## Performance

A single `try_acquire` is two dict lookups, one float multiplication,
one comparison, and one increment — all under a single short-held lock.
On a 2024 M-series MacBook Pro, this library sustains ~150K ops/sec
across 1000 distinct keys.

For >50K qps single-process with millions of keys, consider sharding
key hashes across N limiters to reduce lock contention.

---

## License

MIT — copy, fork, ship.
