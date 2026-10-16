"""Tests for tiny_rate_window v0.1.0."""

from __future__ import annotations

import unittest

from tiny_rate_window import (
    Decision,
    LimiterStats,
    SlidingWindowLimiter,
    __version__,
    http_headers,
)


class _Clock:
    """Deterministic clock for testing the sliding-window behavior."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestVersion(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(__version__, "0.1.0")


class TestConstruction(unittest.TestCase):
    def test_rejects_zero_limit(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowLimiter(limit=0, window_seconds=60)

    def test_rejects_negative_limit(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowLimiter(limit=-1, window_seconds=60)

    def test_rejects_zero_window(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowLimiter(limit=10, window_seconds=0)


class TestBasicAcquire(unittest.TestCase):
    def test_first_acquire_allowed(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clk)
        d = rl.try_acquire("u1")
        self.assertTrue(d.allowed)
        self.assertEqual(d.remaining, 4)
        self.assertEqual(d.limit, 5)
        self.assertEqual(d.window_seconds, 60.0)
        self.assertEqual(d.current_count, 1)

    def test_drains_to_zero_remaining(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=3, window_seconds=60, clock=clk)
        for i in range(3):
            d = rl.try_acquire("u1")
            self.assertTrue(d.allowed, f"acquire {i+1} should be allowed")
            self.assertEqual(d.remaining, 3 - (i + 1))

    def test_blocks_when_full(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=2, window_seconds=60, clock=clk)
        rl.try_acquire("u1")
        rl.try_acquire("u1")
        d = rl.try_acquire("u1")
        self.assertFalse(d.allowed)
        self.assertEqual(d.remaining, 0)
        self.assertGreater(d.retry_after_s, 0.0)
        self.assertLessEqual(d.retry_after_s, 60.0)

    def test_per_key_isolation(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=2, window_seconds=60, clock=clk)
        rl.try_acquire("u1")
        rl.try_acquire("u1")
        d_other = rl.try_acquire("u2")
        self.assertTrue(d_other.allowed)
        self.assertEqual(d_other.current_count, 1)


class TestWeightedSliding(unittest.TestCase):
    def test_old_window_partial_contribution(self) -> None:
        """At t=30s into a new 60s window, prev_count contributes 0.5x."""
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=10, window_seconds=60, clock=clk)
        # Fill previous window.
        for _ in range(8):
            rl.try_acquire("u1")
        # Roll forward 60s. Now in fresh window with prev_count=8 at fraction 1.0
        clk.advance(60)
        # First acquire in new window: effective = 0*1.0 + 0 = 0, allowed.
        d1 = rl.try_acquire("u1")
        self.assertTrue(d1.allowed)
        # Advance 30s and add another: prev still contributes 0.5 * 8 = 4.
        clk.advance(30)
        d2 = rl.try_acquire("u1")
        # effective = 1 (current) + 8 * (1 - 30/60) = 1 + 4 = 5
        self.assertEqual(d2.current_count, 6)  # 1 (current) + 1 (just acquired) + 8 * 0.5 (prev) = 6

    def test_boundary_burst_blocked(self) -> None:
        """A fixed-window limiter would let 2x through at the boundary.
        Sliding window must NOT.
        """
        clk = _Clock(t=1000.0)
        rl = SlidingWindowLimiter(limit=10, window_seconds=60, clock=clk)
        # End of window 1: t=59, fire 10.
        clk.advance(59)
        for _ in range(10):
            d = rl.try_acquire("u1")
            self.assertTrue(d.allowed)
        # Just past the boundary: t=61, current_count=0 but prev=10.
        # effective = 10 * (1 - 1/60) ≈ 9.83 — already over half, so an
        # 11th request would push effective > 10.
        clk.advance(2)
        d = rl.try_acquire("u1")
        self.assertFalse(d.allowed)


class TestCost(unittest.TestCase):
    def test_cost_exceeds_limit_blocks(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clk)
        d = rl.try_acquire("u1", cost=10)
        self.assertFalse(d.allowed)

    def test_cost_partial(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=10, window_seconds=60, clock=clk)
        d = rl.try_acquire("u1", cost=3)
        self.assertTrue(d.allowed)
        self.assertEqual(d.current_count, 3)
        self.assertEqual(d.remaining, 7)

    def test_cost_must_be_positive(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=10, window_seconds=60, clock=clk)
        with self.assertRaises(ValueError):
            rl.try_acquire("u1", cost=0)


class TestPeek(unittest.TestCase):
    def test_peek_does_not_consume(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clk)
        rl.try_acquire("u1")
        rl.try_acquire("u1")
        d = rl.peek("u1")
        self.assertEqual(d.current_count, 2)
        d2 = rl.peek("u1")
        self.assertEqual(d2.current_count, 2)

    def test_peek_unknown_key_is_full(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clk)
        d = rl.peek("never-seen")
        self.assertTrue(d.allowed)
        self.assertEqual(d.current_count, 0)
        self.assertEqual(d.remaining, 5)


class TestStats(unittest.TestCase):
    def test_hits_and_misses(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=2, window_seconds=60, clock=clk)
        rl.try_acquire("u1")
        rl.try_acquire("u1")
        rl.try_acquire("u1")  # miss
        rl.try_acquire("u1")  # miss
        s = rl.stats()
        self.assertEqual(s.hits, 2)
        self.assertEqual(s.misses, 2)
        self.assertEqual(s.hits_per_key["u1"], 4)

    def test_on_reject_callback(self) -> None:
        clk = _Clock()
        seen = []
        rl = SlidingWindowLimiter(
            limit=1,
            window_seconds=60,
            clock=clk,
            on_reject=lambda k, d: seen.append((k, d.allowed)),
        )
        rl.try_acquire("u1")
        rl.try_acquire("u1")  # blocked
        self.assertEqual(seen, [("u1", False)])


class TestReset(unittest.TestCase):
    def test_reset_specific_key(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=2, window_seconds=60, clock=clk)
        rl.try_acquire("u1")
        rl.try_acquire("u1")
        rl.reset("u1")
        d = rl.try_acquire("u1")
        self.assertTrue(d.allowed)
        self.assertEqual(d.current_count, 1)

    def test_reset_all(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=2, window_seconds=60, clock=clk)
        rl.try_acquire("u1")
        rl.try_acquire("u2")
        rl.reset()
        self.assertEqual(len(rl), 0)


class TestEvict(unittest.TestCase):
    def test_evict_idle(self) -> None:
        clk = _Clock(t=1000.0)
        rl = SlidingWindowLimiter(limit=10, window_seconds=60, clock=clk)
        rl.try_acquire("u1")
        clk.advance(120)
        rl.try_acquire("u2")
        evicted = rl.evict_idle(idle_seconds=60)
        self.assertEqual(evicted, 1)
        self.assertEqual(len(rl), 1)


class TestMaxKeys(unittest.TestCase):
    def test_max_keys_enforced(self) -> None:
        clk = _Clock()
        rl = SlidingWindowLimiter(limit=10, window_seconds=60, max_keys=2, clock=clk)
        rl.try_acquire("u1")
        rl.try_acquire("u2")
        with self.assertRaises(RuntimeError):
            rl.try_acquire("u3")


class TestHTTPHeaders(unittest.TestCase):
    def test_allowed_headers(self) -> None:
        d = Decision(allowed=True, limit=100, remaining=99, retry_after_s=0.0,
                     window_seconds=60.0, current_count=1)
        h = http_headers(d)
        self.assertEqual(h["X-RateLimit-Limit"], "100")
        self.assertEqual(h["X-RateLimit-Remaining"], "99")
        self.assertEqual(h["X-RateLimit-Window"], "60")
        self.assertNotIn("Retry-After", h)

    def test_blocked_headers(self) -> None:
        d = Decision(allowed=False, limit=100, remaining=0, retry_after_s=12.4,
                     window_seconds=60.0, current_count=100)
        h = http_headers(d)
        self.assertIn("Retry-After", h)
        self.assertEqual(h["Retry-After"], "13")


if __name__ == "__main__":
    unittest.main()
