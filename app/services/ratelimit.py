"""Client-side rate limiting for the Gemini free tier.

The free tier allows a fixed number of requests per minute per model (15 for Flash-Lite at the
time of writing). Retrying after a 429 wastes requests from the same budget, so the app paces
itself instead: every call waits for a slot in a sliding one-minute window.

When a 429 still happens (another process using the same key, or a changed quota), the Gemini
error says how long to wait ("Please retry in 52.1s" / retryDelay). `retry_delay_s` reads that;
a per-DAY quota is reported as not retryable, since waiting a minute would not help.
"""

import asyncio
import re
import time
from collections import deque


class RateLimiter:
    def __init__(self, per_minute: int, window_s: float = 60.0):
        self.per_minute, self.window_s = max(1, per_minute), window_s
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait for a slot; returns the seconds spent waiting."""
        waited = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window_s:
                    self._calls.popleft()
                if len(self._calls) < self.per_minute:
                    self._calls.append(now)
                    return waited
                pause = self.window_s - (now - self._calls[0]) + 0.05
                waited += pause
                await asyncio.sleep(pause)


_limiters: dict[tuple[str, int], RateLimiter] = {}


def limiter_for(key: str, per_minute: int) -> RateLimiter:
    """One shared limiter per (model, limit) in this process, so parallel drafts share the budget."""
    return _limiters.setdefault((key, per_minute), RateLimiter(per_minute))


_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.I)
_RETRY_DELAY = re.compile(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s")


def is_rate_limit(e: Exception) -> bool:
    return getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)


def retry_delay_s(e: Exception) -> float | None:
    """Seconds the API asked us to wait, or None if waiting won't help (daily quota) / unknown."""
    text = str(e)
    if "PerDay" in text:
        return None
    for pattern in (_RETRY_IN, _RETRY_DELAY):
        if m := pattern.search(text):
            return float(m.group(1))
    return 10.0 if is_rate_limit(e) else None


async def paced_call(make_call, limiter: RateLimiter, max_wait_s: float = 90.0, attempts: int = 3):
    """Wait for a rate-limit slot, call, and on a 429 wait as long as the API asks (bounded)."""
    waited = 0.0
    for attempt in range(attempts):
        await limiter.acquire()
        try:
            return await make_call()
        except Exception as e:
            delay = retry_delay_s(e) if is_rate_limit(e) else None
            if delay is None or attempt == attempts - 1 or waited + delay > max_wait_s:
                raise
            waited += delay + 0.5
            await asyncio.sleep(delay + 0.5)
