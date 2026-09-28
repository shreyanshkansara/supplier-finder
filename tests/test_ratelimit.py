"""Free-tier pacing: wait for a slot, honour the API's retry delay, fail fast on daily quotas."""

import asyncio
import time

import pytest

from app.services.ratelimit import RateLimiter, paced_call, retry_delay_s


class Quota(Exception):
    code = 429


async def test_limiter_spaces_calls_within_the_window():
    lim = RateLimiter(per_minute=2, window_s=0.3)
    t0 = time.monotonic()
    for _ in range(3):
        await lim.acquire()
    assert time.monotonic() - t0 >= 0.25  # the third call had to wait for the window


def test_retry_delay_parsing():
    assert retry_delay_s(Quota("429 RESOURCE_EXHAUSTED ... Please retry in 52.09s.")) == 52.09
    assert retry_delay_s(Quota("429 ... 'retryDelay': '20s'")) == 20.0
    assert retry_delay_s(Quota("429 ... quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")) is None
    assert retry_delay_s(ValueError("bad json")) is None


async def test_paced_call_waits_as_asked_then_succeeds(monkeypatch):
    sleeps, calls = [], []
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: sleeps.append(s) or real_sleep(0))

    async def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise Quota("RESOURCE_EXHAUSTED Please retry in 3s")
        return "ok"

    assert await paced_call(flaky, RateLimiter(100)) == "ok"
    assert sleeps == [3.5] and len(calls) == 2


async def test_paced_call_gives_up_on_daily_quota_and_long_waits():
    async def daily():
        raise Quota("RESOURCE_EXHAUSTED quotaId PerDay")

    with pytest.raises(Quota):
        await paced_call(daily, RateLimiter(100))

    async def slow():
        raise Quota("RESOURCE_EXHAUSTED Please retry in 500s")

    with pytest.raises(Quota):
        await paced_call(slow, RateLimiter(100), max_wait_s=60)
