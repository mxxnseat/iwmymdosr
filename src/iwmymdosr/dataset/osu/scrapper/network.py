import asyncio
from dataclasses import dataclass
import errno
import functools
import inspect
import random
import time
from typing import Callable
import aiohttp
import anyio
from osu.exceptions import RequestException
from iwmymdosr.utils.main import log

class RateLimiter:
    def __init__(self, rate: float):
        self._interval = 1.0 / rate
        self._lock = anyio.Lock()
        self._next_slot = 0.0

    async def wait(self):
        async with self._lock:
            now = anyio.current_time()
            delay = self._next_slot - now
            if delay > 0:
                await anyio.sleep(delay)
            self._next_slot = max(now, self._next_slot) + self._interval

    def penalize(self, seconds: float):
        self._next_slot = max(self._next_slot, anyio.current_time() + seconds)


def _http_error(exc):
    # osu.py wraps the real aiohttp error inside RequestException
    return exc.__cause__ if isinstance(exc, RequestException) else exc

def http_status(exc):
    return getattr(_http_error(exc), "status", None)

def retry_after_seconds(exc, default: float = 30.0) -> float:
    headers = getattr(_http_error(exc), "headers", None)
    try:
        if headers and "Retry-After" in headers:
            return float(headers["Retry-After"])
    except ValueError:
        pass
    return default

@dataclass(frozen=True)
class Retry:
    penalty: float | None = None  # seconds to tell the shared limiter to back off


Policy = Callable[[BaseException], Retry | None]  # None = don't retry

def network_policy(e: BaseException) -> Retry | None:
    status = http_status(e)
    if isinstance(e, RequestException) and status != 429:
        return None  # real API error, not rate limiting
    if status == 429:
        return Retry(penalty=retry_after_seconds(e))
    if isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError, RequestException)):
        return Retry()
    return None


_TRANSIENT = {
    getattr(errno, n)
    for n in ("EAGAIN", "EINTR", "EBUSY", "ETXTBSY", "EMFILE", "ENFILE", "ESTALE")
    if hasattr(errno, n)
}

def io_policy(e: BaseException) -> Retry | None:
    # FileNotFoundError, PermissionError, ENOSPC etc. are not transient: fail fast
    if isinstance(e, OSError) and e.errno in _TRANSIENT:
        return Retry()
    return None

def http_policy_aiohttp(e: BaseException) -> Retry | None:
    # raise_for_status() raises ClientResponseError, which IS a ClientError,
    # so network_policy would retry every 404/403. Handle it first.
    if isinstance(e, aiohttp.ClientResponseError):
        if e.status == 429:
            return Retry(penalty=retry_after_seconds(e))  # check your helper reads e.headers
        if e.status >= 500:
            return Retry()
        return None  # other 4xx: retrying won't help
    return None

def any_of(*policies: Policy) -> Policy:
    def combined(e):
        for p in policies:
            if (d := p(e)) is not None:
                return d
        return None
    return combined

POLICY = any_of(http_policy_aiohttp, network_policy, io_policy)

def retry(policy: Policy, max_attempts: int = 5, base_delay: float = 1.0, limiter=None):
    def decorator(func):
        def next_delay(e: BaseException, attempt: int) -> float | None:
            decision = policy(e)
            if decision is None or attempt == max_attempts - 1:
                return None  # caller re-raises
            if decision.penalty is not None and limiter is not None:
                limiter().penalize(decision.penalty)
                log(func, f"got 429, pausing everyone for {decision.penalty:.0f}s")
            return base_delay * 2**attempt + random.uniform(0, 0.5)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        delay = next_delay(e, attempt)
                        if delay is None:
                            raise
                        await anyio.sleep(delay)
            return awrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    delay = next_delay(e, attempt)
                    if delay is None:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator