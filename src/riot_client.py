"""
Riot API client for long-running collection with a development key.

- RateLimiter: proactive sliding-window limiter per routing value ("americas", "na1", ...),
  so we stay under the app limits instead of bouncing off 429s.
- ApiKeyManager: when the key expires (401/403), all requests pause and .env is re-read
  every few seconds; pasting a new key into .env resumes collection without a restart.
- RiotClient: GET with retries, shared by all collector threads.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

log = logging.getLogger("collector")

# development key app limits; replaced by the X-App-Rate-Limit header once a response arrives
DEFAULT_APP_LIMITS = [(20, 1.0), (100, 120.0)]

KEY_POLL_SECONDS = 30
KEY_REMINDER_SECONDS = 600
KEY_PROBE_ROUTING = "na1"  # any platform works; used to tell an expired key from a forbidden request


class CollectionStopped(Exception):
    """Raised inside worker threads once a shutdown was requested."""


def parse_limit_header(value: str) -> list[tuple[int, float]]:
    """'20:1,100:120' -> [(20, 1.0), (100, 120.0)]"""
    limits = []
    for part in value.split(","):
        n, window = part.split(":")
        limits.append((int(n), float(window)))
    return limits


class RateLimiter:
    """
    Sliding-window limiter for one routing value. Riot counts fixed windows starting at the
    first request, so a sliding window over local send times is slightly conservative.
    """

    def __init__(self, limits: list[tuple[int, float]]):
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()
        self._blocked_until = 0.0
        self._raw_limits: list[tuple[int, float]] = []
        self._limits: list[tuple[int, float]] = []
        self.set_limits(limits)

    def set_limits(self, limits: list[tuple[int, float]]) -> None:
        with self._lock:
            if limits == self._raw_limits:
                return
            self._raw_limits = limits
            # stay one request under each limit to absorb clock/latency differences
            self._limits = [(max(1, n - 1), w) for n, w in limits]

    def block_for(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)

    def acquire(self, stop: threading.Event) -> None:
        while True:
            if stop.is_set():
                raise CollectionStopped
            with self._lock:
                now = time.monotonic()
                longest = max((w for _, w in self._limits), default=0.0)
                while self._calls and now - self._calls[0] >= longest:
                    self._calls.popleft()

                wait = self._blocked_until - now
                for n, w in self._limits:
                    in_window = [t for t in self._calls if now - t < w]
                    if len(in_window) >= n:
                        # the oldest of the last n calls must leave the window first
                        wait = max(wait, in_window[-n] + w - now)

                if wait <= 0:
                    self._calls.append(now)
                    return
            stop.wait(min(wait, 5.0) + 0.01)


class ApiKeyManager:
    def __init__(self, env_path: Path, stop: threading.Event):
        self.env_path = env_path
        self._stop = stop
        self._lock = threading.Lock()
        self._valid = threading.Event()
        self._rejected_key: str | None = None
        self._paused_at = 0.0

        self._key = self._read_key()
        if not self._key:
            raise RuntimeError(f"No RIOT_API_KEY found in {env_path.resolve()}")
        self._valid.set()

        threading.Thread(target=self._watch, name="key-watcher", daemon=True).start()

    def _read_key(self) -> str:
        if not self.env_path.exists():
            return ""
        return (dotenv_values(self.env_path).get("RIOT_API_KEY") or "").strip()

    @property
    def paused(self) -> bool:
        return not self._valid.is_set()

    def get(self) -> str:
        """Current key; blocks while the key is expired."""
        while not self._valid.wait(timeout=1.0):
            if self._stop.is_set():
                raise CollectionStopped
        if self._stop.is_set():
            raise CollectionStopped
        with self._lock:
            return self._key

    def reject(self, key: str) -> None:
        """Called when Riot answers 401/403 for `key`."""
        with self._lock:
            if key != self._key or not self._valid.is_set():
                return  # another thread already reported it, or the key was replaced
            self._valid.clear()
            self._rejected_key = key
            self._paused_at = time.time()
        log.warning(
            "API key rejected (expired?). Collection is PAUSED. "
            "Paste a new key into %s as RIOT_API_KEY=... and it will resume automatically.",
            self.env_path.resolve(),
        )

    def _watch(self) -> None:
        last_reminder = time.time()
        while not self._stop.wait(KEY_POLL_SECONDS):
            if self._valid.is_set():
                continue
            new_key = self._read_key()
            if new_key and new_key != self._rejected_key:
                with self._lock:
                    self._key = new_key
                    self._valid.set()
                paused_for = time.time() - self._paused_at
                log.warning("New API key found in .env after %.0f min paused. Resuming.", paused_for / 60)
            elif time.time() - last_reminder >= KEY_REMINDER_SECONDS:
                last_reminder = time.time()
                log.warning("Still paused: waiting for a new RIOT_API_KEY in %s", self.env_path.resolve())


class RiotClient:
    def __init__(self, keys: ApiKeyManager, stop: threading.Event):
        self.keys = keys
        self.stop = stop
        self._limiters: dict[str, RateLimiter] = {}
        self._limiters_lock = threading.Lock()
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def limiter(self, routing: str) -> RateLimiter:
        routing = routing.lower()
        with self._limiters_lock:
            if routing not in self._limiters:
                self._limiters[routing] = RateLimiter(DEFAULT_APP_LIMITS)
            return self._limiters[routing]

    def _key_works(self, key: str) -> bool | None:
        """True/False if a cheap status call accepts/rejects `key`; None if the check itself failed."""
        self.limiter(KEY_PROBE_ROUTING).acquire(self.stop)
        try:
            r = self._session().get(
                f"https://{KEY_PROBE_ROUTING}.api.riotgames.com/lol/status/v4/platform-data",
                headers={"X-Riot-Token": key},
                timeout=30,
            )
        except requests.RequestException:
            return None
        if r.status_code in (401, 403):
            return False
        return True if r.status_code < 500 else None

    def method_limiter(self, routing: str, method: str) -> RateLimiter:
        """Per-endpoint limiter; starts unlimited and learns its limits from X-Method-Rate-Limit."""
        name = f"{routing.lower()}:{method}"
        with self._limiters_lock:
            if name not in self._limiters:
                self._limiters[name] = RateLimiter([])
            return self._limiters[name]

    def get(
        self,
        routing: str,
        path: str,
        params: dict[str, Any] | None = None,
        method: str = "default",
        max_retries: int = 8,
    ) -> Any:
        """
        GET https://{routing}.api.riotgames.com{path}. `method` names the endpoint for its own
        rate limit. Raises requests.HTTPError for 400/404-style errors, CollectionStopped on shutdown.
        """
        url = f"https://{routing.lower()}.api.riotgames.com{path}"
        limiter = self.limiter(routing)
        method_limiter = self.method_limiter(routing, method)
        failures = 0

        while failures < max_retries:
            key = self.keys.get()
            method_limiter.acquire(self.stop)
            limiter.acquire(self.stop)
            try:
                r = self._session().get(url, params=params, headers={"X-Riot-Token": key}, timeout=30)
            except requests.RequestException as e:
                failures += 1
                log.debug("Network error on %s: %s", url, e)
                self.stop.wait(min(2 ** failures, 60))
                continue

            if "X-App-Rate-Limit" in r.headers:
                limiter.set_limits(parse_limit_header(r.headers["X-App-Rate-Limit"]))
            if "X-Method-Rate-Limit" in r.headers:
                method_limiter.set_limits(parse_limit_header(r.headers["X-Method-Rate-Limit"]))

            if r.status_code in (401, 403):
                # Riot also answers 403 for some bad requests, so check the key separately
                key_ok = self._key_works(key)
                if key_ok is None:
                    failures += 1
                    continue
                if key_ok:
                    r.raise_for_status()
                self.keys.reject(key)
                continue

            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", "5"))
                limit_type = r.headers.get("X-Rate-Limit-Type", "?")
                log.info("429 (%s) on %s %s, waiting %.0fs", limit_type, routing, method, retry_after)
                (method_limiter if limit_type == "method" else limiter).block_for(retry_after + 1)
                continue

            if r.status_code >= 500:
                failures += 1
                self.stop.wait(min(2 ** failures, 60))
                continue

            r.raise_for_status()
            return r.json()

        raise RuntimeError(f"Failed after {max_retries} retries: {url}")
