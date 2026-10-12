"""The one clock the SDK reads and sleeps on.

Every deadline, poll, hold, keepalive tick and ``age_s`` in the SDK goes through this
module rather than :mod:`time` and :mod:`threading` directly, so that one object decides
what time it is and how a background loop is clocked. In a running program that object is
:class:`SystemClock`: ``time.monotonic``, ``time.sleep``, ``threading.Event.wait`` and a
daemon thread per loop. The unit suite installs a clock of its own that advances only when
asked, so a 2 s link timeout or a 0.5 s arming hold is a function call rather than a wait.

What stays on the real clock: HTTP timeouts (``_version``), the LiveKit client's asyncio
loop, and the camera's and microphone's condition-variable waits, which wait on another
thread's delivery rather than on the passage of time.

``install()`` exists for the test suite. It swaps a module-level global and is not
thread-safe: install a clock before any Robot, loop or thread of the SDK is running, from
the thread that will drive it, and restore the previous one the same way.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

#: One iteration of a background loop: do the work, return the delay before the next
#: iteration in seconds, or ``None`` to end the loop.
Step = Callable[[], float | None]


class Loop(Protocol):
    """A handle on a running background loop, as a thread would be."""

    def join(self, timeout: float) -> None:
        """Wait for the loop to end, at most ``timeout`` seconds. Returns at once when
        called from the loop itself (a thread cannot join itself)."""

    def is_alive(self) -> bool: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Block until ``event`` is set or ``timeout`` seconds pass; whether it is set."""

    def run_loop(
        self, name: str, step: Step, *, delay: float, stop: threading.Event | None
    ) -> Loop:
        """Start a background loop: wait ``delay``, run ``step``, wait what it returned,
        and so on until it returns ``None`` or ``stop`` is set (checked before every
        iteration, and a set ``stop`` wakes the loop from its wait at once)."""


class _Thread(threading.Thread):
    def __init__(self, name: str, step: Step, delay: float, stop: threading.Event | None):
        super().__init__(name=name, daemon=True)
        self._step, self._delay, self._halt = step, delay, stop  # not _stop: Thread owns that name

    def run(self) -> None:
        delay: float | None = self._delay
        while delay is not None:
            if self._halt is None:
                time.sleep(delay)
            elif self._halt.wait(delay):
                return
            delay = self._step()

    def join(self, timeout: float | None = None) -> None:
        if self is not threading.current_thread():
            super().join(timeout)


class SystemClock:
    """The real thing: wall time and threads."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def wait(self, event: threading.Event, timeout: float) -> bool:
        return event.wait(timeout)

    def run_loop(
        self, name: str, step: Step, *, delay: float, stop: threading.Event | None
    ) -> Loop:
        thread = _Thread(name, step, delay, stop)
        thread.start()
        return thread


_current: Clock = SystemClock()


def install(clock: Clock) -> Clock:
    """Make ``clock`` the SDK's clock; returns the one it replaces, for restoring."""
    global _current
    previous, _current = _current, clock
    return previous


def monotonic() -> float:
    return _current.monotonic()


def sleep(seconds: float) -> None:
    _current.sleep(seconds)


def wait(event: threading.Event, timeout: float) -> bool:
    return _current.wait(event, timeout)


def run_loop(name: str, step: Step, *, delay: float, stop: threading.Event | None) -> Loop:
    return _current.run_loop(name, step, delay=delay, stop=stop)
