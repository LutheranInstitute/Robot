"""The virtual clock's own promises: one driving thread, and no loop that is always due."""

from __future__ import annotations

import threading

import pytest

from tests.conftest import VirtualClock


def test_only_the_creating_thread_drives_the_clock():
    clock = VirtualClock()
    event = threading.Event()
    seen: list[BaseException | None] = []

    def drive(call) -> None:
        try:
            call()
            seen.append(None)
        except RuntimeError as exc:
            seen.append(exc)

    for call in (
        lambda: clock.sleep(0.1),
        lambda: clock.wait(event, 0.1),
        lambda: clock.advance(0.1),
        lambda: clock.run_until(lambda: False, 0.1),
    ):
        t = threading.Thread(target=drive, args=(call,), name="not-the-driver")
        t.start()
        t.join(5.0)
    assert len(seen) == 4 and all(isinstance(e, RuntimeError) for e in seen), seen
    assert all("'not-the-driver'" in str(e) for e in seen)
    assert clock.now == 1000.0, "nothing moved the clock"
    clock.sleep(0.1)  # the creating thread still can
    assert clock.now == pytest.approx(1000.1)


@pytest.mark.parametrize("period", [0.0, -1.0])
def test_a_loop_that_asks_for_no_delay_cannot_make_advance_spin(period):
    clock = VirtualClock()
    ticks: list[float] = []

    def step() -> float:
        ticks.append(clock.now)
        return period

    clock.run_loop("spinner", step, delay=0.0, stop=None)
    clock.advance(0.01)
    assert len(ticks) == pytest.approx(0.01 / clock.RESOLUTION, abs=1)
    assert clock.now == pytest.approx(1000.01)


def test_call_every_with_no_period_runs_at_the_resolution():
    clock = VirtualClock()
    calls: list[float] = []
    handle = clock.call_every(0.0, lambda: calls.append(clock.now))
    clock.advance(0.005)
    assert len(calls) == 5
    handle.cancel()
    clock.advance(1.0)
    assert len(calls) == 5
