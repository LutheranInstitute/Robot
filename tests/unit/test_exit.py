"""A script that ends without close() still leaves a held velocity at zero.

The scripts run in a child interpreter over real loopback sockets on the real clock: the
point is what the interpreter does at exit, which no in-memory edge can see.
"""

from __future__ import annotations

from tests.conftest import FakeEdge, LoopbackEdge, run_script_over

ZERO = (0.0, 0.0, 0.0)


def _run(edge: LoopbackEdge, body: str, tmp_path) -> tuple[int, str]:
    """Run ``body`` in a child connected to ``edge``; its return code and stderr."""
    code, _out, err = run_script_over(edge, body, tmp_path / "child.err")
    return code, err


def test_unrelated_fake_edge_lifecycle_does_not_start_version_http(clock):
    """UDP-only fixture setup and teardown never allocate an unused HTTP listener."""
    edge = FakeEdge(clock)
    try:
        assert edge._version is None
    finally:
        edge.close()
    assert edge._version is None


def test_opted_in_fake_edge_reuses_version_http_and_stops_it_on_teardown(clock):
    """Explicit preflight access reuses one listener, and teardown stops its server thread."""
    edge = FakeEdge(clock)
    try:
        endpoint = edge.version
        assert edge.version is endpoint
        assert endpoint.thread.is_alive()
    finally:
        edge.close()
    assert not endpoint.thread.is_alive()
    assert endpoint.server.fileno() == -1


def test_a_script_that_exits_holding_a_velocity_sends_zero(loopback_edge, tmp_path):
    loopback_edge.set_mode("move")
    returncode, stderr = _run(
        loopback_edge,
        """
        import time
        robot.set_velocity(vx=0.2, wait=False)  # held until the next command
        time.sleep(0.3)
        # no close(), no `with`: the interpreter exits here
        """,
        tmp_path,
    )
    assert returncode == 0, stderr
    velocities = loopback_edge.velocities
    assert loopback_edge.wait_for(lambda rx: bool(velocities()) and velocities()[-1] == ZERO)
    assert (0.2, 0.0, 0.0) in velocities()


def test_a_script_that_exits_after_a_hold_false_packet_sends_zero(loopback_edge, tmp_path):
    loopback_edge.set_mode("move")
    returncode, stderr = _run(loopback_edge, "robot.set_velocity(vx=0.2, hold=False)\n", tmp_path)
    assert returncode == 0, stderr
    assert loopback_edge.wait_for(lambda rx: loopback_edge.velocities() == [(0.2, 0.0, 0.0), ZERO])


def test_a_script_that_exits_holding_nothing_sends_nothing(loopback_edge, tmp_path):
    loopback_edge.set_mode("move")
    returncode, stderr = _run(loopback_edge, "robot.get_state()\n", tmp_path)
    assert returncode == 0, stderr
    assert loopback_edge.received == []
