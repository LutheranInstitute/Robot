"""The real ``UdpTransport`` over real loopback sockets, on the real clock.

Everything about the wire that an in-memory transport cannot say: binding and the port a
bind fails on, name resolution, the reader thread and its exit, a foreign datagram on the
state port, the sender allowlist, and a Ctrl-C landing in a real wait. Every socket here is
bound to port 0 and kept (see ``LoopbackEdge``): the SDK's state socket is bound by the SDK,
and the edge learns the port from the transport, so no port is ever reserved and released.

Everything else about ``Robot`` runs on the virtual clock against the in-memory edge
(``test_robot.py`` and friends).
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time

import pytest

from menlo.asimov import ConnectError, NotConnectedError, Robot, Velocity
from menlo.asimov.robot import KEEPALIVE_HZ
from menlo.asimov.transport.udp import UdpTransport
from tests.conftest import LoopbackEdge, loopback_transport


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ── binding ───────────────────────────────────────────────────────────────────


def test_the_state_socket_binds_port_zero_and_reports_where(loopback_edge):
    tx = loopback_transport(loopback_edge)
    assert tx._state_address is None, "nothing is bound before open()"
    tx.open()
    try:
        host, port = tx._state_address
        assert host == "127.0.0.1" and port > 0
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            with pytest.raises(OSError):
                probe.bind(("127.0.0.1", port))  # the SDK holds it: nobody else can
        finally:
            probe.close()
    finally:
        tx.close()
    assert tx._state_address is None


def test_a_port_someone_else_holds_is_a_connect_error_naming_it(loopback_edge):
    taken = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    taken.bind(("127.0.0.1", 0))
    port = taken.getsockname()[1]
    try:
        tx = loopback_transport(loopback_edge, state_bind=("127.0.0.1", port))
        with pytest.raises(ConnectError, match=rf"127\.0\.0\.1:{port}"):
            tx.open()
        assert tx._sock is None, "nothing is left open behind the error"
    finally:
        taken.close()


def test_transport_open_twice_fails_loudly_instead_of_leaking(loopback_edge):
    tx = loopback_transport(loopback_edge)
    tx.open()
    try:
        first = tx._sock
        with pytest.raises(ConnectError):
            tx.open()  # an ephemeral bind would silently succeed and orphan the first socket
        assert tx._sock is first
    finally:
        tx.close()


def test_the_transport_reopens_after_close(loopback_edge):
    tx = loopback_transport(loopback_edge)
    seen: list[int] = []
    tx.subscribe_state(lambda s: seen.append(s.sequence))
    for _ in range(2):
        tx.open()
        try:
            assert _wait(lambda: bool(seen)), "a datagram became a State on the reader thread"
        finally:
            tx.close()
        seen.clear()


def test_send_before_open_is_not_connected_not_link_lost():
    with pytest.raises(NotConnectedError):
        UdpTransport("127.0.0.1").send(Velocity())


# ── names ─────────────────────────────────────────────────────────────────────


def test_the_host_is_resolved_once_at_open(loopback_edge):
    tx = UdpTransport(
        "localhost", command_port=loopback_edge.command_port, state_bind=("127.0.0.1", 0)
    )
    assert tx._addr is None
    tx.open()
    try:
        assert tx._addr == ("127.0.0.1", loopback_edge.command_port), (
            "sendto() must get an IP, not a name"
        )
    finally:
        tx.close()


def test_an_unresolvable_host_fails_at_connect():
    tx = UdpTransport("no-such-robot.invalid", command_port=8850, state_bind=("127.0.0.1", 0))
    with pytest.raises(ConnectError):
        tx.open()


def test_an_unresolvable_state_source_is_named_in_the_error():
    tx = UdpTransport(
        "127.0.0.1",
        command_port=8850,
        state_bind=("127.0.0.1", 0),
        state_source="no-such-source.invalid",
    )
    with pytest.raises(ConnectError, match="no-such-source\\.invalid"):
        tx.open()


# ── the wire ──────────────────────────────────────────────────────────────────


def test_connect_refuses_when_nobody_answers():
    """A UDP socket that hears nothing is talking to nobody. Say so, and say the fix."""
    robot = Robot(UdpTransport("127.0.0.1", command_port=1, state_bind=("127.0.0.1", 0)))
    with pytest.raises(ConnectError) as exc:
        robot.open(timeout=0.3)
    assert "udp-control" in str(exc.value)


def test_commands_reach_the_edge_and_state_comes_back(loopback_edge, loopback_robot):
    assert loopback_robot.connected and loopback_robot.info.dof == 25
    loopback_edge.set_mode("move")
    loopback_robot.wait_until(lambda s: s.mode.name == "MOVE", timeout=2.0)
    loopback_robot.set_velocity(vx=0.2, wait=False)
    assert loopback_edge.wait_for(lambda rx: (0.2, 0.0, 0.0) in loopback_edge.velocities())
    c = next(c for c in loopback_edge.received if c.HasField("policy"))
    assert c.mode == 2 and c.protocol_version == 1 and c.sequence >= 1


def test_an_undecodable_datagram_is_dropped_not_fatal(loopback_edge, loopback_robot):
    before = loopback_robot.get_state().sequence
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        out.sendto(b"\xff\xff\xff\xff not a protobuf", loopback_robot._tx._state_address)
    finally:
        out.close()
    assert _wait(lambda: loopback_robot.get_state().sequence > before + 5)
    assert loopback_robot.connected


def _spam_state_port(address: tuple[str, int], payload: bytes, stop: threading.Event) -> None:
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not stop.is_set():
        out.sendto(payload, address)
        time.sleep(0.01)
    out.close()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"", id="empty datagram decodes as a default RobotState"),
        pytest.param(
            __import__("menlo.asimov._proto", fromlist=["load"])
            .load()
            .state.RobotState(protocol_version=1, joint_pos=[0.0, 0.0, 0.0])
            .SerializeToString(),
            id="right protocol, wrong robot (3 joints)",
        ),
    ],
)
def test_a_foreign_state_datagram_does_not_keep_the_link_alive(
    loopback_edge, loopback_robot, payload
):
    loopback_robot.link_timeout = 0.3
    loopback_edge.pushing = False
    stop = threading.Event()
    t = threading.Thread(
        target=_spam_state_port, args=(loopback_robot._tx._state_address, payload, stop)
    )
    t.start()
    try:
        time.sleep(0.9)
        assert not loopback_robot.connected, "foreign datagrams refreshed liveness"
        assert len(loopback_robot.get_state().joints) == 25, (
            "a foreign sample became robot.get_state()"
        )
    finally:
        stop.set()
        t.join(timeout=1.0)


def test_state_source_allowlist_drops_everyone_else(loopback_edge):
    elsewhere = Robot(loopback_transport(loopback_edge, state_source="10.255.255.1"))
    with pytest.raises(ConnectError):
        elsewhere.open(timeout=0.5)
    here = Robot(loopback_transport(loopback_edge, state_source="localhost"))
    here.open(timeout=2.0)
    try:
        assert here.connected
    finally:
        here.close()


# ── threads ───────────────────────────────────────────────────────────────────


def test_close_from_the_reader_thread_still_closes_the_transport(loopback_edge, loopback_robot):
    """A state callback that closes the robot runs on the transport's reader thread; the
    reader must not try to join itself, and the socket must still be released."""
    robot = loopback_robot
    done = threading.Event()

    def on_state(_s):
        if not done.is_set():
            done.set()
            robot.close()

    robot._tx.subscribe_state(on_state)
    assert done.wait(2.0)
    assert _wait(lambda: robot._tx._sock is None, 2.0)
    assert not robot.connected


def test_close_from_the_link_lost_callback_really_closes(loopback_edge, loopback_robot):
    """`on_link_lost` runs on the keepalive thread; the obvious handler closes the robot.
    That must not blow up on a self-join and leave the socket (and port) held."""
    robot = loopback_robot
    robot.link_timeout = 0.3
    robot.on_link_lost = lambda exc: robot.close()
    loopback_edge.pushing = False
    assert _wait(lambda: robot._tx._sock is None, 2.0), (
        "transport still open after close() from the callback"
    )
    assert robot._keepalive is None and not robot.connected


def test_reconnecting_from_on_link_lost_leaves_exactly_one_keepalive_thread(loopback_edge):
    """close()+open() inside on_link_lost runs ON the keepalive thread. The old thread must
    exit; a second one re-sending into the edge would double the command rate."""

    def keepalives() -> list[threading.Thread]:
        return [t for t in threading.enumerate() if t.name == "menlo-sdk-keepalive"]

    before = len(keepalives())
    robot = Robot(loopback_transport(loopback_edge))
    reconnected = threading.Event()

    def reconnect(exc) -> None:
        robot.link_timeout = 2.0  # back to normal before the new session starts
        robot.close()
        robot.open(timeout=3.0)
        reconnected.set()

    robot.on_link_lost = reconnect
    robot.open(timeout=3.0)
    try:
        robot.link_timeout = 0.0  # the next keepalive tick declares the link lost
        assert reconnected.wait(3.0)
        time.sleep(0.3)  # long enough for a stale thread to take its next tick
        assert len(keepalives()) == before + 1
        assert robot.connected
    finally:
        robot.close()
    assert _wait(lambda: len(keepalives()) == before, 2.0)


def test_ctrl_c_during_a_waited_walk_zeroes_the_velocity(loopback_edge, loopback_robot):
    """A real SIGINT lands in a real wait: the KeyboardInterrupt ends it, and leaving the
    ``with`` block sends the zero."""
    robot = loopback_robot
    loopback_edge.set_mode("move")
    robot.wait_until(lambda s: s.mode.name == "MOVE", timeout=2.0)
    threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()  # Ctrl-C
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt), robot:
        robot.set_velocity(vx=0.3, duration=10.0)  # wait=True: blocked in the walk
    assert time.monotonic() - started < 2.0, "Ctrl-C ends the wait, not the duration"
    assert loopback_edge.wait_for(lambda _: loopback_edge.velocities()[-1] == (0.0, 0.0, 0.0))
    n = len(loopback_edge.received)
    time.sleep(3 / KEEPALIVE_HZ)
    assert len(loopback_edge.received) == n, "and nothing is re-sent after the zero"


def test_the_loopback_edge_itself_holds_every_port_it_uses():
    """The fixture's own promise: nothing here is reserved and released."""
    edge = LoopbackEdge()
    try:
        for port in (edge.command_port, edge.push_port):
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                with pytest.raises(OSError):
                    probe.bind(("127.0.0.1", port))
            finally:
                probe.close()
    finally:
        edge.close()
