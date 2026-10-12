"""Shared fixtures.

The unit suite runs the SDK on a :class:`VirtualClock`: time advances only when the test,
or the code under test, sleeps or waits on it, and every background loop the SDK starts
(the keepalive, a ``set_joints`` motion) and every telemetry frame the fake edge pushes is
an event on that clock. A 2 s link timeout is one call, and nothing in a test depends on
how fast the machine is.

The fake edge (:class:`FakeEdge`) speaks the real wire in memory: it parses the same
``RobotCommand`` bytes the SDK encodes and pushes the same ``RobotState`` bytes the SDK
decodes, with no socket in between. :class:`MemoryTransport` is the UDP lane over it, and
:class:`FakeLiveKitClient` the room, so the three connection modes share one edge.

The real ``UdpTransport`` has its own tests over loopback sockets: :class:`LoopbackEdge`
binds every socket to port 0 and keeps it, and learns the SDK's state port from the
transport once it is bound, so no port is ever reserved and released.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from menlo.asimov import (
    Applied,
    ConnectError,
    LinkLostError,
    NotConnectedError,
    Refused,
    Robot,
    UnsupportedError,
    _clock,
)
from menlo.asimov._command import Command
from menlo.asimov._media import AudioChunk
from menlo.asimov._state import TransportKind
from menlo.asimov.transport._livekit_client import identity_from_token
from menlo.asimov.transport._wire import decode_state, encode_command
from menlo.asimov.transport.base import (
    AudioCallback,
    ControllerCallback,
    FrameCallback,
    OutcomeCallback,
    StateCallback,
)
from menlo.asimov.transport.livekit import HybridTransport, LiveKitTransport
from menlo.asimov.transport.udp import UdpTransport

#: ``CONTROL_MODE_FAULT_DAMP``, by value: asimov.io protocol v1.3.0 adds it, and the SDK
#: reads robot modes by value, so it works whatever version of the bindings is installed.
FAULT_DAMP = 5


# ── the clock ─────────────────────────────────────────────────────────────────


class _Task:
    __slots__ = ("cancelled", "fn")

    def __init__(self, fn: Callable[[], None]) -> None:
        self.fn = fn
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _VirtualLoop:
    """What ``run_loop`` hands back: alive until its step ends it or its stop flag is set."""

    def __init__(self, stop: threading.Event | None) -> None:
        self._stop = stop
        self.ended = False

    def is_alive(self) -> bool:
        return not self.ended and not (self._stop is not None and self._stop.is_set())

    def join(self, timeout: float) -> None:
        pass  # the loop runs only when the clock runs: there is nothing to wait for


class VirtualClock:
    """A clock that moves only when asked, with a scheduler for what happens as it moves.

    ``sleep(s)`` and ``wait(event, s)`` run every scheduled task due in the next ``s``
    seconds, in order, on the calling thread, and then set the time; so a test's
    ``clock.sleep(0.3)`` lets the fake edge push 30 frames and the keepalive tick three
    times, and the SDK's own polls and holds do the same from inside a verb.
    ``run_until(predicate, timeout)`` advances event by event until the predicate holds.

    The thread that creates it drives it: ``sleep``, ``wait``, ``advance`` and ``run_until``
    raise from any other thread, since they run tasks and move ``now`` unsynchronised, and
    an event another thread will set is never waited for here. A test that wants something
    to happen at a moment schedules it (``call_later``) rather than sleeping on another
    thread.
    """

    #: ``sleep(0)`` yields the processor; here it yields this much time, so a poll loop
    #: that sleeps nothing still moves towards its deadline.
    RESOLUTION = 0.001

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self._queue: list[tuple[float, int, _Task]] = []
        self._order = itertools.count()
        self._lock = threading.RLock()
        self._owner = threading.get_ident()

    def _driven_here(self, what: str) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError(
                f"VirtualClock.{what}() from thread {threading.current_thread().name!r}: only "
                "the thread that created the clock drives it; schedule with call_later instead"
            )

    # ── menlo.asimov._clock.Clock ────────────────────────────────────────────
    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self._driven_here("sleep")
        self.advance(max(float(seconds), self.RESOLUTION))

    def wait(self, event: threading.Event, timeout: float) -> bool:
        self._driven_here("wait")
        deadline = self.now + max(0.0, float(timeout))
        while not event.is_set():
            if not self._run_next(deadline):
                self.now = max(self.now, deadline)
                return event.is_set()
        return True

    def run_loop(
        self,
        name: str,
        step: Callable[[], float | None],
        *,
        delay: float,
        stop: threading.Event | None,
    ) -> _VirtualLoop:
        loop = _VirtualLoop(stop)

        def tick() -> None:
            if loop.ended or (stop is not None and stop.is_set()):
                loop.ended = True
                return
            again = step()
            if again is None:
                loop.ended = True
                return
            # A step that asks for no delay comes back at the resolution, as ``sleep(0)``
            # does, so ``advance`` cannot spin on a loop that is always due.
            self.call_later(max(again, self.RESOLUTION), tick)

        self.call_later(delay, tick)
        return loop

    # ── scheduling ───────────────────────────────────────────────────────────
    def call_later(self, delay: float, fn: Callable[[], None]) -> _Task:
        task = _Task(fn)
        with self._lock:
            heapq.heappush(self._queue, (self.now + max(0.0, delay), next(self._order), task))
        return task

    def call_every(self, period: float, fn: Callable[[], None]) -> _Task:
        """``fn`` every ``period`` seconds from now, until the returned task is cancelled."""
        period = max(float(period), self.RESOLUTION)  # never a task that is always due
        handle = _Task(fn)

        def tick() -> None:
            if handle.cancelled:
                return
            fn()
            if not handle.cancelled:
                self.call_later(period, tick)

        self.call_later(period, tick)
        return handle

    def advance(self, seconds: float) -> None:
        """Run everything due in the next ``seconds`` seconds, then set the time."""
        self._driven_here("advance")
        limit = self.now + max(0.0, float(seconds))
        while self._run_next(limit):
            pass
        self.now = max(self.now, limit)

    def run_until(self, predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
        """Advance event by event until ``predicate`` holds, at most ``timeout`` seconds."""
        self._driven_here("run_until")
        deadline = self.now + max(0.0, float(timeout))
        while not predicate():
            if not self._run_next(deadline):
                self.now = max(self.now, deadline)
                return predicate()
        return True

    def _run_next(self, limit: float) -> bool:
        """Run the earliest task due at or before ``limit``; whether there was one."""
        with self._lock:
            while self._queue and self._queue[0][2].cancelled:
                heapq.heappop(self._queue)
            if not self._queue or self._queue[0][0] > limit:
                return False
            due, _, task = heapq.heappop(self._queue)
            self.now = max(self.now, due)
        task.fn()
        return True


@pytest.fixture
def clock() -> Iterator[VirtualClock]:
    """The SDK's clock for the duration of the test."""
    virtual = VirtualClock()
    previous = _clock.install(virtual)
    try:
        yield virtual
    finally:
        _clock.install(previous)


# ── the fake edge, in memory ──────────────────────────────────────────────────


def ankle_coupling(positions: list[float]) -> list[float]:
    """What the biped firmware does to a trajectory's ankle entries: it reads them as the
    ankle's (pitch, roll), limits them to 0.35 and 0.1 rad, and drives motors A and B."""
    if len(positions) != 25:
        return positions
    for a, b in ((4, 5), (10, 11)):
        pitch = max(-0.35, min(0.35, positions[a]))
        roll = max(-0.1, min(0.1, positions[b]))
        positions[a] = 2.02 * pitch - 0.8 * roll
        positions[b] = -2.02 * pitch - 0.8 * roll
    return positions


class FakeEdge:
    """The edge's UDP lane as the SDK sees it, in memory: bare ``RobotCommand`` bytes in
    (``receive``), bare ``RobotState`` bytes pushed to every attached sink at ``state_hz``
    on the virtual clock.

    Deliberately shaped like Asimov Edge's ``UdpConnector`` (sender address ignored, state
    to whoever is attached), so a test that passes here says something about the robot.
    Records every decoded command; the state it pushes is whatever the test sets.

    ``firmware=True`` makes the robot mode follow the commands the way the Motion Control
    Board firmware does: STAND and DAMP take effect at once; MOVE (a velocity or a
    trajectory) is entered only from STAND once the robot has been upright (gravity z below
    -0.87) for ``arm_hold_s``, and until then the robot stays in STAND; MOVE -> STAND stays
    armed; a velocity or trajectory while DAMPed is dropped (the edge's DAMP gate); a
    trajectory in MOVE sets the reported joint positions to its targets (after the
    firmware's ankle coupling); and a latched fault (``fault()``, or ``fault_damp()`` for
    robot mode FAULT_DAMP) holds the robot in DAMP and ignores STAND and MOVE.
    """

    def __init__(
        self,
        clock: VirtualClock,
        *,
        state_hz: float = 100.0,
        alerts_every: int = 1,
        firmware: bool = False,
        arm_hold_s: float = 0.5,
    ) -> None:
        from menlo.asimov._proto import load

        pb = load()  # same bindings the SDK uses, whichever source it resolved to
        self._cmd_pb, self._common_pb, self._st_pb = pb.command, pb.common, pb.state
        self.clock = clock
        self._version: FakeVersionEndpoint | None = None
        self.received: list = []  # decoded RobotCommand protos, in arrival order
        self.arrived: list[float] = []  # clock time of receipt of each entry in received
        self.state = self._st_pb.RobotState(
            current_mode=self._common_pb.CONTROL_MODE_DAMP, protocol_version=1
        )
        self.state.joint_pos.extend([0.0] * 25)
        self.state.projected_gravity.extend([0.0, 0.0, -1.0])
        self.pushing = True
        self.alerts_every = alerts_every  # firmware ships the alert block every 20th frame
        self.firmware = firmware
        self.arm_hold_s = arm_hold_s
        self.armed = False
        self.armed_at: float | None = None  # clock time the fake firmware armed
        self.move_entered_at: float | None = None
        self.first_velocity_at: float | None = None  # clock time of the first velocity
        self._upright_since: float | None = None
        self._fw_lock = threading.RLock()
        self._sinks: list[Callable[[bytes], None]] = []
        self._seq = 0
        self._frames = clock.call_every(1.0 / state_hz, self.frame)

    # ── the wire ─────────────────────────────────────────────────────────────
    def attach(self, sink: Callable[[bytes], None]) -> None:
        """``sink`` gets every state frame from now on, as the SDK's socket would."""
        self._sinks.append(sink)

    def detach(self, sink: Callable[[bytes], None]) -> None:
        with suppress(ValueError):
            self._sinks.remove(sink)

    def receive(self, payload: bytes) -> None:
        """One command datagram, as the edge's socket would read it."""
        c = self._cmd_pb.RobotCommand()
        c.ParseFromString(payload)
        if c.HasField("policy") and self.first_velocity_at is None:
            self.first_velocity_at = self.clock.now
        self.arrived.append(self.clock.now)
        self.received.append(c)
        if self.firmware:
            self._follow(c)

    def frame(self) -> None:
        """One telemetry period: the firmware's arming tick, then one state frame."""
        if self.firmware:
            self._arm_tick()
        if not self.pushing:
            return
        self._seq += 1
        self.state.sequence = self._seq
        self.state.timestamp_us = int(self.clock.now * 1e6)
        if self.alerts_every > 1 and self._seq % self.alerts_every:
            frame = self._st_pb.RobotState()
            frame.CopyFrom(self.state)
            del frame.active_alerts[:]
            payload = frame.SerializeToString()
        else:
            payload = self.state.SerializeToString()
        for sink in tuple(self._sinks):
            sink(payload)

    # ── the firmware's mode machine (firmware=True) ──────────────────────────
    def _mode(self) -> int:
        return int(self.state.current_mode)

    def _enter(self, mode: int) -> None:
        stand, move = self._common_pb.CONTROL_MODE_STAND, self._common_pb.CONTROL_MODE_MOVE
        old = self._mode()
        if old == mode:
            return
        if not (old == move and mode == stand):
            self.armed = False
            self._upright_since = None
        if mode == move:
            self.move_entered_at = self.clock.now
        self.state.current_mode = mode

    def _follow(self, c: Any) -> None:
        pb = self._common_pb
        with self._fw_lock:
            drive = c.HasField("policy") or c.HasField("all_trajectory")
            if self.state.error_flags or self._mode() == FAULT_DAMP:
                return  # latched: DAMP until the firmware restarts
            if drive:
                mode = self._mode()
                if mode == pb.CONTROL_MODE_DAMP:
                    return  # the edge drops a velocity or trajectory while DAMPed
                if mode == pb.CONTROL_MODE_STAND and self.armed:
                    self._enter(pb.CONTROL_MODE_MOVE)
                if c.HasField("all_trajectory") and self._mode() == pb.CONTROL_MODE_MOVE:
                    # The joints reach a trajectory's targets at once: tracking is perfect.
                    del self.state.joint_pos[:]
                    self.state.joint_pos.extend(ankle_coupling(list(c.all_trajectory.positions)))
            elif c.mode == pb.CONTROL_MODE_STAND:
                self._enter(pb.CONTROL_MODE_STAND)
            elif c.mode == pb.CONTROL_MODE_DAMP:
                self._enter(pb.CONTROL_MODE_DAMP)

    def _arm_tick(self) -> None:
        with self._fw_lock:
            g = list(self.state.projected_gravity)
            now = self.clock.now
            if g and g[2] < -0.87:
                if self._upright_since is None:
                    self._upright_since = now
                elif (
                    not self.armed
                    and self._mode() == self._common_pb.CONTROL_MODE_STAND
                    and now - self._upright_since >= self.arm_hold_s
                ):
                    self.armed = True
                    self.armed_at = now
            else:
                self._upright_since = None

    def fault(self, alert_id: int = 7) -> None:
        """Latch a critical alert (default FALL_DETECTED): DAMP, error_flags set."""
        with self._fw_lock:
            self.state.error_flags |= 1 | (1 << (1 + alert_id))
            self._enter(self._common_pb.CONTROL_MODE_DAMP)

    def fault_damp(self) -> None:
        """Robot mode FAULT_DAMP: the firmware's latched emergency damping."""
        with self._fw_lock:
            self._enter(FAULT_DAMP)

    # ── test helpers ─────────────────────────────────────────────────────────
    def set_mode(self, mode: str) -> None:
        self.state.current_mode = getattr(self._common_pb, f"CONTROL_MODE_{mode.upper()}")

    def velocities(self) -> list[tuple[float, float, float]]:
        return [
            (round(c.policy.vx, 4), round(c.policy.vy, 4), round(c.policy.vyaw, 4))
            for c in self.received
            if c.HasField("policy")
        ]

    def velocities_between(
        self, start: float, end: float = math.inf
    ) -> list[tuple[float, float, float]]:
        """The velocities received from clock time ``start`` until ``end``."""
        return [
            (round(c.policy.vx, 4), round(c.policy.vy, 4), round(c.policy.vyaw, 4))
            for c, at in zip(tuple(self.received), tuple(self.arrived), strict=False)
            if start <= at < end and c.HasField("policy")
        ]

    def modes(self) -> list[str]:
        names = {0: "damp", 1: "stand", 2: "move"}
        return [names[c.mode] for c in self.received if not c.HasField("policy")]

    def wait_for(self, predicate: Callable[[list], bool], timeout: float = 3.0) -> bool:
        """Run the clock, frame by frame and tick by tick, until ``predicate(received)``."""
        return self.clock.run_until(lambda: predicate(self.received), timeout)

    @property
    def version(self) -> FakeVersionEndpoint:
        """Start real HTTP only for tests that explicitly need version preflight."""
        if self._version is None:
            self._version = FakeVersionEndpoint()
        return self._version

    def close(self) -> None:
        self._frames.cancel()
        if self._version is not None:
            self._version.close()


class MemoryTransport:
    """The ``udp`` lane over a :class:`FakeEdge`, with no socket: the same bytes go the
    same way, synchronously. Plus one test-only door, ``deliver_outcome``: a verdict as if
    the edge had sent one."""

    kind: TransportKind = "udp"
    default_outcome_timeout: float = UdpTransport.default_outcome_timeout
    silence_hint: str = UdpTransport.silence_hint
    capabilities: frozenset[str] = frozenset({"drive", "state"})

    def __init__(self, edge: FakeEdge, *, endpoint: str = "127.0.0.1:8850") -> None:
        self._edge = edge
        self.endpoint = endpoint
        self._is_open = False
        self._seq = 0
        self._lock = threading.Lock()
        self._on_state: list[StateCallback] = []
        self._on_outcome: list[OutcomeCallback] = []
        self._on_controller: list[ControllerCallback] = []

    def open(self) -> None:
        if self._is_open:
            raise ConnectError("this MemoryTransport is already open")
        self._is_open = True
        self._edge.attach(self._deliver)

    def close(self) -> None:
        self._edge.detach(self._deliver)
        self._is_open = False

    def send(self, command: Command) -> int:
        if not self._is_open:
            raise NotConnectedError("this MemoryTransport is not open")
        with self._lock:
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            seq = self._seq
        self._edge.receive(encode_command(command, seq))
        return seq

    def _deliver(self, payload: bytes) -> None:
        try:
            state = decode_state(payload, None)  # Robot names the joints
        except Exception:
            return  # a bad datagram is dropped, as the real reader drops it
        for cb in tuple(self._on_state):
            cb(state)

    def deliver_outcome(self, outcome: Applied | Refused) -> None:
        for cb in tuple(self._on_outcome):
            cb(outcome)

    def subscribe_state(self, callback: StateCallback) -> None:
        self._on_state.append(callback)

    def subscribe_outcome(self, callback: OutcomeCallback) -> None:
        self._on_outcome.append(callback)

    def subscribe_controller_change(self, callback: ControllerCallback) -> None:
        self._on_controller.append(callback)

    def subscribe_frames(self, callback: FrameCallback) -> None:
        raise UnsupportedError("camera", self.kind)

    def subscribe_audio(self, callback: AudioCallback) -> None:
        raise UnsupportedError("microphone", self.kind)

    def play_audio(self, chunk: AudioChunk) -> None:
        raise UnsupportedError("speaker", self.kind)


def route_udp_to(edge: FakeEdge, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``udp`` lane a :class:`ConnectionConfig` builds is a :class:`MemoryTransport`
    over ``edge``, whatever host and ports the config names."""
    from menlo.asimov import connection

    def memory(host: str, *, command_port: int = 8850, **_: object) -> MemoryTransport:
        return MemoryTransport(edge, endpoint=f"{host}:{command_port}")

    monkeypatch.setattr(connection, "UdpTransport", memory)


def connect_udp(
    edge: FakeEdge,
    *,
    limits: Any = None,
    link_timeout: float | None = None,
    **connect_kw: Any,
) -> Robot:
    """``Robot(ConnectionConfig(udp=...)).connect("udp", ...)`` over ``edge``, in one
    call, for tests that only care about the UDP lane."""
    from menlo.asimov.connection import ConnectionConfig, UdpConfig

    robot_kw: dict[str, Any] = {}
    if limits is not None:
        robot_kw["limits"] = limits
    if link_timeout is not None:
        robot_kw["link_timeout"] = link_timeout
    cfg = ConnectionConfig(udp=UdpConfig("127.0.0.1"))
    return Robot(cfg, **robot_kw).connect("udp", **connect_kw)


def put_in(robot: Robot, edge: FakeEdge, mode: str) -> None:
    """Make the fake robot report robot mode ``mode`` and wait until ``robot`` has seen it:
    a motion command checks the robot mode before it sends."""
    edge.set_mode(mode)
    robot.wait_until(lambda s: s.mode.name == mode.upper(), timeout=2.0)


@pytest.fixture
def edge(clock: VirtualClock, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeEdge]:
    """A fake edge that reports what the test sets and follows no command."""
    e = FakeEdge(clock)
    route_udp_to(e, monkeypatch)
    yield e
    e.close()


@pytest.fixture
def fw_edge(clock: VirtualClock, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeEdge]:
    """A fake edge whose robot mode follows the commands as the firmware does."""
    e = FakeEdge(clock, firmware=True, state_hz=200.0)
    route_udp_to(e, monkeypatch)
    yield e
    e.close()


@pytest.fixture
def robot(edge: FakeEdge) -> Iterator[Robot]:
    r = Robot(MemoryTransport(edge))
    r.open(timeout=3.0)
    yield r
    r.close()


# ── the LiveKit client seam, over the same edge ───────────────────────────────


class FakeLiveKitClient:
    """The ``LiveKitClient`` seam with the room replaced by the FakeEdge, in memory.

    This is why no unit test needs livekit installed, or a server running: the SDK's only
    ``livekit`` imports live behind this seam, so faking the seam exercises everything
    above it: the topics, the wire bytes, the capability honesty, the media plumbing.

    ``carry_state=False`` is the hybrid lane's shape: media only, because control and state
    are on the UDP transport beside it.

    ``command_track=True`` is a robot whose Asimov Edge reads the SDK's ``commands`` data
    track (its participant attribute says so); ``refuse_track=True`` is a session the
    server will not let publish one. A frame on the track reaches the FakeEdge as the same
    bytes a packet does, and is recorded in ``frames``. The first frame only starts the
    track's publication and does not go out, as on a real room.
    """

    def __init__(
        self,
        edge: FakeEdge,
        *,
        tracks: tuple[str, ...] = ("camera", "microphone"),
        carry_state: bool = True,
        token: str = "fake-token",
        command_track: bool = False,
        refuse_track: bool = False,
        can_publish_data: bool | None = True,
    ) -> None:
        self._edge = edge
        self.command_track = command_track
        self._refuse_track = refuse_track
        self.frames: list[tuple[str, bytes]] = []  # (data track name, payload)
        self._local_tracks: set[str] = set()
        self._room_tracks = frozenset(tracks)
        self._carry_state = carry_state
        self._token = token
        self.tracks: frozenset[str] = frozenset()
        self.identity: str | None = None  # read out of the token at connect, as the real one does
        self.can_publish_data = can_publish_data
        self.connected = False
        self.played: list = []  # AudioChunks handed to the speaker track
        self.published: list[tuple[str, bytes]] = []  # (topic, payload)
        self.rpc_calls: list[tuple[str, str, str, float]] = []
        self.system_info_response = '{"robot_os_version":"0.2.0","robot_model":"asimov_1"}'
        self.state_publisher_identity = "MENLO-TEST"
        self._data_track_cbs: dict[str, list] = {}
        self._video_cbs: list = []
        self._audio_cbs: list = []
        self._track_cbs: list = []

    # ── the seam ─────────────────────────────────────────────────────────────
    def connect(self) -> None:
        if self.connected:
            raise RuntimeError("already connected")
        if self._carry_state:
            self._edge.attach(self._state_frame)
        self.identity = identity_from_token(self._token)
        self.connected = True
        self._set_tracks(self._room_tracks)

    def close(self) -> None:
        self._edge.detach(self._state_frame)
        self.connected = False
        self.identity = None
        self._local_tracks.clear()
        self._set_tracks(frozenset())

    def wait_for_tracks(self, timeout: float) -> frozenset[str]:
        return self.tracks

    def publish_data(self, payload: bytes, *, topic: str) -> None:
        if not self.connected:
            raise LinkLostError("the fake room is not joined")
        self.published.append((topic, payload))
        self._edge.receive(payload)

    def push_data_frame(self, name: str, payload: bytes) -> bool:
        if not self.connected:
            raise LinkLostError("the fake room is not joined")
        if name not in self._local_tracks:
            if not self._refuse_track:
                self._local_tracks.add(name)
            return False
        self.frames.append((name, payload))
        self._edge.receive(payload)
        return True

    def publisher_attributes(self, track: str) -> dict[str, str]:
        if not (self.connected and self._carry_state and track == "state"):
            return {}
        return {"asimov.commands_track": "1"} if self.command_track else {}

    def wait_for_data_track_publisher(self, name: str, timeout: float) -> str:
        if name != "state" or not self.state_publisher_identity:
            raise ConnectError(f"the fake room did not publish {name!r} within {timeout:g}s")
        return self.state_publisher_identity

    def perform_rpc(
        self,
        destination_identity: str,
        method: str,
        payload: str,
        timeout: float,
    ) -> str:
        if not self.connected:
            raise LinkLostError("the fake room is not joined")
        self.rpc_calls.append((destination_identity, method, payload, timeout))
        return self.system_info_response

    def publish_audio(self, chunk: AudioChunk) -> None:
        if not self.connected:
            raise LinkLostError("the fake room is not joined")
        self.played.append(chunk)

    def on_data_track(self, name: str, callback: Callable[..., None]) -> None:
        self._data_track_cbs.setdefault(name, []).append(callback)

    def on_video(self, callback: Callable[..., None]) -> None:
        self._video_cbs.append(callback)

    def on_audio(self, callback: Callable[..., None]) -> None:
        self._audio_cbs.append(callback)

    def on_tracks(self, callback: Callable[..., None]) -> None:
        self._track_cbs.append(callback)

    # ── plumbing + test helpers ──────────────────────────────────────────────
    def _state_frame(self, payload: bytes) -> None:
        # The FakeEdge speaks UDP, which carries no edge clock: user_timestamp=None.
        for cb in tuple(self._data_track_cbs.get("state", ())):
            cb(payload, None)

    def _set_tracks(self, tracks: frozenset[str]) -> None:
        if tracks == self.tracks:
            return
        self.tracks = tracks
        for cb in tuple(self._track_cbs):
            cb(tracks)

    def push_frame(self, n: int) -> Any:
        from menlo.asimov import Frame

        f = Frame(width=4, height=2, encoding="rgb8", data=bytes(24), stride_bytes=12, sequence=n)
        for cb in tuple(self._video_cbs):
            cb(f)
        return f

    def push_audio(self, n: int, *, samples: int = 160) -> AudioChunk:
        a = AudioChunk(16_000, 1, samples, "pcm_s16le", bytes(2 * samples), sequence=n)
        for cb in tuple(self._audio_cbs):
            cb(a)
        return a


def hybrid_transport(edge: FakeEdge, client: FakeLiveKitClient, **kw: Any) -> HybridTransport:
    """A ``HybridTransport`` whose UDP half is a :class:`MemoryTransport` over ``edge`` and
    whose room is ``client``."""
    tx = HybridTransport(
        "127.0.0.1",
        livekit_url=kw.pop("livekit_url", "ws://fake"),
        room=kw.pop("room", "asimov-room"),
        token=kw.pop("token", "test-token"),
        client=client,
        **kw,
    )
    tx._udp = MemoryTransport(edge)  # the UDP half, in memory; the room half is the client
    tx.endpoint = tx._describe()
    return tx


def make_livekit_robot(edge: FakeEdge, **kw: Any) -> tuple[FakeLiveKitClient, Robot]:
    """Mode B: everything over the (faked) room."""
    client = FakeLiveKitClient(edge, **kw)
    tx = LiveKitTransport("ws://fake", "asimov-room", token="test-token", client=client)
    return client, Robot(tx)


def make_hybrid_robot(edge: FakeEdge, **kw: Any) -> tuple[FakeLiveKitClient, Robot]:
    """Mode A: UDP control, (faked) room media."""
    client = FakeLiveKitClient(edge, carry_state=False, **kw)
    return client, Robot(hybrid_transport(edge, client))


@pytest.fixture
def livekit_robot(edge: FakeEdge) -> Iterator[tuple[FakeLiveKitClient, Robot]]:
    client, r = make_livekit_robot(edge)
    r.open(timeout=3.0)
    yield client, r
    r.close()


@pytest.fixture
def hybrid_robot(edge: FakeEdge) -> Iterator[tuple[FakeLiveKitClient, Robot]]:
    client, r = make_hybrid_robot(edge)
    r.open(timeout=3.0)
    yield client, r
    r.close()


@pytest.fixture(params=["udp", "hybrid", "livekit"])
def any_robot(request: pytest.FixtureRequest, edge: FakeEdge) -> Iterator[tuple[str, Robot]]:
    """The same robot over each of the three lanes. A behaviour that is not mode-agnostic
    is not done: every promise below is asserted three times."""
    if request.param == "udp":
        r: Robot = Robot(MemoryTransport(edge))
    else:
        maker = make_hybrid_robot if request.param == "hybrid" else make_livekit_robot
        _client, r = maker(edge)
    r.open(timeout=3.0)
    yield request.param, r
    r.close()


def route_rooms_to(edge: FakeEdge, monkeypatch: pytest.MonkeyPatch) -> list[FakeLiveKitClient]:
    """Every ``livekit`` and ``hybrid`` lane a :class:`ConnectionConfig` builds joins
    ``edge`` through the fake client. Returns the clients, in the order they were made, so
    a test can push camera frames and audio into them."""
    from menlo.asimov import connection

    clients: list[FakeLiveKitClient] = []

    class Room(LiveKitTransport):
        def __init__(self, url: str, room: str, *, token: Any, **kw: Any) -> None:
            client = FakeLiveKitClient(edge, token=token() if callable(token) else token)
            clients.append(client)
            super().__init__(url, room, token=token, client=client, **kw)

    class Hybrid(HybridTransport):
        def __init__(self, host: str, *, livekit_url: str, room: str, token: Any, **kw: Any):
            client = FakeLiveKitClient(edge, carry_state=False)
            clients.append(client)
            super().__init__(
                host, livekit_url=livekit_url, room=room, token=token, client=client, **kw
            )
            self._udp = MemoryTransport(edge)
            self.endpoint = self._describe()

    monkeypatch.setattr(connection, "LiveKitTransport", Room)
    monkeypatch.setattr(connection, "HybridTransport", Hybrid)
    return clients


def route_manager_rooms_to(edge: FakeEdge, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every ``LiveKitTransport`` a ``ManagerConfig`` builds join the FakeEdge through
    the fake client instead of a real room. Returns the list the tokens each join presented
    are appended to, so a test can see which mint a join used."""
    from menlo.asimov import connection

    seen_tokens: list[str] = []
    real = connection.LiveKitTransport

    class Capturing(real):  # type: ignore[misc,valid-type]
        def __init__(self, url: str, room: str, *, token: Any, **kw: Any) -> None:
            client = FakeLiveKitClient(edge, token=token() if callable(token) else token)
            seen_tokens.append(client._token)
            super().__init__(url, room, token=token, client=client, **kw)

    monkeypatch.setattr(connection, "LiveKitTransport", Capturing)
    return seen_tokens


# ── the manager ───────────────────────────────────────────────────────────────


class FakeManager(HTTPServer):
    """A stand-in for asimov-manager's POST /api/livekit/token."""

    def __init__(self) -> None:
        self.requests: list[tuple[dict, dict]] = []  # (headers, body)
        self.status = 200
        self.reply: dict = {
            "url": "ws://robot:7880",
            "room": "robot-menlo-0042",
            "token": "jwt-1",
            "identity": "sdk-abc",
            "expires_at": "2099-01-01T00:00:00Z",
        }
        self.minted = 0
        self.raw_body: bytes | None = None  # when set, answered verbatim with raw_type
        self.raw_type = "text/html"
        self.redirect_to: str | None = None  # when set, every request is a 302 there
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                body = json.loads(raw) if raw.startswith(b"{") else {}
                server.requests.append((dict(self.headers), body))
                if server.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", server.redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if server.raw_body is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", server.raw_type)
                    self.send_header("Content-Length", str(len(server.raw_body)))
                    self.end_headers()
                    self.wfile.write(server.raw_body)
                    return
                server.minted += 1
                reply = dict(server.reply)
                if server.status == 200:
                    reply["token"] = f"jwt-{server.minted}"
                payload = json.dumps(
                    reply if server.status == 200 else {"error": "no LiveKit here"}
                )
                self.send_response(server.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload.encode())

            def log_message(self, *_: object) -> None:
                pass

        super().__init__(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server_address[1]}"
        # A short poll: shutdown() waits for the serving loop's next poll, 0.5 s by default.
        threading.Thread(target=lambda: self.serve_forever(poll_interval=0.01), daemon=True).start()


@pytest.fixture
def manager() -> Iterator[FakeManager]:
    server = FakeManager()
    yield server
    server.shutdown()


# ── the environment ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _own_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """No test reads the developer's ~/.menlo or environment, and none writes there."""
    monkeypatch.setenv("MENLO_HOME", str(tmp_path / "asimov-home"))
    for name in tuple(os.environ):
        if name.startswith("MENLO_") and name not in ("MENLO_HOME",) and "SDK_" not in name:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def live_host() -> str:
    """A real robot, named by MENLO_SDK_LIVE_HOST. Skips LOUDLY otherwise."""
    host = os.environ.get("MENLO_SDK_LIVE_HOST")
    if not host:
        pytest.skip("MENLO_SDK_LIVE_HOST not set: no live robot to drive")
    return host


# ── the version endpoint ──────────────────────────────────────────────────────


class FakeVersionEndpoint:
    """Loopback Edge facts endpoint; mutable replies exercise real HTTP validation."""

    def __init__(self) -> None:
        self.body = json.dumps({"robot_model": "asimov_1", "robot_os_version": "0.2.0"}).encode()
        self.status = 200
        self.delay = 0.0
        self.requests: list[str] = []
        self._idle = threading.Event()
        self._idle.set()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                # A timed-out request must finish with its admitted reply, not a retry's facts.
                endpoint._idle.clear()
                body, status, delay = endpoint.body, endpoint.status, endpoint.delay
                endpoint.requests.append(self.path)
                try:
                    time.sleep(delay)
                    with suppress(BrokenPipeError, ConnectionResetError):
                        self.send_response(status)
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("Location", "http://wrong-robot.invalid/api/version")
                        self.end_headers()
                        self.wfile.write(body)
                finally:
                    endpoint._idle.set()

            def log_message(self, *_args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True
        )
        self.thread.start()

    def wait_idle(self, timeout: float = 3.0) -> bool:
        """Wait for an admitted response to finish before reconfiguring a retry."""
        return self._idle.wait(timeout)

    def close(self) -> None:
        if self.thread.is_alive():
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2.0)


@pytest.fixture
def version_endpoint() -> Iterator[FakeVersionEndpoint]:
    """A supported Robot OS facts endpoint for transport integration tests."""
    endpoint = FakeVersionEndpoint()
    yield endpoint
    endpoint.close()


@pytest.fixture(autouse=True)
def version_document_fetch(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    """Legacy tests receive supported facts at the version-document network seam.

    Production parsing and compatibility policy remain active. Dedicated version tests
    restore the real fetcher and exercise its loopback HTTP boundary.
    """
    from menlo.asimov import _version

    real_fetch = _version._fetch_version_document
    if any(request.node.get_closest_marker(m) for m in ("integration", "live", "livekit")):
        return real_fetch

    def supported_target(
        _host: str,
        _port: int,
        _timeout: float,
        _endpoint: str,
    ) -> bytes:
        return b'{"robot_model":"asimov_1","robot_os_version":"0.2.0"}'

    monkeypatch.setattr(_version, "_fetch_version_document", supported_target)
    return real_fetch


# ── the edge over real loopback sockets, for the UDP transport's own tests ────


class LoopbackEdge:
    """Asimov Edge's UDP lane over real loopback sockets, on the real clock.

    Every socket is bound to port 0 and KEPT: the command socket the SDK sends to, and the
    socket the state is pushed from. The SDK binds its own state socket to port 0 too, and
    this edge learns where to push from the transport once it is bound (``follow``), or
    from a port a test read out (``push_to``). Nothing is reserved and released, so two
    parties can never end up on one port.

    Records every decoded command; the state it pushes is whatever the test sets. No mode
    machine: these tests are about the wire, not the firmware.
    """

    def __init__(self, *, state_hz: float = 100.0) -> None:
        from menlo.asimov._proto import load

        pb = load()
        self._cmd_pb, self._common_pb, self._st_pb = pb.command, pb.common, pb.state
        self._version: FakeVersionEndpoint | None = None
        self.received: list = []
        self.state = self._st_pb.RobotState(
            current_mode=self._common_pb.CONTROL_MODE_DAMP, protocol_version=1
        )
        self.state.joint_pos.extend([0.0] * 25)
        self.state.projected_gravity.extend([0.0, 0.0, -1.0])
        self.pushing = True
        self.sent = 0  # state datagrams pushed so far
        self._destination: Callable[[], tuple[str, int] | None] = lambda: None
        self._hz = state_hz
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.settimeout(0.05)
        self.command_port: int = self._sock.getsockname()[1]
        self._out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._out.bind(("127.0.0.1", 0))  # bound and kept: it squats on no port the SDK may bind
        self.push_port: int = self._out.getsockname()[1]
        self._rx = threading.Thread(target=self._recv, daemon=True)
        self._tx = threading.Thread(target=self._push, daemon=True)
        self._rx.start()
        self._tx.start()

    def follow(self, tx: UdpTransport) -> None:
        """Push state to wherever ``tx`` binds its state socket, once it has."""
        self._destination = lambda: tx._state_address

    def follow_robot(self, robot: Robot) -> None:
        """Push state to wherever the UDP transport ``robot`` builds on ``connect()`` binds
        its state socket (the UDP half of a hybrid one included), once it has."""

        def where() -> tuple[str, int] | None:
            tx = robot._transport
            udp = getattr(tx, "_udp", tx)
            address: tuple[str, int] | None = getattr(udp, "_state_address", None)
            return address

        self._destination = where

    def push_to(self, port: int, host: str = "127.0.0.1") -> None:
        self._destination = lambda: (host, port)

    def _recv(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            c = self._cmd_pb.RobotCommand()
            c.ParseFromString(data)
            self.received.append(c)

    def _push(self) -> None:
        period = 1.0 / self._hz
        seq = 0
        while not self._stop.is_set():
            dest = self._destination()
            if self.pushing and dest is not None:
                seq += 1
                self.state.sequence = seq
                self.state.timestamp_us = int(time.time() * 1e6)
                with suppress(OSError):
                    self._out.sendto(self.state.SerializeToString(), dest)
                    self.sent += 1
            time.sleep(period)

    # ── test helpers ─────────────────────────────────────────────────────────
    def set_mode(self, mode: str) -> None:
        self.state.current_mode = getattr(self._common_pb, f"CONTROL_MODE_{mode.upper()}")

    def velocities(self) -> list[tuple[float, float, float]]:
        return [
            (round(c.policy.vx, 4), round(c.policy.vy, 4), round(c.policy.vyaw, 4))
            for c in self.received
            if c.HasField("policy")
        ]

    def modes(self) -> list[str]:
        names = {0: "damp", 1: "stand", 2: "move"}
        return [names[c.mode] for c in self.received if not c.HasField("policy")]

    def wait_for(self, predicate: Callable[[list], bool], timeout: float = 3.0) -> bool:
        """Real time: datagrams arrive on their own schedule."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self.received):
                return True
            time.sleep(0.01)
        return False

    @property
    def version(self) -> FakeVersionEndpoint:
        """Start real HTTP only for tests that explicitly need version preflight."""
        if self._version is None:
            self._version = FakeVersionEndpoint()
        return self._version

    def close(self) -> None:
        if self._version is not None:
            self._version.close()
        self._stop.set()
        self._rx.join(timeout=1.0)
        self._tx.join(timeout=1.0)
        self._sock.close()
        self._out.close()


@pytest.fixture
def loopback_edge() -> Iterator[LoopbackEdge]:
    e = LoopbackEdge()
    yield e
    e.close()


def loopback_transport(edge: LoopbackEdge, **kw: Any) -> UdpTransport:
    """A real ``UdpTransport`` to ``edge``: it binds its state socket to port 0, and the
    edge pushes there once it is bound."""
    kw.setdefault("state_bind", ("127.0.0.1", 0))
    tx = UdpTransport("127.0.0.1", command_port=edge.command_port, **kw)
    edge.follow(tx)
    return tx


def run_script_over(
    edge: LoopbackEdge, body: str, stderr_file: Path, *, timeout: float = 30.0
) -> tuple[int, str, str]:
    """Run ``body`` in a child interpreter connected to ``edge`` on ``udp``, as ``robot``,
    once the robot has reported; its return code, stdout and stderr.

    The child binds its state socket to port 0 and prints the port it got, and the edge
    pushes there from then on: no port is reserved and released. The child then waits for
    the first state, as a ``connect`` that requires state would. Its stderr goes to
    ``stderr_file``, never to a pipe nobody drains, and the wait for the port line is
    bounded by ``timeout``: a child that exits first, or says nothing in time, fails the
    test with its stderr."""
    import select
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        from menlo.asimov import ConnectionConfig, Robot, UdpConfig

        udp = UdpConfig(
            "127.0.0.1",
            version_port={edge.version.port}, command_port={edge.command_port},
            state_bind=("127.0.0.1", 0),
        )
        robot = Robot(ConnectionConfig(udp=udp)).connect("udp", timeout=3.0, require_state=False)
        print("state port", robot._transport._state_address[1], flush=True)
        robot.wait_until(lambda s: True, timeout=5.0)
        """
    ) + textwrap.dedent(body)
    deadline = time.monotonic() + timeout
    line, out = "", ""
    with stderr_file.open("w") as err_out:
        child = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=err_out, text=True
        )
    try:
        assert child.stdout is not None
        while child.poll() is None:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            ready, _, _ = select.select([child.stdout], [], [], left)
            if ready:
                line = child.stdout.readline()
                break
        if line.startswith("state port "):
            edge.push_to(int(line.split()[-1]))
            out, _ = child.communicate(timeout=max(0.0, deadline - time.monotonic()))
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    err = stderr_file.read_text()
    assert line.startswith("state port "), f"the child never reported its state port:\n{err}"
    return child.returncode, out, err


@pytest.fixture
def loopback_robot(loopback_edge: LoopbackEdge) -> Iterator[Robot]:
    """A Robot over the real UDP transport and the loopback edge."""
    r = Robot(loopback_transport(loopback_edge))
    r.open(timeout=3.0)
    yield r
    r.close()
