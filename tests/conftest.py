"""Shared fixtures: a fake edge that speaks the real wire, and a live-rig gate.

Every socket a test opens is bound to port 0 and kept: the kernel picks a free port and
the socket owns it from then on. No port number is ever reserved and released, so two
parties cannot end up on one port. The fake edge binds one socket, reports its port
(``command_port``), and pushes state from that same socket; the SDK binds its state
socket to port 0 and the transport seams below tell the edge where it landed
(``push_to``) once it is bound, before ``Robot.open`` starts waiting for state.
"""

from __future__ import annotations

import json
import math
import os
import socket
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from menlo.asimov import Applied, ConnectError, LinkLostError, Refused, Robot
from menlo.asimov.transport._livekit_client import identity_from_token
from menlo.asimov.transport.livekit import HybridTransport, LiveKitTransport
from menlo.asimov.transport.udp import UdpTransport

#: ``CONTROL_MODE_FAULT_DAMP``, by value: asimov.io protocol v1.3.0 adds it, and the SDK
#: reads robot modes by value, so it works whatever version of the bindings is installed.
FAULT_DAMP = 5


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
    """The edge's UDP lane, as the SDK sees it: bare ``RobotCommand`` in, bare
    ``RobotState`` pushed out to a configured host at a configured rate.

    Deliberately shaped like Asimov Edge's ``UdpConnector`` (sender address ignored,
    state to ONE destination, one socket for both), so a test that passes here says
    something about the robot. Records every decoded command; the state it pushes is
    whatever the test sets. The destination is learned from the SDK side once it has
    bound its state socket (``push_to``); until then nothing is pushed.

    ``firmware=True`` makes the robot mode follow the commands the way the Motion Control
    Board firmware does: STAND and DAMP take effect at once; MOVE (a velocity or a
    trajectory) is entered only from STAND once the robot has been upright (gravity z below
    -0.87) for ``arm_hold_s``, and until then the robot stays in STAND; MOVE -> STAND stays
    armed; a velocity or trajectory while DAMPed is dropped (the edge's DAMP gate); a
    trajectory in MOVE sets the reported joint positions to its targets (after the
    firmware's ankle coupling); and a latched
    fault (``fault()``, or ``fault_damp()`` for robot mode FAULT_DAMP) holds the robot in DAMP
    and ignores STAND and MOVE.
    """

    def __init__(
        self,
        *,
        state_hz: float = 100.0,
        alerts_every: int = 1,
        firmware: bool = False,
        arm_hold_s: float = 0.5,
    ) -> None:
        from menlo.asimov._proto import load

        pb = load()  # same bindings the SDK uses, whichever source it resolved to
        self._cmd_pb, self._common_pb, self._st_pb = pb.command, pb.common, pb.state
        self._version: FakeVersionEndpoint | None = None
        self.received: list = []  # decoded RobotCommand protos, in arrival order
        self.arrived: list[float] = []  # monotonic receipt time of each entry in received
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
        self.armed_at: float | None = None  # monotonic time the fake firmware armed
        self.move_entered_at: float | None = None
        self.first_velocity_at: float | None = None  # monotonic receipt of the first velocity
        self._upright_since: float | None = None
        self._fw_lock = threading.Lock()
        self._hz = state_hz
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))  # the kernel picks the port; the socket keeps it
        self._sock.settimeout(0.05)
        self.command_port: int = self._sock.getsockname()[1]
        #: Where state is pushed: the SDK's bound state socket, once it has said where.
        self.destination: tuple[str, int] | None = None
        self._rx = threading.Thread(target=self._recv, daemon=True)
        self._tx = threading.Thread(target=self._push, daemon=True)
        self._rx.start()
        self._tx.start()

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
            if c.HasField("policy") and self.first_velocity_at is None:
                self.first_velocity_at = time.monotonic()
            self.arrived.append(time.monotonic())
            self.received.append(c)
            if self.firmware:
                self._follow(c)

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
            self.move_entered_at = time.monotonic()
        self.state.current_mode = mode

    def _follow(self, c) -> None:
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
            now = time.monotonic()
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

    def _push(self) -> None:
        period = 1.0 / self._hz
        seq = 0
        while not self._stop.is_set():
            if self.firmware:
                self._arm_tick()
            dest = self.destination
            if self.pushing and dest is not None:
                seq += 1
                self.state.sequence = seq
                self.state.timestamp_us = int(time.time() * 1e6)
                if self.alerts_every > 1 and seq % self.alerts_every:
                    frame = self._st_pb.RobotState()
                    frame.CopyFrom(self.state)
                    del frame.active_alerts[:]
                    payload = frame.SerializeToString()
                else:
                    payload = self.state.SerializeToString()
                with suppress(OSError):  # the destination may have closed between checks
                    self._sock.sendto(payload, dest)  # from the one bound socket, as the edge
            time.sleep(period)

    def push_to(self, address: tuple[str, int]) -> None:
        """State goes to ``address`` from now on: the SDK's state socket, as it was bound."""
        self.destination = (address[0], int(address[1]))

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
        """The velocities received from monotonic time ``start`` until ``end``."""
        return [
            (round(c.policy.vx, 4), round(c.policy.vy, 4), round(c.policy.vyaw, 4))
            for c, at in zip(tuple(self.received), tuple(self.arrived), strict=False)
            if start <= at < end and c.HasField("policy")
        ]

    def modes(self) -> list[str]:
        names = {0: "damp", 1: "stand", 2: "move"}
        return [names[c.mode] for c in self.received if not c.HasField("policy")]

    def wait_for(self, predicate: Callable[[list], bool], timeout: float = 3.0) -> bool:
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


def connect_udp(
    host: str,
    *,
    command_port: int = 8850,
    state_bind: tuple[str, int] = ("127.0.0.1", 0),
    state_source: str | None = None,
    limits=None,
    link_timeout: float | None = None,
    **connect_kw,
) -> Robot:
    """``Robot(ConnectionConfig(udp=...)).connect("udp", ...)`` in one call, for tests that
    only care about the UDP lane. The ``udp`` lane the config builds is a
    :class:`SeamUdpTransport` when an ``edge`` fixture (or ``route_udp_to``) is in force,
    so the edge learns the state port the SDK binds."""
    from menlo.asimov.connection import ConnectionConfig, UdpConfig

    cfg = ConnectionConfig(
        udp=UdpConfig(
            host, command_port=command_port, state_bind=state_bind, state_source=state_source
        )
    )
    robot_kw = {}
    if limits is not None:
        robot_kw["limits"] = limits
    if link_timeout is not None:
        robot_kw["link_timeout"] = link_timeout
    return Robot(cfg, **robot_kw).connect("udp", **connect_kw)


def put_in(robot: Robot, edge: FakeEdge, mode: str) -> None:
    """Make the fake robot report robot mode ``mode`` and wait until ``robot`` has seen it:
    a motion command checks the robot mode before it sends."""
    edge.set_mode(mode)
    robot.wait_until(lambda s: s.mode.name == mode.upper(), timeout=2.0)


class SeamUdpTransport(UdpTransport):
    """The real transport plus two test-only doors: once ``open()`` has bound the state
    socket it tells the fake edge where (so the edge pushes there, and ``Robot.open`` finds
    state when it starts waiting), and ``deliver_outcome`` hands in a verdict as if the
    edge had sent one. Built to ``edge``: its command port, and state port 0."""

    def __init__(self, edge: FakeEdge, host: str = "127.0.0.1", **kw) -> None:
        kw.setdefault("command_port", edge.command_port)
        kw.setdefault("state_bind", ("127.0.0.1", 0))
        super().__init__(host, **kw)
        self._edge = edge

    def open(self) -> None:
        super().open()
        assert self._sock is not None
        self._edge.push_to(self._sock.getsockname())

    def deliver_outcome(self, outcome: Applied | Refused) -> None:
        for cb in tuple(self._on_outcome):
            cb(outcome)


class SeamHybridTransport(HybridTransport):
    """``HybridTransport`` whose UDP half reports its bound state port to the fake edge."""

    def __init__(self, edge: FakeEdge, host: str = "127.0.0.1", **kw) -> None:
        kw.setdefault("command_port", edge.command_port)
        kw.setdefault("state_bind", ("127.0.0.1", 0))
        kw.setdefault("livekit_url", "ws://fake")
        kw.setdefault("room", "asimov-room")
        kw.setdefault("token", "test-token")
        super().__init__(host, **kw)
        self._edge = edge

    def open(self) -> None:
        super().open()
        assert self._udp._sock is not None
        self._edge.push_to(self._udp._sock.getsockname())


def route_udp_to(edge: FakeEdge, monkeypatch) -> None:
    """Every ``udp`` and ``hybrid`` lane a :class:`ConnectionConfig` builds is the seam
    transport over ``edge``: the config's ports still apply (tests name the edge's command
    port and state port 0), and the edge learns the state port once it is bound."""
    from menlo.asimov import connection

    monkeypatch.setattr(
        connection, "UdpTransport", lambda host, **kw: SeamUdpTransport(edge, host, **kw)
    )
    monkeypatch.setattr(
        connection, "HybridTransport", lambda host, **kw: SeamHybridTransport(edge, host, **kw)
    )


@pytest.fixture
def edge(monkeypatch):
    e = FakeEdge()
    route_udp_to(e, monkeypatch)
    yield e
    e.close()


@pytest.fixture
def robot(edge):
    r = Robot(SeamUdpTransport(edge, version_port=edge.version.port))
    r.open(timeout=3.0)
    yield r
    r.close()


class FakeLiveKitClient:
    """The ``LiveKitClient`` seam with the room replaced by loopback UDP to the FakeEdge.

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
        self._sock: socket.socket | None = None  # the state track: bound to port 0, kept
        self._out: socket.socket | None = None  # commands leave from here: bound to port 0, kept
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()

    # ── the seam ─────────────────────────────────────────────────────────────
    def connect(self) -> None:
        if self.connected:
            raise RuntimeError("already connected")
        self._out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._out.bind(("127.0.0.1", 0))
        if self._carry_state:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", 0))
            sock.settimeout(0.1)
            self._sock = sock
            self._edge.push_to(sock.getsockname())  # the edge pushes here from now on
            self._stop.clear()
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
        self.identity = identity_from_token(self._token)
        self.connected = True
        self._set_tracks(self._room_tracks)

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._out is not None:
            self._out.close()
            self._out = None
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
        self._send(payload)

    def push_data_frame(self, name: str, payload: bytes) -> bool:
        if not self.connected:
            raise LinkLostError("the fake room is not joined")
        if name not in self._local_tracks:
            if not self._refuse_track:
                self._local_tracks.add(name)
            return False
        self.frames.append((name, payload))
        self._send(payload)
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

    def publish_audio(self, chunk) -> None:
        if not self.connected:
            raise LinkLostError("the fake room is not joined")
        self.played.append(chunk)

    def on_data_track(self, name: str, callback) -> None:
        self._data_track_cbs.setdefault(name, []).append(callback)

    def on_video(self, callback) -> None:
        self._video_cbs.append(callback)

    def on_audio(self, callback) -> None:
        self._audio_cbs.append(callback)

    def on_tracks(self, callback) -> None:
        self._track_cbs.append(callback)

    # ── plumbing + test helpers ──────────────────────────────────────────────
    def _send(self, payload: bytes) -> None:
        assert self._out is not None
        self._out.sendto(payload, ("127.0.0.1", self._edge.command_port))

    def _read(self) -> None:
        while not self._stop.is_set():
            sock = self._sock
            if sock is None:
                return
            try:
                data, _ = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            # The FakeEdge speaks UDP, which carries no edge clock: user_timestamp=None.
            for cb in tuple(self._data_track_cbs.get("state", ())):
                cb(data, None)

    def _set_tracks(self, tracks: frozenset[str]) -> None:
        if tracks == self.tracks:
            return
        self.tracks = tracks
        for cb in tuple(self._track_cbs):
            cb(tracks)

    def push_frame(self, n: int):
        from menlo.asimov import Frame

        f = Frame(width=4, height=2, encoding="rgb8", data=bytes(24), stride_bytes=12, sequence=n)
        for cb in tuple(self._video_cbs):
            cb(f)
        return f

    def push_audio(self, n: int, *, samples: int = 160):
        from menlo.asimov import AudioChunk

        a = AudioChunk(16_000, 1, samples, "pcm_s16le", bytes(2 * samples), sequence=n)
        for cb in tuple(self._audio_cbs):
            cb(a)
        return a


def make_livekit_robot(edge: FakeEdge, **kw) -> tuple[FakeLiveKitClient, Robot]:
    """Mode B: everything over the (faked) room."""
    client = FakeLiveKitClient(edge, **kw)
    tx = LiveKitTransport("ws://fake", "asimov-room", token="test-token", client=client)
    return client, Robot(tx)


def make_hybrid_robot(edge: FakeEdge, **kw) -> tuple[FakeLiveKitClient, Robot]:
    """Mode A: UDP control, (faked) room media."""
    client = FakeLiveKitClient(edge, carry_state=False, **kw)
    tx = SeamHybridTransport(edge, version_port=edge.version.port, client=client)
    return client, Robot(tx)


@pytest.fixture
def livekit_robot(edge):
    client, r = make_livekit_robot(edge)
    r.open(timeout=3.0)
    yield client, r
    r.close()


@pytest.fixture
def hybrid_robot(edge):
    client, r = make_hybrid_robot(edge)
    r.open(timeout=3.0)
    yield client, r
    r.close()


@pytest.fixture(params=["udp", "hybrid", "livekit"])
def any_robot(request, edge):
    """The same robot over each of the three lanes. A behaviour that is not mode-agnostic
    is not done: every promise below is asserted three times."""
    if request.param == "udp":
        r: Robot = Robot(SeamUdpTransport(edge, version_port=edge.version.port))
    else:
        maker = make_hybrid_robot if request.param == "hybrid" else make_livekit_robot
        _client, r = maker(edge)
    r.open(timeout=3.0)
    yield request.param, r
    r.close()


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
        threading.Thread(target=self.serve_forever, daemon=True).start()


@pytest.fixture
def manager():
    server = FakeManager()
    yield server
    server.shutdown()


def route_manager_rooms_to(edge: FakeEdge, monkeypatch) -> list[str]:
    """Make every ``LiveKitTransport`` a ``ManagerConfig`` builds join the FakeEdge through
    the fake client instead of a real room. Returns the list the tokens each join presented
    are appended to, so a test can see which mint a join used."""
    from menlo.asimov import connection

    seen_tokens: list[str] = []
    real = connection.LiveKitTransport

    class Capturing(real):  # type: ignore[misc,valid-type]
        def __init__(self, url, room, *, token, **kw):
            client = FakeLiveKitClient(edge, token=token() if callable(token) else token)
            seen_tokens.append(client._token)
            super().__init__(url, room, token=token, client=client, **kw)

    monkeypatch.setattr(connection, "LiveKitTransport", Capturing)
    return seen_tokens


@pytest.fixture(autouse=True)
def _own_environment(monkeypatch, tmp_path):
    """No test reads the developer's ~/.menlo or environment, and none writes there."""
    monkeypatch.setenv("MENLO_HOME", str(tmp_path / "asimov-home"))
    for name in tuple(os.environ):
        if name.startswith("MENLO_") and name not in ("MENLO_HOME",) and "SDK_" not in name:
            monkeypatch.delenv(name, raising=False)


def run_script_over(
    edge: FakeEdge, body: str, stderr_file: Path, *, timeout: float = 30.0
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
        print("state port", robot._transport._sock.getsockname()[1], flush=True)
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
            edge.push_to(("127.0.0.1", int(line.split()[-1])))
            out, _ = child.communicate(timeout=max(0.0, deadline - time.monotonic()))
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    err = stderr_file.read_text()
    assert line.startswith("state port "), f"the child never reported its state port:\n{err}"
    return child.returncode, out, err


@pytest.fixture
def live_host() -> str:
    """A real robot, named by MENLO_SDK_LIVE_HOST. Skips LOUDLY otherwise."""
    host = os.environ.get("MENLO_SDK_LIVE_HOST")
    if not host:
        pytest.skip("MENLO_SDK_LIVE_HOST not set: no live robot to drive")
    return host


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
def version_endpoint():
    """A supported Robot OS facts endpoint for transport integration tests."""
    endpoint = FakeVersionEndpoint()
    yield endpoint
    endpoint.close()


@pytest.fixture(autouse=True)
def version_document_fetch(monkeypatch, request):
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
