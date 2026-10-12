"""Drive Asimov Edge's REAL `UdpConnector`, in-process, with the SDK's transport.

The fake edge in conftest is shaped like the connector; this proves the shape. It needs
Asimov Edge's source on the path (``ASIMOV_EDGE_SRC``), so it is marked ``integration``
and skips (loudly, by name) when that is missing.

No port is reserved and released: the connector binds its command socket to port 0 and
the SDK its state socket to port 0, and each learns the other's port once it is bound
(the connector's ``state_address`` is read on every state it forwards).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

from menlo.asimov import Velocity
from menlo.asimov.transport.udp import UdpTransport

pytestmark = pytest.mark.integration

EDGE_SRC = os.environ.get("ASIMOV_EDGE_SRC")
if not EDGE_SRC:
    pytest.skip("ASIMOV_EDGE_SRC not set: cannot import Asimov Edge", allow_module_level=True)
sys.path.insert(0, EDGE_SRC)


def _connector(topics):
    from edge.connectors.udp_connector import UdpConnector

    return UdpConnector(
        topics,
        command_host="127.0.0.1",
        command_port=0,
        state_host="127.0.0.1",
        state_port=0,
    )


def _transport_to(conn, version_port: int) -> UdpTransport:
    """A transport to the started connector; once open, the connector pushes state to it."""
    assert conn._socket is not None, "the connector is started"
    command_port = conn._socket.getsockname()[1]
    tx = UdpTransport(
        "127.0.0.1",
        command_port=command_port,
        state_bind=("127.0.0.1", 0),
        version_port=version_port,
    )
    tx.open()
    conn.state_address = tx._state_address
    return tx


def test_the_edge_admits_our_velocity_as_a_udp_source(version_endpoint):
    from edge.control_source import ControlSource
    from edge.topic_registry import TopicRegistry
    from edge.topics import ROBOT_COMMAND_TOPIC

    topics = TopicRegistry()
    sub = topics.subscribe(ROBOT_COMMAND_TOPIC)
    conn = _connector(topics)

    async def main():
        await conn.start()
        tx = _transport_to(conn, version_endpoint.port)
        seq = tx.send(Velocity(vx=0.42))
        got = None
        for _ in range(100):
            try:
                got = sub.get_nowait()
                break
            except Exception:
                await asyncio.sleep(0.02)
        tx.close()
        await conn.stop()
        return seq, got

    seq, got = asyncio.run(main())
    assert got is not None, "the connector published nothing to the arbiter's command topic"
    assert got.source is ControlSource.UDP
    assert got.command.policy.vx == pytest.approx(0.42)
    assert got.command.sequence == seq
    assert conn.connected is False  # stopped


def test_the_edge_forwards_robot_state_to_our_bound_port(version_endpoint):
    from menlo.asimov._proto import load

    st_pb = load().state
    from edge.messages import RobotStateMessage
    from edge.topic_registry import TopicRegistry
    from edge.topics import ROBOT_STATE_TOPIC

    topics = TopicRegistry()
    conn = _connector(topics)
    seen = []

    async def main():
        await conn.start()
        tx = _transport_to(conn, version_endpoint.port)
        tx.subscribe_state(seen.append)
        msg = st_pb.RobotState(current_mode=1, protocol_version=1)
        msg.joint_pos.extend([0.5] * 25)
        topics.publish(ROBOT_STATE_TOPIC, RobotStateMessage(msg))
        deadline = time.monotonic() + 2
        while not seen and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        tx.close()
        await conn.stop()

    asyncio.run(main())
    assert seen, "no RobotState reached the SDK's state port"
    # UdpTransport has no RobotInfo handshake, so this low-level transport state
    # intentionally carries positional joints without robot-specific names.
    assert seen[0].mode.name == "STAND"
    assert seen[0].joints[3].pos == pytest.approx(0.5)
