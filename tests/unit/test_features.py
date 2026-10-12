"""Battery flags, alert names, euler angles, callbacks, recording and set_joints().

Time is the virtual clock's: ``clock.sleep(0.3)`` runs the fake edge, the keepalive and a
``set_joints`` motion for 0.3 s and returns at once.
"""

from __future__ import annotations

import json
import math
import threading

import pytest

from menlo.asimov import (
    Battery,
    BatteryProtection,
    Frame,
    Mode,
    UnsupportedError,
    WaitTimeoutError,
)
from menlo.asimov._media import AudioChunk
from menlo.asimov.recording import load
from tests.conftest import FakeEdge, connect_udp, put_in, route_udp_to


def _pose(value: float) -> list[float]:
    """Every joint at ``value``, the ankles (A, B motors 4, 5, 10, 11) at zero: equal A and B
    is an ankle roll the firmware limits to 0.1 rad, and a set_joints refuses a target past it."""
    pose = [value] * 25
    for i in (4, 5, 10, 11):
        pose[i] = 0.0
    return pose


def test_battery_protection_flags_are_named_and_charging_follows_the_sign():
    b = Battery(48.0, 2.0, 80.0, 30.0, BatteryProtection((1 << 12) | 1))
    assert b.protecting and b.charging
    assert BatteryProtection.CELL_OVERVOLT in b.protection
    assert BatteryProtection.SOFTWARE_MOS_LOCK in b.protection
    assert not Battery(48.0, -1.0, 80.0, 30.0, BatteryProtection(0)).charging


def test_euler_from_quaternion(clock, edge, robot):
    edge.state.base_quat[:] = [1.0, 0.0, 0.0, 0.0]
    clock.sleep(0.05)
    assert robot.get_state().euler == pytest.approx((0.0, 0.0, 0.0))
    yaw = math.pi / 2
    edge.state.base_quat[:] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    clock.sleep(0.05)
    assert robot.get_state().euler[2] == pytest.approx(yaw, abs=1e-6)
    assert robot.get_state().yaw == pytest.approx(yaw, abs=1e-6)
    # a turn is a wrapped difference, so 170° -> -170° reads as +20°, not -340°
    assert math.remainder(math.radians(-170) - math.radians(170), math.tau) == pytest.approx(
        math.radians(20)
    )
    edge.state.base_quat[:] = []
    clock.sleep(0.05)
    assert robot.get_state().yaw is None and robot.get_state().euler is None


def test_alert_carries_first_set_us_and_a_name(clock, edge, robot):
    a = edge.state.active_alerts.add()
    a.id, a.severity, a.first_set_us = 27, 1, 123456
    clock.sleep(0.05)
    alert = robot.get_state().alerts[0]
    assert alert.name == "BMS_LOW_SOC" and alert.first_set_us == 123456 and not alert.critical
    from menlo.asimov import Alert

    assert Alert(14, 0, 0.0, 0.0, 0).name == "INFERENCE_FAILURE"


def test_frame_and_audio_to_numpy():
    np = pytest.importorskip("numpy")
    f = Frame(width=4, height=2, encoding="bgr8", data=bytes(range(24)), stride_bytes=12)
    arr = f.to_numpy()
    assert arr.shape == (2, 4, 3) and arr[1, 3, 2] == 23
    padded = Frame(width=4, height=2, encoding="gray8", data=bytes(range(16)), stride_bytes=8)
    assert padded.to_numpy().shape == (2, 4) and padded.to_numpy()[1, 0] == 8
    with pytest.raises(ValueError):
        Frame(width=1, height=1, encoding="jpeg", data=b"\xff\xd8").to_numpy()
    a = AudioChunk(16_000, 2, 3, "pcm_s16le", bytes(12))
    assert a.to_numpy().shape == (3, 2) and a.to_numpy().dtype == np.int16


def test_require_names_the_first_missing_capability(edge, robot):
    robot.require("drive", "state")
    with pytest.raises(UnsupportedError) as info:
        robot.require("drive", "camera", "speaker")
    assert info.value.capability == "camera"


def test_state_alert_and_mode_callbacks(clock, edge, robot):
    states, alerts, modes = [], [], []
    robot.on_state = states.append
    robot.on_alert = lambda a, ev: alerts.append((a.name, ev))
    robot.on_mode_change = lambda p, c: modes.append((p, c))
    a = edge.state.active_alerts.add()
    a.id, a.severity = 7, 0
    clock.sleep(0.1)
    del edge.state.active_alerts[:]
    edge.set_mode("stand")
    clock.sleep(0.5)  # a cleared alert is carried for ALERT_HOLD_S before it reads as gone
    assert len(states) > 5
    assert alerts == [("FALL_DETECTED", "raised"), ("FALL_DETECTED", "cleared")]
    assert modes == [(Mode.DAMP, Mode.STAND)]


def test_recording_writes_states_and_commands_and_loads_back(clock, edge, robot, tmp_path):
    path = tmp_path / "run.jsonl"
    with robot.record(path) as rec:
        robot.stand(wait=False)  # this fake robot follows no command: it is put in MOVE
        edge.set_mode("move")
        clock.sleep(0.2)
        robot.set_velocity(vx=0.1, wait=False)
        clock.sleep(0.2)
        robot.balance()
    assert rec.samples > 10 and rec.commands_written >= 3
    lines = list(load(path))
    kinds = {line["kind"] for line in lines}
    assert kinds == {"state", "sent"}
    sent = [line for line in lines if line["kind"] == "sent"]
    assert sent[0]["name"] == "stand" and sent[1]["command"]["vx"] == pytest.approx(0.1)
    assert all("mode" in line for line in lines if line["kind"] == "state")
    assert robot.on_state is None, "the recorder restores the previous callback"
    # closed file, valid json on every line
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_goto_clocks_setpoints_from_the_current_pose_then_holds_the_target(clock, edge, robot):
    put_in(robot, edge, "move")
    for i in range(25):
        edge.state.joint_pos[i] = 0.1
    clock.sleep(0.05)
    robot.set_joints(_pose(0.5), duration=0.5, hz=20, wait=False)
    assert edge.wait_for(lambda r: sum(c.HasField("all_trajectory") for c in r) >= 10, timeout=2.0)
    traj = [c.all_trajectory.positions[0] for c in edge.received if c.HasField("all_trajectory")]
    motion = traj[:10]
    assert motion[0] > 0.1 and motion[-1] == pytest.approx(0.5)
    assert motion == sorted(motion), "minimum-jerk from the current pose is monotone"
    assert all(c.mode == 2 for c in edge.received if c.HasField("all_trajectory"))
    # after the motion the target is HELD: more setpoints keep arriving, all at the target
    n = sum(c.HasField("all_trajectory") for c in edge.received)
    clock.sleep(0.35)
    later = [c.all_trajectory.positions[0] for c in edge.received if c.HasField("all_trajectory")]
    assert len(later) >= n + 2 and all(p == pytest.approx(0.5) for p in later[n:])
    robot.damp(wait=False)  # another verb ends the hold
    n = sum(c.HasField("all_trajectory") for c in edge.received)
    clock.sleep(0.3)
    assert sum(c.HasField("all_trajectory") for c in edge.received) <= n + 1


def test_a_velocity_verb_cancels_a_running_set_joints(clock, edge, robot):
    put_in(robot, edge, "move")
    robot.set_joints(_pose(0.5), duration=1.0, hz=20, wait=False)
    clock.sleep(0.15)
    robot.set_velocity(vx=0.1, wait=False)
    n = sum(c.HasField("all_trajectory") for c in edge.received)
    clock.sleep(0.3)
    assert sum(c.HasField("all_trajectory") for c in edge.received) <= n + 1


def test_goto_wait_times_out_when_the_robot_does_not_follow(edge, robot):
    put_in(robot, edge, "move")
    with pytest.raises(WaitTimeoutError):
        robot.set_joints(_pose(0.5), duration=0.2, hz=20, wait=True, timeout=0.5)


def test_alerts_sent_every_20th_frame_are_carried_forward_for_stable_reads(clock, monkeypatch):
    """The firmware includes its alert block in 1 of 20 frames. Per-sample reads must not
    flicker, on_alert must fire once per transition, and a fault must be seen every sample."""
    edge = FakeEdge(clock, state_hz=200.0, alerts_every=20)
    route_udp_to(edge, monkeypatch)
    try:
        with connect_udp(edge, timeout=2) as robot:
            events = []
            robot.on_alert = lambda a, ev: events.append((a.name, ev))
            a = edge.state.active_alerts.add()
            a.id, a.severity = 7, 0
            clock.sleep(0.3)
            reads = [bool(robot.get_state().alerts) for _ in range(40) if not clock.sleep(0.005)]
            assert all(reads), "alerts flickered between frames"
            assert all(robot.get_state().faulted for _ in range(10))
            del edge.state.active_alerts[:]
            clock.sleep(0.6)
            assert robot.get_state().alerts == ()
            assert events == [("FALL_DETECTED", "raised"), ("FALL_DETECTED", "cleared")]
    finally:
        edge.close()


def test_goto_refuses_to_plan_from_a_stale_pose(clock, edge, robot):
    from menlo.asimov import NotReadyError

    put_in(robot, edge, "move")
    robot.link_timeout = 5.0  # long enough that LinkLostError does not fire first
    edge.pushing = False
    clock.sleep(0.7)
    before = len(edge.received)
    with pytest.raises(NotReadyError) as info:
        robot.set_joints(_pose(0.5), duration=0.5, wait=False, timeout=0.2)
    assert info.value.has("stale_state") and info.value.action == "trajectory"
    assert len(edge.received) == before, "nothing is sent from a stale pose"


def test_a_held_goto_stops_when_the_link_is_lost(clock, edge, robot):
    put_in(robot, edge, "move")
    robot.link_timeout = 0.3
    robot.set_joints(_pose(0.5), duration=0.2, hz=20, wait=False)
    clock.sleep(0.3)
    edge.pushing = False
    clock.sleep(0.8)
    assert not robot.connected
    n = sum(c.HasField("all_trajectory") for c in edge.received)
    clock.sleep(0.4)
    assert sum(c.HasField("all_trajectory") for c in edge.received) == n, "hold must stop"


def test_alerts_from_the_previous_session_do_not_haunt_a_reopen(clock, edge, robot):
    a = edge.state.active_alerts.add()
    a.id, a.severity = 7, 0
    clock.sleep(0.1)
    assert robot.get_state().faulted
    robot.close()
    del edge.state.active_alerts[:]
    robot.open(timeout=2.0)
    assert not robot.get_state().faulted, "a carried alert leaked across sessions"
    assert not robot.get_state().alerts
    robot.close()


def test_recording_restores_the_callback_set_after_construction(clock, edge, robot, tmp_path):
    rec = robot.record(tmp_path / "rec-order.jsonl")
    later = []
    robot.on_state = later.append  # set between construction and entering the block
    with rec:
        clock.sleep(0.05)
    assert robot.on_state == later.append
    assert later, "the pre-existing callback kept firing while recording"


def test_goto_captures_its_generation_with_the_first_setpoint(clock, edge, robot):
    """A verb that lands during set_joints()'s first send must cancel the motion. Staged by
    bumping the generation from inside the first trajectory send."""
    put_in(robot, edge, "move")
    real_send = robot._send
    fired = []

    def racing_send(name, command, gen, **kw):
        result = real_send(name, command, gen, **kw)
        if name == "trajectory" and not fired:
            fired.append(1)
            robot.damp(wait=False)  # another verb lands right after the first setpoint left
        return result

    robot._send = racing_send  # type: ignore[method-assign]
    robot.set_joints(_pose(0.5), duration=0.3, hz=20, wait=False)
    clock.sleep(0.5)
    n_traj = sum(c.HasField("all_trajectory") for c in edge.received)
    assert n_traj == 1, f"{n_traj} trajectory setpoints went out after the takeover verb"
    assert "damp" in edge.modes()


@pytest.mark.parametrize(
    "kw", [{"tolerance": float("nan")}, {"timeout": float("inf")}, {"tolerance": -0.1}]
)
def test_goto_validates_its_arguments_before_any_setpoint_leaves(clock, edge, robot, kw):
    with pytest.raises(ValueError):
        robot.set_joints(_pose(0.5), duration=0.3, wait=False, **kw)
    clock.sleep(0.1)
    assert not any(c.HasField("all_trajectory") for c in edge.received)


def test_recording_logs_the_safety_zero_sent_by_close(edge, robot, tmp_path):
    put_in(robot, edge, "move")
    path = tmp_path / "run.jsonl"
    with robot.record(path) as rec:
        robot.set_velocity(vx=0.3, wait=False)
        assert edge.wait_for(lambda r: any(c.HasField("policy") for c in r))
        robot.close()
    assert edge.velocities()[-1] == (0.0, 0.0, 0.0)
    sent = [line for line in load(path) if line["kind"] == "sent"]
    assert sent[-1]["name"] == "set_velocity" and sent[-1]["command"]["vx"] == 0.0
    assert rec.commands_written == len(sent)


def test_recording_logs_the_safety_zero_sent_on_link_loss(clock, edge, robot, tmp_path):
    put_in(robot, edge, "move")
    robot.link_timeout = 0.3
    path = tmp_path / "run.jsonl"
    with robot.record(path):
        robot.set_velocity(vx=0.3, wait=False)
        assert edge.wait_for(lambda r: any(c.HasField("policy") and c.policy.vx > 0 for c in r))
        edge.pushing = False
        clock.sleep(0.8)
        assert not robot.connected
    assert edge.velocities()[-1] == (0.0, 0.0, 0.0)
    sent = [line for line in load(path) if line["kind"] == "sent"]
    assert sent[-1]["name"] == "set_velocity" and sent[-1]["command"]["vx"] == 0.0


def test_damp_from_another_thread_ends_a_running_goto_before_its_next_setpoint(clock, edge, robot):
    """The docs promise set_joints() is fenced: a mode verb from any thread ends it, and no
    setpoint follows the verb. A hand-rolled trajectory loop is the unfenced case. The race
    is not timed here (the motion runs on the virtual clock); test_hardening races a real
    set_joints thread."""
    put_in(robot, edge, "move")
    robot.set_joints(_pose(0.4), duration=5.0, hz=50, wait=False)
    assert edge.wait_for(lambda r: sum(c.HasField("all_trajectory") for c in r) >= 5, timeout=2.0)
    done = threading.Event()

    def other_thread():
        robot.damp(wait=False)
        done.set()

    threading.Thread(target=other_thread).start()
    assert done.wait(1.0)
    clock.sleep(0.3)  # long enough for several 50 Hz setpoints, had the stream continued
    kinds = [("traj" if c.HasField("all_trajectory") else c.mode) for c in edge.received]
    damp_at = max(i for i, k in enumerate(kinds) if k == 0)  # mode 0 = DAMP
    assert "traj" not in kinds[damp_at + 1 :], "a setpoint left after the damp"


@pytest.mark.parametrize(
    ("duration", "hz", "expected"),
    [
        # one step (duration * hz rounds to 1): the target goes out at once and is then held
        # at the hold rate; nothing is sent a period later
        (0.05, 20.0, [0.0, 0.1, 0.2, 0.3]),
        # four steps at 20 Hz: a setpoint every period up to the target, then the hold rate
        (0.2, 20.0, [0.0, 0.05, 0.1, 0.15, 0.25, 0.35]),
    ],
)
def test_set_joints_clocks_the_motion_then_holds_at_the_keepalive_rate(
    clock, edge, robot, duration, hz, expected
):
    """When each setpoint leaves, from the first: ``period`` apart while the motion blends,
    ``1 / KEEPALIVE_HZ`` apart once the target is reached, and for a one-step motion the
    hold rate from the start."""
    put_in(robot, edge, "move")
    started = clock.monotonic()
    robot.set_joints(_pose(0.5), duration=duration, hz=hz, wait=False)
    clock.sleep(expected[-1] + 0.01)
    left_at = [
        round(at - started, 3)
        for c, at in zip(edge.received, edge.arrived, strict=True)
        if c.HasField("all_trajectory")
    ]
    assert left_at[: len(expected)] == pytest.approx(expected)
