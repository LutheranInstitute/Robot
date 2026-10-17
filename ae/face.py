"""æ face bridge for Asimov 1: the robot's reported state becomes a face mood.

Reads state only. Sends nothing to the robot, so it is safe in any robot mode.
Each mood change is printed as one æ.face/0.1 JSON line and written to face.json,
which any screen (a phone in the head, a display, a browser) can read and draw.

Evidence is not authority: the face reports what the robot says about itself.
It never commands the robot.

Run: python ae/face.py            (saved robot, or the MENLO_* environment variables)
     python ae/face.py --demo     (no robot: steps through every mood)
     python ae/serve.py           (same, plus face.json / SSE / kiosk over HTTP)
"""

from __future__ import annotations

import json
import os
import sys
import itertools
import tempfile
import threading
import time
from datetime import datetime, timezone

SCHEMA = "ae.face/0.1"
WHO = "asimov"
AUTHORITY = "æ1"  # a robot publishes its state; it holds no operator authority
STALE_S = 0.5  # the SDK's own freshness limit for decisions to move
LOW_BATTERY = 20  # percent; the SDK's own examples treat below 20 % as unsafe
PERIOD_S = 0.2
OUT = os.environ.get("AE_FACE_OUT", "face.json")

# mood -> what the face means, in the order the rules are checked
MOODS = {
    "hold": "fault latched: robot stays damped until firmware restart",
    "lost": "no fresh state: the face cannot vouch for the robot",
    "tired": "battery low",
    "sleep": "DAMP: no actuator holds a position",
    "think": "STAND, not armed yet",
    "ready": "STAND, armed: MOVE is allowed",
    "work": "MOVE: the walking policy balances the robot",
}


def mood(state, armed: bool | None) -> str:
    """One state sample to one mood. Pure: no robot needed to test it."""
    mode = state.mode.name
    if state.faulted or mode == "FAULT_DAMP":
        return "hold"
    if state.age_s > STALE_S:
        return "lost"
    if state.battery is not None and state.battery.soc_percent < LOW_BATTERY:
        return "tired"
    if mode == "DAMP":
        return "sleep"
    if mode == "STAND":
        return "ready" if armed else "think"
    if mode == "MOVE":
        return "work"
    return "lost"


def face(state, armed: bool | None) -> dict:
    m = mood(state, armed)
    return {
        "schema": SCHEMA,
        "who": WHO,
        "authority": AUTHORITY,
        "mood": m,
        "means": MOODS[m],
        "mode": state.mode.name,
        "armed": armed,
        "battery": None if state.battery is None else round(state.battery.soc_percent),
        "alerts": [a.name for a in state.alerts],
        "age_ms": round(state.age_s * 1000),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ts": round(time.time(), 3),  # unix seconds; added in 0.1, old readers ignore it
    }


_seq = itertools.count(1)
_seq_lock = threading.Lock()


def publish(f: dict) -> dict:
    """Stamp seq (1, 2, 3 ... per process), print one line, write face.json atomically."""
    with _seq_lock:
        f["seq"] = next(_seq)
    f.setdefault("ts", round(time.time(), 3))
    print(json.dumps(f, ensure_ascii=False), flush=True)
    d = os.path.dirname(os.path.abspath(OUT))
    with tempfile.NamedTemporaryFile("w", dir=d, delete=False, encoding="utf-8") as tmp:
        json.dump(f, tmp, ensure_ascii=False, indent=2)
    os.replace(tmp.name, OUT)  # atomic: a screen never reads half a face
    return f


def sample(mode, age=0.05, soc=80, faulted=False, alerts=()):
    """A fake state sample shaped like the SDK's, for the demo and the tests."""
    from types import SimpleNamespace as N

    battery = None if soc is None else N(soc_percent=soc)
    return N(mode=N(name=mode), age_s=age, battery=battery, faulted=faulted,
             alerts=[N(name=a) for a in alerts])


def demo_steps():
    """One (state, armed) per mood, in the order sleep think ready work tired lost hold."""
    return [
        (sample("DAMP"), None), (sample("STAND"), False), (sample("STAND"), True),
        (sample("MOVE"), True), (sample("MOVE", soc=15), True), (sample("MOVE", age=1.2), True),
        (sample("FAULT_DAMP", faulted=True), None),
    ]


def demo(emit=publish, loop: bool = False, step_s: float = 0.3) -> None:
    while True:
        for state, armed in demo_steps():
            emit(face(state, armed))
            time.sleep(step_s)
        if not loop:
            return


def main(emit=publish) -> None:
    """Watch the robot and emit a face on every mood change. Reads only."""
    from menlo.asimov import Robot

    last = None
    with Robot().connect() as robot:
        while True:
            f = face(robot.get_state(), robot.armed)
            if f["mood"] != last:
                emit(f)
                last = f["mood"]
            time.sleep(PERIOD_S)


if __name__ == "__main__":
    try:
        demo() if "--demo" in sys.argv else main()
    except KeyboardInterrupt:
        pass
