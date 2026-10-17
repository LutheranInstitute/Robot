"""Tests for the æ face bridge. No robot, no menlo-sdk, no network beyond localhost.

Run: python3 ae/test_face.py        (or: python3 -m unittest ae/test_face.py -v)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("AE_FACE_OUT", os.path.join(tempfile.mkdtemp(prefix="ae-face-"), "face.json"))

import face as ae  # noqa: E402

# talk to localhost directly, never through a sandbox or corporate proxy
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

S = ae.sample


class MoodPriority(unittest.TestCase):
    """hold > lost > tired > sleep > think > ready > work, checked top to bottom."""

    def test_each_mood(self):
        cases = [
            (S("FAULT_DAMP", faulted=True), None, "hold"),
            (S("MOVE", age=0.6), True, "lost"),
            (S("MOVE", soc=19), True, "tired"),
            (S("DAMP"), None, "sleep"),
            (S("STAND"), False, "think"),
            (S("STAND"), None, "think"),
            (S("STAND"), True, "ready"),
            (S("MOVE"), True, "work"),
        ]
        for state, armed, want in cases:
            with self.subTest(want=want, mode=state.mode.name):
                self.assertEqual(ae.mood(state, armed), want)

    def test_hold_beats_everything(self):
        self.assertEqual(ae.mood(S("MOVE", age=5, soc=1, faulted=True), True), "hold")
        self.assertEqual(ae.mood(S("FAULT_DAMP", age=5, soc=1), None), "hold")

    def test_lost_beats_tired_and_modes(self):
        self.assertEqual(ae.mood(S("STAND", age=0.51, soc=5), True), "lost")
        self.assertEqual(ae.mood(S("DAMP", age=0.51), None), "lost")

    def test_stale_boundary(self):
        self.assertEqual(ae.mood(S("MOVE", age=ae.STALE_S), True), "work")

    def test_tired_beats_modes(self):
        for mode, armed in [("DAMP", None), ("STAND", False), ("STAND", True), ("MOVE", True)]:
            with self.subTest(mode=mode):
                self.assertEqual(ae.mood(S(mode, soc=ae.LOW_BATTERY - 1), armed), "tired")

    def test_battery_boundary_and_unknown(self):
        self.assertEqual(ae.mood(S("MOVE", soc=ae.LOW_BATTERY), True), "work")
        self.assertEqual(ae.mood(S("MOVE", soc=None), True), "work")

    def test_unknown_mode_is_lost(self):
        self.assertEqual(ae.mood(S("BOOT"), None), "lost")

    def test_demo_covers_all_seven(self):
        got = [ae.mood(s, a) for s, a in ae.demo_steps()]
        self.assertEqual(got, ["sleep", "think", "ready", "work", "tired", "lost", "hold"])
        self.assertEqual(set(got), set(ae.MOODS))


class FaceSchema(unittest.TestCase):
    def test_fields_ts_seq(self):
        a = ae.publish(ae.face(S("MOVE", alerts=["IMU_WARN"]), True))
        b = ae.publish(ae.face(S("DAMP"), None))
        for f in (a, b):
            self.assertEqual(f["schema"], "ae.face/0.1")
            self.assertEqual(f["authority"], "æ1")
            self.assertIsInstance(f["ts"], float)
            self.assertLess(abs(f["ts"] - time.time()), 5)
            self.assertIsInstance(f["seq"], int)
        self.assertEqual(b["seq"], a["seq"] + 1)
        self.assertEqual(a["alerts"], ["IMU_WARN"])
        with open(ae.OUT, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["seq"], b["seq"])  # face.json holds the latest


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServeDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.tmp = tempfile.mkdtemp(prefix="ae-serve-")
        env = dict(os.environ, AE_FACE_OUT=os.path.join(cls.tmp, "face.json"))
        cls.proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "serve.py"), "--demo", "--host", "127.0.0.1",
             "--port", str(cls.port), "--step", "0.2"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=cls.tmp)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(cls.base + "/face.json", timeout=1) as r:
                    if r.status == 200:
                        return
            except OSError:
                time.sleep(0.1)
        cls.proc.kill()
        raise RuntimeError("serve.py --demo did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(5)

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=3) as r:
            return r.status, dict(r.headers), r.read()

    def test_face_json(self):
        code, headers, body = self.get("/face.json")
        self.assertEqual(code, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        f = json.loads(body)
        self.assertEqual(f["schema"], "ae.face/0.1")
        self.assertIn(f["mood"], ae.MOODS)
        self.assertIn("ts", f)
        self.assertGreaterEqual(f["seq"], 1)

    def test_health(self):
        code, headers, body = self.get("/health")
        self.assertEqual(code, 200)
        h = json.loads(body)
        self.assertTrue(h["ok"])
        self.assertTrue(h["read_only"])
        self.assertEqual(h["source"], "demo")
        self.assertEqual(h["authority"], "æ1")

    def test_moods_advance(self):
        seen = set()
        deadline = time.time() + 5
        while time.time() < deadline and len(seen) < 7:
            seen.add(json.loads(self.get("/face.json")[2])["mood"])
            time.sleep(0.05)
        self.assertEqual(seen, set(ae.MOODS))

    def test_events_push(self):
        with urllib.request.urlopen(self.base + "/events", timeout=5) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("text/event-stream"))
            faces = []
            while len(faces) < 2:
                line = r.readline().decode()
                if line.startswith("data: "):
                    faces.append(json.loads(line[6:]))
        self.assertGreater(faces[1]["seq"], faces[0]["seq"])
        self.assertNotEqual(faces[1]["mood"], faces[0]["mood"])

    def test_kiosk_page(self):
        code, headers, body = self.get("/")
        self.assertEqual(code, 200)
        page = body.decode()
        self.assertTrue("aeee-lab/lab/face/" in page or os.path.isfile(os.path.join(HERE, "face.html")))
        self.assertIn(b"EventSource", self.get("/local")[2])

    def test_read_only(self):
        req = urllib.request.Request(self.base + "/face.json", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=3)
        self.assertEqual(e.exception.code, 501)  # no write route exists


if __name__ == "__main__":
    unittest.main(verbosity=2)
