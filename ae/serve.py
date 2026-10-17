"""Serve the æ face of Asimov 1 over HTTP. Read-only, standard library only.

    GET /           kiosk page: ae/face.html if present, else the æ lab face in a frame
    GET /local      a tiny built-in face drawn from /events (same origin, works offline)
    GET /face.json  latest ae.face/0.1 face (CORS: *)
    GET /events     Server-Sent Events, one "face" event per mood change
    GET /health     bridge status

The bridge only reads robot state. It has no route, method or code path that sends a
command to the robot: there is no POST, no arm, no mode change. Authority æ1.
Human stays root.

Run: python3 ae/serve.py --demo              (no robot)
     python3 ae/serve.py                     (saved robot or MENLO_* variables)
     python3 ae/serve.py --port 8765 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import face as ae  # noqa: E402

LAB_FACE = "https://lutheraninstitute.github.io/aeee-lab/lab/face/"
KEEPALIVE_S = 15.0


class Hub:
    """Latest face plus a condition that wakes every SSE client on a change."""

    def __init__(self, source: str):
        self.source = source
        self.face: dict | None = None
        self.updated = 0.0
        self.started = time.time()
        self.clients = 0
        self.error: str | None = None
        self.cond = threading.Condition()

    def update(self, f: dict) -> None:
        with self.cond:
            self.face = f
            self.updated = time.time()
            self.cond.notify_all()

    def snapshot(self) -> dict | None:
        with self.cond:
            return None if self.face is None else dict(self.face)

    def wait_newer(self, seq: int, timeout: float) -> dict | None:
        """Block until a face with a higher seq exists, or timeout. Returns it or None."""
        with self.cond:
            self.cond.wait_for(lambda: self.face is not None and self.face.get("seq", 0) > seq, timeout)
            if self.face is not None and self.face.get("seq", 0) > seq:
                return dict(self.face)
            return None


def lost_face(reason: str) -> dict:
    """What the face says when the bridge itself cannot read the robot."""
    f = ae.face(ae.sample("UNKNOWN", age=1e9, soc=None), None)
    f["age_ms"] = None
    f["means"] = f"{ae.MOODS['lost']} ({reason})"
    return f


def run_source(hub: Hub, demo: bool, step_s: float) -> None:
    emit = lambda f: hub.update(ae.publish(f))  # noqa: E731
    if demo:
        ae.demo(emit=emit, loop=True, step_s=step_s)
        return
    while True:  # robot: reconnect forever; a broken link reads as "lost", never as "ok"
        try:
            hub.error = None
            ae.main(emit=emit)
        except Exception as e:  # noqa: BLE001 - any failure means no fresh state
            hub.error = f"{type(e).__name__}: {e}"
            cur = hub.snapshot()
            if cur is None or cur.get("mood") != "lost":
                emit(lost_face(hub.error))
            time.sleep(2.0)


KIOSK = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>æ æ æ · Asimov 1</title>
<style>html,body{{margin:0;height:100%;background:#f4efe3}}iframe{{border:0;width:100%;height:100%}}
a{{position:fixed;right:8px;bottom:6px;font:12px monospace;color:#0008}}</style></head>
<body><iframe src="{src}" allow="fullscreen"></iframe><a href="/local">local face</a></body></html>"""

LOCAL = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>æ · local face</title>
<style>
html,body{margin:0;height:100%;background:#f4efe3;color:#111;font-family:ui-monospace,monospace}
main{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4vh}
#f{font-size:22vmin;letter-spacing:.08em;font-weight:700}
#m{font-size:8vmin;text-transform:uppercase;letter-spacing:.2em}
#w{font-size:3.4vmin;opacity:.7;max-width:90vw;text-align:center}
.hold #f,.lost #f{color:#b00}.tired #f{color:#a60}.work #f{color:#06a}.ready #f{color:#070}
</style></head><body><main><div id="f">æ æ æ</div><div id="m">…</div><div id="w">connecting</div></main>
<script>
const EYES={hold:"æ ✕ æ",lost:"æ ? æ",tired:"æ _ æ",sleep:"– æ –",think:"æ … æ",ready:"æ æ æ",work:"æ ▸ æ"};
function draw(f){document.body.className=f.mood;F.textContent=EYES[f.mood]||"æ æ æ";
M.textContent=f.mood;W.textContent=f.means+(f.battery!=null?" · "+f.battery+"%":"")+" · æ1 · human stays root";}
const F=document.getElementById("f"),M=document.getElementById("m"),W=document.getElementById("w");
fetch("/face.json").then(r=>r.ok&&r.json()).then(f=>f&&draw(f)).catch(()=>{});
const es=new EventSource("/events");es.addEventListener("face",e=>draw(JSON.parse(e.data)));
es.onerror=()=>{M.textContent="lost";W.textContent="bridge unreachable";document.body.className="lost";F.textContent=EYES.lost};
</script></body></html>"""


def make_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ae-face/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep stdout for face lines
            sys.stderr.write("ae.serve %s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_OPTIONS(self):  # CORS preflight; still read-only
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/face.json":
                f = hub.snapshot()
                return self._json(200, f) if f else self._json(503, {"schema": ae.SCHEMA, "error": "no face yet"})
            if path == "/health":
                f = hub.snapshot()
                return self._json(200, {
                    "ok": f is not None and hub.error is None,
                    "schema": ae.SCHEMA,
                    "authority": ae.AUTHORITY,
                    "read_only": True,
                    "source": hub.source,
                    "mood": f and f.get("mood"),
                    "seq": f and f.get("seq"),
                    "since_change_s": round(time.time() - hub.updated, 3) if f else None,
                    "uptime_s": round(time.time() - hub.started, 3),
                    "clients": hub.clients,
                    "error": hub.error,
                })
            if path == "/events":
                return self._events()
            if path in ("/", "/index.html"):
                local = os.path.join(HERE, "face.html")
                if os.path.isfile(local):
                    with open(local, "rb") as fh:
                        return self._send(200, fh.read(), "text/html; charset=utf-8")
                host = self.headers.get("Host") or f"localhost:{self.server.server_port}"
                src = f"{LAB_FACE}?src={quote(f'http://{host}/face.json', safe='')}"
                return self._send(200, KIOSK.format(src=html.escape(src)).encode(), "text/html; charset=utf-8")
            if path == "/local":
                return self._send(200, LOCAL.encode(), "text/html; charset=utf-8")
            return self._json(404, {"error": "not found", "routes": ["/", "/local", "/face.json", "/events", "/health"]})

        def _events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            hub.clients += 1
            try:
                self.wfile.write(b"retry: 2000\n\n")
                f = hub.snapshot()
                seq = 0
                while True:
                    if f is not None:
                        seq = f.get("seq", 0)
                        data = json.dumps(f, ensure_ascii=False)
                        self.wfile.write(f"id: {seq}\nevent: face\ndata: {data}\n\n".encode())
                    else:
                        self.wfile.write(b": ping\n\n")  # keepalive through idle proxies
                    self.wfile.flush()
                    f = hub.wait_newer(seq, KEEPALIVE_S)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.clients -= 1

    return Handler


def build(host: str, port: int, demo: bool, step_s: float = 1.5):
    """Start the face source and the HTTP server. Returns (server, hub); caller serves."""
    hub = Hub("demo" if demo else "robot")
    threading.Thread(target=run_source, args=(hub, demo, step_s), daemon=True, name="ae-source").start()
    srv = ThreadingHTTPServer((host, port), make_handler(hub))
    srv.daemon_threads = True
    return srv, hub


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Serve the æ face (read-only, authority æ1).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--demo", action="store_true", help="no robot: cycle all 7 moods")
    p.add_argument("--step", type=float, default=1.5, help="demo seconds per mood")
    a = p.parse_args(argv)
    srv, _ = build(a.host, a.port, a.demo, a.step)
    sys.stderr.write(f"ae.serve: http://{a.host}:{srv.server_port}/ ({'demo' if a.demo else 'robot'}, read-only, æ1)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
