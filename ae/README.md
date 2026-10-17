# æ face · Asimov 1

The robot has no face. This bridge gives it one without touching its controls.

`face.py` reads the state the SDK reports and turns it into a mood. It sends nothing to the
robot. `serve.py` puts that mood on the network so a phone, a tablet in the head or any
browser can draw the æ æ æ face.

| Robot reports | Mood |
| --- | --- |
| FAULT_DAMP, an error flag or a critical alert | hold |
| no sample in the last 0.5 s | lost |
| battery under 20 % | tired |
| DAMP | sleep |
| STAND, not armed | think |
| STAND, armed | ready |
| MOVE | work |

Rules are checked top to bottom, so safety wins: a faulted robot is `hold` whatever else it says.
If the bridge itself cannot read the robot (SDK missing, link down), the face says `lost`,
never a happy face.

## Run on the robot's companion computer

Python 3.9+, standard library only (plus menlo-sdk from this repo for the real robot).

```bash
git clone -b ae-face https://github.com/LutheranInstitute/Robot && cd Robot
pip install -e .                 # menlo-sdk, only needed for the real robot

python3 ae/serve.py --demo       # no robot: cycles all 7 moods
python3 ae/serve.py              # saved robot or MENLO_* variables
python3 ae/serve.py --port 8765 --host 0.0.0.0   # the defaults
```

Then point a phone on the same network at:

```
http://<robot-ip>:8765/
```

Find the IP on the companion computer with `hostname -I` (Linux) or `ipconfig getifaddr en0` (macOS).

| Route | What it gives |
| --- | --- |
| `GET /` | kiosk: `ae/face.html` if it exists, else the æ lab face in a frame, fed `?src=http://<robot-ip>:8765/face.json` |
| `GET /local` | tiny built-in face, same origin, works offline |
| `GET /face.json` | latest face (`Access-Control-Allow-Origin: *`) |
| `GET /events` | Server-Sent Events: one `face` event per mood change |
| `GET /health` | `ok`, source, mood, seq, error, `read_only: true` |

Browsers block an `https://` page from reading an `http://` robot (mixed content), so the
framed lab face may show no data on a phone. `/local` always works; for the full face offline,
drop the lab's `face.html` into `ae/` and `/` serves it from the robot itself.

`face.py` on its own still works and still writes `face.json` (`AE_FACE_OUT` sets the path):

```bash
python3 ae/face.py --demo
python3 ae/face.py
```

## Schema ae.face/0.1

```json
{"schema":"ae.face/0.1","who":"asimov","authority":"æ1","mood":"work","means":"MOVE: the walking policy balances the robot","mode":"MOVE","armed":true,"battery":80,"alerts":[],"age_ms":50,"at":"2026-10-17T00:20:31+00:00","ts":1792196431.703,"seq":8}
```

`ts` (unix seconds) and `seq` (1, 2, 3 … per bridge process) are additions; readers of the
first 0.1 faces ignore them. A drop in `seq` means the bridge restarted.

## Read-only guarantee

- The bridge calls only `Robot().connect()`, `get_state()` and `armed`. It never arms,
  disarms, changes mode or sends a motion.
- The HTTP server answers GET, HEAD and OPTIONS only. Anything else gets `501`. There is no
  route that writes.
- The robot publishes at æ1 and never gets operator authority from wearing a face.

Evidence ≠ Authority: the face reports, the human decides. **Human stays root.**

## Tests

```bash
python3 ae/test_face.py
```

Covers the 7-priority mapping and its boundaries, `ts`/`seq`, and starts `serve.py --demo`
on a random port to check `/face.json`, `/health`, `/events`, `/` and that POST is refused.
No robot needed.

LAB. Not part of menlo-sdk; lives beside it in this fork. #osw #ioa
