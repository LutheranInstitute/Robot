# æ face · Asimov 1

The robot has no face. This bridge gives it one without touching its controls.

`face.py` reads the state the SDK reports and turns it into a mood. It sends nothing to the
robot. A screen reads `face.json` and draws the æ æ æ face.

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

```bash
python ae/face.py --demo   # no robot needed
python ae/face.py          # saved robot or MENLO_* variables
```

Each change is one `ae.face/0.1` line:

```json
{"schema":"ae.face/0.1","who":"asimov","authority":"æ1","mood":"work","mode":"MOVE","armed":true,"battery":80}
```

Evidence ≠ Authority: the face reports, the human decides. The robot publishes at æ1 and
never gets operator authority from wearing a face.

LAB. Not part of menlo-sdk; lives beside it in this fork.
