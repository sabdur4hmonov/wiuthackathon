# Live demo (Hugging Face Space, Gradio, CPU)

`demo/app.py` wraps the submission's own code: `src/` Stage 1 + rules + Stage 3
(the same calls as `solution.detect_events`) and the causal `RiskEstimator`.
Independent of `web/`: the website links to the Space.

## What it does

Upload a road-camera clip. It returns:

* a summary (events found, processing time, whether the scene matched our camera),
* an annotated playback (boxes, crossings/queue zones, event banners, risk bar),
* an event timeline and the risk curve (0.5 = alarm threshold),
* one short clip per event,
* `prediction.json` in the harness's format (`events` + per-frame `risk`).

**Limits: up to 120 s and 200 MB per clip**, any common container (MP4/H.264
works best). Expect about 2-3x the clip's length on a free CPU Space (2 vCPU).
One job at a time; up to 8 queued.

**Scenes.** The four event rules use zones drawn for the competition's
intersection, aligned to each upload (`src/align.py`). On a different scene the
alignment fails and the app says so: the rules are NOT run (zones on the wrong
street would produce confident nonsense); detections, tracks and the risk
curve are still shown.

**Robustness.** Checked before any work: empty file, non-video, undecodable
first frame, under 2 frames, over 120 s, over 200 MB -> a readable error. An
fps the container reports as 0 or absurd falls back to 25, as the harness
does. Anything else that fails is caught and reported, never a crash.

The demo's risk estimator runs at 2 Hz with no budget guard (there is no 3x
budget in a demo, and a user waiting for a curve wants the whole curve); the
submission runs it at 5 Hz under the guard.

## Run locally

```bash
pip install -r demo/requirements.txt
python demo/app.py            # http://127.0.0.1:7860
```

## Deploy to Hugging Face

```bash
pip install huggingface_hub
python demo/deploy.py <user-or-org>/<space-name> --dry-run   # stage only: demo/_space_build/
HF_TOKEN=hf_xxx python demo/deploy.py <user-or-org>/<space-name>
```

`deploy.py` stages exactly what the Space needs (app.py, the Space card as
README.md, requirements.txt, packages.txt, solution.py, src/, config/zones.json,
config/pose_reference.jpg, weights/yolo11s.pt -- 19.7 MB) and uploads it with
`huggingface_hub` (creates the Space with the Gradio SDK if it does not exist;
the weights go through LFS automatically). The first build takes ~5-10 min
(CPU torch). Then set the Space URL in the website and in `space_README.md`
(`<REPO_URL>` placeholder for the code link).

Space files: `space_README.md` (card + config: Gradio SDK, AGPL-3.0 because
Ultralytics YOLO11 is AGPL-3.0), `requirements.txt` (CPU torch wheel index),
`packages.txt` (libGL for OpenCV).
