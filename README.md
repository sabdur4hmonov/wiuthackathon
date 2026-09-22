# WIUT Hackathon 2026 — CV Track

Traffic event detection (Part A) and causal accident anticipation (Part B) from
a fixed road camera.

**Status: CP1 — four rule classes implemented against synthetic fixtures.**
The foundation (budget manager, perception pass, cache, zone tooling) is from
CP0. `congestion`, `stopped_vehicle`, `jaywalking` and `wrong_way` are now
implemented and tested, plus an HSV traffic-light classifier and a ground-truth
labelling tool. **No real footage has been seen yet**: `config/zones.json` is
still un-authored, so the rules receive `zones=None` and emit nothing, and every
threshold is a placeholder. See [Current state](#current-state).

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
bash weights/download.sh                          # once, with internet
python run_submission.py --videos samples --out predictions_samples.json --team <team>
python evaluate.py --pred predictions_samples.json --validate-only
```

With your own labels:

```bash
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
```

---

## Layout

```
solution.py              the interface the harness imports (CLASSES, detect_events, RiskEstimator)
run_submission.py        organizers' harness — UNCHANGED from the starter kit
evaluate.py              official metric — UNCHANGED from the starter kit
config/zones.json        hand-authored scene geometry (see Scene geometry)
src/
  config.py              tunables, seeds, offline enforcement
  budget.py              wall-clock budget manager
  geometry.py            point-in-polygon, crossings, headings, resampling (pure)
  zones.py               zones.json loader, validator, scene queries
  tracks.py              the tracks table schema
  cache.py               perception cache (dev tool; never a correctness dependency)
  perception.py          STAGE 1: video -> tracks
  thresholds.py          EVERY rule threshold, each with a calibration note
  signal.py              traffic-light ROI -> red / amber / green / unknown
  rules/                 STAGE 2: tracks + zones -> raw frame segments
    congestion.py  stopped_vehicle.py  jaywalking.py  wrong_way.py
    zoneindex.py         vectorised zone membership for a whole table
  postprocess.py         STAGE 3: merge, de-blip, guarantee no same-class overlap
  pipeline.py            Part A orchestration
  risk/                  STAGE B: causal estimator, ISOLATED from everything above
tools/
  extract_frames.py      dump frames to annotate against
  annotate.html          click zones onto a frame, download zones.json (zero setup)
  draw_zones.py          render zones.json over a frame, to check it by eye
  validate_zones.py      fail loudly on a half-authored config
  label.html             scrub a clip, mark [start, end, class], export GT
  bench.py               seconds-per-video-second, per stage
  make_synthetic_clip.py throwaway clip for timing before real samples exist
scripts/offline_check.py proves the pipeline completes with the network blocked
weights/download.sh      fetch weights once, before the offline run
tests/                   unit + end-to-end-through-the-real-harness
  fixtures/synthetic_scene.py   an invented intersection in the zones schema
  fixtures/synthetic_tracks.py  a track builder with the tracker's failure modes
```

### The three stages, and why they are separate

Stage 2 will be iterated on dozens of times over three days. If each iteration
re-ran the detector, the dev loop would be the bottleneck rather than the ideas.
So Stage 1 runs once and serialises to a cache, and Stage 2 runs off that table
in well under a second. Nothing in Stage 2 touches a video file.

---

## The time budget

`run_submission.py` gives **one** budget covering both parts:

```
duration = n_frames / fps          # OpenCV properties, NOT the container duration
budget   = 3.0 * duration
```

Part A runs first. Then `run_risk` **re-decodes the entire video** and calls
`step()` on every frame. If the total exceeds the budget — or if Part A alone
does — the harness replaces the whole entry with `{"events": [], "risk": []}`.

**A slow Part B does not cost Part B. It costs Part A as well.**

`src/budget.py` is built around that:

| limit | value | why |
|---|---|---|
| harness budget | 3.0 × duration | `TIME_FACTOR_DEFAULT` |
| Part A target | 1.2 × duration | shed optional work past here |
| Part A hard stop | min(1.5 × duration, budget − Part B reserve − 2 s) | the second term binds on short clips |
| Part B reserve | 1.2 × duration | a full decode pass; not negotiable |

Running long **degrades, never fails**: `should_stop()` makes perception stop
and emit events from the frames processed so far. `project_overrun()`
extrapolates from the current rate, so a slow run bails at frame 200 of 4000
rather than discovering the problem at frame 3900 with no time left to emit
anything. A partial answer scores; a timeout scores zero for both parts.

### Measured (see [Timing](#timing))

Part B's floor is a pure decode pass — the harness pays it no matter what
`step()` does. Keeping `step()` negligible against that floor is the whole
design constraint, and it is met by three orders of magnitude.

---

## Scene geometry

The task PDF promises `samples/camera.md` describing the road layout. **It is
not in the starter kit** — there is no scene description at all. So the geometry
is authored by hand against a real frame, and `config/zones.json` ships with
**every geometric field `null`**. Nothing about the scene is guessed.

`zones.json` expresses: carriageway polygons; per-lane polygons with a direction
arrow and permitted manoeuvres; stop-line segments with the lanes they govern
and which side vehicles approach from; pedestrian crossing polygons; signal
queue zones (so `stopped_vehicle` can exclude cars waiting on red); traffic
light ROIs; lane markings flagged solid/dashed; and the image size the
coordinates were authored against (everything rescales if the video resolution
differs).

### Authoring workflow

```bash
python tools/extract_frames.py --video samples/<clip>.mp4 --out frames/
# open tools/annotate.html in a browser, load the frame, draw, Download zones.json
mv ~/Downloads/zones.json config/zones.json
python tools/validate_zones.py
python tools/draw_zones.py --image frames/<clip>_f000000.png --out check.png
```

`tools/annotate.html` is a single self-contained file that runs from `file://` —
no server, no install. (An OpenCV click-picker is not an option:
`requirements.txt` pins `opencv-python-headless`, which has no GUI.) It gives
zoom, pan, undo, per-shape metadata, and computes `approach_side` with the same
sign convention as `src/geometry.py`.

**Look at `check.png` before trusting the config.** The validator proves the
file is complete and self-consistent; it cannot tell you that the polygon
labelled "lane 2" is actually the pavement.

`src/zones.py` refuses to load a file that still holds `null`s, and the failure
names every missing field. A half-authored config cannot silently produce
plausible-looking garbage — which matters, because a wrong event scores 0 for
its class *and* enlarges Part A's class set.

---

## Detector and tracker

**YOLO11s @ imgsz 960 + ByteTrack.**

Against a T4 (16 GB) and a 3× budget of which Part B already claims ~0.2×:

- **YOLO11s over YOLO11n**: the nano model drops the small, distant vehicles a
  CCTV view is full of, and every Stage 2 rule needs those tracks to exist.
- **YOLO11s @ 960 over YOLO11m @ 640**: for roughly the same cost, the extra
  input resolution buys small-object recall, which is where this scene's
  difficulty lives. The `m` model's gains are concentrated on large objects we
  already detect.
- **over RT-DETR**: stronger on crowded scenes, but ~2× the latency at equal
  resolution, no NMS knob to trade recall for speed under budget pressure, and a
  less-exercised tracker integration. Losing a whole video to a timeout costs
  more than the accuracy gain.
- **ByteTrack**: its second association pass uses the *low-confidence*
  detections, which is exactly the regime of a partly occluded car at the far
  end of the frame. It needs no ReID model, so it adds nothing to the GPU
  budget.

### ID-switch behaviour — every rule downstream depends on this

ByteTrack is motion-only (IoU + Kalman) with **no appearance model**:

- A track occluded for ≤ `track_buffer` frames is re-associated and **keeps its
  id**. Past that it is deleted and the object gets a **new id** on
  reappearance. At stride 2 on 25 fps the default 30-frame buffer is ≈ 2.4 s of
  real time.
- Two similar boxes crossing with high IoU **can swap ids** — on this camera,
  wherever lanes converge in the distance and boxes are small.

**Consequences for Stage 2:** never assume one id spans a whole manoeuvre; never
treat an id change as evidence that anything physical happened. Fragment
stitching belongs in Stage 3, which merges same-class segments within
`merge_gap_sec` precisely so an id switch mid-event does not split one true
positive into two half-IoU segments that both miss tIoU 0.7.

Detections the tracker did not adopt are kept with `track_id = -1` rather than
discarded: a `stopped_vehicle` or `road_obstacle` can sit untracked for a long
time, and dropping that evidence at Stage 1 would make the class undetectable
no matter what Stage 2 does.

---

## Tracks cache

Keyed on `(sha256 of the video's bytes, hash of PerceptionConfig)`. Editing any
perception tunable or swapping the video invalidates automatically — there is no
manual "clear the cache" step to forget.

**It is a dev tool and can never affect correctness.** Every read and write is
wrapped; any failure degrades silently to recompute. The judges' machine may
have a read-only checkout, a full disk, or no writable temp dir, and none of
those may change a prediction. Location comes from `$WIUT_CACHE_DIR`, falling
back to the OS temp dir — never inside the repo, so a stale entry cannot be
committed.

```bash
WIUT_NO_CACHE=1 python run_submission.py ...   # force recompute
WIUT_CACHE_DIR=/fast/disk python ...           # relocate
```

A truncated table (perception stopped on budget) is **not** cached: that is an
artefact of one slow run, and serving it back would cap every future run at the
same point.

**Format: `.npz`** — a dense `float32` table plus a column-name array and a JSON
metadata blob. Chosen over Parquet (needs `pyarrow`, ~50 MB of dependency for no
gain), JSON (huge and slow for a numeric table), and pickle (unsafe and fragile
across versions). Columns are append-only; a schema change is caught on load and
read as a miss.

Schema: `frame_idx, t_sec, track_id, cls, conf, x1, y1, x2, y2, gx, gy, vx, vy,
speed, heading`. `gx, gy` is the **ground-contact point — bottom-centre of the
box, not the centroid**: on a CCTV view the centroid floats up the object as it
nears the camera, so a lane test on the centroid drifts across lane boundaries
for a vehicle driving perfectly straight. Kinematics are computed once, at
cache-build time, over a centred window of `velocity_window` samples — a
single-frame difference on a jittery CCTV box is mostly detector noise.

---

## Part B is strictly causal

`src/risk/` may import `numpy`, `src.geometry` and `src.config` and **nothing
else**. It may not import `src.perception`, `src.cache`, `src.pipeline`,
`src.tracks`, `src.rules`, `src.postprocess` or `cv2`.

The tracks cache is built by a pass that has already seen the whole video.
Reading it inside `step()` would smuggle future frames into a score the metric
treats as causal — a disqualifying violation, not a bug. So it is enforced
structurally rather than by discipline, and `tests/test_risk_isolation.py`
checks it three ways: an AST scan of the source, a **fresh interpreter** that
imports `src.risk` and asserts no banned module reached `sys.modules`, and a run
with the batch modules made un-importable via a `sys.meta_path` blocker.

**Default-quiet.** Alarm precision pools over *all* videos
(`precision = matched_alarms / all_alarms`), so an alarm on an accident-free
video is a pure false positive that drags down `F1_alarm` for every other video
too. Alarms are runs of `score ≥ 0.5` merged within 2 s, so a score idling near
0.5 manufactures alarms out of nothing. The score therefore sits at 0.0 and must
be pushed up by a real signal.

`step()` is cheap **by construction**: real work happens once every
`work_every_n_frames`; every other call returns the cached score off a branch
and an integer compare.

---

## Offline guarantee

No internet at evaluation. Weights are loaded by **explicit local path** —
`src/perception.py` never passes a bare model *name* to Ultralytics, which is
the code path that downloads.

Ultralytics phones home in three places, each of which would turn into an empty
prediction for the video it happens on:

1. the **AMP self-check** downloads `yolo11n.pt` on the first CUDA inference —
   disabled on the model, its inner module and its predictor;
2. a version / analytics ping — `SETTINGS["sync"] = False`;
3. a font download in its plotting helpers.

Plus `YOLO_OFFLINE`, `ULTRALYTICS_OFFLINE`, `HF_HUB_OFFLINE`, `TORCH_HOME`
pinned into the repo, and `runs_dir`/`weights_dir` pinned so a miss fails loudly
instead of reaching out.

**Asserting env vars proves nothing**, so:

```bash
python scripts/offline_check.py --video samples/<clip>.mp4
```

monkey-patches `socket` so *any* outbound connection raises, verifies the block
itself works, then runs `detect_events` and `RiskEstimator` end to end. Exit 0
means the pipeline completed with every socket blocked. `weights/download.sh` is
provided because the README pattern allows it, but the offline path works
without it.

---

## Determinism

`src/config.seed_everything()` fixes `PYTHONHASHSEED`, `random`, `numpy`,
`torch`, CUDA, `cudnn.deterministic`, and disables `cudnn.benchmark`. Called at
import of `solution.py` and again at the start of every `detect_events`.
`tests/test_harness_contract.py::test_detect_events_is_deterministic` asserts
two runs agree.

---

## Current state

| CP0 deliverable | status |
|---|---|
| repo skeleton, module split | done |
| budget manager | done, unit-tested |
| `tools/extract_frames.py` + annotation workflow | done |
| `zones.json` schema + loader + geometry helpers | done, unit-tested |
| perception pass with disk cache | done |
| `solution.py` wired end to end | done |
| `evaluate.py --validate-only` passing | done |

| CP1 deliverable | status |
|---|---|
| synthetic scene fixture | done, validates through the real validator |
| synthetic track generator + tracker failure modes | done |
| `stopped_vehicle`, `wrong_way`, `jaywalking`, `congestion` | **implemented**, tested clean and degraded |
| traffic-light classifier | done, degrades to "no signal available" |
| labelling tool | done, output scored through the real `evaluate.py` |
| **anything validated against real footage** | **none** |

### Rules are written; they are not calibrated

Every rule is tested against `tests/fixtures/`, which is an **invented** scene
and **invented** tracks. That proves the LOGIC is right — including against id
switches, dropouts, occlusions and bbox jitter. It proves nothing about this
camera.

Concretely, when the real clips land:

- **`config/zones.json` is still un-authored**, so `_get_zones` returns `None`,
  every rule emits nothing, and the pipeline produces `[]`. That is deliberate
  and safe, not a bug — but it means Part A scores 0 until the geometry is drawn.
- **Every pixel and px/s threshold in `src/thresholds.py` is a guess.** The
  synthetic scene's perspective is a linear widening, not a projective mapping,
  so pixels-per-metre is fiction. Speed thresholds are pixels over time and
  inherit the problem twice.
- Only `stopped_vehicle.min_duration_sec = 10.0` has real authority: the task
  PDF defines it.

The retuning loop is cheap by design: Stage 2 runs off the cached tracks in
~0.6 s, so a threshold sweep against a labelled dev set costs seconds, not a
detector re-run. That is what the CP0 cache was for.

### What will need attention first

1. **`stopped_vehicle.speed_px_s`** — measured against synthetic jitter, where a
   stationary vehicle shows ~5.6 × jitter_px of apparent speed. Real box wobble
   will differ, and it varies with object size, so near-field and far-field
   vehicles need different treatment. A perspective-scaled threshold is the
   obvious upgrade.
2. **`congestion`'s queue-zone discriminator** — it works only if the real
   `signal_queue_zones` are drawn generously. Draw them too small and every red
   phase becomes congestion.
3. **`wrong_way` and the intersection box** — in the synthetic scene the NS lanes
   cover the box, so turns through it are judged against an NS lane direction. A
   real config that leaves the box laneless changes that entirely.
4. **`congestion`'s direction grouping** — lanes are grouped by clustering their
   direction arrows, so a sloppily drawn arrow puts a lane in the wrong group.

## Tests

```bash
python -m pytest -q
```

Covers the geometry helpers (sign conventions pinned explicitly — an inverted
sign would not crash, it would silently invert a rule), the budget limits and
degradation path, the zones validator, the cache's failure modes, the causality
isolation, and an end-to-end run through the **real, unmodified**
`run_submission.py` and `evaluate.py`.

## Licences

- **Ultralytics YOLO11** — AGPL-3.0. Weights from the Ultralytics assets
  release, COCO-pretrained.
- No external training datasets used at CP0. Any added later will be listed here
  with its licence, as the task requires.

---

## Timing

Measured on this dev box: **8-core CPU, no GPU**, Python 3.11.9, torch 2.14.0+cpu,
ultralytics 8.4.159, on a synthetic 60 s / 1920×1080 / 25 fps clip
(`tools/make_synthetic_clip.py`). Re-measure on the real clips and on a GPU.

### Part B — the floor (CPU-bound, transfers to any box)

| | value |
|---|---|
| pure decode, median of 3 passes | 9.52 s = **0.159× realtime** (6.34 ms/frame) |
| spread across passes | 9.38 s .. 13.73 s (~20% jitter) |
| `RiskEstimator.step()` in isolation | **0.43 µs/frame** |
| step() over the whole clip | 0.0006 s = **0.007% of the decode floor** |

Part B ≈ the decode floor. `step()` is negligible against it by three orders of
magnitude, which is the design goal: the harness pays the decode no matter what,
so the only thing we control is not adding to it.

**Do not measure step() by subtracting two decode passes** — decode jitter is
~3.3 s, five thousand times step()'s actual cost. `tools/bench.py` times it in
isolation for this reason.

### Part A — detector throughput (YOLO11s, 1920×1080 input)

| imgsz | ms/frame (CPU) | ×realtime @ stride 2 | ×realtime @ stride 5 |
|---|---|---|---|
| 640 | 420.9 | 5.26 | 2.10 |
| **960** (default) | **650.3** | **8.13** | 3.25 |
| 1280 | 1405.8 | 17.57 | 7.03 |

**No CPU configuration fits the 3.0× budget with margin.** This is expected and
does not indicate a problem with the defaults: the evaluation box has a T4, where
YOLO11s in fp16 is roughly an order of magnitude faster than this CPU. The
defaults (960 / stride 2) are chosen for that box.

**The T4 numbers are not measured and must not be trusted until they are.** Run
`python tools/bench.py --video samples/<clip>.mp4` on GPU hardware before relying
on any of this. If Part A lands above ~1.2× there, drop `frame_stride` to 3–5
first — it is linear in cost and cheaper in accuracy than reducing `imgsz`.

### The budget manager does its job

On CPU the hard stop fires exactly as designed. From a real run:

```
[budget] 60.0s video @ 25.0fps | A used 11.11s (0.185x realtime) | A-hard 90.0s
[budget]   model_load                0.836s one-off; excluded from the per-frame rate
[budget]   perception                4.822s 10 frames, dev=cpu
[budget]   ! projected overrun: 10/750 frames in 11.1s, limit 90.0s
```

It extrapolated from 10 frames that finishing would take ~360 s against a 90 s
limit, stopped, and returned a valid (empty) result. The harness recorded
`23.0s` of a `180s` budget and logged **zero** errors. A partial answer scores;
a timeout scores zero for both parts.

Note `model_load` is timed separately and excluded from the per-frame rate — an
early version amortised the one-off load across the first ten frames and aborted
runs that would have finished comfortably.
