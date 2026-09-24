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


---

## CP2: real footage findings (2026-09-23)

The first real camera clip landed (`samples/sample_001.mp4`, gitignored --
5.9 GB, not committed). Everything above this section was measured against
synthetic fixtures. This is the first contact with the real format, and it
changed two things.

### The real format

```
codec:        h264, High 4:2:2 profile (avc1)
pixel format: yuv422p10le  -- 10-bit, 4:2:2 chroma subsampling
resolution:   3840x2160
frame rate:   29.97 fps  (30000/1001, NOT a flat 30/1)
colour:       bt709 -- standard Rec.709, NOT HLG or S-Log (no log gamma to correct for)
bitrate:      ~140 Mbps video
duration:     340.34 s (10200 frames) -- cv2's n_frames/fps agrees with the
              container duration to the millisecond; no VFR mismatch on this
              clip, though src.budget.probe_duration still guards against one
```

The container is XAVC, the format Sony broadcast/pro cameras write. 10-bit
4:2:2 is a materially heavier decode than the 8-bit 4:2:0 every fixture in
this repo was built against.

### Measured: the real Part B floor

`tools/bench.py` gained `--max-frames`, because a full 340 s pass is
impractical for exploratory measurement (the first attempt, unbounded, ran
uninterrupted for many minutes -- `timeout` under this shell could not
actually kill the native Windows process; it had to be force-killed). Sampled
from frame 0, extrapolated to the full clip, on this 8-core CPU machine:

| sample | ms/frame | x realtime |
|---|---|---|
| 100 frames | 49.1 | 1.47x |
| 1500 frames | 59.7 | 1.79x |

The two disagree by ~21%: decode cost is not flat across a clip (GOP
structure, reference-frame buffering, scene motion), and 100 frames from the
very start is not representative. Both numbers are well BELOW the Colab
2-vCPU measurement of 3.05x that motivated this investigation -- this 8-core
machine has real spare cycles that Colab's 2 vCPUs did not. Part B alone,
even at the higher 1.79x, still leaves headroom under the 3.0x harness budget.

**Not yet done: a full-file confirmation pass, and a T4-class GPU
measurement.** Both remain the two biggest open risks. The ~21% intra-clip
variance is folded into `BudgetConfig.part_b_measured_safety` (1.30x) below.

### Tested and rejected: decoding through ffmpeg for Part A

The hypothesis: cv2 decodes and colour-converts every processed frame at full
4K/10-bit before the detector immediately resizes it away, so piping through
ffmpeg's own `scale` filter should decode once at a smaller size instead.
Built (`src/ffdecode.py`, `PerceptionConfig.decoder`), then measured
decode-only cost against the real clip, 300 processed frames, stride 2:

| decoder | ms/processed-frame | vs cv2 |
|---|---|---|
| cv2 grab/retrieve, native 4K | 81.6 | -- |
| ffmpeg piped to 1280 width | 104.0 | 0.78x (slower) |
| ffmpeg piped to 960 width | 93.8 | 0.87x (slower) |
| ffmpeg piped to 640 width | 81.9 | 1.00x (even) |

**The hypothesis was wrong.** Never faster, at any width tested. Two reasons:
H.264 decode of this stream dominates the per-frame cost and happens BEFORE
any scale filter runs, so scaling the output does not reduce it; and cv2's
`grab()`-only skip on non-stride frames is genuinely cheap (decode, no
colour-convert or copy), while the pipe must fully decode, scale, convert and
write EVERY frame to keep frame indices correct, so it pays full cost on the
frames cv2 gets almost free. `PerceptionConfig.decoder` stays `"cv2"`.
`src/ffdecode.py` is kept (tested, opt-in, falls back to cv2 automatically on
any failure) rather than deleted, in case different footage or a `select`
filter closes the gap later -- **do not flip the default without
re-measuring.**

### Built: an adaptive Part B reserve

`BudgetConfig.part_b_reserve` was a flat 1.2x guess, made before any real
footage existed. `detect_events` now measures it instead: at the top of every
call, `src.budget.measure_part_b_floor` reads a handful of real frames with
`cap.read()` -- run_risk's own call, not the cheaper `grab()`/`retrieve()`
Stage 1 uses for itself -- and extrapolates. The probe is bounded twice, the
same way `safety_margin` already was: a flat `part_b_probe_max_sec` (3.0 s)
protects a huge clip from a long probe, and `part_b_probe_max_frac` (5% of
the total budget) protects a tiny clip's already-thin budget from the probe's
own fixed overhead -- the second bound did not exist on the first pass and
was added after it was implicated (see "A real regression, found and fixed"
below). `part_b_measured_safety` (1.30x headroom over the measurement) is set
from the real ~21% intra-clip variance measured above, not a round number.

A real run against the synthetic 60 s fixture: `1.2x fixed guess = 72.0s`
became `measured = 22.7s` -- freeing real Part A time the fixed guess was
reserving unnecessarily on a clip that turned out to be cheap to decode.

### A real regression, found and fixed

The first version of this change opened a second `cv2.VideoCapture` on every
call -- one throwaway open to read native width/height, then a fresh open for
the actual decode loop -- where CP0's original code opened exactly one.
`tools/bench.py`'s own harness-contract tests (tiny ~1-1.6 s synthetic clips,
`run_submission.py` run as a real subprocess, so the model is always cold)
started failing: the clip's whole budget can be under 2 s, and the extra
open/close was enough added overhead, stacked on a cold model load, to blow
it. Fixed: `run_perception` now opens one `VideoCapture` for the default cv2
path, exactly as before; a second, metadata-only open happens ONLY on the
non-default ffmpeg path, which needs native width/height first to compute the
scaled target size.

### Found, not fixed: pre-existing flakiness on tiny clips

After the fix above, `tests/test_harness_contract.py` still occasionally
failed the same way (`test_00N.mp4: 5-6s used of a 4.8s budget`). Verified by
`git stash`-ing every change from today and re-running the identical
harness-contract suite against the last committed state: **the same failure
reproduces on unmodified CP1 code.** `_BUDGET_CHECK_EVERY = 10` means the
first stop-check cannot fire before 10 processed frames; on a clip with only
~20 processed frames total and real (not cached-warm) CPU YOLO inference
occasionally running north of 0.5-0.9 s/frame under load, reaching that first
check point alone can exceed a budget that is only 1.5-3.0 s to begin with.
This is a latent gap in CP0's own architecture, surfaced by today's heavier
system load (large decode benchmarks, an ffmpeg subprocess comparison, a
nearly-full disk), not introduced by anything in this session. Left
undiagnosed further and unfixed: it is out of today's scope, and a proper fix
(checking more eagerly on short clips, or immediately after the very first
processed frame rather than waiting for a batch of 10) deserves its own pass
rather than a rushed patch appended to an unrelated change. 356 of 360 tests
pass; the 4 failures are exactly this pre-existing, load-sensitive gap.

### Immediate next risks, in order

1. **No T4 measurement exists.** Every number above is this 8-core CPU
   machine. YOLO11s in fp16 on a T4 should be an order of magnitude faster,
   but "should be" is a guess until `tools/bench.py` runs on one.
2. **No full-file decode confirmation.** Both real numbers above are
   extrapolated from a sample of the first 100-1500 frames (3-50 s of a
   340 s clip). A full pass, or samples from multiple offsets, would close
   this out.
3. **The `_BUDGET_CHECK_EVERY` gap above**, if the hidden test set turns out
   to include very short clips.
4. **Zones are still un-authored** -- unchanged from CP1, still the single
   blocking item for Part A to score above zero.

## CP3: the four rule classes on real footage (2026-09-24)

Stage 1 was run in full (no budget stop) on all four real clips and cached;
the rules and the unchanged Stage 3 then run off the cache. Zones are aligned
to each clip's camera pose first (`src/align.py`, committed separately), so
sample_003/004 -- framed up to ~150 px differently -- are real dev clips too.
There are still **no labels**: every threshold is biased toward not firing,
because under macro F1 a wrong class costs as much as a missed one.

### What fires (`solution.detect_events`, cache hits)

| clip | jaywalking | stopped_vehicle | wrong_way | congestion |
|---|---|---|---|---|
| sample_001 (340 s) | 1: 01:50–01:56 | 0 | 0 | 0 |
| sample_002 (318 s) | 3: 01:18–01:25, 02:20–02:23, 02:51–02:54 | 0 | 0 | 0 |
| sample_003 (318 s) | 4: 01:27–01:35, 02:52–02:55, 03:01–03:05, 04:19–04:22 | 0 | 0 | 0 |
| sample_004 (128 s) | 2: 00:19–00:25, 00:26–00:35 | 0 | 0 | 0 |

The jaywalking events were checked frame by frame: people walking mid-block,
cutting from the zebra across the box, and walking between queued cars --
except sample_002 02:51, a moped rider the detector did not box as a
motorcycle, so the rider test had nothing to match.

### What the real footage changed

The first real run fired 19 events on sample_002 alone. Each was traced to a
cause in the frames, and fixed at the cause:

* **wrong_way judges painted lanes only.** All 11 hits were lawful traffic in
  the unpainted area right of the refuge, where NB and SB traffic, NB queues
  and turning vehicles share the road. A lane's direction is only as good as
  its paint.
* **The NB queue is bigger than the flow map said.** NB cars queued at red do
  not move, so the optical-flow split missed them; `Q_NB_approach` now covers
  where tracks were measured standing. SB-exit traffic also queues for
  something downstream, out of frame (>= 1 standing vehicle in 54 % of
  sample_001's frames): `Q_SB_exit_downstream`. The bus-stop zone starts
  behind the stop, where traffic waits for a dwelling bus.
* **The far kerb, right of x = 2500, was traced ~100 px too high** in tree
  shade: vehicles never drove within 100 px of it, pedestrians did by the
  thousand. Moved to where vehicles actually are.
* **stopped_vehicle** now ignores stops that are part of a queue, boxes cut off
  by the frame edge, cars pulled up on the kerb line, and very small distant
  boxes. On these four clips it fires nowhere.
* **jaywalking** needs a full body height inside the kerb (bus-stop boarding
  was 14 % of candidates at half a body height), ignores boxes cut by the frame
  and person boxes inside a vehicle box (riders, passengers).
* **congestion** is judged per direction as a whole (SB, NB; named in
  zones.json), over a sliding window, on traffic outside every queue zone. It
  never came close to firing: the SB side has almost nothing countable once
  the approach and the downstream queue are excluded.
* A kerb-margin test must not treat the frame border as a kerb: carriageway
  edges on the authored frame border are marked and skipped.

### Known limits, not fixed

* **Ground contact on tall vehicles**: the bottom-centre of a truck's box sits
  half a lane toward the camera (`geometry.bbox_ground_point`). Lane-level
  rules are biased for trucks and buses.
* **A jam confined to the approach looks like a long red** and is not
  detected: telling them apart needs the signal phase per frame.
* **People walking just outside a zebra** are flagged (accepted).
* **The budget on this CPU machine**: the measured Part B floor on these 4K
  clips leaves Part A's hard limit near zero, so on a real harness run here
  perception -- and alignment -- would barely start. The four
  `test_harness_contract` failures (tiny clips over their 5 s budget) are the
  same pre-existing gap from CP2: identical before and after CP3.

## CP4: keyframe-only Part A (2026-09-24)

The harness's own Part B pass decodes every frame of the original 4K file
(~2.3-2.5x realtime on the 8-core box). Part A used `cv2.grab()` on every
frame too, and `grab()` still runs the H.264 decoder: stride only saved the
colour conversion and the detector. Two full decodes cannot fit in 3x. CP4
makes Part A decode keyframes only.

### The GOP (`tools/gop_probe.py`)

| clip | frames | keyframes | GOP (frames) | GOP (s) | B-frames |
|---|---|---|---|---|---|
| sample_001 | 10200 | 680 | 15 fixed | 0.501 | yes |
| sample_002 | 9525 | 635 | 15 fixed | 0.501 | yes |
| sample_003 | 9525 | 635 | 15 fixed | 0.501 | yes |
| sample_004 | 3825 | 255 | 15 fixed | 0.501 | yes |

Every GOP is exactly 15 frames, with no variation, in display order
`BBI BBP BBP BBP BBP`: one I, four P, ten B, and one third of packets reordered
(pts != dts). A keyframe arrives every 0.5 s -- 2 samples per second -- well
under the ~1 s where tracking would break, so keyframes-only (`NONKEY`) is
the primary mode. `NONREF` would decode I+P, 10 samples per second.

### Part A decodes keyframes only (`src/avdecode.py`)

Stage 1 now decodes through PyAV with `skip_frame = "NONKEY"`: FFmpeg drops
every non-keyframe before decoding it, and the frame is downscaled to the
detector width inside the reformat step (AREA filter, ~9 ms/frame). Boxes are
scaled back to native pixels, so zones, box-height units and the rules see
the same coordinate frame as before. cv2 remains the fallback if PyAV is
missing or cannot open the file.

**Frame threading defeats skip_frame.** With `thread_type = "AUTO"` FFmpeg
picks frame threads, and every frame thread decodes its packet whatever
skip_frame says: NONKEY cost 0.61x realtime that way, against 0.14x with
slice threading or none. The reader uses SLICE and a test pins it.

Full-clip Part A decode cost, 8-core box, keyframes + downscale to 960, no
inference:

| clip | keyframes | wall | x realtime |
|---|---|---|---|
| sample_001 | 680 | 40.1 s | 0.118 |
| sample_002 | 635 | 39.0 s | 0.123 |
| sample_003 | 635 | 38.9 s | 0.122 |
| sample_004 | 255 | 13.7 s | 0.107 |

A first, cold-cache pass measured 0.27x on sample_001 (the 6 GB file coming
off disk). A run on sample_003 once stalled for 77 minutes, most likely the machine sleeping; its
immediate rerun measured 0.114x with no gap over 0.47 s between frames.
Compared with the full decode (~2.3-2.5x), Part A's decode is now about a
twentieth of Part B's.

The frame index of each keyframe is `round((pts - start) * time_base * fps)`
with the harness's fps. Verified against cv2 on sample_004 (keyframes land on
frames 2, 17, 32, 47: each GOP opens with two B-frames in display order) and
in `tests/test_avdecode.py` on an H.264 clip with 15-frame GOPs and B-frames.

### Tracking at 0.5 s per sample (`src/tracker.py`)

Measured on the full-rate CP3 tracks, 0.5 s apart, a moving pedestrian's box
overlaps its own earlier box with IoU < 0.2 in 90-94% of cases (moving cars
40-57%). Stock ByteTrack links on overlap alone, and confirms a new track only
if its second detection overlaps the first, so at keyframe rate walkers were
simply never tracked: replaying stock ByteTrack reproduced 4 of CP3's 10
jaywalking events. Retuning the frame-counted numbers could not fix that.

`KeyframeTracker` keeps ByteTrack (Kalman, two-stage association, Hungarian)
and replaces its similarity with the larger of IoU and a motion term, `1 -
centre distance / (reach x box height)`, gated to the same class group and a
plausible size change. Settings are in seconds and box heights, converted from
the measured sample step: `track_buffer_sec` 2.0 (CP3's 30 frames at stride 2),
reach 1.5 heights for people, 2.5 for vehicles. Score fusion is off: stock
ByteTrack multiplies similarity by confidence, which lowers the bar for a
conf-0.45 pedestrian to a similarity of 0.65.

Tuned by replaying the tracker over stored keyframe detections of all four
clips (the same detector calls Stage 1 makes), scored against CP3:

| association | CP3 people covered | links kept | CP3 events reproduced |
|---|---|---|---|
| IoU only (stock) | -- | -- | 4 / 10 |
| motion term, score fused, reach 1.0 | 79.8% | 97.4% | 7 / 10 |
| motion term, no fusion, reach 1.5 (**default**) | 82.2% | 96.5% | 8 / 10 |

**What the keyframe rate cannot do: direction and speed.** In a platoon moving
about one car-gap per 0.5 s the tracker hops back to the car behind each
sample, and the track drifts backwards. On sample_002 at 0:46, real cars moved
left at 550-680 px/s while keyframe tracks drifted right at 150-200 px/s in
lanes NB1-NB3 -- 11-13 wrong_way events over the four clips, all false.
`wrong_way` and `congestion` therefore stay silent unless samples are at most
0.2 s apart (`max_sample_sec`); neither fired in CP3. `jaywalking` and
`stopped_vehicle` are position-based and unaffected.

**A sample-rate bug in the rules, fixed.** Gap-bridging measured sample to
sample, so two consecutive samples counted one step as "no evidence": 0.07 s
at stride 2, but 0.5 s at keyframe rate, where one missed sample then split
every run under a 1 s allowance. Gaps are now time without evidence. At full
rate CP3's events are unchanged (1/3/4/2).

Jaywalking, keyframe rate vs CP3 (all four clips re-run through Stage 1):

| clip | CP3 | keyframe | lost | new |
|---|---|---|---|---|
| sample_001 | 1 | 1 | -- | -- |
| sample_002 | 3 | 3 | 02:51.4 (the moped-rider false positive) | 04:55.4 |
| sample_003 | 4 | 3 | 02:51.9 | -- |
| sample_004 | 2 | 2 | -- | -- |

Both differences are threshold-edge cases, not tracking losses. The sample_003
pedestrian stands at the one-body-height kerb margin; CP3 fired on samples
alternating pass/fail, and at 2 samples/s only two isolated samples pass. The
new sample_002 event is a person 1.74 s outside the zebra margin in CP3,
under the 2 s minimum; four keyframe samples count as 2.0 s.
