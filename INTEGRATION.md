# Website and demo integration

This is the Codex-owned presentation/integration layer. It does not alter
`solution.py`, `src/`, `config/zones.json`, model weights, `run_submission.py`,
or `evaluate.py`. The official harness remains the submission path.

## Data boundary

The website consumes the official prediction shape:

```json
{"team":"team-name","videos":{"clip.mp4":{"events":[[12.4,18.9,"accident"]],"risk":[[0.0,0.01]]}}}
```

`events` contain `[start_sec, end_sec, official_label]`; `risk` contains
`[t_sec, score]`. The parser in `web/src/lib/predictions.ts` checks the 14
official labels, interval ordering and same-class overlap, risk monotonicity,
finite numeric values and score bounds. It tolerates absent `risk` with a
warning, as the evaluator does. The live API strips harness `log` **before**
responding. Published sample JSON must likewise be exported without logs;
downloading raw harness JSON into the browser would expose diagnostics even if
the parser ignored that field.
The website additionally rejects path-like video keys, unsafe object keys
(`__proto__`, `constructor`, `prototype`), and non-finite values.
The organizer JSON does **not** have a provenance field: the caller supplies
`fixture`, `sample`, or `upload` source metadata separately.

`web/src/fixtures/illustrative_predictions.json` is invented **interface test
data**, not inference or a real camera clip. It is never shown in Results or
EDA as measured evidence. The root `predictions_samples.json` now contains a
synthetic clip **and** two locally held camera clips from the ML owner's
concurrent commit. It includes private harness logs and is not loaded by the
website or treated as an accuracy result.

The Results section loads `web/public/samples/catalog.json`, currently an empty
list. For a future authorized sample, run the unchanged harness on **one**
video, check its errors and the official evaluator, then export only its
sanitized prediction fields:

```powershell
python -m demo_api.sample_export --pred <raw-harness.json> --video <exact-file.mp4> --out <new-sanitized.json>
python evaluate.py --pred <new-sanitized.json> --validate-only
```

Only after verifying camera provenance and redistribution permission, place
the sanitized JSON (and optionally its permitted MP4) under `web/public/samples/`
and add a catalog entry with `id`, `label`, `filename`, `predictionUrl`, and
optional `videoUrl`. URLs must be same-origin `/samples/` paths. The sample
loader rejects extra fields, wrong video keys, invalid event/risk data, unsafe
paths and raw logs; it labels accepted data `VALIDATED REAL SAMPLE`. An empty
catalog shows a clear no-results state, not a synthetic chart.

## Run locally

Use Node.js 20.19+ and Python 3.10+ from the repository root. On a clean
machine, install Python and Node first. The disconnected API uses only the
Python standard library; decoded-duration verification and EDA require
`numpy` and `opencv-python-headless` from `requirements.txt`. Full ML setup
also requires the remaining packages and weights, which are not installed by
the website instructions.

```powershell
cd web
npm ci
npm run dev
```

In a second terminal:

```powershell
python -m demo_api.app
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to the Python server at
`127.0.0.1:8765`. The API binds to localhost only. With no adapter, uploads
are accepted and the job reports `awaiting_model`; no inference is claimed.

Validation:

```powershell
cd web
npm test
npm run build
cd ..
python -m pytest demo_api
python evaluate.py --pred predictions_samples.json --validate-only
```

## Adapter seam for Claude's model

The API accepts a `PredictionAdapter` implementation whose
`predict(video_path: Path, filename: str) -> dict` returns the official
prediction document. Connecting one also **requires** an independent
`DurationVerifier.duration_seconds(video_path)` that verifies actual duration
from decoded frames. `JobStore(adapter=..., duration_verifier=...)` refuses to
start otherwise. `build_store(adapter)` supplies `DecodedDurationVerifier`
automatically; normal `python -m demo_api.app` still calls `build_store()`
without an adapter. The verifier uses the repository's OpenCV dependency to
decode every frame to EOF in a killable subprocess (45-second deadline),
cross-checks decoded frame count and advancing presentation timestamps, and
fails closed on missing/inconsistent data or duration over 120 seconds. This
does not rely on MP4 movie duration alone. Unusual codecs whose timestamps
OpenCV cannot verify will be rejected, and an attacker able to forge both
decoded timestamps and frame rate may still defeat time inference; this is a
localhost/demo guard, not a media-forensics guarantee.

A `HarnessAdapter` now implements that seam but is intentionally **not
injected** into normal startup: real model connection awaits real footage,
authored zones, installed dependencies/weights, and an explicit integration
decision. When injected, it invokes the *unchanged* organizer runner in a
separate process with a 390-second timeout, fixed argument list and no shell.
It suppresses subprocess output, rejects malformed/duplicate/unsafe JSON
keys, and treats harness-reported errors as failed jobs. It strips the runner's
`log`, maps the private upload filename to the visitor's filename, and validates
the final result. Do not call separate
ad-hoc model methods in the website. Before a job can complete,
`demo_api/validation.py` checks the complete event/risk contract and allows
only `team` and the single uploaded video. Extra fields, including harness
`log`, are **rejected**, not returned. The adapter extracts only prediction
fields; no diagnostic log or local path reaches the frontend. The frontend API
and visualizations need no rewrite.

Optional tracking boxes are a **separate** sidecar, never fields added to
official predictions. `AnnotatedPlayer` consumes normalized `TrackBox` values
only if genuinely generated. For letterboxed video, a future adapter must map
coordinates into the displayed image rectangle before overlaying boxes.

## Upload limits and privacy

The local demo accepts only MP4 with valid `ftyp` and `mvhd` headers, at most
100 MiB and a declared duration of at most 120 seconds. This header check is
**not** a decoded-duration guarantee. With no model connected it permits an
upload solely for the interface demo; uploaded bytes are immediately removed.
With a model connected the mandatory verifier must independently confirm a
decoded duration within 120 seconds before inference. The service streams
uploads in 64 KiB chunks with a 10-second socket timeout and 60-second total
upload deadline. At most two uploads/model jobs are active and 32 job records
are retained. A janitor expires jobs after 15 minutes; uploaded files are
removed on completion/failure or expiry (with retry for transient file locks).
Results and statuses are held in memory only. The optional harness adapter
has a killable process deadline, so a hung model process does not hang its
API worker indefinitely. The generic adapter protocol does not make arbitrary
third-party adapters killable; use the subprocess adapter when connecting a
model. Child processes spawned by the harness may need additional process-tree
isolation for a public deployment. API responses contain neither storage paths
nor model exceptions. The service binds to localhost and must not be exposed
directly to the public Internet in its current form.

The UI currently shows upload activity and queued/running/completed/failed or
awaiting-model state, not a fabricated percentage. Real process-level progress
requires a model-side signal from Claude's implementation.
Each upload run retains its submitted `File`; later file selection affects only
the next upload. Old polls and upload requests are aborted and run-ID-guarded.
Browser video metadata, when finite and positive, supplies the real timeline
duration; otherwise views retain their prior safe prediction-based fallback.

## EDA and offline constraints

`python -m eda.summary` without a video reports `real data required`. With an
authorized video it decodes observed metadata; optional sanitized predictions
and measured track sidecars add event counts, trajectories and per-bin unique
track counts. Outputs are labelled provenance-unverified until the team checks
the source. See `eda/README.md`; no real charts or measurements are bundled.

Cache npm/Python dependencies and model weights **before** running on an
offline evaluation machine. This repo does not require paid APIs, but a fresh
`npm ci`, Python package install, or `weights/download.sh` needs network access.
Verify an offline run separately using the repository's offline check once the
ML side and authorized video are ready. Do not publish model scores until the
official evaluator has been run against appropriate ground truth.

## Evidence still required

- Confirmed camera-footage provenance/publication rights and authored scene geometry.
- Permission-cleared, sanitized real-sample predictions and ground-truth-based
  evaluation when available.
- Genuine annotated sample assets or tracks for overlays.
- Measured EDA, team names/roles, technical report findings, weights link and
  final repository/prediction links.

None of those are invented in the site. `web/README.md` and
`demo_api/README.md` detail each local component.

## Phase 3 real-data checkpoint (2026-09-23)

The ignored `samples/sample_001.mp4` and `samples/sample_002.mp4` are present
locally and described as real-camera clips in the ML-owned README. They are
protected inputs, not website assets. A metadata-only FFmpeg probe reported
approximately 340.34 s and 317.82 s respectively, both 3840×2160 at 29.97
fps. These are container-reported values, **not** decoded-duration or
provenance verification. Neither clip has been copied, changed, staged, or
published. Their publication rights have not been established. No ground-truth
annotations or validated real predictions were present at that checkpoint.

The official path for a future local run is:

```powershell
python run_submission.py --videos samples/sample_001.mp4 --out <private-output.json> --team wiut-cv
```

The runner imports the unchanged `solution.py`, calls
`detect_events(str(path))`, then
calls `RiskEstimator.reset` with video ID/FPS/size/frame count and `step` on
each decoded frame at `frame_index / fps`. It writes `team`, a filename-keyed
`videos` map of event intervals and risk points, plus a private `log`. The
runner may exit successfully while recording per-video errors and replacing a
timed-out result with empty arrays; the log must therefore be checked before
format validation. Its nominal budget is 3× the video duration, measured by
the runner's video metadata. Use the unchanged `evaluate.py --validate-only`
for format validation, then `demo_api.sample_export` to strip the log only
after confirming an error-free result. Accuracy requires actual ground truth;
format validation is not accuracy validation.

No official real-video run was completed in this checkpoint. The repository's
Python 3.11 virtual environment refers to a base interpreter that this
execution environment denies launching (`Access is denied`). The bundled
Python 3.12 runtime lacks OpenCV and Ultralytics and cannot use the virtual
environment's CPython 3.11 native packages. The local YOLO weights file is
present, but a runnable environment and a full 3×-budget measurement remain
unverified here. Independently, `config/zones.json` still contains null scene
geometry; the registered event rules return nothing without authored zones.
The current risk estimator stays at its quiet default. These ML-side blockers
belong to the ML owner and have not been altered by the website work.

Both real clips also exceed the localhost demo's 100 MiB / 120-second upload
limits. They must be run through the unchanged official harness offline, not
sent through the current demo API. The sample catalog remains empty. Once the
ML owner supplies authored zones and a working local environment, run one
authorized clip with the official harness, inspect its log and runtime,
validate the exact filename/events/risk with `evaluate.py --validate-only`,
then export a log-free JSON result. Only after separately confirming footage
provenance and publication rights should that result be added to the catalog.
No raw footage or harness diagnostics belong in the website.

## Phase 4 local execution checkpoint (2026-09-24)

**Verified environment.** The existing `.venv` uses Python 3.11.9 but its
base interpreter is denied execution in this Codex sandbox; `py -0p` reports
no registered Python. An accessible Python 3.12.14 was used to create a
separate, gitignored `venv_phase4/` without changing `.venv`. The local
runtime has OpenCV 5.0.0 (`opencv-python`), NumPy 2.5.2, CPU-only PyTorch
2.14.0, torchvision 0.29.0, Ultralytics 8.4.160, SciPy 1.18.1, and `lap`
0.5.13. `torch.cuda.is_available()` is false. Ultralytics needs a writable
config directory; set `YOLO_CONFIG_DIR` to the local venv before import/run.
The default OpenCV decode path does not need the optional `imageio-ffmpeg`.
On a machine with an accessible Python 3.12 on PATH, the isolated setup is:

```powershell
python -m venv venv_phase4
.\venv_phase4\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\venv_phase4\Scripts\python.exe -m pip install ultralytics scipy pytest lap
$env:YOLO_CONFIG_DIR = (Resolve-Path .\venv_phase4).Path
```

This local setup is not a GPU performance endorsement. `lap` was required by
the installed Ultralytics tracker even though `requirements.txt` describes a
SciPy fallback; the ML owner should reconcile that deployment dependency.
No system Python, weights, or ML implementation was changed.

**Real clip attempt, not a completed result.** OpenCV opened
`samples/sample_001.mp4` and decoded its first 3840×2160 frame. It reported
10,200 frames at 29.97002997 fps, or 340.34 s from header metadata; full
decoded duration was not independently checked. The unchanged command was
`venv_phase4\Scripts\python.exe run_submission.py --videos
samples/sample_001.mp4 --out predictions_phase4_real.json --team wiut-cv`
with default flags. The runner's budget was 1,021 s. Part A reported 39.44 s
and failed at the first tracker call because `lap` was then missing. Its short
Part B probe estimated a 5,021.7 s reserve (14.755× video duration), far
above the whole budget. The attempt was interrupted rather than spending the
remaining budget on an invalid result: exit code 1, no prediction JSON, no
event/risk counts, and no validated real output **from this attempt**. The interrupt prevented a
reliable total elapsed-time measurement. After `lap` was installed, tracker
import succeeded; a second full real-clip run was not attempted because the
measured probe still predicts an over-budget CPU run. That projection is not
an observed full-run time. Peak memory was not measured.

**Separate ML/data blockers.** The unchanged zone validator reports
`config/zones.json: NOT AUTHORED` with 18 problems. The current registered
event rules cannot use missing scene geometry, and the risk estimator remains
at its quiet default. No ground-truth labels or publication clearance are
available. None of these is repaired or inferred by the website layer. The
sample catalog stays empty, and no real evidence artifact is published.

**Checks.** Frontend: 13 tests and production build passed. The official
validator accepted the shipped mixed synthetic/camera `predictions_samples.json`; this is
format validation only. On the new CPU environment, the focused harness test
suite had 4 performance-related failures: a tiny synthetic clip took 12.8 s
against a 4.8 s budget, so the unchanged runner discarded its events/risk.
The demo API suite had 21 passes and one Windows temporary-directory lock
failure in its timeout test (the test passes in isolation). These failures
remain visible; no tests or ML code were changed to hide them.

**Concurrent ML-owner result, checked read-only.** While this Phase 4 work was
in progress, HEAD advanced independently to `0b44db2` (the only committed
file was the protected `predictions_samples.json`). For `sample_001.mp4`, that
file's harness log reports 340.34 s of footage, 786.4 s total runtime against
a 1,021.0 s budget (reported ratio 2.3106×), zero harness errors, zero events,
and 10,200 risk points, all with score 0. The unchanged official validator
and the existing single-video sanitizer accept its event/risk structure. This
is format/log validation of a committed ML-owner artifact, **not** an
independently reproduced run, accuracy measurement, or publication clearance.
The raw file contains harness logs and must never be linked from the website.
`evidence/sample_001.metadata.json` records only safe counts/status and the
source file fingerprint for handoff. It deliberately omits video and raw logs.
The public sample catalog remains empty until provenance and redistribution
rights are confirmed and a log-free prediction is explicitly approved.

## Phase 6 current checkpoint (2026-09-24)

This checkpoint supersedes earlier statements that scene geometry was still
being developed; the historical Phase 3 and Phase 4 notes above are retained
as records of those earlier states. Phase 6 geometry was authored against a
3840×2160 authoring frame at frame 3400 / 113.4467s. The zones validator
passed with zero errors. The checkpoint inventory recorded 12 lanes, 1 stop
line, 3 crossings, 1 queue zone, 1 signal, and 7 markings. This validator
result establishes geometry/schema validity only. It does not establish
detection accuracy. The backend working tree, including its separately
protected current four queue zones, is outside the website ownership boundary
and was not changed by this checkpoint documentation.

The Phase 6 run did not establish full-video perception coverage. Part A
stopped early because the Part B reserve estimate became approximately 3452s;
the Part A hard limit became 0s. Actual Part A coverage was approximately 0.6s
/ 10 processed frames. The total observed runtime was 1003.2s / 340.34s =
2.9476x, including Part B at 954.3s. The recorded risk output was all zero.
That is recorded model output, not proof that the scene was safe and not an
accuracy claim. An approximately 22 px camera-pose offset was observed during
the first approximately 30s.

The prediction/schema format was validated, but format-valid output is not a
meaningful full-video evaluation. Empty events mean only that no events are
present in the supplied prediction; they do not mean the entire video was
checked or that no events occurred. No accuracy claim is made without verified
ground truth, and full-video coverage remains unverified/partial for this run.

The website now reads the existing `/api/health` response and reports model
adapter availability only from its explicit `model_connected` boolean. It
distinguishes adapter available, adapter unavailable/disconnected, and health
status unknown. It also labels fixture or catalog data as supplied prediction
data viewed without live model execution. Sample labels say prediction
format/schema validated and explicitly do not imply detection accuracy.
