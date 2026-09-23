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
EDA as measured evidence. The existing root `predictions_samples.json` was
generated from a synthetic clip and is also not presented as real footage.

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

- Organizer camera video(s) and scene geometry, with redistribution permission.
- Real sample predictions from the official harness and validation output.
- Genuine annotated sample assets or tracks for overlays.
- Measured EDA, team names/roles, technical report findings, weights link and
  final repository/prediction links.

None of those are invented in the site. `web/README.md` and
`demo_api/README.md` detail each local component.
