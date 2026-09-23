# Website and demo integration (Phase 1)

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
warning, as the evaluator does. Extra harness fields such as `log` are ignored.
The website additionally rejects path-like video keys, unsafe object keys
(`__proto__`, `constructor`, `prototype`), and non-finite values.
The organizer JSON does **not** have a provenance field: the caller supplies
`fixture`, `sample`, or `upload` source metadata separately.

`web/src/fixtures/illustrative_predictions.json` is invented **interface test
data**, not inference or a real camera clip. It is never shown in Results or
EDA as measured evidence. The existing root `predictions_samples.json` was
generated from a synthetic clip and is also not presented as real footage.

## Run locally

Use Node.js 20.19+ and Python 3.10+ from the repository root:

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
python -m unittest demo_api.test_jobs -v
python evaluate.py --pred predictions_samples.json --validate-only
```

## Adapter seam for Claude's model

The API accepts a `PredictionAdapter` implementation whose
`predict(video_path: Path, filename: str) -> dict` returns the official
prediction document. Connecting one also **requires** an independent
`DurationVerifier.duration_seconds(video_path)` that verifies actual duration
from decoded frames. `JobStore(adapter=..., duration_verifier=...)` refuses to
start otherwise. The verifier runs before the adapter, so header metadata
alone cannot authorize model execution. No decoded-frame verifier is shipped
in this phase: adding one correctly depends on the eventual video runtime.

A future adapter should invoke the *unchanged* organizer runner and parse its
output, maintaining the harness's timing and output semantics. Do not call
separate ad-hoc model methods in the website. Before a job can complete,
`demo_api/validation.py` checks the complete event/risk contract and allows
only `team` and the single uploaded video. Extra fields, including harness
`log`, are **rejected**, not returned. The adapter must extract only the
prediction fields, never forward diagnostic logs or paths. The frontend API
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
Results and statuses are held in memory only. The server cannot forcibly stop
a hung future adapter thread; that adapter must enforce its own execution
deadline. API responses contain neither storage paths nor model exceptions.
The service binds to localhost and must not be exposed directly to the public
Internet in its current form.

The UI currently shows upload activity and queued/running/completed/failed or
awaiting-model state, not a fabricated percentage. Real process-level progress
requires a model-side signal from Claude's implementation.

## Evidence still required

- Organizer camera video(s) and scene geometry, with redistribution permission.
- Real sample predictions from the official harness and validation output.
- Genuine annotated sample assets or tracks for overlays.
- Measured EDA, team names/roles, technical report findings, weights link and
  final repository/prediction links.

None of those are invented in the site. `web/README.md` and
`demo_api/README.md` detail each local component.
