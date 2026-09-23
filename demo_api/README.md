# Local demo API foundation

Run from the repository root with `python -m demo_api.app`. Standard library
only. Routes:

| Method | Path | Result |
| --- | --- | --- |
| GET | `/api/health` | Service and adapter availability |
| POST | `/api/jobs` | Raw `video/mp4` bytes, URI-encoded `X-File-Name`; returns job ID/status |
| GET | `/api/jobs/{id}` | Safe job status and message |
| GET | `/api/jobs/{id}/result` | Official prediction JSON after completion; otherwise 409 |

Without an injected `PredictionAdapter`, accepted jobs remain
`awaiting_model` and their uploaded bytes are deleted immediately. A future
adapter must be paired with an independent decoded-frame `DurationVerifier`;
`JobStore` refuses model execution without one. Its result is validated and
sanitized before completion: only official events/risk for the uploaded video
are returned, never the harness log or local paths. The service streams at
most 100 MiB in 64 KiB chunks, limits active uploads/jobs to two, expires
records after 15 minutes, and cleans files after completion/failure/expiry.
The normal server startup remains disconnected. It checks declared MP4
duration for upload acceptance, but does not run a decoder or model on an
accepted upload. `build_store(adapter)` automatically pairs a future adapter
with `DecodedDurationVerifier`: OpenCV decodes frames to EOF in a separate
process, checks advancing presentation timestamps and decoded frame count,
enforces 120 seconds, and fails closed if verification cannot complete within
45 seconds. Container movie duration alone cannot authorize inference.

`HarnessAdapter` is available but not enabled by default. If explicitly
injected later, it runs the unchanged `run_submission.py` with fixed arguments,
no shell, and a 390-second timeout. It discards the harness `log`, rejects
duplicate/unsafe JSON keys and malformed predictions, and returns only the
existing sanitized frontend contract. Harness-reported errors fail the job
instead of being displayed as genuine no-event results. A process killed on
timeout may still leave child processes it created; this localhost/demo boundary is not a
public-deployment security guarantee. See [integration notes](../INTEGRATION.md).

For a future authorized website sample, `python -m demo_api.sample_export
--pred <raw-harness.json> --video <exact-file.mp4> --out <new-file.json>`
uses the same harness-result sanitizer without running the model. It fails on
harness errors or invalid predictions, removes `log`, and will not overwrite an
existing output. The operator must separately verify that the video is real,
authorized, and publishable before adding it to the website catalog.
