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
The disconnected demo checks declared MP4 duration, **not** actual decoded
duration. This localhost foundation is not production hardened; see
[integration notes](../INTEGRATION.md).
