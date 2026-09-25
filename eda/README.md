# Real-clip EDA and evidence-only CLI

The website evidence in `site_assets/eda/` was built from the four real sample
clips, their Stage 1 perception cache, and the harness output in
`predictions_samples.json`. It is model-derived evidence, not ground truth or
an official accuracy grade. `site_assets/eda/summary.json` records the cached
track and detection statistics; `format_findings.json` records the GOP and
decode measurements. The heatmaps, trajectories, and per-sample object counts
for every clip are in the same directory.

All four clips are 3840x2160, 29.97 fps Sony XAVC MP4 (H.264 High 4:2:2,
10-bit). The measured GOP is fixed at 15 frames, approximately 0.501 seconds,
with one I, four P, and ten B frames. The keyframe-only Stage 1 pass samples
approximately twice per second. These are real source-footage measurements,
not illustrative fixture values.

| Clip | Duration | Frames | Cached track IDs | Detections / Stage 1 sample | Person IDs | Car IDs |
|---|---:|---:|---:|---:|---:|---:|
| sample_001 | 340.3 s | 10,200 | 782 | 38.4 | 312 | 446 |
| sample_002 | 317.8 s | 9,525 | 753 | 39.4 | 313 | 422 |
| sample_003 | 317.8 s | 9,525 | 957 | 43.4 | 427 | 497 |
| sample_004 | 127.6 s | 3,825 | 393 | 40.7 | 172 | 191 |

Track IDs by class are not mutually exclusive because a track can change
class. The four clips have 2,885 unique per-clip track IDs in total; that sum
does not imply identity continuity between recordings. Counts describe the
cache, not detector precision or recall. See `site_assets/eda/dev_error_analysis.json`
for the separate comparison against the team's labels.

The `eda.summary` CLI below remains a bounded, evidence-only tool for an
additional local clip. Without an authorized `--video`, it reports `real data
required` and exits. It does not infer that a file is authentic: new output is
labelled `local_video_provenance_unverified` until the team separately verifies
source and publication rights.

From the repository root, with `opencv-python-headless` installed:

```powershell
python -m eda.summary --video <authorized-clip.mp4>
python -m eda.summary --video <authorized-clip.mp4> --pred <sanitized-predictions.json>
python -m eda.summary --video <authorized-clip.mp4> --pred <sanitized-predictions.json> --tracks <measured-tracks.json>
```

The tool decodes frames to EOF (up to 150,000), reports observed width, height,
FPS, decoded frame count and duration; `reported_frame_count` is null unless
container count agrees with decoded count. Predictions must be sanitized
single-video official JSON. `event_distribution` counts actual predicted
labels, not accuracy or ground truth; it is null when no prediction is given.

Optional track sidecar schema:

```json
{"video":"clip.mp4","points":[{"t_sec":0.0,"track_id":1,"x":0.5,"y":0.5}]}
```

`x` and `y` are measured normalized ground-point coordinates. The tool emits
trajectory points and unique track IDs per five-second bin as hooks for future
plots; this is **not** object density ground truth. No sidecar means null hooks,
not simulated trajectories or density.

The sample videos remain protected camera inputs and are not committed. The
published real-clip summaries above came from the separately run perception
cache and asset-building workflow, not from this CLI's optional track sidecar.
Without annotations, do not infer detection accuracy or failure rates from a
prediction's event counts.
