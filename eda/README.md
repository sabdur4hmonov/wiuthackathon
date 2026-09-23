# Evidence-only EDA contract

No camera EDA numbers or charts are bundled. Without an authorized `--video`,
the tool reports `real data required` and exits. It does not infer that a file
is authentic: output is labelled `local_video_provenance_unverified` until the
team separately verifies source and publication rights.

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
