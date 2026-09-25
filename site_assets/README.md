# site_assets/ -- results and EDA for the team website

Built by `python tools/build_site_assets.py` from the four real sample clips
(sample_001-004: one 4K intersection camera, 2-6 min each), the perception
cache and `predictions_samples.json`. Everything here is OUR output on the
sample clips; nothing is an official grade. Times are in seconds from the
start of each clip; clip names match `predictions_samples.json`.

`team.json` contains the member names and profile links published in the
project README. It is the website's Team section data source.

## Per clip (`<clip>` = sample_001 ... sample_004)

| file | what it is |
|---|---|
| `videos/<clip>_annotated.mp4` | Web-sized (854 px, H.264, 15 fps, real-time) playback: tracked boxes (green people, orange vehicles), crossings (cyan) and signal queue zones (grey) aligned to this clip's camera pose, a banner while an event is active, and the risk score as a bar (red above the 0.5 alarm threshold). Same drawing as the live demo. |
| `events/<clip>_events.json` | `{"clip", "duration_sec", "events": [[start, end, label], ...]}` -- exactly the submission's Part A output. |
| `risk/<clip>_risk.json` | `{"clip", "alarm_threshold": 0.5, "risk": [[t, score], ...]}` -- the Part B curve, downsampled to 5 Hz from the per-frame harness output. |
| `event_clips/<clip>_event_NN_<label>_<start>s.mp4` | One clip per detected event, cut from the annotated video with 1.5 s either side. |

## Failure cases (`failures/`)

`failures.json` lists each case with `title`, `what` (what goes wrong and why)
and `status` (what we did about it, honestly). `failures/<id>.mp4` is the
annotated clip.

* `moped_rider` -- a rider flagged as a jaywalker (CP3); why it no longer
  fires, and why it still could.
* `far_kerb_pedestrians` -- people ~30 px tall at the far kerb: too small for
  the kerb-margin test either way.
* `platoon_aliasing` -- at 2 samples/s a platoon of cars makes the tracker drift
  backwards; why wrong_way/congestion are off at keyframe rate.

## EDA (`eda/`)

| file | what it is |
|---|---|
| `<clip>_object_counts.json` | `{"t": [...], "person": [...], "car": [...], ...}`: tracked objects per class at every Stage 1 sample (every 0.5 s). |
| `<clip>_density.json` | `{"t": [...], "southbound": [...], "northbound": [...], "other": [...]}`: vehicles inside each direction's lanes per sample -- traffic density over time. |
| `<clip>_motion_heatmap.jpg` | Where things MOVE: ground points of objects moving faster than half a body height per second, accumulated over the clip, over a real frame. |
| `<clip>_trajectories_lanes.jpg` | Every track's path (vehicles coloured by heading, people white) with the authored lanes and their direction arrows. The lane arrows were inferred from optical flow; the trajectories are the check. |
| `summary.json` | Per clip: duration, number of tracks, detections per sample, tracks per class, the camera pose the zones were aligned with. |
| `format_findings.json` | The footage format, the GOP, the measured decode costs and budget, and the pipeline decisions each one forced. |
| `dev_error_analysis.json` | (after labelling) per-video TP/FP/FN, misses and false positives against the team's labels. |
