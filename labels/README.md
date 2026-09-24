# Labels for the four real clips

**Nothing here is the official grade.** No ground truth for sample_001-004 was
supplied; everything in this folder is ours, for internal self-scoring.

## `gt_approx_cp3.json` -- APPROXIMATE, circular

The 10 jaywalking events the CP3 full-rate pipeline (stride 2) produced,
eyeballed on a thumbnail montage, minus one rejected there (sample_002 02:51,
a moped rider the detector did not box as a motorcycle): 9 events.

* It is our own rule's output, so scoring the pipeline against it measures
  agreement between two versions of the pipeline, not accuracy. Its
  boundaries are the rule's boundaries, which flatters tIoU 0.7.
* Only jaywalking was ever looked at. stopped_vehicle, wrong_way,
  congestion and every other class are UNLABELLED here, not absent.
* Every video entry carries `"approximate": true` and a `source` note.

Score of `predictions_samples.json` (keyframe pipeline, 2026-09-24) against it:

| | F1@0.3 | F1@0.5 | F1@0.7 | TP/FP/FN @0.5 |
|---|---|---|---|---|
| jaywalking | 0.889 | 0.778 | 0.778 | 7 / 2 / 2 |

Score A 0.815. The 2 FP: sample_002 04:55 (new at keyframe rate) and
sample_002 02:21 (bounds 1.5 s off CP3's: tIoU 0.43). The 2 FN: that same
02:20 event, and sample_003 02:52 (lost at keyframe rate).

## Getting a real ground truth (~30-40 min for all four clips)

```bash
python tools/review_candidates.py samples/sample_00*.mp4 --seed labels/gt_approx_cp3.json --out labels/review --scan --proxies
```

then open `labels/review/index.html` in a browser (straight from disk).
Each candidate is a short clip with the source time burned in:

* **predicted / seed**: what the pipeline or CP3 reported.
* **recall probe**: near misses the pipeline did NOT report (the same rules
  with thresholds loosened). Confirming one records a miss.

Per candidate: **Y** correct, **N** reject, or edit start / end / class and
**F** (`[` / `]` copy the playing time into start / end). "Add a missed event"
takes anything else seen. **Export** writes `ground_truth.json` in
evaluate.py's shape; decisions autosave in the browser and an export can be
re-imported. Then:

```bash
python evaluate.py --pred predictions_samples.json --gt ground_truth.json --per-video
```

### Scan windows: stopped_vehicle, wrong_way, congestion (`--scan`)

Those three classes fire nothing on the sample clips, so they have no
candidates -- a miss there is invisible. `--scan` adds windows flagged by
`tools/scan_windows.py`: places a person should LOOK, deliberately far looser
than the rules (it is a search aid, not a detector: nothing in src/ imports
it, it registers no rule, and tests/test_scan_windows.py pins that).

* **stationary**: a vehicle near-stationary >= 5 s anywhere on the road (the
  rule wants 10 s, outside queue zones, off the kerb); outside-queue stops
  rank first. Mostly parked, kerb or bus-stop stops -- deciding which, if
  any, is a `stopped_vehicle` is the human's call.
* **direction**: the wrong_way rule, loosened and with its keyframe-rate gate
  lifted. Most will be platoon ALIASING (the card says so): check the traffic.
* **slow traffic**: the congestion rule, loosened the same way.

Capped at 6 / 4 / 3 per clip; overlapping windows merge; windows over 20 s
play sped up (2-4x, shown on the card; start/end still convert exactly).

### How long a full pass takes

`--scan` gives 61 cards over the four clips: 10 reported events, 22 recall
probes (jaywalking), 29 scan windows (10-11 per clip before merging). Their
clips total 10.4 minutes of playback. At ~30 s per card to watch and decide,
plus ~1 min each to set tight bounds on the ~10-15 that turn out real:
**about 40-45 minutes** for all four clips.

That covers what the tools can point at. Classes nothing flags (red_light,
near_miss, accident, illegal turns ...) need the full-length proxies: 18.8
minutes of footage, ~10 minutes at 2x plus pauses, **another 20-25 minutes**.
Watching all four clips raw instead, for all three classes at once across a
wide 4K scene, realistically takes 3-4x the footage length (60-80 minutes)
with more misses.

Candidates only cover what the rules can see. For classes no rule looks at,
watch the full-length proxies (`labels/review/media/*_proxy.mp4`, written by
`--proxies`) in `tools/label.html`: the 4K source does not play in a browser.

`labels/review/` is generated (clips, ~100 MB with proxies) and not committed.
