"""Dev-set error analysis: which events were hit, missed, or false -- and why.

    python tools/error_analysis.py --gt labels/ground_truth.json \
        --pred predictions_samples.json --out site_assets/eda/dev_error_analysis.json

Uses evaluate.py's own tIoU and greedy one-to-one matching (per class, per
video, at each official threshold), then lists every missed and false event
with its best overlap, and a class-agnostic "confusion" where a prediction
overlaps a ground-truth event of a DIFFERENT class (the same moment, the wrong
label).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402


def _match(gt, pred, thr):
    """evaluate.match_segments, but returning WHICH pairs matched."""
    pairs = sorted(((evaluate.tiou(g, p), i, j) for i, g in enumerate(gt)
                    for j, p in enumerate(pred) if evaluate.tiou(g, p) >= thr), reverse=True)
    used_g, used_p, out = set(), set(), []
    for iou, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, iou))
    return out


def analyse(gt: dict, pred: dict, thr: float = 0.5) -> dict:
    per_video, totals = {}, {}
    confusion = []
    for vid, g in gt.items():
        p_events = pred.get(vid, {}).get("events", [])
        classes = sorted({e[2] for e in g["events"]} | {e[2] for e in p_events})
        rows = {}
        for c in classes:
            gs = [(s, e) for s, e, lab in g["events"] if lab == c]
            ps = [(s, e) for s, e, lab in p_events if lab == c]
            m = _match(gs, ps, thr)
            hit_g, hit_p = {i for i, _, _ in m}, {j for _, j, _ in m}
            best = lambda seg, others: max((evaluate.tiou(seg, o) for o in others), default=0.0)  # noqa: E731
            rows[c] = {
                "tp": len(m), "fp": len(ps) - len(m), "fn": len(gs) - len(m),
                "matched": [{"gt": gs[i], "pred": ps[j], "tiou": round(iou, 3)} for i, j, iou in m],
                "missed": [{"gt": gs[i], "best_tiou": round(best(gs[i], ps), 3)}
                           for i in range(len(gs)) if i not in hit_g],
                "false": [{"pred": ps[j], "best_tiou": round(best(ps[j], gs), 3)}
                          for j in range(len(ps)) if j not in hit_p],
            }
            t = totals.setdefault(c, {"tp": 0, "fp": 0, "fn": 0})
            for k in ("tp", "fp", "fn"):
                t[k] += rows[c][k]
        for ps, pe, pl in p_events:
            for gs_, ge, gl in g["events"]:
                if gl != pl and evaluate.tiou((gs_, ge), (ps, pe)) >= 0.3:
                    confusion.append({"video": vid, "gt": [gs_, ge, gl], "pred": [ps, pe, pl]})
        per_video[vid] = rows
    for c, t in totals.items():
        t.update(evaluate.prf(t["tp"], t["fp"], t["fn"]))
    return {"tiou_threshold": thr, "per_class": totals, "per_video": per_video,
            "confusion_other_class_overlap": confusion}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--pred", default=ROOT / "predictions_samples.json", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tiou", type=float, default=0.5)
    args = ap.parse_args()
    gt = json.loads(args.gt.read_text())
    pred = json.loads(args.pred.read_text())["videos"]
    rep = analyse(gt, pred, args.tiou)
    rep["gt_file"] = str(args.gt.name)
    rep["full_report"] = evaluate.evaluate(gt, {"videos": pred}, per_video=True)
    text = json.dumps(rep, indent=1, default=float)
    if args.out:
        args.out.write_text(text)
    for c, t in rep["per_class"].items():
        print(f"{c:<18} TP {t['tp']:3d} FP {t['fp']:3d} FN {t['fn']:3d}  "
              f"P {t['precision']:.2f} R {t['recall']:.2f} F1 {t['f1']:.2f}  (tIoU {args.tiou})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
