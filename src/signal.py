"""Traffic-light state from an ROI crop: red / amber / green / unknown.

This unlocks red_light, stop_line and failure_to_yield later. It is built now
because it costs nothing now and is on the critical path for three classes.

=============================================================================
THE DESIGN RULE: RETURN `unknown` RATHER THAN GUESS
=============================================================================
A wrongly-emitted class is a guaranteed zero in macro F1 -- it scores 0 for that
class AND enlarges the denominator. The three rules this feeds are all of the
form "vehicle did X while the signal was red", so a signal misread turns lawful
traffic into a stream of violations. There is no symmetric upside: a missed red
costs one FN, a false red costs many FPs.

So the classifier is deliberately reluctant. A crop must clear three separate
bars before it gets an answer:

  1. enough LIT pixels at all (min_lit_pixels) -- an unlit, blown-out or
     badly-cropped ROI is unknown, not "whatever colour is least dark";
  2. the winning colour must hold a majority of those lit pixels
     (min_winner_fraction);
  3. and beat the runner-up by a margin (min_winner_ratio), so an amber/red
     ambiguity resolves to unknown instead of a coin flip.

=============================================================================
AND THE WHOLE PATH MUST DEGRADE TO "NO SIGNAL AVAILABLE"
=============================================================================
We do not know whether a traffic light is even visible in this camera's frame:
the starter kit has no camera.md, and config/zones.json ships with an EMPTY
traffic_lights list being a legal, meaningful state ("no signal in view").
SignalReader.available is False in that case and every read returns UNKNOWN, so
a downstream rule written against this API simply never fires rather than
breaking.

Downstream rules must treat UNKNOWN as "do not emit". `is_red()` returns a
tri-state (True / False / None) precisely so that "not red" and "don't know"
cannot be confused by an `if not ...` written in haste.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .thresholds import TH, SignalThresholds

RED = "red"
AMBER = "amber"
GREEN = "green"
UNKNOWN = "unknown"
STATES = (RED, AMBER, GREEN, UNKNOWN)


@dataclass(frozen=True)
class SignalState:
    """One classification, with enough detail to debug a bad call."""

    state: str = UNKNOWN
    confidence: float = 0.0
    lit_pixels: int = 0
    counts: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.state != UNKNOWN

    def __bool__(self) -> bool:
        # Deliberately NOT "is the light on": an UNKNOWN state is falsey so a
        # careless `if state:` fails safe (emits nothing) rather than unsafe.
        return self.known


def classify_crop(crop: np.ndarray,
                  th: SignalThresholds | None = None) -> SignalState:
    """Classify a BGR uint8 crop of a traffic-light ROI.

    Robust to a None or empty crop: those are UNKNOWN, not an exception. This
    runs inside Stage 2 and inside the risk estimator's reach, and neither can
    afford to raise.
    """
    th = th or TH.signal
    if crop is None:
        return SignalState(reason="no crop")
    arr = np.asarray(crop)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        return SignalState(reason=f"bad crop shape {getattr(arr, 'shape', None)}")

    try:
        import cv2

        hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    except Exception as e:  # cv2 missing, or a crop it cannot convert
        return SignalState(reason=f"hsv conversion failed: {e!r}")

    h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1], hsv[..., 2]
    lit = (s >= th.min_saturation) & (v >= th.min_value)
    n_lit = int(lit.sum())
    if n_lit < th.min_lit_pixels:
        return SignalState(lit_pixels=n_lit,
                           reason=f"only {n_lit} lit pixels "
                                  f"(need {th.min_lit_pixels})")

    def band(lo_hi) -> np.ndarray:
        lo, hi = lo_hi
        return (h >= lo) & (h <= hi)

    counts = {
        RED: int((lit & (band(th.red_hue_lo) | band(th.red_hue_hi))).sum()),
        AMBER: int((lit & band(th.amber_hue)).sum()),
        GREEN: int((lit & band(th.green_hue)).sum()),
    }
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    (best, best_n), (_, second_n) = ranked[0], ranked[1]

    if best_n == 0:
        return SignalState(lit_pixels=n_lit, counts=counts,
                           reason="no lit pixel fell in a known hue band")

    fraction = best_n / n_lit
    ratio = best_n / max(second_n, 1)
    if fraction < th.min_winner_fraction:
        return SignalState(lit_pixels=n_lit, counts=counts,
                           reason=f"{best} holds only {fraction:.2f} of lit "
                                  f"pixels (need {th.min_winner_fraction})")
    if second_n and ratio < th.min_winner_ratio:
        return SignalState(lit_pixels=n_lit, counts=counts,
                           reason=f"{best} beats runner-up by only {ratio:.2f}x "
                                  f"(need {th.min_winner_ratio}x)")

    return SignalState(state=best, confidence=float(fraction),
                       lit_pixels=n_lit, counts=counts, reason="ok")


class SignalReader:
    """Reads every traffic-light ROI a Zones object declares.

    Degrades cleanly in two ways that both matter here:
      * zones is None, or declares no traffic lights -> `available` is False and
        every read is UNKNOWN. That is the expected state until someone
        confirms a signal is visible in this camera's frame.
      * a crop falls outside the frame, or the frame is malformed -> UNKNOWN for
        that light, no exception.
    """

    def __init__(self, zones, th: SignalThresholds | None = None) -> None:
        self.zones = zones
        self.th = th or TH.signal
        self.lights = tuple(getattr(zones, "traffic_lights", ()) or ())

    @property
    def available(self) -> bool:
        """False means: this scene has no signal we can read. Not an error."""
        return bool(self.lights)

    def crop(self, frame: np.ndarray, light) -> np.ndarray | None:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim != 3:
            return None
        height, width = arr.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in light.roi)
        x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return arr[y1:y2, x1:x2]

    def read(self, frame: np.ndarray) -> dict[str, SignalState]:
        """light id -> SignalState. Empty dict when no signal is available."""
        out: dict[str, SignalState] = {}
        for light in self.lights:
            try:
                out[light.id] = classify_crop(self.crop(frame, light), self.th)
            except Exception as e:  # noqa: BLE001 - never raise into a rule
                out[light.id] = SignalState(reason=f"read failed: {e!r}")
        return out

    def state_for_lane(self, frame: np.ndarray, lane_id: str) -> SignalState:
        """The state of whichever signal controls `lane_id`.

        UNKNOWN when no signal claims that lane, or when the signals that do
        disagree -- a contradiction is exactly the case where guessing is worst.
        """
        relevant = [lt for lt in self.lights if lane_id in lt.controls_lanes]
        if not relevant:
            return SignalState(reason=f"no signal controls lane {lane_id!r}")
        states = [classify_crop(self.crop(frame, lt), self.th) for lt in relevant]
        known = [st for st in states if st.known]
        if not known:
            return SignalState(reason="no controlling signal could be read")
        distinct = {st.state for st in known}
        if len(distinct) > 1:
            return SignalState(reason=f"controlling signals disagree: "
                                      f"{sorted(distinct)}")
        return max(known, key=lambda st: st.confidence)

    def is_red(self, frame: np.ndarray, lane_id: str) -> bool | None:
        """True / False / None. None means 'do not emit'.

        Tri-state on purpose: a plain bool would let `if not reader.is_red(...)`
        silently treat an unreadable signal as a green light.
        """
        st = self.state_for_lane(frame, lane_id)
        if not st.known:
            return None
        return st.state == RED
