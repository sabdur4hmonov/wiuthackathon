"""An invented four-way intersection, in the CP0 zones.json schema.

Rules consume TRACKS + ZONES, never video. Both can be synthesized, so every
rule can be written and tested before a real clip exists. This module is the
ZONES half.

=============================================================================
WHAT THIS SCENE IS
=============================================================================
A 1920x1080 oblique CCTV view of a four-way intersection, hand-authored (the
numbers below were chosen, not traced from anything):

  * a major north-south carriageway running from far-top to near-bottom, its
    width growing with y to mimic perspective, its centreline drifting right
    so the view is oblique rather than axis-aligned;
  * a minor east-west carriageway crossing it as a tilted band;
  * 2 lanes per direction per carriageway (8 lanes), each with a direction
    vector drawn tail->head along the lane;
  * one stop line per approach (4), each with approach_side DERIVED from a
    point where a waiting vehicle would sit -- never guessed;
  * two pedestrian crossings;
  * two signal queue zones, on the two NS approaches;
  * one traffic-light ROI;
  * solid and dashed lane markings.

Right-hand traffic. Looking roughly north: east is screen-right, so northbound
traffic occupies the RIGHT half of the NS road and eastbound traffic the LOWER
half of the EW band.

=============================================================================
WHAT THIS SCENE DOES *NOT* MODEL
=============================================================================
Read this before treating a passing test as a working rule.

  1. NO LENS DISTORTION. Perspective here is a linear widening with y. A real
     camera has radial distortion and a true projective mapping; straight lines
     near the frame edge will not be straight in real footage.
  2. NO REAL CALIBRATION. All distances are pixels. Pixels-per-metre varies
     enormously with y in a real oblique view, so every speed threshold tuned
     against this fixture is tuned against a fiction. Speed thresholds are the
     single most likely thing to need retuning on real footage.
  3. LANES OVERLAP IN THE INTERSECTION. NS and EW lane polygons both cover the
     intersection box. Zones.lane_of returns the FIRST match and NS lanes are
     listed first, so the intersection resolves to the major road. A real
     hand-authored config may instead leave the intersection box as carriageway
     with no lane, which changes what lane_of returns there -- and therefore
     changes which rules have an opinion inside the box.
  4. NO OCCLUSION GEOMETRY. Nothing here says a bus hides the lane behind it,
     or that the far approach is only 40 px tall. Real small-object dropout at
     the far end is modelled in synthetic_tracks.py, not here.
  5. NO SIGNAL PHASE. The traffic-light ROI is a rectangle with nothing in it.
     Whether this camera can even see a signal is unknown: the starter kit has
     no camera.md. src/signal.py must degrade to "no signal available".
  6. IDEALISED MARKINGS. Two markings, cleanly solid or dashed. Real markings
     are worn, occluded by traffic, and change style mid-line.
  7. THE SCENE IS SYMMETRIC AND TIDY. No bus stop, no parking bay, no side
     entrance, no roadworks -- all of which generate legitimate stopped
     vehicles that a real stopped_vehicle rule must survive.

A rule that passes here has correct LOGIC. It has not been shown to work.
"""
from __future__ import annotations

import json
from pathlib import Path

from src import geometry as g
from src.zones import Zones, load_zones

# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1920, 1080

# ---------------------------------------------------------------------------
# north-south carriageway (the major road)
# ---------------------------------------------------------------------------
# The road is a trapezoid: at NS_Y_FAR it is narrow, at NS_Y_NEAR it is wide.
# The centreline drifts right with y, which is what makes the view oblique
# rather than a top-down grid.
NS_Y_FAR, NS_Y_NEAR = 320.0, 1080.0
NS_HALFW_FAR, NS_HALFW_NEAR = 70.0, 520.0
NS_CX_FAR, NS_CX_NEAR = 880.0, 980.0


def ns_t(y: float) -> float:
    return (y - NS_Y_FAR) / (NS_Y_NEAR - NS_Y_FAR)


def ns_halfwidth(y: float) -> float:
    return NS_HALFW_FAR + (NS_HALFW_NEAR - NS_HALFW_FAR) * ns_t(y)


def ns_centre(y: float) -> float:
    return NS_CX_FAR + (NS_CX_NEAR - NS_CX_FAR) * ns_t(y)


def ns_x(y: float, offset: float) -> float:
    """x at row y, `offset` in [-1, +1] across the carriageway.

    -1 is the left kerb, 0 the centreline, +1 the right kerb.
    """
    return ns_centre(y) + offset * ns_halfwidth(y)


def ns_band(offset_a: float, offset_b: float,
            y_from: float = NS_Y_FAR, y_to: float = NS_Y_NEAR,
            rows: int = 5) -> list[list[float]]:
    """A polygon between two normalised offsets, sampled down the road.

    Sampled at several rows rather than as a bare quad so the polygon has the
    vertex count a hand-traced one would, and so the geometry helpers are
    exercised on something other than a rectangle.
    """
    ys = [y_from + (y_to - y_from) * i / (rows - 1) for i in range(rows)]
    left = [[round(ns_x(y, offset_a), 1), round(y, 1)] for y in ys]
    right = [[round(ns_x(y, offset_b), 1), round(y, 1)] for y in reversed(ys)]
    return left + right


# ---------------------------------------------------------------------------
# east-west carriageway (the minor road)
# ---------------------------------------------------------------------------
# A band that tilts up towards the right: the left end is nearer the camera,
# so it sits lower and is thicker.
EW_TOP_LEFT, EW_TOP_RIGHT = 690.0, 600.0
EW_BOT_LEFT, EW_BOT_RIGHT = 870.0, 760.0


def ew_top(x: float) -> float:
    return EW_TOP_LEFT + (EW_TOP_RIGHT - EW_TOP_LEFT) * (x / WIDTH)


def ew_bottom(x: float) -> float:
    return EW_BOT_LEFT + (EW_BOT_RIGHT - EW_BOT_LEFT) * (x / WIDTH)


def ew_y(x: float, offset: float) -> float:
    """y at column x, `offset` in [-1, +1] across the band (-1 = top kerb)."""
    top, bot = ew_top(x), ew_bottom(x)
    mid, half = (top + bot) / 2.0, (bot - top) / 2.0
    return mid + offset * half


def ew_band(offset_a: float, offset_b: float,
            x_from: float = 0.0, x_to: float = float(WIDTH),
            cols: int = 5) -> list[list[float]]:
    xs = [x_from + (x_to - x_from) * i / (cols - 1) for i in range(cols)]
    top = [[round(x, 1), round(ew_y(x, offset_a), 1)] for x in xs]
    bot = [[round(x, 1), round(ew_y(x, offset_b), 1)] for x in reversed(xs)]
    return top + bot


# ---------------------------------------------------------------------------
# where the two roads meet
# ---------------------------------------------------------------------------
# The EW band crosses the NS road around x ~ 900, so the intersection box spans
# roughly these rows. The stop lines sit just outside it.
INTERSECTION_Y_TOP = round(ew_top(900.0), 1)        # ~647.8
INTERSECTION_Y_BOTTOM = round(ew_bottom(900.0), 1)  # ~818.4

SL_NS_SOUTH_Y = 830.0     # northbound approach stops here (just south of the box)
SL_NS_NORTH_Y = 635.0     # southbound approach stops here (just north of it)
SL_EW_WEST_X = 560.0      # eastbound approach
SL_EW_EAST_X = 1360.0     # westbound approach


def _approach_side(segment: list[list[float]], waiting_point: tuple[float, float]) -> int:
    """Derive approach_side from where a waiting vehicle sits.

    Exactly what tools/annotate.html does with its third click. Never written
    by hand: the sign depends on the segment's authored direction, and guessing
    it inverts every stop-line test downstream without failing anything loudly.
    """
    seg = ((segment[0][0], segment[0][1]), (segment[1][0], segment[1][1]))
    side = g.side_of_line(waiting_point, seg)
    if side == 0:
        raise ValueError(f"waiting point {waiting_point} lies ON {segment}")
    return side


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------
def build_zones_dict() -> dict:
    """The zones.json content for this scene, fully authored."""
    # -- stop lines, authored left-to-right as the approaching driver sees it --
    # Northbound driver faces up-screen: their left is screen-left.
    sl_ns_south_seg = [[round(ns_x(SL_NS_SOUTH_Y, 0.0), 1), SL_NS_SOUTH_Y],
                       [round(ns_x(SL_NS_SOUTH_Y, 1.0), 1), SL_NS_SOUTH_Y]]
    # Southbound driver faces down-screen, towards us: their left is screen-RIGHT.
    sl_ns_north_seg = [[round(ns_x(SL_NS_NORTH_Y, 0.0), 1), SL_NS_NORTH_Y],
                       [round(ns_x(SL_NS_NORTH_Y, -1.0), 1), SL_NS_NORTH_Y]]
    # Eastbound driver faces screen-right: their left is up-screen (smaller y).
    sl_ew_west_seg = [[SL_EW_WEST_X, round(ew_y(SL_EW_WEST_X, 0.0), 1)],
                      [SL_EW_WEST_X, round(ew_y(SL_EW_WEST_X, 1.0), 1)]]
    # Westbound driver faces screen-left: their left is down-screen (larger y).
    sl_ew_east_seg = [[SL_EW_EAST_X, round(ew_y(SL_EW_EAST_X, 0.0), 1)],
                      [SL_EW_EAST_X, round(ew_y(SL_EW_EAST_X, -1.0), 1)]]

    return {
        "_README": [
            "SYNTHETIC SCENE -- invented, not traced from any camera.",
            "Generated by tests/fixtures/synthetic_scene.py. Do not ship this as",
            "config/zones.json: read that module's docstring for the list of real",
            "camera properties it does not model.",
        ],
        "schema_version": 1,

        "authored_against": {
            "image_width": WIDTH,
            "image_height": HEIGHT,
            "source_video": "synthetic_intersection.mp4",
            "source_frame_idx": 0,
        },

        # Two carriageways. on_carriageway() is the union, so the intersection
        # box is covered twice -- harmless, and it matches how a human tracing
        # two roads would draw them.
        "carriageway": [
            {"id": "ns_main", "polygon": ns_band(-1.0, 1.0)},
            {"id": "ew_cross", "polygon": ew_band(-1.0, 1.0)},
        ],

        # ORDER MATTERS: Zones.lane_of returns the first polygon that contains
        # the point, and NS lanes are listed first so the intersection box
        # resolves to the major road rather than to whichever EW lane happens
        # to overlap it.
        "lanes": [
            # -- northbound: right half, travelling away from the camera ----
            {"id": "ns_nb_inner", "polygon": ns_band(0.0, 0.5),
             "direction_from": [round(ns_x(1000.0, 0.25), 1), 1000.0],
             "direction_to": [round(ns_x(420.0, 0.25), 1), 420.0],
             "permitted_manoeuvres": ["through", "left"],
             "governed_by_stop_line": "sl_ns_south"},
            {"id": "ns_nb_outer", "polygon": ns_band(0.5, 1.0),
             "direction_from": [round(ns_x(1000.0, 0.75), 1), 1000.0],
             "direction_to": [round(ns_x(420.0, 0.75), 1), 420.0],
             "permitted_manoeuvres": ["through", "right"],
             "governed_by_stop_line": "sl_ns_south"},
            # -- southbound: left half, travelling towards the camera -------
            {"id": "ns_sb_inner", "polygon": ns_band(-0.5, 0.0),
             "direction_from": [round(ns_x(420.0, -0.25), 1), 420.0],
             "direction_to": [round(ns_x(1000.0, -0.25), 1), 1000.0],
             "permitted_manoeuvres": ["through", "left"],
             "governed_by_stop_line": "sl_ns_north"},
            {"id": "ns_sb_outer", "polygon": ns_band(-1.0, -0.5),
             "direction_from": [round(ns_x(420.0, -0.75), 1), 420.0],
             "direction_to": [round(ns_x(1000.0, -0.75), 1), 1000.0],
             "permitted_manoeuvres": ["through", "right"],
             "governed_by_stop_line": "sl_ns_north"},
            # -- eastbound: lower half of the band, travelling screen-right --
            {"id": "ew_eb_inner", "polygon": ew_band(0.0, 0.5),
             "direction_from": [200.0, round(ew_y(200.0, 0.25), 1)],
             "direction_to": [1700.0, round(ew_y(1700.0, 0.25), 1)],
             "permitted_manoeuvres": ["through", "left"],
             "governed_by_stop_line": "sl_ew_west"},
            {"id": "ew_eb_outer", "polygon": ew_band(0.5, 1.0),
             "direction_from": [200.0, round(ew_y(200.0, 0.75), 1)],
             "direction_to": [1700.0, round(ew_y(1700.0, 0.75), 1)],
             "permitted_manoeuvres": ["through", "right"],
             "governed_by_stop_line": "sl_ew_west"},
            # -- westbound: upper half, travelling screen-left --------------
            {"id": "ew_wb_inner", "polygon": ew_band(-0.5, 0.0),
             "direction_from": [1700.0, round(ew_y(1700.0, -0.25), 1)],
             "direction_to": [200.0, round(ew_y(200.0, -0.25), 1)],
             "permitted_manoeuvres": ["through", "left"],
             "governed_by_stop_line": "sl_ew_east"},
            {"id": "ew_wb_outer", "polygon": ew_band(-1.0, -0.5),
             "direction_from": [1700.0, round(ew_y(1700.0, -0.75), 1)],
             "direction_to": [200.0, round(ew_y(200.0, -0.75), 1)],
             "permitted_manoeuvres": ["through", "right"],
             "governed_by_stop_line": "sl_ew_east"},
        ],

        "stop_lines": [
            {"id": "sl_ns_south", "segment": sl_ns_south_seg,
             "approach_side": _approach_side(
                 sl_ns_south_seg, (ns_x(880.0, 0.5), 880.0)),
             "governs_lanes": ["ns_nb_inner", "ns_nb_outer"]},
            {"id": "sl_ns_north", "segment": sl_ns_north_seg,
             "approach_side": _approach_side(
                 sl_ns_north_seg, (ns_x(560.0, -0.5), 560.0)),
             "governs_lanes": ["ns_sb_inner", "ns_sb_outer"]},
            {"id": "sl_ew_west", "segment": sl_ew_west_seg,
             "approach_side": _approach_side(
                 sl_ew_west_seg, (480.0, ew_y(480.0, 0.5))),
             "governs_lanes": ["ew_eb_inner", "ew_eb_outer"]},
            {"id": "sl_ew_east", "segment": sl_ew_east_seg,
             "approach_side": _approach_side(
                 sl_ew_east_seg, (1450.0, ew_y(1450.0, -0.5))),
             "governs_lanes": ["ew_wb_inner", "ew_wb_outer"]},
        ],

        # Just outside each stop line, where pedestrians cross the major road
        # and the minor road respectively.
        "crossings": [
            {"id": "x_ns_south", "polygon": ns_band(-1.0, 1.0,
                                                    y_from=838.0, y_to=902.0, rows=3)},
            {"id": "x_ew_west", "polygon": ew_band(-1.0, 1.0,
                                                   x_from=470.0, x_to=546.0, cols=3)},
        ],

        # Where vehicles legitimately sit still on a red. stopped_vehicle is
        # suppressed here, and congestion ignores vehicles inside it -- that is
        # what separates "jammed approach" from "normal red-light queue".
        "signal_queue_zones": [
            {"id": "q_ns_south", "polygon": ns_band(0.0, 1.0,
                                                    y_from=SL_NS_SOUTH_Y, y_to=960.0,
                                                    rows=3),
             "lanes": ["ns_nb_inner", "ns_nb_outer"]},
            {"id": "q_ns_north", "polygon": ns_band(-1.0, 0.0,
                                                    y_from=470.0, y_to=SL_NS_NORTH_Y,
                                                    rows=3),
             "lanes": ["ns_sb_inner", "ns_sb_outer"]},
        ],

        "traffic_lights": [
            {"id": "sig_ns_south", "roi": [1498.0, 182.0, 1546.0, 302.0],
             "controls_lanes": ["ns_nb_inner", "ns_nb_outer"]},
        ],

        "lane_markings": [
            # The NS centreline: solid, so crossing it is a violation.
            {"id": "m_ns_centre",
             "segment": [[round(ns_x(NS_Y_FAR, 0.0), 1), NS_Y_FAR],
                         [round(ns_x(NS_Y_NEAR, 0.0), 1), NS_Y_NEAR]],
             "style": "solid",
             "separates": ["ns_nb_inner", "ns_sb_inner"]},
            # The divider between the two northbound lanes: dashed, so changing
            # lanes across it is legal.
            {"id": "m_ns_nb_divider",
             "segment": [[round(ns_x(NS_Y_FAR, 0.5), 1), NS_Y_FAR],
                         [round(ns_x(NS_Y_NEAR, 0.5), 1), NS_Y_NEAR]],
             "style": "dashed",
             "separates": ["ns_nb_inner", "ns_nb_outer"]},
            # The EW centreline: double solid.
            {"id": "m_ew_centre",
             "segment": [[0.0, round(ew_y(0.0, 0.0), 1)],
                         [float(WIDTH), round(ew_y(float(WIDTH), 0.0), 1)]],
             "style": "double_solid",
             "separates": ["ew_eb_inner", "ew_wb_inner"]},
        ],

        "notes": ("Synthetic scene for rule development. Right-hand traffic. "
                  "Northbound = right half of the NS road, eastbound = lower "
                  "half of the EW band. The intersection box is covered by the "
                  "NS lanes, which are listed first so lane_of resolves there."),
    }


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------
def write_zones_json(path: str | Path) -> Path:
    """Write the scene to disk, e.g. to run tools/validate_zones.py against it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_zones_dict(), indent=2), encoding="utf-8")
    return p


def scene_zones(tmp_path: str | Path | None = None,
                frame_size: tuple[int, int] | None = None) -> Zones:
    """The loaded, validated Zones object for this scene."""
    import tempfile

    base = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    p = write_zones_json(base / "synthetic_zones.json")
    return load_zones(p, frame_size=frame_size)


# ---------------------------------------------------------------------------
# named places, so tests read as sentences rather than coordinates
# ---------------------------------------------------------------------------
def point_in_lane(lane_id: str, along: float) -> tuple[float, float]:
    """A ground point inside `lane_id`, `along` in [0, 1] from its start.

    "Start" means the tail of the lane's direction arrow, so along=0 is where a
    vehicle enters the lane and along=1 is where it leaves.
    """
    ns_lanes = {
        "ns_nb_inner": (0.25, 1000.0, 420.0),
        "ns_nb_outer": (0.75, 1000.0, 420.0),
        "ns_sb_inner": (-0.25, 420.0, 1000.0),
        "ns_sb_outer": (-0.75, 420.0, 1000.0),
    }
    ew_lanes = {
        "ew_eb_inner": (0.25, 200.0, 1700.0),
        "ew_eb_outer": (0.75, 200.0, 1700.0),
        "ew_wb_inner": (-0.25, 1700.0, 200.0),
        "ew_wb_outer": (-0.75, 1700.0, 200.0),
    }
    if lane_id in ns_lanes:
        off, y0, y1 = ns_lanes[lane_id]
        y = y0 + (y1 - y0) * along
        return (ns_x(y, off), y)
    if lane_id in ew_lanes:
        off, x0, x1 = ew_lanes[lane_id]
        x = x0 + (x1 - x0) * along
        return (x, ew_y(x, off))
    raise KeyError(f"unknown lane {lane_id!r}")


#: Somewhere on the carriageway that is NOT inside any crossing -- jaywalking.
JAYWALK_POINT: tuple[float, float] = (ns_x(1000.0, -0.2), 1000.0)

#: Inside the southern pedestrian crossing -- lawful for a pedestrian.
CROSSING_POINT: tuple[float, float] = (ns_x(870.0, 0.0), 870.0)

#: Inside the northbound signal queue zone -- lawful place to sit still.
QUEUE_POINT: tuple[float, float] = (ns_x(890.0, 0.5), 890.0)

#: On the northbound carriageway but OUTSIDE the queue zone -- a real stop.
STOPPED_POINT: tuple[float, float] = (ns_x(1030.0, 0.5), 1030.0)

#: Off the road entirely (pavement, top-left) -- nothing should ever fire here.
OFF_ROAD_POINT: tuple[float, float] = (120.0, 200.0)
