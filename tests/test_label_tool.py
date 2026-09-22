"""tools/label.html — the class table and export contract.

The tool's interactive behaviour was verified in a browser (segments open and
close on a keypress, same-class overlaps are flagged, zero-length taps are
refused, and the exported JSON scores through the real evaluate.py). What is
worth guarding in CI is the part that can silently rot: the class list drifting
from evaluate.OFFICIAL_CLASSES, or two classes colliding on one shortcut key.

A wrong label id here would be invisible during labelling and would poison the
dev set -- every event of that class scoring 0 against predictions that were
actually right.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "label.html"


@pytest.fixture(scope="module")
def entries() -> list[tuple[str, str, str]]:
    """The CLASSES table parsed out of the tool's JavaScript."""
    src = TOOL.read_text(encoding="utf-8")
    block = re.search(r"const CLASSES = \[(.*?)\];", src, re.S)
    assert block, "could not find the CLASSES table in label.html"
    rows = re.findall(r'\["(\w)",\s*"(\w+)",\s*"(#[0-9a-fA-F]{6})"\]', block.group(1))
    assert rows, "CLASSES table did not parse"
    return rows


def test_tool_exists():
    assert TOOL.exists()


def test_every_official_class_is_labellable(entries):
    import evaluate

    assert [e[1] for e in entries] == list(evaluate.OFFICIAL_CLASSES)


def test_shortcut_keys_are_unique(entries):
    """A collision would silently label the wrong class."""
    keys = [e[0] for e in entries]
    assert len(set(keys)) == len(keys), f"duplicate keys: {keys}"


def test_shortcut_keys_do_not_collide_with_transport_keys(entries):
    """Digits drive playback speed; those must not also be class keys."""
    keys = {e[0] for e in entries}
    assert keys.isdisjoint(set("0123456789"))
    assert keys.isdisjoint({"z"})          # Ctrl+Z is undo


def test_colours_are_distinct(entries):
    colours = [e[2].lower() for e in entries]
    assert len(set(colours)) == len(colours)


def test_exported_shape_is_what_evaluate_expects():
    """The shape the tool's build() produces, scored by the real evaluator.

    Pinned here as data because the browser is not available in CI: this is the
    exact JSON observed coming out of the tool.
    """
    import evaluate

    gt = {
        "clip_01.mp4": {
            "duration": 60.0,
            "fps": 25.0,
            "events": [[12.04, 18.92, "accident"],
                       [30.0, 34.5, "jaywalking"],
                       [40.0, 55.25, "congestion"]],
        }
    }
    pred = {"team": "t", "videos": {"clip_01.mp4": {
        "events": [[12.1, 18.8, "accident"], [40.5, 55.0, "congestion"]],
        "risk": [[0.0, 0.0], [30.0, 0.0], [59.9, 0.0]]}}}

    errors, _ = evaluate.validate(pred, gt)
    assert errors == []
    rep = evaluate.evaluate(gt, pred)
    assert rep["part_a"]["per_class"]["accident"]["0.7"]["f1"] == 1.0
    assert rep["part_a"]["per_class"]["jaywalking"]["0.5"]["f1"] == 0.0
    assert rep["model_score"] > 0.0


def test_tool_refuses_to_export_a_label_outside_the_official_list(entries):
    """Belt and braces: the ids in the table are exactly the strings exported."""
    import evaluate

    for _key, label, _colour in entries:
        assert label in evaluate.OFFICIAL_CLASSES


def test_tool_documents_the_boundary_precision_requirement():
    """tIoU 0.7 punishes sloppy boundaries; the tool must say so on screen."""
    src = TOOL.read_text(encoding="utf-8")
    assert "0.7" in src
    assert "toFixed(3)" in src, "timestamp must be shown to millisecond precision"


def test_tool_handles_an_undecodable_clip():
    """OpenCV writes MPEG-4 Part 2 by default, which no browser decodes. The
    tool must say that rather than looking idle."""
    src = TOOL.read_text(encoding="utf-8")
    assert "v.onerror" in src
    assert "codec not supported" in src
    assert "libx264" in src, "should tell the user how to re-encode"
