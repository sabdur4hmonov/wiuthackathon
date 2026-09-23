"""Offline export of one validated, log-free harness result for the website.

This never runs the model and never asserts that footage is real or authorized.
The operator must verify provenance before adding a catalog entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness_adapter import (
    MAX_OUTPUT_BYTES, _reject_constant, _unique_safe_object,
    sanitized_harness_result,
)


def export_sample(source: Path, filename: str, destination: Path) -> None:
    if source.stat().st_size > MAX_OUTPUT_BYTES:
        raise ValueError("Prediction file exceeds the demo output limit.")
    raw = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_safe_object,
        parse_constant=_reject_constant,
    )
    clean = sanitized_harness_result(raw, filename, filename)
    # Exclusive creation avoids silently replacing a verified published asset.
    with destination.open("x", encoding="utf-8") as output:
        json.dump(clean, output, indent=2, allow_nan=False)
        output.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one sanitized harness result; does not verify footage provenance.")
    parser.add_argument("--pred", required=True, type=Path, help="Existing single-video harness JSON")
    parser.add_argument("--video", required=True, help="Exact .mp4 key in the harness JSON")
    parser.add_argument("--out", required=True, type=Path, help="New sanitized JSON file")
    args = parser.parse_args()
    try:
        export_sample(args.pred, args.video, args.out)
    except Exception:
        print("Sample export failed: check input, output location, and prediction validity.")
        return 1
    print("Sanitized sample prediction exported. Verify footage provenance before publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
