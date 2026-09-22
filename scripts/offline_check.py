#!/usr/bin/env python3
"""Prove the pipeline completes with networking disabled.

    python scripts/offline_check.py --video samples/clip1.mp4

There is no internet at evaluation. Ultralytics in particular reaches out in
three places that are easy to miss, and each one turns into an empty prediction
for the video it happens on:

  1. the AMP self-check downloads yolo11n.pt on the first CUDA inference;
  2. a version / analytics ping on import;
  3. a font download inside its plotting helpers.

Asserting that we set the right env vars proves nothing -- the only convincing
test is to make the network unusable and run the real thing. This script
monkey-patches socket so that ANY outbound connection raises, then runs
detect_events and the RiskEstimator end to end. If a hidden code path tries to
reach the network, it fails here rather than on the judges' machine.

Exit 0 = the pipeline completed with every socket blocked.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ATTEMPTS: list[str] = []


class NetworkBlocked(RuntimeError):
    pass


def block_network() -> None:
    """Make every outbound connection raise, recording who tried."""
    real_socket = socket.socket
    real_create = socket.create_connection
    real_getaddr = socket.getaddrinfo

    def _record(what: str) -> None:
        stack = "".join(traceback.format_stack(limit=8)[:-1])
        _ATTEMPTS.append(f"{what}\n{stack}")

    class BlockedSocket(real_socket):  # type: ignore[misc, valid-type]
        def connect(self, *a, **k):
            _record(f"socket.connect{a}")
            raise NetworkBlocked(f"outbound connection blocked: {a}")

        def connect_ex(self, *a, **k):
            _record(f"socket.connect_ex{a}")
            raise NetworkBlocked(f"outbound connection blocked: {a}")

    def blocked_create_connection(address, *a, **k):
        _record(f"socket.create_connection({address})")
        raise NetworkBlocked(f"outbound connection blocked: {address}")

    def blocked_getaddrinfo(host, port, *a, **k):
        # Localhost stays resolvable: torch uses it for intra-process bits.
        if host in ("localhost", "127.0.0.1", "::1", None):
            return real_getaddr(host, port, *a, **k)
        _record(f"socket.getaddrinfo({host}, {port})")
        raise NetworkBlocked(f"DNS blocked: {host}")

    socket.socket = BlockedSocket  # type: ignore[assignment]
    socket.create_connection = blocked_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = blocked_getaddrinfo  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path,
                    help="a clip to run through. Without it, only imports and "
                         "model loading are exercised.")
    ap.add_argument("--frames", type=int, default=150,
                    help="how many frames to push through RiskEstimator")
    args = ap.parse_args()

    print("blocking all outbound sockets ...")
    block_network()

    # Sanity: the block itself must work, or the whole test is vacuous.
    try:
        socket.create_connection(("example.com", 80), timeout=1)
        print("FAIL: the network block did not take effect", file=sys.stderr)
        return 1
    except NetworkBlocked:
        print("  ok: outbound connections raise NetworkBlocked")
    _ATTEMPTS.clear()

    print("importing solution (this imports ultralytics + torch) ...")
    t0 = time.perf_counter()
    try:
        import solution
    except Exception:
        print("FAIL: importing solution.py raised:\n" + traceback.format_exc(),
              file=sys.stderr)
        return 1
    print(f"  ok in {time.perf_counter() - t0:.1f}s; CLASSES = {solution.CLASSES}")

    print("loading the detector from local weights ...")
    try:
        from src.perception import PerceptionUnavailable, load_model, weights_path

        try:
            load_model()
            print(f"  ok: loaded {weights_path()}")
        except PerceptionUnavailable as e:
            print(f"  ! detector unavailable ({e})")
            print("  ! the pipeline will still complete, but with no detections.")
            print("  ! run weights/download.sh ONCE, with internet, before evaluation.")
    except Exception:
        print("FAIL: loading the model raised:\n" + traceback.format_exc(),
              file=sys.stderr)
        return 1

    if args.video and args.video.exists():
        print(f"running detect_events on {args.video.name} ...")
        try:
            events = solution.detect_events(str(args.video))
            print(f"  ok: {len(events)} events")
        except Exception:
            print("FAIL: detect_events raised:\n" + traceback.format_exc(),
                  file=sys.stderr)
            return 1

        print(f"running RiskEstimator over {args.frames} frames ...")
        try:
            import cv2

            cap = cv2.VideoCapture(str(args.video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            est = solution.RiskEstimator()
            est.reset({"video_id": args.video.name, "fps": fps,
                       "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})
            n = 0
            while n < args.frames:
                ok, frame = cap.read()
                if not ok:
                    break
                est.step(frame, n / fps)
                n += 1
            cap.release()
            print(f"  ok: {n} frames stepped")
        except Exception:
            print("FAIL: RiskEstimator raised:\n" + traceback.format_exc(),
                  file=sys.stderr)
            return 1
    else:
        print("no --video given: skipped the end-to-end run "
              "(imports and weight loading were still exercised)")

    print()
    if _ATTEMPTS:
        print(f"FAIL: {len(_ATTEMPTS)} network attempt(s) were made:\n",
              file=sys.stderr)
        for a in _ATTEMPTS[:5]:
            print(a, file=sys.stderr)
        return 1

    print("PASS: the pipeline completed with every outbound socket blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
