"""Stage and upload the Hugging Face Space.

    pip install huggingface_hub
    HF_TOKEN=hf_... python demo/deploy.py <user-or-org>/<space-name>

Stages exactly what the Space needs into demo/_space_build/ (git-ignored) and
uploads it with huggingface_hub, which stores the 19 MB weights via LFS:

    app.py  README.md  requirements.txt  packages.txt
    solution.py  src/  config/zones.json  config/pose_reference.jpg
    weights/yolo11s.pt

--dry-run stages without uploading.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "_space_build"


def stage() -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir()
    shutil.copy(HERE / "app.py", BUILD / "app.py")
    shutil.copy(HERE / "space_README.md", BUILD / "README.md")
    shutil.copy(HERE / "requirements.txt", BUILD / "requirements.txt")
    shutil.copy(HERE / "packages.txt", BUILD / "packages.txt")
    shutil.copy(ROOT / "solution.py", BUILD / "solution.py")
    shutil.copytree(ROOT / "src", BUILD / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (BUILD / "config").mkdir()
    for name in ("zones.json", "pose_reference.jpg"):
        shutil.copy(ROOT / "config" / name, BUILD / "config" / name)
    (BUILD / "weights").mkdir()
    shutil.copy(ROOT / "weights" / "yolo11s.pt", BUILD / "weights" / "yolo11s.pt")
    return BUILD


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("space", help="<user-or-org>/<space-name>")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hardware", default="zero-a10g",
                    help="Space hardware: zero-a10g (ZeroGPU; needs PRO or a community grant) or cpu-basic")
    args = ap.parse_args()
    build = stage()
    size = sum(f.stat().st_size for f in build.rglob("*") if f.is_file())
    print(f"staged {build} ({size / 1e6:.1f} MB)")
    if args.dry_run:
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("set HF_TOKEN to a write token", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", space_hardware=args.hardware,
                    exist_ok=True)
    api.upload_folder(folder_path=str(build), repo_id=args.space, repo_type="space",
                      commit_message="Deploy the WIUT traffic demo")
    print(f"uploaded: https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
