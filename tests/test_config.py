"""The detector knobs in src/config.py must stay runnable offline."""
from src.config import CFG, REPO_ROOT


def test_configured_weights_are_fetched_by_download_script():
    # The run is offline: a model variant that download.sh does not fetch
    # means no detector at evaluation time and an empty prediction per video.
    script = (REPO_ROOT / "weights" / "download.sh").read_text(encoding="utf-8")
    listed = [line.strip().strip('"').split("|")[0]
              for line in script.splitlines()
              if line.strip().startswith('"') and "|http" in line]
    assert CFG.perception.weights in listed, (
        f"{CFG.perception.weights} is not fetched by weights/download.sh ({listed})")


def test_speed_knobs_are_sane():
    p = CFG.perception
    assert p.imgsz % 32 == 0          # YOLO stride; Ultralytics would round it anyway
    assert p.frame_stride >= 1
