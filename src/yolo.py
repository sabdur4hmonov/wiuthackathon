"""YOLO helpers shared by Stage 1 (src/perception.py) and the causal risk
estimator (src/risk/). Deliberately tiny and pipeline-free: the risk package
may import this, and must never import perception."""
from __future__ import annotations


def _disable_amp_check(model) -> None:
    """Stop Ultralytics' AMP self-check, which downloads yolo11n.pt.

    The check runs on the first CUDA inference and fetches a nano checkpoint to
    compare fp16 and fp32 outputs. With no internet that is at best a long
    timeout and at worst an exception inside detect_events -- i.e. an empty
    prediction for the video. We pin the flag on every object that owns one.
    """
    for obj in (model, getattr(model, "model", None), getattr(model, "predictor", None)):
        if obj is None:
            continue
        for attr in ("amp", "_amp_checked"):
            try:
                setattr(obj, attr, False if attr == "amp" else True)
            except Exception:
                pass
    try:
        args = getattr(getattr(model, "model", None), "args", None)
        if isinstance(args, dict):
            args["amp"] = False
    except Exception:
        pass


def _fp16_kwarg() -> dict:
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:  # noqa: BLE001 - fall back to the old name
        pass
    return {"half": True}


def device_string() -> str:
    """'0' for the first CUDA device, else 'cpu'. Reported in the timing log."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
