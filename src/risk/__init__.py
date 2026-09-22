"""Part B -- causal accident anticipation.

This package is deliberately ISOLATED from the batch pipeline. It may import
numpy and src.geometry (pure maths) and src.config (constants only). It may NOT
import src.perception, src.cache, src.pipeline, src.tracks, src.rules or
src.postprocess, and it may not import cv2.

The reason is not tidiness. The tracks cache is built by a pass that has already
seen the whole video; reading it inside step() would smuggle future frames into
a score the rules call causal. That is a disqualifying violation, not a bug. The
import ban makes it structurally impossible rather than a matter of discipline,
and tests/test_risk_isolation.py enforces it in a fresh interpreter.
"""
from .estimator import RiskEstimator

__all__ = ["RiskEstimator"]
