"""The cache is a dev tool. These tests exist to prove it can never be anything
more than that: every failure mode must degrade to "recompute", silently.

The judges' machine may have a read-only checkout, a full disk, or no writable
temp dir. If any of those changed the predictions, the cache would have turned
a convenience into a correctness dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src import cache as C
from src.config import PerceptionConfig, perception_config_hash
from src.tracks import COLUMNS, TrackTable


@pytest.fixture
def cache_at(tmp_path, monkeypatch):
    monkeypatch.setenv(C.CACHE_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.delenv(C.DISABLE_ENV_VAR, raising=False)
    return tmp_path / "cache"


def table(n_rows: int = 5, complete: bool = True) -> TrackTable:
    data = np.arange(n_rows * len(COLUMNS), dtype=np.float32).reshape(n_rows, -1)
    return TrackTable(data=data, fps=25.0, duration=60.0, n_frames=1500,
                      width=1920, height=1080, frame_stride=2,
                      complete=complete, processed_until_sec=60.0 if complete else 12.0)


# ---------------------------------------------------------------------------
# location and policy
# ---------------------------------------------------------------------------
def test_location_comes_from_the_env_var(cache_at):
    assert C.cache_dir() == cache_at
    assert C.is_enabled() is True


def test_falls_back_to_temp_when_unset(monkeypatch):
    monkeypatch.delenv(C.CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(C.DISABLE_ENV_VAR, raising=False)
    d = C.cache_dir()
    assert d is not None and d.exists()


def test_never_lives_inside_the_repo(monkeypatch):
    """A stale entry must not be committable."""
    monkeypatch.delenv(C.CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(C.DISABLE_ENV_VAR, raising=False)
    repo = Path(__file__).resolve().parent.parent
    assert repo not in C.cache_dir().resolve().parents


@pytest.mark.parametrize("val", ["1", "true", "yes"])
def test_disable_env_var_turns_it_off(cache_at, monkeypatch, val):
    monkeypatch.setenv(C.DISABLE_ENV_VAR, val)
    assert C.cache_dir() is None
    assert C.is_enabled() is False
    assert C.load("v", "c") is None
    assert C.store("v", "c", table()) is False


@pytest.mark.parametrize("val", ["", "0", "false", "False"])
def test_disable_env_var_off_values_keep_it_on(cache_at, monkeypatch, val):
    monkeypatch.setenv(C.DISABLE_ENV_VAR, val)
    assert C.is_enabled() is True


def test_unwritable_dir_disables_rather_than_raises(monkeypatch, tmp_path):
    """A read-only repo on the judges' box must not be fatal."""
    target = tmp_path / "nope"
    monkeypatch.setenv(C.CACHE_ENV_VAR, str(target))

    def boom(*a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "mkdir", boom)
    assert C.cache_dir() is None
    assert C.load("v", "c") is None
    assert C.store("v", "c", table()) is False


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------
def test_video_hash_is_content_based(tmp_path):
    a, b, c = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    c.write_bytes(b"other bytes")
    assert C.video_hash(a) == C.video_hash(b)        # a copy hits
    assert C.video_hash(a) != C.video_hash(c)        # different content misses


def test_config_hash_changes_with_the_config():
    h = PerceptionConfig().hash()
    assert PerceptionConfig().hash() == h            # stable
    assert PerceptionConfig(imgsz=640).hash() != h
    assert PerceptionConfig(frame_stride=5).hash() != h
    assert PerceptionConfig(weights="yolo11m.pt").hash() != h


def test_entry_path_includes_both_hashes(cache_at):
    p = C.entry_path("VHASH", "CHASH")
    assert "VHASH" in p.name and "CHASH" in p.name


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------
def test_round_trip_preserves_everything(cache_at):
    t = table(7)
    assert C.store("v1", "c1", t) is True
    got = C.load("v1", "c1")
    assert got is not None
    assert np.array_equal(got.data, t.data)
    assert (got.fps, got.duration, got.n_frames) == (t.fps, t.duration, t.n_frames)
    assert (got.width, got.height, got.frame_stride) == (t.width, t.height,
                                                         t.frame_stride)
    assert got.complete is True


def test_miss_on_a_different_key(cache_at):
    C.store("v1", "c1", table())
    assert C.load("v2", "c1") is None
    assert C.load("v1", "c2") is None


def test_changing_the_perception_config_invalidates(cache_at):
    """No manual 'clear the cache' step to forget."""
    h1 = PerceptionConfig().hash()
    h2 = PerceptionConfig(imgsz=1280).hash()
    C.store("v1", h1, table())
    assert C.load("v1", h1) is not None
    assert C.load("v1", h2) is None


def test_truncated_tables_are_not_cached(cache_at):
    """A budget stop is a property of one slow run, not of the video."""
    assert C.store("v1", "c1", table(complete=False)) is False
    assert C.load("v1", "c1") is None


def test_empty_table_round_trips(cache_at):
    t = TrackTable.empty(fps=25.0, duration=10.0, n_frames=250)
    assert C.store("v1", "c1", t) is True
    got = C.load("v1", "c1")
    assert got is not None and len(got) == 0


# ---------------------------------------------------------------------------
# corruption
# ---------------------------------------------------------------------------
def test_corrupt_entry_reads_as_a_miss(cache_at):
    C.store("v1", "c1", table())
    p = C.entry_path("v1", "c1")
    p.write_bytes(b"garbage, not an npz")
    assert C.load("v1", "c1") is None                # degrade, do not raise


def test_truncated_file_reads_as_a_miss(cache_at):
    C.store("v1", "c1", table())
    p = C.entry_path("v1", "c1")
    p.write_bytes(p.read_bytes()[: max(1, p.stat().st_size // 3)])
    assert C.load("v1", "c1") is None


def test_schema_drift_reads_as_a_miss(cache_at):
    """An old entry written before a column was added must not be trusted."""
    p = C.entry_path("v1", "c1")
    np.savez_compressed(p, data=np.zeros((2, 3), dtype=np.float32),
                        columns=np.array(["a", "b", "c"]),
                        meta=np.array('{"fps": 25.0}'))
    assert C.load("v1", "c1") is None


def test_store_failure_is_swallowed(cache_at, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", boom)
    assert C.store("v1", "c1", table()) is False     # no exception escapes


def test_store_cleans_up_its_temp_file_on_failure(cache_at, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    C.store("v1", "c1", table())
    assert list(cache_at.glob("*.tmp")) == []


def test_load_of_a_missing_entry_is_a_quiet_miss(cache_at):
    assert C.load("never", "written") is None


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------
def test_clear(cache_at):
    C.store("v1", "c1", table())
    C.store("v2", "c1", table())
    assert C.clear() == 2
    assert C.load("v1", "c1") is None


def test_describe(cache_at, monkeypatch):
    assert "entries" in C.describe()
    monkeypatch.setenv(C.DISABLE_ENV_VAR, "1")
    assert "DISABLED" in C.describe()
