"""Regression tests for the train.py review findings (TR-*).

Pure: no model, no tokenizer, no Hub. Each test drives the inlined helper with
fake objects, because train.py must stay a single uploadable file.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

_PI05 = Path(__file__).resolve().parents[1]


def _load_train():
    spec = importlib.util.spec_from_file_location("_pi05_train_fixes", _PI05 / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ck = _load_train()


# ── TR-1: resume must keep the normalization stats ──────────────────────

def _stats():
    return {"action": {"mean": np.zeros(8), "std": np.ones(8)}}


@pytest.fixture
def captured(monkeypatch):
    """Replace lerobot's factory with a recorder: building the real pi05
    pipeline would fetch the PaliGemma tokenizer."""
    seen = {}

    def fake_original(policy_cfg, pretrained_path=None, pretrained_revision=None, **kw):
        seen.update(kw)
        return "pre", "post"

    monkeypatch.setattr(ck, "_original", fake_original)
    return seen


def test_resume_recovers_stats_from_the_overrides(captured):
    """lerobot e40b58a8 (the pinned revision) omits dataset_stats on
    --resume=true and passes the stats ONLY through the overrides."""
    stats = _stats()
    ck._fresh_pipeline(
        object(), pretrained_path="ckpt/pretrained_model",
        preprocessor_overrides={"normalizer_processor": {"stats": stats}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": stats}})
    assert captured.get("dataset_stats") is stats, "resume dropped normalization stats"
    assert "preprocessor_overrides" not in captured


def test_fresh_start_keeps_dataset_stats(captured):
    stats = _stats()
    ck._fresh_pipeline(object(), pretrained_path="lerobot/pi05_base", dataset_stats=stats,
                       preprocessor_overrides={"normalizer_processor": {"stats": {}}})
    assert captured["dataset_stats"] is stats


def test_no_stats_anywhere_refuses(captured):
    """lerobot 0.6.1 forwards no stats at all on resume. Training on an identity
    normalizer must fail loudly, not silently."""
    with pytest.raises(RuntimeError, match="normalization stats"):
        ck._fresh_pipeline(
            object(), pretrained_path="ckpt/pretrained_model",
            preprocessor_overrides={"normalizer_processor": {"features": {}}})
