"""Regression tests for the train.py review findings (TR-*).

Pure: no model, no tokenizer, no Hub. Each test drives the inlined helper with
fake objects, because train.py must stay a single uploadable file.
"""

import importlib.util
import re
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


# ── TR-3: the diverse split on per-step DELTA actions ───────────────────

@pytest.mark.xfail(strict=True, reason=(
    "TR-3 NEEDS DECISION: episode_descriptors reads action[:, :3] as a POSITION; "
    "on 11D per-step-delta datasets it is a velocity, so farthest-point sampling "
    "picks the glitchiest episodes. Any fix changes the held-out set of existing "
    "delta-dataset runs."))
def test_diverse_split_on_deltas_ignores_a_glitch():
    rng = np.random.default_rng(0)
    by_ep = {}
    for e in range(40):
        a = np.zeros((100, 11))
        a[:, :3] = rng.normal(scale=0.0005, size=(100, 3))   # SLAM jitter deltas
        if e == 7:
            a[60, :3] = [0.02, 0.0, 0.0]                      # one 20 mm spike at closure
        a[60:, 10] = 1.0
        by_ep[e] = a
    picked = ck.select_diverse(ck.episode_descriptors(by_ep, closure_col=10), 4)
    assert 7 not in picked, "the lone glitch episode was held out"


# ── smoke_generation diagnostics (TR-4, TR-9) ───────────────────────────

def _smoke():
    spec = importlib.util.spec_from_file_location("_sg_fixes", _PI05 / "smoke_generation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sg = _smoke()


def test_differences_within_sampling_noise_do_not_pass():
    """TR-4: pi05 re-sampled on identical input moves by ~3e-3; an episode
    difference of that size is no evidence of input dependence."""
    assert sg.input_dependence_verdict(diff=3e-3, floor=3e-3) != "PASS"
    assert sg.input_dependence_verdict(diff=3e-2, floor=3e-3) == "PASS"


def test_deterministic_decoder_keeps_the_absolute_threshold():
    assert sg.input_dependence_verdict(diff=1e-3, floor=0.0) == "PASS"
    assert sg.input_dependence_verdict(diff=0.0, floor=0.0) != "PASS"


def test_net_motion_over_executed_window_for_deltas(capsys):
    """TR-9: 15 clean 2.83 mm/step deltas travel 42.45 mm, not 0."""
    chunk = np.tile([0.00283, 0, 0] + [0] * 8, (50, 1)).astype(float)
    sg.report_execution_quality(chunk.copy(), chunk, 15, differenced=False)
    out = capsys.readouterr().out
    m = re.search(r"net ([0-9.]+) mm of ([0-9.]+) mm travelled", out)
    assert m, out
    assert float(m.group(1)) == pytest.approx(42.45, abs=0.1)
    assert float(m.group(2)) == pytest.approx(42.45, abs=0.1)
