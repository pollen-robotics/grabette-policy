"""Checks and pipeline install of the ACT launcher (train.py).

The helpers live inside train.py because `hf jobs uv run` uploads one file;
they are loaded from there rather than kept in a second copy to test.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

_ACT = Path(__file__).resolve().parents[1]
K = 50


def _load_train():
    spec = importlib.util.spec_from_file_location("_act_train", _ACT / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load_train()


def act_cfg(action_dim=8, **kw):
    return ACTConfig(
        device="cpu",
        chunk_size=K,
        n_action_steps=kw.pop("n_action_steps", K),
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            "observation.images.cam0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        **kw,
    )


def _moments(dim):
    return {"mean": np.zeros(dim), "std": np.ones(dim), "min": -np.ones(dim), "max": np.ones(dim)}


def stats(chunk_size=K):
    """Stats as lerobot hands them over: every leaf a numpy array."""
    s = {
        "action": _moments(8),
        "observation.state": _moments(2),
        "observation.images.cam0": {"mean": np.full((3, 1, 1), 0.5), "std": np.full((3, 1, 1), 0.25)},
    }
    if chunk_size is not None:
        s["action_relative_meta"] = {"chunk_size": np.array([chunk_size])}
    return s


def names(pipe):
    return [type(s).__name__ for s in pipe.steps]


# ── stats ───────────────────────────────────────────────────────────────

def test_check_relative_stats_rejects_absolute():
    with pytest.raises(RuntimeError, match="write_relative_action_stats"):
        tr.check_relative_stats(stats(chunk_size=None), K)


def test_check_relative_stats_rejects_chunk_mismatch():
    with pytest.raises(RuntimeError, match=r"50.*100|100.*50"):
        tr.check_relative_stats(stats(chunk_size=50), 100)


def test_check_relative_stats_accepts_match():
    tr.check_relative_stats(stats(chunk_size=K), K)


# ── config ──────────────────────────────────────────────────────────────

def test_check_act_config_rejects_temporal_ensembling():
    with pytest.raises(RuntimeError, match="temporal"):
        tr.check_act_config(act_cfg(temporal_ensemble_coeff=0.01, n_action_steps=1))


def test_check_act_config_rejects_wrong_action_dim():
    with pytest.raises(RuntimeError, match="8"):
        tr.check_act_config(act_cfg(action_dim=11))


def test_check_act_config_rejects_missing_state():
    """Datasets older than generate_dataset.py's observation.state would
    otherwise fail deep in ACT's forward with a bare KeyError."""
    cfg = act_cfg()
    del cfg.input_features["observation.state"]
    with pytest.raises(RuntimeError, match="observation.state"):
        tr.check_act_config(cfg)


def test_check_act_config_rejects_other_policy():
    with pytest.raises(RuntimeError, match="act"):
        tr.check_act_config(DiffusionConfig())


# ── pipeline ────────────────────────────────────────────────────────────

def test_fresh_pipeline_installs_steps():
    pre, post = tr._fresh_pipeline(act_cfg(), dataset_stats=stats())
    assert "ChunkRelativeActionsStep" in names(pre)
    assert "AbsoluteFromChunkRelativeStep" in names(post)


def test_fresh_pipeline_passthrough_when_pretrained(tmp_path):
    pre, post = tr._fresh_pipeline(act_cfg(), dataset_stats=stats())
    pre.save_pretrained(str(tmp_path))
    post.save_pretrained(str(tmp_path))
    pre2, _ = tr._fresh_pipeline(act_cfg(), pretrained_path=str(tmp_path), dataset_stats=stats())
    assert names(pre2).count("ChunkRelativeActionsStep") == 1


def _saved_chunkrel_pipeline(tmp_path):
    pre, post = tr._fresh_pipeline(act_cfg(), dataset_stats=stats())
    pre.save_pretrained(str(tmp_path))
    post.save_pretrained(str(tmp_path))


def test_pretrained_path_checks_stats_lerobot_overrides_with(tmp_path):
    """On resume/fine-tune lerobot swaps the saved normaliser stats for the
    current dataset's (normalizer_processor override): those must be checked too."""
    _saved_chunkrel_pipeline(tmp_path)
    overrides = {"normalizer_processor": {"stats": stats(chunk_size=None)}}
    with pytest.raises(RuntimeError, match="write_relative_action_stats"):
        tr._fresh_pipeline(act_cfg(), pretrained_path=str(tmp_path),
                           preprocessor_overrides=overrides)


def test_pretrained_path_checks_config(tmp_path):
    _saved_chunkrel_pipeline(tmp_path)
    cfg = act_cfg(temporal_ensemble_coeff=0.01, n_action_steps=1)
    with pytest.raises(RuntimeError, match="temporal"):
        tr._fresh_pipeline(cfg, pretrained_path=str(tmp_path), dataset_stats=stats())


def test_remote_job_is_refused():
    """--job.target hands training to a pod running plain lerobot-train,
    without this wrapper: it would train on absolute actions with relative stats."""
    with pytest.raises(RuntimeError, match="hf jobs uv run"):
        tr.lerobot_train.submit_to_hf(None)


def test_fresh_pipeline_rejects_pretrained_without_steps(tmp_path):
    cfg = act_cfg()
    pre, post = make_act_pre_post_processors(
        cfg, {k: {m: torch.as_tensor(v) for m, v in d.items()} for k, d in stats(None).items()})
    pre.save_pretrained(str(tmp_path))
    post.save_pretrained(str(tmp_path))
    with pytest.raises(RuntimeError, match="chunk-relative"):
        tr._fresh_pipeline(cfg, pretrained_path=str(tmp_path), dataset_stats=stats())
