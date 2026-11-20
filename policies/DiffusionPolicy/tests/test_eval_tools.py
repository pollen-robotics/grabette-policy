"""Regression tests for the DiffusionPolicy eval / conversion review findings.

Pure: fake policies and synthetic poses only.
"""
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import offline_eval  # noqa: E402
import ood_check  # noqa: E402
import vision_check  # noqa: E402


# ── TR-5: vision_check must condition on BOTH frames of the pair ────────

class QueueingPolicy:
    """Mimics lerobot DiffusionPolicy's queues: select_action only runs the
    model when the action queue is empty; predict_action_chunk on an empty
    queue uses the (B, T, ...) batch directly."""

    config = SimpleNamespace(n_obs_steps=2, n_action_steps=8)

    def reset(self):
        self._queues = {"img": deque(maxlen=2), "action": deque(maxlen=8)}

    @staticmethod
    def _generate(imgs):                  # imgs: (1, T, C, H, W)
        a = torch.zeros(1, 8, 11)
        a[..., 0] = imgs[:, -1].mean()    # depends on the CURRENT frame
        a[..., 1] = imgs[:, 0].mean()     # depends on the previous frame
        a[..., 2] = torch.arange(8.0)     # action index, to see where the chunk starts
        return a

    def select_action(self, batch):
        img, q = batch["observation.images.cam0"], self._queues["img"]
        q.extend([img] * (2 - len(q)) if not q else [img])
        if not self._queues["action"]:
            chunk = self._generate(torch.stack(list(q), dim=1))
            self._queues["action"].extend(chunk.transpose(0, 1))
        return self._queues["action"].popleft()

    def predict_action_chunk(self, batch):
        assert not any(self._queues.values()), "expects the offline (empty-queue) mode"
        return self._generate(batch["observation.images.cam0"])


def _chunk(img_prev, img_now):
    ident = lambda x: x  # noqa: E731
    return vision_check.get_chunk(QueueingPolicy(), ident, ident, "cpu",
                                  img_prev, img_now, np.zeros(2), n_samples=1)


def test_img_now_changes_the_chunk():
    zeros, ones = torch.zeros(3, 4, 4), torch.ones(3, 4, 4)
    a, b = _chunk(zeros, ones), _chunk(zeros, zeros)
    assert not np.allclose(a, b), "the current frame never reached the model"
    assert a[0, 2] == 0 and len(a) == 8, "chunk must start at the current action"


# ── TR-6: gripper_lag sign ──────────────────────────────────────────────

def test_late_prediction_has_positive_lag():
    t = np.arange(200)
    gt = (t > 100).astype(float)
    assert offline_eval.gripper_lag((t > 105).astype(float), gt)[0] == 5
    assert offline_eval.gripper_lag((t > 95).astype(float), gt)[0] == -5


# ── TR-16: ood_check with an 11D (relative) state ──────────────────────

def test_ood_feature_extractor_handles_11d_state():
    seen = {}

    def pre(batch):
        seen["state"] = batch["observation.state"].shape
        return batch

    policy = SimpleNamespace(
        diffusion=SimpleNamespace(rgb_encoder=lambda x: x.flatten(1)[:, :4]),
        config=SimpleNamespace(robot_state_feature=SimpleNamespace(shape=(11,))))
    feats = ood_check.FeatureExtractor(policy, pre, "cpu")([torch.zeros(3, 4, 4)] * 3)
    assert feats.shape == (3, 4) and seen["state"] == (3, 11)


# ── TR-13: the fallback split must match train.py's ────────────────────

def test_fallback_split_honours_exclusions():
    excl = [3, 7]
    remaining = [e for e in range(100) if e not in excl]
    expected = remaining[::10]                     # train.py: stride over the kept set
    assert offline_eval.val_episodes_like_train(100, 0.1, exclude=excl) == expected


def test_fallback_split_unchanged_without_exclusions():
    assert offline_eval.val_episodes_like_train(100, 0.1) == list(range(0, 100, 10))
    assert offline_eval.val_episodes_like_train(20, 0.1, "tail") == [18, 19]


def test_split_is_read_from_a_hub_checkpoint(monkeypatch, tmp_path):
    f = tmp_path / "val_episodes.json"
    f.write_text(json.dumps({"val_episodes": [0, 9]}))
    calls = []

    def fake_download(repo_id, filename, **_k):
        calls.append((repo_id, filename))
        return str(f)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(hf_hub_download=fake_download))
    assert offline_eval.val_episodes_from_checkpoint("user/model-best") == [0, 9]
    assert calls == [("user/model-best", "val_episodes.json")]


def test_missing_hub_split_falls_back(monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("no such file")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=boom))
    assert offline_eval.val_episodes_from_checkpoint("user/old-model") is None
