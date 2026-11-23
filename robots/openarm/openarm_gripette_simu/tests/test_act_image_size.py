"""Tests for evaluate.py's ACT frame rescaling.

ACT has no internal resize, so a Gripette frame at another size than the
training images must be rescaled before inference; other policies resize
internally and are left untouched.

Needs the `eval` extra (evaluate.py imports lerobot/torch).
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_EVALUATE = Path(__file__).resolve().parents[1] / "examples" / "evaluate.py"

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("_evaluate", _EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy(kind, shape=(3, 720, 960)):
    feature = SimpleNamespace(shape=shape)
    return SimpleNamespace(config=SimpleNamespace(
        type=kind, input_features={"observation.images.cam0": feature}))


def test_act_reads_the_training_size(ev):
    assert ev.act_image_size(policy("act")) == (720, 960)


def test_other_policies_are_not_rescaled(ev):
    assert ev.act_image_size(policy("diffusion")) is None
    assert ev.act_image_size(None) is None             # remote client mode
    assert ev.act_image_size(SimpleNamespace()) is None  # fake policy without config


def test_gripette_frame_is_rescaled_to_the_training_size(ev):
    frame = np.zeros((486, 648, 3), np.uint8)
    out = ev.match_image_size(frame, (720, 960))
    assert out.shape == (720, 960, 3) and out.dtype == np.uint8


def test_matching_frame_or_no_target_is_untouched(ev):
    frame = np.zeros((486, 648, 3), np.uint8)
    assert ev.match_image_size(frame, (486, 648)) is frame
    assert ev.match_image_size(frame, None) is frame
