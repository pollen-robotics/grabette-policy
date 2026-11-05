"""What a checkpoint's action numbers MEAN, on an arm of a given joint count.

The width is the only evidence the checkpoint offers, and on a 6-DoF arm it
runs out: joint space is `arm_dof + 2` = 8D, and chunk-relative is Cartesian
and therefore 8D on any arm. The two readings are not close - six joint angles
read as chunk-relative become three metres of translation and a rotation
vector - so an eval that guesses here drives a real arm to an arbitrary pose
instead of failing.
"""
import importlib.util
import pathlib
import sys

import pytest

_EV = pathlib.Path(__file__).resolve().parents[1] / "examples" / "evaluate.py"


@pytest.fixture(scope="module")
def ev():
    """Import evaluate.py without running it, and without lerobot."""
    spec = importlib.util.spec_from_file_location("_evaluate_under_test", _EV)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # pragma: no cover
        pytest.skip(f"evaluate.py needs the eval extra ({e})")
    return mod


# --------------------------------------------------------------------------- #
# Unambiguous widths behave as they always did
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm_dof", [6, 7])
def test_11d_is_cartesian_on_any_arm(ev, arm_dof):
    """Cartesian deltas say nothing about joint count, so they are portable -
    which is the whole reason one policy can drive either arm."""
    assert ev.resolve_action_space(11, arm_dof, None, None) is False


def test_9d_on_a_7dof_arm_is_joint_space(ev):
    assert ev.resolve_action_space(9, 7, None, None) is True


def test_8d_on_a_7dof_arm_is_chunk_relative(ev):
    """Unchanged from before: 7 + 2 = 9, so 8D cannot be joint space there."""
    assert ev.resolve_action_space(8, 7, True, None) is False


def test_8d_on_a_7dof_arm_without_the_flag_still_refuses(ev):
    """The pre-existing guard: running chunk-relative actions as per-step
    deltas reads the rotation's last dim as a gripper command."""
    with pytest.raises(SystemExit, match="--chunk_relative on"):
        ev.resolve_action_space(8, 7, None, None)


# --------------------------------------------------------------------------- #
# The 6-DoF collision
# --------------------------------------------------------------------------- #

def test_8d_on_a_6dof_arm_refuses_rather_than_guessing(ev):
    """6 + 2 = 8, the same width as chunk-relative. With no checkpoint to ask,
    the only safe answer is to stop and say which two readings are in play."""
    with pytest.raises(SystemExit) as e:
        ev.resolve_action_space(8, 6, None, None)
    msg = str(e.value)
    assert "joint space" in msg and "chunk-relative" in msg
    assert "--chunk_relative" in msg


def test_the_flag_settles_the_6dof_collision(ev):
    assert ev.resolve_action_space(8, 6, True, None) is False


def test_the_checkpoint_settles_the_6dof_collision(ev, monkeypatch, tmp_path):
    """Preferred over the flag because it is evidence, not an assertion: the
    preprocessor config records the registered step names."""
    monkeypatch.setattr(ev, "detect_chunk_relative", lambda ckpt: False)
    assert ev.resolve_action_space(8, 6, None, str(tmp_path)) is True

    monkeypatch.setattr(ev, "detect_chunk_relative", lambda ckpt: True)
    assert ev.resolve_action_space(8, 6, None, str(tmp_path)) is False


def test_an_unreadable_checkpoint_still_refuses(ev, monkeypatch, tmp_path):
    """detect_chunk_relative returns None when it cannot tell. That is not a
    licence to pick one."""
    monkeypatch.setattr(ev, "detect_chunk_relative", lambda ckpt: None)
    with pytest.raises(SystemExit, match="could not settle"):
        ev.resolve_action_space(8, 6, None, str(tmp_path))


def test_a_width_that_fits_nothing_is_refused(ev):
    with pytest.raises(SystemExit, match="emits 12D"):
        ev.resolve_action_space(12, 6, None, None)


def test_joint_space_width_follows_the_arm(ev):
    """8D is joint space on a 6-DoF arm and 9D on a 7-DoF one. A fixed `== 9`
    would simply never see the reBot's joint-space checkpoints."""
    assert ev.resolve_action_space(8, 6, True, None) is False   # flag wins
    assert ev.resolve_action_space(9, 7, None, None) is True
    with pytest.raises(SystemExit):
        ev.resolve_action_space(9, 6, None, None)               # 6+2 != 9
