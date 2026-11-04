"""Regression tests for evaluate.py fixes from the 2026-11 code review (EV-*).

Like test_chunk_relative_eval.py, these load the REAL examples/evaluate.py and
drive its functions / run_episode() against small fakes, so a fix is checked in
the code that ships rather than in a mirror of it.

Needs the `eval` extra (evaluate.py imports lerobot/torch).
"""

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_EVALUATE = Path(__file__).resolve().parents[1] / "examples" / "evaluate.py"

torch = pytest.importorskip("torch")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

from openarm_gripette_simu.rotation import delta_6d_to_matrix  # noqa: E402


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("_evaluate", _EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _server_angle_deg(dr6):
    """Rotation the arm server would actually apply for this 6D delta."""
    R, _ = delta_6d_to_matrix(dr6)
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


# ── EV-1: --clamp_rot_deg must not turn a "hold" into a rotation ──────────

@pytest.mark.parametrize("dr6", [
    np.array([0.003, -0.002, 0.001, 0.0, 0.002, -0.001]),   # typical hold noise
    np.random.default_rng(0).normal(scale=0.08, size=6),    # reviewer's repro
])
def test_clamp_keeps_a_degenerate_hold_a_hold(ev, dr6):
    assert _server_angle_deg(dr6) == 0.0          # the server reads it as "hold"
    _dp, out, _was = ev.clamp_delta(np.zeros(3), dr6.copy(), None, np.radians(2.0))
    assert _server_angle_deg(out) == 0.0, "clamp re-encoded a hold as a real rotation"


def test_clamp_of_an_exactly_zero_6d_does_not_raise(ev):
    # Raising here kills the async executor thread (run_episode_async then spins).
    dp, out, was = ev.clamp_delta(np.zeros(3), np.zeros(6), 0.005, np.radians(2.0))
    assert _server_angle_deg(out) == 0.0 and not was


def test_clamp_still_caps_a_real_rotation(ev):
    from openarm_gripette_simu.rotation import rotation_matrix_to_6d
    dr6 = rotation_matrix_to_6d(Rotation.from_rotvec([0, 0, np.radians(10)]).as_matrix())
    _dp, out, was = ev.clamp_delta(np.zeros(3), dr6, None, np.radians(2.0))
    assert was and abs(_server_angle_deg(out) - 2.0) < 1e-6


# ── fakes for run_episode() ───────────────────────────────────────────────

IDENTITY_R6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def hold_action(grip):
    """An 11D per-step action: no motion, the given gripper channels."""
    return np.concatenate([np.zeros(3), IDENTITY_R6D, np.asarray(grip, float)])


class ConstPolicy:
    """Returns the same action every call (no chunk structure needed)."""

    def __init__(self, action):
        self._a = torch.from_numpy(np.asarray(action, np.float32))
        self.config = SimpleNamespace(n_action_steps=8)

    def select_action(self, batch):
        return self._a.unsqueeze(0)


class FakeArm:
    def __init__(self, displacement=0.0):
        self.deltas = []
        self.status_calls = 0
        self._disp = displacement
        self.frozen = False

    def SendCartesianDelta(self, msg):
        self.deltas.append((np.array([msg.dx, msg.dy, msg.dz]), np.array(msg.dr6d)))
        if self.frozen:
            return SimpleNamespace(success=False, error="motion frozen by watchdog")
        return SimpleNamespace(success=True, error="")

    def GetSuccessStatus(self, _req):
        self.status_calls += 1
        return SimpleNamespace(goal_reached=False, cube_displacement=self._disp)

    def GetArmState(self, _req):
        return SimpleNamespace(joint_positions=[0.0] * 7)


class FakeGripper:
    def __init__(self, start=(0.0, 0.0)):
        self.goals = []
        self.pos = tuple(start)   # where the (free, unobstructed) fingers are

    def SendMotorCommand(self, msg):
        self.goals.append((msg.motor1_goal, msg.motor2_goal))
        self.pos = (msg.motor1_goal, msg.motor2_goal)   # arrives instantly
        return SimpleNamespace(success=True)


class FakeCamera:
    """Reports the gripper's TRUE joint angles, like the gripette service."""

    def __init__(self, gripper, load=(0.0, 0.0)):
        self._g = gripper
        self._load = load
        self.n = 0

    def get(self):
        self.n += 1
        return (np.zeros((8, 8, 3), np.uint8), np.array(self._g.pos, np.float32),
                time.perf_counter() * 1000.0)

    def get_load(self):
        return self._load


def fake_pb2():
    return SimpleNamespace(
        CartesianDelta=lambda dx, dy, dz, dr6d: SimpleNamespace(dx=dx, dy=dy, dz=dz, dr6d=dr6d),
        SuccessStatusRequest=lambda: None,
        GetArmStateRequest=lambda: None,
    )


def fake_grip_pb2():
    return SimpleNamespace(
        MotorCommand=lambda motor1_goal, motor2_goal, motor1_torque_limit=0.0,
        motor2_torque_limit=0.0: SimpleNamespace(motor1_goal=motor1_goal,
                                                 motor2_goal=motor2_goal))


def run(ev, policy, arm=None, gripper=None, camera=None, max_steps=20, **kw):
    arm = arm or FakeArm()
    gripper = gripper or FakeGripper()
    camera = camera or FakeCamera(gripper)
    result = ev.run_episode(
        policy=policy, preprocessor=lambda b: b, postprocessor=lambda a: a,
        arm_stub=arm, gripper_stub=gripper,
        arm_pb2=fake_pb2(), gripper_pb2=fake_grip_pb2(),
        camera=camera, device=torch.device("cpu"),
        max_steps=max_steps, fps=1000.0, success_check_freq=1000,
        debug=False, use_relative_proprio=False, start_pos=None, start_rot=None,
        task="t", **kw)
    return arm, gripper, camera, result


# ── EV-2: GripAssist must see measured angles, not (strategy, closure) ────

def test_grip_assist_measures_lag_in_radians_under_the_projection(ev):
    """Free fingers that ARRIVED at the sent pose have ~zero lag. Fed the
    projected (s, c) instead of angles, the assist read |angle - closure| as a
    stall and latched GRIPPED on an empty gripper (holding load ~60)."""
    from grabette_dataset_tools.grasp_projection import GraspProjection

    assist = ev.GripAssist(50.0, ref=(0.0, 0.0), stable_ticks=3)
    gripper = FakeGripper()
    _arm, _g, _c, _r = run(
        ev, ConstPolicy(hold_action((0.5, 0.6))),
        gripper=gripper, camera=FakeCamera(gripper, load=(60.0, 10.0)), max_steps=12,
        grasp_projection=GraspProjection(), grip_assist=assist)
    assert assist.lag == assist.lag and assist.lag < 0.01, f"lag {assist.lag}"
    assert assist.state != "GRIPPED" and assist.n_gripped == 0
