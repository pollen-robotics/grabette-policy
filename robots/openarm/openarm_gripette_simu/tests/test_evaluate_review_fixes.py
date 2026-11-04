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


def run(ev, policy, arm=None, gripper=None, camera=None, max_steps=20, fps=1000.0, **kw):
    arm = arm or FakeArm()
    gripper = gripper or FakeGripper()
    camera = camera or FakeCamera(gripper)
    result = ev.run_episode(
        policy=policy, preprocessor=lambda b: b, postprocessor=lambda a: a,
        arm_stub=arm, gripper_stub=gripper,
        arm_pb2=fake_pb2(), gripper_pb2=fake_grip_pb2(),
        camera=camera, device=torch.device("cpu"),
        max_steps=max_steps, fps=fps, success_check_freq=1000,
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


# ── main() / argument handling (no policy load, no network) ────────────────

class _ReachedPolicyLoad(Exception):
    """main() got past argument validation to the policy load."""


def run_main(ev, monkeypatch, argv):
    """Run main() with argv; stop it at the policy load so nothing is fetched."""
    def _no_load(_ckpt):
        raise _ReachedPolicyLoad
    monkeypatch.setattr(ev, "_load_policy_any", _no_load)
    monkeypatch.setattr("sys.argv", ["evaluate.py", *argv])
    ev.main()


# ── EV-3: --async_exec does not implement the grasp projection ─────────────

def test_async_exec_with_the_grasp_projection_is_refused(ev, monkeypatch):
    with pytest.raises(SystemExit, match="grasp_projection"):
        run_main(ev, monkeypatch, ["--checkpoint", "/nonexistent", "--async_exec",
                                   "--grasp_projection", "on", "--chunk_relative", "off"])


def test_async_exec_with_raw_angles_still_starts(ev, monkeypatch):
    with pytest.raises(_ReachedPolicyLoad):
        run_main(ev, monkeypatch, ["--checkpoint", "/nonexistent", "--async_exec",
                                   "--grasp_projection", "off", "--chunk_relative", "off"])


# ── EV-4: --max_ticks > 1 must not generate from a duplicated obs pair ─────

class QueueDiffusion:
    """lerobot DiffusionPolicy.select_action's contract: EVERY call appends the
    batch to an obs queue (maxlen n_obs_steps=2, primed by duplicating the first
    obs), and a chunk is generated from that queue when the action queue is
    empty. Records the obs pair each generation saw."""

    N_ACT = 3

    def __init__(self):
        from collections import deque
        self._obs = deque(maxlen=2)
        self._queues = {"action": deque()}
        self.config = SimpleNamespace(n_action_steps=self.N_ACT)
        self.generation_pairs = []

    def select_action(self, batch):
        frame_id = float(batch["observation.state"][0, 0])
        if not self._obs:
            self._obs.extend([frame_id, frame_id])
        else:
            self._obs.append(frame_id)
        if not self._queues["action"]:
            self.generation_pairs.append(tuple(self._obs))
            self._queues["action"].extend(
                torch.from_numpy(hold_action((0.0, 0.0)).astype(np.float32))
                for _ in range(self.N_ACT))
        time.sleep(0.004)          # slower than the 2 ms tick -> catch-up kicks in
        return self._queues["action"].popleft().unsqueeze(0)


class FrameCounterCamera(FakeCamera):
    """A new, numbered frame per observation (state[0] = frame id)."""

    def get(self):
        self.n += 1
        return (np.zeros((8, 8, 3), np.uint8), np.array([self.n, 0.0], np.float32),
                time.perf_counter() * 1000.0)


@pytest.mark.parametrize("max_ticks", [1, 2, 3])
def test_no_generation_sees_a_duplicated_obs_pair(ev, max_ticks):
    pol = QueueDiffusion()
    gripper = FakeGripper()
    arm, *_ = run(ev, pol, gripper=gripper, camera=FrameCounterCamera(gripper),
                  max_steps=15, fps=500.0, max_ticks=max_ticks)
    later = pol.generation_pairs[1:]             # the first is primed by design
    assert len(later) >= 3
    dup = [p for p in later if p[0] == p[1]]
    assert not dup, f"generations from a duplicated pair: {dup} of {later}"
    if max_ticks > 1:
        assert len(arm.deltas) > 15, "catch-up never sent more than one tick/iteration"


# ── EV-5: --grip_gain must stay inside what the gripper service accepts ────

@pytest.mark.parametrize("g1,g2", [(1.3, 1.9), (-1.3, 1.9)])   # +/- closing models
def test_grip_gain_respects_the_service_limits(ev, g1, g2):
    import struct
    p, d = ev.apply_grip_gain(g1, g2, 1.5, (0.0, 0.0))
    for value, (_lo, hi) in zip((p, d), ev.GRIPPER_LIMITS):
        on_wire = struct.unpack("f", struct.pack("f", value))[0]
        assert -hi <= on_wire <= hi, f"{on_wire} outside +/-{hi}"
    assert abs(p) > 1.4 and abs(d) > 1.5            # still a deep close, just bounded


def test_grip_gain_one_is_still_the_identity(ev):
    # Gain 1.0 must stay a pure pass-through (sign conventions vary by model).
    assert ev.apply_grip_gain(-1.55, 1.7, 1.0, (0.0, 0.0)) == (-1.55, 1.7)


# ── main() with a fake loaded policy (no checkpoint, no server) ────────────

class _ReachedConnect(Exception):
    """main() got past model loading to the gRPC connection."""


class _FakeLoadedPolicy:
    def __init__(self, state_dim, action_dim):
        self.config = SimpleNamespace(
            type="diffusion", n_action_steps=8,
            robot_state_feature=SimpleNamespace(shape=(state_dim,)),
            action_feature=SimpleNamespace(shape=(action_dim,)))

    def to(self, _device):
        return self

    def eval(self):
        return self


def run_main_loaded(ev, monkeypatch, argv, state_dim=2, action_dim=11):
    """main() with a fake policy; stops at the first gRPC channel."""
    def _no_channel(_addr):
        raise _ReachedConnect
    monkeypatch.setattr(ev, "_load_policy_any",
                        lambda _c: _FakeLoadedPolicy(state_dim, action_dim))
    monkeypatch.setattr(ev, "make_pre_post_processors",
                        lambda *a, **k: (lambda b: b, SimpleNamespace(steps=[])))
    monkeypatch.setattr(ev.grpc, "insecure_channel", _no_channel)
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--device", "cpu", "--checkpoint",
                                     "/nonexistent", "--grasp_projection", "off",
                                     "--chunk_relative", "off", *argv])
    ev.main()


# ── SIM-8: --joint_mode needs an RPC no arm.proto defines ──────────────────

def test_joint_space_checkpoint_is_refused_before_connecting(ev, monkeypatch):
    with pytest.raises(SystemExit, match="SendJointCommand"):
        run_main_loaded(ev, monkeypatch, [], state_dim=9, action_dim=9)


def test_cartesian_checkpoint_still_connects(ev, monkeypatch):
    with pytest.raises(_ReachedConnect):
        run_main_loaded(ev, monkeypatch, [])


# ── EV-14: a frozen-arm episode still reports the real displacement ────────

def test_frozen_episode_reports_the_measured_displacement(ev):
    arm = FakeArm(displacement=0.031)
    arm.frozen = True
    _a, _g, _c, r = run(ev, ConstPolicy(hold_action((0.0, 0.0))), arm=arm, max_steps=5)
    assert not r["success"] and r["steps"] == 1
    assert r["displacement_mm"] == pytest.approx(31.0, abs=1e-6)


# ── EV-15: flags that a mode silently ignores now say so ───────────────────

def _parse(ev, monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--checkpoint", "x", *argv])
    return ev.parse_args()


@pytest.mark.parametrize("argv,flag", [
    (["--commit_close", "0.5"], "--commit_close"),                        # sync mode
    (["--policy_addr", "h:1", "--skip_stale"], "--skip_stale"),           # remote
    (["--grip_assist", "50", "--latch_close", "0.5"], "--latch_close"),
    (["--grip_assist", "50", "--grip_gain", "1.3"], "--grip_gain"),
    (["--async_exec", "--max_ticks", "3"], "--max_ticks"),
    (["--async_exec", "--skip_stale"], "--skip_stale"),
])
def test_inert_flag_combinations_warn(ev, monkeypatch, caplog, argv, flag):
    with caplog.at_level("WARNING"):
        _parse(ev, monkeypatch, argv)
    assert any(flag in r.message and "ignored" in r.message for r in caplog.records), \
        caplog.text


def test_meaningful_flags_do_not_warn(ev, monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        _parse(ev, monkeypatch, ["--async_exec", "--commit_close", "0.5",
                                 "--latch_close", "0.5", "--grip_gain", "1.3"])
        _parse(ev, monkeypatch, ["--skip_stale", "--max_ticks", "2", "--latch_close", "0.5"])
    assert "ignored" not in caplog.text
