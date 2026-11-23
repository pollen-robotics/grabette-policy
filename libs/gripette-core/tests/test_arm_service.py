"""Behavioural tests for the shared ArmServicer, with no hardware.

The real-arm server had no tests at all before this: every safety behaviour in
it (IK-jump watchdog, target-lead contact guard, reset ramp, torque gating)
was only ever exercised against a powered robot, which is the worst place to
discover a regression. A fake ArmInterface exercises all of it in-process.

Each test runs against BOTH arms — the 7-DoF OpenArm and the 6-DoF reBot —
because the whole reason this class is shared is that one policy must get
identical behaviour from either. A 6-DoF arm is the interesting case: it has
no null space, so the IK cannot trade orientation against position.

Skipped wholesale without placo or the model packages, i.e. in a base install.
"""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("placo", reason="needs the [kinematics] extra")

from gripette_core.arm_service import ArmServicer  # noqa: E402
from gripette_core.kinematics import ArmSpec, Kinematics  # noqa: E402
from gripette_core.rotation import rotation_matrix_to_6d  # noqa: E402


# --------------------------------------------------------------------------- #
# Arm specs under test
# --------------------------------------------------------------------------- #

def _openarm_spec():
    m = pytest.importorskip("openarm_gripette_model")
    return ArmSpec(
        model_dir=str(m.OPENARM_RIGHT_DIR),
        joint_names=("r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_elbow",
                     "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch"),
        masked_dofs=("proximal", "distal", "r_wrist_roll_mimic"),
    ), np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])


def _rebot_spec():
    m = pytest.importorskip("rebot_gripette_model")
    # Home is NOT all-zeros: shoulder_lift and elbow_flex both start at 0, so
    # zero sits exactly on two hard stops. Pick a pose inside the envelope.
    return ArmSpec(
        model_dir=str(m.REBOT_B601RS_DIR),
        joint_names=("shoulder_pan", "shoulder_lift", "elbow_flex",
                     "wrist_flex", "wrist_yaw", "wrist_roll"),
        masked_dofs=("proximal", "distal"),
    ), np.deg2rad([0.0, 60.0, 60.0, 0.0, 0.0, 0.0])


ARMS = {"openarm_7dof": _openarm_spec, "rebot_6dof": _rebot_spec}


class FakeArm:
    """An ArmInterface that tracks perfectly, or refuses to move.

    `stalled=True` models the arm pressing into something: commands are
    accepted and ignored, which is exactly the condition the target-lead
    guard exists to detect.
    """

    def __init__(self, n, start=None):
        self._lock = threading.Lock()
        self.positions = np.zeros(n) if start is None else np.array(start, float)
        self.stalled = False
        self.torque = True
        self.commands = []
        self.torque_calls = []
        # A real driver may REFUSE to report a pose rather than invent one: the
        # reBot raises when a motor sent no feedback frame, because the plugin
        # would otherwise substitute 0.0 deg and zero is a hard stop on two of
        # its joints.
        self.read_fails = False

    def get_positions(self):
        with self._lock:
            if self.read_fails:
                raise RuntimeError("no feedback from ['elbow_flex']")
            return self.positions.copy()

    def send_command_rad(self, joint_angles_rad):
        with self._lock:
            self.commands.append(np.asarray(joint_angles_rad).copy())
            if not self.stalled:
                self.positions = np.asarray(joint_angles_rad, float).copy()

    def set_torque(self, enable):
        with self._lock:
            self.torque = enable
            self.torque_calls.append(enable)


@pytest.fixture(scope="session")
def kin_cache():
    """Build each arm's Kinematics once per session.

    Placo parses the URDF and every mesh it references on construction — 121
    of them for the reBot — which dominated the runtime of this file when it
    was per-test (4m18s for 34 tests). The solver carries per-instance state
    (the frame task target, the joint configuration), but every test below
    sets that state explicitly through the servicer before asserting on it,
    so sharing one instance per arm is safe here.
    """
    return {}


@pytest.fixture(params=list(ARMS), ids=list(ARMS))
def rig(request, kin_cache):
    """(servicer, fake arm, kinematics, home) for one arm, torn down cleanly."""
    spec, home = ARMS[request.param]()
    if request.param not in kin_cache:
        kin_cache[request.param] = Kinematics(spec)
    kin = kin_cache[request.param]
    arm = FakeArm(len(spec.joint_names), start=home)
    # interp_hz high so tests converge fast; alpha 1.0 = no lag, which keeps
    # the assertions about *logic* free of assertions about settling time.
    servicer = ArmServicer(arm, kin, home, start_time=0.0,
                           interp_hz=200.0, interp_alpha=1.0)
    yield servicer, arm, kin, home
    servicer.stop()


def _settle(seconds=0.25):
    time.sleep(seconds)


class _Ctx:
    """Minimal stand-in for a grpc ServicerContext (unused by this servicer)."""


# --------------------------------------------------------------------------- #
# Basics
# --------------------------------------------------------------------------- #

def test_get_arm_state_is_fk_of_measured_joints(rig):
    servicer, arm, kin, home = rig
    st = servicer.GetArmState(None, _Ctx())
    expected = kin.forward(arm.get_positions())
    assert np.allclose(st.joint_positions, arm.get_positions())
    assert st.x == pytest.approx(expected[0, 3], abs=1e-6)
    assert st.y == pytest.approx(expected[1, 3], abs=1e-6)
    assert st.z == pytest.approx(expected[2, 3], abs=1e-6)
    assert len(st.r6d) == 6


def test_joint_count_follows_the_arm(rig):
    """The servicer used to hardcode 7. A 6-DoF arm must report 6."""
    servicer, arm, kin, home = rig
    st = servicer.GetArmState(None, _Ctx())
    assert len(st.joint_positions) == len(kin.joint_names)


def test_ping_reports_uptime(rig):
    servicer, _, _, _ = rig
    assert servicer.Ping(None, _Ctx()).status == "ok"


def test_home_joints_length_is_validated(rig):
    """A home pose of the wrong length is a configuration bug; fail loudly."""
    _, arm, kin, _ = rig
    with pytest.raises(ValueError, match="home_joints"):
        ArmServicer(arm, kin, np.zeros(len(kin.joint_names) + 1), start_time=0.0)


# --------------------------------------------------------------------------- #
# Cartesian deltas
# --------------------------------------------------------------------------- #

class _Delta:
    def __init__(self, dx=0.0, dy=0.0, dz=0.0, dr6d=None):
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dr6d = dr6d if dr6d is not None else rotation_matrix_to_6d(np.eye(3)).tolist()


def test_a_small_delta_moves_the_arm(rig):
    servicer, arm, kin, _ = rig
    before = kin.forward(arm.get_positions())[:3, 3].copy()
    resp = servicer.SendCartesianDelta(_Delta(dz=0.005), _Ctx())
    assert resp.success, resp.error
    _settle()
    after = kin.forward(arm.get_positions())[:3, 3]
    assert np.linalg.norm(after - before) > 1e-4, "arm did not move at all"


def test_dr6d_must_have_six_values(rig):
    servicer, _, _, _ = rig
    resp = servicer.SendCartesianDelta(_Delta(dr6d=[0.0, 0.0, 0.0]), _Ctx())
    assert not resp.success
    assert "6 values" in resp.error


def test_a_rejected_delta_is_rolled_back_not_leaked(rig):
    """A failed command must not leave its delta in the integrator."""
    servicer, _, _, _ = rig
    before = servicer._target_pos.copy()
    servicer.SendCartesianDelta(_Delta(dz=0.01, dr6d=[1.0, 2.0]), _Ctx())
    assert np.allclose(servicer._target_pos, before)


# --------------------------------------------------------------------------- #
# The guards — the reason this class is worth sharing rather than copying
# --------------------------------------------------------------------------- #

def test_target_lead_guard_rejects_when_the_arm_does_not_track(rig):
    """Arm stalled (in contact): the open-loop target must not run away."""
    servicer, arm, _, _ = rig
    arm.stalled = True
    errors = []
    for _ in range(12):
        r = servicer.SendCartesianDelta(_Delta(dz=0.02), _Ctx())
        if not r.success:
            errors.append(r.error)
    assert errors, "target marched ahead of a stalled arm with no complaint"
    assert any("lead" in e or "contact" in e for e in errors), errors


def test_contact_relaxes_the_target_to_the_measured_pose(rig):
    """>=3 lead trips in 8 commands = confirmed contact; press must release."""
    servicer, arm, kin, _ = rig
    arm.stalled = True
    saw_relax = False
    for _ in range(20):
        r = servicer.SendCartesianDelta(_Delta(dz=0.02), _Ctx())
        if not r.success and "relaxed" in r.error:
            saw_relax = True
            break
    assert saw_relax, "contact was never confirmed; press force would grow"
    measured_fk = kin.forward(arm.get_positions())
    assert np.allclose(servicer._target_pos, measured_fk[:3, 3], atol=1e-6)


def test_lead_reducing_commands_still_pass_under_contact(rig):
    """Retreats must always be accepted, or corrections look ignored."""
    servicer, arm, _, _ = rig
    arm.stalled = True
    for _ in range(6):
        servicer.SendCartesianDelta(_Delta(dz=0.02), _Ctx())
    arm.stalled = False
    r = servicer.SendCartesianDelta(_Delta(dz=-0.02), _Ctx())
    assert r.success or "lead" not in r.error, r.error


def test_ik_jump_watchdog_latches_and_freezes_motion(rig):
    """After the latch, commands are refused outright until Reset."""
    servicer, _, _, _ = rig
    servicer._interp_enabled = False      # the post-latch state
    r = servicer.SendCartesianDelta(_Delta(dz=0.001), _Ctx())
    assert not r.success
    assert "frozen" in r.error and "Reset" in r.error


# --------------------------------------------------------------------------- #
# Reset / torque
# --------------------------------------------------------------------------- #

class _ResetReq:
    def __init__(self, joint_positions=()):
        self.joint_positions = list(joint_positions)


def test_reset_ramps_to_home_and_rearms_the_watchdog(rig):
    servicer, arm, _, home = rig
    servicer._interp_enabled = False
    servicer._ik_jump_violations = 99
    resp = servicer.Reset(_ResetReq(), _Ctx())
    assert resp.success, resp.error
    assert np.allclose(arm.get_positions(), home, atol=1e-6)
    assert servicer._ik_jump_violations == 0
    assert servicer._interp_enabled


def test_reset_accepts_an_explicit_pose_of_the_right_length(rig):
    servicer, arm, kin, home = rig
    want = home + np.deg2rad(2.0)
    resp = servicer.Reset(_ResetReq(want.tolist()), _Ctx())
    assert resp.success
    assert np.allclose(arm.get_positions(), want, atol=1e-6)


def test_reset_with_a_wrong_length_pose_falls_back_to_home(rig):
    """A 7-value reset sent to the 6-DoF arm must not be applied blindly."""
    servicer, arm, kin, home = rig
    wrong = [0.1] * (len(kin.joint_names) + 1)
    resp = servicer.Reset(_ResetReq(wrong), _Ctx())
    assert resp.success
    assert np.allclose(arm.get_positions(), home, atol=1e-6)


def test_reset_ramps_rather_than_jumping(rig):
    """A single large step to home would be violent on a real arm."""
    servicer, arm, _, home = rig
    far = home + np.deg2rad(20.0)
    servicer.Reset(_ResetReq(far.tolist()), _Ctx())
    arm.commands.clear()
    servicer.Reset(_ResetReq(home.tolist()), _Ctx())
    steps = [c for c in arm.commands]
    assert len(steps) > 50, f"reset issued only {len(steps)} commands"
    biggest = max(float(np.max(np.abs(b - a))) for a, b in zip(steps, steps[1:]))
    assert biggest < np.deg2rad(5.0), f"reset step of {np.rad2deg(biggest):.1f} deg"


def test_set_torque_false_stops_commanding(rig):
    servicer, arm, _, _ = rig
    assert servicer.SetTorque(type("R", (), {"enable": False})(), _Ctx()).success
    assert arm.torque is False
    assert servicer._latest_target_joints is None
    arm.commands.clear()
    _settle(0.15)
    assert not arm.commands, "kept streaming commands after torque off"


def test_set_torque_true_holds_the_current_pose(rig):
    """A hand-placed arm must stay where it was put, not sag or snap."""
    servicer, arm, _, _ = rig
    servicer.SetTorque(type("R", (), {"enable": False})(), _Ctx())
    moved = arm.get_positions() + np.deg2rad(3.0)
    with arm._lock:
        arm.positions = moved.copy()
    servicer.SetTorque(type("R", (), {"enable": True})(), _Ctx())
    assert arm.torque is True
    assert np.allclose(servicer._latest_target_joints, moved, atol=1e-9)
    _settle()
    assert np.allclose(arm.get_positions(), moved, atol=1e-3)


# --------------------------------------------------------------------------- #
# A driver that refuses to report a pose
# --------------------------------------------------------------------------- #

def test_the_interp_thread_survives_a_failing_read(rig):
    """A raising get_positions() must skip a tick, not kill the interpolator.

    The seeding read sits in the interp loop, and nothing joins that thread —
    so an escaping exception would stop the arm being driven with no error
    surfacing anywhere. This could not happen while drivers substituted 0.0 for
    a missing reading; it can now that the reBot refuses to, which is why the
    read belongs inside the loop's try.
    """
    servicer, arm, _, _ = rig
    servicer.SetTorque(type("R", (), {"enable": True})(), _Ctx())
    assert servicer._interp_thread.is_alive()

    # Force the loop back through its seeding path, then break the read.
    arm.read_fails = True
    servicer._current_cmd_joints = None
    _settle(0.15)
    assert servicer._interp_thread.is_alive(), "interp thread died on a failed read"
    arm.commands.clear()
    _settle(0.1)
    assert not arm.commands, "commanded a pose while unable to read one"

    # And it recovers once feedback returns, rather than needing a restart.
    arm.read_fails = False
    _settle(0.2)
    assert arm.commands, "interpolator never resumed after the read recovered"
