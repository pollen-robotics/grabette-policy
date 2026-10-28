"""The parts of the bring-up examples that can be checked without an arm.

The examples exist to be run against hardware, but the arithmetic they do
BEFORE commanding anything is exactly where a mistake is expensive: a trajectory
duration, a waypoint list, an abort threshold. All of that is pure, so it is
tested here and the hardware run is left to confirm only what hardware can.

Importing an example must not open a bus: motorbridge and the plugin are
imported inside main(), never at module level. These tests would fail at
collection if that regressed.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import _bringup  # noqa: E402


# --------------------------------------------------------------------------- #
# Importing an example must touch no hardware
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["read_arm_state", "reset_arm", "set_arm_pose",
                                  "cartesian_square", "read_torque_limits",
                                  "mirror_arm", "capture_pose", "check_connect",
                                  "measure_release"])
def test_example_imports_without_a_bus(name):
    """A module that opens CAN at import is a trap: an IDE 'run file', a doc
    tool or a stray import would touch the arm. limits.py already had its
    __main__ moved out for this reason."""
    __import__(name)


# --------------------------------------------------------------------------- #
# Trajectory arithmetic
# --------------------------------------------------------------------------- #

def test_ease_starts_and_ends_at_rest():
    assert _bringup.ease(0.0) == pytest.approx(0.0)
    assert _bringup.ease(1.0) == pytest.approx(1.0)
    # Zero velocity at both ends is the point: the derivative vanishes there.
    assert _bringup.ease(0.01) < 0.01
    assert _bringup.ease(0.99) > 0.99


def test_ease_is_monotonic_and_clamped():
    vals = [_bringup.ease(f / 50) for f in range(51)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert _bringup.ease(-1.0) == pytest.approx(0.0)
    assert _bringup.ease(2.0) == pytest.approx(1.0)


def test_leg_duration_scales_with_the_move_and_has_a_floor():
    """A bigger move takes longer, it does not go faster."""
    assert _bringup.leg_duration(20.0, deg_per_s=2.0) == pytest.approx(10.0)
    assert _bringup.leg_duration(40.0, deg_per_s=2.0) == pytest.approx(20.0)
    # A tiny move still gets a floor, so a 0.1 deg correction is not a step.
    assert _bringup.leg_duration(0.1) == _bringup.MIN_LEG_S
    assert _bringup.leg_duration(-20.0, deg_per_s=2.0) == pytest.approx(10.0)


def test_the_rate_limiter_sits_above_gravity_droop():
    """max_relative_target below the 1-2 deg steady-state droop would clamp
    continuously during normal motion (13_end_to_end's finding)."""
    assert _bringup.MAX_RELATIVE_TARGET > 2.0


# --------------------------------------------------------------------------- #
# The square
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def square():
    import cartesian_square
    return cartesian_square


def test_square_starts_and_ends_at_the_start_point(square):
    start = (0.3, 0.0, 0.5)
    pts = square.square_waypoints(start, axis=0, size_m=0.1, steps_per_side=5)
    assert np.allclose(pts[0], start), "the first waypoint must command no motion"
    assert np.allclose(pts[-1], start), "the square must close"


def test_square_lies_in_the_plane_normal_to_the_axis(square):
    start = (0.3, 0.0, 0.5)
    for axis in (0, 1, 2):
        pts = square.square_waypoints(start, axis=axis, size_m=0.1, steps_per_side=4)
        assert all(p[axis] == pytest.approx(start[axis]) for p in pts)


def test_square_has_the_requested_side_length(square):
    pts = square.square_waypoints((0.3, 0.0, 0.5), axis=0, size_m=0.1,
                                  steps_per_side=4)
    pts = np.array(pts)
    for i in (1, 2):
        extent = pts[:, i].max() - pts[:, i].min()
        assert extent == pytest.approx(0.1)


def test_square_visits_four_corners(square):
    """steps_per_side=1 reduces the path to the corners themselves."""
    pts = square.square_waypoints((0.0, 0.0, 0.0), axis=0, size_m=0.1,
                                  steps_per_side=1)
    corners = {(round(p[1], 6), round(p[2], 6)) for p in pts}
    assert corners == {(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)}


# --------------------------------------------------------------------------- #
# Solving it — needs placo and the model, not hardware
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def kin():
    pytest.importorskip("placo", reason="needs the [kinematics] extra")
    pytest.importorskip("rebot_gripette_model")
    from gripette_core.kinematics import Kinematics
    from rebot_gripette.spec import REBOT_SPEC
    return Kinematics(REBOT_SPEC)


@pytest.fixture(scope="module")
def q_start():
    """An arbitrary in-envelope pose. NOT a recommended home pose."""
    return np.deg2rad([0.0, 60.0, 60.0, 0.0, 0.0, 0.0])


def test_a_small_square_solves_cleanly(square, kin, q_start):
    from gripette_core.kinematics import CONTROL_FRAME
    _, report, problems = square.solve_square(
        kin, CONTROL_FRAME, q_start, axis=0, size_m=0.02,
        steps_per_side=10, max_joint_step_deg=15.0)
    assert problems == []
    assert max(r[2] for r in report) < 1.0, "IK residual over 1 mm on a 2 cm square"


def test_an_oversized_square_is_refused_rather_than_commanded(square, kin, q_start):
    """The failure that matters: unreachable along the PATH, not just far away.

    This is the check that has to happen offline, because on 6 DoF the failure
    presents as a branch flip between neighbouring waypoints - a large joint
    step - and by the time the arm shows you that, it has already done it.
    """
    from gripette_core.kinematics import CONTROL_FRAME
    _, _, problems = square.solve_square(
        kin, CONTROL_FRAME, q_start, axis=0, size_m=0.40,
        steps_per_side=10, max_joint_step_deg=15.0)
    assert problems, "a 40 cm square from this pose should not solve"
    assert any("branch flip" in p for p in problems)


def test_the_step_limit_matches_the_servers_watchdog(square):
    """Divergence here would mean the dry run blesses a path the server aborts."""
    src = (EXAMPLES.parent / "rebot_gripette" / "grpc_server_real.py").read_text()
    assert '"--max_ik_jump_deg", type=float, default=15.0' in src


# --------------------------------------------------------------------------- #
# Shared constants stay tied to the model
# --------------------------------------------------------------------------- #

def test_urdf_limits_cover_every_arm_joint():
    pytest.importorskip("rebot_gripette_model")
    from rebot_gripette.spec import ARM_JOINT_NAMES

    limits = _bringup.urdf_limits()
    for name in ARM_JOINT_NAMES:
        lo, hi = limits[name]
        assert lo < hi


def test_the_motor_table_matches_the_spec_order():
    """ARM_MOTORS drives the printouts; a mismatch would mislabel every row."""
    pytest.importorskip("rebot_gripette_model")
    from rebot_gripette.spec import ARM_JOINT_NAMES

    assert tuple(n for n, _, _ in _bringup.ARM_MOTORS) == ARM_JOINT_NAMES
    assert [c for _, c, _ in _bringup.ARM_MOTORS] == [1, 2, 3, 4, 5, 6]
    assert "gripper" not in [n for n, _, _ in _bringup.ARM_MOTORS]


def test_cold_read_reports_missing_joints_instead_of_defaulting():
    """A joint that did not answer must be ABSENT, never 0.0 — zero is a hard
    stop on two of these joints, so a defaulted read is a command into one."""
    class Dead:
        def robstride_get_param_f32(self, *_):
            raise OSError("no answer")

    class Live:
        def robstride_get_param_f32(self, *_):
            return math.radians(12.0)

    handles = {n: (Live() if n == "shoulder_pan" else Dead())
               for n, _, _ in _bringup.ARM_MOTORS}
    got = _bringup.cold_read_positions(handles)
    assert set(got) == {"shoulder_pan"}
    assert got["shoulder_pan"] == pytest.approx(12.0)
    with pytest.raises(SystemExit, match="nothing was enabled"):
        _bringup.require_full_read(got)


# --------------------------------------------------------------------------- #
# The live mirror (rung 0)
# --------------------------------------------------------------------------- #

def test_the_mirror_cannot_energize_the_arm():
    """Its whole safety claim is textual: no enable, no command, anywhere.

    Unlike the other rungs there is no guard to assert on, because the point
    is the ABSENCE of a code path. A future edit that adds a compliant mode to
    this file must also move it off rung 0.
    """
    src = (EXAMPLES / "mirror_arm.py").read_text()
    body = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    body = body.split('"""', 2)[-1]          # drop the module docstring
    for forbidden in ("enable(", "send_mit(", "send_action(", "disable("):
        assert forbidden not in body, f"mirror_arm.py may not call {forbidden}"


def test_the_mirror_drives_the_model_from_motor_space_unchanged():
    mirror = __import__("mirror_arm")
    pose = {"elbow_flex": 60.0, "shoulder_pan": 10.0}
    assert mirror.to_space(pose, "motor", None) == pose


def test_the_vendor_mapping_mirrors_exactly_the_three_reversed_joints():
    """The prediction the hardware run is asked to confirm by eye."""
    mirror = __import__("mirror_arm")
    directions = {"shoulder_pan": 1.0, "shoulder_lift": 1.0, "elbow_flex": -1.0,
                  "wrist_flex": -1.0, "wrist_yaw": -1.0, "wrist_roll": 1.0}
    pose = {n: 30.0 for n in directions}
    got = mirror.to_space(pose, "vendor", directions)
    for name, direction in directions.items():
        assert got[name] == pytest.approx(30.0 * direction)


def test_a_silent_joint_is_absent_so_the_viewer_holds_its_last_pose():
    """Zero is a hard stop on two joints; a defaulted read would draw the arm
    folded onto them and look like a real pose."""
    mirror = __import__("mirror_arm")

    class Dead:
        def robstride_get_param_f32(self, *_):
            raise OSError("no answer")

    class Live:
        def robstride_get_param_f32(self, *_):
            return math.radians(45.0)

    handles = {n: (Dead() if n == "elbow_flex" else Live())
               for n, _, _ in _bringup.ARM_MOTORS}
    got = mirror.read_positions(handles)
    assert "elbow_flex" not in got
    assert got["shoulder_lift"] == pytest.approx(45.0)


# --------------------------------------------------------------------------- #
# Capturing a held pose
# --------------------------------------------------------------------------- #

def test_the_median_rejects_a_single_drifted_sample():
    """Why the median and not the mean: one sample taken as the holder adjusts
    their grip must not drag the recorded pose with it."""
    capture = __import__("capture_pose")
    steady = [{"elbow_flex": 60.0}, {"elbow_flex": 60.2}, {"elbow_flex": 59.9}]
    drifted = steady + [{"elbow_flex": 95.0}]
    median, spread, n = capture.summarise(drifted)["elbow_flex"]
    assert median == pytest.approx(60.1)
    assert n == 4
    # The spread still reports the excursion - it is not hidden, just not averaged in.
    assert spread == pytest.approx(35.1)


def test_the_spread_is_peak_to_peak_per_joint():
    capture = __import__("capture_pose")
    samples = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 10.5}, {"a": 2.0, "b": 9.5}]
    stats = capture.summarise(samples)
    assert stats["a"] == (pytest.approx(2.0), pytest.approx(2.0), 3)
    assert stats["b"] == (pytest.approx(10.0), pytest.approx(1.0), 3)


# --------------------------------------------------------------------------- #
# Measuring the release damping
# --------------------------------------------------------------------------- #

class _FakeState:
    """Just the MotorState fields descent_summary reads."""

    def __init__(self, pos_deg, vel_deg_s, torq):
        self.pos = math.radians(pos_deg)
        self.vel = math.radians(vel_deg_s)
        self.torq = torq


def _sink(kd, w_deg_s, from_deg=25.0, n=20):
    """A joint sinking at terminal velocity: tau = kd*w throughout."""
    tau = kd * math.radians(w_deg_s)
    travel = w_deg_s * (n - 1) / 50.0
    return [(i / 50.0,
             _FakeState(from_deg - travel * i / (n - 1), -w_deg_s, -tau))
            for i in range(n)]


def test_the_two_kd_estimates_agree_at_terminal_velocity():
    """They come from different measurements - commanded damping x observed
    speed, and observed torque. Agreement is the evidence that the sink
    actually reached terminal velocity, so the estimate means something."""
    mr = __import__("measure_release")
    d = mr.descent_summary(_sink(kd=2.5, w_deg_s=40.0), kd_used=2.5,
                           target_deg_s=10.0)
    assert d["kd_from_speed"] == pytest.approx(d["kd_from_torque"], rel=1e-6)


def test_the_recommended_kd_scales_to_the_target_speed():
    """Sinking 4x too fast means 4x the damping: tau is fixed by gravity, so
    kd and w trade off inversely."""
    mr = __import__("measure_release")
    d = mr.descent_summary(_sink(kd=2.5, w_deg_s=40.0), kd_used=2.5,
                           target_deg_s=10.0)
    assert d["kd_from_speed"] == pytest.approx(10.0)
    assert d["peak_deg_s"] == pytest.approx(40.0)


def test_an_unloaded_joint_gets_no_estimate():
    """wrist_yaw barely sees gravity at the home pose. Extrapolating a kd from
    a joint that did not sink would recommend a near-zero damping for the one
    case where damping is the only thing holding the arm."""
    mr = __import__("measure_release")
    still = [(i / 50.0, _FakeState(0.0, 0.0, 0.0)) for i in range(20)]
    d = mr.descent_summary(still, kd_used=0.72, target_deg_s=10.0)
    assert d["kd_from_speed"] is None
    assert d["kd_from_torque"] is None
    assert d["travel_deg"] == pytest.approx(0.0)


def test_the_measurement_does_not_write_the_safety_constant():
    """It prints what it implies and stops there. A script that edited
    RELEASE_KD from one run on one pose would turn thin evidence into shipped
    behaviour without anyone reading it."""
    src = (EXAMPLES / "measure_release.py").read_text()
    assert "RELEASE_KD =" not in src
    assert "write_text" not in src


def test_the_measurement_run_outlasts_the_production_timeout():
    """RELEASE_MAX_S is one of the things this run is measuring, so measuring
    with it would truncate the descent at exactly the point the production
    path gives up and disables."""
    import rebot_gripette.follower as follower_mod
    src = (EXAMPLES / "measure_release.py").read_text()
    assert '"--max_s", type=float, default=30.0' in src
    assert follower_mod.RELEASE_MAX_S < 30.0


def test_the_connect_check_never_builds_a_trajectory():
    """Rung 1.5 exists to exercise the connect path WITHOUT a move. If it ever
    grows a run_leg it is rung 2 again and has stopped isolating anything."""
    src = (EXAMPLES / "check_connect.py").read_text()
    assert "run_leg" not in src
    assert "leg_duration" not in src
