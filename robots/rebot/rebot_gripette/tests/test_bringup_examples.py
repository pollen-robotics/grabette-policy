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
                                  "verify_descent"])
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


def _ramp(from_deg, to_deg, w_deg_s, torq=1.0, n=20):
    """A joint being lowered along a ramp at a roughly constant rate."""
    return [(i / 50.0,
             _FakeState(from_deg + (to_deg - from_deg) * i / (n - 1),
                        -w_deg_s, -torq))
            for i in range(n)]


def test_a_lowered_joint_is_scored_against_the_floor():
    """residual is distance from where it was TOLD to end, not from where it
    started - that is what says whether the arm actually got down."""
    vd = __import__("verify_descent")
    d = vd.descent_errors(_ramp(25.0, 0.4, w_deg_s=10.0), started_at=25.0,
                          floor=0.0, rate_deg_s=10.0)
    assert d["travel_deg"] == pytest.approx(24.6)
    assert d["residual_deg"] == pytest.approx(0.4)
    assert d["overspeed"] == pytest.approx(1.0)


def test_a_held_joint_is_scored_against_where_it_started():
    """A held joint's floor IS its start, so the same arithmetic scores both
    roles and a held joint's residual is its drift."""
    vd = __import__("verify_descent")
    d = vd.descent_errors(_ramp(-70.0, -70.0, w_deg_s=0.0), started_at=-70.0,
                          floor=-70.0, rate_deg_s=10.0)
    assert d["travel_deg"] == pytest.approx(0.0)
    assert d["residual_deg"] == pytest.approx(0.0)


def test_a_joint_that_falls_reads_as_overspeed():
    """The number that distinguishes a descent from a drop. Position control
    tracks the commanded rate; a joint running far past it is not being
    lowered, it is falling."""
    vd = __import__("verify_descent")
    d = vd.descent_errors(_ramp(25.0, 0.0, w_deg_s=120.0), started_at=25.0,
                          floor=0.0, rate_deg_s=10.0)
    assert d["overspeed"] == pytest.approx(12.0)


def test_a_traced_path_is_not_a_sum_of_minimum_length_legs():
    """Why run_path exists. leg_duration floors every leg at MIN_LEG_S, so the
    square's 41 segments of 0.5-1.3 deg each took 2 s regardless - 82 s of
    stop-start for 36 deg of travel. Priced as one continuous motion the same
    path is an order of magnitude shorter."""
    seg_deltas = [1.3] * 20 + [0.5] * 20
    per_leg = sum(_bringup.leg_duration(d, deg_per_s=2.0) for d in seg_deltas)
    as_one_path = sum(seg_deltas) / 2.0
    assert per_leg == pytest.approx(80.0)     # 40 segments pinned at MIN_LEG_S
    assert as_one_path == pytest.approx(18.0)
    assert per_leg > 4 * as_one_path


def test_the_path_distance_metric_matches_leg_duration():
    """run_path measures distance in the same currency leg_duration prices a
    single leg in - the largest joint delta - so --deg_per_s means one thing
    across the package rather than two."""
    segs = _bringup.path_segments([{"a": 0.0, "b": 0.0},
                                   {"a": 3.0, "b": 1.0},
                                   {"a": 3.0, "b": 5.0}])
    assert [s[2] for s in segs] == [3.0, 4.0]
    # One leg of 3 deg and one of 4 deg, priced the same way.
    assert _bringup.leg_duration(3.0, deg_per_s=1.0) == pytest.approx(3.0)


def test_a_path_interpolates_across_waypoint_boundaries():
    """The point of tracing as one motion: distance past a waypoint continues
    into the next segment instead of stopping there."""
    wps = [{"a": 0.0}, {"a": 10.0}, {"a": 20.0}]
    segs = _bringup.path_segments(wps)
    assert _bringup.pose_along(segs, 0.0, wps[-1])["a"] == pytest.approx(0.0)
    assert _bringup.pose_along(segs, 5.0, wps[-1])["a"] == pytest.approx(5.0)
    # 15 is past the first waypoint - it must land mid-second-segment.
    assert _bringup.pose_along(segs, 15.0, wps[-1])["a"] == pytest.approx(15.0)
    assert _bringup.pose_along(segs, 20.0, wps[-1])["a"] == pytest.approx(20.0)


def test_a_path_never_runs_past_its_last_waypoint():
    """Distance beyond the end clamps to the final pose rather than
    extrapolating off it."""
    wps = [{"a": 0.0}, {"a": 10.0}]
    segs = _bringup.path_segments(wps)
    assert _bringup.pose_along(segs, 999.0, wps[-1])["a"] == pytest.approx(10.0)


def test_a_repeated_waypoint_does_not_divide_by_zero():
    """Two identical consecutive waypoints are a zero-length segment. A square
    whose last waypoint returns to the first produces them."""
    wps = [{"a": 0.0}, {"a": 0.0}, {"a": 4.0}]
    segs = _bringup.path_segments(wps)
    assert [s[2] for s in segs] == [0.0, 4.0]
    assert _bringup.pose_along(segs, 0.0, wps[-1])["a"] == pytest.approx(0.0)
    assert _bringup.pose_along(segs, 2.0, wps[-1])["a"] == pytest.approx(2.0)


def test_the_cruise_rate_is_the_rate_that_was_asked_for():
    """The ease is over a fixed time at each end, not a cosine across the whole
    path. A cosine would speed the middle up to make up for the slow ends, so
    --deg_per_s would not be the speed the arm moves at."""
    total, rate, ease_s = 100.0, 5.0, 1.0
    d0 = _bringup.distance_along(10.0, total, rate, ease_s)
    d1 = _bringup.distance_along(11.0, total, rate, ease_s)
    assert d1 - d0 == pytest.approx(rate)   # one second of cruise = one rate


def test_a_path_starts_and_ends_at_rest():
    """Zero velocity at both ends: the arm eases in rather than being stepped,
    and settles rather than stopping dead."""
    total, rate = 100.0, 5.0
    assert _bringup.distance_along(0.0, total, rate) == pytest.approx(0.0)
    # The first tenth of the ease covers far less than a tenth of cruise.
    assert _bringup.distance_along(0.1, total, rate) < 0.1 * rate * 0.5
    end = _bringup.path_duration(total, rate)
    assert _bringup.distance_along(end, total, rate) == pytest.approx(total)


def test_the_eased_ends_cost_exactly_one_ease_window():
    """Stated in path_duration's docstring, so it should be true: the two
    half-eases together add ease_s to the constant-rate time."""
    assert _bringup.path_duration(100.0, 5.0, ease_s=1.0) == pytest.approx(21.0)
    assert _bringup.path_duration(100.0, 5.0, ease_s=2.0) == pytest.approx(22.0)


def test_the_square_closure_compares_two_measurements():
    """Closure against the COMMANDED start pose reports steady-state droop -
    2.5 deg on this arm - and calls it closure error. Both ends must be
    measured for droop to cancel."""
    src = (EXAMPLES / "cartesian_square.py").read_text()
    assert "end[name] - before[name]" in src
    assert "end[name] - start_motor[name]" not in src


def test_the_square_solved_from_the_stops_hugs_a_limit():
    """Why cartesian_square.py grew --home_joints. The solver reports IK
    residual and joint step; neither says anything about how close the path
    sits to a hard stop. From the folded pose the 2 cm square is 'clean' and
    spends the whole trace a fraction of a degree off shoulder_lift's stop."""
    pytest.importorskip("placo")
    pytest.importorskip("rebot_gripette_model")
    cs = __import__("cartesian_square")
    from gripette_core.kinematics import CONTROL_FRAME, Kinematics
    from rebot_gripette.spec import ARM_JOINT_NAMES, REBOT_SPEC, urdf_limits

    kin = Kinematics(REBOT_SPEC)
    lift = ARM_JOINT_NAMES.index("shoulder_lift")
    lo = urdf_limits()["shoulder_lift"][0]

    def lift_margin(start_deg):
        poses, _, problems = cs.solve_square(
            kin, CONTROL_FRAME, np.deg2rad(start_deg), cs.AXES["x"], 0.02, 10, 15.0)
        assert not problems, "both start poses are supposed to solve cleanly"
        return min(np.rad2deg(p[lift]) for p in poses) - lo

    folded = lift_margin([-0.44, 0.07, 4.09, -10.17, 0.53, -0.42])
    home = lift_margin([0.0, 15.0, 25.0, -10.0, 0.0, 0.0])
    assert folded < 1.0, f"expected the folded start to hug the stop, got {folded:.2f}"
    assert home > 10.0, f"expected the home start to keep margin, got {home:.2f}"


def test_the_square_returns_the_arm_to_where_it_was_found():
    """--home_joints moves the arm somewhere it was not. The run has to put it
    back, including after an aborted trace, or the pose it stopped at silently
    becomes the next run's start pose - and the teardown's descent only lowers
    two joints."""
    src = (EXAMPLES / "cartesian_square.py").read_text()
    back = src.index('"back"')
    assert src.index("[back] returning") < back
    # The return must not be inside the `if ok:` that guards the closure
    # report, or an abort skips it.
    assert "Back to where the arm was found, whatever happened" in src


def test_the_descent_check_does_not_reach_for_damping():
    """Rung 2.6 used to measure a kd. Damping was measured useless on this arm
    on 2026-10-28; a script still tuning it would imply it was an option. The
    docstring still TELLS that story, which is the point - what must be gone is
    the machinery: raw MIT frames and the kd table."""
    src = (EXAMPLES / "verify_descent.py").read_text()
    assert "send_mit" not in src
    assert "RELEASE_KD" not in src.replace("the shipped RELEASE_KD", "")


def test_the_connect_check_never_builds_a_trajectory():
    """Rung 1.5 exists to exercise the connect path WITHOUT a move. If it ever
    grows a run_leg it is rung 2 again and has stopped isolating anything."""
    src = (EXAMPLES / "check_connect.py").read_text()
    assert "run_leg" not in src
    assert "leg_duration" not in src
