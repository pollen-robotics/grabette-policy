"""The bench clearance gate on the REAL server.

The IK-jump watchdog catches the solver flipping branch. Driving into the
bench the arm is bolted to is not a branch flip - it is a smooth run of
well-conditioned IK solutions - so the watchdog cannot see it. This is the
guard for that, and the point of these tests is that it guards the HARDWARE
path, not only the simulator.
"""
import numpy as np
import pytest

from rebot_gripette.obstacles import GUARDED_BODIES, OBSTACLE_GEOMS, TABLE_MARGIN


def test_the_declaration_lives_with_the_robot_not_the_simulator():
    """It is a statement about the arm's own links, and the real server needs
    it as much as the sim does."""
    import rebot_gripette.obstacles as mod

    assert mod.__name__.startswith("rebot_gripette.")
    assert GUARDED_BODIES and OBSTACLE_GEOMS


def test_both_work_surfaces_are_named():
    """One declaration serves two models: the simulator's furnished scene uses
    'tabletop', the bare model the real server loads uses 'bench'. A gate
    given the other scene must still find its obstacle."""
    assert "tabletop" in OBSTACLE_GEOMS
    assert "bench" in OBSTACLE_GEOMS


def test_the_base_links_are_deliberately_unguarded():
    """The arm is BOLTED to the work surface, so axis_1..3 touch it by
    construction and at the folded pose it powers on in. Guarding them would
    fire the gate forever."""
    for link in ("axis_1", "axis_2", "axis_3"):
        assert link not in GUARDED_BODIES


# --------------------------------------------------------------------------- #
# Geometry. Needs the optional clearance extra.
# --------------------------------------------------------------------------- #

mujoco = pytest.importorskip("mujoco", reason="needs rebot-gripette[clearance]")
pytest.importorskip("gripette_sim", reason="needs rebot-gripette[clearance]")


@pytest.fixture(scope="module")
def bench_clearance():
    from gripette_sim.clearance import GeomClearance
    from gripette_sim.simulation import _load_model
    from rebot_gripette.spec import REBOT_SPEC
    from rebot_gripette_model import REBOT_B601RS_BENCH

    model = _load_model(REBOT_B601RS_BENCH)
    return GeomClearance(model, OBSTACLE_GEOMS, GUARDED_BODIES,
                         list(REBOT_SPEC.joint_names), margin=TABLE_MARGIN)


def test_the_bench_model_arms_the_gate(bench_clearance):
    """A gate that found no geometry would pass every behavioural test below
    while guarding nothing."""
    assert bench_clearance.enabled
    assert bench_clearance.obstacle_gids and bench_clearance.arm_gids


def test_the_bench_surface_is_the_plane_the_arm_is_bolted_to():
    """z=0 by construction - the arm bolts to it - so it needs no measurement.
    The slab must sit entirely BELOW that, or it would swallow the base."""
    from rebot_gripette_model import REBOT_B601RS_BENCH

    xml = REBOT_B601RS_BENCH.read_text()
    assert 'name="bench"' in xml
    assert 'size="1.0 1.0 0.05" pos="0 0 -0.05"' in xml


def test_driving_into_the_bench_is_refused(bench_clearance):
    gate = bench_clearance.as_joint_gate()
    into = np.deg2rad([0, 60, 20, -60, 0, 0])
    assert bench_clearance.clearance(into) < 0
    assert gate(into) is not None


@pytest.mark.parametrize("deg,label", [
    ([0, 0, 0, 0, 0, 0], "the folded pose it powers on in"),
    ([0, 15, 25, -10, 0, 0], "the home pose"),
    ([0, 19, 28, -13, 0, 12], "a rung 5 square corner"),
])
def test_poses_already_run_on_the_arm_are_still_allowed(bench_clearance, deg, label):
    """A gate that refused work already validated on hardware would be wrong
    about the hardware, not protective of it. Rungs 2-5 all ran through this
    region."""
    gate = bench_clearance.as_joint_gate()
    assert gate(np.deg2rad(deg)) is None, f"the gate now refuses {label}"


def test_retreating_from_inside_the_margin_is_allowed(bench_clearance):
    """Without this an arm inside the margin is deadlocked - every command
    that could free it is also inside the margin - and the only way out is
    killing the server, which is the state the gate exists to avoid."""
    stuck = np.deg2rad([0, 90, 10, -80, 0, 0])
    assert 0 < bench_clearance.clearance(stuck) < bench_clearance.margin

    gate = bench_clearance.as_joint_gate(current_joints=lambda: stuck)
    retreat = np.deg2rad([0, 90, 40, -80, 0, 0])      # elbow up, away from it
    assert bench_clearance.clearance(retreat) > bench_clearance.clearance(stuck)
    assert gate(retreat) is None


def test_a_breach_is_refused_however_deep_it_is(bench_clearance):
    """mj_geomDistance does not report a penetration DEPTH for a mesh inside a
    box - it returns 0 - so the number stops being meaningful once a link is
    through the surface, and the simulator's scene and the bare bench disagree
    about it there. The VERDICT does not disagree: anything under the margin
    is refused either way, which is the property the gate rests on."""
    gate = bench_clearance.as_joint_gate()
    for deg in ([0, 90, 10, -80, 0, 0], [0, 90, 5, -80, 0, 0],
                [0, 100, 10, -80, 0, 0], [0, 60, 20, -60, 0, 0]):
        assert gate(np.deg2rad(deg)) is not None, f"{deg} was allowed"


def test_the_bench_and_the_sim_scene_agree_on_approach(bench_clearance):
    """Two models, one declaration, so the simulator must be testing the same
    gate that guards the arm - in the region that matters, which is BEFORE
    contact. Past the surface mj_geomDistance returns 0 for an intersecting
    mesh/box rather than a depth, and the two scenes differ there; both still
    refuse, which is what the gate is for."""
    pytest.importorskip("rebot_gripette_simu")
    from gripette_sim.clearance import GeomClearance
    from gripette_sim.simulation import _load_model
    from rebot_gripette.spec import REBOT_SPEC
    from rebot_gripette_simu.server import DEFAULT_SCENE

    sim = GeomClearance(_load_model(DEFAULT_SCENE), OBSTACLE_GEOMS,
                        GUARDED_BODIES, list(REBOT_SPEC.joint_names),
                        margin=TABLE_MARGIN)
    for deg in ([0, 15, 25, -10, 0, 0], [0, 0, 0, 0, 0, 0],
                [0, 90, 10, -80, 0, 0], [0, 95, 10, -80, 0, 0],
                [0, 90, 10, -90, 0, 0]):
        q = np.deg2rad(deg)
        assert bench_clearance.clearance(q) == pytest.approx(
            sim.clearance(q), abs=1e-6), f"models disagree at {deg}"
