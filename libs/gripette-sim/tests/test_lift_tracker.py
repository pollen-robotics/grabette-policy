"""The lift tracker is what makes a simulated episode scoreable.

It replaced a `GetSuccessStatus` that returned False unconditionally, so the
thing most worth pinning down is that it reports a real number and that Reset
PUTS THE OBJECT BACK - the failure that would otherwise make every episode
after a dropped can score 0 without saying why.
"""
import mujoco
import numpy as np
import pytest

from gripette_sim.lift_tracker import LIFT_SUCCESS_THRESHOLD, LiftTracker

SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="can" pos="0.3 0.1 0.06">
      <freejoint name="can_joint"/>
      <geom type="cylinder" size="0.025 0.06" mass="0.1"/>
    </body>
    <body name="bolted" pos="0 0 0.5">
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


class FakeSim:
    """Just the two attributes the tracker touches."""

    def __init__(self, xml=SCENE):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)


@pytest.fixture
def sim():
    return FakeSim()


def test_reports_the_objects_start_position(sim):
    t = LiftTracker(sim, "can")
    assert t.enabled
    assert t.reset() == pytest.approx((0.3, 0.1, 0.06))


def test_untouched_object_has_not_been_lifted(sim):
    t = LiftTracker(sim, "can")
    t.reset()
    reached, lift = t.status()
    assert not reached
    assert lift == pytest.approx(0.0, abs=1e-9)


def test_lift_is_measured_from_the_start_height(sim):
    t = LiftTracker(sim, "can")
    t.reset()
    qadr = sim.model.jnt_qposadr[sim.model.body_jntadr[
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "can")]]
    sim.data.qpos[qadr + 2] += 0.08
    mujoco.mj_forward(sim.model, sim.data)
    reached, lift = t.status()
    assert lift == pytest.approx(0.08, abs=1e-6)
    assert reached


def test_a_lift_under_the_threshold_is_not_success(sim):
    t = LiftTracker(sim, "can")
    t.reset()
    qadr = sim.model.jnt_qposadr[sim.model.body_jntadr[
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "can")]]
    sim.data.qpos[qadr + 2] += LIFT_SUCCESS_THRESHOLD * 0.5
    mujoco.mj_forward(sim.model, sim.data)
    reached, lift = t.status()
    assert not reached
    assert lift > 0


def test_a_horizontal_shove_is_not_a_pick(sim):
    """A nudge across the bench moves the object a long way and lifts it none."""
    t = LiftTracker(sim, "can")
    t.reset()
    qadr = sim.model.jnt_qposadr[sim.model.body_jntadr[
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "can")]]
    sim.data.qpos[qadr] += 0.5
    mujoco.mj_forward(sim.model, sim.data)
    reached, lift = t.status()
    assert not reached
    assert lift == pytest.approx(0.0, abs=1e-9)


def test_reset_puts_a_knocked_away_object_back(sim):
    """THE one that matters across episodes: without this, episode 2 onwards
    would run with no object in reach and score 0 for a reason nobody sees."""
    t = LiftTracker(sim, "can")
    t.reset()
    qadr = sim.model.jnt_qposadr[sim.model.body_jntadr[
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "can")]]
    sim.data.qpos[qadr:qadr + 3] = [1.5, -1.5, 0.9]   # knocked off the bench
    mujoco.mj_forward(sim.model, sim.data)
    assert t.status()[1] > 0.5

    assert t.reset() == pytest.approx((0.3, 0.1, 0.06))
    reached, lift = t.status()
    assert not reached
    assert lift == pytest.approx(0.0, abs=1e-9)


def test_reset_stops_the_object_moving(sim):
    """A can still travelling from the last episode would carry that into this
    one and could 'lift' itself without ever being grasped."""
    t = LiftTracker(sim, "can")
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "can")
    dofadr = sim.model.jnt_dofadr[sim.model.body_jntadr[bid]]
    sim.data.qvel[dofadr:dofadr + 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t.reset()
    assert np.allclose(sim.data.qvel[dofadr:dofadr + 6], 0.0)


def test_reset_restores_orientation(sim):
    t = LiftTracker(sim, "can")
    qadr = sim.model.jnt_qposadr[sim.model.body_jntadr[
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "can")]]
    sim.data.qpos[qadr + 3:qadr + 7] = [0.707, 0.707, 0.0, 0.0]   # tipped over
    t.reset()
    assert sim.data.qpos[qadr + 3:qadr + 7] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_a_missing_body_disables_rather_than_raises(sim):
    """A scene with nothing to pick up is legitimate; refusing to start would
    only mean a server that cannot run its own bare scene."""
    t = LiftTracker(sim, "no_such_body")
    assert not t.enabled
    assert t.reset() == (0.0, 0.0, 0.0)
    assert t.status() == (False, 0.0)


def test_a_bolted_body_disables_too(sim):
    """No free joint means it cannot move, so a lift could never be measured —
    reporting 0.0 forever would look like a policy that never succeeds."""
    t = LiftTracker(sim, "bolted")
    assert not t.enabled
    assert t.status() == (False, 0.0)


def test_threshold_is_configurable_and_defaults_to_the_dataset_figure(sim):
    assert LIFT_SUCCESS_THRESHOLD == 0.05
    t = LiftTracker(sim, "can", lift_threshold=0.01)
    t.reset()
    qadr = sim.model.jnt_qposadr[sim.model.body_jntadr[
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "can")]]
    sim.data.qpos[qadr + 2] += 0.02
    mujoco.mj_forward(sim.model, sim.data)
    assert t.status()[0]
