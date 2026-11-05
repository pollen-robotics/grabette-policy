"""Did the arm actually pick the thing up?

`gripette_core.arm_service.ArmServicer` is the service the REAL arm runs, and
a real arm cannot see an object: its `GetSuccessStatus` returns False and its
`Reset` hands back dummy coordinates, both correctly. The simulator serves that
same class - that sharing is the point of the package - and so inherited the
blind answer, which made every simulated episode score 0/1 no matter what
happened. `--num_episodes 20` produced a number that meant nothing.

This supplies the eyes, the same way `clearance.GeomClearance` supplies the
bench geometry: core takes a plain object and never imports mujoco.

WHY RESET REPOSITIONS, AND NOT ONLY RECORDS
    `evaluate.py` calls Reset between episodes. If episode 1 knocks the can off
    the bench and Reset only noted where it was, episode 2 starts with no can
    in the scene and scores 0 - silently, and so does every episode after it. A
    20-episode run would report 1/20 and look like a policy problem. So reset()
    puts the object back and zeroes its velocity.

WHAT COUNTS AS SUCCESS
    The object risen `lift_threshold` above where reset() left it. 0.05 m by
    default, which is `grabette_trajectory.LIFT_SUCCESS_THRESHOLD` - the figure
    the dataset was collected under and the OpenArm's own servicer uses, so the
    two arms' numbers mean the same thing. A horizontal nudge is not a pick,
    which is why this is a vertical test.

    The reading is INSTANTANEOUS. `evaluate.py` polls it during the episode and
    latches (`success = success or goal_reached`), so an object lifted and then
    dropped still counts. That is the OpenArm's behaviour too; matching it
    keeps the arms comparable, but it means this measures "was it ever lifted",
    not "was it still held at the end".
"""

import logging
import threading

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# Metres the object must rise to count as picked. Keep equal to
# grabette_trajectory.LIFT_SUCCESS_THRESHOLD.
LIFT_SUCCESS_THRESHOLD = 0.05


class LiftTracker:
    """Tracks one free-floating body, and says whether it has been lifted.

    `body_name` must name a body with a free joint - without one the object
    cannot move, so nothing could ever be measured. An absent body disables the
    tracker (`enabled = False`) rather than raising: a scene legitimately may
    have nothing to pick up, and a server that refuses to start because its
    bench is empty would be a nuisance.
    """

    # `lock` is unannotated on purpose: threading.Lock is a factory, not a
    # type, so `threading.Lock | None` raises at import time.
    def __init__(self, sim, body_name: str, lock=None,
                 lift_threshold: float = LIFT_SUCCESS_THRESHOLD):
        self._sim = sim
        self._body_name = body_name
        # The physics loop steps on the main thread while gRPC answers on
        # others; this touches live qpos, so it shares the server's lock. Same
        # arrangement as GripperServicer.
        self._lock = lock or threading.Lock()
        self.lift_threshold = float(lift_threshold)

        model = sim.model
        self._bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.enabled = self._bid >= 0
        if not self.enabled:
            logger.warning("lift tracker: no body %r in this scene - success "
                           "will always report False", body_name)
            self._qadr = None
            self._start = np.zeros(3)
            return

        jadr = model.body_jntadr[self._bid]
        if jadr < 0 or model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
            self.enabled = False
            self._qadr = None
            self._start = np.zeros(3)
            logger.warning("lift tracker: body %r has no free joint - it cannot "
                           "move, so a lift could never be measured", body_name)
            return

        self._qadr = int(model.jnt_qposadr[jadr])
        self._dofadr = int(model.jnt_dofadr[jadr])
        # Where the scene author put it. reset() returns the object here, so
        # "the start pose" is a property of the scene, not of whatever the last
        # episode left behind.
        self._home_pos = np.array(model.body_pos[self._bid], dtype=float)
        self._home_quat = np.array(model.body_quat[self._bid], dtype=float)
        self._start = self._home_pos.copy()

    def _position(self) -> np.ndarray:
        return np.array(self._sim.data.xpos[self._bid], dtype=float)

    def reset(self) -> tuple[float, float, float]:
        """Put the object back, stop it moving, and record its start height.

        Returns its (x, y, z), which `Reset` reports to the client.
        """
        if not self.enabled:
            return (0.0, 0.0, 0.0)
        with self._lock:
            d = self._sim.data
            d.qpos[self._qadr:self._qadr + 3] = self._home_pos
            d.qpos[self._qadr + 3:self._qadr + 7] = self._home_quat
            # Velocity too: a can still travelling from the last episode would
            # carry that motion into this one.
            d.qvel[self._dofadr:self._dofadr + 6] = 0.0
            mujoco.mj_forward(self._sim.model, d)
            self._start = self._position()
        logger.info("lift tracker: %s reset to [%+.3f %+.3f %+.3f]",
                    self._body_name, *self._start)
        return tuple(float(v) for v in self._start)

    def status(self) -> tuple[bool, float]:
        """`(goal_reached, lift_m)` - how far the object is above its start."""
        if not self.enabled:
            return (False, 0.0)
        with self._lock:
            lift = float(self._position()[2] - self._start[2])
        return (lift > self.lift_threshold, lift)
