"""Clearance between the arm's links and named obstacle geoms.

Generalised from the OpenArm's `pedestal.py`, which guarded one box proxy
around its mounting column. The reBot's hazard is a different shape - it is
bolted to a bench, so the obstacle is the table it stands on and can reach
down into - and the query is the same either way: for a candidate joint
configuration, how close does any guarded link come to any obstacle geom.

WHY IT TAKES A CONFIGURATION AND NOT THE LIVE STATE
    This is asked BEFORE commanding, about a pose the arm is not in yet, so it
    runs FK on a scratch MjData. Asking after the fact would be a collision
    report, not a gate.

WHY THE RESULT IS A CALLABLE
    `gripette_core.arm_service.ArmServicer` takes a `joint_gate`: a plain
    function from joint angles to a reason-or-None. Core must not import
    mujoco - the real-hardware drivers import core - so the geometry lives
    here and crosses as a closure. `as_joint_gate()` builds it.

LIMITS, WORTH STATING
    Only the geoms named as obstacles are checked, and only the bodies named
    as guarded. It is not a general self-collision check, and it knows nothing
    about anything the gripper is holding.
"""

import logging

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# Default clearance to keep, in metres. The OpenArm uses 2 cm against its
# column; the same figure is a reasonable default for a table.
DEFAULT_MARGIN = 0.02


class GeomClearance:
    """Min distance from guarded arm links to named obstacle geoms.

    `obstacle_geoms` and `guarded_bodies` are names. Missing names are skipped
    with a warning rather than raising: a scene legitimately may not contain a
    table, and a gate that refuses to construct would be a gate nobody fits.
    An instance with nothing to check reports `enabled = False` and returns
    +inf, which never blocks.
    """

    def __init__(self, model, obstacle_geoms, guarded_bodies, arm_joint_names,
                 margin: float = DEFAULT_MARGIN):
        self.model = model
        self.margin = float(margin)

        self.obstacle_gids = []
        for name in obstacle_geoms:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                logger.warning("clearance: no geom %r in this scene - not guarded", name)
            else:
                self.obstacle_gids.append(gid)

        guarded = set()
        for name in guarded_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                logger.warning("clearance: no body %r in this model", name)
            else:
                guarded.add(bid)

        # Collision geoms only (contype/conaffinity set). Visual-only geoms
        # are not the arm's physical extent.
        self.arm_gids = [g for g in range(model.ngeom)
                         if model.geom_bodyid[g] in guarded
                         and (model.geom_contype[g] or model.geom_conaffinity[g])]

        self.enabled = bool(self.obstacle_gids and self.arm_gids)
        if not self.enabled:
            logger.warning("clearance: disabled (%d obstacle geoms, %d arm geoms)",
                           len(self.obstacle_gids), len(self.arm_gids))

        self._data = mujoco.MjData(model)
        self._qadr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in arm_joint_names]

    def clearance(self, arm_joints) -> float:
        """Signed min distance (m) at `arm_joints`. Negative is penetration.

        Capped above at `margin`: mj_geomDistance is given that as its cutoff,
        so anything further away is simply "far enough" and costs nothing to
        evaluate. +inf when disabled.
        """
        if not self.enabled:
            return np.inf
        m, d = self.model, self._data
        for adr, val in zip(self._qadr, arm_joints):
            d.qpos[adr] = val
        mujoco.mj_kinematics(m, d)
        dmin = self.margin
        for obstacle in self.obstacle_gids:
            for g in self.arm_gids:
                dmin = min(dmin, mujoco.mj_geomDistance(m, d, obstacle, g,
                                                        self.margin, None))
        return dmin

    def collides(self, arm_joints, margin: float | None = None) -> bool:
        """True iff `arm_joints` brings a guarded link within `margin`."""
        return self.clearance(arm_joints) < (self.margin if margin is None else margin)

    def as_joint_gate(self, current_joints=None):
        """A `joint_gate` for `ArmServicer`: joints -> reason string or None.

        Moves that INCREASE clearance are always allowed. Without that, an arm
        that somehow ends up inside the margin is deadlocked there - every
        command it could use to escape is also a command inside the margin, so
        the gate would refuse all of them and the only way out would be to kill
        the server, which is the state this exists to avoid.
        """
        if not self.enabled:
            return None

        def gate(target_joints):
            d = self.clearance(target_joints)
            if d >= self.margin:
                return None
            if current_joints is not None:
                now = self.clearance(current_joints())
                if d > now:
                    return None        # retreating: always allowed
            return (f"clearance {d * 1000:.1f} mm < {self.margin * 1000:.0f} mm "
                    f"to a guarded obstacle")

        return gate
