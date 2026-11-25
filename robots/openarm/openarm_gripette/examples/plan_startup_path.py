"""Plan a safe move from the power-on (hanging) pose to the manipulation pose,
with a table in front of the arm. Offline: prints reset_arm.py command lines and
the matching evaluate.py --home_joints, never talks to the arm.

Manipulation pose: shoulder pitched back, elbow and wrist pitch bent so the
gripper is level and pointing at the table, its centre ~10 cm in front of and
~27 cm below the shoulder axis. Pulled back this far, the gripper has more room
to move forward over the table, while shoulder pitch (-59, stop -80) and wrist
pitch (46, stop 90) keep some travel. Hanging pose: all joints at 0.

THE PATH
--------
A straight move from hanging to the manipulation pose swings the gripper
forward from below the elbow, through the space where the table edge usually
is. Instead the arm goes BACKWARDS first, away from the table (each hop is one
ArmService.Reset, a straight joint-space line):

  1. shoulder pitch -> -BACK      the straight arm swings back, away from the table
  2. elbow -> TUCK, wrist pitch   the forearm folds up against the upper arm,
     -> TUCK_WRIST                 the gripper tilts back, away from the table
  3. shoulder pitch -> MANIP      the tucked arm swings forward, gripper high
  4. elbow, wrist pitch -> MANIP  the forearm comes DOWN onto the manipulation
                                   pose, from above the table

Folding the elbow straight to its final angle at step 2 (no tuck) is worse: step 3 then
swings the gripper forward and UP from below the table top, into its edge.
Leaving the wrist straight while tucked is worse too: at step 3 the fingertips
point forward, at the table edge. Tilting it the other way (positive wrist
pitch) presses the gripper shell against the shoulder once the elbow is tucked
(seen on the arm).
Shoulder pitch cannot go further back than -80 deg (joint limit), and the elbow
limit is 140 deg, so both are kept 5 deg short of their stops.

Shutdown is the same path reversed, ending hanging (torque still on).

The path is checked against the table, modelled as a solid block from the floor
to the table top, starting at its near edge and extending 1 m away and 1 m to
each side. It is also checked for the gripper hitting the shoulder (the profile
and every link up to the upper arm). Distances use the arm's collision meshes
(MuJoCo model). If any hop comes closer than --margin to either, nothing is
printed but the reason.

Model frame: origin at the bottom of the arm's vertical 60x60 mm profile, x
forward (towards the table), y left, z up.

Usage:
  uv run python examples/plan_startup_path.py \\
      --table_height 0.75 --shoulder_height 0.99 --table_distance 0.20
"""

import argparse

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from openarm_gripette_model import OPENARM_RIGHT_DIR
from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES

PROFILE_HALF_WIDTH = 0.03  # the vertical profile is 60x60 mm, centred on x = 0
TABLE_DEPTH, TABLE_HALF_WIDTH = 1.0, 1.0  # assumed: deep and wide (worst case)
SHOULDER_SEARCH = 0.1  # gripper-to-shoulder distances above this are reported as this (m)

HANG_DEG = [0.0] * 7
MANIP_DEG = [-59.0, 0.0, 0.0, 103.0, 0.0, 0.0, 46.0]
TUCK_WRIST_DEG = -45.0  # wrist pitch while the elbow is tucked (steps 2-3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--table_height", type=float, required=True,
                   help="Table top, from the floor (m).")
    p.add_argument("--shoulder_height", type=float, required=True,
                   help="Shoulder joint axis (pitch/roll, they are at the same height), "
                        "from the floor (m).")
    p.add_argument("--table_distance", type=float, required=True,
                   help="Horizontal distance from the FRONT face of the profile to the "
                        "table's near edge (m).")
    p.add_argument("--back_deg", type=float, default=75.0,
                   help="How far the straight arm swings back (shoulder pitch, max 80). "
                        "Needs ~0.6 m of free space behind the arm.")
    p.add_argument("--tuck_deg", type=float, default=135.0,
                   help="Elbow angle while the arm swings forward (max 140).")
    p.add_argument("--margin", type=float, default=0.03,
                   help="Minimum allowed distance from the arm to the table, and from "
                        "the gripper to the shoulder (m).")
    p.add_argument("--arm_addr", default="localhost:50052", help="Only used in the printed lines.")
    return p.parse_args()


def surface_points(m, g, step=0.004, seed=0):
    """Points on the surface of mesh geom g (its own frame), about one per step x step.

    Mesh vertices alone are sparse on large flat faces, so the gap between two
    meshes would be overestimated there.
    """
    mid = m.geom_dataid[g]
    verts = m.mesh_vert[m.mesh_vertadr[mid]:][:m.mesh_vertnum[mid]]
    tri = verts[m.mesh_face[m.mesh_faceadr[mid]:][:m.mesh_facenum[mid]]]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    idx = np.repeat(np.arange(len(tri)), np.maximum(1, np.ceil(area / step**2)).astype(int))
    # Uniform random points in each triangle (folded back when outside it).
    r = np.random.default_rng(seed).random((len(idx), 2))
    r[r.sum(axis=1) > 1] = 1 - r[r.sum(axis=1) > 1]
    pts = tri[idx, 0] + r[:, :1] * e1[idx] + r[:, 1:] * e2[idx]
    _, keep = np.unique(np.round(pts / step), axis=0, return_index=True)
    return pts[keep]


class ClearanceCheck:
    """Distance from the arm's collision meshes to the table block, and from the
    gripper to the shoulder."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(OPENARM_RIGHT_DIR / "robot.xml"))
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
        # Shoulder axis height above the model origin (bottom of the profile).
        mujoco.mj_kinematics(self.m, self.d)
        self.shoulder_z = self.d.xanchor[self.m.joint("r_arm_pitch").id][2]
        self.mimic = self.m.joint("r_wrist_roll_mimic").qposadr[0]
        # Vertices of every collision mesh that moves with the arm (the profile
        # is the root body). One per 5 mm voxel: ~560k vertices otherwise, and
        # the margins checked are centimetres.
        root = self.m.body("hfsb6_6060_750_1_step").id
        self.geoms = []
        for g in range(self.m.ngeom):
            if (self.m.geom_bodyid[g] in (0, root) or self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                    or not (self.m.geom_contype[g] or self.m.geom_conaffinity[g])):
                continue
            mid = self.m.geom_dataid[g]
            verts = self.m.mesh_vert[self.m.mesh_vertadr[mid]:][:self.m.mesh_vertnum[mid]]
            _, keep = np.unique(np.round(verts / 0.005), axis=0, return_index=True)
            self.geoms.append((g, verts[keep]))
        # Gripper: the wrist pitch link and everything it carries. Shoulder: the
        # profile and the links up to the upper arm (moved by pitch, roll, yaw).
        wrist = self.m.joint("r_wrist_pitch").bodyid[0]
        gripper = [b for b in range(self.m.nbody) if self._carried_by(b, wrist)]
        shoulder = [root] + [self.m.joint(n).bodyid[0] for n in ARM_JOINT_NAMES[:3]]
        self.gripper = [(g, surface_points(self.m, g)) for g, _ in self.geoms
                        if self.m.geom_bodyid[g] in gripper]
        # One KD-tree per shoulder link, in the link's own frame (built once:
        # the links are rigid, so the gripper points are moved into them instead).
        self.shoulder = []
        for b in shoulder:
            geoms = [(g, surface_points(self.m, g)) for g in range(self.m.ngeom)
                     if self.m.geom_bodyid[g] == b
                     and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                     and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
            local = (self._world(geoms) - self.d.xpos[b]) @ self.d.xmat[b].reshape(3, 3)
            self.shoulder.append((b, cKDTree(local)))

    def _carried_by(self, b, ancestor):
        while b not in (0, ancestor):
            b = self.m.body_parentid[b]
        return b == ancestor

    def _world(self, geoms):
        """World positions of the (geom, local points) pairs at the current pose."""
        return np.concatenate([self.d.geom_xpos[g] + v @ self.d.geom_xmat[g].reshape(3, 3).T
                               for g, v in geoms])

    def _set(self, q_rad):
        self.d.qpos[self.qadr] = q_rad
        # r_wrist_roll_mimic = -r_wrist_roll (a MuJoCo equality, not applied by mj_kinematics).
        self.d.qpos[self.mimic] = -q_rad[ARM_JOINT_NAMES.index("r_wrist_roll")]
        mujoco.mj_kinematics(self.m, self.d)

    def shoulder_distance(self, q_rad):
        """Distance (m) from the gripper to the shoulder at joints q (0 = touching).

        Capped at SHOULDER_SEARCH: searching only that close is ~10x faster.
        """
        self._set(q_rad)
        pts = self._world(self.gripper)
        return min(SHOULDER_SEARCH, *(
            tree.query((pts - self.d.xpos[b]) @ self.d.xmat[b].reshape(3, 3),
                       distance_upper_bound=SHOULDER_SEARCH, workers=-1)[0].min()
            for b, tree in self.shoulder))

    def distance(self, q_rad):
        """Signed distance (m) from the arm to the table at joints q (negative = inside)."""
        self._set(q_rad)
        pts = self._world(self.geoms)
        # Point-to-box signed distance.
        c, h = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        e = np.abs(pts - c) - h
        return (np.linalg.norm(np.maximum(e, 0.0), axis=1) + np.minimum(e.max(axis=1), 0.0)).min()

    def hop(self, a_deg, b_deg, step_deg=1.0):
        """Smallest (table, shoulder) distances along the straight joint-space move a -> b."""
        a, b = np.radians(a_deg), np.radians(b_deg)
        n = int(np.ceil(np.degrees(np.abs(b - a).max()) / step_deg)) + 1
        qs = [a + t * (b - a) for t in np.linspace(0, 1, n)]
        return min(map(self.distance, qs)), min(map(self.shoulder_distance, qs))


def main():
    args = parse_args()
    # Table block in the model frame (origin = bottom of the profile).
    check = ClearanceCheck()
    floor = check.shoulder_z - args.shoulder_height
    top = floor + args.table_height
    edge = PROFILE_HALF_WIDTH + args.table_distance
    check.lo = np.array([edge, -TABLE_HALF_WIDTH, floor])
    check.hi = np.array([edge + TABLE_DEPTH, TABLE_HALF_WIDTH, top])
    print(f"Table top is {1000 * (args.shoulder_height - args.table_height):.0f} mm below the "
          f"shoulder axis, near edge {1000 * args.table_distance:.0f} mm in front of the profile")
    print(f"  hanging pose: {1000 * check.distance(np.radians(HANG_DEG)):5.0f} mm from the table, "
          f"manipulation pose: {1000 * check.distance(np.radians(MANIP_DEG)):5.0f} mm")

    back = [-args.back_deg, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    back_tucked = [-args.back_deg, 0.0, 0.0, args.tuck_deg, 0.0, 0.0, TUCK_WRIST_DEG]
    tucked = [MANIP_DEG[0], 0.0, 0.0, args.tuck_deg, 0.0, 0.0, TUCK_WRIST_DEG]
    hops = [("swing arm back", HANG_DEG, back),
            ("tuck elbow", back, back_tucked),
            ("swing tucked arm forward", back_tucked, tucked),
            ("lower forearm onto pose", tucked, MANIP_DEG)]

    safe = True
    for name, a, b in hops:
        table, shoulder = check.hop(a, b)
        ok = min(table, shoulder) >= args.margin
        safe &= ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:30s} closest to table: {1000 * table:5.0f} mm, "
              f"gripper to shoulder: {1000 * shoulder:5.0f} mm")
    if not safe:
        final = check.distance(np.radians(MANIP_DEG))
        print(f"\nUNSAFE: the arm comes within {1000 * args.margin:.0f} mm of the table "
              "or of itself.")
        if final < args.margin:
            print(f"The manipulation pose itself is {1000 * final:.0f} mm from the table: "
                  "move the table away or lower it (or raise the arm).")
        raise SystemExit(1)

    def line(waypoints):
        wps = " ".join("--waypoint_deg " + " ".join(f"{v:g}" for v in w) for w in waypoints)
        return f"uv run python examples/reset_arm.py --arm_addr {args.arm_addr} {wps}"

    print("\nStartup (hanging -> manipulation pose):")
    print("  " + line([back, back_tucked, tucked, MANIP_DEG]))
    print("Shutdown (manipulation pose -> hanging):")
    print("  " + line([tucked, back_tucked, back, HANG_DEG]))
    print("Episode start for evaluate.py (the manipulation pose, rad):")
    print("  --home_joints " + " ".join(f"{v:.4f}" for v in np.radians(MANIP_DEG)))


if __name__ == "__main__":
    main()
