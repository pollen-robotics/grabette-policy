"""Plan a safe move from the power-on (hanging) pose to the manipulation pose,
with a table in front of the arm. Offline: prints reset_arm.py command lines,
never talks to the arm.

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
     -> MANIP                      the gripper tilts back
  3. shoulder pitch -> MANIP      the tucked arm swings forward, gripper high
  4. elbow -> MANIP               the forearm comes DOWN onto the manipulation
                                   pose, from above the table

Folding the elbow straight to its final angle at step 2 (no tuck) is worse: step 3 then
swings the gripper forward and UP from below the table top, into its edge.
Leaving the wrist straight while tucked is worse too: at step 3 the fingertips
point forward, at the table edge.
Shoulder pitch cannot go further back than -80 deg (joint limit), and the elbow
limit is 140 deg, so both are kept 5 deg short of their stops.

Shutdown is the same path reversed, ending hanging (torque still on).

The path is checked against the table, modelled as a solid block from the floor
to the table top, starting at its near edge and extending 1 m away and 1 m to
each side. Distances use the arm's collision meshes (MuJoCo model). If any hop
comes closer than --margin, nothing is printed but the reason.

Model frame: origin at the bottom of the arm's vertical 60x60 mm profile, x
forward (towards the table), y left, z up.

Usage:
  uv run python examples/plan_startup_path.py \\
      --table_height 0.75 --shoulder_height 0.99 --table_distance 0.20
"""

import argparse

import mujoco
import numpy as np
from openarm_gripette_model import OPENARM_RIGHT_DIR
from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES

PROFILE_HALF_WIDTH = 0.03  # the vertical profile is 60x60 mm, centred on x = 0
TABLE_DEPTH, TABLE_HALF_WIDTH = 1.0, 1.0  # assumed: deep and wide (worst case)

HANG_DEG = [0.0] * 7
MANIP_DEG = [-59.0, 0.0, 0.0, 103.0, 0.0, 0.0, 46.0]


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
                   help="Minimum allowed distance from the arm to the table (m).")
    p.add_argument("--arm_addr", default="localhost:50052", help="Only used in the printed lines.")
    return p.parse_args()


class TableCheck:
    """Distance from the arm's collision meshes to the table block."""

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

    def distance(self, q_rad):
        """Signed distance (m) from the arm to the table at joints q (negative = inside)."""
        self.d.qpos[self.qadr] = q_rad
        # r_wrist_roll_mimic = -r_wrist_roll (a MuJoCo equality, not applied by mj_kinematics).
        self.d.qpos[self.mimic] = -q_rad[ARM_JOINT_NAMES.index("r_wrist_roll")]
        mujoco.mj_kinematics(self.m, self.d)
        pts = np.concatenate([self.d.geom_xpos[g] + v @ self.d.geom_xmat[g].reshape(3, 3).T
                              for g, v in self.geoms])
        # Point-to-box signed distance.
        c, h = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        e = np.abs(pts - c) - h
        return (np.linalg.norm(np.maximum(e, 0.0), axis=1) + np.minimum(e.max(axis=1), 0.0)).min()

    def hop(self, a_deg, b_deg, step_deg=1.0):
        """Smallest distance along the straight joint-space move a -> b."""
        a, b = np.radians(a_deg), np.radians(b_deg)
        n = int(np.ceil(np.degrees(np.abs(b - a).max()) / step_deg)) + 1
        return min(self.distance(a + t * (b - a)) for t in np.linspace(0, 1, n))


def main():
    args = parse_args()
    # Table block in the model frame (origin = bottom of the profile).
    check = TableCheck()
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
    back_tucked = [-args.back_deg, 0.0, 0.0, args.tuck_deg, 0.0, 0.0, MANIP_DEG[6]]
    tucked = [MANIP_DEG[0], 0.0, 0.0, args.tuck_deg, 0.0, 0.0, MANIP_DEG[6]]
    hops = [("swing arm back", HANG_DEG, back),
            ("tuck elbow", back, back_tucked),
            ("swing tucked arm forward", back_tucked, tucked),
            ("lower forearm onto pose", tucked, MANIP_DEG)]

    safe = True
    for name, a, b in hops:
        dist = check.hop(a, b)
        ok = dist >= args.margin
        safe &= ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:30s} closest to table: {1000 * dist:5.0f} mm")
    if not safe:
        final = check.distance(np.radians(MANIP_DEG))
        print(f"\nUNSAFE: the arm comes within {1000 * args.margin:.0f} mm of the table.")
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


if __name__ == "__main__":
    main()
