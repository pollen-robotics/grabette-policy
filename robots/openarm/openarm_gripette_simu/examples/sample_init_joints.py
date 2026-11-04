"""Sample one training-distribution home pose and print the arm joints.

The default IK seed used by the sim server is just "a good starting point
for IK convergence" — it is NOT a pose the policy ever saw at training
time. If the deployment arm starts at an out-of-distribution gripper
pose, the visual encoder produces OOD features and the policy outputs
near-mean actions (which on this dataset look like a slow lift with
gripper drifting closed) regardless of cube position.

This script samples a home pose from the SAME distribution
`collect_grasp_dataset.py` uses, runs IK to the control frame (oak_l), prints the
7 joint values formatted for `--initial-joints`. Pass them to the server.

Usage:
    cd /home/steve/Project/Repo/GRABETTE/openarm_gripette_simu
    uv run python examples/sample_init_joints.py
    uv run python examples/sample_init_joints.py --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grabette_trajectory import (  # noqa: E402
    pose_T,
    sample_episode_waypoints,
)
from openarm_gripette_simu import IKFeasibilityChecker, Kinematics  # noqa: E402
from openarm_gripette_simu.kinematics import CONTROL_FRAME  # noqa: E402

ARM_IK_SEED = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5])


def parse_args():
    p = argparse.ArgumentParser(description="Print --initial-joints for a training-distribution home pose.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_attempts", type=int, default=200)
    return p.parse_args()


def sample_init_joints(seed: int = 0, max_attempts: int = 200):
    """Sample a feasible training-distribution plan and IK its home pose.

    Returns (arm_q, waypoints), or (None, None) if no feasible plan was found.
    """
    rng = np.random.default_rng(seed)
    kin = Kinematics()

    # The plan's poses are CONTROL_FRAME (oak_l) poses — grabette's
    # BODY_TO_CAMERA is the identity because the mocap is welded at oak_l — so
    # check and IK that frame, exactly like the collector and the sim server's
    # reset. Targeting the URDF "camera" site (40 mm / 20 deg from oak_l) put
    # the camera, not oak_l, at the home pose.
    checker = IKFeasibilityChecker(
        Kinematics(),
        frame=CONTROL_FRAME,
        seed_joints=ARM_IK_SEED,
        pos_tol_m=0.015,
        rot_tol_deg=5.0,
        n_iter=200,
    )

    # Use the same rejection sampler the dataset collector uses, so we land
    # on a plan whose entire trajectory is arm-reachable. We only need the
    # home pose, but it has to come from a fully-feasible plan.
    from grabette_trajectory import episode_target_poses  # noqa: E402

    def builder():
        wp = sample_episode_waypoints(rng)
        return episode_target_poses(wp), wp

    poses, wp, stats = checker.sample_feasible_trajectory(
        builder, max_attempts=max_attempts,
    )
    if wp is None:
        return None, None

    home_T = pose_T(wp.home_xyz, wp.home_quat)
    arm_q = kin.inverse(home_T, current_joint_positions=ARM_IK_SEED.copy(),
                         n_iter=300, frame=CONTROL_FRAME)
    return arm_q, wp


def main():
    args = parse_args()
    kin = Kinematics()
    arm_q, wp = sample_init_joints(args.seed, args.max_attempts)
    if wp is None:
        print(f"Could not find a feasible plan in {args.max_attempts} attempts.", file=sys.stderr)
        sys.exit(1)

    print("Sampled training-distribution home pose:")
    print(f"  cube_xyz   : {wp.cube_xyz}")
    print(f"  home_xyz   : {wp.home_xyz}  (body frame, in world coords)")
    print(f"  home pitch : {wp.home_dbg['pitch_deg']:.1f} deg")
    print(f"  home yaw   : {wp.home_dbg['yaw_deg']:.1f} deg")

    # Verify
    T_actual = kin.forward(arm_q, frame=CONTROL_FRAME)
    pos_err_mm = np.linalg.norm(T_actual[:3, 3] - wp.home_xyz) * 1000
    print(f"  IK residual ({CONTROL_FRAME}): {pos_err_mm:.2f} mm "
          f"(the sim server's reset accepts < 20 mm)")

    print()
    print("Arm joints (rad):")
    print("  " + ", ".join(f"{q:+.4f}" for q in arm_q))
    print()
    print("Pass to the sim server like this:")
    joint_str = " ".join(f"{q:.4f}" for q in arm_q)
    print("  uv run python -m openarm_gripette_simu \\")
    print("      --scene scenes/table_grasp.xml \\")
    print(f"      --initial-joints {joint_str}")


if __name__ == "__main__":
    main()
