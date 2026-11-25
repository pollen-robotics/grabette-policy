"""Joint-space retargeting of handheld (Grabette) demonstrations onto the OpenArm.

A Grabette recording gives the oak_l (control-frame) pose per frame in a SLAM
world that is re-anchored at frame 0 and carries no gravity. To express it as
OpenArm joint trajectories we must choose where that anchor sits relative to
the robot. The frame-0 depth image fixes three of the six degrees of freedom:
the table plane seen by the camera gives the camera height above the table and
its tilt/roll (assuming a level table). The remaining three — the anchor's
position on the table (x, y) and its heading psi about the vertical — plus the
table height in the robot frame form a `Placement`.

For a joint-space POLICY the placement is a labelling choice: the wrist-camera
images do not depend on it, but the joint labels do. So a dataset uses ONE
nominal placement (the setup the real robot is arranged in) and each episode
only gets a small, bounded correction (dx, dy, dpsi) when the nominal one is
infeasible; see `select_delta`.

Frames
------
* frame-0 camera frame: OpenCV optical (x right, y down, z forward) of oak_l
  at the first frame of the episode. `relative_poses` expresses the whole
  episode in it.
* level frame (rows of `level_frame(n)`): forward = optical axis projected on
  the table, left, up = table normal n (pointing from the table to the camera).
* robot base frame (the URDF/MuJoCo world): the anchor camera is at
  (x, y, table_z + h0), its level frame rotated by psi about +z.

This module is pure numpy + Placo (+ MuJoCo for the collision checks). Dataset
I/O lives in examples/convert_to_jointspace.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from .kinematics import ARM_JOINT_NAMES, CONTROL_FRAME
from .pedestal import PEDESTAL_MARGIN, PedestalClearance

# Scene with the pedestal column proxy (`pedestal_box`) the collision gate needs.
DEFAULT_SCENE = Path(__file__).resolve().parent.parent / "scenes" / "table_grasp.xml"

# Arm links that must stay above the table plane (body ORIGINS, a coarse but
# conservative proxy given TABLE_MARGIN). Upper arm, elbow, forearm and wrist
# only: the gripper and its fingertips legitimately touch the table at the grasp,
# exactly as in the human demonstration.
TABLE_GUARDED_BODIES = ("dm_j4340_2_step", "dm_j4310_1_2_step", "j6_a_3_step", "j7_e_2_step")

# IK seeds tried for the first frame of an episode; the rest of the episode is
# warm-started frame to frame. The first one is the derived red-cube start pose
# (camera ~20 cm above the table, looking down and forward): it biases every
# episode toward the same elbow configuration.
# Tolerance on joint-limit margins: the IK clamps joints AT their limits, which
# reads back as a margin of about -1e-12, i.e. zero, not a violation.
MARGIN_EPS = 1e-6

HOME_SEEDS = (
    np.array([-0.259, 1.017, 0.244, 2.018, 1.062, 0.0, -0.654]),
    np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5]),
)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rotation vectors (N, 3) -> rotation matrices (N, 3, 3) (Rodrigues)."""
    rv = np.asarray(rv, dtype=float).reshape(-1, 3)
    ang = np.linalg.norm(rv, axis=1)
    k = np.divide(rv, ang[:, None], out=np.zeros_like(rv), where=ang[:, None] > 1e-12)
    K = np.zeros((len(rv), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def relative_poses(pose8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recorded poses (N, >=6: x y z rotvec) -> motion in the frame-0 camera frame.

    Returns (t_rel (N, 3), R_rel (N, 3, 3)) with T_rel(t) = T(0)^-1 T(t); frame 0
    is the identity whatever SLAM world the recording used.
    """
    pose8 = np.asarray(pose8, dtype=float)
    R = rotvec_to_matrix(pose8[:, 3:6])
    R0 = R[0]
    return (pose8[:, :3] - pose8[0, :3]) @ R0, np.einsum("ji,njk->nik", R0, R)


def level_frame(n: np.ndarray) -> np.ndarray:
    """Rows = (forward, left, up) of the level frame, expressed in the camera frame.

    up = table normal n; forward = optical axis (+z) projected on the table.
    """
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    f = np.array([0.0, 0.0, 1.0]) - n[2] * n
    f /= np.linalg.norm(f)
    return np.stack([f, np.cross(n, f), n])


def depth_to_points(depth_m: np.ndarray, K: dict, mask: np.ndarray | None, rng: np.random.Generator,
                    max_points: int = 6000, z_range: tuple[float, float] = (0.04, 1.5)) -> np.ndarray:
    """Depth image (metres) -> random subset of 3D points in the depth camera frame.

    K holds fx, fy, cx, cy. `mask` (same shape, >0 = usable) removes the
    gripper/body pixels.
    """
    valid = (depth_m > z_range[0]) & (depth_m < z_range[1])
    if mask is not None and mask.shape == depth_m.shape:
        valid &= mask > 0
    v, u = np.nonzero(valid)
    sel = rng.choice(len(v), min(max_points, len(v)), replace=False)
    v, u = v[sel], u[sel]
    z = depth_m[v, u]
    return np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], axis=1)


def fit_table_plane(points: np.ndarray, rng: np.random.Generator, n_trials: int = 400,
                    inlier_tol: float = 0.008) -> tuple[np.ndarray, float, float]:
    """Dominant plane by RANSAC, refined by SVD on the inliers.

    Returns (n, h, inlier_fraction): unit normal pointing toward the camera
    (origin) and the camera's distance to the plane.
    """
    best, n_best = None, -1
    for _ in range(n_trials):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((points - a) @ n) < inlier_tol
        if inl.sum() > n_best:
            best, n_best = inl, int(inl.sum())
    Q = points[best]
    ctr = Q.mean(axis=0)
    n = np.linalg.svd(Q - ctr)[2][-1]
    if n @ (-ctr) < 0:
        n = -n
    return n, float(n @ (-ctr)), n_best / len(points)


@dataclass(frozen=True)
class Placement:
    """Where an episode's frame-0 camera sits, in the robot base frame.

    table_z: table top height. (x, y): anchor camera position. psi_deg: heading
    of the level forward axis about +z (0 = robot +x).
    """
    table_z: float
    x: float
    y: float
    psi_deg: float

    def shifted(self, dx: float, dy: float, dpsi_deg: float) -> "Placement":
        return replace(self, x=self.x + dx, y=self.y + dy, psi_deg=self.psi_deg + dpsi_deg)


def placement_from_start_pose(T0: np.ndarray, table_z: float) -> Placement:
    """The placement whose anchor is the given robot-frame start camera pose."""
    z_axis = T0[:3, 2]
    return Placement(table_z, float(T0[0, 3]), float(T0[1, 3]), float(np.degrees(np.arctan2(z_axis[1], z_axis[0]))))


def world_poses(t_rel: np.ndarray, R_rel: np.ndarray, n: np.ndarray, h0: float,
                placement: Placement) -> np.ndarray:
    """Episode motion (frame-0 camera frame) -> control-frame poses (N, 4, 4) in the robot frame."""
    c, s = np.cos(np.radians(placement.psi_deg)), np.sin(np.radians(placement.psi_deg))
    Rw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ level_frame(n)
    T = np.tile(np.eye(4), (len(t_rel), 1, 1))
    T[:, :3, :3] = np.einsum("ij,njk->nik", Rw, R_rel)
    T[:, :3, 3] = t_rel @ Rw.T + np.array([placement.x, placement.y, placement.table_z + h0])
    return T


# --------------------------------------------------------------------------- #
# Feasibility
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Limits:
    """Per-frame feasibility thresholds.

    qstep_max: max |dq| per frame (rad). 0.10 rad/frame = 3 rad/s at 30 fps, a
    conservative stand-in for the arm's speed limit; it also catches IK branch
    flips. Joint position limits are enforced inside the IK (incl. the driver's
    wrist-roll clip), so they show up as pose error, not here.
    """
    pos_tol: float = 0.010
    rot_tol_deg: float = 5.0
    qstep_max: float = 0.10
    pedestal_margin: float = PEDESTAL_MARGIN
    table_margin: float = 0.02


@dataclass
class ChainResult:
    """Outcome of retargeting one episode at one placement."""
    ok: bool
    fail_frame: int | None
    reason: str | None
    q: np.ndarray                 # (n_solved, 7) joint trajectory (best effort past a failure)
    max_pos_err: float
    max_rot_err_deg: float
    max_step: float
    min_joint_margin: float       # rad to the nearest IK joint limit, over the episode
    min_pedestal: float           # m, capped at the pedestal margin
    min_link_height: float        # m above the table, guarded links


def load_retarget_model(scene: str | Path = DEFAULT_SCENE) -> mujoco.MjModel:
    """MuJoCo model with the pedestal proxy, loaded the way the simulator does
    (resolves the included robot's mesh paths)."""
    from .simulation import Simulation
    return Simulation(str(scene)).model


def _rot_err_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1.0) / 2.0, -1.0, 1.0))))


class Retargeter:
    """Warm-started IK along an episode + the feasibility checks.

    Args:
        kin: OpenArm `Kinematics`.
        model: MuJoCo model with a `pedestal_box` geom (see load_retarget_model).
        limits: thresholds.
        home_seeds: IK seeds for the first frame (best one kept).
        n_iter: Placo iterations per warm-started frame (10 is converged at
            30 fps motion; more iterations can jump IK branches).
    """

    def __init__(self, kin, model: mujoco.MjModel, limits: Limits = Limits(),
                 home_seeds=HOME_SEEDS, n_iter: int = 10, home_n_iter: int = 300):
        self.kin, self.model, self.limits = kin, model, limits
        self.home_seeds = [np.asarray(s, dtype=float) for s in home_seeds]
        self.n_iter, self.home_n_iter = n_iter, home_n_iter
        self.pedestal = PedestalClearance(model)
        self._data = mujoco.MjData(model)
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self._qadr = [model.jnt_qposadr[jid(n)] for n in ARM_JOINT_NAMES]
        self._mimic_qadr = model.jnt_qposadr[jid("r_wrist_roll_mimic")]
        self._roll_idx = ARM_JOINT_NAMES.index("r_wrist_roll")
        self._guard = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in TABLE_GUARDED_BODIES]
        pm = kin.robot.model
        idx = [kin.robot.get_joint_offset(n) for n in ARM_JOINT_NAMES]
        self.q_lo = np.array([pm.lowerPositionLimit[i] for i in idx])
        self.q_hi = np.array([pm.upperPositionLimit[i] for i in idx])

    def _links_min_height(self, q: np.ndarray) -> float:
        d = self._data
        for adr, v in zip(self._qadr, q):
            d.qpos[adr] = v
        d.qpos[self._mimic_qadr] = -q[self._roll_idx]
        mujoco.mj_kinematics(self.model, d)
        return float(min(d.xpos[b][2] for b in self._guard))

    def _solve(self, T: np.ndarray, seed: np.ndarray, n_iter: int):
        q = self.kin.inverse(T, current_joint_positions=seed, n_iter=n_iter, frame=CONTROL_FRAME)
        Ta = self.kin.forward(q, frame=CONTROL_FRAME)
        return q, float(np.linalg.norm(Ta[:3, 3] - T[:3, 3])), _rot_err_deg(Ta[:3, :3], T[:3, :3])

    def chain(self, Ts: np.ndarray, table_z: float, early_stop: bool = True) -> ChainResult:
        """IK along the poses Ts (N, 4, 4); first failing check decides the reason."""
        L = self.limits
        best = None
        for s in self.home_seeds:
            sol = self._solve(Ts[0], s, self.home_n_iter)
            if best is None or sol[1] + sol[2] / 1000 < best[1] + best[2] / 1000:
                best = sol
        q, pe, re = best
        qs = [q]
        w = dict(pe=pe, re=re, step=0.0, margin=np.inf, ped=np.inf, link=np.inf)
        fail = reason = None
        for i in range(len(Ts)):
            if i > 0:
                if fail is not None and early_stop:
                    break
                qn, pe, re = self._solve(Ts[i], q, self.n_iter)
                step = float(np.abs(qn - q).max())
                q = qn
                qs.append(q)
            else:
                step = 0.0
            margin = float(min((q - self.q_lo).min(), (self.q_hi - q).min()))
            ped = self.pedestal.clearance(q)
            link = self._links_min_height(q) - table_z
            w = dict(pe=max(w["pe"], pe), re=max(w["re"], re), step=max(w["step"], step),
                     margin=min(w["margin"], margin), ped=min(w["ped"], ped), link=min(w["link"], link))
            if fail is None:
                pose_bad = pe > L.pos_tol or re > L.rot_tol_deg
                r = (f"home unreachable ({pe * 1000:.0f}mm/{re:.0f}deg)" if pose_bad and i == 0 else
                     f"pos {pe * 1000:.0f}mm" if pe > L.pos_tol else
                     f"rot {re:.0f}deg" if re > L.rot_tol_deg else
                     f"step {step:.2f}rad" if step > L.qstep_max else
                     f"pedestal {ped * 1000:.0f}mm" if ped < L.pedestal_margin else
                     f"link below table margin ({link * 1000:.0f}mm)" if link < L.table_margin else None)
                if r is not None:
                    fail, reason = i, r
        return ChainResult(ok=fail is None, fail_frame=fail, reason=reason, q=np.array(qs),
                           max_pos_err=w["pe"], max_rot_err_deg=w["re"], max_step=w["step"],
                           min_joint_margin=w["margin"], min_pedestal=w["ped"], min_link_height=w["link"])

    def max_fk_error(self, q: np.ndarray, Ts: np.ndarray) -> float:
        """Max control-frame position error of FK(q) against Ts."""
        return float(max(np.linalg.norm(self.kin.forward(qi, frame=CONTROL_FRAME)[:3, 3] - T[:3, 3])
                         for qi, T in zip(q, Ts)))

    def choose_delta(self, t_rel, R_rel, n, h0, placement: Placement, deltas, margin_target: float,
                     bounds: tuple[float, float]):
        """Per-episode correction: the nominal placement if it is feasible with an
        adequate joint margin, else `select_delta` over the candidate grid.

        Returns (delta | None, ChainResult | None, failure-reason Counter)."""
        zero = (0.0, 0.0, 0.0)
        r0 = self.chain(world_poses(t_rel, R_rel, n, h0, placement), placement.table_z)
        if r0.ok and r0.min_joint_margin >= margin_target - MARGIN_EPS:
            return zero, r0, Counter()
        cands = [(zero, r0)] + [
            (d, self.chain(world_poses(t_rel, R_rel, n, h0, placement.shifted(*d)), placement.table_z))
            for d in deltas if tuple(d) != zero]
        return select_delta(cands, margin_target, bounds)


def select_delta(candidates, margin_target: float, bounds: tuple[float, float]):
    """Pick the correction for one episode among evaluated candidates.

    candidates: [((dx, dy, dpsi_deg), ChainResult), ...]. Among the feasible
    ones, prefer (1) a joint margin >= margin_target, then (2) the smallest
    correction, measured in bound units |dxy|/bounds[0] + |dpsi|/bounds[1]
    (consistency: joint labels should come from one placement), then (3) the
    larger margin; ties keep candidate order.

    Returns (delta, ChainResult, Counter()) or (None, None, Counter of the
    failure reasons' first word) when nothing is feasible.
    """
    feasible = [(d, r) for d, r in candidates if r.ok]
    if not feasible:
        return None, None, Counter(r.reason.split()[0] for _, r in candidates)
    size = lambda d: np.hypot(d[0], d[1]) / bounds[0] + abs(d[2]) / bounds[1]  # noqa: E731
    d, r = sorted(feasible, key=lambda c: (c[1].min_joint_margin < margin_target - MARGIN_EPS, size(c[0]),
                                           -c[1].min_joint_margin))[0]
    return d, r, Counter()


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


def repair_joint_spikes(q: np.ndarray, thr: float = 0.3) -> int:
    """Replace ISOLATED one-frame IK glitches by the neighbour midpoint (in place).

    Frame k is a glitch if it jumps > thr (max over joints) from both neighbours
    while the neighbours agree (< thr). Sustained jumps (reconfigurations) are
    left alone — the feasibility step limit rejects those episodes instead.
    Returns the number of frames repaired. One episode per call.
    """
    fixed = 0
    for k in range(1, len(q) - 1):
        if (np.abs(q[k] - q[k - 1]).max() > thr and np.abs(q[k + 1] - q[k]).max() > thr
                and np.abs(q[k + 1] - q[k - 1]).max() < thr):
            q[k] = 0.5 * (q[k - 1] + q[k + 1])
            fixed += 1
    return fixed


def joint_state_action(q: np.ndarray, grip_state: np.ndarray,
                       grip_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One episode -> state = [q_t, grip_state_t], action = [q_t+1 (last repeated), grip_action_t].

    The gripper channels are copied unchanged from the Cartesian dataset being
    mirrored, so both representations share the exact same gripper supervision:
    with the grasp projection the commanded closure (action) is NOT the next
    measured closure (state) — it is 1.0 through the grasp — so it must not be
    derived from the state.
    """
    q_next = np.concatenate([q[1:], q[-1:]], axis=0)
    state = np.concatenate([q, grip_state], axis=1).astype(np.float32)
    action = np.concatenate([q_next, grip_action], axis=1).astype(np.float32)
    return state, action
