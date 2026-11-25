"""Retarget a handheld Grabette dataset to OpenArm JOINT space (for joint-vs-Cartesian A/B).

The output mirrors a prepared Cartesian dataset (--src_root, e.g. run_pipeline.sh's
graspproj or cartesian_480): same images, same episodes (minus the infeasible
ones), same gripper channels copied unchanged; only the arm part changes:

    observation.state = [q0..q6, gripper_state(2)]
    action            = [q0..q6 at t+1, gripper_action(2)]

so the gripper representation is chosen by the source: graspproj -> projected
(strategy, closure), cartesian_480 -> raw joint angles.

How the joints are obtained (see openarm_gripette_simu/retarget.py):
 1. Per episode, the 8D absolute poses (--poses_root, the cleaned dataset, same
    episode order as --src_root) are re-anchored at frame 0; the table plane in
    the frame-0 depth image (--raw_repo, one dir per ORIGINAL episode) gives the
    camera height/tilt/roll above a level table.
 2. One nominal placement for the whole dataset: table height fixed by
    --table_below_shoulder, anchor position/heading searched (coarse to fine) on
    a subsample to maximise episodes feasible as-is — or given by --nominal.
 3. Per episode: the nominal placement if feasible with margin, else the
    smallest bounded correction (+-dxy, +-dpsi) that is; else the episode is
    DROPPED (reasons recorded).
 4. Final IK chain, isolated-glitch repair, dataset written.

meta/retarget.json records the setup (placement, table height below the
shoulder), every episode's correction and errors, and the dropped episodes.
--cartesian_out writes the Cartesian source cut to the same episode set, for a
fair A/B.

Run in the simu env with the dataset extra, e.g.:
  uv run --package openarm-gripette-simu --extra dataset python \\
    robots/openarm/openarm_gripette_simu/examples/convert_to_jointspace.py \\
    --src_root   ~/grabette-work/put_red_cube_green_box/pipeline/graspproj \\
    --poses_root ~/grabette-work/put_red_cube_green_box/pipeline/clean \\
    --orig_root  ~/grabette-work/put_red_cube_green_box/raw \\
    --raw_repo   glannuzel/put-red-cube-in-green-box-raw \\
    --output_root ~/grabette-work/put_red_cube_green_box/pipeline/jointspace \\
    --cartesian_out ~/grabette-work/put_red_cube_green_box/pipeline/graspproj_jointset
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger("convert_to_jointspace")

JOINT_NAMES = [f"q{i}" for i in range(7)]


# --------------------------------------------------------------------------- #
# Dataset helpers (pyarrow only; unit-tested)
# --------------------------------------------------------------------------- #


def read_episodes(root: Path, column: str) -> list[np.ndarray]:
    """Per-episode arrays of a list column, ordered by episode_index then frame_index."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.concat_tables([pq.read_table(p, columns=["episode_index", "frame_index", column])
                          for p in sorted((Path(root) / "data").rglob("*.parquet"))]).to_pydict()
    ep, fr = np.asarray(t["episode_index"]), np.asarray(t["frame_index"])
    vals = np.asarray(t[column], dtype=np.float64)
    out = []
    for e in range(int(ep.max()) + 1):
        m = ep == e
        out.append(vals[m][np.argsort(fr[m])])
    return out


def match_episodes(cleaned: list[np.ndarray], orig: list[np.ndarray]) -> list[int]:
    """Original index of each cleaned episode: same length and identical first frame.

    Cleaning only deletes whole episodes (and renumbers), so this is exact."""
    keys: dict = {}
    for i, a in enumerate(orig):
        keys.setdefault((len(a), a[0].tobytes()), []).append(i)
    out = []
    for i, a in enumerate(cleaned):
        cands = keys.get((len(a), np.asarray(a[0], dtype=orig[0].dtype).tobytes()), [])
        if len(cands) != 1:
            raise ValueError(f"cleaned episode {i}: {len(cands)} matching original episodes (expected 1)")
        out.append(cands[0])
    return out


def drop_undeclared_episode_stats(root: Path) -> list[str]:
    """Remove per-episode `stats/<feature>/...` columns of features absent from info.json (in place).

    lerobot's remove_feature (used by clean_dataset.py to drop extra cameras)
    leaves the removed camera's per-episode stats behind; delete_episodes then
    cannot reshape them (it only does so for declared features) and fails with
    "Shape of quantile 'min' must be (3,1,1)". Returns the features dropped."""
    import pyarrow.parquet as pq

    declared = set(json.loads((Path(root) / "meta" / "info.json").read_text())["features"])
    dropped = set()
    for p in sorted((Path(root) / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(p)
        stale = [c for c in t.column_names if c.startswith("stats/") and c.split("/")[1] not in declared]
        if stale:
            dropped |= {c.split("/")[1] for c in stale}
            pq.write_table(t.drop_columns(stale), p)
    return sorted(dropped)


def rewrite_joint_columns(root: Path, state_by_ep: dict, action_by_ep: dict, gripper_names: list[str]) -> None:
    """Replace `observation.state` / `action` row by row from per-episode arrays
    (indexed by episode_index, frame_index) and update meta/info.json."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for p in sorted((Path(root) / "data").rglob("*.parquet")):
        t = pq.read_table(p)
        ep, fr = t.column("episode_index").to_pylist(), t.column("frame_index").to_pylist()
        cols = {c: t.column(c) for c in t.column_names}
        cols["observation.state"] = pa.array([state_by_ep[e][f].tolist() for e, f in zip(ep, fr)], pa.list_(pa.float32()))
        cols["action"] = pa.array([action_by_ep[e][f].tolist() for e, f in zip(ep, fr)], pa.list_(pa.float32()))
        pq.write_table(pa.table(cols), p)
    info_path = Path(root) / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        info["features"][key] = {"dtype": "float32", "shape": [9], "names": JOINT_NAMES + list(gripper_names)}
    info_path.write_text(json.dumps(info, indent=4))


# --------------------------------------------------------------------------- #
# Table planes from the raw depth (frame 0)
# --------------------------------------------------------------------------- #


def table_planes(orig_indices: list[int], raw_repo: str, cache: Path) -> dict[int, dict]:
    """Frame-0 table plane per ORIGINAL episode (raw dirs sorted == original order). Cached."""
    import av
    import cv2
    from huggingface_hub import HfApi, hf_hub_download

    from openarm_gripette_simu.retarget import depth_to_points, fit_table_plane

    planes = {int(k): v for k, v in json.loads(cache.read_text()).items()} if cache.is_file() else {}
    todo = [i for i in orig_indices if i not in planes]
    if todo:
        dirs = sorted({f.split("/")[0] for f in HfApi().list_repo_files(raw_repo, repo_type="dataset") if "/" in f})
        rng = np.random.default_rng(0)
        for k, i in enumerate(todo):
            get = lambda name: hf_hub_download(raw_repo, f"{dirs[i]}/right/{name}", repo_type="dataset")  # noqa: E731
            K = json.loads(Path(get("dcam_calib_offline.json")).read_text())
            mask = cv2.imread(get("dcam_mask.png"), cv2.IMREAD_GRAYSCALE)
            with av.open(get("dcam_depth.mkv")) as c:
                depth = next(c.decode(video=0)).to_ndarray().astype(np.float32) / 1000.0   # mm -> m
            n, h0, frac = fit_table_plane(depth_to_points(depth, K, mask, rng), rng)
            planes[i] = dict(dir=dirs[i], n=n.tolist(), h0=h0, inliers=frac)
            logger.info(f"  plane {k + 1}/{len(todo)} orig ep {i}: h0 {h0 * 1000:.0f} mm, inliers {frac:.2f}")
            cache.write_text(json.dumps(planes, indent=1))
    return planes


# --------------------------------------------------------------------------- #
# Parallel workers (one Retargeter per process)
# --------------------------------------------------------------------------- #

_W: dict = {}


def _init_worker(limits_kw: dict, n_iter: int, episodes: dict | None):
    from openarm_gripette_simu.kinematics import Kinematics
    from openarm_gripette_simu.retarget import Limits, Retargeter, load_retarget_model

    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        _W["rt"] = Retargeter(Kinematics(), load_retarget_model(), Limits(**limits_kw), n_iter=n_iter)
    _W["eps"] = episodes


def _score_nominal(placement):
    """(placement, #episodes feasible as-is, mean feasible prefix) on the search subsample."""
    from openarm_gripette_simu.retarget import world_poses

    ok, prefix = 0, []
    for e in _W["eps"].values():
        r = _W["rt"].chain(world_poses(e["t"], e["R"], e["n"], e["h0"], placement), placement.table_z)
        ok += r.ok
        prefix.append(1.0 if r.ok else r.fail_frame / len(e["t"]))
    return placement, ok, float(np.mean(prefix))


def _retarget_episode(args):
    """Choose the correction, then return the full joint trajectory (or the failure reasons)."""
    from openarm_gripette_simu.retarget import repair_joint_spikes, world_poses

    idx, e, placement, deltas, margin_target, bounds = args
    rt = _W["rt"]
    delta, res, reasons = rt.choose_delta(e["t"], e["R"], e["n"], e["h0"], placement, deltas, margin_target, bounds)
    if delta is None:
        return idx, None, dict(reasons)
    Ts = world_poses(e["t"], e["R"], e["n"], e["h0"], placement.shifted(*delta))
    q = res.q.copy()
    n_fixed = repair_joint_spikes(q)
    return idx, dict(delta=list(delta), q=q, n_repaired=n_fixed, max_fk_err=rt.max_fk_error(q, Ts),
                     max_pos_err=res.max_pos_err, max_rot_err_deg=res.max_rot_err_deg, max_step=res.max_step,
                     min_joint_margin=res.min_joint_margin, min_pedestal=res.min_pedestal,
                     min_link_height=res.min_link_height), {}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def search_nominal(pool, table_z: float, n_workers: int):
    """Coarse grid then a finer grid around the best; returns the best Placement and its score."""
    from openarm_gripette_simu.retarget import Placement

    def run(grid):
        res = pool.map(_score_nominal, grid, chunksize=max(1, len(grid) // (4 * n_workers)))
        return max(res, key=lambda r: (r[1], r[2]))

    coarse = [Placement(table_z, x, y, p) for x, y, p in itertools.product(
        np.arange(0.05, 0.401, 0.05), np.arange(-0.40, 0.101, 0.05), np.arange(-45.0, 60.1, 15.0))]
    best, ok, pf = run(coarse)
    logger.info(f"coarse nominal search ({len(coarse)}): best {best} -> {ok} as-is, prefix {pf:.2f}")
    fine = [best.shifted(dx, dy, dp) for dx, dy, dp in itertools.product(
        np.arange(-0.05, 0.051, 0.0125), np.arange(-0.05, 0.051, 0.0125), np.arange(-15.0, 15.1, 3.75))]
    best, ok, pf = run(fine + [best])
    logger.info(f"fine nominal search ({len(fine)}): best {best} -> {ok} as-is, prefix {pf:.2f}")
    return best, ok


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_root", required=True, help="Prepared Cartesian dataset to mirror (images, episodes, gripper).")
    p.add_argument("--poses_root", required=True, help="Cleaned 8D dataset: absolute poses, same episode order as src.")
    p.add_argument("--orig_root", required=True, help="Pre-cleaning 8D dataset (original episode order == raw dirs).")
    p.add_argument("--raw_repo", required=True, help="Hub repo with the raw recordings (depth), one dir per episode.")
    p.add_argument("--output_root", required=True)
    p.add_argument("--repo_id", default=None, help="Local id for the output (default local/<output dir name>).")
    p.add_argument("--cartesian_out", default=None, help="Also write --src_root cut to the kept episodes here.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--table_below_shoulder", type=float, default=0.34,
                   help="Table top distance below the shoulder pitch axis (m). The real setup must match.")
    p.add_argument("--nominal", type=float, nargs=3, metavar=("X", "Y", "PSI_DEG"), default=None,
                   help="Nominal anchor placement in the robot base frame; searched if omitted.")
    p.add_argument("--search_episodes", type=int, default=30, help="Episodes (evenly spaced) for the nominal search.")
    p.add_argument("--dxy", type=float, default=0.05, help="Per-episode correction bound, position (m).")
    p.add_argument("--dpsi", type=float, default=15.0, help="Per-episode correction bound, heading (deg).")
    p.add_argument("--grid_steps", type=int, default=5, help="Correction grid points per axis (odd).")
    p.add_argument("--margin_target", type=float, default=0.1, help="Preferred min joint-limit margin (rad).")
    p.add_argument("--pos_tol", type=float, default=0.010)
    p.add_argument("--rot_tol_deg", type=float, default=5.0)
    p.add_argument("--qstep_max", type=float, default=0.10, help="Max joint step per frame (rad).")
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()

    from openarm_gripette_simu.retarget import Placement, load_retarget_model, relative_poses

    src, out = Path(args.src_root).expanduser(), Path(args.output_root).expanduser()
    for d in [out] + ([Path(args.cartesian_out).expanduser()] if args.cartesian_out else []):
        if d.exists():
            if not args.overwrite:
                raise SystemExit(f"{d} exists; pass --overwrite.")
            shutil.rmtree(d)

    # ---- episodes: cleaned poses, mapping to original indices, src consistency
    poses = read_episodes(Path(args.poses_root).expanduser(), "action")
    orig_map = match_episodes(poses, read_episodes(Path(args.orig_root).expanduser(), "action"))
    src_state = read_episodes(src, "observation.state")
    src_action = read_episodes(src, "action")
    if [len(a) for a in src_state] != [len(a) for a in poses]:
        raise SystemExit("--src_root and --poses_root episodes differ in number or length.")
    logger.info(f"{len(poses)} episodes; original indices {orig_map[:5]}...")

    planes = table_planes(orig_map, args.raw_repo, out.parent / f"{out.name}_planes.json")
    episodes = {}
    for i, a in enumerate(poses):
        t, R = relative_poses(a)
        pl = planes[orig_map[i]]
        episodes[i] = dict(t=t, R=R, n=np.array(pl["n"]), h0=pl["h0"])

    # ---- table height from the shoulder
    import mujoco
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        model = load_retarget_model()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    shoulder = data.xanchor[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_arm_pitch")].copy()
    table_z = float(shoulder[2] - args.table_below_shoulder)
    logger.info(f"shoulder pitch axis at {np.round(shoulder, 3).tolist()} -> table_z {table_z:.3f} m (base frame)")

    limits_kw = dict(pos_tol=args.pos_tol, rot_tol_deg=args.rot_tol_deg, qstep_max=args.qstep_max)
    sub = {i: episodes[i] for i in np.linspace(0, len(episodes) - 1, min(args.search_episodes, len(episodes))).astype(int)}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(limits_kw, args.n_iter, sub)) as pool:
        if args.nominal:
            nominal, n_asis = Placement(table_z, *args.nominal), None
        else:
            nominal, n_asis = search_nominal(pool, table_z, args.workers)
    logger.info(f"nominal placement: {nominal}")

    axis = lambda b: np.linspace(-b, b, args.grid_steps)  # noqa: E731
    deltas = [tuple(float(v) for v in d) for d in itertools.product(axis(args.dxy), axis(args.dxy), axis(args.dpsi))]
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(limits_kw, args.n_iter, None)) as pool:
        results = pool.map(_retarget_episode, [(i, e, nominal, deltas, args.margin_target, (args.dxy, args.dpsi))
                                               for i, e in episodes.items()])
    kept = {i: r for i, r, _ in results if r is not None}
    dropped = {i: reasons for i, r, reasons in results if r is None}
    logger.info(f"kept {len(kept)}/{len(episodes)}; dropped {sorted(dropped)}")

    # ---- write: src minus dropped -> output, then joint columns
    from lerobot.datasets import LeRobotDataset, delete_episodes
    from lerobot.datasets.dataset_tools import recompute_stats

    repo_id = args.repo_id or f"local/{out.name}"
    # Work from a scratch copy of src (never modified) with stale per-episode stats removed.
    tmp = out.parent / f"{out.name}_srctmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    stale = drop_undeclared_episode_stats(tmp)
    if stale:
        logger.info(f"dropped stale per-episode stats of undeclared features {stale}")
    targets = [(out, repo_id)] + ([(Path(args.cartesian_out).expanduser(), f"local/{Path(args.cartesian_out).name}")]
                                  if args.cartesian_out else [])
    for dst, rid in targets:
        if dropped:
            delete_episodes(LeRobotDataset(f"local/{tmp.name}", root=tmp), sorted(dropped), output_dir=dst, repo_id=rid)
        else:
            shutil.copytree(tmp, dst)
    shutil.rmtree(tmp)
    from openarm_gripette_simu.retarget import joint_state_action
    state_by_ep, action_by_ep = {}, {}
    for new_i, old_i in enumerate(sorted(kept)):          # delete_episodes renumbers in order
        state_by_ep[new_i], action_by_ep[new_i] = joint_state_action(
            kept[old_i]["q"], src_state[old_i], src_action[old_i][:, -2:])
    src_info = json.loads((src / "meta" / "info.json").read_text())
    grip_names = (src_info["features"]["observation.state"].get("names") or ["gripper_0", "gripper_1"])[-2:]
    rewrite_joint_columns(out, state_by_ep, action_by_ep, grip_names)
    stats_path = out / "meta" / "stats.json"
    if stats_path.is_file():   # stale provenance of the Cartesian action stats (see grasp_projection_convert)
        stats = json.loads(stats_path.read_text())
        for k in ("action_relative_meta", "action_absolute"):
            stats.pop(k, None)
        stats_path.write_text(json.dumps(stats, indent=2))
    recompute_stats(LeRobotDataset(repo_id, root=out), skip_image_video=True)

    meta = dict(
        source=dict(src_root=str(src), poses_root=args.poses_root, orig_root=args.orig_root, raw_repo=args.raw_repo),
        setup=dict(table_below_shoulder_m=args.table_below_shoulder, shoulder_pitch_axis_base_frame=shoulder.tolist(),
                   table_z_base_frame=table_z, nominal=dict(x=nominal.x, y=nominal.y, psi_deg=nominal.psi_deg),
                   nominal_feasible_as_is_on_search_subset=n_asis,
                   search_subset=None if args.nominal else sorted(int(i) for i in sub)),
        limits=dict(**limits_kw, n_iter=args.n_iter, dxy=args.dxy, dpsi=args.dpsi, grid_steps=args.grid_steps,
                    margin_target=args.margin_target),
        episodes={str(new_i): dict(src_episode=old_i, orig_episode=orig_map[old_i],
                                   **{k: v for k, v in kept[old_i].items() if k != "q"})
                  for new_i, old_i in enumerate(sorted(kept))},
        dropped={str(i): dict(orig_episode=orig_map[i], reasons=r) for i, r in sorted(dropped.items())},
        kept_src_episodes=sorted(kept),
    )
    (out / "meta" / "retarget.json").write_text(json.dumps(meta, indent=1, default=float))
    d = np.array([kept[i]["delta"] for i in kept])
    logger.info(
        f"done: {out}\n  kept {len(kept)}/{len(episodes)} | corrected {int((np.abs(d).sum(1) > 0).sum())} | "
        f"max FK err {max(k['max_fk_err'] for k in kept.values()) * 1000:.1f} mm | "
        f"min joint margin {min(k['min_joint_margin'] for k in kept.values()):.2f} rad | "
        f"repaired frames {sum(k['n_repaired'] for k in kept.values())}")


if __name__ == "__main__":
    main()
