"""Re-express a dataset's gripper channels as (strategy, commanded closure).

WHAT CHANGES
------------
The two gripper channels are replaced, in place of the raw angles:

    action[-2:]            -> (s, c_target)
    observation.state[-2:] -> (s_obs, c_obs)

`c_target` is what the policy should COMMAND:

    close      -> 1.0        drive fully closed along `s`; the OBJECT stops the
                             fingers, at the servo's torque cap
    otherwise  -> passthrough of the closure the human actually used

The full close fires at the onset of the closing ramp by default. For operators
who close slowly while still approaching, `--close_at` / `--close_stop_speed`
move it to the grasp itself (see `delay_close_onsets`, and "When the full close
fires" in docs/grasp_projection.md).

So the ONLY thing this conversion changes is that the grasp becomes a full close.
Everything else is the demonstration unaltered, which keeps the distribution shift
as small as it can be.

An earlier version commanded `c = 0` (fully open) on frames labelled "open". That
was wrong: humans open only about as wide as the grasp needs, and most detected
opens in the existing data are start-of-episode pre-shape adjustments, not
releases. Commanding a full open there changed the approach posture by up to 76
degrees and would have put the observed gripper out of distribution. Detecting
opens still matters — it is what stops a full close being commanded during the
approach or a release — it just does not dictate the value.

`c_obs` stays CONTINUOUS on purpose. The command is the same for rest and for a
grasp, so "am I actually holding something" lives in the observation — the jaws
reach the mechanical stop when empty and are blocked short when not. That is the
right place for it: something the policy can condition on rather than infer.

WHY
---
Replayed angles under-close. The demonstrated angle is where the human's fingers
sat *while pressing* the object, and a position servo reproducing it stops just
short. Measured over these datasets, demonstrations use only 38-60% of the
proximal range. Commanding `c = 1` removes the need to predict an
object-dependent angle at all.

Non-destructive: the source dataset is copied, never modified.

PUSHING THE RESULT TO THE HUB
----------------------------
`hf upload` is not sufficient on its own. LeRobot resolves a dataset by a git TAG
matching its `codebase_version` (e.g. `v3.0`), and `hf upload` only writes `main`.
Without the tag, training dies in `get_safe_version` with a
`RevisionNotFoundError` that lerobot then re-raises incorrectly, surfacing as an
unrelated `TypeError: HfHubHTTPError.__init__() missing ... 'response'`.

So after uploading:

    from huggingface_hub import create_tag
    create_tag(repo_id, tag="v3.0", repo_type="dataset", revision="main")

using whatever `codebase_version` meta/info.json reports.

The other thing `hf upload` skips is the dataset CARD. LeRobot's own
`push_to_hub` renders one from a template; uploading files directly does not, so
the repo lands with no README, no `LeRobot` tag, and — the part that is easy to
miss — no link to the online viewer. Include this block in the card, since the
Space takes the repo id as `path` (NOT `dataset`, which silently shows nothing):

    <a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=<repo_id>">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
    </a>

and keep `tags: [LeRobot]` in the front matter, which is what makes the dataset
discoverable as a LeRobot one.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from grabette_dataset_tools.grasp_projection import (
    PROJECTION_SIDECAR,
    GraspProjection,
    normalize_closing_sign,
    segment_grip,
)

# Card front matter for a converted dataset. `grasp-projection` is queryable
# through the Hub API without downloading anything, which is what makes a
# projected dataset identifiable from a listing; the channel names in info.json
# remain the authoritative signal.
PROJECTION_HUB_TAG = "grasp-projection"
_CARD_TAGS = ["LeRobot", "grabette", "gripette", PROJECTION_HUB_TAG]

logger = logging.getLogger(__name__)

C_CLOSE = 1.0   # every other frame passes the demonstrated closure through

# Label for the frames of a closing ramp BEFORE the delayed full close fires
# (see `delay_close_onsets`). Anything but "close" passes the observed closure
# through, so these frames replay the human's own gradual squeeze.
CLOSING = "closing"
# Width of the centred moving average applied to the hand speed before the
# stop test: one 30 fps frame of SLAM jitter must not read as "stopped".
SPEED_SMOOTH = 5


def hand_speed(pos, fps: float) -> np.ndarray:
    """Per-frame hand speed (m/s) from positions, smoothed over SPEED_SMOOTH frames.

    Frame 0 copies frame 1's step (there is no step before it). The average is
    normalised by the number of frames actually inside the window, so the ends of
    the episode are not biased towards zero — which would read as "stopped".
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    if n < 2:
        return np.zeros(n)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) * float(fps)
    v = np.concatenate([step[:1], step])
    half = SPEED_SMOOTH // 2
    csum = np.concatenate([[0.0], np.cumsum(v)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (csum[hi] - csum[lo]) / (hi - lo)


def delay_close_onsets(
    labels: list[str],
    closure: list[float],
    close_at: float | None = None,
    speed=None,
    stop_speed: float | None = None,
) -> tuple[list[str], list[int]]:
    """Move the start of each full close from the ramp onset to the grasp itself.

    `segment_grip` fires the close at the START of the closing ramp — right when
    the human commits. That is correct for operators who close in one quick
    squeeze once in place, but some close slowly WHILE still approaching: on the
    put-red-cube dataset (Nov. 2026) the ramp onset came a median 0.77 s and
    8.4 cm before the hand stopped on the cube, at 0.20 m/s. A servo commanded to
    c = 1 there closes at its own speed, far faster than the hand did, and shuts
    the fingers before the object is between them.

    For every span of "close" frames that starts after a non-close frame, the full
    close is moved to the EARLIEST of:

      - `close_at`: the first frame where the observed closure has covered this
        fraction of the span's rise (from its value at the onset to its maximum
        within the span);
      - `stop_speed`: the first frame where the hand speed drops below this
        value (m/s).

    Using the earlier of the two is deliberate: the fraction alone sometimes fires
    late, when the operator keeps squeezing during transport; the stop alone never
    fires in episodes where the hand does not pause. Frames in between are
    relabelled CLOSING, so the action replays the human's own closure there.

    A span starting at frame 0 (rest-closed convention, idle hand) is left alone:
    there is no ramp to delay. Returns the new labels and the delay applied to
    each delayed span, in frames.
    """
    labels = list(labels)
    if close_at is None and stop_speed is None:
        return labels, []
    if close_at is not None and not 0.0 < close_at <= 1.0:
        raise ValueError(f"close_at must be in (0, 1], got {close_at}")
    if stop_speed is not None:
        if speed is None:
            raise ValueError("stop_speed needs the hand speed (positions + fps)")
        if len(speed) != len(labels):
            raise ValueError(f"speed has {len(speed)} frames, labels {len(labels)}")
    c = np.asarray(closure, dtype=np.float64)
    n = len(labels)
    delays = []
    t0 = 1
    while t0 < n:
        if labels[t0] != "close" or labels[t0 - 1] == "close":
            t0 += 1
            continue
        t1 = t0
        while t1 < n and labels[t1] == "close":
            t1 += 1
        candidates = []
        if close_at is not None:
            lo, hi = c[t0], c[t0:t1].max()
            reached = np.nonzero(c[t0:t1] >= lo + close_at * (hi - lo))[0]
            candidates.append(t0 + int(reached[0]))   # the maximum always qualifies
        if stop_speed is not None:
            slow = np.nonzero(np.asarray(speed[t0:t1]) < stop_speed)[0]
            if len(slow):
                candidates.append(t0 + int(slow[0]))
        trigger = min(candidates) if candidates else t0
        for i in range(t0, trigger):
            labels[i] = CLOSING
        if trigger > t0:
            delays.append(trigger - t0)
        t0 = t1
    return labels, delays


def _gripper_columns(names: list[str] | None, dim: int) -> tuple[int, int]:
    """Locate (proximal, distal) BY NAME. Layouts differ between dataset eras —
    8-D [x,y,z,ax,ay,az,proximal,distal] and 11-D [dx,dy,dz,dr6d_0..5,
    proximal,distal] — so a positional guess silently reads rotation channels
    as gripper angles.
    """
    if names:
        low = [str(n).lower() for n in names]
        ip = next((i for i, n in enumerate(low) if "proximal" in n), None)
        idl = next((i for i, n in enumerate(low) if "distal" in n and "dr6d" not in n), None)
        if ip is not None and idl is not None:
            return ip, idl
    logger.warning("Gripper channels not found by name; falling back to the last two")
    return dim - 2, dim - 1


def _pose_columns(names: list[str] | None) -> tuple[list[int], bool] | None:
    """Locate the hand position channels BY NAME: ([ix, iy, iz], is_delta).

    Absolute layouts name them x/y/z (or right_x...), delta layouts dx/dy/dz;
    deltas are integrated by the caller. None when neither is present.
    """
    if not names:
        return None
    low = [str(n).lower() for n in names]
    for prefix, is_delta in (("", False), ("d", True)):
        idx = []
        for axis in "xyz":
            key = prefix + axis
            idx.append(next((i for i, n in enumerate(low)
                             if n == key or n.endswith("_" + key)), None))
        if None not in idx:
            return idx, is_delta
    return None


def convert_episode(
    prox: list[float],
    dist: list[float],
    gp: GraspProjection,
    rest_is_closed: bool,
    close_at: float | None = None,
    stop_speed: float | None = None,
    pos=None,
    fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One episode -> (action gripper pairs, state gripper pairs, report).

    `close_at` / `stop_speed` delay each full close from the ramp onset to the
    grasp (see `delay_close_onsets`); `stop_speed` needs `pos` (N x 3, metres)
    and `fps`. Both None (the default) keeps the original behaviour.
    """
    prox, flipped_p = normalize_closing_sign(prox)
    dist, flipped_d = normalize_closing_sign(dist)

    # Clamp negatives: the recorded open pose sits slightly BELOW zero (the
    # device's zero is a calibration, good to ~1 deg), and a negative angle has
    # no meaning in the projection.
    s_obs, c_obs = [], []
    for p_, d_ in zip(prox, dist):
        s, c = gp.encode(max(p_, 0.0), max(d_, 0.0))
        s_obs.append(s)
        c_obs.append(c)

    labels, events = segment_grip(c_obs, rest_is_closed=rest_is_closed)

    speed = None
    if stop_speed is not None:
        if pos is None or fps is None:
            raise ValueError("stop_speed needs the hand positions and the fps")
        if len(pos) != len(c_obs):
            raise ValueError(f"pos has {len(pos)} frames, gripper {len(c_obs)}")
        speed = hand_speed(pos, fps)
    labels, delays = delay_close_onsets(labels, c_obs, close_at=close_at,
                                        speed=speed, stop_speed=stop_speed)

    # Only the grasp is rewritten; approach, open and rest pass through, so the
    # decoded command matches the demonstration everywhere except where the
    # under-close happens.
    c_target = [C_CLOSE if lab == "close" else c_obs[i]
                for i, lab in enumerate(labels)]

    # LATCH the strategy across each close. Left per-frame, `s` drifts while the
    # fingers settle onto the object, and since the commanded pose is
    # decode(s, 1.0), the target then wanders — observed swinging between 57 and
    # 93 degrees of proximal mid-grasp, which would make a servo already stalled
    # on the object shuffle. A grasp has ONE shape, so it is taken from the pose
    # the hand ends in (the plateau) and held from the onset — or from the
    # delayed full close, when one was applied: before it the gripper replays
    # the human's own pose, strategy included.
    s_cmd = list(s_obs)
    for ev in (e for e in events if e.kind == "close"):
        tail = s_obs[max(ev.end - 5, ev.onset):ev.end + 1]
        if not tail:
            continue
        grasp_s = sorted(tail)[len(tail) // 2]
        start = next((i for i in range(ev.onset, len(labels)) if labels[i] == "close"),
                     None)
        if start is None:
            continue
        end = next((i for i in range(start, len(labels)) if labels[i] != "close"),
                   len(labels))
        for i in range(start, end):
            s_cmd[i] = grasp_s
    return (
        np.asarray(list(zip(s_cmd, c_target)), dtype=np.float32),
        np.asarray(list(zip(s_obs, c_obs)), dtype=np.float32),
        {
            "n_close": sum(1 for e in events if e.kind == "close"),
            "n_open": sum(1 for e in events if e.kind == "open"),
            "sign_flipped": bool(flipped_p or flipped_d),
            "frac_close": labels.count("close") / max(len(labels), 1),
            "close_delays": delays,
        },
    )


def _drop_stale_action_stats(dst_root: Path) -> None:
    """Remove action-stats provenance copied from the source dataset.

    The source tree is copied wholesale, so if it carried chunk-relative action
    stats its `action_relative_meta` marker survives — while this conversion has
    just recomputed `action` for the NEW gripper channels, i.e. as absolute
    stats. A dataset then claims relative stats while holding absolute ones,
    which is exactly the mismatch the training guard exists to catch, and the
    stale marker would make that guard PASS.

    `action_absolute` goes for the same reason: it archives the pre-relative
    stats of a different representation.
    """
    path = dst_root / "meta" / "stats.json"
    if not path.is_file():
        return
    stats = json.loads(path.read_text())
    dropped = [k for k in ("action_relative_meta", "action_absolute") if k in stats]
    if not dropped:
        return
    for k in dropped:
        del stats[k]
    path.write_text(json.dumps(stats, indent=2) + "\n")
    logger.info("Dropped stale action-stats provenance: %s", ", ".join(dropped))


def _write_sidecar(dst_root: Path, gp: GraspProjection, src_root: Path,
                   close_trigger: dict | None = None) -> None:
    """Record which projection built this dataset, and from what.

    Written every conversion, unconditionally: a projected dataset with no record
    of its calibration is the ambiguity this file exists to remove.

    `close_trigger` records when the full close fires, since it changes the
    training targets as much as the calibration does. It is always written —
    null fields mean the original ramp-onset behaviour.
    """
    path = dst_root / PROJECTION_SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"source_root": str(src_root)}
    if close_trigger is not None:
        extra["close_trigger"] = close_trigger
    path.write_text(json.dumps(gp.to_metadata(**extra), indent=2) + "\n")
    logger.info("Wrote %s", path)


def _write_card(dst_root: Path, repo_id: str | None = None) -> None:
    """Write a minimal dataset card if there is none.

    `hf upload` does not render LeRobot's card template, so a hand-uploaded
    dataset otherwise lands with no README, no tags and no viewer link. Never
    overwrites: a hand-written card carries more than this can.

    The viewer badge needs the repo id, which only exists at upload time — hence
    `repo_id`. Without it the card is still written (tags are the valuable part)
    and check_publish reports the missing viewer link.

    CAVEAT: `LeRobotDataset.push_to_hub` REPLACES README.md with LeRobot's own
    template, so this card does not survive that route — the template keeps a
    correct viewer badge (set `ds.repo_id` before pushing) but carries only the
    `LeRobot` tag, and check_publish then reports the missing `grasp-projection`
    tag. After such a push the tags have to be merged back into the uploaded
    card, preserving the template's `configs:` block, which drives the Hub
    dataset viewer. This card survives an `hf upload`, which renders no template
    of its own.
    """
    path = dst_root / "README.md"
    if path.exists():
        logger.info("%s exists; leaving it alone", path)
        return
    tags = "\n".join(f"- {t}" for t in _CARD_TAGS)
    badge = ""
    if repo_id:
        badge = (
            f'\n<a class="flex" href="https://huggingface.co/spaces/lerobot/'
            f'visualize_dataset?path={repo_id}">\n'
            '<img class="block dark:hidden" src="https://huggingface.co/datasets/'
            'huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>\n'
            '<img class="hidden dark:block" src="https://huggingface.co/datasets/'
            'huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>\n'
            "</a>\n"
        )
    path.write_text(
        f"---\ntask_categories:\n- robotics\ntags:\n{tags}\n---\n{badge}\n"
        f"# {dst_root.name}\n\n"
        "Gripper channels are **(strategy, closure)**, not joint angles: the last "
        "two channels of `action` and `observation.state` are normalised to the "
        "gripper's reachable travel. In `action`, `closure = 1.0` means \"drive "
        "fully closed\" and the object stops the fingers; in `observation.state` "
        "closure stays continuous, so \"am I holding something\" remains "
        "observable.\n\n"
        f"The exact calibration is recorded in `{PROJECTION_SIDECAR}`. Decode with "
        "`grabette_dataset_tools.grasp_projection.GraspProjection` — see "
        "[grasp_projection.md](https://github.com/pollen-robotics/grabette-policy/"
        "blob/develop/libs/grabette-dataset-tools/docs/grasp_projection.md).\n"
    )
    logger.info("Wrote %s", path)


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    rest_is_closed: bool = False,
    overwrite: bool = False,
    stats_repo_id: str | None = None,
    close_at: float | None = None,
    stop_speed: float | None = None,
) -> dict:
    """Copy the dataset and rewrite its gripper channels. Returns a report.

    `close_at` / `stop_speed`: see `delay_close_onsets`. Checked before anything
    is copied, so a bad value or a dataset without position channels fails fast.
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if close_at is not None and not 0.0 < close_at <= 1.0:
        raise ValueError(f"close_at must be in (0, 1], got {close_at}")
    if stop_speed is not None and stop_speed <= 0.0:
        raise ValueError(f"stop_speed must be > 0 m/s, got {stop_speed}")
    src_info = json.loads((src_root / "meta" / "info.json").read_text())
    pose_cols = None
    if stop_speed is not None:
        pose_cols = _pose_columns(src_info["features"]["action"].get("names"))
        if pose_cols is None:
            raise ValueError("stop_speed needs position channels in `action` "
                             "(x/y/z or dx/dy/dz); none found by name")
    fps = float(src_info["fps"])
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst_root}. Pass --overwrite.")
        logger.warning("Overwriting %s", dst_root)
        shutil.rmtree(dst_root)

    logger.info("Copying %s -> %s", src_root, dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    # Dereferences symlinks, so the result is a self-contained snapshot and the
    # source (a shared HF cache) cannot be touched by anything downstream.
    shutil.copytree(src_root, dst_root, symlinks=False)

    # Open the dataset BEFORE rewriting anything.
    #
    # LeRobotDataset syncs from the Hub on construction even when `root` is
    # local, and it must be given a repo_id that exists (an invented one 404s,
    # and HF_HUB_OFFLINE makes the check raise rather than skip). Constructing it
    # here means that download lands on the freshly-copied files, which are still
    # identical to the source. Constructing it AFTER the rewrite re-downloads the
    # source parquet over the converted data and silently undoes everything —
    # observed, and only caught by re-running the round-trip check.
    ds = None
    if stats_repo_id:
        from lerobot.datasets import LeRobotDataset

        ds = LeRobotDataset(repo_id=stats_repo_id, root=dst_root)
        logger.info("Opened %s at %s for the stats pass", stats_repo_id, dst_root)

    info_path = dst_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    a_names = info["features"]["action"].get("names")
    a_dim = int(info["features"]["action"]["shape"][0])
    s_names = info["features"]["observation.state"].get("names")
    s_dim = int(info["features"]["observation.state"]["shape"][0])
    a_ip, a_id = _gripper_columns(a_names, a_dim)
    s_ip, s_id = _gripper_columns(s_names, s_dim)
    logger.info("action gripper cols %s, state gripper cols %s", (a_ip, a_id), (s_ip, s_id))

    gp = GraspProjection()
    reports = []
    for pf in sorted((dst_root / "data").rglob("*.parquet")):
        table = pq.read_table(pf)
        actions = np.array(table.column("action").to_pylist(), dtype=np.float32)
        states = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
        eps = np.array(table.column("episode_index").to_pylist())

        for ep in np.unique(eps):
            m = eps == ep
            pos = None
            if pose_cols is not None:
                idx, is_delta = pose_cols
                pos = actions[m][:, idx].astype(np.float64)
                if is_delta:
                    # Per-step deltas: integrate. The speed only uses differences,
                    # so the unknown starting point does not matter.
                    pos = np.cumsum(pos, axis=0)
            a_pair, s_pair, rep = convert_episode(
                actions[m, a_ip].tolist(), actions[m, a_id].tolist(), gp, rest_is_closed,
                close_at=close_at, stop_speed=stop_speed, pos=pos, fps=fps,
            )
            actions[m, a_ip], actions[m, a_id] = a_pair[:, 0], a_pair[:, 1]
            states[m, s_ip], states[m, s_id] = s_pair[:, 0], s_pair[:, 1]
            rep["episode"] = int(ep)
            reports.append(rep)

        cols = {}
        for c in table.column_names:
            if c == "action":
                cols[c] = pa.array(actions.tolist(), type=pa.list_(pa.float32()))
            elif c == "observation.state":
                cols[c] = pa.array(states.tolist(), type=pa.list_(pa.float32()))
            else:
                cols[c] = table.column(c)
        pq.write_table(pa.table(cols), pf)
        logger.info("  rewrote %s (%d rows)", pf.name, len(actions))

    # Rename the two channels so the new meaning is visible in the metadata
    # rather than implied. Shapes are unchanged.
    def _renamed(names, ip, idl):
        out = list(names)
        out[ip], out[idl] = "strategy", "closure"
        return out

    if a_names:
        info["features"]["action"]["names"] = _renamed(a_names, a_ip, a_id)
    if s_names:
        info["features"]["observation.state"]["names"] = _renamed(s_names, s_ip, s_id)
    info_path.write_text(json.dumps(info, indent=4))
    logger.info("Updated %s", info_path)

    if ds is not None:
        # Mandatory: `closure` is now 0..1 where the raw angle was 0..1.6 rad, so
        # stale stats would mis-scale the channel. Uses the object opened above —
        # constructing a new one here is what caused the clobber.
        from lerobot.datasets.dataset_tools import recompute_stats

        logger.info("Recomputing stats...")
        recompute_stats(ds, skip_image_video=True)
        logger.info("Stats recomputed")

    _drop_stale_action_stats(dst_root)
    _write_sidecar(dst_root, gp, src_root, close_trigger={
        "close_at": close_at, "stop_speed_m_s": stop_speed})
    _write_card(dst_root, repo_id=stats_repo_id)

    # WARN, don't refuse. A converted dataset that only ever lives locally for an
    # experiment is legitimate, so publication rules must not block conversion —
    # but the operator should hear about them here, while the context is fresh,
    # rather than from a training run that dies in `get_safe_version` an hour in.
    # Run `checks.publish.check_publish` as the hard gate before uploading.
    from grabette_dataset_tools.checks.publish import check_publish

    audit = check_publish(dst_root)
    for msg in audit["errors"]:
        logger.warning("publish check: %s", msg)
    for msg in audit["warnings"]:
        logger.warning("publish check: %s", msg)
    if audit["errors"] or audit["warnings"]:
        logger.warning(
            "%d publish issue(s) above — fix them before uploading; "
            "the git tag and card checks only run against a pushed repo",
            len(audit["errors"]) + len(audit["warnings"]),
        )

    return {
        "episodes": len(reports),
        "publish_errors": audit["errors"],
        "publish_warnings": audit["warnings"],
        "single_close": sum(1 for r in reports if r["n_close"] == 1),
        "no_close": sum(1 for r in reports if r["n_close"] == 0),
        "multi_close": sum(1 for r in reports if r["n_close"] > 1),
        "with_open": sum(1 for r in reports if r["n_open"] > 0),
        "sign_flipped": sum(1 for r in reports if r["sign_flipped"]),
        "frac_close_median": float(np.median([r["frac_close"] for r in reports])),
        "close_delays_s": [d / fps for r in reports for d in r["close_delays"]],
        "per_episode": reports,
    }


@click.command()
@click.option("--src_root", required=True, type=click.Path(exists=True),
              help="Source dataset root (a LeRobot dataset directory).")
@click.option("--dst_root", required=True, type=click.Path(),
              help="Destination root. The source is never modified.")
@click.option("--rest_is_closed", is_flag=True,
              help="Recording convention where the operator holds the gripper CLOSED "
                   "when idle. Then rest and grasp are the same command and only "
                   "opens are detected. Use for datasets recorded that way.")
@click.option("--close_at", type=click.FloatRange(0.0, 1.0, min_open=True), default=None,
              help="Delay the full close until the human closure has covered this "
                   "fraction of its rise (e.g. 0.8). Default: fire at the ramp onset. "
                   "For operators who close slowly while still approaching.")
@click.option("--close_stop_speed", type=click.FloatRange(0.0, min_open=True), default=None,
              help="Also fire the full close once the hand slows below this speed "
                   "(m/s, e.g. 0.03), whichever comes first with --close_at. Needs "
                   "x/y/z or dx/dy/dz channels in `action`.")
@click.option("--overwrite", is_flag=True, help="Replace the destination if it exists.")
@click.option("--recompute_stats/--no_recompute_stats", default=True,
              help="Recompute normalisation stats. The gripper channels change "
                   "meaning and RANGE, so stale stats mis-normalise training.")
@click.option("--repo_id", default=None,
              help="repo_id to open the converted dataset under, for the stats "
                   "pass only; the data always comes from --dst_root. A LOCAL id "
                   "(local/anything) is the right choice and needs no Hub access: "
                   "LeRobotDataset only queries the Hub when the root is "
                   "incomplete, and by this point it is fully written. Do NOT pass "
                   "the SOURCE dataset's Hub id — opening the converted dataset "
                   "under it has re-downloaded source parquet over the converted "
                   "files, silently reverting the conversion.")
def main(src_root, dst_root, rest_is_closed, close_at, close_stop_speed, overwrite,
         recompute_stats, repo_id):
    """Re-express a dataset's gripper channels as (strategy, commanded closure)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = convert_dataset(Path(src_root), Path(dst_root),
                            rest_is_closed=rest_is_closed, overwrite=overwrite,
                            stats_repo_id=repo_id if recompute_stats else None,
                            close_at=close_at, stop_speed=close_stop_speed)

    n = report["episodes"]
    click.echo("")
    click.echo(f"episodes:            {n}")
    click.echo(f"  exactly one close: {report['single_close']} "
               f"({100 * report['single_close'] / max(n, 1):.1f}%)")
    click.echo(f"  no close found:    {report['no_close']}")
    click.echo(f"  multiple closes:   {report['multi_close']}")
    click.echo(f"  with open events:  {report['with_open']}")
    click.echo(f"  sign-flipped:      {report['sign_flipped']}")
    click.echo(f"  median frames labelled close: "
               f"{100 * report['frac_close_median']:.0f}%")
    if close_at is not None or close_stop_speed is not None:
        d = report["close_delays_s"]
        click.echo(f"  full close delayed: {len(d)} close(s)"
                   + (f", by {np.median(d):.2f} s median (max {max(d):.2f} s)" if d else ""))

    click.echo("")
    click.echo(f"Done: {dst_root}")


if __name__ == "__main__":
    main()
