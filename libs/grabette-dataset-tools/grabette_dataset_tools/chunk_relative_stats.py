"""Action stats for the chunk-relative representation.

WHY THIS EXISTS
---------------
The pi05 pipeline order is `raw -> relative -> NORMALIZE -> model`, so the
normaliser sees the RELATIVE actions while `meta/stats.json` describes the
ABSOLUTE poses the dataset stores. Measured on test_pick_mustard_200, feeding
relative actions through absolute-pose quantiles puts the rotation channels
*outside* [-1, 1] and compresses them into ~10% of the dynamic range:

    channel   normalised q01..q99 using ABSOLUTE stats
    x/y/z     ~[0.04, 0.93]        off-centre, 42% of the range
    ax        [-1.364, -1.162]     outside the range, 10% of it
    az        [+1.062, +1.283]     outside the range, 11% of it

That is the documented failure mode where the model is fed values it never saw
in training. lerobot solves the same problem for its built-in step with
`lerobot-edit-dataset --operation.relative_action true`; this is the equivalent
for ours.

pi05 normalises ACTION with QUANTILES, which reads `q01`/`q99`. lerobot's own
`compute_stats` does emit them (with q10/q50/q90), but over ABSOLUTE actions,
so they are computed here, alongside min/max/mean/std, over relative ones.

The stats depend on the CHUNK LENGTH: a 50-frame offset is larger than a
15-frame one, so `--chunk-size` must match the policy's `chunk_size`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grabette_chunkrel.chunk_relative import to_chunk_relative

# Quantiles pi05's QUANTILES mode reads, plus the moments other modes use.
_QUANTILES = {"q01": 0.01, "q99": 0.99}


def _chunk_indices(n_frames: int, chunk_size: int, stride: int) -> list[np.ndarray]:
    """Frame indices of each chunk TRAINING draws from an episode.

    Mirrors the LeRobotDataset sampler: every frame is a start, and indices past
    the episode end are clamped to its last frame
    (`max(ep_start, min(ep_end - 1, idx + delta))` in lerobot's dataset_reader).
    pi05 ignores `action_is_pad`, so those padded tails are trained on and belong
    in the stats, as do episodes shorter than `chunk_size`.

    The previous enumeration (non-overlapping, unpadded chunks) never saw the
    padded tail and dropped short episodes entirely; on the real pick3 /
    sugar_cup datasets it shifted q01/q99 by up to ~8% of their span.
    """
    return [np.minimum(np.arange(t, t + chunk_size), n_frames - 1)
            for t in range(0, n_frames, stride)]


def relative_action_stats(
    root: Path, chunk_size: int, stride: int | None = None, rot: str = "rotvec"
) -> tuple[dict, dict]:
    """Stats over the chunk-relative actions a dataset would produce.

    Args:
        root: dataset root (must contain meta/ and data/).
        chunk_size: MUST match the policy's `chunk_size`; the stats scale with it.
        stride: chunk start spacing. Defaults to 1, every start, which is what
            training samples; a larger value subsamples the same distribution.
        rot: rotation encoding, as passed to the processor step.

    Returns:
        (stats dict for the `action` key, report dict).

    Holds all relative actions in memory: chunk_size x 8 float64 per frame is
    ~130 MB for a 40k-frame dataset at chunk 50; pass `stride` if that matters.
    """
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    stride = stride or 1

    parquets = sorted((root / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no data parquet under {root / 'data'}")
    df = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "action"]) for p in parquets],
        ignore_index=True,
    )

    rel, n_jumps, n_rot_jumps, n_chunks = [], 0, 0, 0
    for _epi, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(np.float64)
        for idx in _chunk_indices(len(a), chunk_size, stride):
            out, rep = to_chunk_relative(a[idx], rot=rot)
            rel.append(out)
            n_jumps += rep["n_jumps_spliced"]
            n_rot_jumps += rep["n_rot_jumps_spliced"]
            n_chunks += 1
    if not rel:
        raise ValueError(f"no frames under {root / 'data'}")
    arr = np.concatenate(rel).astype(np.float64)

    stats = {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }
    for key, q in _QUANTILES.items():
        stats[key] = np.quantile(arr, q, axis=0).tolist()

    return stats, {
        "chunks": n_chunks,
        "frames": len(arr),
        "chunk_size": chunk_size,
        "stride": stride,
        "n_jumps_spliced": n_jumps,
        "n_rot_jumps_spliced": n_rot_jumps,
        "declared_fps": info.get("fps"),
    }


def write_relative_action_stats(
    root: Path, chunk_size: int, stride: int | None = None, rot: str = "rotvec"
) -> dict:
    """Replace the `action` entry of meta/stats.json with relative-action stats.

    Every other key is left untouched: observation stats are unaffected by the
    action representation. The original absolute stats are kept under
    `action_absolute` so the change is reversible and auditable — a dataset whose
    action stats silently no longer match its stored actions is exactly the kind
    of thing that is impossible to diagnose later.
    """
    root = Path(root)
    path = root / "meta" / "stats.json"
    all_stats = json.loads(path.read_text())

    stats, report = relative_action_stats(root, chunk_size, stride, rot)
    if "action_absolute" not in all_stats and "action" in all_stats:
        all_stats["action_absolute"] = all_stats["action"]
    all_stats["action"] = stats
    all_stats["action_relative_meta"] = report
    path.write_text(json.dumps(all_stats, indent=2) + "\n")
    return report
