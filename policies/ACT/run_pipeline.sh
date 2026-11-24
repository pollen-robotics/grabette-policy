#!/usr/bin/env bash
#
# One-shot DATA-PREP pipeline for chunk-relative ACT.
#
#   raw dataset (8D, with is_lost)  ──▶  clean_dataset.py          (reject unrecoverable-lost episodes,
#                                                                    keep the training camera only)
#                                   ──▶  normalize_raw             (add observation.state, camera -> cam0,
#                                                                    strip the arm prefix; actions stay 8D absolute)
#                                   ──▶  grasp_projection_convert  (--grasp-projection, optional)
#                                   ──▶  resize_dataset_videos.py  (648x486: the Gripette's stream size)
#                                   ──▶  check_dataset_videos.py   (decode-check every episode)
#                                   ──▶  analyze_dataset.py        (QA, optional)
#                                   ──▶  write_relative_action_stats  (chunk-relative stats; LAST, because
#                                        the projection recomputes the stats it would overwrite)
#
# Unlike the Diffusion pipeline there is no delta conversion (ACT trains on
# absolute poses made chunk-relative by train.py). ACT does not resize images
# internally, so the training images are resized to what the Gripette streams
# at inference (648x486 by default, same camera and FOV as the Grabette).
# evaluate.py also rescales live frames to the checkpoint's size.
#
# The rotation-glitch cap of the stats follows the dataset fps, so the fps in
# meta/info.json must be the real one (datasets built before grabette-slam's
# 30 fps fix declare 50 while being ~30).
#
# Training is NOT run here; the exact train.py command is printed and saved.
#
# Usage:
#   ./run_pipeline.sh <raw_repo_id> [options]
#
# Options:
#   --raw-root DIR        local root of the raw dataset (omit if it's on the Hub)
#   --work DIR            scratch dir for intermediate datasets
#                         (default: ~/.cache/grabette_pipeline/<name>_act — real disk, not /tmp)
#   --camera NAME         camera stream to train on (default: cam0; e.g. right_cam0)
#   --strip-prefix P      channel-name prefix to drop (e.g. right_)
#   --chunk-size K        ACT chunk_size the stats are computed for (default: 50)
#   --max-lost-run N      clean: reject if longest lost run > N frames (default: script's 10)
#   --image-size WxH      training image size (default: 648x486, the Gripette's stream size)
#   --no-resize           keep the dataset's image size
#   --grasp-projection    re-express the gripper channels as (strategy, closure)
#   --rest-is-closed      --grasp-projection: operator holds the gripper CLOSED when idle
#   --close-at F          --grasp-projection: fire the full close once the human closure
#                         has covered fraction F of its ramp (e.g. 0.8) instead of at the
#                         ramp onset — for operators who close slowly while approaching
#   --close-stop-speed V  --grasp-projection: or once the hand slows below V m/s (e.g.
#                         0.03), whichever comes first
#   --no-qa              skip the analyze_dataset.py QA step
#   -h, --help            show this help
#
# Example:
#   ./run_pipeline.sh simheo/red-cube-blue-box --camera right_cam0 --strip-prefix right_

set -euo pipefail

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

RAW="" RAW_ROOT="" WORK="" CAMERA="cam0" PREFIX="" CHUNK=50 MAX_LOST_RUN=""
DO_PROJECTION=0 REST_IS_CLOSED=0 DO_QA=1 IMAGE_SIZE="648x486" DO_RESIZE=1
CLOSE_AT="" CLOSE_STOP_SPEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-root)         RAW_ROOT="$2"; shift 2 ;;
    --work)             WORK="$2"; shift 2 ;;
    --camera)           CAMERA="$2"; shift 2 ;;
    --strip-prefix)     PREFIX="$2"; shift 2 ;;
    --chunk-size)       CHUNK="$2"; shift 2 ;;
    --max-lost-run)     MAX_LOST_RUN="$2"; shift 2 ;;
    --image-size)       IMAGE_SIZE="$2"; shift 2 ;;
    --no-resize)        DO_RESIZE=0; shift ;;
    --grasp-projection) DO_PROJECTION=1; shift ;;
    --rest-is-closed)   REST_IS_CLOSED=1; shift ;;
    --close-at)         CLOSE_AT="$2"; shift 2 ;;
    --close-stop-speed) CLOSE_STOP_SPEED="$2"; shift 2 ;;
    --no-qa)           DO_QA=0; shift ;;
    -h|--help)          usage; exit 0 ;;
    -*)                 echo "Unknown option: $1" >&2; exit 1 ;;
    *)                  if [[ -z "$RAW" ]]; then RAW="$1"; shift; else echo "Unexpected arg: $1" >&2; exit 1; fi ;;
  esac
done

[[ -n "$RAW" ]] || { echo "ERROR: raw dataset repo_id is required." >&2; echo; usage; exit 1; }
[[ "$IMAGE_SIZE" =~ ^([0-9]+)x([0-9]+)$ ]] || { echo "ERROR: --image-size must be WxH, got '$IMAGE_SIZE'." >&2; exit 1; }
IMG_W="${BASH_REMATCH[1]}" IMG_H="${BASH_REMATCH[2]}"

# Run from this script's directory (the ACT uv project); the other steps run
# in the projects that own them.
cd "$(dirname "$0")"
DIFFUSION=../DiffusionPolicy
TOOLS=../../libs/grabette-dataset-tools

BASE="${RAW##*/}"
WORK="${WORK:-${XDG_CACHE_HOME:-$HOME/.cache}/grabette_pipeline/${BASE}_act}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"   # absolute: the steps run from different directories
WORK_FSTYPE=$(df -PT "$WORK" 2>/dev/null | awk 'NR==2{print $2}' || true)
if [[ "$WORK_FSTYPE" == "tmpfs" || "$WORK_FSTYPE" == "ramfs" ]]; then
  echo "WARNING: work dir '$WORK' is on $WORK_FSTYPE (RAM-backed). Pass --work <dir-on-disk>." >&2
fi
CLEAN_ID="local/${BASE}_clean";  CLEAN_ROOT="$WORK/clean"
TRAIN_ID="local/${BASE}_act";    TRAIN_ROOT="$WORK/act"

# Expanded as ${A[@]+"${A[@]}"}: bash 3.2 (macOS) treats an empty array as unset under set -u.
RAW_ROOT_ARG=(); [[ -n "$RAW_ROOT" ]]     && RAW_ROOT_ARG=(--root "$RAW_ROOT")
CLEAN_EXTRA=();  [[ -n "$MAX_LOST_RUN" ]] && CLEAN_EXTRA=(--max_lost_run "$MAX_LOST_RUN")
PREFIX_ARG=();   [[ -n "$PREFIX" ]]       && PREFIX_ARG=(--strip_prefix "$PREFIX")

echo "════════════════════════════════════════════════════════════════"
echo "  ACT (chunk-relative) — data prep"
echo "    raw dataset : $RAW ${RAW_ROOT:+(root $RAW_ROOT)}"
echo "    work dir    : $WORK"
echo "    camera      : $CAMERA   prefix: ${PREFIX:-none}   chunk_size: $CHUNK"
echo "    image size  : $([[ "$DO_RESIZE" == 1 ]] && echo "$IMAGE_SIZE" || echo "unchanged")"
echo "════════════════════════════════════════════════════════════════"

echo; echo "==> [1] clean — reject episodes with unrecoverable SLAM loss, keep $CAMERA only"
uv run --project "$DIFFUSION" python "$DIFFUSION/clean_dataset.py" \
  --repo_id "$RAW" ${RAW_ROOT_ARG[@]+"${RAW_ROOT_ARG[@]}"} \
  --output_repo_id "$CLEAN_ID" --output_root "$CLEAN_ROOT" --overwrite_output \
  --keep_cameras "$CAMERA" ${CLEAN_EXTRA[@]+"${CLEAN_EXTRA[@]}"}

echo; echo "==> [2] normalize — observation.state, $CAMERA -> cam0, unprefixed names"
uv run --project "$TOOLS" python -m grabette_dataset_tools.normalize_raw \
  --src_root "$CLEAN_ROOT" --dst_root "$TRAIN_ROOT" --repo_id "$TRAIN_ID" \
  --camera "$CAMERA" ${PREFIX_ARG[@]+"${PREFIX_ARG[@]}"} --overwrite

if [[ "$DO_PROJECTION" == 1 ]]; then
  PROJ_ID="local/${BASE}_act_graspproj"; PROJ_ROOT="$WORK/act_graspproj"
  PROJ_EXTRA=(); [[ "$REST_IS_CLOSED" == 1 ]] && PROJ_EXTRA=(--rest_is_closed)
  [[ -n "$CLOSE_AT" ]] && PROJ_EXTRA+=(--close_at "$CLOSE_AT")
  [[ -n "$CLOSE_STOP_SPEED" ]] && PROJ_EXTRA+=(--close_stop_speed "$CLOSE_STOP_SPEED")
  echo; echo "==> [3] grasp projection — gripper angles -> (strategy, closure)"
  uv run --project "$TOOLS" python -m grabette_dataset_tools.grasp_projection_convert \
    --src_root "$TRAIN_ROOT" --dst_root "$PROJ_ROOT" --overwrite \
    --repo_id "$PROJ_ID" ${PROJ_EXTRA[@]+"${PROJ_EXTRA[@]}"}
  TRAIN_ID="$PROJ_ID"; TRAIN_ROOT="$PROJ_ROOT"
fi

if [[ "$DO_RESIZE" == 1 ]]; then
  RESIZE_ID="${TRAIN_ID}_${IMG_W}x${IMG_H}"; RESIZE_ROOT="${TRAIN_ROOT}_${IMG_W}x${IMG_H}"
  echo; echo "==> [3b] resize — ${IMG_W}x${IMG_H} training copy (ACT sees images at this size)"
  # resize_dataset_videos.py refuses an existing output; re-runs regenerate it.
  rm -rf "$RESIZE_ROOT"
  uv run --project "$DIFFUSION" python "$DIFFUSION/resize_dataset_videos.py" \
    --repo_id "$TRAIN_ID" --root "$TRAIN_ROOT" --output_root "$RESIZE_ROOT" \
    --width "$IMG_W" --height "$IMG_H"
  TRAIN_ID="$RESIZE_ID"; TRAIN_ROOT="$RESIZE_ROOT"
fi

# The steps above re-encode video; an encoder glitch only surfaces hours into
# training. Decode-check every episode now, through the path training uses.
echo; echo "==> [4] video integrity — decode-check every episode"
if ! uv run --project "$DIFFUSION" python "$DIFFUSION/check_dataset_videos.py" \
    --repo_id "$TRAIN_ID" --dataset_root "$TRAIN_ROOT"; then
  echo "ERROR: corrupt video segment(s) — see above. Re-run the pipeline." >&2
  exit 1
fi

if [[ "$DO_QA" == 1 ]]; then
  echo; echo "==> [5] analyze — QA on the prepared dataset"
  uv run --project "$DIFFUSION" python "$DIFFUSION/analyze_dataset.py" \
    --repo_id "$TRAIN_ID" --root "$TRAIN_ROOT"
fi

echo; echo "==> [6] chunk-relative action stats (chunk_size=$CHUNK)"
uv run --project "$TOOLS" python -c "
from pathlib import Path
from grabette_dataset_tools.chunk_relative_stats import write_relative_action_stats as w
print(w(Path('$TRAIN_ROOT'), chunk_size=$CHUNK))"

FPS=$(python3 -c "import json; print(json.load(open('$TRAIN_ROOT/meta/info.json'))['fps'])")

echo; echo "════════════════════════════════════════════════════════════════"
echo "  Data prep complete. Training dataset:"
echo "    repo_id : $TRAIN_ID"
echo "    root    : $TRAIN_ROOT"
echo "    fps     : $FPS (evaluate with --fps $FPS)"
echo "    images  : $(python3 -c "import json; s=json.load(open('$TRAIN_ROOT/meta/info.json'))['features']['observation.images.cam0']['shape']; print(f'{s[2]}x{s[1]}')")"
if [[ "$DO_PROJECTION" == 1 ]]; then
  echo "    gripper : (strategy, closure) — projected (evaluate.py --grasp_projection auto detects it)"
else
  echo "    gripper : RAW JOINT ANGLES (radians) — not projected"
fi
TRAIN_CMD_FILE="$WORK/train_command.txt"
cat > "$TRAIN_CMD_FILE" <<EOF
# Generated by run_pipeline.sh — train on the PREPARED dataset (not the raw repo id!)
uv run python train.py --policy.type=act \\
    --policy.chunk_size=$CHUNK --policy.n_action_steps=$CHUNK \\
    --policy.device=cuda --policy.push_to_hub=false \\
    --dataset.repo_id=$TRAIN_ID --dataset.root=$TRAIN_ROOT \\
    --batch_size=16 --num_workers=12 --steps=80000 --save_freq=10000 --log_freq=50 \\
    --output_dir=\$PWD/outputs/${BASE}_act --job_name=${BASE}_act
# evaluate with: evaluate.py --checkpoint <ckpt>/pretrained_model --chunk_relative auto --fps $FPS
EOF
echo
echo "  Train with (also saved to $TRAIN_CMD_FILE):"
sed 's/^/    /' "$TRAIN_CMD_FILE"
echo "════════════════════════════════════════════════════════════════"
