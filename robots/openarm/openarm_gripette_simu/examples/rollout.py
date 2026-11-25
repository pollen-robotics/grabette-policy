"""Roll out a trained policy on a Gripette arm (sim or real), in the style of
`lerobot-rollout`.

A fixed-rate control loop observes, asks an inference engine for the next
action and sends it, so the arm always runs on the dataset's clock. Two
engines, mirroring lerobot.rollout.inference:

  sync  select_action on every tick. When the policy's queue runs dry the
        loop blocks for one inference (one chunk every n_action_steps).
  rtc   Real-Time Chunking. A background thread keeps generating chunks with
        predict_action_chunk while the loop drains a shared ActionQueue. Each
        chunk is guided to agree with the actions still queued, then replaces
        the queue minus the actions executed during its own inference.
        Flow-matching policies only (pi0, pi05).

Observation building, gripper decoding and the safety helpers are
evaluate.py's. Supports per-step Cartesian-delta policies (11D actions) with
gripper-only or relative proprioception; chunk-relative and joint-space
checkpoints are refused (use evaluate.py for the former).

Usage:
  uv run python examples/rollout.py --checkpoint <hub-id-or-dir> \\
      --inference rtc --fps 30 --task "put the red cube in the green box"
"""

import argparse
import atexit
import json
import logging
import math
import threading
import time

import grpc
import numpy as np
import torch
from grabette_dataset_tools.grasp_projection import GraspProjection, clamp_to_command_limits
from lerobot.policies import make_pre_post_processors
from lerobot.policies.rtc import ActionQueue, LatencyTracker, RTCConfig

from evaluate import (
    CameraStream,
    _load_policy_any,
    act_image_size,
    apply_grip_gain,
    build_observation,
    capture_start_pose,
    clamp_delta,
    detect_grasp_projection,
    lift_then_reset,
    match_image_size,
)

logger = logging.getLogger(__name__)

# [dp(3), dr6d(6), gripper(2)]
_CARTESIAN_ACTION_DIM = 11


class SyncEngine:
    def __init__(self, policy, preprocessor, postprocessor):
        self.policy, self.pre, self.post = policy, preprocessor, postprocessor
        self._batch = None

    def start(self):
        self.policy.reset()

    def stop(self):
        pass

    def notify_observation(self, batch):
        self._batch = batch

    def get_action(self):
        with torch.no_grad():
            action = self.policy.select_action(self.pre(self._batch))
        return self.post(action).squeeze(0).cpu().numpy()

    def summary(self):
        return ""


class RTCEngine:
    def __init__(self, policy, preprocessor, postprocessor, rtc_config, fps, queue_threshold):
        if not hasattr(policy, "init_rtc_processor"):
            raise SystemExit(f"--inference rtc needs a flow-matching policy (pi0, pi05), "
                             f"got {policy.config.type}.")
        policy.config.rtc_config = rtc_config
        policy.init_rtc_processor()
        self.policy, self.pre, self.post = policy, preprocessor, postprocessor
        self.rtc_config = rtc_config
        self.dt = 1.0 / fps
        self.queue_threshold = queue_threshold
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        self.policy.reset()
        self.queue = ActionQueue(self.rtc_config)
        self.latency = LatencyTracker()
        self.dropped = []
        self.error = None
        self._batch = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rtc-inference")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def notify_observation(self, batch):
        with self._lock:
            self._batch = batch

    def get_action(self):
        if self.error is not None:
            raise RuntimeError("RTC inference thread failed") from self.error
        action = self.queue.get()
        return None if action is None else action.numpy()

    def summary(self):
        if not len(self.latency):
            return " | rtc: no chunk produced"
        return (f" | rtc: {len(self.dropped)} chunks, infer p95 {self.latency.p95() * 1000:.0f}ms"
                f" max {self.latency.max() * 1000:.0f}ms, dropped/chunk median "
                f"{int(np.median(self.dropped))}")

    def _loop(self):
        try:
            while not self._stop.is_set():
                with self._lock:
                    batch = self._batch
                if batch is None or self.queue.qsize() > self.queue_threshold:
                    time.sleep(0.005)
                    continue
                t0 = time.perf_counter()
                idx_before = self.queue.get_action_index()
                latency = self.latency.max()
                with torch.no_grad():
                    chunk = self.policy.predict_action_chunk(
                        self.pre(batch),
                        inference_delay=math.ceil(latency / self.dt) if latency else 0,
                        prev_chunk_left_over=self.queue.get_left_over(),
                    )
                processed = self.post(chunk).squeeze(0).cpu()
                self.latency.add(time.perf_counter() - t0)
                # Drop what the arm actually executed while this chunk was computed
                # (fewer than the elapsed ticks if the queue ran dry meanwhile).
                consumed = self.queue.get_action_index() - idx_before
                self.queue.merge(chunk.squeeze(0), processed, consumed, idx_before)
                self.dropped.append(consumed)
        except Exception as e:  # noqa: BLE001 - surfaced by get_action on the control thread
            logger.exception("RTC inference thread failed")
            self.error = e


def send_action(a, arm_stub, arm_pb2, gripper_stub, gripper_pb2, projection, args, stats):
    """Send one 11D Cartesian-delta action. Returns the server's error once the arm froze."""
    dp, dr6, (g1, g2) = a[:3] * args.translation_gain, a[3:9], a[9:11]
    if args.clamp_pos_mm is not None or args.clamp_rot_deg is not None:
        dp, dr6, was = clamp_delta(
            dp, dr6,
            args.clamp_pos_mm / 1000.0 if args.clamp_pos_mm else None,
            np.deg2rad(args.clamp_rot_deg) if args.clamp_rot_deg else None)
        stats["clamped"] += int(was)
    resp = arm_stub.SendCartesianDelta(arm_pb2.CartesianDelta(
        dx=float(dp[0]), dy=float(dp[1]), dz=float(dp[2]), dr6d=dr6.tolist()))
    if not resp.success:
        stats["rejected"] += 1
        if "frozen" in resp.error:
            return resp.error
        if stats["rejected"] <= 5 or stats["rejected"] % 25 == 0:
            print(f"ARM REJECTED delta #{stats['rejected']}: {resp.error}", flush=True)
    g1, g2 = float(g1), float(g2)
    if projection is not None:
        g1, g2 = projection.decode(g1, g2)
        g1, g2, _ = clamp_to_command_limits(g1, g2)
    g1, g2 = apply_grip_gain(g1, g2, args.grip_gain, tuple(args.start_gripper))
    # Fire-and-forget: a blocking round trip to the Pi would eat the tick budget.
    gripper_stub.SendMotorCommand.future(gripper_pb2.MotorCommand(
        motor1_goal=g1, motor2_goal=g2,
        motor1_torque_limit=args.grip_torque_limit,
        motor2_torque_limit=args.grip_torque_limit))
    return None


def run_episode(engine, arm_stub, arm_pb2, gripper_stub, gripper_pb2, camera, projection,
                use_relative_proprio, image_hw, device, args):
    start_pos = start_rot = None
    if use_relative_proprio:
        start_pos, start_rot = capture_start_pose(arm_stub, arm_pb2)
    stats = {"underruns": 0, "rejected": 0, "clamped": 0}
    dt = 1.0 / args.fps
    success, step = False, 0
    t_start = next_t = time.monotonic()
    engine.start()
    try:
        for step in range(1, args.max_steps + 1):
            img, state, _ts = build_observation(arm_stub, arm_pb2, camera, use_relative_proprio,
                                                start_pos, start_rot, grasp_projection=projection)
            image = torch.from_numpy(match_image_size(img, image_hw)).float().div_(255.0)
            engine.notify_observation({
                "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(device),
                "observation.images.cam0": image.permute(2, 0, 1).contiguous().unsqueeze(0).to(device),
                "task": args.task,
            })
            action = engine.get_action()
            if action is None:
                stats["underruns"] += 1
            elif (frozen := send_action(action, arm_stub, arm_pb2, gripper_stub, gripper_pb2,
                                        projection, args, stats)) is not None:
                print(f"\nARM MOTION FROZEN at step {step}: {frozen}\nAborting episode; "
                      "re-home the arm before the next one.", flush=True)
                break
            if step % args.success_check_freq == 0 and arm_stub.GetSuccessStatus(
                    arm_pb2.SuccessStatusRequest()).goal_reached:
                success = True
                break
            next_t += dt
            if (sleep_for := next_t - time.monotonic()) > 0:
                time.sleep(sleep_for)
            elif sleep_for < -0.2:  # long stall (blocking inference, GC): resync, don't burst
                next_t = time.monotonic()
    finally:
        engine.stop()
    status = arm_stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
    return {
        "success": success or status.goal_reached,
        "steps": step,
        "displacement_mm": status.cube_displacement * 1000,
        "duration_s": time.monotonic() - t_start,
        **stats,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Policy checkpoint: local dir or Hub repo id")
    p.add_argument("--arm_addr", default="localhost:50052")
    p.add_argument("--gripper_addr", default="localhost:50051")
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default="grasp and lift cube",
                   help="Language instruction for VLA policies, verbatim as in training")
    p.add_argument("--fps", type=float, required=True,
                   help="Control rate. Must equal the dataset fps: one action is one dataset tick")
    p.add_argument("--inference", choices=["sync", "rtc"], default="sync")
    p.add_argument("--n_action_steps", type=int, default=None,
                   help="sync: actions executed per chunk before re-inferring (default: checkpoint's)")
    p.add_argument("--rtc_execution_horizon", type=int, default=10,
                   help="rtc: steps over which a new chunk is guided towards the queued actions")
    p.add_argument("--rtc_max_guidance_weight", type=float, default=10.0)
    p.add_argument("--rtc_queue_threshold", type=int, default=None,
                   help="rtc: start the next inference once at most this many actions are queued. "
                        "Default: the chunk size, i.e. re-infer back to back (most reactive)")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--success_check_freq", type=int, default=10)
    p.add_argument("--grasp_projection", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--start_gripper", type=float, nargs=2, default=[0.0, 0.0], metavar=("PROX", "DIST"))
    p.add_argument("--grip_gain", type=float, default=1.0)
    p.add_argument("--grip_torque_limit", type=float, default=0.0)
    p.add_argument("--translation_gain", type=float, default=1.0)
    p.add_argument("--clamp_pos_mm", type=float, default=None)
    p.add_argument("--clamp_rot_deg", type=float, default=None)
    p.add_argument("--home_joints", type=float, nargs="+", default=None, metavar="RAD")
    p.add_argument("--home_lift", type=float, default=0.05, metavar="M",
                   help="Lift the gripper straight up by this much (m) before every Reset to home. 0 = no lift")
    p.add_argument("--no_reset", action="store_true", help="Start each episode from the current arm pose")
    p.add_argument("--return_home", action="store_true", help="Reset the arm home on exit")
    p.add_argument("--ask_success", default=None, metavar="RESULTS_JSONL",
                   help="Real arm: ask the operator for success after each episode and append to this file")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    policy = _load_policy_any(args.checkpoint)
    if args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    action_dim = policy.config.action_feature.shape[0]
    state_dim = policy.config.robot_state_feature.shape[0]
    if action_dim != _CARTESIAN_ACTION_DIM or state_dim not in (2, 11):
        raise SystemExit(
            f"rollout.py drives per-step Cartesian-delta policies (11D actions, 2D or 11D state); "
            f"this checkpoint has action_dim={action_dim}, state_dim={state_dim}. Chunk-relative "
            "(8D) checkpoints run with evaluate.py; the arm API has no joint-command RPC.")

    projection = {"on": True, "off": False}.get(args.grasp_projection)
    if projection is None:
        projection = detect_grasp_projection(args.checkpoint)
        if projection is None:
            raise SystemExit("Could not tell whether the gripper actions are grasp-projected; "
                             "pass --grasp_projection on|off.")
    projection = GraspProjection() if projection else None

    if args.inference == "rtc":
        engine = RTCEngine(
            policy, preprocessor, postprocessor,
            RTCConfig(execution_horizon=args.rtc_execution_horizon,
                      max_guidance_weight=args.rtc_max_guidance_weight),
            args.fps,
            args.rtc_queue_threshold if args.rtc_queue_threshold is not None else policy.config.chunk_size)
    else:
        engine = SyncEngine(policy, preprocessor, postprocessor)
    logger.info(f"Policy {policy.config.type}: chunk {policy.config.chunk_size}, "
                f"inference={args.inference}, gripper "
                f"{'(strategy, closure) decoded' if projection else 'raw angles'}, "
                f"state {'relative 11D' if state_dim == 11 else 'gripper 2D'}, {args.fps:g} Hz")

    from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc, gripper_pb2, gripper_pb2_grpc

    arm_channel = grpc.insecure_channel(args.arm_addr)
    gripper_channel = grpc.insecure_channel(args.gripper_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(arm_channel)
    gripper_stub = gripper_pb2_grpc.GripperServiceStub(gripper_channel)
    arm_stub.Ping(arm_pb2.ArmPingRequest())
    gripper_stub.Ping(gripper_pb2.PingRequest())
    camera = CameraStream(gripper_stub, gripper_pb2)
    camera.get()

    def reset_home():
        return lift_then_reset(arm_stub, arm_pb2, args.home_joints, args.home_lift, args.fps)

    open_gripper = gripper_pb2.MotorCommand(
        motor1_goal=args.start_gripper[0], motor2_goal=args.start_gripper[1],
        motor1_torque_limit=args.grip_torque_limit, motor2_torque_limit=args.grip_torque_limit)

    # The servo holds its last goal: never leave the gripper clamped on exit.
    def teardown():
        if getattr(teardown, "done", False):
            return
        teardown.done = True
        try:
            gripper_stub.SendMotorCommand(open_gripper)
            if args.return_home:
                reset_home()
        except Exception as e:  # noqa: BLE001 - best effort on the way out
            print(f"WARNING: teardown failed ({e}); reopen the gripper by hand.", flush=True)
        for close in (camera.stop, arm_channel.close, gripper_channel.close):
            try:
                close()
            except Exception:  # noqa: BLE001 - already going away
                pass

    atexit.register(teardown)

    results = []
    for ep in range(args.num_episodes):
        if not args.no_reset and not (resp := reset_home()).success:
            logger.error(f"Reset failed: {resp.error}")
            continue
        gripper_stub.SendMotorCommand(open_gripper)
        time.sleep(0.5)
        result = run_episode(engine, arm_stub, arm_pb2, gripper_stub, gripper_pb2, camera, projection,
                             state_dim == 11, act_image_size(policy), device, args)
        if args.ask_success:
            result["success"] = input(f"  Episode {ep + 1}: success? [y/N] ").strip().lower() in ("y", "yes")
            with open(args.ask_success, "a") as f:
                f.write(json.dumps({"episode": ep, "checkpoint": args.checkpoint,
                                    "inference": args.inference, **result}) + "\n")
        results.append(result)
        logger.info(
            f"Episode {ep + 1}/{args.num_episodes}: {'SUCCESS' if result['success'] else 'FAIL'} | "
            f"steps {result['steps']} | {result['duration_s']:.1f}s | underruns {result['underruns']} | "
            f"rejected {result['rejected']} | clamped {result['clamped']}{engine.summary()}")

    if results:
        n_success = sum(r["success"] for r in results)
        print(f"\nSuccess {n_success}/{len(results)} ({100 * n_success / len(results):.0f}%) | "
              f"inference={args.inference} | mean steps {np.mean([r['steps'] for r in results]):.0f}")
    teardown()


if __name__ == "__main__":
    main()
