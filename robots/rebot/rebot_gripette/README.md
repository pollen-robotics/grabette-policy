# rebot-gripette

Real-hardware driver and `ArmService` gRPC server for the **reBot B601-RS +
Gripette**.

Arm only — six joints. The Gripette's Feetech servos are not on the CAN bus
and serve their own `GripperService`, the same one the OpenArm uses, so there
is no gripper code here: point the client at `--gripper_addr` as usual.

## How this relates to the OpenArm

The `ArmService` implementation — Cartesian integrator, joint interpolator,
IK-jump watchdog, target-lead contact guard, reset ramp — is **not** in this
package. It lives in [`gripette_core.arm_service`](../../../libs/gripette-core),
shared with the OpenArm. That is deliberate: a policy must get identical
behaviour from either arm, and two copies of 400 lines of safety logic would
drift.

This package supplies only the reBot-specific half:

| Piece | What |
|---|---|
| `follower.py` | LeRobot follower, with two upstream plugin defects corrected |
| `limits.py` | Per-joint torque clamps, re-applied on every connect |
| `calibration.py` | Calibration-file guard and logical range handling |
| `arm_interface.py` | The three-method `ArmInterface` over motorbridge/CAN |
| `grpc_server_real.py` | `REBOT_SPEC`, the home pose, and the CLI |

## Two things that bite on this arm

**Zero is a hard stop.** `shoulder_lift` is 0…170° and `elbow_flex` 0…200° —
both *start* at zero. The OpenArm homes at `[0,0,0,1.57,0,0,0]`; the
equivalent here would sit exactly on two limits, with Placo's joint-limit
constraint fighting the solver and no room on one side. The home pose is
therefore chosen inside the envelope, not at zero.

**Six DoF has no null space.** The OpenArm is redundant and can trade
orientation against position; this arm cannot — position plus orientation is
exactly determined. The IK branch-flip problems the shared watchdog exists to
catch get *worse* here, not better, so expect a tighter reachable envelope and
possibly different solver weights.

## Torque limits are volatile

The RobStride `0x7000` register range is runtime-only. `store_parameters()`
does not persist it — only `0x2000` survives a power cycle. Limits must be
re-applied on **every** connect, which `limits.py` does; do not "optimise"
that away after seeing the values read back correctly within one session.

## Bring-up order

Never skip a rung, and never run any of it without a human in the loop and a
hand on the power:

1. `read_arm_state` — no motion, confirms the bus and the sign conventions
2. `reset_arm` — the ramp to home
3. `set_arm_pose` — a single absolute pose
4. `cartesian_square` 2 cm
5. `cartesian_square` 10 cm

If the 10 cm square is not clean, stop. The OpenArm README is blunt about why:
a policy run on an arm that cannot trace a square produces garbage silently.
