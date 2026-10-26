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

## Starting pose vs home pose

Two different things, and conflating them is how an arm gets driven into a
stop:

| | What it is |
|---|---|
| **Starting pose** | Where the robot physically *is* at power-on — on this arm, resting on its hard mechanical stops. An observation about the robot. Nothing commands it. |
| **Home pose** | The startup position **for an experiment**. `Reset` ramps the arm to it, so it is the first commanded motion of a run. A property of the experiment, not of the robot. |

The OpenArm's home pose is its simulator's `START_JOINTS` — the pose its
dataset was recorded from and its policy expects to begin at. **The reBot has
no such experiment yet**, so there is nothing to inherit and nothing to guess.
`--home_joints` is therefore **required** and has no default: a pose chosen
merely for being "inside the limits" is still a guess, and `Reset` would drive
the arm into it.

When the reBot sim scene exists (it defines the start joints), set the home
pose from *that* and keep the two equal.

`spec.check_home_pose()` validates whatever you pass — correct joint count,
and every joint ≥5° clear of a hard stop — **before** the CAN bus is opened.
These are *necessary* conditions only: it knows the URDF and nothing else, so
it cannot tell you a pose is right for your experiment, and it knows nothing
about your table, the pedestal, the Gripette's envelope or self-collision.

**Zero is a hard stop**, which is the specific trap when translating from the
OpenArm: `shoulder_lift` is 0…170° and `elbow_flex` 0…200°, so both *start* at
zero and an all-zeros pose sits on two stops. The guard rejects it.

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
