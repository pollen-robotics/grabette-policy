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

## `disconnect()` wants to move the arm

The upstream plugin's `disconnect()` calls `safe_zero()`, which interpolates
all six arm joints to **motor zero** — and stage 2 of that sweep is CAN ids 2
and 3, i.e. `shoulder_lift` and `elbow_flex`, the two joints whose zero *is*
their lower hard stop. So an unmodified disconnect lowers the arm onto its
stops.

Parking the arm is fine when a human asks for it. Happening because someone hit
Ctrl-C is not — an interrupt is the one moment a motion must never start. The
server calls `ReBotFollower.skip_safe_zero_on_disconnect()` before **every**
`disconnect()`, so the arm holds position and then goes limp.

Two things worth knowing:

- This was already the behaviour, *by accident*: torque is disabled first,
  disabled RobStride motors emit no feedback, `safe_zero`'s position read
  raised, and `disconnect()` swallowed the exception. A plugin update that
  merely fixed the exception path would have silently restored the motion.
- The accident never covered the single-instance guard's cleanup path. There,
  `connect()` has already run on a bus a *first* server is driving with torque
  live, so feedback works and `safe_zero` would really haul the arm to zero
  while the other process commands it.

`tests/test_safe_zero_guard.py` pins both the upstream premise and our
suppression, and fails if a `disconnect()` ever appears without one.

## Bring-up order

Never skip a rung, and never run any of it without a human in the loop and a
hand on the power:

| # | Script | Moves the arm? |
|---|---|---|
| 1 | `read_arm_state.py` | **No.** Motors stay disabled; position comes from parameter reads |
| 2 | `reset_arm.py --home_joints …` | Yes — the first commanded motion |
| 3 | `set_arm_pose.py --joints …` | Yes — any absolute pose |
| 4 | `cartesian_square.py --size_cm 2` | Yes — first rung with IK in the loop |
| 5 | `cartesian_square.py --size_cm 10` | Yes |

`cartesian_square.py --dry_run --from_joints …` solves the whole path offline
and reports the IK residual and largest joint step per waypoint. Run it before
the hardware rung: if the dry run cannot close the square, the arm certainly
cannot, and nothing had to move to find out.

If the 10 cm square is not clean, stop. The OpenArm README is blunt about why:
a policy run on an arm that cannot trace a square produces garbage silently.

Shared discipline, all in `examples/_bringup.py`, every constant traced back to
the `rebot-lerobot` experiments that measured it:

- **Preflight reads disabled motors.** A RobStride sends no feedback frames
  until enabled, so `get_state()` is `None` beforehand. A joint that does not
  answer is *absent* from the read, never `0.0` — zero is a hard stop on two of
  these joints, so a defaulted read is a command into one. Refuse to energize
  an arm you cannot fully read.
- **The first commanded pose is the measured pose.** Scripts 11/12 ramped `kp`
  from 0 over 1.5 s so the arm never got a step change. Through the plugin the
  gains are not ours to ramp, so the equivalent is to begin the trajectory
  exactly where the arm already is.
- **All six joints are held,** not just the one moving — unpowered links swing
  off their stops unpredictably.
- **Abort envelope** on every leg: position deviation (5°), torque (80 % of the
  per-joint clamp) and MOSFET temperature (60 °C). An abort leaves the arm
  holding, not dropped.
- **`max_relative_target = 5.0`,** not 2.0: steady-state droop under gravity is
  already 1–2°, so a 2° cap clamps continuously during normal motion.
- **Gravity compensation defaults off.** The exported inertias still put each
  link's COM at its motor, so the predicted torque has the right magnitude at
  the calibration pose and the wrong angle dependence everywhere else — fine
  for position control, not as a feedforward term.

### Rung 1 also confirms the sign convention

Three spaces, two of which get confused:

| | What it is |
|---|---|
| **motor** | What the motor reports. The upstream plugin's `get_observation` is `math.degrees(state.pos)` — raw, untouched by calibration. |
| **logical** | motor ÷ `joint_directions`, i.e. the *leader* arm's convention. Mirrored against motor on `elbow_flex`, `wrist_flex`, `wrist_yaw` (direction −1.0). `ReBotFollower.get_observation` converts into it so observation → `send_action` round-trips, which teleop, replay and rollout all need. |
| **URDF** | **== motor.** `22_mirror.py` drove the exported model from raw `mechPos` with every sign at `+1` and confirmed joint-by-joint agreement. |

So the follower's logical output has to be converted **back** before it reaches
the model. `ReBotArmInterface` does that now, in both directions. An earlier
version passed logical angles through unconverted; at a representative pose
that put the control frame **745 mm** from where it really was, sent
`SendCartesianDelta` the wrong way, and would have commanded a URDF-validated
home pose mirrored — `elbow_flex` `+60°` arriving as `−60°`, past its lower
stop at 0.

It hid because reads *and* writes were both logical: holds held, replays
replayed, and every round-trip test passed. Only the model disagreed, and
nothing compared the model to the arm. `tests/test_arm_interface_space.py` now
does, and asserts the two spaces still differ enough at the test pose to be
able to tell.

Rung 1 prints the control-frame position under both readings, so the fix is
confirmed against a ruler rather than only against source.
