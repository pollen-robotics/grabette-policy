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
| `follower.py` | LeRobot follower: one joint space, pinned and enforced |
| `limits.py` | Per-joint torque clamps, re-applied on every connect |
| `calibration.py` | Calibration-file guard and joint-range handling |
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

### The working home pose

Measured 2026-10-27. Pass it explicitly; it is deliberately **not** a default,
because the reasoning above has not changed — there is still no experiment, and
this pose is validated against the geometry only.

    --home_joints 0 15 25 -10 0 0

It is the pose a human held, cleaned up — nothing more. `capture_pose.py`
recorded `3.2 14.8 26.5 -10.7 5.8 1.3` (median of 82 samples, up to 9.4° of
wobble). The cleanup rounds the angles, zeroes `shoulder_pan`, `wrist_yaw` and
`wrist_roll` as the hand tremor they were, and sets `wrist_flex` so the
Gripette is exactly level. The result sits **9.6 mm** from where the hand was:

| | |
|---|---|
| control frame | x −165, y −3, z +371 mm |
| gripper pitch | 0.00°, level |
| nearest hard stop | 15° (`shoulder_lift`) |

The Gripette is level exactly when `shoulder_lift + elbow_flex + wrist_flex =
30°`, so rounding and levelling are one operation.

**Known limitation, now measured.** Dry runs from this pose on 2026-10-28:

| square | result |
|---|---|
| 2 cm, x plane | clean — 0.00 mm residual, worst joint step 1.3° |
| 10 cm, x plane | clean — worst joint step 6.3° |
| 10 cm, y plane | **fails** — waypoint 20 residual 8.4 mm, waypoint 21 steps 90° |
| 10 cm, z plane | **fails** — waypoint 20 residual 4.5 mm, waypoint 21 steps 54.8° |

Both failures are at the same corner and both are branch flips rather than
solver errors — the signature of 6 DoF with no null space to escape through.

**Rung 4 ran on the arm on 2026-10-28** — the 2 cm square in the x plane, from
this pose, clean. The first attempt traced it as 41 separate legs and was
unreadable by eye: `leg_duration` floors a leg at `MIN_LEG_S = 2.0`, the
segments are 0.5–1.3°, so the arm stopped 41 times and took 80 s to cover 36°
of joint travel. `_bringup.run_path` traces the whole waypoint list as one
motion; at `--deg_per_s 6` the square takes about 7 s and looks like a square.

A search over 4 500 poses found **2 143** that trace 10 cm cleanly in all three
planes, the nearest being `0 35 25 -10 0 0` — the home pose with `shoulder_lift`
raised 20°, control frame 234 mm out instead of 165. Worth knowing, but every
candidate sits at 11–14° worst joint step against the 15° limit, where the x
plane from home manages 6.3°. Passing is not the same as having margin.
The gripper sits 165 mm from the base, close to the shoulder axis, where
lateral motion conditions worst.

This is **not** a reason to move the home pose. The square validates the
kinematics chain — URDF, joint signs, calibration, solver — and the experiment's
start pose does not have to be the pose that validation runs from. Trace the
square from wherever it solves cleanly; keep the home pose where the operator
put it.

### Steady-state droop, measured twice

With gravity compensation off, two joints sit short of their targets at the
home pose, reproducibly:

| joint | commanded | rung 2 | rung 3 |
|---|---|---|---|
| `elbow_flex` | 25° | 22.35 (−2.65) | 22.46 (−2.54) |
| `wrist_flex` | −10° | −12.35 (−2.35) | −12.37 (−2.37) |

That is the figure `ABORT_DEG = 5.0` has to live above: ~2.4° of headroom
before normal droop alone trips the envelope, which is why `--deg_per_s` cannot
rise far above 2 without widening the envelope to match.

**The `shoulder_lift` overshoot does not appear here.** At 75/75 it overshot
~2° and stayed; the rung 3 return leg landed within **0.07°** of its start, and
every other joint within 0.25°. The overshoot is specific to the over-centre
region, where gravity assists the motion, not a property of the joint.

### Starting pose, for the record

Where this arm sits at power-on, read with rung 1 on 2026-10-27:

| joint | deg |
|---|---|
| shoulder_pan | +0.8 … +19.1 (free; wherever it was left) |
| shoulder_lift | −0.3 (on its lower stop) |
| elbow_flex | 0.0 (on its lower stop) |
| wrist_flex | −0.2 |
| wrist_yaw | +1.1 … +3.0 |
| wrist_roll | free; wherever it was left |

`shoulder_lift` and `elbow_flex` are the two that matter: both rest on their
lower stop and read a few hundredths *past* the URDF limit, because that is
where the limit was defined. A solver handed this pose must clamp. Supply
measured 23.2–24.2 V across the six at idle.

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

## `disconnect()` wants to move the arm, and the destructor calls it

The upstream plugin's `disconnect()` calls `safe_zero()`, which interpolates
all six arm joints to **motor zero** — and stage 2 of that sweep is CAN ids 2
and 3, i.e. `shoulder_lift` and `elbow_flex`, the two joints whose zero *is*
their lower hard stop. Stage 2 also reaches for the gripper: *"if it is
currently past 180°, bring it back toward 170°"*. So an unmodified disconnect
lowers the arm onto its stops and moves motor 7.

`ReBotFollower.__init__` sets `_skip_safe_zero_on_disconnect = True` at
**construction**, and that placement is the whole point.

The obvious place for the guard is the shutdown path, and that is where it was
first put: before each `disconnect()` in the server. It is not enough. The
dangerous case is a connect that **fails** — then the object loses its last
reference, Python garbage-collects it, `Robot.__del__` calls `disconnect()`,
and no call site ever gets a say. That is not hypothetical: on 2026-10-27
`detect_gripper_zero()` raised during rung 2's connect, the destructor ran the
sweep, and it left the rack gripper wide open. The arm was still folded on its
stops with nowhere to be driven; the same failure after a ramp to the home pose
would have brought it down onto them.

Parking the arm is fine when a human asks for it. Happening because a connect
threw is not.

`tests/test_safe_zero_guard.py` pins the upstream premise and the suppression,
and fails if a `disconnect()` ever appears without one.

## Putting the arm down

`disconnect()` disables every motor, so ending a run with the arm up is a drop.
It dropped the arm from 77° on 2026-10-27. Every teardown now ends with
`ReBotFollower.controlled_descent()`: `shoulder_lift` and `elbow_flex` are
lowered to 0° — their lower hard stops, where the arm rests at power-on — under
position control at 10°/s, while the other four joints are held where they are
so the gripper goes down rather than sweeping across the workspace.

**It is not a damped release, and that was measured, not assumed.** The first
attempt used MIT with `kp=0` and a per-joint `kd`, on the theory that damping
torque would turn a drop into a sink. Run from the 15/25 home on 2026-10-28 it
was indistinguishable from cutting torque. The arithmetic says it always would
be: a damped joint sinks at the speed where `kd·ω` balances gravity, so 10°/s
against the ~7 N·m this arm carries at the shoulder needs `kd ≈ 40` — four
times the largest value the vendor plugin ever sends, and that value alone only
buys ~40°/s. Damping cannot hold this arm up. Stiffness can, and position
control already applies it.

Rung 2.6, `verify_descent.py`, scores a descent against its commanded ramp: did
the lowered joints reach the floor, did the held joints hold, did anything run
faster than the commanded rate. Run from the 15/25 home on 2026-10-28: the arm
was put down as commanded. That is the first teardown on this arm that carried
its weight all the way to the stops rather than letting go of it.

"The floor" is approximate, and harmlessly so. `shoulder_lift` ends on its stop,
but `elbow_flex` was read at **+3.8°** at the start of the next run — the
descent commands 0 and the same droop that leaves it 2.5° short at the home
pose leaves it a few degrees high here, and it stays there when the motors are
disabled rather than settling further. A few degrees of fold is not energy
worth chasing; what matters is that the arm is at rest before the disable, and
it has been on every run since.

## Motor 7 is read-only

The reBot's rack gripper is still physically mounted and still on the CAN bus,
but the Gripette replaces it and serves its own `GripperService` over Feetech.
Nothing in this package commands motor 7, and that is enforced in one place —
`ReBotFollower.send_action` strips every `gripper.*` key before delegating, and
`hold_action()` leaves it out.

Enforced centrally because the property used to rest on six separate call sites
each happening to build an arm-only action, which is not a property at all once
one of them forgets.

`detect_gripper_zero()` is overridden to do nothing for the same reason. The
upstream version drives the gripper closed with torque, waits for a stall,
releases, and then raises unless the motor is near a home position that is
meaningless for this arm — and `connect()` calls it, so a gripper nobody asked
about can abort the whole bring-up. It measured −235.29° and did exactly that.

## Bring-up order

Never skip a rung, and never run any of it without a human in the loop and a
hand on the power:

| # | Script | Moves the arm? |
|---|---|---|
| 0 | `mirror_arm.py` | **No.** Torque off throughout; you move the arm by hand |
| 1 | `read_arm_state.py` | **No.** Motors stay disabled; position comes from parameter reads |
| 1.5 | `check_connect.py` | Energizes, holds where the arm already is. No trajectory |
| 2 | `reset_arm.py --home_joints …` | Yes — the first commanded motion |
| 2.6 | `verify_descent.py --home_joints …` | Yes — checks how the arm gets put down |
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

### One joint space

Motor degrees are the URDF's angles. Nothing in this package mirrors, scales
or converts a joint angle anywhere, and that is enforced rather than assumed.

The upstream plugin ships a `joint_directions` table — `−1.0` for
`elbow_flex`, `wrist_flex` and `wrist_yaw`, `6.0` for `gripper`. It is not
geometry: `6.0` is a gear ratio. It maps a **leader arm's** joint convention
and scale onto this follower's motors, for Seeed's own teleop pairing. Worse,
the plugin applies it on one side only:

| | |
|---|---|
| `send_action` | `position * joint_directions.get(name, 0.0)` |
| `get_observation` | `math.degrees(state.pos)` — raw motor |

so feeding an observation back as an action — teleop recording, dataset
replay, policy rollout — commands the mirror of the current pose.

**This stack has no leader arm** (a policy drives it over gRPC) **and does not
use the reBot's rack gripper** (the Gripette replaces it, on its own Feetech
bus behind its own service). The mapping buys nothing here, so
`ReBotFollowerConfig` pins every entry to `1.0` and the second space stops
existing.

The dict is pinned rather than emptied because `send_action` reads it with
`.get(name, 0.0)`: a **missing** joint is multiplied by zero and commanded to
its zero, which on this arm is a hard stop for `shoulder_lift` and
`elbow_flex`. An empty table would be the most dangerous possible value.
`connect()` refuses to open the bus unless every joint is present and exactly
`1.0`, so a plugin upgrade restoring the vendor default fails loudly instead
of silently mirroring three joints.

**What this replaced.** `ReBotArmInterface` used to convert between the two
spaces on every read and write. One early version missed the conversion, and
at a representative pose that put the control frame **745 mm** from where it
really was — a URDF-validated home pose would have been commanded mirrored,
`elbow_flex` `+60°` arriving as `−60°`, past its lower stop at 0. It hid
because reads *and* writes were in the same wrong space: holds held, replays
replayed, every round-trip test passed. Only the model disagreed, and nothing
compared the model to the arm. `tests/test_arm_interface_space.py` now does.

### Confirming it against the hardware

Source can establish which space the code uses; it cannot establish that the
model matches the robot. Rung 0 does. `mirror_arm.py` streams the disabled
motors' parameter reads into a live MuJoCo viewer — no torque, no command, no
`enable()` anywhere in the file — so you move the arm by hand and watch the
model follow. A sign error is then not a number to interpret but a joint
visibly going the wrong way, across the whole range rather than at one pose.

`--space vendor` drives the model through the vendor mapping on purpose:
`elbow_flex`, `wrist_flex` and `wrist_yaw` should run backwards, which is what
makes a clean `--space motor` run mean something.

Rung 1 also prints the control frame, but treat it as a tape-measure sanity
check only. At the power-on pose the three vendor-mapped joints all rest
within a degree of zero, so a mirror would move the control frame about 4 mm —
below the noise floor of any measurement you can make by hand.
