# reBot Gripette Simulation

MuJoCo simulation of the reBot B601-RS + Gripette, serving the **same two gRPC
services as the real arm** — because they are the same code, not a port of it.

```bash
# the bare table scene, any valid pose
uv run python -m rebot_gripette_simu --home_joints 0 15 25 -10 0 0 --headless

# the grasp scene and the start pose a policy actually picks from
uv run python -m rebot_gripette_simu \
  --home_joints -25.2 127.0 65.7 27.5 -61.8 -30.5 \
  --scene robots/rebot/rebot_gripette_simu/scenes/table_grasp.xml \
  --no_clearance_gate
```

Driving `SteveNguyen/test_pick_can_200_smooth_trim-best` at the second of
those: **11/20 episodes lift the can**, average displacement 38.8 mm. The full
two-terminal recipe, and the three ways to get it wrong, are in
[`../README.md`](../README.md).

| Service | Port | Supplied by |
|---|---|---|
| GripperService | 50051 | `gripette_sim.gripper_service` — shared with the OpenArm |
| ArmService | 50052 | `gripette_core.arm_service` — shared with the OpenArm **and with the real reBot** |

## What is reBot-specific here

Almost nothing, and that is the design:

| file | what it is |
|---|---|
| `simulation.py` | two class attributes: the arm's joint names and its default scene |
| `sim_arm_interface.py` | `ArmInterface` over MuJoCo instead of over CAN — the only genuinely new code |
| `server.py` | wiring, plus the camera render loop |
| `scenes/table.xml` | a bare bench placed for **this** arm, which reaches in −x |
| `scenes/table_grasp.xml` | the same bench plus the can — the grasp scene every measurement here uses |

`rebot_gripette/grpc_server_real.py` hands `ArmServicer` a CAN-backed
interface; this hands it a MuJoCo-backed one. The Cartesian integrator, the
setpoint interpolator, the IK-jump watchdog and the reset ramp are then
literally the same object in sim and on hardware — so a policy that passes here
and fails there fails for reasons about the robot, not because the two ran
different control code.

## `--home_joints` is required, here too

Same reasoning as the real server, and the same guard (`spec.check_home_pose`):
the *starting pose* is where the robot is at power-on, the *home pose* is the
startup position for an experiment, and the reBot has no experiment yet. A pose
picked for being inside the limits is still a guess, and `Reset` drives the arm
into it. In sim a bad guess costs nothing — which is exactly why it must not
become the value someone later copies onto the hardware.

## The arm is bolted to the bench, and reaches in −x

Two things the first version of this scene got wrong.

**Bolted, not standing on the floor.** The raw Onshape export gave the root
body a `<freejoint>`, so the whole robot was a free body resting on the ground
by contact — visibly wobbly, and worse, its base could translate. Placo's FK,
the Cartesian integrator and every pose in a recorded dataset are expressed
from a base at the world origin, so a drifting base makes all of them quietly
wrong. `postprocess_rebot.py` now strips it (the script had already listed
"the base weld" as not done yet), and both the model tests and this package's
tests fail if a re-export puts it back. The OpenArm export has no freejoint
anywhere, which is what a bolted arm should look like.

**The table is built downward from z=0**, rather than the arm being raised
onto a surface — same reason: the base stays at the origin, so nothing
downstream is offset by the table height.

| | |
|---|---|
| tabletop | z −0.030 … **0.000** (the arm bolts to its top face) |
| legs | z −0.730 … −0.030 |
| floor | z −0.730 |

At the home pose `0 15 25 -10 0 0` the control frame (`oak_l`) sits at x −165,
y −3, z +371 mm and `gripper_center` at x −224, z **+318 mm above the table
surface** — a working height, with the cube under it. The work area is at
negative x, so this is **not** a mirror of the OpenArm's `table_red_cube.xml`.

The 318 mm is measured from the model.

The bench **extent** is not a guess any more, but it is also not the real
bench: it was narrowed to **0.85 × 0.36 m**, ending 150 mm past the can, so
the camera sees a patch of table comparable to the one the policy trained on.
THE BENCH EXTENT IS PART OF THE OBSERVATION — training used a 300 × 300 mm
table floating in a void, and this is the closest match that still supports
the arm. Measure the real bench before trusting any of it on hardware.

## Tracking: reconciled

The simulated arm used to droop **7.4°** at the elbow against the real arm's
2.5°, because the export shipped `kp=50` on every position actuator. That is
not merely soft, it breaks the control loop: the arm settled 55 mm from the
pose it was commanded to hold, and `arm_service` then read a 50 mm target lead
as contact, so the guard fired on physics alone. (The cap is now 80 mm, the
OpenArm's hardware-tuned value; 55 mm of sag would leave it almost no margin.) `postprocess_rebot.py` now
writes `kp=500` (120 on the gripper, as on the OpenArm), measured at 5 mm
static error with no overshoot. See that script's ACTUATOR GAINS note for the
sweep and for why tracking, not fidelity, is the target here.

## Scoring an episode

`gripette_core.arm_service.ArmServicer` is the class the REAL arm serves, so
its `GetSuccessStatus` correctly reports that a real robot cannot see the
object. This simulator serves that same class and therefore inherited the
blind `False` — every episode scored 0/1 whatever happened, and a run whose
gripper closed on empty air was once recorded as a successful lift.

`server.py` now passes an `object_tracker`: `gripette_sim.LiftTracker`,
injected the same way the bench guard is injected as a `joint_gate`, so core
still imports no mujoco. `Reset` repositions the can, zeroes its velocity and
records its height; `GetSuccessStatus` reports the rise against that.

* Success = the can up **≥ 50 mm**, `LIFT_SUCCESS_THRESHOLD` — equal to
  `grabette_trajectory`'s, so this arm's numbers mean what the OpenArm's mean.
  A horizontal nudge is not a pick.
* `Reset` REPOSITIONS rather than only recording, so an episode that knocks
  the can off the bench does not silently score every later episode 0.
* The reading is instantaneous and `evaluate.py` latches it, so a can lifted
  and then dropped still counts. That is the OpenArm's behaviour too.

A scene with no `red_cube`, or one where it is bolted down, disables the
tracker rather than raising.

## The clearance gate

Two different failures, two different guards:

| guard | catches | where |
|---|---|---|
| IK-jump watchdog | the solver flipping branch — a 90° step in one tick | `gripette_core.arm_service`, 15° |
| clearance gate | a perfectly smooth solution that goes **into the bench** | `gripette_sim.clearance`, 2 cm |

The watchdog would not have caught the second: driving into the table is a
sequence of small, well-conditioned IK solutions.

`ArmServicer` takes a `joint_gate` — a plain callable from joint angles to a
reason-or-`None`, vetoing a solution before it is commanded, with the same
integrator rollback a jump gets. It is a callable rather than geometry because
`gripette-core` must not import mujoco; `gripette_sim.clearance` supplies the
MuJoCo-backed implementation, and the hook is on the **shared** service, so the
real arm can be given one too.

`obstacles.py` declares what is guarded. Measured by sweeping the joint
envelope and recording each link's closest approach to the tabletop:

| link | closest approach |
|---|---|
| `实体1` (gripper mount) | **−69.2 mm** |
| `axis_5` / `axis_4` | −62.5 / −61.6 mm |
| `axis_6` | −48.9 mm |
| gripette distal / proximal | −44.8 / −44.4 mm |
| `axis_1` / `2` / `3` | 0.0 mm — permanent contact |

The last row is why the base links are **not** guarded: the arm is bolted to
this surface, so they touch it by construction and at the folded resting pose.
Guarding them would fire the gate in the pose the arm powers on in and never
clear it.

**Retreating is always allowed.** Without that, an arm that ends up inside the
margin is deadlocked — every command that could free it is also a command
inside the margin, so the gate refuses them all and the only way out is killing
the server, which is the state this exists to avoid.

### The gate cannot express a PICK — `--no_clearance_gate`

One margin applies to every guarded body, and a grasp needs two things at
once:

> the **fingertips** are supposed to reach the bench; the **arm links** are not.

The start pose is clear — no guarded geom within 20 mm — but descending ~8° to
close on the can puts `gripette_r_distal` 5.8 mm under the bench top. That is
the task, not a fault, and no single margin admits it:

| margin | outcome, driving the pick policy |
|---|---|
| 20 mm (default) | 236 of 250 commands refused, 0.0 mm displacement |
| 0 mm | 236 of 250 refused — the fingertips still go under |
| −15 mm | 238 of 250 refused; the policy then asks for −17 to −21 mm |
| gate off | 19 of 250 refused, episode runs, **11/20 lift the can** |

The number was never the problem. So the simulator takes
`--no_clearance_gate`, and **the real server deliberately gets no such flag**:
on hardware, over-blocking is the safe way to be wrong.

The real fix is a margin **per guarded body** — fingers near 0, links at
20 mm — in `obstacles.py` and `clearance.py`. Until then the sim runs
unguarded for pick tasks, and you will see the gripper touch the bench.

### Do not raise the query's `distmax`

`clearance()` passes the margin itself as `mj_geomDistance`'s `distmax`, so
anything further away comes back clamped and the solver is never asked to
measure it. Keep it that way: MuJoCo 3.9's convex-distance solver returns a
hard 0.0 for some mesh/box pairs when asked for a LONG separation —
reproducible with nothing in the model but one mesh and one box 375 mm apart,
where it reports 0.0 while filling a 455 mm witness segment, and unaffected by
`ccd_iterations`, `ccd_tolerance` or the collision mask. A `min` over many
pairs means one such reading pins the whole result to 0.0.

Inside the margin the query is sound: over 3000 random configurations checked
against a closed-form hull-vertex-to-box distance, **0 false passes and 1
false block**, agreeing to a median of 0.02 mm.

### Still not guarded

The table legs (nothing reaches them without hitting the top first), anything
held in the gripper, and general self-collision. The gate checks the named
obstacles against the named links, nothing else.
