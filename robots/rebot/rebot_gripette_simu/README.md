# reBot Gripette Simulation

MuJoCo simulation of the reBot B601-RS + Gripette, serving the **same two gRPC
services as the real arm** — because they are the same code, not a port of it.

```bash
uv run python -m rebot_gripette_simu --home_joints 0 15 25 -10 0 0
uv run python -m rebot_gripette_simu --home_joints 0 15 25 -10 0 0 --headless
```

Run and confirmed working on 2026-10-28: viewer, both services, and the
fisheye camera stream.

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
| `scenes/table.xml` | a table placed for **this** arm, which reaches in −x |

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

The 318 mm is measured from the model. The table's **extent** (0.80 × 0.70 m,
arm 10 cm in from one edge) is a guess — measure the real bench and change it.

## Known gap: the sim is softer than the arm

With torque on at the home pose the simulated `elbow_flex` droops **7.4°**
where the real arm droops **2.5°** — the MuJoCo actuator gains are softer than
the follower's `mit_kp`. Nothing depends on it yet, but a policy trained here
would see tracking the hardware does not reproduce, so this is the first thing
to reconcile before any dataset is recorded in sim.

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

### Still not guarded

The table legs (nothing reaches them without hitting the top first), anything
held in the gripper, and general self-collision. The gate checks the named
obstacles against the named links, nothing else.
