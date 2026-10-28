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

## Known gap: no collision gate

The OpenArm's `arm_servicer` rejects any configuration bringing a guarded link
within 2 cm of its mounting column. That gate lives in the OpenArm sim package,
not in the shared service, and the reBot has no pedestal geometry modelled, so
**the reBot currently has IK-jump protection but no clearance gate**. The jump
watchdog catches a branch flip; nothing yet catches the arm reaching into its
own table.
