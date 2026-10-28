# reBot Gripette Simulation

MuJoCo simulation of the reBot B601-RS + Gripette, serving the **same two gRPC
services as the real arm** — because they are the same code, not a port of it.

```bash
uv run python -m rebot_gripette_simu --home_joints 0 15 25 -10 0 0
uv run python -m rebot_gripette_simu --home_joints 0 15 25 -10 0 0 --headless
```

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

## The arm reaches in −x

Unlike the OpenArm. At the home pose `0 15 25 -10 0 0` the control frame
(`oak_l`) sits at x −165, y −3, z +371 mm, so the table is at negative x and
the scene is **not** a mirror of `table_red_cube.xml`. See `scenes/table.xml`
for the measured numbers it was placed from.

## Known gap: no collision gate

The OpenArm's `arm_servicer` rejects any configuration bringing a guarded link
within 2 cm of its mounting column. That gate lives in the OpenArm sim package,
not in the shared service, and the reBot has no pedestal geometry modelled, so
**the reBot currently has IK-jump protection but no clearance gate**. The jump
watchdog catches a branch flip; nothing yet catches the arm reaching into its
own table.
