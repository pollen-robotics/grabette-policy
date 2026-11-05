"""What the reBot must not drive into, and which of its links can get there.

Lives in the HARDWARE package, not the simulator: this is a statement about the
robot's own links, and the gate it feeds is on the shared arm service, so the
real server needs it as much as the sim does.

The OpenArm guards a box proxy around its mounting column. The reBot's hazard
is the bench it is bolted to: a large horizontal surface it can fold down onto
and reach into. Same query (`gripette_sim.clearance`), different geometry.

WHICH LINKS ARE GUARDED, AND WHY NOT ALL OF THEM
    Measured by sweeping shoulder_pan x shoulder_lift x elbow_flex x wrist_flex
    over their full travel and recording each body's closest approach to the
    tabletop geom:

        实体1 (gripper base)   -69.2 mm      axis_6            -48.9 mm
        axis_5               -62.5 mm      gripette distal   -44.8 mm
        axis_4               -61.6 mm      gripette proximal -44.4 mm
        axis_1 / 2 / 3         0.0 mm      (permanent contact - see below)

    Negative is penetration. The six outward links genuinely reach into the
    table, by up to 7 cm, and are guarded.

    axis_1, axis_2 and axis_3 read exactly 0.0 because the arm is BOLTED to
    this surface - the base sits on it by construction, and the inner links
    rest near it when the arm is folded on its lower stops, which is where it
    sits at power-on. Guarding them would make the gate fire in the arm's
    resting pose and never clear. The OpenArm excludes its two base links for
    the same kind of reason.

LIMITS
    The legs are not guarded: nothing can reach them that has not already hit
    the top. Nothing here knows about an object in the gripper.
"""

# Geoms the arm must stay clear of. Two scenes define a work surface and the
# gate is given whichever is loaded: "tabletop" in the simulator's furnished
# scene, "bench" in the bare slab the real server measures against. Naming
# both means one declaration serves sim and hardware.
OBSTACLE_GEOMS = ("tabletop", "bench")

# Links that can actually reach them. Names are MuJoCo body names from the
# export; 实体1 is the gripper mount plate, named by the Onshape source.
GUARDED_BODIES = (
    "axis_4",
    "axis_5",
    "axis_6",
    "实体1",
    "gripette_r_proximal__configuration_default",
    "gripette_r_distal__configuration_default",
)

# Clearance to keep from the table, in metres. Matches the OpenArm's column
# margin; there is no measurement behind either number, only that 2 cm is
# comfortably more than the solver's own tolerance and visibly a gap.
TABLE_MARGIN = 0.02
