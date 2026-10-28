"""Re-export shim: the Gripette camera moved to `gripette_sim.camera`.

It is a property of the GRIPPER, not of the OpenArm - same calibration, same
KannalaBrandt8 coefficients, on any arm wearing it. Kept importable from here
because call sites and docs spell it this way.
"""

from gripette_sim.camera import *  # noqa: F401,F403
from gripette_sim.camera import FisheyeCamera  # noqa: F401
