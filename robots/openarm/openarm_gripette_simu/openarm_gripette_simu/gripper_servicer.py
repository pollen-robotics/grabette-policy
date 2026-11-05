"""Re-export shim: the gripper service moved to `gripette_sim.gripper_service`.

Same gripper on every arm, so one implementation. Kept importable under the
old module name because call sites spell it this way.
"""

from gripette_sim.gripper_service import *  # noqa: F401,F403
from gripette_sim.gripper_service import GripperServicer  # noqa: F401
