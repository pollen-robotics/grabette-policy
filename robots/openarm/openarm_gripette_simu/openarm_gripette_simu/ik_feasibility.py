"""Back-compat shim: IK feasibility moved to `gripette_core.ik_feasibility`.

It was already robot-agnostic — it only ever needed a Placo-style kinematics
object — so the move was a relocation, not a rewrite. Kept so existing
`openarm_gripette_simu.ik_feasibility` imports keep resolving.
"""

from gripette_core.ik_feasibility import (  # noqa: F401  (re-exported)
    IKFeasibilityChecker,
    PoseFeasibility,
    RejectionSamplingStats,
    TrajectoryFeasibility,
)

__all__ = [
    "IKFeasibilityChecker",
    "PoseFeasibility",
    "TrajectoryFeasibility",
    "RejectionSamplingStats",
]
