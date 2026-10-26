"""Back-compat shim: the 6D rotation helpers moved to `gripette_core.rotation`.

Kept so existing `openarm_gripette_simu.rotation` imports keep resolving.
New code should import from `gripette_core` directly.
"""

from gripette_core.rotation import (  # noqa: F401  (re-exported)
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
)

__all__ = ["rotation_matrix_to_6d", "rotation_6d_to_matrix"]
