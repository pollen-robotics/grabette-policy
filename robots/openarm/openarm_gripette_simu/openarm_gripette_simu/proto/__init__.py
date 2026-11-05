"""Back-compat shim: the generated stubs moved to `gripette_core.proto`.

Re-exported rather than duplicated, deliberately: this makes
`openarm_gripette_simu.proto.arm_pb2` and `gripette_core.proto.arm_pb2` the
SAME module object, so `isinstance` holds across both import paths and there
is one descriptor registration instead of two.

Note the descriptor-pool hazard is not the tripwire it looks like: protobuf
keys its pool by .proto filename, but deduplicates byte-identical serialized
descriptors, so two identical generated copies coexist quietly and only raise
`duplicate file name` once they diverge. Re-exporting sidesteps it entirely.
Divergence against the gripper's own copy is guarded separately by the
`proto-sync` CI job, which diffs against the pinned grabette commit.
"""

import sys

from gripette_core.proto import (  # noqa: F401  (re-exported)
    arm_pb2,
    arm_pb2_grpc,
    gripper_pb2,
    gripper_pb2_grpc,
)

__all__ = ["arm_pb2", "arm_pb2_grpc", "gripper_pb2", "gripper_pb2_grpc"]

# Also make the dotted form (`import openarm_gripette_simu.proto.arm_pb2`)
# resolve to the same module, not a second import of the generated file.
for _name in __all__:
    sys.modules[f"{__name__}.{_name}"] = globals()[_name]
del _name
