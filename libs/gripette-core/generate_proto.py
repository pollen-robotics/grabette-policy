"""Regenerate gRPC Python files from proto/gripper.proto for gripette_core.

Run from libs/gripette-core/ — the paths below are relative to it.

Mirrors packages/gripette/generate_proto.py in the grabette repo. The gripper.proto
here MUST stay field-for-field identical to grabette's
packages/gripette/proto/gripper.proto (CI's proto-sync job checks it) — the eval
client (built from these stubs) talks to BOTH the sim server and the real
device over the wire, so field numbers must match.

After generation, fixes the grpc import to a relative import (grpc_tools emits
an absolute import that doesn't resolve inside the package).
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("gripette_core/proto")


def main():
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        "--proto_path=proto",
        f"--python_out={OUTPUT_DIR}",
        f"--grpc_python_out={OUTPUT_DIR}",
        f"--pyi_out={OUTPUT_DIR}",
        "gripper.proto",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    grpc_file = OUTPUT_DIR / "gripper_pb2_grpc.py"
    text = grpc_file.read_text()
    text = text.replace(
        "import gripper_pb2 as gripper__pb2",
        "from . import gripper_pb2 as gripper__pb2",
    )
    grpc_file.write_text(text)
    print(f"Proto files generated and fixed in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
