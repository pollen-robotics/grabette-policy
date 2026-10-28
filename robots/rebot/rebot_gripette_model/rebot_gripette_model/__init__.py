"""reBot Gripette model — paths to URDF, MuJoCo XML, and mesh assets.

Mirrors `openarm_gripette_model` so the two arms are interchangeable to
anything that takes a model directory (notably `gripette_core.ArmSpec`).
"""

from pathlib import Path

# Root of the model data
MODEL_DIR = Path(__file__).resolve().parent

# reBot B601-RS + Gripette. Named for the variant, not just "rebot", so a
# second arm revision can sit alongside it without renaming this one.
REBOT_B601RS_DIR = MODEL_DIR / "rebot_b601rs"
REBOT_B601RS_URDF = REBOT_B601RS_DIR / "robot.urdf"
REBOT_B601RS_XML = REBOT_B601RS_DIR / "robot.xml"
REBOT_B601RS_SCENE = REBOT_B601RS_DIR / "scene.xml"
# The arm plus a bare bench slab at z=0 and nothing else. Geometry for the
# clearance gate to measure against, including on a machine driving the real
# arm where the simulator's scene is not wanted. See the file's own comment.
REBOT_B601RS_BENCH = REBOT_B601RS_DIR / "bench.xml"
REBOT_B601RS_ASSETS = REBOT_B601RS_DIR / "assets"
