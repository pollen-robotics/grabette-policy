"""Shared test setup.

Simulation() creates an offscreen MuJoCo renderer eagerly, so tests that build
a scene need a headless GL backend. EGL works without a display; an explicit
MUJOCO_GL from the environment still wins. Must run before mujoco is imported.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
