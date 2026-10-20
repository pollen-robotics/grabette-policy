# Third-Party Notices

GRABETTE POLICY is © Pollen Robotics and released under the Apache License 2.0
(see [`LICENSE`](LICENSE)). It incorporates, depends on, or derives from the
third-party components below, each under its own license. This file is provided
for attribution and license compliance.

---

### OpenArm & openarm_can — Apache-2.0 (Enactic, Inc.)
<https://github.com/enactic/openarm> · <https://github.com/enactic/openarm_can>

- The robot description in `robots/openarm/openarm_gripette_model` is
  derived from the OpenArm arm model, adapted to carry the Gripette end-effector.
- `openarm_can` is a dependency of `robots/openarm/openarm_gripette`.
- `robots/openarm/openarm_gripette/examples/calibrate_arm_no_gripper.py` is
  adapted from OpenArm's zero-position calibration procedure.

### LeRobot — Apache-2.0 (Hugging Face, Inc.)
<https://github.com/huggingface/lerobot>

- Used as a dependency for LeRobot-format dataset generation and policy training.
- `robots/openarm/openarm_gripette/openarm_gripette/openarm7_follower.py`
  and `config_openarm7_follower.py` subclass LeRobot's `OpenArmFollower` and
  retain the original Hugging Face copyright headers.

### onshape-to-robot — MIT (Rhoban)
<https://github.com/Rhoban/onshape-to-robot>

Used to generate the OpenArm-Gripette URDF / MuJoCo model from Onshape CAD.

### 6-D rotation representation — Zhou et al., CVPR 2019
*"On the Continuity of Rotation Representations in Neural Networks."*

`policies/DiffusionPolicy/rotation.py` is a Pollen Robotics implementation of
this representation (Apache-2.0), vendored to avoid depending on a fork.

