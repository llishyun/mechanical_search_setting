"""GR00T-N1.7 policy module for Franka Panda in IsaacLab mechanical search.

Embodiment : OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT (single-arm, relative EEF)
Video keys : exterior_image_1_left (overhead), wrist_image_left (wrist or fallback)
State keys : eef_9d [9], gripper_position [1], joint_position [7]
Action keys: eef_9d, gripper_position, joint_position  (horizon=40)
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np

# Isaac-GR00T must be on the path
sys.path.insert(0, "/home/leesu37/AP-project/Isaac-GR00T")

MODEL_PATH = "/home/leesu37/AP-project/GR00T-N1.7-3B"

# delta_indices: [-15, 0] → need 16-frame ring buffer
_VIDEO_BUF = 16
# action horizon from modality config
ACTION_HORIZON = 40


class FrankaGR00TPolicy:
    """GR00T inference wrapper for a single Franka Panda arm.

    Call reset() at the start of each episode, then get_action() every step.

    Args:
        device: torch device string, e.g. "cuda:0" or "cpu".

    Input per step
    --------------
    exterior_rgb : np.ndarray [H, W, 3] uint8   — overhead camera
    eef_pos      : np.ndarray [3]        float32 — EEF world position (x, y, z)
    eef_rotmat   : np.ndarray [3, 3]     float32 — EEF rotation matrix (world frame)
    gripper_pos  : float                         — gripper width [0=closed, 1=open]
    joint_pos    : np.ndarray [7]        float32 — Franka joint angles (rad)
    wrist_rgb    : np.ndarray [H, W, 3] uint8    — wrist camera (optional; falls back to exterior)

    Output per step
    ---------------
    dict with keys (all np.float32):
        "eef_9d"           : [ACTION_HORIZON, 9]   relative EEF delta (pos3 + 6D-rot)
        "gripper_position" : [ACTION_HORIZON, 1]   gripper target
        "joint_position"   : [ACTION_HORIZON, 7]   joint position targets
    """

    def __init__(self, device: str = "cuda:0"):
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy import Gr00tPolicy

        self._policy = Gr00tPolicy(
            model_path=MODEL_PATH,
            embodiment_tag=EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
            device=device,
            strict=False,
        )
        self._instruction: str = ""
        self._ext_buf: deque[np.ndarray] = deque(maxlen=_VIDEO_BUF)
        self._wri_buf: deque[np.ndarray] = deque(maxlen=_VIDEO_BUF)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, instruction: str = "pick and place objects") -> None:
        """Reset frame buffers and policy state for a new episode."""
        self._instruction = instruction
        self._ext_buf.clear()
        self._wri_buf.clear()
        self._policy.reset()

    def get_action(
        self,
        exterior_rgb: np.ndarray,
        eef_pos: np.ndarray,
        eef_rotmat: np.ndarray,
        gripper_pos: float,
        joint_pos: np.ndarray,
        wrist_rgb: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run one GR00T inference step and return actions.

        Returns action dict indexed by action key, each array shape
        [ACTION_HORIZON, D] float32.
        """
        if wrist_rgb is None:
            wrist_rgb = exterior_rgb

        # --- update ring buffers ---
        self._ext_buf.append(exterior_rgb.copy())
        self._wri_buf.append(wrist_rgb.copy())

        # warm-start: pad with the first frame until buffer is full
        while len(self._ext_buf) < _VIDEO_BUF:
            self._ext_buf.appendleft(self._ext_buf[0].copy())
            self._wri_buf.appendleft(self._wri_buf[0].copy())

        frames = list(self._ext_buf)
        wrists = list(self._wri_buf)

        # delta_indices [-15, 0] → frames[0] (oldest) and frames[-1] (current)
        ext_video = np.stack([frames[0], frames[-1]], axis=0)   # [T=2, H, W, 3]
        wri_video = np.stack([wrists[0], wrists[-1]], axis=0)   # [T=2, H, W, 3]

        # --- build state ---
        # eef_9d: position (3) + two columns of rotation matrix (6) = 9
        rot_6d = np.concatenate([eef_rotmat[:, 0], eef_rotmat[:, 1]])
        eef_9d = np.concatenate([eef_pos, rot_6d]).astype(np.float32)          # [9]
        gripper = np.array([gripper_pos], dtype=np.float32)                     # [1]
        joints = np.asarray(joint_pos, dtype=np.float32)                        # [7]

        obs = {
            "video": {
                "exterior_image_1_left": ext_video[np.newaxis],  # [1, 2, H, W, 3]
                "wrist_image_left":      wri_video[np.newaxis],  # [1, 2, H, W, 3]
            },
            "state": {
                "eef_9d":           eef_9d[np.newaxis, np.newaxis],    # [1, 1, 9]
                "gripper_position": gripper[np.newaxis, np.newaxis],   # [1, 1, 1]
                "joint_position":   joints[np.newaxis, np.newaxis],    # [1, 1, 7]
            },
            "language": {
                "annotation.language.language_instruction": [[self._instruction]],
            },
        }

        action, _ = self._policy.get_action(obs)
        return action


# ------------------------------------------------------------------
# Standalone smoke-test (no Isaac Lab needed)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Loading FrankaGR00TPolicy...")
    t0 = time.time()
    policy = FrankaGR00TPolicy(device="cuda:0")
    print(f"Loaded in {time.time() - t0:.1f}s")

    policy.reset(instruction="push objects to find the target")

    # dummy inputs matching a 480x640 overhead camera
    H, W = 480, 640
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    eef_pos = np.array([0.5, 0.0, 0.3], dtype=np.float32)
    eef_rotmat = np.eye(3, dtype=np.float32)
    gripper_pos = 0.04
    joint_pos = np.zeros(7, dtype=np.float32)

    print("Running inference...")
    t0 = time.time()
    action = policy.get_action(rgb, eef_pos, eef_rotmat, gripper_pos, joint_pos)
    print(f"Inference in {time.time() - t0:.3f}s")

    for k, v in action.items():
        print(f"  action['{k}']: shape={v.shape}, dtype={v.dtype}")
