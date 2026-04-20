"""MechanicalSearchEnv: drops 20 objects on a desk, selects a random target."""

from __future__ import annotations

import math
import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sensors.camera import Camera

from configs.scene_cfg import (
    DROP_X,
    DROP_Y,
    DROP_Z,
    NUM_OBJECTS,
    PILE_NAMES,
    PILE_TO_OBJECT_NAME,
    SETTLE_STEPS,
    MechSearchSceneCfg,
)


class MechanicalSearchEnv:
    """Drops 20 objects onto a desk and tracks a randomly chosen target object."""

    def __init__(self, cfg: MechSearchSceneCfg, sim: sim_utils.SimulationContext):
        self.cfg   = cfg
        self.sim   = sim
        self.scene = InteractiveScene(cfg)

        self.robot:        Articulation      = self.scene["robot"]
        self.camera:       Camera            = self.scene["camera"]
        self.wrist_camera: Camera            = self.scene["wrist_camera"]
        self.objects:      list[RigidObject] = [self.scene[n] for n in PILE_NAMES]

        self.target_idx:  int = -1
        self.target_name: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, settle: bool = True) -> str:
        """Reset the scene, select a target, and drop all objects.

        Args:
            seed:   RNG seed for reproducibility. None = non-deterministic.
            settle: If True, run SETTLE_STEPS physics steps before returning.

        Returns:
            target_name: The USD directory name of the selected target object.
        """
        if seed is not None:
            random.seed(seed)

        self.scene.reset()

        # Target selection — stored as plain text
        self.target_idx  = random.randint(0, NUM_OBJECTS - 1)
        self.target_name = PILE_TO_OBJECT_NAME[PILE_NAMES[self.target_idx]]

        device     = self.sim.device
        num_envs   = self.cfg.num_envs
        env_origins = self.scene.env_origins  # [N, 3]

        # Overhead camera pose — must be set AFTER scene.reset()
        cam_pos    = env_origins + torch.tensor([[0.5, 0.0, 2.5]], device=device)
        cam_target = env_origins + torch.tensor([[0.5, 0.0, 0.0]], device=device)
        self.camera.set_world_poses_from_view(cam_pos, cam_target)

        # Drop all 20 objects at random positions within the drop zone
        vel = torch.zeros(num_envs, 6, device=device)
        for obj in self.objects:
            pos  = self._random_drop_pos(num_envs, device)
            quat = self._random_yaw_quat(num_envs, device)
            obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
            obj.write_root_velocity_to_sim(vel)

        self.scene.write_data_to_sim()

        if settle:
            self._settle()

        return self.target_name

    def step(self, render: bool = True) -> None:
        """Advance one physics step.

        Args:
            render: Pass True when you need fresh camera data this step.
                    Pass False during settle to skip RTX rendering overhead.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=render)
        self.scene.update(self.sim.get_physics_dt())

    def get_rgb(self) -> torch.Tensor:
        """Overhead RGB: [N, 480, 640, 3] uint8."""
        self.camera.update(dt=self.sim.get_physics_dt())
        return self.camera.data.output["rgb"][..., :3]

    def get_wrist_rgb(self) -> torch.Tensor:
        """Wrist RGB: [N, 256, 256, 3] uint8."""
        self.wrist_camera.update(dt=self.sim.get_physics_dt())
        return self.wrist_camera.data.output["rgb"][..., :3]

    def get_rgbd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Overhead (RGB, depth). depth: [N, 480, 640, 1] float32 metres."""
        self.camera.update(dt=self.sim.get_physics_dt())
        rgb   = self.camera.data.output["rgb"][..., :3]
        depth = self.camera.data.output["depth"]
        return rgb, depth

    def get_obs(self) -> dict:
        """All sensor observations bundled as a dict.

        Returns:
            overhead_rgb:   [N, 480, 640, 3]  uint8
            overhead_depth: [N, 480, 640, 1]  float32 (metres)
            wrist_rgb:      [N, 256, 256, 3]  uint8
            target_idx:     int   index into PILE_NAMES (0–19)
            target_name:    str   USD directory name of target object
        """
        rgb, depth = self.get_rgbd()
        return {
            "overhead_rgb":   rgb,
            "overhead_depth": depth,
            "wrist_rgb":      self.get_wrist_rgb(),
            "target_idx":     self.target_idx,
            "target_name":    self.target_name,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _settle(self) -> None:
        """Run physics without rendering to let the pile come to rest."""
        for _ in range(SETTLE_STEPS):
            self.sim.step(render=False)
            self.scene.update(self.sim.get_physics_dt())

    @staticmethod
    def _random_drop_pos(num_envs: int, device: str) -> torch.Tensor:
        x = torch.rand(num_envs, device=device) * (DROP_X[1] - DROP_X[0]) + DROP_X[0]
        y = torch.rand(num_envs, device=device) * (DROP_Y[1] - DROP_Y[0]) + DROP_Y[0]
        z = torch.rand(num_envs, device=device) * (DROP_Z[1] - DROP_Z[0]) + DROP_Z[0]
        return torch.stack([x, y, z], dim=-1)  # [N, 3]

    @staticmethod
    def _random_yaw_quat(num_envs: int, device: str) -> torch.Tensor:
        """Random rotation around Z axis only (WXYZ convention)."""
        yaw = torch.rand(num_envs, device=device) * 2.0 * math.pi
        quat = torch.zeros(num_envs, 4, device=device)
        quat[:, 0] = torch.cos(yaw / 2)  # w
        quat[:, 3] = torch.sin(yaw / 2)  # z
        return quat  # [N, 4]
