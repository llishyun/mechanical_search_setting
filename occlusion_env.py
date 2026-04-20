"""Occlusion environment: drops objects onto a table to form a pile.

Objects are placed at random positions above the table, then fall under gravity
to create a natural pile (occlusion). OpenArm bimanual robot is stationary for now.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sensors.camera import Camera

from occlusion_env_cfg import OcclusionSceneCfg


# Pile object names — 20 occluders
OBJECT_NAMES = [f"pile_{i:02d}" for i in range(20)]

# Drop zone: inside the basket (centered at x=0.5, y=0).
# Basket is scaled 2x so interior is roughly ±0.20m in x/y.
# Drop from 0.4~0.8m above the basket bottom (z=0) so fall is visible but not too high.
DROP_X_RANGE = (0.30, 0.70)
DROP_Y_RANGE = (-0.20, 0.20)
DROP_Z_RANGE = (0.10, 0.30)

# Steps to let the full pile settle before agent acts
SETTLE_STEPS = 600


class OcclusionEnv:
    """Scene manager for the occlusion pile environment."""

    def __init__(self, cfg: OcclusionSceneCfg, sim: sim_utils.SimulationContext):
        self.cfg = cfg
        self.sim = sim
        self.scene = InteractiveScene(cfg)

        # Grab handles after scene is built
        self.robot: Articulation = self.scene["robot"]
        self.camera: Camera = self.scene["camera"]
        self.wrist_camera: Camera = self.scene["wrist_camera"]
        self.objects: list[RigidObject] = [self.scene[name] for name in OBJECT_NAMES]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, settle: bool = False):
        """Reset scene: place objects at random drop positions above table.

        Args:
            settle: If True, run SETTLE_STEPS physics steps before returning
                    (objects will be piled up already). If False, objects start
                    mid-air so the caller can record the fall.
        """
        self.scene.reset()

        # Set camera to look at the pile on the table
        num_envs = self.cfg.num_envs #4
        device = self.sim.device  #GPU

        env_origins = self.scene.env_origins  
        # [4, 3], 각 env의 원점 위치 (x, y, z) 좌표, 4개의 env가 x축 방향으로 2m 간격으로 배치되어 있으므로, env_origins는 [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]] 형태일 것임
        # pile 중심(x=0.5) 수직 내려보기, 높이 2.5m
        cam_pos    = env_origins + torch.tensor([[0.5, 0.0, 2.5]], device=device)
        cam_target = env_origins + torch.tensor([[0.5, 0.0, 0.0]], device=device)
        self.camera.set_world_poses_from_view(cam_pos, cam_target)

        # 모든 물체를 동시에 staggered 높이로 배치 → recording 시작 즉시 낙하가 찍힘
        quat = torch.zeros(num_envs, 4, device=device) #회전 zero
        quat[:, 0] = 1.0 # unit quaternion (w, x, y, z) 형식에서 w=1, x=y=z=0이면 회전이 없는 상태 
        vel = torch.zeros(num_envs, 6, device=device) 
        #sh

        for obj in self.objects:
            x = torch.rand(num_envs, device=device) * (DROP_X_RANGE[1] - DROP_X_RANGE[0]) + DROP_X_RANGE[0]
            y = torch.rand(num_envs, device=device) * (DROP_Y_RANGE[1] - DROP_Y_RANGE[0]) + DROP_Y_RANGE[0]
            z = torch.rand(num_envs, device=device) * (DROP_Z_RANGE[1] - DROP_Z_RANGE[0]) + DROP_Z_RANGE[0]
            pos = torch.stack([x, y, z], dim=-1)

            obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
            obj.write_root_velocity_to_sim(vel)

        self.scene.write_data_to_sim()

        if settle:
            self._settle()

    def step(self):
        """Single physics + render step."""
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

    def get_camera_rgb(self) -> torch.Tensor:
        """Return overhead RGB frames: [num_envs, H, W, 3] uint8."""
        self.camera.update(dt=self.sim.get_physics_dt())
        rgb = self.camera.data.output["rgb"]   # [N, H, W, 4] RGBA
        return rgb[..., :3]                    # drop alpha

    def get_wrist_camera_rgb(self) -> torch.Tensor:
        """Return wrist camera RGB frames: [num_envs, H, W, 3] uint8."""
        self.wrist_camera.update(dt=self.sim.get_physics_dt())
        rgb = self.wrist_camera.data.output["rgb"]  # [N, H, W, 4] RGBA
        return rgb[..., :3]

    def get_camera_rgbd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (RGB, depth) frames.

        Returns:
            rgb:   [num_envs, H, W, 3] uint8
            depth: [num_envs, H, W, 1] float32, in metres (inf/nan for invalid pixels)
        """
        self.camera.update(dt=self.sim.get_physics_dt())
        rgb   = self.camera.data.output["rgb"][..., :3]    # drop alpha
        depth = self.camera.data.output["depth"]           # [N, H, W, 1]
        return rgb, depth

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _settle(self):
        """Run physics steps without rendering to let objects settle."""
        for _ in range(SETTLE_STEPS):
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
