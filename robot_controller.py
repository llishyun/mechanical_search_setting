"""Robot controller for Mechanical Search using Differential IK.

Robot: Franka Emika Panda (7-DOF fixed-base manipulator)

Provides blocking motion primitives:
    - move_to_pose(pos, quat, timeout_steps)
    - open_gripper() / close_gripper()
    - execute_push(p_start, p_end)
    - execute_grasp(grasp_pos)
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms

# Franka Panda constants
FRANKA_JOINT_NAMES  = ["panda_joint[1-7]"]
FRANKA_EE_BODY      = "panda_hand"
FRANKA_FINGER_JOINTS = ["panda_finger_joint.*"]
GRIPPER_OPEN  = 0.04   # meters per finger (Franka max = 40mm)
GRIPPER_CLOSE = 0.0    # meters per finger (fully closed)

# Motion parameters
POS_THRESHOLD = 0.005   # 5mm convergence threshold
MAX_MOVE_STEPS = 300    # max IK steps per move_to_pose call
MAX_PUSH_STEPS = 200    # max IK steps per push segment
GRIPPER_STEPS = 50      # steps to actuate gripper


class RobotController:
    """Blocking motion primitive controller for Franka Panda using Differential IK.

    All motion primitives run the physics simulation internally until the motion
    is complete or a timeout is reached. This is appropriate for scripted policies
    where each action is a discrete, blocking primitive.

    Usage:
        ctrl = RobotController(robot, scene, sim)
        ctrl.execute_push(p_start, p_end)
        ctrl.execute_grasp(grasp_pos)
    """

    def __init__(
        self,
        robot: Articulation,
        scene: InteractiveScene,
        sim: sim_utils.SimulationContext,
        robot_key: str = "robot",
        on_step=None,
    ):
        self.robot = robot
        self.scene = scene
        self.sim = sim
        self.on_step = on_step   # 매 physics step마다 호출되는 선택적 콜백
        self._sim_dt = sim.get_physics_dt()
        self._num_envs = scene.num_envs
        self._device = sim.device

        # --- Resolve arm joint/body indices ---
        self._arm_cfg = SceneEntityCfg(
            robot_key,
            joint_names=FRANKA_JOINT_NAMES,
            body_names=[FRANKA_EE_BODY],
        )
        self._arm_cfg.resolve(scene)

        # For fixed-base robots, Jacobian frame index = body_id - 1
        # (root body is excluded from the Jacobian matrix)
        self._ee_jacobi_idx = self._arm_cfg.body_ids[0] - 1

        # --- Resolve finger joint indices ---
        finger_cfg = SceneEntityCfg(robot_key, joint_names=FRANKA_FINGER_JOINTS)
        finger_cfg.resolve(scene)
        self._finger_joint_ids = finger_cfg.joint_ids

        # --- Differential IK controller (DLS = Damped Least Squares) ---
        # DLS is more robust near singularities than plain pseudo-inverse
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        )
        self._ik = DifferentialIKController(
            ik_cfg,
            num_envs=self._num_envs,
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Public motion primitives
    # ------------------------------------------------------------------

    def move_to_pose(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        timeout_steps: int = MAX_MOVE_STEPS,
    ) -> bool:
        """Move end-effector to target pose, stepping sim until convergence.

        Args:
            target_pos:    [num_envs, 3] target position in world frame.
            target_quat:   [num_envs, 4] target orientation (w, x, y, z) in world frame.
            timeout_steps: max physics steps before giving up.

        Returns:
            True if all envs converged within POS_THRESHOLD.
        """
        # Convert world-frame target to robot root frame (IK controller works in root frame)
        root_pose_w = self.robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pos, target_quat,
        )
        command = torch.cat([target_pos_b, target_quat_b], dim=-1)  # [N, 7]

        self._ik.reset()
        self._ik.set_command(command)

        for _ in range(timeout_steps):
            # --- IK: compute desired joint positions ---
            jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self._ee_jacobi_idx, :, self._arm_cfg.joint_ids
            ]
            ee_pose_w = self.robot.data.body_pose_w[:, self._arm_cfg.body_ids[0]]
            root_pose_w = self.robot.data.root_pose_w

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
            )
            joint_pos = self.robot.data.joint_pos[:, self._arm_cfg.joint_ids]
            joint_pos_des = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # --- Apply and step ---
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_cfg.joint_ids)
            self._step_sim()

            # --- Check convergence (position only) ---
            ee_pose_w = self.robot.data.body_pose_w[:, self._arm_cfg.body_ids[0]]
            pos_error = torch.norm(ee_pose_w[:, 0:3] - target_pos, dim=-1)
            if pos_error.max().item() < POS_THRESHOLD:
                return True

        return False

    def open_gripper(self):
        """Open Franka gripper fully."""
        target = torch.full(
            (self._num_envs, len(self._finger_joint_ids)),
            GRIPPER_OPEN,
            device=self._device,
        )
        self._set_gripper(target)

    def close_gripper(self):
        """Close Franka gripper fully."""
        target = torch.full(
            (self._num_envs, len(self._finger_joint_ids)),
            GRIPPER_CLOSE,
            device=self._device,
        )
        self._set_gripper(target)

    def execute_push(
        self,
        p_start: torch.Tensor,
        p_end: torch.Tensor,
        approach_height: float = 0.20,
    ) -> bool:
        """Execute a linear push from p_start to p_end.

        Motion sequence:
            1. Move above p_start (pre-approach, clears wall at z=0.12)
            2. Descend to p_start
            3. Push linearly to p_end
            4. Retract above p_end

        Args:
            p_start:         [num_envs, 3] push start position (world frame).
            p_end:           [num_envs, 3] push end position (world frame).
            approach_height: meters above p_start/p_end. Must exceed wall height
                             (0.12m) since robot base is outside wall at x=-0.2.

        Returns:
            True if all motion segments completed.
        """
        push_quat = self._downward_quat()

        # 1. Pre-approach: above p_start (high enough to clear the boundary wall)
        pre_pos = p_start.clone()
        pre_pos[:, 2] += approach_height
        self.move_to_pose(pre_pos, push_quat, timeout_steps=MAX_PUSH_STEPS)

        # 2. Descend to p_start
        self.move_to_pose(p_start, push_quat, timeout_steps=MAX_PUSH_STEPS)

        # 3. Push to p_end
        self.move_to_pose(p_end, push_quat, timeout_steps=MAX_PUSH_STEPS)

        # 4. Retract above p_end
        retract_pos = p_end.clone()
        retract_pos[:, 2] += approach_height
        self.move_to_pose(retract_pos, push_quat, timeout_steps=MAX_PUSH_STEPS)

        return True

    def execute_grasp(
        self,
        grasp_pos: torch.Tensor,
        approach_height: float = 0.15,
        lift_height: float = 0.20,
    ) -> bool:
        """Execute a top-down grasp.

        Motion sequence:
            1. Open gripper
            2. Move above grasp_pos (pre-grasp)
            3. Descend to grasp_pos
            4. Close gripper
            5. Lift to lift_height above grasp_pos

        Args:
            grasp_pos:       [num_envs, 3] grasp position (world frame).
            approach_height: meters above grasp_pos for pre-grasp waypoint.
            lift_height:     meters above grasp_pos to lift after closing.

        Returns:
            True if lift motion converged (proxy for successful grasp).
        """
        grasp_quat = self._downward_quat()

        # 1. Open gripper
        self.open_gripper()

        # 2. Pre-grasp: above target
        pre_pos = grasp_pos.clone()
        pre_pos[:, 2] += approach_height
        self.move_to_pose(pre_pos, grasp_quat, timeout_steps=MAX_MOVE_STEPS)

        # 3. Descend to grasp position
        self.move_to_pose(grasp_pos, grasp_quat, timeout_steps=MAX_MOVE_STEPS)

        # 4. Close gripper
        self.close_gripper()

        # 5. Lift (convergence here = object likely in hand)
        lift_pos = grasp_pos.clone()
        lift_pos[:, 2] += lift_height
        success = self.move_to_pose(lift_pos, grasp_quat, timeout_steps=MAX_MOVE_STEPS)

        return success

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_sim(self):
        """Advance one physics step and update scene buffers."""
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self._sim_dt)
        if self.on_step is not None:
            self.on_step()

    def _set_gripper(self, target: torch.Tensor):
        """Hold gripper position target for GRIPPER_STEPS physics steps."""
        for _ in range(GRIPPER_STEPS):
            self.robot.set_joint_position_target(target, joint_ids=self._finger_joint_ids)
            self._step_sim()

    def _downward_quat(self) -> torch.Tensor:
        """Return [num_envs, 4] quaternion (w,x,y,z) for downward-facing gripper.

        Represents a 180° rotation around the X axis, which points the Franka
        hand's approach axis straight down (-Z world direction).
        """
        q = torch.zeros(self._num_envs, 4, device=self._device)
        q[:, 0] = 0.0  # w
        q[:, 1] = 1.0  # x
        q[:, 2] = 0.0  # y
        q[:, 3] = 0.0  # z
        return q
