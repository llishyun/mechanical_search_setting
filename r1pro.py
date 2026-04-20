# Copyright (c) 2024, The AP-Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Stanford BEHAVIOR R1 Pro robot.

The R1 Pro is a mobile manipulation robot with:
  - Holonomic base  : steer_motor_joint1-3 (revolute) + wheel_motor_joint1-3 (continuous)
  - Torso           : torso_joint1-4 (revolute)
  - Left  arm       : left_arm_joint1-7  (revolute)
  - Left  gripper   : left_gripper_finger_joint1-2 (prismatic, 0~0.05 m)
  - Right arm       : right_arm_joint1-7 (revolute)
  - Right gripper   : right_gripper_finger_joint1-2 (prismatic, 0~0.05 m)
  Total active DOF  : 28  (6 base + 4 torso + 7+2 left + 7+2 right)

USD 생성 방법 (Isaac Sim 필요, 최초 1회):
  cd /home/leesu37/AP-project/IsaacLab
  ./isaaclab.sh -p scripts/tools/convert_urdf.py \\
      /home/leesu37/AP-project/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf \\
      /home/leesu37/AP-project/mechanical_search_setting/assets/r1pro/r1pro.usd \\
      --merge-joints \\
      --headless

Note:
  - --merge-joints  : fixed joint(gripper_joint, realsense_joint 등)를 상위 링크로 합쳐 prim 수 절감
  - base는 floating (고정 안 함) → 로봇이 바닥 위에 서 있음
  - GR00T BEHAVIOR_R1_PRO embodiment tag와 함께 사용
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_R1PRO_USD = (
    "/home/leesu37/AP-project/mechanical_search_setting/assets/r1pro/r1pro.usd"
)

R1PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_R1PRO_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 테이블 뒤쪽으로 충분히 이격 (팔 닿는 거리 무관)
        pos=(2.0, 0.0, -1.05),
        rot=(0.0, 0.0, 0.0, 1.0),  # 180° around Z → -x 방향(pile 쪽) 바라봄
        joint_pos={
            # 전 관절 중립 (0) — viewer에서 자세 확인 후 조정
            ".*": 0.0,
        },
    ),
    actuators={
        # ── 베이스 ─────────────────────────────────────────────────────────
        "base_steer": ImplicitActuatorCfg(
            joint_names_expr=["steer_motor_joint[1-3]"],
            effort_limit_sim=50.0,
            velocity_limit_sim=20.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "base_wheel": ImplicitActuatorCfg(
            joint_names_expr=["wheel_motor_joint[1-3]"],
            effort_limit_sim=50.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,   # velocity-controlled wheel
            damping=20.0,
        ),
        # ── 토르소 ─────────────────────────────────────────────────────────
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint[1-4]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=2.5,
            stiffness=400.0,
            damping=40.0,
        ),
        # ── 왼팔 ───────────────────────────────────────────────────────────
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint[1-7]"],
            effort_limit_sim={
                "left_arm_joint[1-2]": 55.0,
                "left_arm_joint[3-4]": 25.0,
                "left_arm_joint[5-7]": 18.0,
            },
            velocity_limit_sim={
                "left_arm_joint[1-2]": 7.1209,
                "left_arm_joint[3-4]": 8.3776,
                "left_arm_joint[5-7]": 10.472,
            },
            stiffness=400.0,
            damping=40.0,
        ),
        # ── 오른팔 ─────────────────────────────────────────────────────────
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint[1-7]"],
            effort_limit_sim={
                "right_arm_joint[1-2]": 55.0,
                "right_arm_joint[3-4]": 25.0,
                "right_arm_joint[5-7]": 18.0,
            },
            velocity_limit_sim={
                "right_arm_joint[1-2]": 7.1209,
                "right_arm_joint[3-4]": 8.3776,
                "right_arm_joint[5-7]": 10.472,
            },
            stiffness=400.0,
            damping=40.0,
        ),
        # ── 왼손 그리퍼 ────────────────────────────────────────────────────
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper_finger_joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=0.25,
            stiffness=2e3,
            damping=1e2,
        ),
        # ── 오른손 그리퍼 ──────────────────────────────────────────────────
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_gripper_finger_joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=0.25,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""R1 Pro ArticulationCfg — floating base, full 28 DOF."""


R1PRO_HIGH_PD_CFG = R1PRO_CFG.copy()
R1PRO_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
R1PRO_HIGH_PD_CFG.actuators["torso"].stiffness    = 800.0
R1PRO_HIGH_PD_CFG.actuators["torso"].damping      = 80.0
R1PRO_HIGH_PD_CFG.actuators["left_arm"].stiffness  = 800.0
R1PRO_HIGH_PD_CFG.actuators["left_arm"].damping    = 80.0
R1PRO_HIGH_PD_CFG.actuators["right_arm"].stiffness = 800.0
R1PRO_HIGH_PD_CFG.actuators["right_arm"].damping   = 80.0
"""R1 Pro — High PD variant (gravity off, stiff gains).

Task-space / IK 제어나 GR00T zero-shot 평가 초기 테스트에 적합.
"""
