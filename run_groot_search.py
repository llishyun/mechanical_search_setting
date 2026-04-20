"""GR00T-N1.7 + OcclusionEnv main loop.

Usage:
    cd /home/leesu37/AP-project/IsaacLab
    # random target (recommended):
    ./isaaclab.sh -p /home/leesu37/AP-project/mechanical_search_setting/run_groot_search.py \\
        --headless --enable_cameras --num_steps 500

    # explicit target:
    ./isaaclab.sh -p /home/leesu37/AP-project/mechanical_search_setting/run_groot_search.py \\
        --headless --enable_cameras --num_steps 500 --target pile_03

Action format (OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT):
    - joint_position  : RELATIVE delta (rad) → add to current joint positions
    - gripper_position: ABSOLUTE [0=closed, 1=open] → map to finger joints (×0.04 m)
    - eef_9d          : RELATIVE delta (pos3 + 6D-rot) — not used (joint control)
"""

import argparse
import os
import random
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI args  (must be before AppLauncher)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="GR00T-N1.7 search in Isaac Lab")
parser.add_argument("--num_steps",   type=int,  default=500,
                    help="Total physics steps to run.")
parser.add_argument("--target",      type=str,  default=None,
                    help="Target pile name (e.g. pile_03). Random if omitted.")
parser.add_argument("--action_freq", type=int,  default=1,
                    help="Re-query GR00T every N sim steps (receding horizon).")
parser.add_argument("--record",      action="store_true",
                    help="Save overhead camera video to outputs/groot_<target>.mp4")
parser.add_argument("--record_every", type=int, default=4,
                    help="Capture 1 frame every N physics steps (default 4).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Isaac Lab imports  (AFTER AppLauncher.app is created)
# ---------------------------------------------------------------------------
import datetime                                                  # noqa: E402
import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import isaaclab.sim as sim_utils                               # noqa: E402
from isaaclab.managers import SceneEntityCfg                   # noqa: E402
from isaaclab.utils.math import matrix_from_quat               # noqa: E402

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from occlusion_env     import OcclusionEnv, OBJECT_NAMES, SETTLE_STEPS  # noqa: E402
from occlusion_env_cfg import OcclusionSceneCfg, PILE_TO_OBJECT_NAME    # noqa: E402
from groot_module      import FrankaGR00TPolicy                          # noqa: E402

# Franka joint name patterns (match SceneEntityCfg regex)
_ARM_JOINTS    = ["panda_joint[1-7]"]
_FINGER_JOINTS = ["panda_finger_joint.*"]
_EE_BODY       = "panda_hand"

# Franka finger scale: gripper_position [0,1] → finger joint position [0, 0.04 m]
_FINGER_SCALE  = 0.04


def _make_instruction(object_name: str) -> str:
    """Convert raw USD object name to a natural language instruction.

    e.g. "Hey_You_Pikachu_Nintendo_64" → "find the hey you pikachu nintendo 64"
    """
    readable = object_name.replace("_", " ").lower()
    return f"find the {readable}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Target selection --------------------------------------------------
    if args_cli.target is not None:
        target_name = args_cli.target
    else:
        target_name = random.choice(OBJECT_NAMES)

    object_name = PILE_TO_OBJECT_NAME[target_name]
    instruction = _make_instruction(object_name)

    print(f"[GR00T] Target : {target_name} ({object_name})")
    print(f"[GR00T] Instruction: '{instruction}'")

    # ---- Simulation context ------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
    sim     = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, -1.5, 1.5], target=[0.5, 0.0, 0.2])

    # ---- Scene (single env) ------------------------------------------------
    scene_cfg = OcclusionSceneCfg(num_envs=1, env_spacing=3.0)
    # GR00T only needs RGB; drop depth/semantic_segmentation to avoid a
    # dtype=0 crash in the Isaac Sim SyntheticData annotator pipeline.
    scene_cfg.camera.data_types = ["rgb"]
    env = OcclusionEnv(cfg=scene_cfg, sim=sim)

    print("[GR00T] Simulation ready.")

    # ---- Render pipeline warmup (CRITICAL) ---------------------------------
    # Problem: headless + enable_cameras 모드에서 Isaac Sim의 SyntheticData
    # annotator가 render var (e.g. LdrColorSD)를 비동기로 등록한다.
    # sim.reset() → render() 가 너무 일찍 호출되면 annotator.attach()가
    #   TypeError: Unable to write from unknown dtype, kind=f, size=0
    # 로 크래시. simulation_app.update()를 먼저 여러 번 호출해 파이프라인을
    # 워밍업한 뒤 sim.reset()을 호출하면 이 문제가 해결된다.
    print("[GR00T] Warming up render pipeline…")
    for _ in range(30):
        simulation_app.update()

    # ---- Initialize PhysX views --------------------------------------------
    # sim.reset(): PhysX를 play 상태로 전환 → 모든 Articulation의
    #   _initialize_impl() 호출 → _root_physx_view, actuators 등 생성.
    # 이 호출 없이 env.reset()을 먼저 하면:
    #   AttributeError: 'Articulation' object has no attribute 'actuators'
    sim.reset()

    # render=False 로 한 스텝 더 밟아 PhysX 내부 뷰를 완전히 구축.
    sim.step(render=False)

    # ---- Resolve joint / body indices (AFTER PhysX is initialized) ---------
    arm_cfg = SceneEntityCfg("robot", joint_names=_ARM_JOINTS, body_names=[_EE_BODY])
    arm_cfg.resolve(env.scene)
    arm_joint_ids = arm_cfg.joint_ids   # list[int], length 7
    ee_body_id    = arm_cfg.body_ids[0] # int

    finger_cfg = SceneEntityCfg("robot", joint_names=_FINGER_JOINTS)
    finger_cfg.resolve(env.scene)
    finger_joint_ids = finger_cfg.joint_ids  # list[int], length 2

    dev = sim.device

    # ---- Load GR00T policy -------------------------------------------------
    print("[GR00T] Loading policy…")
    policy = FrankaGR00TPolicy(device=dev if "cuda" in dev else "cpu")
    policy.reset(instruction=instruction)
    print("[GR00T] Policy ready.")

    # ---- Settle objects ----------------------------------------------------
    print(f"[GR00T] Settling objects ({SETTLE_STEPS} steps)…")
    env.reset(settle=True)

    print("[GR00T] Scene settled. Starting GR00T loop.")

    # ---- Recording setup ---------------------------------------------------
    frame_buffer: list[np.ndarray] = [] if args_cli.record else None

    # ---- Cached action horizon state ---------------------------------------
    _cached_action: dict | None = None
    _horizon_idx: int           = 0

    # ---- Main loop ---------------------------------------------------------
    for step in range(args_cli.num_steps):

        # 1. Overhead + wrist camera RGB  [H, W, 3] uint8  (numpy)
        rgb_tensor   = env.get_camera_rgb()                 # [1, H, W, 3]
        exterior_rgb = rgb_tensor[0].cpu().numpy()          # [H, W, 3]
        wrist_tensor = env.get_wrist_camera_rgb()           # [1, 256, 256, 3]
        wrist_rgb    = wrist_tensor[0].cpu().numpy()        # [256, 256, 3]

        # 2. EEF pose in world frame
        ee_pose_w  = env.robot.data.body_pose_w[:, ee_body_id]  # [1, 7]
        eef_pos    = ee_pose_w[0, :3].cpu().numpy()             # [3]
        ee_quat_t  = ee_pose_w[0, 3:7].unsqueeze(0)            # [1, 4] wxyz
        eef_rotmat = matrix_from_quat(ee_quat_t)[0].cpu().numpy()  # [3, 3]

        # 3. Arm joint positions  [7]
        arm_joint_pos = env.robot.data.joint_pos[:, arm_joint_ids]  # [1, 7]
        joint_pos_np  = arm_joint_pos[0].cpu().numpy()              # [7]

        # 4. Gripper state: mean finger joint / 0.04 → [0, 1]
        finger_pos  = env.robot.data.joint_pos[:, finger_joint_ids]  # [1, 2]
        gripper_pos = float(finger_pos[0].mean().cpu()) / _FINGER_SCALE

        # 5. Query GR00T (re-query every action_freq steps or when horizon exhausted)
        horizon_exhausted = (
            _cached_action is None
            or _horizon_idx >= _cached_action["joint_position"].shape[1]
        )
        if horizon_exhausted or (step % args_cli.action_freq == 0):
            _cached_action = policy.get_action(
                exterior_rgb, eef_pos, eef_rotmat, gripper_pos, joint_pos_np,
                wrist_rgb=wrist_rgb,
            )
            _horizon_idx = 0

        # 6. Extract current-step action from cached horizon
        # joint_position: RELATIVE delta [rad]
        joint_delta = _cached_action["joint_position"][0, _horizon_idx]  # [7] float32
        # gripper_position: ABSOLUTE [0, 1]
        gripper_cmd = float(_cached_action["gripper_position"][0, _horizon_idx, 0])
        _horizon_idx += 1

        # 7. Apply arm joint command  (relative → add delta to current joints)
        delta_t = torch.from_numpy(joint_delta).to(dev).unsqueeze(0)  # [1, 7]
        new_arm = arm_joint_pos + delta_t                              # [1, 7]
        env.robot.set_joint_position_target(new_arm, joint_ids=arm_joint_ids)

        # 8. Apply gripper command  (absolute: map [0,1] → [0, 0.04 m])
        finger_target = torch.full(
            (1, len(finger_joint_ids)),
            float(np.clip(gripper_cmd, 0.0, 1.0)) * _FINGER_SCALE,
            device=dev,
        )
        env.robot.set_joint_position_target(finger_target, joint_ids=finger_joint_ids)

        # 9. Step physics
        env.step()

        # 10. Record frame
        if frame_buffer is not None and step % args_cli.record_every == 0:
            frame = rgb_tensor[0].cpu().numpy().astype(np.uint8)  # [H, W, 3]
            frame_buffer.append(frame.copy())

        if step % 50 == 0:
            delta_mag = float(np.abs(joint_delta).max())
            print(
                f"[GR00T] step={step:4d}  "
                f"eef={np.round(eef_pos, 3)}  "
                f"gripper={gripper_pos:.2f}→{gripper_cmd:.2f}  "
                f"max_joint_delta={delta_mag:.4f} rad"
            )

    print("[GR00T] Done.")

    # ---- Save video --------------------------------------------------------
    if frame_buffer:
        import imageio
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.datetime.now().strftime("%H%M%S")
        out_dir  = os.path.join(_dir, "outputs", date_str)
        os.makedirs(out_dir, exist_ok=True)
        fname    = f"{time_str}_groot_{target_name}.mp4"
        out_path = os.path.join(out_dir, fname)
        fps = max(1, int(1.0 / (1/120) / args_cli.record_every))
        imageio.mimsave(out_path, frame_buffer, fps=fps)
        print(f"[GR00T] Video saved → {out_path}  ({len(frame_buffer)} frames, {fps} fps)")


if __name__ == "__main__":
    main()
    simulation_app.close()