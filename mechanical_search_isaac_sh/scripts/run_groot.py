"""GR00T-N1.7 closed-loop control for MechanicalSearchEnv.

GR00T 모델은 별도 프로세스(groot_server.py)에서 실행되며,
이 스크립트는 Isaac Lab 시뮬레이션을 구동하는 클라이언트 역할.

Embodiment: OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
Camera mapping:
    exterior_image_1_left ← env.get_rgb()      (overhead 480×640)
    wrist_image_left      ← env.get_wrist_rgb() (wrist 256×256 on panda_hand)
Control:
    joint_position  RELATIVE delta → add to current joints
    gripper_position ABSOLUTE [0,1] → × 0.04 m → finger joints

Usage:
    # 터미널 1: GR00T 서버 먼저 기동
    cd /home/leesu37/AP-project/Isaac-GR00T_N1.7
    uv run python \\
        /home/leesu37/AP-project/mechanical_search_setting/mechanical_search_isaac_sh/scripts/groot_server.py

    # 터미널 2: Isaac Lab 클라이언트 실행
    cd /home/leesu37/AP-project/IsaacLab
    ./isaaclab.sh -p \\
        /home/leesu37/AP-project/mechanical_search_setting/mechanical_search_isaac_sh/scripts/run_groot.py \\
        --headless --enable_cameras

Options:
    --num_steps    Total physics steps          (default: 500)
    --action_freq  Re-query server every N steps (default: 1)
    --seed         RNG seed
    --instruction  Language instruction (auto if omitted)
    --groot_port   GR00T server TCP port        (default: 5000)
    --record       Save overhead video
    --record_every Capture 1 frame every N steps (default: 4)
    --no_settle    Skip pile-settle phase
"""

import argparse
import os
import sys

for _flag in ("--/rtx/post/aa/op=0", "--/rtx/post/dlss/execMode=0"):
    if _flag not in sys.argv:
        sys.argv.append(_flag)

from isaaclab.app import AppLauncher  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="GR00T closed-loop (socket client)")
parser.add_argument("--num_steps",    type=int,  default=500)
parser.add_argument("--action_freq",  type=int,  default=1,
                    help="Re-query GR00T server every N sim steps.")
parser.add_argument("--seed",         type=int,  default=None)
parser.add_argument("--instruction",  type=str,  default=None)
parser.add_argument("--groot_port",   type=int,  default=5000,
                    help="TCP port of groot_server.py (default: 5000)")
parser.add_argument("--record",       action="store_true")
parser.add_argument("--record_every", type=int,  default=4)
parser.add_argument("--no_settle",    action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# carb settings (DLSS off)
# ---------------------------------------------------------------------------
import carb as _carb  # noqa: E402
_s = _carb.settings.get_settings()
_s.set("/rtx/post/dlss/execMode",        0)
_s.set("/rtx/post/aa/op",                0)
_s.set("/rtx/post/histogram/enabled",    False)
_s.set("/rtx/post/lensFlares/enabled",   False)
_s.set("/rtx/post/motionblur/enabled",   False)

# ---------------------------------------------------------------------------
# Post-AppLauncher imports
# ---------------------------------------------------------------------------
import datetime        # noqa: E402
import numpy as np     # noqa: E402
import torch           # noqa: E402
import imageio         # noqa: E402

import isaaclab.sim as sim_utils                 # noqa: E402
from isaaclab.managers import SceneEntityCfg     # noqa: E402
from isaaclab.utils.math import matrix_from_quat # noqa: E402

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)

from configs.scene_cfg import MechSearchEnvCfg, PILE_TO_OBJECT_NAME  # noqa: E402
from envs.mechanical_search_env import MechanicalSearchEnv            # noqa: E402
from envs.groot_client import GR00TSocketClient                       # noqa: E402

_ARM_JOINTS    = ["panda_joint[1-7]"]
_FINGER_JOINTS = ["panda_finger_joint.*"]
_EE_BODY       = "panda_hand"
_FINGER_SCALE  = 0.04


# ---------------------------------------------------------------------------
# SyntheticData patch
# ---------------------------------------------------------------------------

_MAX_RETRIES = 10

def _patch_syntheticdata() -> None:
    try:
        from omni.syntheticdata.scripts.SyntheticData import SyntheticData as _SD
        _orig = _SD._add_node_downstream_intergraph_dependency
        _app  = simulation_app

        @staticmethod
        def _safe(node, downstream_node_handle):
            try:
                return _orig(node, downstream_node_handle)
            except TypeError:
                for _ in range(_MAX_RETRIES):
                    _app.update()
                    try:
                        return _orig(node, downstream_node_handle)
                    except TypeError:
                        continue
                print("[Patch][Warning] OmniGraph node still uninitialised after retries.")
                return 1

        _SD._add_node_downstream_intergraph_dependency = _safe
        print("[Patch] SyntheticData patch applied.")
    except Exception as exc:
        print(f"[Patch][Warning] SyntheticData patch failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = os.path.join(_proj_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── Simulation context ───────────────────────────────────────────────
    env_cfg = MechSearchEnvCfg()
    sim     = sim_utils.SimulationContext(env_cfg.sim)
    sim.set_camera_view(eye=[0.5, -1.5, 1.5], target=[0.5, 0.0, 0.2])

    env = MechanicalSearchEnv(cfg=env_cfg.scene, sim=sim)

    _patch_syntheticdata()
    print("[Setup] Warming up render pipeline (5 frames)...")
    for _ in range(5):
        simulation_app.update()

    sim.reset()
    sim.step(render=False)

    # ── Joint / body index resolution ────────────────────────────────────
    arm_cfg = SceneEntityCfg("robot", joint_names=_ARM_JOINTS, body_names=[_EE_BODY])
    arm_cfg.resolve(env.scene)
    arm_joint_ids = arm_cfg.joint_ids
    ee_body_id    = arm_cfg.body_ids[0]

    finger_cfg = SceneEntityCfg("robot", joint_names=_FINGER_JOINTS)
    finger_cfg.resolve(env.scene)
    finger_joint_ids = finger_cfg.joint_ids

    dev = sim.device

    # ── GR00T 소켓 클라이언트 연결 ────────────────────────────────────────
    print(f"[Client] Connecting to GR00T server (port {args_cli.groot_port})...")
    client = GR00TSocketClient(host="localhost", port=args_cli.groot_port)

    # ── Reset env + settle ───────────────────────────────────────────────
    target_name = env.reset(seed=args_cli.seed, settle=not args_cli.no_settle)
    object_name = PILE_TO_OBJECT_NAME.get(target_name, target_name)
    instruction = args_cli.instruction or f"find the {object_name.replace('_', ' ').lower()}"

    print(f"[GR00T] Target     : {target_name} ({object_name})")
    print(f"[GR00T] Instruction: '{instruction}'")

    client.reset(instruction=instruction)

    # ── Recording setup ──────────────────────────────────────────────────
    frame_buffer: list[np.ndarray] = [] if args_cli.record else None

    # ── Receding horizon cache ───────────────────────────────────────────
    _cached_action: dict | None = None
    _horizon_idx: int           = 0

    # ── Main loop ────────────────────────────────────────────────────────
    print(f"[GR00T] Starting loop ({args_cli.num_steps} steps)...")
    try:
        for step in range(args_cli.num_steps):

            # 1. Camera observations
            # exterior_image_1_left ← overhead camera
            exterior_rgb = env.get_rgb()[0].cpu().numpy()        # [480, 640, 3] uint8
            # wrist_image_left ← wrist camera on panda_hand
            wrist_rgb    = env.get_wrist_rgb()[0].cpu().numpy()  # [256, 256, 3] uint8

            # 2. EEF pose (world frame)
            ee_pose_w  = env.robot.data.body_pose_w[:, ee_body_id]           # [1, 7]
            eef_pos    = ee_pose_w[0, :3].cpu().numpy()                      # [3]
            eef_rotmat = matrix_from_quat(
                ee_pose_w[0, 3:7].unsqueeze(0)
            )[0].cpu().numpy()                                                # [3, 3]

            # 3. Arm joint positions
            arm_jpos  = env.robot.data.joint_pos[:, arm_joint_ids]  # [1, 7]
            joint_pos = arm_jpos[0].cpu().numpy()                   # [7]

            # 4. Gripper state → [0, 1]
            finger_pos  = env.robot.data.joint_pos[:, finger_joint_ids]  # [1, 2]
            gripper_pos = float(finger_pos[0].mean().cpu()) / _FINGER_SCALE

            # 5. Query GR00T server (receding horizon)
            horizon_exhausted = (
                _cached_action is None
                or _horizon_idx >= _cached_action["joint_position"].shape[0]
            )
            if horizon_exhausted or (step % args_cli.action_freq == 0):
                _cached_action = client.get_action(
                    exterior_rgb=exterior_rgb,
                    wrist_rgb=wrist_rgb,
                    eef_pos=eef_pos,
                    eef_rotmat=eef_rotmat,
                    gripper_pos=gripper_pos,
                    joint_pos=joint_pos,
                )
                _horizon_idx = 0

            # 6. Extract current-step action
            joint_delta = _cached_action["joint_position"][_horizon_idx]       # [7] float32
            gripper_cmd = float(_cached_action["gripper_position"][_horizon_idx, 0])
            _horizon_idx += 1

            # 7. Apply arm: current + relative delta
            delta_t = torch.from_numpy(joint_delta).to(dev).unsqueeze(0)  # [1, 7]
            new_arm = arm_jpos + delta_t
            env.robot.set_joint_position_target(new_arm, joint_ids=arm_joint_ids)

            # 8. Apply gripper: absolute → meters
            finger_target = torch.full(
                (1, len(finger_joint_ids)),
                float(np.clip(gripper_cmd, 0.0, 1.0)) * _FINGER_SCALE,
                device=dev,
            )
            env.robot.set_joint_position_target(finger_target, joint_ids=finger_joint_ids)

            # 9. Step physics
            env.step(render=True)

            # 10. Record
            if frame_buffer is not None and step % args_cli.record_every == 0:
                frame_buffer.append(exterior_rgb.copy())

            if step % 50 == 0:
                delta_mag = float(np.abs(joint_delta).max())
                print(
                    f"[GR00T] step={step:4d}  "
                    f"eef={np.round(eef_pos, 3)}  "
                    f"gripper={gripper_pos:.2f}→{gripper_cmd:.2f}  "
                    f"max_joint_delta={delta_mag:.4f} rad"
                )

    finally:
        client.close()

    print("[GR00T] Done.")

    # ── Save video ───────────────────────────────────────────────────────
    if frame_buffer:
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        vid_path = os.path.join(out_dir, f"{stamp}_groot_{target_name}.mp4")
        fps      = max(1, int((1.0 / env_cfg.sim.dt) / args_cli.record_every))
        imageio.mimsave(vid_path, frame_buffer, fps=fps)
        print(f"[GR00T] Video → {vid_path}  ({len(frame_buffer)} frames, {fps} fps)")


if __name__ == "__main__":
    main()
    simulation_app.close()
