"""Mechanical Search 실행 엔트리포인트.

Usage:
    cd /home/leesu37/AP-project/IsaacLab
    # target 랜덤 선택 (권장):
    ./isaaclab.sh -p /home/leesu37/AP-project/mechanical_search_setting/run_mechanical_search.py \\
        --headless --enable_cameras \\
        --method random --max_actions 20

    # target 직접 지정:
    ./isaaclab.sh -p /home/leesu37/AP-project/mechanical_search_setting/run_mechanical_search.py \\
        --headless --enable_cameras \\
        --target pile_03 --method largest_first --max_actions 20
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------
# Args (AppLauncher.add_app_launcher_args 보다 먼저 정의)
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Mechanical Search in Isaac Lab")
parser.add_argument(
    "--target", type=str, default=None,
    help="목표 오브젝트 pile name (예: pile_03). 미지정 시 랜덤 선택.",
)
parser.add_argument(
    "--method", type=str, default="random",
    choices=["random", "largest_first"],
    help="Priority list 생성 방식: random | largest_first",
)
parser.add_argument(
    "--max_actions", type=int, default=20,
    help="최대 push 횟수",
)
parser.add_argument(
    "--record", action="store_true",
    help="영상 녹화 (mechanical_search.mp4 저장)",
)
parser.add_argument(
    "--record_every", type=int, default=4,
    help="N physics step마다 1프레임 캡처 (기본 4)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------
# Isaac Lab imports (AppLauncher 이후)
# -----------------------------------------------------------------------
import random  # noqa: E402

import imageio  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from occlusion_env import OcclusionEnv, OBJECT_NAMES  # noqa: E402
from occlusion_env_cfg import OcclusionSceneCfg, PILE_TO_OBJECT_NAME  # noqa: E402
from robot_controller import RobotController  # noqa: E402
from mechanical_search import MechanicalSearch  # noqa: E402


def main():
    # ---- Simulation context ----
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.0, 1.5], target=[0.5, 0.0, 0.3])

    # ---- Scene (num_envs=1: Mechanical Search는 순차 알고리즘) ----
    scene_cfg = OcclusionSceneCfg(num_envs=1, env_spacing=3.0)
    env = OcclusionEnv(cfg=scene_cfg, sim=sim)

    sim.reset()
    print("[INFO]: Simulation ready.")

    # ---- Target 선택 (랜덤 or 직접 지정) ----
    if args_cli.target is None:
        target_name = random.choice(OBJECT_NAMES)
        print(f"[INFO]: target 랜덤 선택 → {target_name}")
    else:
        target_name = args_cli.target
        print(f"[INFO]: target 직접 지정 → {target_name}")
    target_object_name = PILE_TO_OBJECT_NAME[target_name]
    print(f"[INFO]: target object = {target_object_name}")
    print(f"[INFO]: method={args_cli.method}")

    # ---- Controller (Franka Panda) ----
    ctrl = RobotController(
        env.robot, env.scene, sim,
        robot_key="robot",
    )

    # ---- Mechanical Search 실행 ----
    frame_buffer = [] if args_cli.record else None
    ms = MechanicalSearch(
        env, ctrl,
        method=args_cli.method,
        frame_buffer=frame_buffer,
        record_every=args_cli.record_every,
    )
    result = ms.run(
        target_name=target_name,
        max_actions=args_cli.max_actions,
        env_idx=0,
        target_object_name=target_object_name,
    )

    # ---- 결과 출력 ----
    print("\n" + "=" * 60)
    print(f"  Result  : {'SUCCESS' if result['success'] else 'FAILURE'}")
    print(f"  Method  : {args_cli.method}")
    print(f"  Target  : {target_name} ({target_object_name})")
    print(f"  Actions : {result['n_actions']} / {args_cli.max_actions}")
    print("=" * 60)
    print("\n[Step-by-step history]")
    for h in result["history"]:
        status = "FOUND" if h["target_visible"] else "hidden"
        print(
            f"  step {h['step']:02d}: target={status} | "
            f"top3={h['priority_list'][:3]}"
        )

    # ---- 영상 저장 ----
    if args_cli.record and frame_buffer:
        import datetime
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.datetime.now().strftime("%H%M%S")
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", date_str)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{time_str}_{target_name}_{target_object_name}_{args_cli.method}.mp4"
        out_path = os.path.join(out_dir, filename)
        fps = max(1, int(1.0 / sim_cfg.dt / args_cli.record_every))
        imageio.mimsave(out_path, frame_buffer, fps=fps)
        print(f"\n[INFO]: 영상 저장 완료 → {out_path}  ({len(frame_buffer)} frames, {fps} fps)")


if __name__ == "__main__":
    main()
    simulation_app.close()
