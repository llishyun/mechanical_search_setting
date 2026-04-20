"""Record overhead RGB video of pile formation.

Usage:
    cd /home/leesu37/AP-project/IsaacLab
    ./isaaclab.sh -p \\
        /home/leesu37/AP-project/mechanical_search_setting/mechanical_search_isaac_sh/scripts/record_pile.py \\
        --headless --enable_cameras

Options:
    --num_steps     Total physics steps after settle  (default: 300)
    --record_every  Capture 1 frame every N steps     (default: 4)
    --settle        Run SETTLE_STEPS before recording (default: True)

Output:
    <project>/outputs/pile_rgb_YYYYMMDD_HHMMSS.mp4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI — must come before AppLauncher
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Record pile RGB video")
parser.add_argument("--num_steps",    type=int,            default=300)
parser.add_argument("--record_every", type=int,            default=4)
parser.add_argument("--no_settle",    action="store_true", help="Skip settle phase")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# DLSS가 활성화되면 렌더 해상도가 낮을 때 사선 줄무늬 테스트 패턴을 출력함.
# AppLauncher 직후에 DLSS를 비활성화해야 실제 렌더링이 캡처된다.
import carb as _carb  # noqa: E402
_carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)

# ---------------------------------------------------------------------------
# Post-AppLauncher imports
# ---------------------------------------------------------------------------
import datetime        # noqa: E402
import numpy as np     # noqa: E402
import imageio         # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)

from configs.scene_cfg import PileSceneCfg   # noqa: E402
from envs.pile_env     import PileEnv        # noqa: E402


# ---------------------------------------------------------------------------
# SyntheticData patch
# ---------------------------------------------------------------------------

def _patch_syntheticdata_dep() -> None:
    """Fix SyntheticData._add_node_downstream_intergraph_dependency TypeError.

    The error (kind=f/u, size=0) happens because the OmniGraph node is
    brand-new and its state attribute is uninitialized at attachment time.
    One simulation_app.update() call evaluates the OmniGraph so the
    attribute gets initialized, then a retry succeeds.

    The patch is applied before sim.reset(). camera.py applies its own
    patch during sim.reset(), storing ours as _orig_dep, so the retry
    path reaches the truly-original function through both layers.
    """
    try:
        from omni.syntheticdata.scripts.SyntheticData import SyntheticData as _SD
        _orig_dep = _SD._add_node_downstream_intergraph_dependency
        _app = simulation_app  # capture in closure

        @staticmethod
        def _user_safe_dep(node, downstream_node_handle):
            try:
                return _orig_dep(node, downstream_node_handle)
            except TypeError:
                # Node just created → attributes uninitialized.
                # One app.update() evaluates OmniGraph and initialises
                # state:_sdp_intergraph_downstream_node_handles_.
                _app.update()
                try:
                    return _orig_dep(node, downstream_node_handle)
                except TypeError:
                    return 1  # give up gracefully

        _SD._add_node_downstream_intergraph_dependency = _user_safe_dep
        print("[Patch] SyntheticData dependency patch applied.")
    except Exception as _e:
        print(f"[Patch][Warning] SyntheticData dependency patch failed: {_e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Simulation context ------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
    sim     = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, -1.5, 1.5], target=[0.5, 0.0, 0.2])

    # ---- Scene -------------------------------------------------------------
    scene_cfg = PileSceneCfg(num_envs=1, env_spacing=3.0)
    env = PileEnv(cfg=scene_cfg, sim=sim)

    # Same patch timing as run_occlusion_record.py: after env creation,
    # before sim.reset().
    _patch_syntheticdata_dep()

    # Flush the async SyntheticData render-var registration before PhysX starts.
    print("[Record] Warming up render pipeline…")
    for _ in range(30):
        simulation_app.update()

    # ---- PhysX initialisation ----------------------------------------------
    sim.reset()
    sim.step(render=False)  # complete PhysX view initialisation

    # ---- Settle pile -------------------------------------------------------
    env.reset(settle=not args_cli.no_settle)

    # ---- Recording loop ----------------------------------------------------
    print(f"[Record] Recording {args_cli.num_steps} steps…")
    rgb_frames: list[np.ndarray] = []

    for step in range(args_cli.num_steps):
        env.step()

        if step % args_cli.record_every == 0:
            rgb = env.get_rgb()                            # [1, H, W, 3]
            frame = rgb[0].cpu().numpy().astype(np.uint8)   # [H, W, 3]
            if frame.ndim == 3 and frame.shape[0] > 0:
                rgb_frames.append(frame)

    # ---- Save video --------------------------------------------------------
    if not rgb_frames:
        print("[Record] No frames captured — camera data was empty.")
        return

    now      = datetime.datetime.now()
    stamp    = now.strftime("%Y%m%d_%H%M%S")
    out_dir  = os.path.join(_proj_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    fps      = max(1, int(1.0 / (1 / 120) / args_cli.record_every))
    out_path = os.path.join(out_dir, f"pile_rgb_{stamp}.mp4")

    imageio.mimsave(out_path, rgb_frames, fps=fps)
    print(f"[Record] → {out_path}  ({len(rgb_frames)} frames, {fps} fps)")


if __name__ == "__main__":
    main()
    simulation_app.close()
