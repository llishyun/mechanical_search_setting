"""Run and record a Mechanical Search episode.

Usage:
    cd /home/leesu37/AP-project/IsaacLab
    ./isaaclab.sh -p \\
        /home/leesu37/AP-project/mechanical_search_setting/mechanical_search_isaac_sh/scripts/run_mechanical_search.py \\
        --headless --enable_cameras

Options:
    --num_episodes  Number of episodes to record          (default: 1)
    --num_steps     Physics steps to record per episode   (default: 300)
    --record_every  Capture 1 frame every N steps         (default: 4)
    --no_settle     Skip the pile-settle phase
    --seed          Base RNG seed (incremented per episode)
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Inject DLSS-disable flags BEFORE AppLauncher so the Kit renderer starts
# without DLSS.  carb.settings.set() after AppLauncher is too late — the
# renderer has already initialised with DLSS enabled at that point.
# ---------------------------------------------------------------------------
for _flag in ("--/rtx/post/aa/op=0", "--/rtx/post/dlss/execMode=0"):
    if _flag not in sys.argv:
        sys.argv.append(_flag)

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI — must be parsed before AppLauncher
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Mechanical Search recorder")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--num_steps",    type=int, default=300)
parser.add_argument("--record_every", type=int, default=4)
parser.add_argument("--no_settle",    action="store_true")
parser.add_argument("--seed",         type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
# parse_known_args: argparse가 모르는 --/rtx/... Kit 인자를 에러 없이 무시.
# 해당 인자들은 sys.argv에 남아 있어 Kit 앱이 직접 처리한다.
args_cli, _ = parser.parse_known_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Disable DLSS immediately after AppLauncher.
# execMode=0 alone is NOT enough — it sets DLSS to "Auto" mode, not Off.
# Setting aa/op=0 turns off the AA pipeline entirely, preventing DLSS from
# running and producing the diagonal stripe test pattern in headless renders.
# ---------------------------------------------------------------------------
import carb as _carb  # noqa: E402
_s = _carb.settings.get_settings()
_s.set("/rtx/post/dlss/execMode", 0)           # DLSS quality mode: off
_s.set("/rtx/post/aa/op", 0)                  # AA pipeline: None (key fix for stripe pattern)
# Scene space effects — not related to the stripe pattern, but can cause
# blown-out / dark / color-shifted frames in headless camera captures.
_s.set("/rtx/post/histogram/enabled", False)   # auto-exposure off
_s.set("/rtx/post/lensFlares/enabled", False)  # lens flare off
_s.set("/rtx/post/motionblur/enabled", False)  # motion blur off (objects are slow)

# ---------------------------------------------------------------------------
# Post-AppLauncher imports
# ---------------------------------------------------------------------------
import datetime        # noqa: E402
import json            # noqa: E402

import numpy as np     # noqa: E402
import imageio         # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)

from configs.scene_cfg import MechSearchEnvCfg                   # noqa: E402
from envs.mechanical_search_env import MechanicalSearchEnv       # noqa: E402


# ---------------------------------------------------------------------------
# SyntheticData patch
# ---------------------------------------------------------------------------

_MAX_RETRIES = 10  # app.update() attempts before giving up

def _patch_syntheticdata() -> None:
    """Retry on TypeError during OmniGraph node initialisation.

    The TypeError (kind=f, size=0) means a freshly-created OmniGraph node has
    uninitialised float attributes.  Each app.update() evaluates the graph and
    may initialise those attributes.  We retry up to _MAX_RETRIES times.

    Returning 1 on true failure fakes 'success' to the caller — the annotator
    is registered but its render-product connection is BROKEN, which is why
    the camera outputs the DLSS stripe test pattern instead of scene content.
    We therefore must succeed here, not just swallow the error.
    """
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

    # ── Simulation context ──────────────────────────────────────────────
    env_cfg = MechSearchEnvCfg()
    sim_cfg = env_cfg.sim
    sim     = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, -1.5, 1.5], target=[0.5, 0.0, 0.2])

    # ── Create scene (registers camera annotators) ──────────────────────
    env = MechanicalSearchEnv(cfg=env_cfg.scene, sim=sim)

    # ── Apply patch BEFORE sim.reset() ──────────────────────────────────
    _patch_syntheticdata()

    # ── Warm-up: let the annotation system register render vars ─────────
    # Must happen before sim.reset() so semantic annotators are ready.
    print("[Setup] Warming up render pipeline (5 frames)…")
    for _ in range(5):
        simulation_app.update()

    # ── PhysX initialisation ─────────────────────────────────────────────
    sim.reset()
    sim.step(render=False)  # completes PhysX view initialisation

    # ── Episode loop ─────────────────────────────────────────────────────
    episode_log: list[dict] = []

    for ep in range(args_cli.num_episodes):
        seed = None if args_cli.seed is None else args_cli.seed + ep

        print(f"\n[Episode {ep}] Resetting scene…")
        target_name = env.reset(seed=seed, settle=False)
        print(f"[Episode {ep}] Target object: {target_name}")

        # A few render steps after reset so the RTX buffer is populated
        # before we start capturing frames.
        for _ in range(5):
            env.step(render=True)

        rgb_frames: list[np.ndarray] = []

        # ── Settle loop (recorded) ────────────────────────────────────
        if not args_cli.no_settle:
            from configs.scene_cfg import SETTLE_STEPS
            print(f"[Episode {ep}] Settling ({SETTLE_STEPS} steps, recording every {args_cli.record_every})…")
            for step in range(SETTLE_STEPS):
                env.step(render=True)
                if step % args_cli.record_every == 0:
                    obs   = env.get_obs()
                    frame = obs["overhead_rgb"][0].cpu().numpy().astype(np.uint8)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        rgb_frames.append(frame)

        # ── Recording loop ────────────────────────────────────────────
        for step in range(args_cli.num_steps):
            # render=True every step ensures the camera buffer is always fresh;
            # set to False if you only need frames at record_every intervals and
            # want to reduce rendering overhead.
            env.step(render=True)

            if step % args_cli.record_every == 0:
                obs   = env.get_obs()
                frame = obs["overhead_rgb"][0].cpu().numpy().astype(np.uint8)
                if frame.ndim == 3 and frame.shape[2] == 3:
                    rgb_frames.append(frame)

        # ── Save video ────────────────────────────────────────────────
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ep_stamp = f"ep{ep:03d}_{stamp}"

        if rgb_frames:
            fps      = max(1, int((1.0 / (1.0 / 120)) / args_cli.record_every))
            vid_path = os.path.join(out_dir, f"mechanical_search_{ep_stamp}.mp4")
            imageio.mimsave(vid_path, rgb_frames, fps=fps)
            print(f"[Episode {ep}] Video → {vid_path}  ({len(rgb_frames)} frames, {fps} fps)")
        else:
            print(f"[Episode {ep}] No frames captured.")

        # ── Save target name ──────────────────────────────────────────
        episode_log.append({
            "episode":     ep,
            "target_idx":  env.target_idx,
            "target_name": target_name,
            "seed":        seed,
        })

    # ── Write episode log (target names etc.) ────────────────────────────
    log_path = os.path.join(out_dir, f"episode_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump(episode_log, f, indent=2)
    print(f"\n[Done] Episode log → {log_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
