"""Mechanical Search algorithm for Isaac Lab occlusion environment.

Priority list generation (정보 1 기반):
    - "random":        Preempted Random  — target first, rest randomly shuffled
    - "largest_first": Largest-First     — target first, rest sorted by
                                           visible pixel count (descending)

Push policy  : Signed Distance Transform 방식 (논문 방식)
Grasp policy : overhead segmentation mask centroid → depth deproject → top-down grasp
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from occlusion_env_cfg import _OBJECT_USDS

# 영상 오버레이용 간단한 5x7 비트맵 폰트 (ASCII 32-95)
# 각 문자: 7행 × 5비트 (MSB = 왼쪽)
_FONT: dict[int, dict[int, int]] = {
    # Space
    32: {},
    # 숫자
    48: {0:0x1C,1:0x22,2:0x26,3:0x2A,4:0x32,5:0x22,6:0x1C},  # 0
    49: {0:0x08,1:0x18,2:0x08,3:0x08,4:0x08,5:0x08,6:0x1C},  # 1
    50: {0:0x1C,1:0x22,2:0x02,3:0x0C,4:0x10,5:0x20,6:0x3E},  # 2
    51: {0:0x3E,1:0x02,2:0x04,3:0x0C,4:0x02,5:0x22,6:0x1C},  # 3
    52: {0:0x04,1:0x0C,2:0x14,3:0x24,4:0x3E,5:0x04,6:0x04},  # 4
    53: {0:0x3E,1:0x20,2:0x3C,3:0x02,4:0x02,5:0x22,6:0x1C},  # 5
    54: {0:0x0C,1:0x10,2:0x20,3:0x3C,4:0x22,5:0x22,6:0x1C},  # 6
    55: {0:0x3E,1:0x02,2:0x04,3:0x08,4:0x10,5:0x10,6:0x10},  # 7
    56: {0:0x1C,1:0x22,2:0x22,3:0x1C,4:0x22,5:0x22,6:0x1C},  # 8
    57: {0:0x1C,1:0x22,2:0x22,3:0x1E,4:0x02,5:0x04,6:0x18},  # 9
    # 콜론
    58: {1:0x08,2:0x00,4:0x08,5:0x00},
    # 대문자
    65: {0:0x08,1:0x14,2:0x22,3:0x3E,4:0x22,5:0x22,6:0x22},  # A
    66: {0:0x3C,1:0x22,2:0x22,3:0x3C,4:0x22,5:0x22,6:0x3C},  # B
    67: {0:0x1C,1:0x22,2:0x20,3:0x20,4:0x20,5:0x22,6:0x1C},  # C
    68: {0:0x3C,1:0x22,2:0x22,3:0x22,4:0x22,5:0x22,6:0x3C},  # D
    69: {0:0x3E,1:0x20,2:0x20,3:0x3C,4:0x20,5:0x20,6:0x3E},  # E
    70: {0:0x3E,1:0x20,2:0x20,3:0x3C,4:0x20,5:0x20,6:0x20},  # F
    71: {0:0x1C,1:0x22,2:0x20,3:0x2E,4:0x22,5:0x22,6:0x1E},  # G
    72: {0:0x22,1:0x22,2:0x22,3:0x3E,4:0x22,5:0x22,6:0x22},  # H
    73: {0:0x1C,1:0x08,2:0x08,3:0x08,4:0x08,5:0x08,6:0x1C},  # I
    74: {0:0x02,1:0x02,2:0x02,3:0x02,4:0x22,5:0x22,6:0x1C},  # J
    75: {0:0x22,1:0x24,2:0x28,3:0x30,4:0x28,5:0x24,6:0x22},  # K
    76: {0:0x20,1:0x20,2:0x20,3:0x20,4:0x20,5:0x20,6:0x3E},  # L
    77: {0:0x22,1:0x36,2:0x2A,3:0x22,4:0x22,5:0x22,6:0x22},  # M
    78: {0:0x22,1:0x32,2:0x2A,3:0x26,4:0x22,5:0x22,6:0x22},  # N
    79: {0:0x1C,1:0x22,2:0x22,3:0x22,4:0x22,5:0x22,6:0x1C},  # O
    80: {0:0x3C,1:0x22,2:0x22,3:0x3C,4:0x20,5:0x20,6:0x20},  # P
    81: {0:0x1C,1:0x22,2:0x22,3:0x22,4:0x2A,5:0x24,6:0x1A},  # Q
    82: {0:0x3C,1:0x22,2:0x22,3:0x3C,4:0x28,5:0x24,6:0x22},  # R
    83: {0:0x1C,1:0x22,2:0x20,3:0x1C,4:0x02,5:0x22,6:0x1C},  # S
    84: {0:0x3E,1:0x08,2:0x08,3:0x08,4:0x08,5:0x08,6:0x08},  # T
    85: {0:0x22,1:0x22,2:0x22,3:0x22,4:0x22,5:0x22,6:0x1C},  # U
    86: {0:0x22,1:0x22,2:0x22,3:0x22,4:0x22,5:0x14,6:0x08},  # V
    87: {0:0x22,1:0x22,2:0x22,3:0x2A,4:0x2A,5:0x36,6:0x22},  # W
    88: {0:0x22,1:0x22,2:0x14,3:0x08,4:0x14,5:0x22,6:0x22},  # X
    89: {0:0x22,1:0x22,2:0x14,3:0x08,4:0x08,5:0x08,6:0x08},  # Y
    90: {0:0x3E,1:0x02,2:0x04,3:0x08,4:0x10,5:0x20,6:0x3E},  # Z
    # 언더스코어
    95: {6:0x3E},
    # 기타 (fallback = ?)
    63: {0:0x1C,1:0x22,2:0x02,3:0x04,4:0x08,5:0x00,6:0x08},  # ?
}

# class_name (semantic tag) → pile_name 매핑
# _make_pile_cfg: semantic_tags=[("class", obj_name)] 에서 obj_name = usd_path.split("/")[-2]
CLASS_TO_PILE: dict[str, str] = {
    _OBJECT_USDS[i].split("/")[-2]: f"pile_{i:02d}" for i in range(20)
}
# Isaac Lab이 소문자로 내려주므로 lowercase 버전도 준비
CLASS_TO_PILE_LOWER: dict[str, str] = {k.lower(): v for k, v in CLASS_TO_PILE.items()}

# SDT push 파라미터
PUSH_OFFSET_PX = 50       # object COM 뒤쪽 push 시작점 (pixel 단위)
POST_PUSH_SETTLE = 120     # push 후 물리 안정화 step 수


# ---------------------------------------------------------------------------
# Priority list
# ---------------------------------------------------------------------------

def generate_priority_list_random(
    visible_objects: list[str],
    target_name: str,
) -> list[str]:
    """Preempted Random: target 먼저, 나머지 무작위 셔플."""
    others = [o for o in visible_objects if o != target_name]
    random.shuffle(others)
    if target_name in visible_objects:
        return [target_name] + others
    return others


def generate_priority_list_largest_first(
    visible_objects: list[str],
    visible_areas: dict[str, int],
    target_name: str,
) -> list[str]:
    """Largest-First: target 먼저, 나머지 visible pixel 수 내림차순."""
    others = [o for o in visible_objects if o != target_name]
    others.sort(key=lambda o: visible_areas.get(o, 0), reverse=True)
    if target_name in visible_objects:
        return [target_name] + others
    return others


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def get_visible_areas(camera, dt: float, env_idx: int = 0) -> dict[str, int]:
    """overhead 카메라 semantic segmentation에서 pile별 pixel 수 계산.

    visible_area 정의: segmentation mask 에서 해당 object의 pixel 수.

    Returns:
        {pile_name: pixel_count} — pixel > 0 인 오브젝트만 포함
    """
    camera.update(dt=dt)
    seg_rgba, color_to_pile = _parse_seg(camera, env_idx)

    visible: dict[str, int] = {}
    for color, pile_name in color_to_pile.items():
        r, g, b, a = color
        mask = (
            (seg_rgba[:, :, 0] == r) &
            (seg_rgba[:, :, 1] == g) &
            (seg_rgba[:, :, 2] == b) &
            (seg_rgba[:, :, 3] == a)
        )
        count = int(mask.sum())
        if count > 0:
            visible[pile_name] = count
    return visible


def debug_seg_output(camera, env_idx: int = 0):
    """segmentation 출력 구조 디버그 출력."""
    seg_raw = camera.data.output["semantic_segmentation"]
    print(f"[DEBUG] seg_raw type={type(seg_raw).__name__}, shape={seg_raw.shape}")
    try:
        env_info = camera.data.info[env_idx]
        seg_info = env_info.get("semantic_segmentation", {})
        labels = seg_info.get("idToLabels", seg_info)
        print(f"[DEBUG] idToLabels entries ({len(labels)}): {dict(list(labels.items())[:5])} ...")
    except Exception as e:
        print(f"[DEBUG] info 접근 실패: {e}")


def _parse_seg(camera, env_idx: int):
    """segmentation 파싱 → (seg_rgba [H,W,4] uint8 numpy, color_to_pile dict).

    Isaac Lab의 실제 포맷 (확인됨):
        camera.data.output["semantic_segmentation"]  : Tensor [N, H, W, 4] RGBA uint8
        camera.data.info[env_idx]["semantic_segmentation"]["idToLabels"]
            : {'(R,G,B,A)': {'class': 'lowercase_name'}, ...}

    color_to_pile: {(R,G,B,A): pile_name}
    """
    # --- RGBA seg 이미지 ---
    seg_raw = camera.data.output["semantic_segmentation"]   # [N, H, W, 4]
    seg_rgba = seg_raw[env_idx].cpu().numpy()               # [H, W, 4] uint8

    # --- idToLabels 파싱 ---
    color_to_pile: dict[tuple, str] = {}
    try:
        env_info   = camera.data.info[env_idx]
        seg_info   = env_info["semantic_segmentation"]
        id_to_labels = seg_info.get("idToLabels", seg_info)   # 키 이름 두 가지 대응
    except (AttributeError, KeyError, TypeError, IndexError):
        id_to_labels = {}

    for color_str, meta in id_to_labels.items():
        class_name = meta.get("class", "") if isinstance(meta, dict) else str(meta)
        # 소문자로 비교 (Isaac Lab이 lowercase로 내려줌)
        pile_name = CLASS_TO_PILE_LOWER.get(class_name.lower())
        if pile_name is None:
            continue
        # '(R, G, B, A)' 문자열 → tuple
        try:
            color_tuple = tuple(int(x.strip()) for x in color_str.strip("()").split(","))
            if len(color_tuple) == 4:
                color_to_pile[color_tuple] = pile_name
        except ValueError:
            pass

    return seg_rgba, color_to_pile


def _get_depth_np(camera, env_idx: int) -> np.ndarray:
    """depth 이미지 → (H, W) float32 numpy array (단위: meter)."""
    depth_raw = camera.data.output["depth"]
    if depth_raw.dim() == 4:
        return depth_raw[env_idx, :, :, 0].cpu().numpy().astype(np.float32)
    return depth_raw[env_idx].cpu().numpy().astype(np.float32)


def _deproject(u: int, v: int, depth_np: np.ndarray, K: np.ndarray,
               cam_pos: np.ndarray, cam_quat_wxyz: np.ndarray) -> np.ndarray | None:
    """pixel (u, v) + depth → world 3D 좌표 (numpy [3]).

    Args:
        u, v         : pixel column, row
        depth_np     : (H, W) depth map in meters
        K            : (3, 3) intrinsic matrix
        cam_pos      : (3,) camera world position
        cam_quat_wxyz: (4,) camera quaternion (w, x, y, z) in world frame
    """
    d = float(depth_np[v, u])
    if not np.isfinite(d) or d <= 0.0:
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # camera frame
    x_c = (u - cx) * d / fx
    y_c = (v - cy) * d / fy
    z_c = d

    # quaternion → rotation matrix
    w, qx, qy, qz = cam_quat_wxyz
    R = np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*w),   2*(qx*qz + qy*w)],
        [2*(qx*qy + qz*w),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*w)],
        [2*(qx*qz - qy*w),   2*(qy*qz + qx*w),   1 - 2*(qx**2 + qy**2)],
    ])
    return R @ np.array([x_c, y_c, z_c]) + cam_pos


def _get_cam_info(camera, env_idx: int):
    """카메라 intrinsic + extrinsic 반환."""
    K = camera.data.intrinsic_matrices[env_idx].cpu().numpy()
    cam_pos = camera.data.pos_w[env_idx].cpu().numpy()
    cam_quat = camera.data.quat_w_world[env_idx].cpu().numpy()   # (w,x,y,z)
    return K, cam_pos, cam_quat


# ---------------------------------------------------------------------------
# Grasp policy: segmentation mask centroid → depth deproject
# ---------------------------------------------------------------------------

def _get_obj_mask(seg_rgba: np.ndarray, color_to_pile: dict, obj_name: str) -> np.ndarray | None:
    """obj_name에 해당하는 RGBA 마스크 반환 (없으면 None)."""
    for color, pile_name in color_to_pile.items():
        if pile_name != obj_name:
            continue
        r, g, b, a = color
        mask = (
            (seg_rgba[:, :, 0] == r) &
            (seg_rgba[:, :, 1] == g) &
            (seg_rgba[:, :, 2] == b) &
            (seg_rgba[:, :, 3] == a)
        )
        if mask.any():
            return mask
    return None


def compute_grasp_point(camera, env, obj_name: str, env_idx: int,
                        device) -> torch.Tensor:
    """target object grasp 위치 계산 → [num_envs, 3] world tensor.

    방법:
        1. RGBA segmentation mask에서 target pixel 추출
        2. centroid (u, v) 계산
        3. depth로 deproject → world 3D
        4. 실패 시 physics root_pos_w로 fallback
    """
    seg_rgba, color_to_pile = _parse_seg(camera, env_idx)
    mask = _get_obj_mask(seg_rgba, color_to_pile, obj_name)

    if mask is not None:
        rows, cols = np.where(mask)
        v, u = int(rows.mean()), int(cols.mean())

        depth_np = _get_depth_np(camera, env_idx)
        K, cam_pos, cam_quat = _get_cam_info(camera, env_idx)
        p_world = _deproject(u, v, depth_np, K, cam_pos, cam_quat)

        if p_world is not None:
            pos = torch.tensor(p_world, dtype=torch.float32, device=device)
            return pos.unsqueeze(0).expand(env.scene.num_envs, -1).clone()

    # fallback: physics position
    return env.scene[obj_name].data.root_pos_w


# ---------------------------------------------------------------------------
# Push policy: Signed Distance Transform (논문 방식)
# ---------------------------------------------------------------------------

def compute_push_points_sdt(camera, env, obj_name: str, env_idx: int,
                             device) -> tuple[torch.Tensor | None,
                                              torch.Tensor | None,
                                              bool]:
    """SDT 기반 push 시작점(p)과 끝점(p') 계산.

    논문 알고리즘:
        1. overhead seg에서 binary obstacle mask 생성 (오브젝트 + bin 벽)
        2. distance_transform_edt → 가장 빈 공간의 픽셀 p' 찾기
        3. p → p' 직선이 object COM을 통과하도록 p 계산
        4. 2D pixel → depth deproject → 3D world 좌표

    Returns:
        (p_start, p_end, valid): [num_envs, 3] tensors, bool
    """
    seg_rgba, color_to_pile = _parse_seg(camera, env_idx)
    H, W = seg_rgba.shape[:2]

    # object RGBA 마스크
    obj_mask = _get_obj_mask(seg_rgba, color_to_pile, obj_name)
    if obj_mask is None:
        return None, None, False

    rows, cols = np.where(obj_mask)
    v_com, u_com = int(rows.mean()), int(cols.mean())

    # --- SDT: 가장 빈 공간 p' 찾기 ---
    # obstacle = alpha==255 인 픽셀 (오브젝트 + bin 경계), background는 alpha==0
    obstacle_mask = (seg_rgba[:, :, 3] == 255)
    # free 픽셀에서 가장 가까운 obstacle까지의 거리
    sdt = distance_transform_edt(~obstacle_mask)
    # 거리가 최대인 픽셀 = 가장 빈 공간
    p_prime_flat = int(sdt.argmax())
    v_prime, u_prime = divmod(p_prime_flat, W)

    # --- p_start: COM 기준으로 p'의 반대 방향에 PUSH_OFFSET_PX만큼 ---
    dv = v_com - v_prime
    du = u_com - u_prime
    norm = np.sqrt(dv**2 + du**2) + 1e-6
    dv_n, du_n = dv / norm, du / norm

    v_start = int(np.clip(v_com + dv_n * PUSH_OFFSET_PX, 0, H - 1))
    u_start = int(np.clip(u_com + du_n * PUSH_OFFSET_PX, 0, W - 1))

    # --- 2D → 3D deproject (depth 기반) ---
    depth_np = _get_depth_np(camera, env_idx)
    K, cam_pos, cam_quat = _get_cam_info(camera, env_idx)

    p_end_arr = _deproject(u_prime, v_prime, depth_np, K, cam_pos, cam_quat)
    p_start_arr = _deproject(u_start, v_start, depth_np, K, cam_pos, cam_quat)

    # --- fallback: depth deproject 실패 시 physics position 기반 계산 ---
    obj_pos_w = env.scene[obj_name].data.root_pos_w[env_idx].cpu().numpy()  # [3]
    obj_z = float(obj_pos_w[2])

    if p_end_arr is None or p_start_arr is None:
        print(f"  [WARN] depth deproject 실패 → physics position fallback 사용")
        p_end_arr, p_start_arr = _physics_push_fallback(obj_pos_w)

    # 테이블 범위 클리핑 (바닥 픽셀 deproject 방지)
    # 테이블: X [-0.1, 1.1], Y [-0.3, 0.3] (벽 안쪽 약간 여유)
    _TX = (-0.08, 1.08)
    _TY = (-0.58, 0.58)
    p_end_arr[0]   = np.clip(p_end_arr[0],   *_TX)
    p_end_arr[1]   = np.clip(p_end_arr[1],   *_TY)
    p_start_arr[0] = np.clip(p_start_arr[0], *_TX)
    p_start_arr[1] = np.clip(p_start_arr[1], *_TY)

    # push z는 오브젝트 실제 물리 높이로 덮어쓰기 (depth보다 안정적)
    p_start_arr[2] = obj_z
    p_end_arr[2]   = obj_z

    print(f"  [PUSH] p_start={p_start_arr.round(3)}  p_end={p_end_arr.round(3)}")

    N = env.scene.num_envs
    p_start = torch.tensor(p_start_arr, dtype=torch.float32, device=device)
    p_end   = torch.tensor(p_end_arr,   dtype=torch.float32, device=device)

    return (
        p_start.unsqueeze(0).expand(N, -1).clone(),
        p_end.unsqueeze(0).expand(N, -1).clone(),
        True,
    )


# 파일 전역 상수 (fallback push 거리)
_PILE_CENTER = np.array([0.5, 0.0])
_PUSH_DISTANCE = 0.25   # meters


def _physics_push_fallback(obj_pos_w: np.ndarray):
    """depth deproject 실패 시 physics position 기반으로 push 좌표 계산.

    push 방향: 오브젝트 → pile center 반대 방향 (= 더미 바깥)
    p_start   = 오브젝트 반대편 (오브젝트 뒤에서 밀기 시작)
    p_end     = pile center 반대편 (오브젝트를 밀어낼 목표)
    """
    obj_xy = obj_pos_w[:2]
    direction = obj_xy - _PILE_CENTER
    norm = np.linalg.norm(direction) + 1e-6
    direction = direction / norm

    p_end_arr   = np.array([obj_xy[0] + direction[0] * _PUSH_DISTANCE,
                             obj_xy[1] + direction[1] * _PUSH_DISTANCE,
                             obj_pos_w[2]])
    p_start_arr = np.array([obj_xy[0] - direction[0] * 0.10,
                             obj_xy[1] - direction[1] * 0.10,
                             obj_pos_w[2]])
    return p_end_arr, p_start_arr


# ---------------------------------------------------------------------------
# MechanicalSearch
# ---------------------------------------------------------------------------

class MechanicalSearch:
    """Mechanical Search 탐색 정책.

    매 step:
        1. overhead camera segmentation 관찰
        2. target이 보이면 → grasp → SUCCESS
        3. 안 보이면 → priority list 생성 → 최우선 non-target 오브젝트 push → settle
        4. max_actions 초과 시 → FAILURE

    Args:
        env:         OcclusionEnv instance
        ctrl:        RobotController instance
        method:      "random" | "largest_first"
        frame_buffer: 녹화 시 프레임을 쌓을 list (None이면 녹화 안 함)
        record_every: N physics step마다 1프레임 캡처
    """

    def __init__(
        self,
        env,
        ctrl,
        method: Literal["random", "largest_first"] = "random",
        frame_buffer: list | None = None,
        record_every: int = 4,
    ):
        self.env = env
        self.ctrl = ctrl
        self.method = method
        self._device = env.sim.device
        self._dt = env.sim.get_physics_dt()
        self._frame_buffer = frame_buffer
        self._record_every = record_every
        self._step_count = 0        # 전체 physics step 카운터
        self._target_name: str | None = None   # run() 시작 시 설정

        # 녹화 활성화 시 ctrl의 on_step 콜백 등록
        if frame_buffer is not None:
            self.ctrl.on_step = self._capture_frame

    def run(
        self,
        target_name: str,
        max_actions: int = 20,
        env_idx: int = 0,
        target_object_name: str | None = None,
    ) -> dict:
        """Mechanical Search 1 에피소드 실행.

        Args:
            target_name: 목표 오브젝트 pile name (예: "pile_03")
            max_actions: 최대 push 횟수
            env_idx:     사용할 환경 인덱스 (single-env이면 0)

        Returns:
            {
                "success":   bool,
                "n_actions": int,
                "history":   list[dict],
            }
        """
        self._target_name = target_name
        self._target_object_name = target_object_name
        self.env.reset(settle=True)

        history: list[dict] = []
        n_actions = 0

        for step in range(max_actions + 1):

            # ---- 관찰 ----
            self.env.camera.update(dt=self._dt)

            visible_areas = get_visible_areas(self.env.camera, self._dt, env_idx)
            visible_objects = list(visible_areas.keys())
            target_visible = target_name in visible_objects

            # ---- priority list 생성 ----
            if self.method == "random":
                priority_list = generate_priority_list_random(
                    visible_objects, target_name
                )
            else:
                priority_list = generate_priority_list_largest_first(
                    visible_objects, visible_areas, target_name
                )

            history.append({
                "step":           step,
                "visible_areas":  dict(visible_areas),
                "priority_list":  list(priority_list),
                "target_visible": target_visible,
            })

            print(
                f"[Step {step:02d}] target_visible={target_visible} | "
                f"method={self.method} | visible={len(visible_objects)} objs"
            )
            print(f"         priority={priority_list[:5]}")

            # ---- target 발견 → grasp ----
            if target_visible:
                print(f"  → '{target_name}' 발견! Grasping...")
                grasp_pos = compute_grasp_point(
                    self.env.camera, self.env, target_name, env_idx, self._device
                )
                self.ctrl.execute_grasp(grasp_pos)
                return {"success": True, "n_actions": n_actions, "history": history}

            if step == max_actions:
                break

            if not priority_list:
                print("  → visible 오브젝트 없음. 중단.")
                break

            # ---- push 실행 ----
            obj_to_push = priority_list[0]   # target 미발견이므로 non-target
            area = visible_areas.get(obj_to_push, 0)
            print(f"  → Push '{obj_to_push}' (area={area} px) [SDT policy]")

            p_start, p_end, valid = compute_push_points_sdt(
                self.env.camera, self.env, obj_to_push, env_idx, self._device
            )

            if valid:
                self.ctrl.execute_push(p_start, p_end)
            else:
                print(f"     SDT 실패. '{obj_to_push}' 건너뜀.")

            n_actions += 1
            self._settle(POST_PUSH_SETTLE)

        return {"success": False, "n_actions": n_actions, "history": history}

    def _capture_frame(self):
        """매 physics step마다 ctrl.on_step 콜백으로 호출됨."""
        self._step_count += 1
        if self._frame_buffer is None:
            return
        if self._step_count % self._record_every != 0:
            return
        self.env.camera.update(dt=self._dt)
        rgb = self.env.camera.data.output["rgb"]        # [N, H, W, 4]
        frame = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8).copy()
        frame = self._overlay_target(frame)
        self._frame_buffer.append(frame)

    def _overlay_target(self, frame: np.ndarray) -> np.ndarray:
        """target object 픽셀을 초록색으로 하이라이트한 프레임 반환."""
        if self._target_name is None or self._frame_buffer is None:
            return frame

        seg_rgba, color_to_pile = _parse_seg(self.env.camera, 0)
        target_mask = _get_obj_mask(seg_rgba, color_to_pile, self._target_name)

        if target_mask is not None:
            # 초록색 반투명 오버레이 (alpha=0.5)
            overlay = frame.copy()
            overlay[target_mask] = [0, 220, 0]
            frame = (frame * 0.5 + overlay * 0.5).astype(np.uint8)

        # 상단에 target 이름 표시
        # 오브젝트 이름이 있으면 2줄, 없으면 1줄
        line1 = f"TARGET: {self._target_name}".upper()
        line2 = self._target_object_name.upper() if self._target_object_name else None
        bar_h = 20 if line2 is None else 32
        frame[:bar_h, :] = 0

        def _draw_text(text: str, y_offset: int):
            x_offset = 4
            for ch in text:
                code = ord(ch)
                for bit_row in range(8):
                    byte = _FONT.get(code, _FONT[63]).get(bit_row, 0)
                    for bit_col in range(6):
                        if byte & (1 << (5 - bit_col)):
                            px = x_offset + bit_col
                            py = y_offset + bit_row
                            if 0 <= px < frame.shape[1] and 0 <= py < bar_h:
                                frame[py, px] = [255, 255, 255]
                x_offset += 7

        _draw_text(line1, y_offset=1)
        if line2 is not None:
            _draw_text(line2, y_offset=13)

        return frame

    def _settle(self, n_steps: int):
        for _ in range(n_steps):
            self.env.sim.step()
            self.env.scene.update(self._dt)
            self._step_count += 1
            if self._frame_buffer is not None and self._step_count % self._record_every == 0:
                self.env.camera.update(dt=self._dt)
                rgb = self.env.camera.data.output["rgb"]
                frame = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8).copy()
                frame = self._overlay_target(frame)
                self._frame_buffer.append(frame)
