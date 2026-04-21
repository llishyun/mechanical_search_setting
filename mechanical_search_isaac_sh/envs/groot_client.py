"""GR00T socket client — Isaac Lab (env_isaaclab) 쪽에서 사용.

gr00t / tyro 의존성 없음. Python 기본 socket + pickle만 사용.
FrankaGR00TPolicy와 동일한 인터페이스 제공 (drop-in replacement).
"""

from __future__ import annotations

import pickle
import socket
import struct

import numpy as np

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5000


def _recvall(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly.")
        buf += chunk
    return buf


def _send_msg(conn: socket.socket, data: dict) -> None:
    payload = pickle.dumps(data, protocol=4)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_msg(conn: socket.socket) -> dict:
    raw_len = _recvall(conn, 4)
    msglen  = struct.unpack(">I", raw_len)[0]
    return pickle.loads(_recvall(conn, msglen))


class GR00TSocketClient:
    """Isaac Lab 측 GR00T 클라이언트.

    FrankaGR00TPolicy와 동일한 인터페이스:
        client.reset(instruction)
        action = client.get_action(exterior_rgb, wrist_rgb, eef_pos,
                                   eef_rotmat, gripper_pos, joint_pos)

    Returns per get_action():
        "joint_position"  : np.ndarray [40, 7] float32 — relative delta (rad)
        "gripper_position": np.ndarray [40, 1] float32 — absolute [0,1]
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
        print(f"[Client] Connected to GR00T server at {host}:{port}")

    def reset(self, instruction: str = "pick and place objects") -> None:
        _send_msg(self._sock, {"type": "reset", "instruction": instruction})
        resp = _recv_msg(self._sock)
        if resp.get("status") != "ok":
            raise RuntimeError(f"Server reset failed: {resp}")

    def get_action(
        self,
        exterior_rgb: np.ndarray,
        wrist_rgb:    np.ndarray,
        eef_pos:      np.ndarray,
        eef_rotmat:   np.ndarray,
        gripper_pos:  float,
        joint_pos:    np.ndarray,
    ) -> dict[str, np.ndarray]:
        """현재 관찰을 서버로 전송하고 action dict를 수신."""
        _send_msg(self._sock, {
            "type":         "step",
            "exterior_rgb": np.ascontiguousarray(exterior_rgb),
            "wrist_rgb":    np.ascontiguousarray(wrist_rgb),
            "eef_pos":      np.asarray(eef_pos,    dtype=np.float32),
            "eef_rotmat":   np.asarray(eef_rotmat, dtype=np.float32),
            "gripper_pos":  float(gripper_pos),
            "joint_pos":    np.asarray(joint_pos,  dtype=np.float32),
        })
        return _recv_msg(self._sock)

    def close(self) -> None:
        self._sock.close()
        print("[Client] Disconnected from GR00T server.")
