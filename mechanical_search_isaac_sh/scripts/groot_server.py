"""GR00T-N1.7 inference server — runs in env_groot / uv run context.

Isaac Lab과 환경이 충돌하므로 별도 프로세스로 분리.
ring buffer, policy state 등 GR00T 관련 상태를 모두 이쪽에서 관리.

Usage (터미널 1):
    # uv 사용 시:
    cd /home/leesu37/AP-project/Isaac-GR00T_N1.7
    uv run python \\
        /home/leesu37/AP-project/mechanical_search_setting/mechanical_search_isaac_sh/scripts/groot_server.py

    # conda env_groot 사용 시:
    conda activate env_groot
    python \\
        /home/leesu37/AP-project/mechanical_search_setting/mechanical_search_isaac_sh/scripts/groot_server.py \\
        --port 5000 --device cuda:0

Protocol (request-response, length-prefixed pickle):
    reset → {"type":"reset", "instruction":str}
          ← {"status":"ok"}
    step  → {"type":"step", "exterior_rgb":ndarray, "wrist_rgb":ndarray,
              "eef_pos":ndarray, "eef_rotmat":ndarray,
              "gripper_pos":float, "joint_pos":ndarray}
          ← {"joint_position":ndarray[40,7], "gripper_position":ndarray[40,1]}
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys

# Isaac-GR00T_N1.7 패키지 경로 추가 (gr00t 임포트용)
sys.path.insert(0, "/home/leesu37/AP-project/Isaac-GR00T_N1.7")

# groot_module.py 임포트를 위해 프로젝트 루트 추가
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)

from envs.groot_module import FrankaGR00TPolicy  # noqa: E402

DEFAULT_PORT = 5000


# ---------------------------------------------------------------------------
# Socket helpers (length-prefixed: 4-byte big-endian uint32 + pickle payload)
# ---------------------------------------------------------------------------

def _recvall(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly.")
        buf += chunk
    return buf


def send_msg(conn: socket.socket, data: dict) -> None:
    payload = pickle.dumps(data, protocol=4)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def recv_msg(conn: socket.socket) -> dict:
    raw_len = _recvall(conn, 4)
    msglen  = struct.unpack(">I", raw_len)[0]
    return pickle.loads(_recvall(conn, msglen))


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GR00T inference server")
    parser.add_argument("--port",   type=int, default=DEFAULT_PORT,
                        help="TCP port to listen on (default: 5000)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device for inference (default: cuda:0)")
    args = parser.parse_args()

    print(f"[Server] Loading FrankaGR00TPolicy on {args.device}...")
    policy = FrankaGR00TPolicy(device=args.device)
    print("[Server] Policy ready.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("localhost", args.port))
        srv.listen(1)
        print(f"[Server] Listening on localhost:{args.port} ...")

        # 클라이언트 재접속을 허용하기 위해 outer while loop
        while True:
            conn, addr = srv.accept()
            print(f"[Server] Client connected: {addr}")
            with conn:
                try:
                    while True:
                        msg = recv_msg(conn)

                        if msg["type"] == "reset":
                            instruction = msg.get("instruction", "pick and place objects")
                            policy.reset(instruction=instruction)
                            send_msg(conn, {"status": "ok"})
                            print(f"[Server] Reset. Instruction: '{instruction}'")

                        elif msg["type"] == "step":
                            action = policy.get_action(
                                exterior_rgb=msg["exterior_rgb"],   # [H, W, 3] uint8
                                eef_pos=msg["eef_pos"],             # [3] float32
                                eef_rotmat=msg["eef_rotmat"],       # [3,3] float32
                                gripper_pos=msg["gripper_pos"],     # float
                                joint_pos=msg["joint_pos"],         # [7] float32
                                wrist_rgb=msg.get("wrist_rgb"),     # [H, W, 3] uint8 or None
                            )
                            # action["joint_position"]  : [1, 40, 7]
                            # action["gripper_position"]: [1, 40, 1]
                            send_msg(conn, {
                                "joint_position":   action["joint_position"][0],    # [40, 7]
                                "gripper_position": action["gripper_position"][0],  # [40, 1]
                            })

                        else:
                            send_msg(conn, {"error": f"Unknown message type: {msg['type']}"})

                except ConnectionError as e:
                    print(f"[Server] Client disconnected: {e}")


if __name__ == "__main__":
    main()
