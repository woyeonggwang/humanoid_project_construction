"""VLM 장면 배치 도우미: 받침대·상자·로봇을 정해진 배치대로 옮기고, 조명을 바꾸고, 머리 카메라 사진을 읽는다.

scripts/make_vlm_data.py (데이터 생성)와 scripts/vlm_pick_place_demo.py (연결 데모)가 같이 쓴다.
isaaclab 을 import 하므로 Isaac Sim 앱이 켜진 뒤에만 import 한다.

배치(layout) 형식 — 환경 하나에 dict 하나:
  {"boxes": {"stand_L": "red" 또는 None, "stand_C": ..., "stand_R": ...},   어느 받침대에 어떤 색 상자
   "stands": {"stand_L": (dx, dy, yaw), ...},                                받침대의 기본 위치 대비 이동(m)·방향(rad). 없으면 0
   "box_offset": {"stand_L": (dx, dy, yaw), ...},                            상자의 받침대 대비 이동·방향. 없으면 0
   "box_z": {"red": z, ...},                                                 상자 중심 높이 (상자 크기를 바꿨을 때). 없으면 기본 크기 기준
   "robot": (x, y, z, roll, pitch, yaw)}                                     로봇 골반 자세 (몸 고정일 때만). 없으면 그대로
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_from_euler_xyz

from .pick_place_cfg import BOX_Z_ON_TABLE, TABLE_SIZE
from .vlm_scene_cfg import BOX_COLORS, HIDDEN_POS, STAND_NAMES, STAND_X, STAND_Y


def _quat_yaw(yaw: float, device) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=device)


def apply_layouts(env, layouts: list[dict]) -> dict:
    """환경 i 에 layouts[i] 를 적용한다. 돌려주는 값: 받침대·상자의 월드 위치 등 (데모가 걷기·잡기 목표로 씀)."""
    device = env.device
    n = len(layouts)
    origins = env.scene.env_origins[:n]
    ids = torch.arange(n, device=device)
    stand_pose = torch.zeros(len(STAND_NAMES), n, 7, device=device)
    stand_yaw = torch.zeros(len(STAND_NAMES), n, device=device)
    box_pose = {c: torch.zeros(n, 7, device=device) for c in BOX_COLORS}
    box_used = {c: torch.zeros(n, dtype=torch.bool, device=device) for c in BOX_COLORS}
    for c_i, color in enumerate(BOX_COLORS):  # 기본은 화면 밖
        hidden = torch.tensor([HIDDEN_POS[0], HIDDEN_POS[1] + 0.6 * c_i, HIDDEN_POS[2]], device=device)
        box_pose[color][:, :3] = origins + hidden
        box_pose[color][:, 3] = 1.0
    for i, lay in enumerate(layouts):
        for k, name in enumerate(STAND_NAMES):
            dx, dy, yaw = lay.get("stands", {}).get(name, (0.0, 0.0, 0.0))
            pos = origins[i] + torch.tensor([STAND_X + dx, STAND_Y[k] + dy, TABLE_SIZE[2] / 2], device=device)
            stand_pose[k, i, :3], stand_pose[k, i, 3:], stand_yaw[k, i] = pos, _quat_yaw(yaw, device), yaw
            color = lay["boxes"].get(name)
            if color:
                bx, by, byaw = lay.get("box_offset", {}).get(name, (0.0, 0.0, 0.0))
                c, s = math.cos(yaw), math.sin(yaw)
                box_pose[color][i, 0] = pos[0] + c * bx - s * by
                box_pose[color][i, 1] = pos[1] + s * bx + c * by
                box_pose[color][i, 2] = lay.get("box_z", {}).get(color, BOX_Z_ON_TABLE)
                box_pose[color][i, 3:] = _quat_yaw(yaw + byaw, device)
                box_used[color][i] = True
    zeros6 = torch.zeros(n, 6, device=device)
    for k, name in enumerate(STAND_NAMES):
        env.scene[name].write_root_pose_to_sim(stand_pose[k], env_ids=ids)
    for color in BOX_COLORS:
        box = env.scene[f"{color}_box"]
        box.write_root_pose_to_sim(box_pose[color], env_ids=ids)
        box.write_root_velocity_to_sim(zeros6, env_ids=ids)
    if any("robot" in lay for lay in layouts):
        robot = env.scene["robot"]
        pose = robot.data.root_state_w[:n, :7].clone()
        for i, lay in enumerate(layouts):
            if "robot" in lay:
                x, y, z, roll, pitch, yaw = lay["robot"]
                pose[i, :3] = origins[i] + torch.tensor([x, y, z], device=device)
                pose[i, 3:] = quat_from_euler_xyz(torch.tensor(roll, device=device), torch.tensor(pitch, device=device),
                                                  torch.tensor(yaw, device=device))
        robot.write_root_pose_to_sim(pose, env_ids=ids)
        robot.write_root_velocity_to_sim(zeros6, env_ids=ids)
    return {"stand_pos": stand_pose[..., :3], "stand_yaw": stand_yaw, "box_used": box_used}


def set_lighting(dome_intensity: float, sun_intensity: float, sun_elevation_deg: float, sun_azimuth_deg: float):
    """조명 세기·해 방향을 바꾼다 (조명은 모든 환경이 함께 쓰므로 한 번에 한 값)."""
    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    stage.GetPrimAtPath("/World/skyLight").GetAttribute("inputs:intensity").Set(float(dome_intensity))
    sun = stage.GetPrimAtPath("/World/sunLight")
    sun.GetAttribute("inputs:intensity").Set(float(sun_intensity))
    # 방향성 조명은 자기 좌표의 -z 쪽으로 비춘다. 고도 90° = 바로 위에서 아래로
    tilt = math.radians(90.0 - sun_elevation_deg)
    q = quat_from_euler_xyz(torch.tensor(0.0), torch.tensor(tilt), torch.tensor(math.radians(sun_azimuth_deg)))
    sun.GetAttribute("xformOp:orient").Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))


def read_camera(env) -> torch.Tensor:
    """머리 카메라 사진 (N, 높이, 폭, 3) uint8."""
    return env.scene.sensors["head_cam"].data.output["rgb"][..., :3]
