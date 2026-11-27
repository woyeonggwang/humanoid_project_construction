"""9단계 전체 연결용 장면: 집는 받침대(색 상자 두 자리)와 내려놓을 받침대(색 판 두 개)를 함께 놓는다.

scripts/make_vlm_full_data.py (VLM 데이터 생성)와 scripts/vla_full_eval.py (전체 연결 실행·평가)가 같이 쓴다.
isaaclab 을 import 하므로 Isaac Sim 앱이 켜진 뒤에만 import 한다. 과제는 Humanoid-G1-VLA-Full-v0.

배치: 로봇은 원점에서 +x 를 보고 시작한다. 집는 받침대는 왼쪽 앞, 내려놓을 받침대는 오른쪽 앞에 로봇 쪽을 보게 둔다 (둘 사이 2m).
VLM 전용 넓은 카메라의 사진에서 집는 받침대가 왼쪽, 내려놓을 받침대가 오른쪽에 보인다.

장면(scene) 형식 — 환경 하나에 dict 하나. 자리는 받침대를 마주 본 로봇 기준 +1 = 왼쪽, -1 = 오른쪽:
  {"pick": (dx, dy, dyaw), "place": (dx, dy, dyaw),      두 받침대의 기본 자리 대비 이동(m)·방향(rad)
   "boxes": {1: 색 또는 None, -1: …}, "box_offset": {1: (bx, by, byaw), -1: …},   집는 받침대의 상자
   "pads": {1: 색, -1: 색}, "pad_offset": {1: (px, py), -1: …},                    내려놓을 받침대의 판
   "on_pad": {1: 색 또는 None, -1: …}, "on_pad_offset": {…},                        판 위에 이미 놓인 상자
   "mass": {색: kg}, "robot": (x, y, z, roll, pitch, yaw)}                          robot 은 몸을 고정했을 때만 (없으면 그대로)
"""

from __future__ import annotations

import math
import random

import torch

from . import vla_pick as vp
from .pick_place_cfg import BOX_Z_ON_TABLE
from .vla_pick_cfg import SLOT_Y, WIDE_STAND_SIZE
from .vla_place import BOX_REST_Z
from .vla_place_cfg import PAD_SLOT_Y, PAD_TOP_Z
from .vlm_scene import apply_layouts
from .vlm_scene_cfg import BOX_COLORS, HIDDEN_POS, STAND_NAMES, STAND_X

PICK_XY, PLACE_XY = (1.7, 1.0), (1.7, -1.0)  # 두 받침대의 기본 자리 (환경 원점 기준 m). 둘 사이 2.0m
PICK_YAW, PLACE_YAW = math.atan2(PICK_XY[1], PICK_XY[0]), math.atan2(PLACE_XY[1], PLACE_XY[0])  # 처음 자리의 로봇을 마주 보는 방향
# 쓰지 않는 7단계 받침대를 치우는 이동량: 환경 간격(30m) 때문에 이웃 환경의 앞쪽에 놓이지 않게 뒤로 35m (25m 면 이웃의 앞 6m 에 보임)
HIDE = (-35.0, 0.0, 0.0)
SIDE_NAME = {1: "L", -1: "R"}


def sample_scene(rng: random.Random, n_boxes: int | None = None, on_pad: bool | None = None,
                 pos_jitter: float = 0.05, yaw_jitter_deg: float = 8.0, mass_range=(2.0, 3.0)) -> dict:
    """장면 하나를 무작위로. n_boxes: 집는 받침대의 상자 수 (None 이면 0·1·2 를 5:25:70), on_pad: 판 하나에 상자가 이미 있는지 (None 이면 35%)."""
    colors = list(BOX_COLORS)
    if n_boxes is None:
        n_boxes = rng.choices([0, 1, 2], weights=[5, 25, 70])[0]
    if on_pad is None:
        on_pad = rng.random() < 0.35
    order = rng.sample(colors, 4)
    box_colors = order[:n_boxes]
    sides = rng.sample([1, -1], 2)
    boxes = {1: None, -1: None}
    for s, c in zip(sides, box_colors):
        boxes[s] = c
    pad_colors = rng.sample(colors, 2)
    occupied = {1: None, -1: None}
    left = [c for c in colors if c not in box_colors]
    if on_pad and left:
        occupied[rng.choice([1, -1])] = rng.choice(left)

    def stand():
        return (rng.uniform(-pos_jitter, pos_jitter), rng.uniform(-pos_jitter, pos_jitter), math.radians(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)))

    return {
        "pick": stand(), "place": stand(), "boxes": boxes, "pads": {1: pad_colors[0], -1: pad_colors[1]}, "on_pad": occupied,
        "box_offset": {s: (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), math.radians(rng.uniform(-3.0, 3.0))) for s in (1, -1)},
        "pad_offset": {s: (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01)) for s in (1, -1)},
        "on_pad_offset": {s: (rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), math.radians(rng.uniform(-8.0, 8.0))) for s in (1, -1)},
        "mass": {c: rng.uniform(*mass_range) for c in colors},
    }


def describe(scene: dict) -> dict:
    """문장 템플릿(vlm/templates_full.py)에 넘길 형식: 자리 이름을 L·R 로."""
    return {k: {SIDE_NAME[s]: scene[k][s] for s in (1, -1)} for k in ("boxes", "pads", "on_pad")}


def _yaw_pose(x: float, y: float, z: float, yaw: float, device) -> torch.Tensor:
    return torch.tensor([x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=device)


def _local(xy, yaw: float, lx: float, ly: float):
    return xy[0] + math.cos(yaw) * lx - math.sin(yaw) * ly, xy[1] + math.sin(yaw) * lx + math.cos(yaw) * ly


class FullLayout:
    """apply_scenes 가 돌려주는 것: 두 받침대와 자리들의 월드 위치."""

    def __init__(self, n: int, device):
        z = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        self.pick_xy, self.pick_yaw = z(n, 2), z(n)  # 집는 받침대 중심·로봇이 마주 볼 방향
        self.place_xy, self.place_yaw = z(n, 2), z(n)
        self.slot_xy = z(n, 2, 2)  # 집는 받침대의 상자 자리 [환경, 0 = 왼쪽 / 1 = 오른쪽, xy]
        self.pad_xy = z(n, 2, 2)  # 판 가운데


def apply_scenes(env, scenes: list[dict]) -> FullLayout:
    """환경 i 에 scenes[i] 를 적용한다 (받침대 둘, 판, 상자, 상자 무게). 쓰지 않는 물체는 화면 밖으로 치운다."""
    n, device = len(scenes), env.device
    origins = env.scene.env_origins[:n]
    ids = torch.arange(n, device=device)
    hidden = [{"boxes": {s: None for s in STAND_NAMES}, "stands": {s: HIDE for s in STAND_NAMES}} for _ in range(n)]
    for lay, sc in zip(hidden, scenes):
        if "robot" in sc:
            lay["robot"] = sc["robot"]
    apply_layouts(env, hidden)
    lay = FullLayout(n, device)
    stand_pose = {"wide_stand": torch.zeros(n, 7, device=device), "place_stand": torch.zeros(n, 7, device=device)}
    pad_pose = {c: torch.zeros(n, 7, device=device) for c in BOX_COLORS}
    box_pose = {c: torch.zeros(n, 7, device=device) for c in BOX_COLORS}
    masses = {c: torch.full((n,), 2.5) for c in BOX_COLORS}
    for c_i, c in enumerate(BOX_COLORS):  # 기본은 숨김
        pad_pose[c][:, :3] = origins + torch.tensor([HIDDEN_POS[0] - 4.0, HIDDEN_POS[1] + 0.6 * c_i, HIDDEN_POS[2]], device=device)
        pad_pose[c][:, 3] = 1.0
        box_pose[c][:, :3] = origins + torch.tensor([HIDDEN_POS[0] - 2.0, HIDDEN_POS[1] + 0.6 * c_i, HIDDEN_POS[2]], device=device)
        box_pose[c][:, 3] = 1.0
    half = WIDE_STAND_SIZE[2] / 2
    for i, sc in enumerate(scenes):
        ox, oy = float(origins[i, 0]), float(origins[i, 1])
        px, py = ox + PICK_XY[0] + sc["pick"][0], oy + PICK_XY[1] + sc["pick"][1]
        qx, qy = ox + PLACE_XY[0] + sc["place"][0], oy + PLACE_XY[1] + sc["place"][1]
        pyaw, qyaw = PICK_YAW + sc["pick"][2], PLACE_YAW + sc["place"][2]
        stand_pose["wide_stand"][i] = _yaw_pose(px, py, half, pyaw, device)
        stand_pose["place_stand"][i] = _yaw_pose(qx, qy, half, qyaw, device)
        lay.pick_xy[i], lay.pick_yaw[i] = torch.tensor([px, py], device=device), pyaw
        lay.place_xy[i], lay.place_yaw[i] = torch.tensor([qx, qy], device=device), qyaw
        for k, s in enumerate((1, -1)):
            bx, by, byaw = sc["box_offset"][s]
            wx, wy = _local((px, py), pyaw, bx, s * SLOT_Y + by)
            lay.slot_xy[i, k] = torch.tensor([wx, wy], device=device)
            if sc["boxes"][s]:
                box_pose[sc["boxes"][s]][i] = _yaw_pose(wx, wy, BOX_Z_ON_TABLE, pyaw + byaw, device)
            ax, ay = sc["pad_offset"][s]
            wx, wy = _local((qx, qy), qyaw, ax, s * PAD_SLOT_Y + ay)
            lay.pad_xy[i, k] = torch.tensor([wx, wy], device=device)
            pad_pose[sc["pads"][s]][i] = _yaw_pose(wx, wy, (PAD_TOP_Z + WIDE_STAND_SIZE[2]) / 2, qyaw, device)
            if sc["on_pad"][s]:
                cx, cy, cyaw = sc["on_pad_offset"][s]
                ux, uy = _local((wx, wy), qyaw, cx, cy)
                box_pose[sc["on_pad"][s]][i] = _yaw_pose(ux, uy, BOX_REST_Z, qyaw + cyaw, device)
        for c in BOX_COLORS:
            masses[c][i] = sc["mass"][c]
    zeros6 = torch.zeros(n, 6, device=device)
    for name, pose in stand_pose.items():
        env.scene[name].write_root_pose_to_sim(pose, env_ids=ids)
    for c in BOX_COLORS:
        env.scene[f"pad_{c}"].write_root_pose_to_sim(pad_pose[c], env_ids=ids)
        box = env.scene[f"small_{c}_box"]
        box.write_root_pose_to_sim(box_pose[c], env_ids=ids)
        box.write_root_velocity_to_sim(zeros6, env_ids=ids)
        vp.set_box_mass(box, masses[c])
    return lay


def read_vlm_camera(env) -> torch.Tensor:
    """VLM 전용 넓은 머리 카메라의 사진 (N, 높이, 폭, 3) uint8."""
    return env.scene.sensors["vlm_cam"].data.output["rgb"][..., :3]
