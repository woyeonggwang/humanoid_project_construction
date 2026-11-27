"""10단계: 공사장 장면의 배치 자료 (모델 목록, 놓는 자리, 단순 상자 충돌, 철골의 용접선 좌표).

isaaclab 을 import 하지 않는다 (장면 설정 site_cfg.py 와, 다음 단계의 용접 작업이 같이 쓴다).
좌표는 환경 원점 기준 m, 로봇은 원점에서 +x 를 보고 시작한다. 방향(yaw)은 z 축 둘레 각도(도).
모델 파일은 assets/usd/<이름>.usda (Z 위쪽, 미터, 원점 = 모델 바닥 가운데. 펜스만 원점이 한쪽 끝이고 +x 로 2m 뻗음).

배치 원칙 (근거는 CLAUDE.md 10단계)
- 받침대·상자·판과 그 둘레는 그대로 둔다. VLM 카메라는 처음 자리에서 앞쪽(+x)을 지평선까지 보고,
  VLA 카메라는 받침대 앞에 선 로봇의 발밑·앞쪽 약 3.3m 까지를 본다.
- A 배치: 모든 모델을 로봇 처음 자리의 뒤쪽·옆쪽(x <= 0)에 둔다. 그림자가 카메라에 보이는 바닥에 닿지 않게,
  보이는 구역에서 '모델 높이 x 1.2' 보다 멀리 둔다 (해 고도 40° 이상).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"
USD_DIR = ASSET_DIR / "usd"
WELD_SEAMS_FILE = ASSET_DIR / "weld_seams.json"

# 모델별 크기 (x, y, z) m — USD 의 extent 로 확인한 값. 원점은 바닥 가운데 (펜스는 x 0~2)
MODEL_SIZE = {
    "SM_FencePanel": (2.0, 0.6, 1.9),
    "SM_SafetyCone": (0.38, 0.38, 0.70),
    "SM_Barricade": (1.2, 0.48, 0.83),
    "SM_RebarBundle": (4.08, 0.50, 0.18),
    "SM_BrickPallet": (1.1, 1.1, 0.55),
    "SM_BlockPallet": (1.2, 1.0, 0.91),
    "SM_LumberPile": (2.78, 0.70, 0.45),
    "SM_CementPallet": (1.2, 1.0, 0.80),
    "SM_SteelFrame": (4.32, 0.32, 2.0),
    "SM_ScaffoldTower": (1.95, 1.35, 6.12),
    "SM_SiteOffice": (6.42, 3.08, 2.79),
    "SM_Floor_Concrete": (10.0, 10.0, 0.021),
    "SM_Floor_Dirt": (10.0, 10.0, 0.021),
}

# 단순 상자 충돌: 모델 좌표의 (중심, 크기). 로봇이 부딪힐 수 있는 모델만. 비계·사무실·바닥은 겉모습만 둔다.
COLLIDERS = {
    "SM_FencePanel": [((1.0, 0.0, 0.95), (2.0, 0.30, 1.9))],  # 철망 면을 감싸는 판 하나 (받침 블록은 좌우 0.6 까지 나오지만 높이 0.12 라 뺌)
    "SM_SafetyCone": [((0.0, 0.0, 0.35), (0.38, 0.38, 0.70))],
    "SM_Barricade": [((0.0, 0.0, 0.415), (1.2, 0.48, 0.83))],
    "SM_RebarBundle": [((0.0, 0.0, 0.09), (4.08, 0.50, 0.18))],
    "SM_BrickPallet": [((0.0, 0.0, 0.275), (1.1, 1.1, 0.55))],
    "SM_BlockPallet": [((0.0, 0.0, 0.455), (1.2, 1.0, 0.91))],
    "SM_LumberPile": [((0.0, 0.0, 0.225), (2.78, 0.70, 0.45))],
    "SM_CementPallet": [((0.0, 0.0, 0.40), (1.2, 1.0, 0.80))],
    # 철골: 기둥 3개(H 형강 0.2 x 0.2, 높이 2m)와 그 사이의 보 2개(길이 1.8m, 폭 0.15, 높이 0.9~1.1m). 기둥 사이는 비워 팔이 들어갈 수 있게
    "SM_SteelFrame": [((-2.0, 0.0, 1.0), (0.20, 0.20, 2.0)), ((0.0, 0.0, 1.0), (0.20, 0.20, 2.0)), ((2.0, 0.0, 1.0), (0.20, 0.20, 2.0)),
                      ((-1.0, 0.0, 1.0), (1.8, 0.15, 0.20)), ((1.0, 0.0, 1.0), (1.8, 0.15, 0.20))],
}

# 철골(다음 단계의 용접 대상)을 둔 자리. 길이 방향이 y 축과 나란하고, 모델의 -y 면(세로 용접선이 있는 쪽)이 +x(로봇 처음 자리 쪽)를 봄
STEEL_FRAME_POS = (-3.5, 0.0, 0.0)
STEEL_FRAME_YAW_DEG = 90.0

# ---- A 배치: 두 카메라에 보이지 않는 곳 (이름, 모델, (x, y), yaw 도) ----
PLACEMENTS_A = (
    [(f"fence_back_{i}", "SM_FencePanel", (-10.0, -6.0 + 2.0 * i), 90.0) for i in range(6)]  # 뒤쪽 줄 x = -10, y -6 ~ 6
    + [(f"fence_left_{i}", "SM_FencePanel", (-10.0 + 2.0 * i, 6.0), 0.0) for i in range(5)]  # 왼쪽 줄 y = +6, x -10 ~ 0
    + [(f"fence_right_{i}", "SM_FencePanel", (-10.0 + 2.0 * i, -6.0), 0.0) for i in range(5)]  # 오른쪽 줄 y = -6
    + [
        ("steel_frame", "SM_SteelFrame", STEEL_FRAME_POS[:2], STEEL_FRAME_YAW_DEG),
        ("scaffold", "SM_ScaffoldTower", (-8.6, 3.6), 0.0),
        ("site_office", "SM_SiteOffice", (-8.0, -2.6), 90.0),
        ("rebar", "SM_RebarBundle", (-5.8, 4.9), 0.0),
        ("lumber", "SM_LumberPile", (-5.6, 3.9), 0.0),
        ("brick_0", "SM_BrickPallet", (-1.6, 4.6), 0.0),
        ("brick_1", "SM_BrickPallet", (-2.9, 4.7), 8.0),
        ("block", "SM_BlockPallet", (-1.8, -4.6), 0.0),
        ("cement", "SM_CementPallet", (-3.6, -4.6), -6.0),
        ("cone_0", "SM_SafetyCone", (-2.4, 2.7), 0.0),
        ("cone_1", "SM_SafetyCone", (-2.4, -2.7), 0.0),
        ("cone_2", "SM_SafetyCone", (-4.6, 2.7), 0.0),
        ("cone_3", "SM_SafetyCone", (-4.6, -2.7), 0.0),
        ("barricade_0", "SM_Barricade", (-5.0, 1.2), 90.0),
        ("barricade_1", "SM_Barricade", (-5.0, -1.2), 90.0),
    ]
)

# ---- B 배치에 더하는 것: 카메라에 보이는 곳 (VLM 카메라의 배경). 받침대 앞에 선 로봇의 VLA 카메라가 보는 바닥(앞쪽 약 3.3m)은 피함 ----
PLACEMENTS_B_VISIBLE = (
    [(f"fence_front_{i}", "SM_FencePanel", (8.0, -6.0 + 2.0 * i), 90.0) for i in range(6)]  # 앞쪽 줄 x = +8 (받침대 뒤 약 6m)
    + [(f"fence_left_f{i}", "SM_FencePanel", (2.0 * i, 6.0), 0.0) for i in range(4)]  # 왼쪽 줄을 x 0 ~ 8 로 이음
    + [(f"fence_right_f{i}", "SM_FencePanel", (2.0 * i, -6.0), 0.0) for i in range(4)]
    + [
        ("cement_f", "SM_CementPallet", (5.6, 4.0), 12.0),
        ("block_f", "SM_BlockPallet", (6.6, 2.4), -5.0),
        ("lumber_f", "SM_LumberPile", (6.2, -3.6), 20.0),
        ("brick_f", "SM_BrickPallet", (5.2, -4.6), 0.0),
        ("barricade_f", "SM_Barricade", (6.8, 0.0), 90.0),
        ("cone_f0", "SM_SafetyCone", (5.0, 0.6), 0.0),
        ("cone_f1", "SM_SafetyCone", (5.0, -0.6), 0.0),
    ]
)

# 바닥 겉모습: 10m x 10m 판을 깐다 (충돌 없음. 물리 바닥은 원본 평면 그대로). 물리 바닥의 겉면(z = 0)과 겹쳐 깜빡이지 않게 FLOOR_Z 만큼 올림
FLOOR_Z = 0.01
FLOOR_TILES_XY = [(-5.0 + 10.0 * i, -20.0 + 10.0 * j) for i in range(6) for j in range(5)]  # x -10 ~ 50, y -25 ~ 25


def floor_placements(model: str = "SM_Floor_Concrete"):
    return [(f"floor_{k}", model, xy, 0.0) for k, xy in enumerate(FLOOR_TILES_XY)]


def model_path(model: str) -> str:
    return str(USD_DIR / f"{model}.usda")


def to_world(local, pos, yaw_deg: float, origin=(0.0, 0.0, 0.0)):
    """모델 좌표의 점 local 을, 모델을 pos·yaw 로 놓았을 때의 좌표로 바꾼다 (origin = 환경 원점)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    z0 = pos[2] if len(pos) > 2 else 0.0
    return (origin[0] + pos[0] + c * local[0] - s * local[1], origin[1] + pos[1] + s * local[0] + c * local[1], origin[2] + z0 + local[2])


def colliders_of(model: str, pos, yaw_deg: float):
    """모델을 pos·yaw 로 놓았을 때의 충돌 상자들: [(중심 (x, y, z), 크기, yaw 도)]. 환경 원점 기준."""
    return [(to_world(center, pos, yaw_deg), size, yaw_deg) for center, size in COLLIDERS.get(model, [])]


def weld_seams_local() -> dict:
    """용접선 (철골 원점 기준 m): {이름: {"start": [x, y, z], "end": [x, y, z]}}. assets/weld_seams.json 을 읽는다."""
    data = json.loads(WELD_SEAMS_FILE.read_text(encoding="utf-8"))
    return data["seams"]


def weld_seams_world(env_origin=(0.0, 0.0, 0.0), frame_pos=STEEL_FRAME_POS, frame_yaw_deg: float = STEEL_FRAME_YAW_DEG) -> dict:
    """용접선을 월드 좌표로: {이름: {"start": (x, y, z), "end": (x, y, z)}}.

    env_origin 은 그 환경의 원점 (env.scene.env_origins[i]). 철골을 다른 자리에 두었으면 frame_pos·frame_yaw_deg 를 넘긴다.
    """
    origin = tuple(float(v) for v in env_origin)
    return {name: {key: to_world(seam[key], frame_pos, frame_yaw_deg, origin) for key in ("start", "end")}
            for name, seam in weld_seams_local().items()}
