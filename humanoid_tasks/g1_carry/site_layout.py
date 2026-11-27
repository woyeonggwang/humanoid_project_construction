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


# ---------------------------------------------------------------------------
# 11단계 용접: 철골의 실제 모양과 용접선의 바깥 방향
# ---------------------------------------------------------------------------
def _steel_frame_plates():
    """철골을 이루는 판 18개: 모델 좌표의 (중심, 크기). USD 메시를 이어진 덩어리로 나눠 확인한 치수.

    기둥(H 형강 0.2 x 0.2, 높이 2m) = 바닥판 + 플랜지 2장(두께 12mm, 보가 붙는 면) + 웨브(8mm).
    보(H 형강 폭 0.15, 높이 0.9~1.1m) = 위·아래 플랜지(두께 10mm) + 웨브(8mm). 충돌 상자(COLLIDERS)는 보를 꽉 찬 상자로 두지만,
    용접 토치가 닿았는지는 이 판들과의 거리로 판정한다 (세로 용접선은 위·아래 플랜지 사이, 웨브 면에 있음).
    """
    plates = []
    for cx in (-2.0, 0.0, 2.0):
        plates.append(((cx, 0.0, 0.008), (0.32, 0.32, 0.016)))
        plates.append(((cx - 0.094, 0.0, 1.008), (0.012, 0.20, 1.984)))
        plates.append(((cx + 0.094, 0.0, 1.008), (0.012, 0.20, 1.984)))
        plates.append(((cx, 0.0, 1.008), (0.176, 0.008, 1.984)))
    for cx in (-1.0, 1.0):
        plates.append(((cx, 0.0, 0.905), (1.8, 0.15, 0.01)))
        plates.append(((cx, 0.0, 1.095), (1.8, 0.15, 0.01)))
        plates.append(((cx, 0.0, 1.0), (1.8, 0.008, 0.18)))
    return plates


STEEL_FRAME_PLATES = _steel_frame_plates()


def weld_seam_info() -> dict:
    """용접선마다 종류와 바깥 방향(모델 좌표)을 붙인다: {이름: {"start", "end", "kind", "column", "beam_side", "normal"}}.

    바깥 방향(normal)은 JSON 에 없어 철골 모양으로 정한다. 용접선은 두 면이 만나는 구석에 있으므로 두 면의 법선을 더한 방향(45°):
    - 가로선(kind "h", 보 윗면과 기둥 면이 만나는 선): 기둥 면의 법선(보 쪽, 모델 ±x) + 보 윗면의 법선(+z)
    - 세로선(kind "v", 보 웨브와 기둥 면이 만나는 선): 기둥 면의 법선(보 쪽, 모델 ±x) + 웨브 면의 법선(모델 -y)
    beam_side = 기둥에서 볼 때 보가 있는 쪽 (모델 x 의 +1 / -1).
    """
    r = math.sqrt(0.5)
    out = {}
    for name, seam in weld_seams_local().items():
        start, end = [float(v) for v in seam["start"]], [float(v) for v in seam["end"]]
        col_x = min((-2.0, 0.0, 2.0), key=lambda c: abs(c - start[0]))
        beam_side = 1.0 if start[0] > col_x else -1.0
        kind = "h" if abs(start[2] - end[2]) < 1.0e-6 else "v"
        normal = (beam_side * r, 0.0, r) if kind == "h" else (beam_side * r, -r if start[1] < 0 else r, 0.0)
        out[name] = {"start": start, "end": end, "kind": kind, "column": name.split("_")[1], "beam_side": beam_side, "normal": normal}
    return out


def weld_seam_frames(env_origin=(0.0, 0.0, 0.0), frame_pos=STEEL_FRAME_POS, frame_yaw_deg: float = STEEL_FRAME_YAW_DEG) -> dict:
    """weld_seam_info 를 월드 좌표로: start·end 는 점, normal 은 방향(길이 1)."""
    origin = tuple(float(v) for v in env_origin)
    out = {}
    for name, info in weld_seam_info().items():
        item = dict(info)
        item["start"] = to_world(info["start"], frame_pos, frame_yaw_deg, origin)
        item["end"] = to_world(info["end"], frame_pos, frame_yaw_deg, origin)
        item["normal"] = to_world(info["normal"], (0.0, 0.0, 0.0), frame_yaw_deg)
        out[name] = item
    return out


# 용접선마다 로봇이 설 자리: (D, L, yaw) — D = 철골 가운데 면에서 골반까지의 거리(용접선이 있는 면 쪽으로),
# L = 용접선에서 옆으로 벗어난 거리(철골 길이 방향, 모델 +x 쪽이 +), yaw = 철골을 정면으로 볼 때에서 돌아선 각도(도, 왼쪽이 +).
# 값은 scripts/weld_check.py --mode reach (몸 고정)로 후보를 고르고 scripts/weld_eval.py (서서)로 확인해 정한다.
WELD_STANCE = {
    "column_A_seam_2": (0.53, -0.10, -20.0),
    "column_B_seam_2": (0.64, -0.20, 20.0),
    "column_B_seam_4": (0.53, -0.10, -20.0),
    "column_C_seam_2": (0.64, -0.20, 20.0),
}
# 용접선의 한국어 이름 (철골을 마주 보고 왼쪽부터 A·B·C 기둥. B 기둥은 양쪽에 보가 붙어 왼쪽·오른쪽으로 나눈다)
SEAM_KO = {
    "column_A_seam_1": "A 기둥 가로 이음부", "column_A_seam_2": "A 기둥 세로 이음부",
    "column_B_seam_1": "B 기둥 왼쪽 가로 이음부", "column_B_seam_2": "B 기둥 왼쪽 세로 이음부",
    "column_B_seam_3": "B 기둥 오른쪽 가로 이음부", "column_B_seam_4": "B 기둥 오른쪽 세로 이음부",
    "column_C_seam_1": "C 기둥 가로 이음부", "column_C_seam_2": "C 기둥 세로 이음부",
}
# 손이 닿는 용접선 = 세로선 4개. 가로선 4개(보 윗면, 높이 1.1m)는 닿지 않는 것으로 둔다 (근거는 CLAUDE.md 11단계)
WELD_REACHABLE = tuple(WELD_STANCE)

# 기준 자세: 그 자리에 서서 토치 끝을 용접선의 네 곳에 두고 토치를 용접선 쪽으로 향하게 한 오른팔 관절값
# (어깨 pitch·roll·yaw, 팔꿈치 pitch·roll). far = 거치는 점(용접선 가운데에서 바깥으로 18cm), mid·start·end = 용접선 가운데·시작·끝에서 1cm.
# 용접 IK 가 이 자세들 사이를 이어 기준으로 삼는다 (팔이 다른 꼴로 풀리지 않게). weld_check.py --mode reach 가 출력한 값
_POSTURE_BEAM_RIGHT = {"far": (-0.97, -1.31, 0.86, 0.33, -0.15), "mid": (-1.41, -0.55, 1.07, 1.03, -0.28),
                       "start": (-1.24, -0.47, 0.98, 1.08, -0.30), "end": (-1.60, -0.59, 1.14, 1.05, -0.28)}  # 보가 로봇의 오른쪽 (A 기둥, B 기둥의 4번)
_POSTURE_BEAM_LEFT = {"far": (-2.16, 0.99, -2.35, 0.12, -0.01), "mid": (-1.58, 0.24, -1.96, 0.79, -0.21),
                      "start": (-1.41, 0.25, -1.94, 0.83, -0.24), "end": (-1.82, 0.15, -2.10, 0.80, -0.23)}  # 보가 로봇의 왼쪽 (B 기둥의 2번, C 기둥)
WELD_POSTURE = {
    "column_A_seam_2": _POSTURE_BEAM_RIGHT,
    "column_B_seam_2": _POSTURE_BEAM_LEFT,
    "column_B_seam_4": _POSTURE_BEAM_RIGHT,
    "column_C_seam_2": _POSTURE_BEAM_LEFT,
}


def weld_stance(name: str, env_origin=(0.0, 0.0, 0.0), stance=None, frame_pos=STEEL_FRAME_POS, frame_yaw_deg: float = STEEL_FRAME_YAW_DEG):
    """용접선 name 을 용접할 때 설 자리: (x, y, 방향 rad). stance 를 주면 WELD_STANCE 대신 그 (D, L, yaw) 를 쓴다."""
    d, lat, yaw_off = stance if stance is not None else WELD_STANCE[name]
    info = weld_seam_info()[name]
    mid_x = 0.5 * (info["start"][0] + info["end"][0])
    x, y, _ = to_world((mid_x + lat, -d, 0.0), frame_pos, frame_yaw_deg, tuple(float(v) for v in env_origin))
    return x, y, math.radians(frame_yaw_deg + 90.0 + yaw_off)
