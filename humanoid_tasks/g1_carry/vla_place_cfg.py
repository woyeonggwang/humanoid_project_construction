"""9단계: 내려놓기도 VLA 로 하는 과제와, 집기·내려놓기를 잇는 전체 장면의 설정.

8단계 2차 장면(넓은 받침대 + 작은 색 상자 4개)에 더하는 것
- 내려놓을 받침대(`place_stand`): 집는 받침대와 같은 크기
- 색 판 4개(`pad_<색>`): 내려놓을 받침대 위에 두 개를 좌우로 깐다. 명령으로 어느 색 판에 놓을지 정한다. 쓰지 않는 판은 스크립트가 멀리 치운다
- (전체 장면만) VLM 전용 넓은 머리 카메라(`vlm_cam`): 2m 떨어진 두 받침대를 한 장에 담는다. VLA 는 8단계의 아래를 보는 카메라(`vla_cam`)를 그대로 쓴다

판의 크기와 두께는 PAD_SIZE (두께를 정한 근거는 CLAUDE.md 9단계).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .vla_pick_cfg import SLOT_Y, WIDE_STAND_SIZE, G1VlaPick2EnvCfg
from .vlm_scene_cfg import BOX_COLORS, CAM_APERTURE, CAM_PITCH_DEG, CAM_POS, CAM_SIZE, HIDDEN_POS, STAND_X

PAD_SIZE = (0.24, 0.24, 0.04)  # 판 (앞뒤, 좌우, 두께) m. 상자(앞뒤 0.20, 좌우 0.14)보다 조금 큼
PAD_SLOT_Y = SLOT_Y  # 판 자리: 받침대 가운데에서 좌우로 (집을 때의 상자 자리와 같음)
PAD_TOP_Z = WIDE_STAND_SIZE[2] + PAD_SIZE[2]  # 판 윗면 높이
PAD_SHADE = 0.75  # 판의 색 = 상자 색보다 조금 어둡게 (같은 색 상자가 올라가도 구분되게)
PAD_NAMES = tuple(f"pad_{c}" for c in BOX_COLORS)

# VLM 전용 카메라: 7단계의 머리 카메라와 같은 자리·숙인 각도, 가로 시야각만 약 90° 로 넓힘
VLM_CAM_FOCAL = 10.5


def _kinematic_cuboid(name: str, size, color, pos) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class G1VlaPlaceEnvCfg(G1VlaPick2EnvCfg):
    """8단계 2차 장면 + 내려놓을 받침대(`place_stand`)와 색 판 4개(`pad_<색>`). 쓰지 않는 물체는 스크립트가 멀리 치운다."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.place_stand = _kinematic_cuboid("place_stand", WIDE_STAND_SIZE, (0.55, 0.56, 0.60),
                                                   (STAND_X, 0.0, WIDE_STAND_SIZE[2] / 2))
        for i, (color, rgb) in enumerate(BOX_COLORS.items()):
            setattr(self.scene, f"pad_{color}", _kinematic_cuboid(
                f"pad_{color}", PAD_SIZE, tuple(v * PAD_SHADE for v in rgb),
                (HIDDEN_POS[0] - 4.0, HIDDEN_POS[1] + 0.6 * i, HIDDEN_POS[2])))


@configclass
class G1VlaFullEnvCfg(G1VlaPlaceEnvCfg):
    """내려놓기 장면 + VLM 전용 넓은 머리 카메라(`vlm_cam`). 집는 받침대와 내려놓을 받침대를 한 사진에 담아 계획을 세울 때 쓴다."""

    def __post_init__(self):
        super().__post_init__()
        half = math.radians(CAM_PITCH_DEG) / 2
        self.scene.vlm_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/vlm_cam",
            offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=(math.cos(half), 0.0, math.sin(half), 0.0), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=VLM_CAM_FOCAL, horizontal_aperture=CAM_APERTURE, clipping_range=(0.05, 15.0)),
            width=CAM_SIZE,
            height=CAM_SIZE,
        )
