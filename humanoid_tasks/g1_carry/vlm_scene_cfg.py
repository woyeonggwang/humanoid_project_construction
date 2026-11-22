"""7단계: VLM 판단 층용 장면 설정.

로봇 앞에 받침대 3개(로봇이 볼 때 왼쪽 stand_L, 가운데 stand_C, 오른쪽 stand_R)와 색 상자 4개(red, blue, green, yellow)를 두고,
로봇 머리 위치에 카메라를 단다. 상자를 어느 받침대에 올릴지·받침대 위치·조명 등은 스크립트가 정한다
(scripts/make_vlm_data.py, scripts/vlm_pick_place_demo.py). 쓰지 않는 상자는 화면 밖 먼 곳에 둔다.

걷기 정책은 v2(G1LocoV2FlatEnvCfg)를 쓰고, 팔은 스크립트(IK)가 움직이므로 팔 무작위 이벤트와 손 무게 무작위화를 끈다.
카메라가 있으므로 실행할 때 --enable_cameras 가 필요하다.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .env_cfg import G1LocoV2FlatEnvCfg_PLAY
from .pick_place_cfg import BOX_MASS, BOX_SIZE, TABLE_SIZE, apply_script_control

STAND_NAMES = ("stand_L", "stand_C", "stand_R")  # 로봇이 볼 때 왼쪽·가운데·오른쪽
STAND_X = 1.45  # 로봇 시작 위치에서 받침대까지 앞쪽 거리 (m)
STAND_Y = (0.65, 0.0, -0.65)  # 왼쪽(+y)·가운데·오른쪽. 가로 시야각 70° 안에 세 받침대가 여유 있게 들어오는 간격
BOX_COLORS = {
    "red": (0.80, 0.07, 0.07),
    "blue": (0.07, 0.18, 0.85),
    "green": (0.08, 0.58, 0.14),
    "yellow": (0.93, 0.80, 0.08),
}
BOX_NAMES = tuple(f"{c}_box" for c in BOX_COLORS)
HIDDEN_POS = (-30.0, 0.0, 0.2)  # 쓰지 않는 상자를 두는 곳 (환경 원점 기준, 카메라 뒤쪽 멀리)

# 머리 카메라: 몸통(torso_link) 기준 위치·아래로 숙인 각도. 모델 입력에 맞춰 448x448
CAM_SIZE = 448
CAM_POS = (0.10, 0.0, 0.43)
CAM_PITCH_DEG = 26.0
CAM_FOCAL, CAM_APERTURE = 15.0, 20.955  # 가로 시야각 약 70°


def _stand_cfg(name: str, y: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.56, 0.60)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(STAND_X, y, TABLE_SIZE[2] / 2)),
    )


def _box_cfg(name: str, color: tuple[float, float, float], index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(HIDDEN_POS[0], HIDDEN_POS[1] + 0.6 * index, HIDDEN_POS[2])),
    )


@configclass
class G1VlmSceneEnvCfg(G1LocoV2FlatEnvCfg_PLAY):
    """받침대 3개 + 색 상자 4개 + 머리 카메라. 걷기·팔·붙이기·장면 배치는 스크립트가 맡는다."""

    def __post_init__(self):
        super().__post_init__()
        apply_script_control(self)
        self.scene.num_envs = 1
        self.scene.env_spacing = 30.0  # 다른 환경의 로봇·상자가 사진에 찍히지 않게 멀리 (카메라는 15m 까지만 그림)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        for name, y in zip(STAND_NAMES, STAND_Y):
            setattr(self.scene, name, _stand_cfg(name, y))
        for i, (color, rgb) in enumerate(BOX_COLORS.items()):
            setattr(self.scene, f"{color}_box", _box_cfg(f"{color}_box", rgb, i))
        half = math.radians(CAM_PITCH_DEG) / 2
        self.scene.head_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_cam",
            offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=(math.cos(half), 0.0, math.sin(half), 0.0), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=CAM_FOCAL, horizontal_aperture=CAM_APERTURE,
                                             clipping_range=(0.05, 15.0)),
            width=CAM_SIZE,
            height=CAM_SIZE,
        )
        # 조명: 스크립트가 세기·방향을 바꿀 수 있게 방향성 조명을 하나 더 둠
        self.scene.sun_light = AssetBaseCfg(
            prim_path="/World/sunLight",
            spawn=sim_utils.DistantLightCfg(intensity=1500.0, angle=1.0, color=(1.0, 0.98, 0.95)),
        )
