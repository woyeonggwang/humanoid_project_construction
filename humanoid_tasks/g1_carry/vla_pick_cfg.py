"""8단계: VLA(SmolVLA) 로 팔을 움직여 상자를 집는 과제의 장면 설정.

7단계 VLM 장면(받침대 3개 + 색 상자 4개)을 그대로 쓰고, 카메라만 바꾼다.
- 7단계의 머리 카메라는 멀리 있는 받침대 세 개를 보려고 26° 만 숙였다. 받침대 바로 앞에 서면 상자와 손이 화면 아래로 벗어난다.
- 팔을 움직일 때는 손과 상자가 화면에 들어와야 하므로, 같은 자리(몸통 기준)에서 더 숙인 카메라를 쓴다.

걷기는 정책 v2 가 맡고, 팔 목표값과 붙이기는 스크립트 또는 VLA 가 정한다 (scripts/collect_vla_demos.py, scripts/vla_pick_eval.py).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .vlm_scene_cfg import CAM_APERTURE, CAM_FOCAL, CAM_POS, G1VlmSceneEnvCfg

# 팔 조작용 머리 카메라: 256x256. SmolVLA 는 사진을 512x512 로 맞춰 넣으므로(비율 유지, 남는 곳 채움) 정사각형이면 채우는 곳이 없고,
# smolvla_base 가 기대하는 카메라 입력 모양도 (3, 256, 256) 이다. 받침대 앞 0.36m 에 섰을 때 상자와 양손이 화면에 들어오는 각도로 숙인다.
VLA_CAM_SIZE = 256
VLA_CAM_PITCH_DEG = 58.0


def vla_cam_cfg(pitch_deg: float = VLA_CAM_PITCH_DEG, size: int = VLA_CAM_SIZE, name: str = "vla_cam") -> TiledCameraCfg:
    half = math.radians(pitch_deg) / 2
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/torso_link/{name}",
        offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=(math.cos(half), 0.0, math.sin(half), 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAM_FOCAL, horizontal_aperture=CAM_APERTURE, clipping_range=(0.05, 15.0)),
        width=size,
        height=size,
    )


@configclass
class G1VlaPickEnvCfg(G1VlmSceneEnvCfg):
    """받침대·색 상자 장면 + 아래를 보는 머리 카메라 (`vla_cam`). 7단계의 먼 곳을 보는 카메라는 뺀다."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.head_cam = None
        self.scene.vla_cam = vla_cam_cfg()


# ---------------------------------------------------------------------------
# 2차: 넓은 받침대 하나에 색이 다른 작은 상자 두 개를 나란히 둔다 (명령으로 집을 색을 정함)
# ---------------------------------------------------------------------------
# 로봇은 받침대 가운데 앞에 서고 팔만 움직이므로 두 상자가 모두 팔이 닿는 곳에 있어야 한다.
# 상자 폭 14cm, 가운데에서 좌우 18cm: 바깥 손은 좌우 0.32m, 안쪽 손은 몸 가운데 근처에 오고, 두 상자 사이 틈 22cm 로 안쪽 손이 들어간다
# (몸을 고정한 IK 시험에서 이 자리들에 오차 0 으로 닿는 것을 확인, 2026-11-23)
SMALL_BOX_SIZE = (0.20, 0.14, 0.16)  # 앞뒤, 좌우, 높이 (m)
SLOT_Y = 0.18  # 상자 자리: 받침대 가운데에서 좌우로 (m)
WIDE_STAND_SIZE = (0.10, 0.56, 0.75)
SMALL_BOX_NAMES = tuple(f"small_{c}_box" for c in ("red", "blue", "green", "yellow"))


@configclass
class G1VlaPick2EnvCfg(G1VlaPickEnvCfg):
    """1차 장면에 넓은 받침대(`wide_stand`)와 작은 색 상자 4개(`small_<색>_box`)를 더한다. 쓰지 않는 물체는 스크립트가 멀리 치운다."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.assets import RigidObjectCfg

        from .pick_place_cfg import BOX_MASS
        from .vlm_scene_cfg import BOX_COLORS, HIDDEN_POS, STAND_X

        self.scene.wide_stand = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/wide_stand",
            spawn=sim_utils.CuboidCfg(
                size=WIDE_STAND_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.56, 0.60)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(STAND_X, 0.0, WIDE_STAND_SIZE[2] / 2)),
        )
        for i, (color, rgb) in enumerate(BOX_COLORS.items()):
            setattr(self.scene, f"small_{color}_box", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/small_{color}_box",
                spawn=sim_utils.CuboidCfg(
                    size=SMALL_BOX_SIZE,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.6),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(HIDDEN_POS[0] - 2.0, HIDDEN_POS[1] + 0.6 * i, HIDDEN_POS[2])),
            ))
