"""10단계: 공사장 장면 설정. 9단계 전체 연결 장면(G1VlaFullEnvCfg)을 상속해 공사장 모델(assets/usd)을 더한다.

- 받침대·상자·판·카메라·로봇·바닥(물리)은 9단계 그대로. 더하는 것은 겉모습(USD 모델)과 단순 상자 충돌뿐
- 모델을 놓는 곳: 바닥을 만드는 쪽(SiteTerrainImporter)이 환경 원점마다 /World/Site/env_<번호>/ 아래에 불러온다.
  환경 원점이 정해진 뒤, 시뮬레이션이 시작되기 전에 놓아야 하고, 환경 묶음(/World/envs) 밖에 두어야 하기 때문 (아래)
- 환경을 한 줄로: 원본의 격자 배치에서는 앞쪽 환경의 공사장(높은 비계·사무실)이 뒤쪽 환경의 카메라에 보이고 바닥 반사도 바꾼다
  (카메라의 그리는 거리 15m 설정으로는 가려지지 않았음). 그래서 환경을 로봇이 보는 방향(+x)과 직각인 y 축을 따라 한 줄로 세운다
- 충돌: 로봇이 부딪힐 수 있는 모델에만, 보이지 않는 상자(움직이지 않는 충돌체)를 같은 자리에 따로 둔다. 모델 파일은 고치지 않는다
- 배치 자료와 용접선 좌표는 site_layout.py

과제: Humanoid-G1-Site-A-v0 (A 배치: 두 카메라에 보이지 않는 곳에만),
      Humanoid-G1-Site-B1-v0 (A + VLM 카메라의 배경에 보이는 펜스·자재), Humanoid-G1-Site-B2-v0 (B1 + 콘크리트 바닥 겉모습),
      Humanoid-G1-Site-Empty-v0 (비교용: 환경 배치만 같고 모델은 없음)
"""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import create_prim, get_current_stage
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from . import site_layout as sl
from .vla_place_cfg import G1VlaFullEnvCfg

SITE_ROOT = "/World/Site"
SITE_ENV_SPACING = 80.0  # 한 줄로 세운 환경 사이 거리 (m). 공사장 너비 12m
# 줄의 기준점: 바닥 격자 무늬가 로봇 발밑에서 9단계의 학습·평가 때와 같은 자리에 오도록 원본 격자 배치(간격 30m → 원점이 15m 의 홀수 배)와 맞춤
SITE_LINE_XY = (15.0, 15.0)


def _yaw_quat(yaw_deg: float):
    half = math.radians(yaw_deg) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def spawn_site(env_origins, placements, z: float = 0.0, collide: bool = True) -> None:
    """환경 원점마다 placements [(이름, 모델, (x, y), yaw 도)] 를 불러온다. 모델은 <SITE_ROOT>/env_<i>/<이름>, 충돌 상자는 <이름>_col<k>."""
    stage = get_current_stage()
    for i, origin in enumerate(env_origins.tolist()):
        root = f"{SITE_ROOT}/env_{i}"
        for path in (SITE_ROOT, root):  # 묶음을 여러 번 놓을 수 있으므로 없을 때만 만든다
            if not stage.GetPrimAtPath(path).IsValid():
                create_prim(path, "Xform")
        for name, model, xy, yaw in placements:
            usd_cfg = sim_utils.UsdFileCfg(usd_path=sl.model_path(model))
            usd_cfg.func(f"{root}/{name}", usd_cfg, translation=(origin[0] + xy[0], origin[1] + xy[1], origin[2] + z),
                         orientation=_yaw_quat(yaw))
            if not collide:
                continue
            for k, (center, size, box_yaw) in enumerate(sl.colliders_of(model, (xy[0], xy[1], z), yaw)):
                box_cfg = sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(), visible=False)
                box_cfg.func(f"{root}/{name}_col{k}", box_cfg, translation=tuple(o + c for o, c in zip(origin, center)),
                             orientation=_yaw_quat(box_yaw))


class SiteTerrainImporter(TerrainImporter):
    """원본 바닥(평면)을 그대로 만들고, 환경을 y 축을 따라 한 줄로 세운 뒤, 환경 원점마다 공사장 모델을 놓는다."""

    def __init__(self, cfg: "SiteTerrainImporterCfg"):
        super().__init__(cfg)
        for placements, z, collide in cfg.site_layers:
            spawn_site(self.env_origins, placements, z, collide)

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        origins = torch.zeros(num_envs, 3, device=self.device)
        origins[:, 0] = SITE_LINE_XY[0]
        origins[:, 1] = SITE_LINE_XY[1] + (torch.arange(num_envs, device=self.device) - num_envs // 2) * env_spacing
        return origins


@configclass
class SiteTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = SiteTerrainImporter
    site_layers: list = []
    """놓을 모델 묶음들: [(placements, 높이 z, 충돌을 붙일지)]"""


def use_site_terrain(env_cfg, layers) -> None:
    """env_cfg 의 바닥 설정을 값은 그대로 둔 채 SiteTerrainImporterCfg 로 바꾸고 공사장 모델 묶음을 넣는다."""
    old = env_cfg.scene.terrain
    new = SiteTerrainImporterCfg()
    for key, value in old.__dict__.items():
        if key != "class_type":
            setattr(new, key, value)
    new.site_layers = list(layers)
    env_cfg.scene.terrain = new
    env_cfg.scene.env_spacing = SITE_ENV_SPACING


@configclass
class G1SiteEmptyEnvCfg(G1VlaFullEnvCfg):
    """비교용: 환경을 한 줄로 세운 배치만 같고 공사장 모델은 없는 장면 (모델의 영향과 배치의 영향을 나눠 볼 때)."""

    def __post_init__(self):
        super().__post_init__()
        use_site_terrain(self, [])


@configclass
class G1SiteAEnvCfg(G1VlaFullEnvCfg):
    """9단계 전체 연결 장면 + 공사장 모델 (A 배치: 로봇 처음 자리의 뒤쪽·옆쪽, 두 카메라에 보이지 않는 곳)."""

    def __post_init__(self):
        super().__post_init__()
        use_site_terrain(self, [(sl.PLACEMENTS_A, 0.0, True)])


@configclass
class G1SiteB1EnvCfg(G1VlaFullEnvCfg):
    """A 배치 + 카메라에 보이는 배치 후보 1: 받침대 뒤쪽에 펜스를 둘러 작업 구역을 닫고, 그 안쪽 먼 곳에 자재 더미·안전 고깔을 둔다 (VLM 카메라의 배경)."""

    def __post_init__(self):
        super().__post_init__()
        use_site_terrain(self, [(sl.PLACEMENTS_A, 0.0, True), (sl.PLACEMENTS_B_VISIBLE, 0.0, True)])


@configclass
class G1SiteB2EnvCfg(G1VlaFullEnvCfg):
    """후보 2: 후보 1 + 콘크리트 바닥 겉모습 (두 카메라 모두 바닥이 달라짐). 물리 바닥은 그대로."""

    def __post_init__(self):
        super().__post_init__()
        use_site_terrain(self, [(sl.PLACEMENTS_A, 0.0, True), (sl.PLACEMENTS_B_VISIBLE, 0.0, True),
                                (sl.floor_placements("SM_Floor_Concrete"), sl.FLOOR_Z, False)])


@configclass
class G1SiteB3EnvCfg(G1VlaFullEnvCfg):
    """후보 1 + 흙 바닥 겉모습 (11단계: VLM 학습 사진의 배경을 섞을 때 씀). 물리 바닥은 그대로."""

    def __post_init__(self):
        super().__post_init__()
        use_site_terrain(self, [(sl.PLACEMENTS_A, 0.0, True), (sl.PLACEMENTS_B_VISIBLE, 0.0, True),
                                (sl.floor_placements("SM_Floor_Dirt"), sl.FLOOR_Z, False)])
