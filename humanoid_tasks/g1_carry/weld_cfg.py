"""11단계: 용접 장면 설정. 공사장 장면에 용접 토치를 든 로봇을 둔다.

- 토치: 손잡이 + 목 + 노즐 + 팁을 원통으로 만든 겉모습만의 모델. 충돌·질량 없음.
  로봇 USD 를 불러온 직후 오른손(right_palm_link) 아래에 붙인다 → 링크의 자식이라 손을 그대로 따라오고, 머리 카메라에도 보인다.
  환경 묶음을 복제하기 전에 붙이므로 모든 환경에 같이 복제된다.
- 토치의 치수와 기준점(노즐 끝 = tip)은 TORCH 에 모아 둔다 (palm 링크 좌표, m). 용접 IK(weld.py)와 겉모습이 같은 값을 쓴다.
- 과제: Humanoid-G1-Weld-v0 (공사장 A 배치 + 토치, 머리 카메라 포함), Humanoid-G1-Weld-NoCam-v0 (카메라 없음. 용접 동작 평가용),
  Humanoid-G1-Weld-Probe-v0 (공사장 없이 로봇만. 닿는 범위를 몸을 고정하고 잴 때)
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass


from .pick_place_cfg import G1ArmIKV2EnvCfg
from .site_cfg import G1SiteAEnvCfg, G1SiteB1EnvCfg, G1SiteB2EnvCfg

# ---------------------------------------------------------------------------
# 토치 (오른손 palm 링크 좌표: x = 손가락 쪽(아래팔이 뻗은 방향), y = 손바닥 안쪽(손가락이 접히는 쪽), z = 검지 쪽)
# ---------------------------------------------------------------------------
TORCH_BEND_DEG = 45.0  # 목이 손잡이 축(z)에서 손가락 쪽(x)으로 꺾인 각도


def _torch(bend_deg: float = TORCH_BEND_DEG) -> dict:
    b = math.radians(bend_deg)
    axis = (math.sin(b), 0.0, math.cos(b))  # 노즐이 향하는 방향
    grip = (0.060, 0.030, 0.0)  # 손잡이 가운데: 접은 손가락과 손바닥 사이
    top = (grip[0], grip[1], 0.055)  # 손잡이 위 끝 (검지 쪽)
    bottom = (grip[0], grip[1], -0.065)

    def along(d):
        return tuple(t + d * a for t, a in zip(top, axis))

    return {
        "axis": axis,
        "tip": along(0.220),  # 기준점: 팁 끝 (손잡이 위 끝에서 22cm)
        # 겉모습 원통: (이름, 시작점, 끝점, 반지름, 재질)
        "parts": [
            ("handle", bottom, top, 0.013, "grip"),
            ("neck", top, along(0.150), 0.007, "metal"),
            ("nozzle", along(0.150), along(0.205), 0.006, "brass"),
            ("tip", along(0.205), along(0.220), 0.001, "brass"),  # 팁(와이어, 지름 2mm): 용접하는 동안 이 끝이 용접선에서 약 1cm
        ],
    }


TORCH = _torch()
TORCH_PRIM = "right_palm_link/weld_torch"
# 토치를 쥔 손가락 자세 (오른손 관절 목표값, rad): 두 손가락을 손잡이 둘레로 접고 엄지는 맞은편에서 받친다
TORCH_GRIP = {"right_three_joint": 1.45, "right_five_joint": 1.45, "right_four_joint": 1.40, "right_six_joint": 1.40,
              "right_zero_joint": 0.0, "right_one_joint": -1.0, "right_two_joint": -0.75}
_MATERIALS = {"grip": ((0.05, 0.05, 0.06), 0.7, 0.0), "metal": ((0.55, 0.56, 0.58), 0.35, 0.9), "brass": ((0.78, 0.55, 0.22), 0.3, 0.9)}


def add_torch_prims(parent_path: str, torch: dict = TORCH) -> None:
    """parent_path(= 오른손 링크) 아래에 토치 겉모습을 만든다. 물리 속성은 붙이지 않는다."""
    from pxr import Gf, UsdGeom, UsdShade

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    root = f"{parent_path}/{TORCH_PRIM.split('/')[-1]}"
    UsdGeom.Xform.Define(stage, root)
    mats = {}
    for name, (color, rough, metal) in _MATERIALS.items():
        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=rough, metallic=metal)
        cfg.func(f"{root}/mat_{name}", cfg)
        mats[name] = UsdShade.Material(stage.GetPrimAtPath(f"{root}/mat_{name}"))
    for name, p0, p1, radius, mat in torch["parts"]:
        a, b = Gf.Vec3d(*p0), Gf.Vec3d(*p1)
        cyl = UsdGeom.Cylinder.Define(stage, f"{root}/{name}")
        cyl.CreateAxisAttr("Z")
        cyl.CreateHeightAttr((b - a).GetLength())
        cyl.CreateRadiusAttr(radius)
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.AddTranslateOp().Set((a + b) * 0.5)
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), (b - a).GetNormalized()).GetQuat())
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mats[mat])


@clone
def spawn_g1_with_torch(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """로봇 USD 를 불러온 뒤 오른손 링크 아래에 토치를 붙인다 (복제 전, 첫 환경의 로봇에)."""
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    add_torch_prims(f"{prim_path}/right_palm_link")
    return prim


def use_torch_robot(env_cfg) -> None:
    """env_cfg 의 로봇을 '토치를 든 로봇'으로 바꾼다 (원본 설정 객체는 건드리지 않고 복사해서 불러오는 함수만 바꿈)."""
    env_cfg.scene.robot = env_cfg.scene.robot.replace(spawn=env_cfg.scene.robot.spawn.replace(func=spawn_g1_with_torch))


@configclass
class G1WeldEnvCfg(G1SiteAEnvCfg):
    """공사장 A 배치 + 토치를 든 로봇. 9단계 전체 연결 장면의 받침대·상자·판·머리 카메라는 그대로."""

    def __post_init__(self):
        super().__post_init__()
        use_torch_robot(self)
        self.episode_length_s = 600.0


@configclass
class G1WeldNoCamEnvCfg(G1WeldEnvCfg):
    """용접 동작 평가용: 머리 카메라 두 대를 뺀다 (환경을 여러 개 돌릴 때 가볍게)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.vla_cam = None
        self.scene.vlm_cam = None


@configclass
class G1WeldB1EnvCfg(G1SiteB1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_torch_robot(self)
        self.episode_length_s = 600.0


@configclass
class G1WeldB2EnvCfg(G1SiteB2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_torch_robot(self)
        self.episode_length_s = 600.0


@configclass
class G1WeldProbeEnvCfg(G1ArmIKV2EnvCfg):
    """공사장 없이 토치를 든 로봇만 (닿는 범위를 계산으로 잴 때. 철골은 좌표로만 다룸)."""

    def __post_init__(self):
        super().__post_init__()
        use_torch_robot(self)
