"""5·6단계: 상자 집어 옮기기용 환경 설정.

팔은 스크립트(IK)가 매 스텝 목표값을 넣으므로 팔 무작위 이벤트와 손 무게 무작위화를 끈다.
리셋 이벤트 hold_arms 는 남겨 둔다: 리셋 직후 팔 목표값을 짐 든 자세로 넣고, 스크립트의 IK 도 같은 값에서 다시 시작한다.

- G1ArmIKEnvCfg / G1PickPlaceEnvCfg: 5단계. 걷기 정책 = 팔 무작위판(G1CarryGaitArmRandFlatEnvCfg)
- G1ArmIKV2EnvCfg / G1PickPlaceV2EnvCfg: 6단계. 걷기 정책 = v2(G1LocoV2FlatEnvCfg). 관측의 박자 신호가
  '걸음을 마무리한 뒤 정지' 방식이라 v2 정책은 v2 설정에서만 쓴다
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .env_cfg import G1CarryGaitArmRandFlatEnvCfg_PLAY, G1LocoV2FlatEnvCfg_PLAY

# ---------------------------------------------------------------------------
# 상자·받침대 배치 (환경 원점 기준 m). 로봇은 원점에서 +x 를 보고 시작한다.
# 순서: 받침대 A 앞까지 걷기 → 상자 집기 → 뒤로 돌아 받침대 B 앞까지 걷기 → 내려놓기
# ---------------------------------------------------------------------------
TABLE_SIZE = (0.10, 0.30, 0.75)  # 받침대 (앞뒤, 좌우, 높이). 앞뒤를 좁게 해서 로봇 발이 닿지 않게 함
BOX_SIZE = (0.20, 0.30, 0.16)  # 상자 (앞뒤, 좌우, 높이). 받침대보다 앞뒤로 5cm 씩 나옴
BOX_MASS = 3.0  # kg
TABLE_A_XY = (1.20, 0.0)  # 상자가 처음 놓인 받침대
TABLE_B_XY = (-1.90, 0.0)  # 내려놓을 받침대
BOX_Z_ON_TABLE = TABLE_SIZE[2] + BOX_SIZE[2] / 2  # 받침대 위에 놓인 상자 중심 높이


def _table_cfg(name: str, xy: tuple[float, float]) -> RigidObjectCfg:
    """받침대: 움직이지 않는 강체(kinematic). 스크립트가 시작할 때 위치를 옮길 수 있게 RigidObject 로 둔다."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.47, 0.52)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], TABLE_SIZE[2] / 2)),
    )


def apply_script_control(cfg):
    """스크립트가 팔과 속도 명령을 직접 정하도록: 팔 무작위·손 무게 무작위를 끄고, 속도 명령은 리셋 때 0(정지)으로."""
    cfg.events.arm_motion = None
    cfg.events.add_hand_mass = None
    cmd = cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.rel_turn_envs = 0.0
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.ranges.lin_vel_x = (0.0, 0.0)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.debug_vis = False
    cfg.episode_length_s = 120.0


def add_pick_place_scene(cfg):
    """받침대 A·B 와 상자를 더하고, 로봇이 원점에서 앞(+x)을 보고 시작하게 한다."""
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 6.0  # 환경마다 약 2.3m 를 걸어가므로 넓게
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.scene.table_a = _table_cfg("TableA", TABLE_A_XY)
    cfg.scene.table_b = _table_cfg("TableB", TABLE_B_XY)
    cfg.scene.box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.10)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(TABLE_A_XY[0], TABLE_A_XY[1], BOX_Z_ON_TABLE)),
    )


@configclass
class G1ArmIKEnvCfg(G1CarryGaitArmRandFlatEnvCfg_PLAY):
    """5단계: IK 로 팔을 움직이는 확인용 (상자 없음). 명령은 스크립트가 매 스텝 정한다."""

    def __post_init__(self):
        super().__post_init__()
        apply_script_control(self)


@configclass
class G1PickPlaceEnvCfg(G1ArmIKEnvCfg):
    """5단계: 받침대 두 개와 상자를 더한 장면. 걷기·팔·붙이기는 scripts/pick_place_demo.py 가 맡는다."""

    def __post_init__(self):
        super().__post_init__()
        add_pick_place_scene(self)


@configclass
class G1ArmIKV2EnvCfg(G1LocoV2FlatEnvCfg_PLAY):
    """6단계: 걷기 정책 v2 로 같은 확인을 할 때."""

    def __post_init__(self):
        super().__post_init__()
        apply_script_control(self)


@configclass
class G1PickPlaceV2EnvCfg(G1ArmIKV2EnvCfg):
    """6단계: 걷기 정책 v2 + 받침대 두 개와 상자."""

    def __post_init__(self):
        super().__post_init__()
        add_pick_place_scene(self)
