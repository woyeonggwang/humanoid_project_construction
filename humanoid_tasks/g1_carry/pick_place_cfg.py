"""5단계: 상자 집어 옮기기용 환경 설정.

걷기 정책은 팔 무작위판(G1CarryGaitArmRandFlatEnvCfg)에서 학습한 것을 그대로 쓰므로 관측·동작은 바꾸지 않는다.
팔은 스크립트(IK)가 매 스텝 목표값을 넣으므로 팔 무작위 이벤트와 손 무게 무작위화를 끈다.
리셋 이벤트 hold_arms 는 남겨 둔다: 리셋 직후 팔 목표값을 짐 든 자세로 넣고, 스크립트의 IK 도 같은 값에서 다시 시작한다.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .env_cfg import G1CarryGaitArmRandFlatEnvCfg_PLAY

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


@configclass
class G1ArmIKEnvCfg(G1CarryGaitArmRandFlatEnvCfg_PLAY):
    """IK 로 팔을 움직이는 확인용 (상자 없음). 명령은 스크립트가 매 스텝 정한다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.arm_motion = None
        self.events.add_hand_mass = None
        # 속도 명령: 리셋 때 0 으로 뽑히게 하고(정지), 이후에는 스크립트가 덮어씀
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_standing_envs = 0.0
        cmd.rel_turn_envs = 0.0
        cmd.resampling_time_range = (1.0e6, 1.0e6)
        cmd.ranges.lin_vel_x = (0.0, 0.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.debug_vis = False
        self.episode_length_s = 120.0


@configclass
class G1PickPlaceEnvCfg(G1ArmIKEnvCfg):
    """받침대 두 개와 상자를 더한 장면. 걷기·팔·붙이기는 scripts/pick_place_demo.py 가 맡는다."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 6.0  # 환경마다 약 2.3m 를 걸어가므로 넓게
        # 원점에서 앞(+x)을 보고 시작
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.scene.table_a = _table_cfg("TableA", TABLE_A_XY)
        self.scene.table_b = _table_cfg("TableB", TABLE_B_XY)
        self.scene.box = RigidObjectCfg(
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
