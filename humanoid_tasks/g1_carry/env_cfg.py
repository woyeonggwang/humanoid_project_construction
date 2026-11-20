"""G1 짐 든 채 걷기 환경 설정 (Isaac Lab G1 Flat 상속).

G1 Flat 과 달라지는 점
1. 정책이 움직이는 관절: 37개(팔·손가락 포함) → 다리 12 + 허리 1
2. 팔: 시작 자세를 '앞으로 내밀어 짐 든 자세'로 바꾸고, 정책이 건드리지 않음
   (동작에서 빠진 관절은 목표값이 0 으로 남으므로 hold_arms 이벤트가 리셋마다 목표값을 기본 자세로 넣음)
3. 손에 0~5kg 무게를 무작위로 더하고, 몸통 무게중심을 조금씩 무작위로 바꿈
4. 움직이지 않는 팔·손가락 관련 벌점 제거
5. 원본 구동기 'arms'(어깨·팔꿈치·손가락 한 그룹, 강성 40)를 어깨·팔꿈치(강성을 올림)와 손가락(원본 값)으로 나눔
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg, ObservationsCfg

from . import carry_mdp

# 정책이 움직이는 관절 (다리 12 + 허리 1)
LEG_JOINTS = [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint",
              ".*_ankle_pitch_joint", ".*_ankle_roll_joint", "torso_joint"]

# 짐 든 자세 (라디안). 어깨 pitch 음수 = 팔이 앞으로 (2026-11-19 inspect_g1.py·시험 영상으로 확인:
# 리셋 직후 손이 골반 기준 앞 0.28m, 높이는 골반과 같음, 좌우 간격 0.42m)
CARRY_POSE = {
    "left_shoulder_pitch_joint": -0.5,
    "right_shoulder_pitch_joint": -0.5,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_pitch_joint": 1.0,
    ".*_elbow_roll_joint": 0.0,
}

# 정책이 움직이지 않고 기본 자세로 고정할 관절 (팔 10 + 손가락 14)
HOLD_JOINTS = [".*_shoulder_.*", ".*_elbow_.*", ".*_zero_joint", ".*_one_joint", ".*_two_joint",
               ".*_three_joint", ".*_four_joint", ".*_five_joint", ".*_six_joint"]

# 무게를 더할 손 링크 (inspect_g1.py 로 확인, 2026-11-19: 각 0.281kg)
HAND_LINKS = ["left_palm_link", "right_palm_link"]
HAND_MASS_RANGE = (0.0, 5.0)  # kg, 손 하나당 더하는 무게

# 원본 구동기 'arms' 를 나눌 관절 (원본 joint_names_expr 를 그대로 둘로 나눔)
ARM_JOINTS = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
              ".*_elbow_pitch_joint", ".*_elbow_roll_joint"]
FINGER_JOINTS = [".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint",
                 ".*_zero_joint", ".*_one_joint", ".*_two_joint"]
# 어깨·팔꿈치 PD 이득 (2026-11-19 inspect_g1.py --fix_base 로 비교해 결정, 근거는 CLAUDE.md)
# 강성: 손 5.28kg 에서 어깨 pitch 중력 토크 약 16Nm → 강성 40 이면 처짐 약 0.3 rad, 200 이면 약 0.1 rad
# 감쇠: 무거운 손 기준 임계 감쇠 약 40 → 30 이면 넘침 약 4% (20 은 17%, 10 은 42%)
ARM_STIFFNESS = 200.0
ARM_DAMPING = 30.0


@configclass
class CarryEventCfg(EventCfg):
    """기존 이벤트 + 팔 자세 고정(리셋마다) + 손 무게 무작위화(시작할 때 환경마다 한 번)."""

    hold_arms = EventTerm(
        func=carry_mdp.hold_joints_at_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HOLD_JOINTS)},
    )

    add_hand_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=HAND_LINKS),
            "mass_distribution_params": HAND_MASS_RANGE,
            "operation": "add",
        },
    )


@configclass
class G1CarryFlatEnvCfg(G1FlatEnvCfg):
    events: CarryEventCfg = CarryEventCfg()

    def __post_init__(self):
        # G1 Flat 설정(지형·보상 가중치·명령 범위 등)을 먼저 전부 적용
        super().__post_init__()

        # 1. 팔 시작 자세: 기존 어깨·팔꿈치 항목을 빼고 짐 든 자세로 교체
        #    (같은 관절에 정규식 두 개가 겹치면 오류가 나므로 기존 항목을 지운다)
        robot = self.scene.robot
        joint_pos = {k: v for k, v in robot.init_state.joint_pos.items() if "shoulder" not in k and "elbow" not in k}
        joint_pos.update(CARRY_POSE)

        # 5. 구동기 'arms' 를 어깨·팔꿈치와 손가락으로 나눔
        #    새 dict 를 만들어 원본 G1 설정(모듈 전역 객체)의 actuators 는 건드리지 않는다
        #    armature 는 원본 값(어깨·팔꿈치 0.01, 손가락 0.001)을 그룹별 숫자로 넣음
        actuators = {k: v for k, v in robot.actuators.items() if k != "arms"}
        arms = robot.actuators["arms"]
        actuators["arms"] = arms.replace(
            joint_names_expr=ARM_JOINTS, stiffness=ARM_STIFFNESS, damping=ARM_DAMPING, armature=0.01
        )
        actuators["fingers"] = arms.replace(joint_names_expr=FINGER_JOINTS, armature=0.001)  # 강성 40, 감쇠 10 그대로

        self.scene.robot = robot.replace(
            init_state=robot.init_state.replace(joint_pos=joint_pos), actuators=actuators
        )

        # 2. 정책은 다리와 허리만 움직임
        self.actions.joint_pos.joint_names = LEG_JOINTS

        # 3. 몸통 무게중심 무작위화 (G1 Flat 에서는 꺼져 있음)
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
            },
        )

        # 4. 정책이 움직이지 않는 관절의 벌점 제거
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_fingers = None


@configclass
class G1CarryFlatEnvCfg_PLAY(G1CarryFlatEnvCfg):
    """영상 확인용 (G1FlatEnvCfg_PLAY 와 같은 변경)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ===========================================================================
# 걸음 보상 추가판 (Humanoid-G1-Carry-Gait-Flat-v0, gait_reward 브랜치)
# 짐 든 채 걷기 설정은 그대로 두고, 사람에 가까운 걸음을 위한 관측·보상을 더한다.
# 가중치·목표값은 시작값 (eval_carry.py 걸음 지표와 시험 학습을 보며 조정, 확인 필요)
# ===========================================================================

GAIT_PERIOD = 0.8  # 걸음 주기(초). 한 주기에 좌우 한 걸음씩 → 2.5걸음/초 (기준 정책은 5~8걸음/초)
STANCE_RATIO = 0.55  # 한 주기 중 한 발이 딛는 비율. 0.5 보다 조금 크게 → 발을 바꿀 때 양발이 잠깐 같이 닿음
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]  # [왼쪽, 오른쪽] 순서 (carry_mdp 의 걸음 함수가 가정)
KNEES = ["left_knee_joint", "right_knee_joint"]
FEET_SENSOR = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
FEET_ASSET = SceneEntityCfg("robot", body_names=FEET, preserve_order=True)


@configclass
class GaitObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # 걸음 주기 신호 (sin, cos) — 관측 맨 끝에 붙음
        gait_phase = ObsTerm(func=carry_mdp.gait_phase_obs, params={"period": GAIT_PERIOD})

    policy: PolicyCfg = PolicyCfg()


@configclass
class GaitRewardsCfg(G1Rewards):
    # 걸음 박자: 주기 앞부분은 왼발, 반 주기 늦게 오른발이 딛으면 보상 (명령 0 이면 양발 딛기가 정답)
    gait_contact = RewTerm(
        func=carry_mdp.gait_contact,
        weight=0.3,
        params={
            "sensor_cfg": FEET_SENSOR,
            "command_name": "base_velocity",
            "period": GAIT_PERIOD,
            "stance_ratio": STANCE_RATIO,
            "stand_threshold": 0.1,
        },
    )
    # 딛는 발 무릎: 0.05~0.3 rad(약 3~17°) 밖이면 벌점 (기준 정책은 약 45° 웅크림, 상한만 둔 2차 시험은 과신전 -1~-4°)
    stance_knee = RewTerm(
        func=carry_mdp.stance_knee_range,
        weight=-1.0,
        params={
            "sensor_cfg": FEET_SENSOR,
            "asset_cfg": SceneEntityCfg("robot", joint_names=KNEES, preserve_order=True),
            "min_angle": 0.05,
            "max_angle": 0.3,
        },
    )
    # 골반 높이: 0.72m 근처 유지 (기준 정책 약 0.64m. 0.74m 목표는 무릎을 완전히 펴야 해서 낮춤, 2026-11-20)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.72})
    # 발 높이: 박자가 '떠 있을 구간'이라고 정한 동안 발 링크 원점이 0.04m(딛는 높이) → 0.12m(+8cm) → 0.04m
    # 사인 궤적을 따르도록 차이 제곱 벌점 (시험 1~3차 방식의 허점 때문에 바꿈, 2026-11-20)
    feet_swing_height = RewTerm(
        func=carry_mdp.feet_swing_trajectory_l2,
        weight=-30.0,
        params={
            "asset_cfg": FEET_ASSET,
            "command_name": "base_velocity",
            "period": GAIT_PERIOD,
            "stance_ratio": STANCE_RATIO,
            "stand_threshold": 0.1,
            "ground_height": 0.04,
            "swing_height": 0.08,
        },
    )
    # 발 간격: 골반 기준 좌우 15~30cm 밖이면 벌점 (기준 정책은 20~23cm, 발 꼬임 방지)
    feet_distance = RewTerm(
        func=carry_mdp.feet_lateral_distance,
        weight=-10.0,
        params={"asset_cfg": FEET_ASSET, "min_dist": 0.15, "max_dist": 0.30},
    )
    # 좌우 대칭: 좌우 발의 공중·접촉 시간 분산
    feet_symmetry = RewTerm(func=carry_mdp.feet_air_time_variance, weight=-10.0, params={"sensor_cfg": FEET_SENSOR})


@configclass
class G1CarryGaitFlatEnvCfg(G1CarryFlatEnvCfg):
    observations: GaitObservationsCfg = GaitObservationsCfg()
    rewards: GaitRewardsCfg = GaitRewardsCfg()

    def __post_init__(self):
        # 짐 든 채 걷기 설정(G1 Flat 변경 포함)을 먼저 전부 적용
        super().__post_init__()
        # 발을 오래 띄울수록 주던 보상을 크게 낮춤 (G1 Flat 0.75) — 발 높이는 feet_swing_height 가 맡음
        self.rewards.feet_air_time.weight = 0.1
        # 회전 연습: 환경의 절반은 방향 오차 대신 회전 속도(-1~1 rad/s)를 직접 뽑아 다시 뽑을 때(10초)까지 유지
        # (원래 1.0 = 모든 환경이 '방향 오차 × 0.5' 로 회전 → 오래 도는 상황이 드묾. 시험 3차에서 제자리 회전 중 넘어짐 62%)
        self.commands.base_velocity.rel_heading_envs = 0.5


@configclass
class G1CarryGaitFlatEnvCfg_PLAY(G1CarryGaitFlatEnvCfg):
    """영상 확인용 (G1CarryFlatEnvCfg_PLAY 와 같은 변경)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
