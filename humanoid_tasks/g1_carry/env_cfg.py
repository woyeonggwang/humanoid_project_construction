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
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

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
