"""5단계: 상자 집어 옮기기용 환경 설정.

걷기 정책은 팔 무작위판(G1CarryGaitArmRandFlatEnvCfg)에서 학습한 것을 그대로 쓰므로 관측·동작은 바꾸지 않는다.
팔은 스크립트(IK)가 매 스텝 목표값을 넣으므로 팔 무작위 이벤트와 손 무게 무작위화를 끈다.
리셋 이벤트 hold_arms 는 남겨 둔다: 리셋 직후 팔 목표값을 짐 든 자세로 넣고, 스크립트의 IK 도 같은 값에서 다시 시작한다.
"""

from isaaclab.utils import configclass

from .env_cfg import G1CarryGaitArmRandFlatEnvCfg_PLAY


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
