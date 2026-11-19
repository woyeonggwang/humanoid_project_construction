"""G1 짐 든 채 걷기 PPO 설정. 1차 학습(G1 Flat)과 비교할 수 있도록 기록 폴더 이름만 바꾼다."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1CarryFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        # 기록 폴더: <실행 위치>\logs\rsl_rl\g1_carry_flat\<날짜시간>
        self.experiment_name = "g1_carry_flat"
