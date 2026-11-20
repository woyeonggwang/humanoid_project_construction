"""G1 짐 든 채 걷기 과제 등록.

Isaac Sim 앱이 켜지기 전에 import 되므로 isaaclab 모듈은 import 하지 않고
설정 위치를 문자열로만 넘긴다. 실제 설정 클래스는 앱이 켜진 뒤 train.py/play.py 가 불러온다.
"""

import gymnasium as gym

_ENV_CFG = "humanoid_tasks.g1_carry.env_cfg"
_AGENT_CFG = "humanoid_tasks.g1_carry.agents:G1CarryFlatPPORunnerCfg"

gym.register(
    id="Humanoid-G1-Carry-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1CarryFlatEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT_CFG,
    },
)

gym.register(
    id="Humanoid-G1-Carry-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1CarryFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _AGENT_CFG,
    },
)

# 걸음 보상 추가판 (사람에 가까운 걸음: 걸음 박자·무릎·골반 높이·발 높이 등)
_GAIT_AGENT_CFG = "humanoid_tasks.g1_carry.agents:G1CarryGaitFlatPPORunnerCfg"

gym.register(
    id="Humanoid-G1-Carry-Gait-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1CarryGaitFlatEnvCfg",
        "rsl_rl_cfg_entry_point": _GAIT_AGENT_CFG,
    },
)

gym.register(
    id="Humanoid-G1-Carry-Gait-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1CarryGaitFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _GAIT_AGENT_CFG,
    },
)
