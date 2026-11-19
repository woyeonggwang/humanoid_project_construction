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
