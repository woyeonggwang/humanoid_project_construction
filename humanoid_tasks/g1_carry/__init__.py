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

# 근접 비교 영상용 (한 대, 옆에서 따라가는 카메라, scripts/record_compare.py 에서 사용)
gym.register(
    id="Humanoid-G1-Carry-Flat-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1CarryFlatEnvCfg_RECORD",
        "rsl_rl_cfg_entry_point": _AGENT_CFG,
    },
)

gym.register(
    id="Humanoid-G1-Carry-Gait-Flat-Record-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1CarryGaitFlatEnvCfg_RECORD",
        "rsl_rl_cfg_entry_point": _GAIT_AGENT_CFG,
    },
)

# 팔이 움직여도 버티는 걷기·서기 (12차 설정 + 어깨·팔꿈치 목표 자세 무작위화)
_ARMRAND_AGENT_CFG = "humanoid_tasks.g1_carry.agents:G1CarryGaitArmRandFlatPPORunnerCfg"

gym.register(
    id="Humanoid-G1-Carry-Gait-ArmRand-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1CarryGaitArmRandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": _ARMRAND_AGENT_CFG,
    },
)

# 5단계: 상자 집어 옮기기 (걷기 정책은 팔 무작위판 그대로, 팔은 스크립트의 IK 가 움직임)
_PICK_CFG = "humanoid_tasks.g1_carry.pick_place_cfg"

gym.register(
    id="Humanoid-G1-ArmIK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PICK_CFG}:G1ArmIKEnvCfg",
        "rsl_rl_cfg_entry_point": _ARMRAND_AGENT_CFG,
    },
)

gym.register(
    id="Humanoid-G1-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PICK_CFG}:G1PickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": _ARMRAND_AGENT_CFG,
    },
)

gym.register(
    id="Humanoid-G1-Carry-Gait-ArmRand-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1CarryGaitArmRandFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _ARMRAND_AGENT_CFG,
    },
)

# 걷기 정책 v2: 팔 관절 전체·손가락 무작위화, 뒤로 걷기, 걸음을 마무리한 뒤 정지, 걸을 때 뒤로 젖히기 억제
_LOCO_V2_AGENT_CFG = "humanoid_tasks.g1_carry.agents:G1LocoV2FlatPPORunnerCfg"

gym.register(
    id="Humanoid-G1-Loco-V2-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1LocoV2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": _LOCO_V2_AGENT_CFG,
    },
)

gym.register(
    id="Humanoid-G1-Loco-V2-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:G1LocoV2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _LOCO_V2_AGENT_CFG,
    },
)

# 걷기 정책 v2 로 상자 집어 옮기기 (6단계), VLM 판단 층용 장면 (7단계: 받침대 3개·색 상자 4개·머리 카메라)
gym.register(
    id="Humanoid-G1-ArmIK-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PICK_CFG}:G1ArmIKV2EnvCfg",
        "rsl_rl_cfg_entry_point": _LOCO_V2_AGENT_CFG,
    },
)

gym.register(
    id="Humanoid-G1-PickPlace-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PICK_CFG}:G1PickPlaceV2EnvCfg",
        "rsl_rl_cfg_entry_point": _LOCO_V2_AGENT_CFG,
    },
)

gym.register(
    id="Humanoid-G1-VLM-Scene-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "humanoid_tasks.g1_carry.vlm_scene_cfg:G1VlmSceneEnvCfg",
        "rsl_rl_cfg_entry_point": _LOCO_V2_AGENT_CFG,
    },
)
