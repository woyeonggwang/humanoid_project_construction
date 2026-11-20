"""짐 든 채 걷기 전용 이벤트·관측·보상 함수.

발·무릎 순서: 아래 걸음 함수들은 [왼쪽, 오른쪽] 순서를 가정한다.
→ SceneEntityCfg 에 이름을 왼쪽·오른쪽 순으로 적고 preserve_order=True 로 넘긴다.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def hold_joints_at_default(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg):
    """정책이 움직이지 않는 관절의 모터 목표값을 기본 자세(init_state.joint_pos)로 맞춘다.

    동작(action)에 들어가지 않은 관절은 목표값을 설정하는 곳이 없어 0 으로 남는다
    (inspect_g1.py 로 확인, 2026-11-19). 리셋 때마다 목표값을 기본 자세로 넣어 팔이 자세를 유지하게 한다.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    target = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    asset.set_joint_position_target(target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# 걸음 박자 (gait clock)
# ---------------------------------------------------------------------------


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """에피소드 시작부터의 시간으로 정한 걸음 주기 위상 φ ∈ [0, 1). 리셋되면 0 부터 다시 시작."""
    t = env.episode_length_buf.float() * env.step_dt
    return torch.remainder(t / period, 1.0)


def gait_phase_obs(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """관측: 걸음 주기 신호 (sin 2πφ, cos 2πφ)."""
    phi = gait_phase(env, period) * (2.0 * math.pi)
    return torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)


def feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """발 접촉 여부 (N, 발 수). 접촉 센서의 현재 접촉 시간이 0 보다 크면 닿아 있음."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0


def feet_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """발마다의 위상 (N, 2): [왼발 φ, 오른발 (φ+0.5) mod 1]. 각 발은 위상 < stance_ratio 일 때 딛는 구간."""
    phi = gait_phase(env, period)
    return torch.stack([phi, torch.remainder(phi + 0.5, 1.0)], dim=1)


def is_standing_command(env: ManagerBasedRLEnv, command_name: str, stand_threshold: float) -> torch.Tensor:
    """명령이 거의 0 인지 (N,): 선속도 크기·회전 속도 모두 stand_threshold 미만."""
    cmd = env.command_manager.get_command(command_name)
    return (torch.norm(cmd[:, :2], dim=1) < stand_threshold) & (cmd[:, 2].abs() < stand_threshold)


def gait_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    period: float,
    stance_ratio: float,
    stand_threshold: float,
) -> torch.Tensor:
    """걸음 박자 보상: 발마다 '닿아 있어야 할 때 닿고, 떠 있어야 할 때 떠 있으면' +1 (최대 2).

    - 왼발은 φ < stance_ratio 일 때, 오른발은 반 주기 늦게 ((φ+0.5) mod 1 < stance_ratio) 딛는 구간
      → stance_ratio 가 0.5 보다 조금 크면 발을 바꿀 때 양발이 잠깐 같이 닿음
    - 명령이 거의 0 이면(선속도 크기·회전 속도 모두 stand_threshold 미만) 양발 모두 딛는 것이 정답
    """
    contact = feet_contact(env, sensor_cfg)
    expected = feet_phase(env, period) < stance_ratio
    expected = expected | is_standing_command(env, command_name, stand_threshold)[:, None]
    return torch.sum((contact == expected).float(), dim=1)


# ---------------------------------------------------------------------------
# 자세·발 높이·발 간격·대칭
# ---------------------------------------------------------------------------


def stance_knee_range(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    min_angle: float,
    max_angle: float,
) -> torch.Tensor:
    """벌점: 딛고 있는 발 쪽 무릎이 [min_angle, max_angle](rad) 를 벗어난 만큼 (양발 합).

    max_angle 보다 굽으면 웅크린 자세, min_angle 보다 펴지면(과신전) 다리를 잠근 자세.
    """
    asset = env.scene[asset_cfg.name]
    contact = feet_contact(env, sensor_cfg).float()
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]
    excess = torch.clamp(knee - max_angle, min=0.0) + torch.clamp(min_angle - knee, min=0.0)
    return torch.sum(excess * contact, dim=1)


def feet_swing_trajectory_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    period: float,
    stance_ratio: float,
    stand_threshold: float,
    ground_height: float,
    swing_height: float,
) -> torch.Tensor:
    """벌점: 걸음 박자가 '떠 있을 구간'이라고 정한 동안 (발 높이 − 목표 궤적)², 양발 합.

    목표 궤적 = ground_height + swing_height × sin(π × 진행률), 진행률 = 떠 있을 구간 안에서 0 → 1.
    발 높이 = 발 링크 원점(지면 0), ground_height = 딛고 있을 때의 발 링크 원점 높이.
    실제 접촉·발 속도와 관계없이 박자로 구간을 정하므로 발을 안 들거나 톡톡 딛어 피할 수 없음.
    명령이 거의 0 이면 떠 있을 구간이 없으므로 벌점 없음.
    (바꾼 이유: 닿는 순간 보상 → 톡톡 딛기, 떠 있는 동안 높이 벌점 → 끌기, 발 속도 비례 벌점 → 천천히 걸을 때 약함. 2026-11-20 시험 1~3차)
    """
    asset = env.scene[asset_cfg.name]
    phase = feet_phase(env, period)
    in_swing = (phase >= stance_ratio) & ~is_standing_command(env, command_name, stand_threshold)[:, None]
    progress = torch.clamp((phase - stance_ratio) / (1.0 - stance_ratio), 0.0, 1.0)
    target = ground_height + swing_height * torch.sin(math.pi * progress)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return torch.sum(torch.square(foot_z - target) * in_swing.float(), dim=1)


def feet_lateral_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float, max_dist: float
) -> torch.Tensor:
    """벌점: 골반 기준 좌우(y) 발 간격(왼발 − 오른발)이 [min_dist, max_dist] 를 벗어난 만큼 (m).

    부호를 그대로 쓰므로 발이 꼬여 왼발이 오른쪽으로 넘어가면(음수) 큰 벌점.
    """
    asset = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids]
    lateral = quat_apply_inverse(asset.data.root_quat_w, feet[:, 0] - feet[:, 1])[:, 1]
    return torch.clamp(min_dist - lateral, min=0.0) + torch.clamp(lateral - max_dist, min=0.0)


def feet_air_time_variance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """벌점: 좌우 발의 직전 공중 시간·접촉 시간의 분산 (좌우 걸음 대칭).

    Isaac Lab Spot 과제의 air_time_variance_penalty 와 같은 식 (0.5초에서 자름).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    last_air = torch.clip(sensor.data.last_air_time[:, sensor_cfg.body_ids], max=0.5)
    last_contact = torch.clip(sensor.data.last_contact_time[:, sensor_cfg.body_ids], max=0.5)
    return torch.var(last_air, dim=1) + torch.var(last_contact, dim=1)
