"""짐 든 채 걷기 전용 이벤트·관측·보상 함수.

발·무릎 순서: 아래 걸음 함수들은 [왼쪽, 오른쪽] 순서를 가정한다.
→ SceneEntityCfg 에 이름을 왼쪽·오른쪽 순으로 적고 preserve_order=True 로 넘긴다.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


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


def gait_phase_obs(env: ManagerBasedRLEnv, period: float, command_name: str, stand_threshold: float) -> torch.Tensor:
    """관측: 걸음 주기 신호 (sin 2πφ, cos 2πφ). 명령이 거의 0 이면 (0, 0).

    정지 명령에서는 박자 보상의 정답이 '양발 딛기'이므로 박자 신호도 꺼서 걷는 박자와 구분되게 한다
    (신호가 계속 돌면 정지 명령에도 박자대로 제자리걸음. 2026-11-20 시험 7·8차).
    """
    phi = gait_phase(env, period) * (2.0 * math.pi)
    clock = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)
    return clock * (~is_standing_command(env, command_name, stand_threshold)).float()[:, None]


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


class lin_vel_avg_error(ManagerTermBase):
    """벌점: (명령 − 실제) 이동 속도를 약 한 걸음 주기 동안 평균한 값의 크기 (m/s).

    속도 따라가기 보상(exp)은 오차가 작을 때 거의 평평해서, 명령보다 조금 느리게 걸어도 보상을 거의 다 받는다
    (2026-11-20: 0.3m/s 명령에 0.26m/s 로 걸어도 98%). 순간 속도는 걸음마다 출렁이므로 평균한 오차에 벌점을 준다.
    평균은 지수 이동 평균(시간 상수 time_constant 초), 리셋되면 0 부터.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.avg_error = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.avg_error[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        time_constant: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        vel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)[:, :2]
        error = env.command_manager.get_command(command_name)[:, :2] - vel
        self.avg_error += (env.step_dt / time_constant) * (error - self.avg_error)
        return torch.norm(self.avg_error, dim=1)


def stand_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, stand_threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """벌점: 정지 명령일 때 관절이 기본 자세에서 벗어난 정도 (절댓값 합, rad).

    정지 명령에서는 양발만 딛고 있으면 박자 보상을 다 받으므로 서 있는 자세가 정해지지 않는다
    (2026-11-20 본 학습: 발과 상체는 앞을 보는데 골반만 약 36° 돌아간 자세로 섬).
    """
    asset = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(deviation.abs(), dim=1) * is_standing_command(env, command_name, stand_threshold).float()


def stand_flat_orientation_l2(
    env: ManagerBasedRLEnv, command_name: str, stand_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """벌점: 정지 명령일 때 골반이 기운 정도 (골반 좌표계에서 본 중력 방향의 수평 성분 제곱).

    stand_joint_deviation_l1 은 골반·허리를 돌리는 관절만 보므로 앞뒤로 기운 자세는 잡지 못한다
    (2026-11-20 본 학습: 정지 명령에 뒤로 약 16° 젖힌 채 서고, 양손 +4~5kg 에서 넘어짐).
    """
    asset = env.scene[asset_cfg.name]
    tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return tilt * is_standing_command(env, command_name, stand_threshold).float()


def feet_slip(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, max_speed: float
) -> torch.Tensor:
    """벌점: 바닥에 닿아 있는 발의 수평 속도 제곱 (양발 합, 발마다 max_speed² 에서 자름).

    걸음 박자 보상은 '닿음 여부'만 보므로 발을 바닥에 살짝 댄 채 앞으로 끌어도 보상을 받는다.
    2026-11-20 평가에서 한 걸음 약 36cm 중 21cm 를 닿은 채 끌어서 이동 (기준 정책은 1cm).
    닿아 있는 발은 멈춰 있어야 하므로 실제로 발을 들어 옮기는 걸음이 되게 한다.
    제곱이라 닿는 순간의 작은 잔여 속도에는 약하고, 빠르게 끄는 동작에는 강하다.
    """
    asset = env.scene[asset_cfg.name]
    contact = feet_contact(env, sensor_cfg).float()
    speed_sq = torch.sum(torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=-1)
    return torch.sum(torch.clamp(speed_sq, max=max_speed**2) * contact, dim=1)
