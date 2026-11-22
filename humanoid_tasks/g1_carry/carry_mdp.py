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
    """명령이 거의 0 인지 (N,): 선속도 크기·회전 속도 모두 stand_threshold 미만.

    환경 설정에 `stand_latch_period`(걸음 주기, 초)가 있으면 '걸음을 마무리한 뒤의 정지 상태'를 돌려준다 (걷기 정책 v2):
    정지 명령이 들어와도 박자상 양발이 함께 닿는 구간이 올 때까지는 걷는 중으로 보고, 그 구간에서 정지 상태로 바뀐다.
    명령이 다시 걷기로 바뀌면 바로 풀린다. 박자 신호·박자 보상·서 있는 자세 벌점이 모두 이 함수를 쓰므로 함께 바뀐다.
    """
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :2], dim=1) < stand_threshold) & (cmd[:, 2].abs() < stand_threshold)
    period = getattr(env.cfg, "stand_latch_period", None)
    if period is None:
        return standing
    return _latched_standing(env, standing, period, getattr(env.cfg, "stand_latch_overlap", 0.05))


def _latched_standing(env: ManagerBasedRLEnv, standing_cmd: torch.Tensor, period: float, overlap: float) -> torch.Tensor:
    """정지 명령 → 정지 상태 전환을 '양발이 함께 닿는 박자 구간'까지 미룬 값. 한 스텝에 한 번만 갱신한다.

    박자상 양발이 함께 닿는 구간: 위상 φ 가 [0, overlap) 또는 [0.5, 0.5 + overlap) (왼발 딛기 φ < 0.5 + overlap, 오른발은 반 주기 뒤).
    리셋 직후(에피소드 0스텝)는 미루지 않고 명령 그대로 본다.
    """
    step = int(env.common_step_counter)
    latch = getattr(env, "_stand_latch", None)
    if latch is None or latch.shape != standing_cmd.shape:
        env._stand_latch = standing_cmd.clone()
        env._stand_latch_step = step
    elif env._stand_latch_step != step:
        phi = gait_phase(env, period)
        both_feet = (phi < overlap + 0.01) | ((phi >= 0.5) & (phi < 0.5 + overlap + 0.01))
        env._stand_latch = standing_cmd & (latch | both_feet)
        env._stand_latch_step = step
    return torch.where(env.episode_length_buf == 0, standing_cmd, env._stand_latch)


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


# ---------------------------------------------------------------------------
# 팔 목표 자세 무작위화 (팔이 움직여도 버티는 걷기·서기, arm_rand)
# ---------------------------------------------------------------------------

# 무작위로 바꾸는 관절 순서: [왼 어깨 pitch, 오른 어깨 pitch, 왼 팔꿈치 pitch, 오른 팔꿈치 pitch]
ARM_TARGET_JOINTS = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
]


def arm_goal_from_uniform(
    u: torch.Tensor,
    carry: torch.Tensor,
    shoulder_range: tuple[float, float],
    elbow_range: tuple[float, float],
    random_prob: float | torch.Tensor,
    symmetric_prob: float,
) -> torch.Tensor:
    """0~1 균일 난수 u (n, 6) → 팔 목표 자세 (n, 4), 순서는 ARM_TARGET_JOINTS.

    - u[:, 0] < random_prob 이면 무작위 자세, 아니면 짐 든 자세(carry, (n, 4))로 돌아감 (random_prob 는 숫자 또는 환경별 (n,) 텐서)
    - u[:, 1] < symmetric_prob 이면 양팔 같은 자세 (상자를 양손으로 드는 경우), 아니면 좌우 따로
    - u[:, 2:4] = 왼팔 (어깨, 팔꿈치), u[:, 4:6] = 오른팔
    학습 이벤트(randomize_arm_targets)와 평가 스크립트(eval_carry.py --arm_motion)가 같은 규칙을 쓰도록 함수로 둠.
    """
    s_lo, s_hi = shoulder_range
    e_lo, e_hi = elbow_range
    left_s = s_lo + (s_hi - s_lo) * u[:, 2]
    left_e = e_lo + (e_hi - e_lo) * u[:, 3]
    sym = u[:, 1] < symmetric_prob
    right_s = torch.where(sym, left_s, s_lo + (s_hi - s_lo) * u[:, 4])
    right_e = torch.where(sym, left_e, e_lo + (e_hi - e_lo) * u[:, 5])
    rand_goal = torch.stack([left_s, right_s, left_e, right_e], dim=1)
    return torch.where((u[:, 0] < random_prob)[:, None], rand_goal, carry)


def step_toward(current: torch.Tensor, goal: torch.Tensor, max_step: float) -> torch.Tensor:
    """current 를 goal 쪽으로 한 스텝에 최대 max_step 만큼 옮긴 값 (관절 목표값이 순간 이동하지 않게)."""
    return current + torch.clamp(goal - current, -max_step, max_step)


class randomize_arm_targets(ManagerTermBase):
    """interval 이벤트 (매 스텝 호출): 어깨·팔꿈치 pitch 의 모터 목표값을 무작위 자세로 옮긴다.

    - 환경마다 change_interval_s 범위(초)에서 뽑은 간격으로 새 목표 자세를 정함
      (random_prob 확률로 무작위 자세, 나머지는 짐 든 자세로 돌아감. 정지 명령인 환경은 random_prob_standing 확률)
    - 목표값은 한 번에 바꾸지 않고 max_speed(rad/s) 로 옮김 (IK 로 팔을 내리는 동작처럼 연속으로 움직이게)
    - 리셋되면 짐 든 자세에서 다시 시작 (리셋 이벤트 hold_arms 가 넣는 목표값과 같음)
    매 스텝 목표값을 직접 옮겨야 하므로 interval_range_s=(0, 0), is_global_time=True 로 등록해 매 스텝 부르고,
    환경별 간격은 이 클래스가 따로 센다.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.joint_ids, _ = self.asset.find_joints(ARM_TARGET_JOINTS, preserve_order=True)
        self.carry = self.asset.data.default_joint_pos[:, self.joint_ids].clone()  # (N, 4) 짐 든 자세
        self.goal = self.carry.clone()
        self.current = self.carry.clone()
        self.time_left = self._sample_interval(env.num_envs, cfg.params["change_interval_s"])

    def _sample_interval(self, n: int, interval: tuple[float, float]) -> torch.Tensor:
        lo, hi = interval
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
            n = self.num_envs
        else:
            n = len(env_ids)
        self.goal[env_ids] = self.carry[env_ids]
        self.current[env_ids] = self.carry[env_ids]
        self.time_left[env_ids] = self._sample_interval(n, self.cfg.params["change_interval_s"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        change_interval_s: tuple[float, float],
        random_prob: float,
        symmetric_prob: float,
        shoulder_range: tuple[float, float],
        elbow_range: tuple[float, float],
        max_speed: float,
        random_prob_standing: float | None = None,
        command_name: str = "base_velocity",
        stand_threshold: float = 0.1,
    ):
        self.time_left -= env.step_dt
        due = (self.time_left <= 0.0).nonzero().flatten()
        if len(due) > 0:
            u = torch.rand(len(due), 6, device=self.device)
            prob = random_prob
            if random_prob_standing is not None:  # 서 있는 환경은 팔을 더 자주 움직임
                standing = is_standing_command(env, command_name, stand_threshold)[due]
                prob = torch.where(standing, random_prob_standing, random_prob)
            self.goal[due] = arm_goal_from_uniform(
                u, self.carry[due], shoulder_range, elbow_range, prob, symmetric_prob
            )
            self.time_left[due] = self._sample_interval(len(due), change_interval_s)
        self.current = step_toward(self.current, self.goal, max_speed * env.step_dt)
        self.asset.set_joint_position_target(self.current, joint_ids=self.joint_ids)


class stand_com_range(ManagerTermBase):
    """벌점: 정지 명령일 때 몸 전체 무게중심이 양 발목 가운데보다 진행 방향으로 [min_x, max_x](m) 밖이면 벗어난 거리.

    팔 무작위판 2차는 팔을 아래·뒤로 내린 채 서 있으면 무게중심이 발목보다 최대 6cm 뒤로 가서 뒤로 넘어짐
    (가만히 서 있을 때도 무게중심이 발목 바로 위, 2026-11-20 eval_carry.py 진단). 발바닥은 발목보다 앞쪽이 길므로
    무게중심을 발목보다 조금 앞에 두게 한다.
    질량은 시작 이벤트(손 무게 무작위화)가 보상 관리자보다 늦게 실행되므로 첫 호출 때 읽어 둔다 (원본 코드로 확인).
    asset_cfg 의 body_names 는 [왼발, 오른발] 발 링크.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.mass = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        stand_threshold: float,
        min_x: float,
        max_x: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        if self.mass is None:
            self.mass = asset.root_physx_view.get_masses().to(env.device)
        com = (asset.data.body_com_pos_w * self.mass[..., None]).sum(dim=1) / self.mass.sum(dim=1, keepdim=True)
        mid = asset.data.body_pos_w[:, asset_cfg.body_ids, :2].mean(dim=1)
        yaw = asset.data.heading_w
        x = (com[:, 0] - mid[:, 0]) * torch.cos(yaw) + (com[:, 1] - mid[:, 1]) * torch.sin(yaw)
        excess = torch.clamp(min_x - x, min=0.0) + torch.clamp(x - max_x, min=0.0)
        return excess * is_standing_command(env, command_name, stand_threshold).float()


# ---------------------------------------------------------------------------
# 걷기 정책 v2 (loco_v2): 팔 관절 전체·손가락 무작위화, 걸을 때 뒤로 젖히기 억제, 서 있을 때 무릎 하한
# ---------------------------------------------------------------------------

# 무작위로 바꾸는 팔 관절 순서 (arm_ik.DualArmIK 와 같음): 왼팔 5개, 오른팔 5개
ARM_WIDE_JOINTS = [f"{side}_{name}_joint" for side in ("left", "right")
                   for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll")]
# 오른팔은 왼팔 범위를 좌우 대칭으로 뒤집어 씀 (roll·yaw 관절은 부호 반대)
ARM_MIRROR_SIGN = (1.0, -1.0, -1.0, 1.0, -1.0)
FINGER_JOINT_EXPR = [".*_zero_joint", ".*_one_joint", ".*_two_joint", ".*_three_joint", ".*_four_joint",
                     ".*_five_joint", ".*_six_joint"]


def arm_goal_wide_from_uniform(
    u: torch.Tensor,
    carry: torch.Tensor,
    ranges,
    random_prob: float | torch.Tensor,
    symmetric_prob: float,
) -> torch.Tensor:
    """0~1 균일 난수 u (n, 12) → 팔 목표 자세 (n, 10), 순서는 ARM_WIDE_JOINTS.

    - u[:, 0] < random_prob 이면 무작위 자세, 아니면 짐 든 자세(carry, (n, 10))
    - u[:, 1] < symmetric_prob 이면 양팔 좌우 대칭 자세, 아니면 좌우 따로
    - u[:, 2:7] = 왼팔 5개 관절, u[:, 7:12] = 오른팔
    - ranges: 왼팔 기준 (최소, 최대) 5쌍. 오른팔은 ARM_MIRROR_SIGN 으로 뒤집음
    학습 이벤트와 평가 스크립트(eval_carry.py --arm_motion wide)가 같은 규칙을 쓰도록 함수로 둠.
    """
    lo = torch.tensor([r[0] for r in ranges], device=u.device)
    hi = torch.tensor([r[1] for r in ranges], device=u.device)
    sign = torch.tensor(ARM_MIRROR_SIGN, device=u.device)
    left = lo + (hi - lo) * u[:, 2:7]
    sym = (u[:, 1] < symmetric_prob)[:, None]
    right = torch.where(sym, left, lo + (hi - lo) * u[:, 7:12]) * sign
    rand_goal = torch.cat([left, right], dim=1)
    return torch.where((u[:, 0] < random_prob)[:, None], rand_goal, carry)


class randomize_upper_body_targets(ManagerTermBase):
    """interval 이벤트 (매 스텝 호출): 팔 관절 10개와 손가락 14개의 모터 목표값을 무작위 자세로 옮긴다.

    randomize_arm_targets(어깨·팔꿈치 pitch 만)를 넓힌 것. 이유 (2026-11-22, 5단계):
    IK 가 어깨 roll·yaw 를 움직이자 그 관절이 바뀌는 것을 본 적 없는 정책이 넘어져, 정책에 넣는 관측을 걸러야 했음.
    손가락도 같은 이유로 움직일 수 없었음 (엄지를 펴서 상자에 손바닥을 붙이지 못함).
    - 새 목표는 환경마다 change_interval_s 간격. random_prob 확률로 무작위 자세(팔 + 손가락), 아니면 짐 든 자세
    - 손가락은 무작위 자세일 때 finger_prob 확률로 관절 한계 안의 무작위 값, 아니면 기본 자세
    - 목표값은 max_speed(rad/s) 로 옮김. 리셋되면 짐 든 자세에서 다시 시작
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.arm_ids, _ = self.asset.find_joints(ARM_WIDE_JOINTS, preserve_order=True)
        self.finger_ids, _ = self.asset.find_joints(FINGER_JOINT_EXPR)
        self.joint_ids = list(self.arm_ids) + list(self.finger_ids)
        self.n_arm = len(self.arm_ids)
        self.carry = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        limits = self.asset.data.soft_joint_pos_limits[:, self.finger_ids]
        self.finger_lo, self.finger_hi = limits[..., 0].clone(), limits[..., 1].clone()
        self.goal = self.carry.clone()
        self.current = self.carry.clone()
        self.time_left = self._sample_interval(env.num_envs, cfg.params["change_interval_s"])

    def _sample_interval(self, n: int, interval) -> torch.Tensor:
        lo, hi = interval
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
            n = self.num_envs
        else:
            n = len(env_ids)
        self.goal[env_ids] = self.carry[env_ids]
        self.current[env_ids] = self.carry[env_ids]
        self.time_left[env_ids] = self._sample_interval(n, self.cfg.params["change_interval_s"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        change_interval_s,
        random_prob: float,
        symmetric_prob: float,
        arm_ranges,
        finger_prob: float,
        max_speed: float,
        random_prob_standing: float | None = None,
        command_name: str = "base_velocity",
        stand_threshold: float = 0.1,
    ):
        self.time_left -= env.step_dt
        due = (self.time_left <= 0.0).nonzero().flatten()
        if len(due) > 0:
            n = len(due)
            u = torch.rand(n, 12, device=self.device)
            prob = random_prob
            if random_prob_standing is not None:
                standing = is_standing_command(env, command_name, stand_threshold)[due]
                prob = torch.where(standing, random_prob_standing, random_prob)
            carry = self.carry[due]
            arm_goal = arm_goal_wide_from_uniform(u, carry[:, : self.n_arm], arm_ranges, prob, symmetric_prob)
            v = torch.rand(n, len(self.finger_ids), device=self.device)
            finger_rand = self.finger_lo[due] + (self.finger_hi[due] - self.finger_lo[due]) * v
            move_fingers = ((u[:, 0] < prob) & (torch.rand(n, device=self.device) < finger_prob))[:, None]
            finger_goal = torch.where(move_fingers, finger_rand, carry[:, self.n_arm:])
            self.goal[due] = torch.cat([arm_goal, finger_goal], dim=1)
            self.time_left[due] = self._sample_interval(n, change_interval_s)
        self.current = step_toward(self.current, self.goal, max_speed * env.step_dt)
        self.asset.set_joint_position_target(self.current, joint_ids=self.joint_ids)


def walk_pitch_range_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    stand_threshold: float,
    back_limit: float,
    forward_limit: float,
    forward_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """벌점: 걷는 동안(정지 상태가 아닐 때) 골반 앞뒤 기울기가 허용 범위를 벗어난 만큼 (rad).

    기울기 = 골반 좌표에서 본 중력 방향으로 계산 (+ 앞으로 숙임). 뒤로는 back_limit, 앞으로는 forward_limit 까지 벌점 0.
    뒤로 넘은 만큼은 그대로, 앞으로 넘은 만큼은 forward_scale 배로 약하게 (앞으로 조금 숙이는 걸음은 허용).
    이유 (2026-11-20 팔 무작위판 처음부터 학습): 천천히 걸을 때 -10.8°, 회전할 때 -13.2° 로 뒤로 젖힘.
    서 있을 때의 기울기는 stand_flat_orientation_l2 가 맡는다.
    """
    asset = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b
    pitch = torch.atan2(grav[:, 0], -grav[:, 2])
    excess = torch.clamp(-pitch - back_limit, min=0.0) + forward_scale * torch.clamp(pitch - forward_limit, min=0.0)
    return excess * (~is_standing_command(env, command_name, stand_threshold)).float()


def stand_knee_min(
    env: ManagerBasedRLEnv, command_name: str, stand_threshold: float, asset_cfg: SceneEntityCfg, min_angle: float
) -> torch.Tensor:
    """벌점: 정지 상태에서 무릎이 min_angle(rad) 보다 펴진 만큼 (양 무릎 합).

    stance_knee_range(-1.0)의 하한 0.05 rad 로는 약해서, 팔 무작위판 처음부터 학습은 서 있을 때 무릎을 끝까지 폄 (-1.6°).
    """
    asset = env.scene[asset_cfg.name]
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]
    short = torch.sum(torch.clamp(min_angle - knee, min=0.0), dim=1)
    return short * is_standing_command(env, command_name, stand_threshold).float()
