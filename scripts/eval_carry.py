"""짐 든 채 걷기 정책 평가 (영상 없이 숫자만).

학습된 정책을 명령 속도 시나리오 × 손 무게 구간으로 나눈 환경에서 돌리고 기록한다.
- 넘어짐: 환경별 base_contact 종료 횟수
- 팔 흔들림: 매 스텝 어깨·팔꿈치 관절의 (실제값 - 기본값) → 평균(꾸준한 처짐), 표준편차·최대(흔들림 폭)
- 손 위치: 골반 기준 손 위치(cm)의 평균 이동, 표준편차, 최대-최소
- 속도 따라가기: 명령과 실제 속도의 차이
- 걸음 지표 (발 접촉 = 접촉 센서의 현재 접촉 시간 > 0, 발 = ankle_roll_link)
  - 딛는 발 무릎 각도: 그 발이 닿아 있는 동안의 무릎 관절 각도 평균 (웅크림 정도)
  - 골반 높이: 골반(root) 높이 평균 (평지라 지면 높이 0)
  - 걸음/초: 발이 닿는 순간(공중 → 접촉) 횟수 ÷ 시간, 양발 합계
  - 발 들어 올리는 높이: 한 번 옮기는 동안 발(링크 원점)의 최고 높이 − 닿는 순간 높이
  - 한발 지지 / 양발 지지 / 양발 공중: 시간 비율
  - 번갈아 딛기: 발이 닿을 때 직전에 닿은 발과 반대쪽인 비율 (L-R-L-R 이면 100%)
  - 양발 좌우 간격: 골반 기준 좌우(y) 거리

설정
- 랜덤 밀기·외력 끔, 관측 노이즈 끔 (Play 설정)
- 명령은 시나리오 값으로 고정: 방향 명령·정지 환경을 끄고, 매 스텝 덮어쓴 뒤 관측을 다시 계산
  (넘어져 리셋된 환경은 명령이 다시 뽑히므로 매 스텝 덮어씀)
- 손 무게: 시나리오 × 구간마다 같은 수의 환경. 구간 안에서 무작위, 양손 같은 무게 (상자를 드는 상황)
  학습 때의 무게 무작위화(add_hand_mass)는 끄고 같은 방식(질량 + 관성 비율)으로 직접 넣음
- 흔들림·손 위치 통계는 에피소드 시작 후 --warmup 스텝(처음 처지는 구간)을 뺀다. 넘어짐은 전체 구간
- --arm_motion: 어깨·팔꿈치 pitch 목표값을 미리 정한 순서로 움직임 (팔 무작위판 1차 학습 이벤트와 같은 규칙·범위·속도로 고정,
  난수는 --seed 로 고정 → 어느 정책이든 같은 팔 움직임). 학습 과제에 팔 무작위 이벤트가 있어도 평가에서는 끄고 이 순서를 씀
  이때 팔 처짐·흔들림·손 위치 값은 일부러 움직인 몫이 섞이므로 의미 없음 (넘어짐·속도·걸음·자세 지표를 봄)
  --arm_motion wide: 걷기 정책 v2 의 넓힌 범위 (팔 관절 10개 + 손가락, carry_mdp.randomize_upper_body_targets 와 같은 규칙)
- --scenarios back: 명령 시나리오를 뒤로 걷기 두 가지(-0.3, -0.5m/s)로 바꿈 (기본 네 가지와 따로 실행해 기존 결과와 환경 배정이 같게 유지)

예) python -u scripts\\eval_carry.py --headless --load_run 2026-11-19_17-33-52
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate G1 carry policy (no video).")
parser.add_argument("--task", type=str, default="Humanoid-G1-Carry-Flat-Play-v0")
parser.add_argument("--load_run", type=str, default=".*", help="logs/rsl_rl/g1_carry_flat 아래 폴더 이름 (기본: 가장 최근)")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt", help="체크포인트 파일 패턴 (기본: 마지막)")
parser.add_argument("--per_cell", type=int, default=50, help="시나리오 × 무게 구간 한 칸당 환경 수")
parser.add_argument("--steps", type=int, default=1000, help="진행 스텝 수 (50스텝 = 1초, 에피소드 최대 1000)")
parser.add_argument("--warmup", type=int, default=50, help="에피소드 시작 후 흔들림 통계에서 뺄 스텝 수")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--arm_motion", nargs="?", const="narrow", default=None, choices=["narrow", "wide"],
                    help="팔 목표값을 무작위 순서로 움직임. narrow(값 없이 쓰면 이것): 어깨·팔꿈치 pitch, wide: 팔 관절 10개 + 손가락")
parser.add_argument("--scenarios", type=str, default="default", choices=["default", "back"],
                    help="back: 뒤로 걷기 두 가지로 바꿈")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

# 명령 시나리오 (vx m/s, vy m/s, wz rad/s). 학습 범위: vx 0~1, vy -0.5~0.5, wz -1~1
SCENARIOS = [
    ("제자리", (0.0, 0.0, 0.0)),
    ("천천히 0.3m/s", (0.3, 0.0, 0.0)),
    ("빠르게 1.0m/s", (1.0, 0.0, 0.0)),
    ("회전 1.0rad/s", (0.0, 0.0, 1.0)),
]
if args_cli.scenarios == "back":
    SCENARIOS = [("뒤로 0.3m/s", (-0.3, 0.0, 0.0)), ("뒤로 0.5m/s", (-0.5, 0.0, 0.0))]
MASS_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]  # 손 하나에 더하는 무게 (kg)
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]
SIDES = ("left", "right")
RAD2DEG = 57.29578

n_scen, n_bins, per_cell = len(SCENARIOS), len(MASS_BINS), args_cli.per_cell
num_envs = n_scen * n_bins * per_cell

# ---- 환경 설정 ----
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
env_cfg.seed = args_cli.seed
env_cfg.events.add_hand_mass = None  # 손 무게는 아래에서 구간별로 직접 넣음
cmd_cfg = env_cfg.commands.base_velocity
cmd_cfg.heading_command = False
cmd_cfg.rel_standing_envs = 0.0
cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)
if hasattr(cmd_cfg, "rel_turn_envs"):  # 걸음 보상 추가판의 제자리 회전 명령 섞기도 끔 (명령은 아래에서 고정)
    cmd_cfg.rel_turn_envs = 0.0
cmd_cfg.debug_vis = False
if hasattr(env_cfg.events, "arm_motion"):  # 팔 무작위 움직임판의 학습 이벤트는 끄고, 필요하면 아래 고정 순서를 씀
    env_cfg.events.arm_motion = None

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
unwrapped = env.unwrapped
robot = unwrapped.scene["robot"]
device = unwrapped.device

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
print(f"[eval] 체크포인트: {resume_path}")

# ---- 환경 배정: 환경 i → 시나리오 i // (구간 수 × 칸당), 무게 구간 (i // 칸당) % 구간 수 ----
env_idx = torch.arange(num_envs)
scen_of_env = env_idx // (n_bins * per_cell)
bin_of_env = (env_idx // per_cell) % n_bins

g = torch.Generator().manual_seed(args_cli.seed)
bin_lo = torch.tensor([lo for lo, _ in MASS_BINS])[bin_of_env]
bin_w = torch.tensor([hi - lo for lo, hi in MASS_BINS])[bin_of_env]
added_mass = bin_lo + torch.rand(num_envs, generator=g) * bin_w  # 양손 같은 무게

# 손 무게 넣기 (원본 randomize_rigid_body_mass 와 같은 방식: 기본 질량 + 추가, 관성은 질량 비율만큼)
palm_ids = [robot.body_names.index(f"{s}_palm_link") for s in SIDES]
view = robot.root_physx_view
cpu_ids = torch.arange(num_envs)
default_mass = robot.data.default_mass.cpu()
default_inertia = robot.data.default_inertia.cpu()
masses = view.get_masses().clone()
for b in palm_ids:
    masses[:, b] = default_mass[:, b] + added_mass
view.set_masses(masses, cpu_ids)
ratios = masses[:, palm_ids] / default_mass[:, palm_ids]
inertias = view.get_inertias().clone()
inertias[:, palm_ids] = default_inertia[:, palm_ids] * ratios[..., None]
view.set_inertias(inertias, cpu_ids)
check = view.get_masses()[:, palm_ids]
print(f"[eval] 손 질량 확인: 최소 {check.min().item():.3f} kg, 최대 {check.max().item():.3f} kg")

# 명령 고정
cmd_term = unwrapped.command_manager.get_term("base_velocity")
target_cmd = torch.tensor([c for _, c in SCENARIOS], device=device)[scen_of_env.to(device)]

# 팔 움직임 순서 (--arm_motion): 학습 이벤트(carry_mdp.randomize_arm_targets)와 같은 규칙을 CPU 난수로 미리 계산
from humanoid_tasks.g1_carry import carry_mdp  # noqa: E402

arm_target_ids = [robot.joint_names.index(n) for n in carry_mdp.ARM_TARGET_JOINTS]
arm_sched = None  # (스텝, 환경, 관절) 목표값
arm_sched_pitch = None  # 넘어짐 진단용: [왼 어깨 pitch, 오른 어깨 pitch, 왼 팔꿈치 pitch, 오른 팔꿈치 pitch]
if args_cli.arm_motion == "narrow":
    # 평가 조건은 고정 (팔 무작위판 1차 학습 이벤트와 같은 값). 학습 설정을 바꿔도 이전 결과와 비교할 수 있게 숫자로 둠
    ap = {"change_interval_s": (2.0, 5.0), "random_prob": 0.5, "symmetric_prob": 0.5,
          "shoulder_range": (-1.2, 0.3), "elbow_range": (0.3, 1.5), "max_speed": 2.0}
    ga = torch.Generator().manual_seed(args_cli.seed + 1)
    carry = robot.data.default_joint_pos[:, arm_target_ids].cpu()
    cur, goal = carry.clone(), carry.clone()
    lo_i, hi_i = ap["change_interval_s"]
    t_left = lo_i + (hi_i - lo_i) * torch.rand(num_envs, generator=ga)
    sched = torch.zeros(args_cli.steps, num_envs, len(arm_target_ids))
    for t in range(args_cli.steps):
        t_left -= unwrapped.step_dt
        due = (t_left <= 0.0).nonzero().flatten()
        if len(due) > 0:
            u = torch.rand(len(due), 6, generator=ga)
            goal[due] = carry_mdp.arm_goal_from_uniform(
                u, carry[due], ap["shoulder_range"], ap["elbow_range"], ap["random_prob"], ap["symmetric_prob"]
            )
            t_left[due] = lo_i + (hi_i - lo_i) * torch.rand(len(due), generator=ga)
        cur = carry_mdp.step_toward(cur, goal, ap["max_speed"] * unwrapped.step_dt)
        sched[t] = cur
    arm_sched = sched.to(device)
    arm_sched_pitch = arm_sched
    moved = ((sched - carry[None]).abs() > 0.05).any(dim=-1).float().mean().item()
    print(f"[eval] 팔 움직임 순서: 간격 {ap['change_interval_s']}초, 무작위 비율 {ap['random_prob']}, 어깨 {ap['shoulder_range']}, "
          f"팔꿈치 {ap['elbow_range']}, 속도 {ap['max_speed']} rad/s → 짐 든 자세에서 벗어난 시간 비율 {moved * 100:.1f}%")
elif args_cli.arm_motion == "wide":
    # 걷기 정책 v2 의 넓힌 범위 (학습 이벤트 randomize_upper_body_targets 의 값을 숫자로 고정)
    ap = {"change_interval_s": (2.0, 5.0), "random_prob": 0.5, "symmetric_prob": 0.5, "finger_prob": 0.5, "max_speed": 2.0,
          "arm_ranges": ((-1.6, 0.4), (0.0, 0.9), (-0.3, 0.3), (0.0, 1.8), (-0.2, 0.2))}
    arm_ids, _ = robot.find_joints(carry_mdp.ARM_WIDE_JOINTS, preserve_order=True)
    finger_ids, _ = robot.find_joints(carry_mdp.FINGER_JOINT_EXPR)
    arm_target_ids = list(arm_ids) + list(finger_ids)
    n_arm = len(arm_ids)
    ga = torch.Generator().manual_seed(args_cli.seed + 1)
    carry = robot.data.default_joint_pos[:, arm_target_ids].cpu()
    f_lim = robot.data.soft_joint_pos_limits[:, finger_ids].cpu()
    cur, goal = carry.clone(), carry.clone()
    lo_i, hi_i = ap["change_interval_s"]
    t_left = lo_i + (hi_i - lo_i) * torch.rand(num_envs, generator=ga)
    sched = torch.zeros(args_cli.steps, num_envs, len(arm_target_ids))
    for t in range(args_cli.steps):
        t_left -= unwrapped.step_dt
        due = (t_left <= 0.0).nonzero().flatten()
        if len(due) > 0:
            u = torch.rand(len(due), 12, generator=ga)
            arm_goal = carry_mdp.arm_goal_wide_from_uniform(u, carry[due][:, :n_arm], ap["arm_ranges"], ap["random_prob"],
                                                            ap["symmetric_prob"])
            v = torch.rand(len(due), len(finger_ids), generator=ga)
            f_rand = f_lim[due][..., 0] + (f_lim[due][..., 1] - f_lim[due][..., 0]) * v
            move_f = ((u[:, 0] < ap["random_prob"]) & (torch.rand(len(due), generator=ga) < ap["finger_prob"]))[:, None]
            goal[due] = torch.cat([arm_goal, torch.where(move_f, f_rand, carry[due][:, n_arm:])], dim=1)
            t_left[due] = lo_i + (hi_i - lo_i) * torch.rand(len(due), generator=ga)
        cur = carry_mdp.step_toward(cur, goal, ap["max_speed"] * unwrapped.step_dt)
        sched[t] = cur
    arm_sched = sched.to(device)
    arm_sched_pitch = arm_sched[:, :, [0, 5, 3, 8]]
    moved = ((sched[..., :n_arm] - carry[None, :, :n_arm]).abs() > 0.05).any(dim=-1).float().mean().item()
    print(f"[eval] 팔 움직임 순서 (넓힌 범위): 간격 {ap['change_interval_s']}초, 무작위 비율 {ap['random_prob']}, "
          f"팔 범위(왼팔 기준 어깨 pitch·roll·yaw, 팔꿈치 pitch·roll) {ap['arm_ranges']}, 손가락 무작위 {ap['finger_prob']}, "
          f"속도 {ap['max_speed']} rad/s → 짐 든 자세에서 벗어난 시간 비율 {moved * 100:.1f}%")

# 기록할 관절·링크
joint_ids = [robot.joint_names.index(f"{s}_{j}_joint") for s in SIDES for j in ARM_JOINTS]  # (좌 5, 우 5)
default_q = robot.data.default_joint_pos[:, joint_ids]

T, N, J = args_cli.steps, num_envs, len(joint_ids)
dev_hist = torch.zeros(T, N, J, device=device)
hand_hist = torch.zeros(T, N, 2, 3, device=device)
mask_hist = torch.zeros(T, N, dtype=torch.bool, device=device)
falls = torch.zeros(N, dtype=torch.long, device=device)
lin_err_sum = torch.zeros(N, device=device)
yaw_err_sum = torch.zeros(N, device=device)
vx_sum = torch.zeros(N, device=device)
wz_sum = torch.zeros(N, device=device)

# 걸음 지표용
contact_sensor = unwrapped.scene.sensors["contact_forces"]
foot_sensor_ids = [contact_sensor.body_names.index(f"{s}_ankle_roll_link") for s in SIDES]
foot_body_ids = [robot.body_names.index(f"{s}_ankle_roll_link") for s in SIDES]
knee_ids = [robot.joint_names.index(f"{s}_knee_joint") for s in SIDES]
dt = unwrapped.step_dt
g_stance_knee = torch.zeros(N, 2, device=device)
g_stance_cnt = torch.zeros(N, 2, device=device)
g_stance_footz = torch.zeros(N, 2, device=device)
g_pelvis_z = torch.zeros(N, device=device)
g_touchdowns = torch.zeros(N, device=device)
g_swing_peak = torch.zeros(N, device=device)
g_single = torch.zeros(N, device=device)
g_double = torch.zeros(N, device=device)
g_flight = torch.zeros(N, device=device)
g_alt_hits = torch.zeros(N, device=device)
g_alt_total = torch.zeros(N, device=device)
g_feet_lat = torch.zeros(N, device=device)
contact_hist = torch.zeros(T, N, 2, dtype=torch.bool, device=device)  # 접촉 패턴·구간 길이 진단용
footz_hist = torch.zeros(T, N, 2, device=device)  # 떨림을 걸러낸 걸음 지표용 발 높이
lift_hist = torch.full((T, N, 2), float("nan"), device=device)  # 발이 닿는 순간의 들어 올린 높이
prev_contact = torch.ones(N, 2, dtype=torch.bool, device=device)
swing_max = torch.zeros(N, 2, device=device)
last_td_foot = torch.full((N,), -1, dtype=torch.long, device=device)

# 자세·보폭·넘어진 시점·항목별 보상 진단용
reward_manager = unwrapped.reward_manager
reward_names = list(reward_manager.active_terms)
rew_sum = torch.zeros(N, len(reward_names), device=device)  # 항목별 보상(가중치 곱한 값)의 합
torso_joint_id = robot.joint_names.index("torso_joint")
pitch_sum = torch.zeros(N, device=device)  # 골반 앞뒤 기울기 (+ = 앞으로 숙임)
tilt_sum = torch.zeros(N, device=device)  # 골반 전체 기울기 (연직에서 벗어난 각도, 방향 무관)
vx_sq_sum = torch.zeros(N, device=device)
wz_sq_sum = torch.zeros(N, device=device)
pz_sq_sum = torch.zeros(N, device=device)
waist_sum = torch.zeros(N, device=device)
waist_sq_sum = torch.zeros(N, device=device)
knee_hist = torch.zeros(T, N, 2, device=device)
sepx_hist = torch.zeros(T, N, device=device)  # 골반 기준 앞뒤 발 간격 (왼발 − 오른발)
footxy_hist = torch.zeros(T, N, 2, 2, device=device)  # 발 수평 위치 (닿은 채 미끄러진 거리 계산용)
heading_hist = torch.zeros(T, N, 2, device=device)  # 골반이 향하는 방향 단위 벡터
fall_step = torch.full((N,), -1, dtype=torch.long, device=device)  # 처음 넘어진 시점 (에피소드 스텝)
fall_t = torch.full((N,), -1, dtype=torch.long, device=device)  # 처음 넘어진 시점 (평가 시작 후 스텝)
# 몸 전체 무게중심: 링크별 질량(손 무게 포함)으로 가중 평균. 발목(ankle_roll_link) 가운데 기준 진행 방향 앞뒤 거리
body_mass = view.get_masses().to(device)  # (N, 링크 수), 시작 후 바뀌지 않음
com_hist = torch.zeros(T, N, device=device)
fall_pitch = torch.zeros(N, device=device)  # 넘어지기 한 스텝 전 기울기
fall_roll = torch.zeros(N, device=device)
prev_len = unwrapped.episode_length_buf.clone()
prev_pitch = torch.zeros(N, device=device)
prev_roll = torch.zeros(N, device=device)


def hand_pos_in_pelvis() -> torch.Tensor:
    """손(palm) 링크 위치, 골반(root) 기준 좌표 (N, 2, 3), 단위 m."""
    rel = robot.data.body_pos_w[:, palm_ids] - robot.data.root_pos_w[:, None, :]
    quat = robot.data.root_quat_w[:, None, :].expand(-1, 2, -1)
    return quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(N, 2, 3)


# 기준 손 위치: 리셋 직후 (모든 관절이 기본 자세)
hand_ref = hand_pos_in_pelvis().mean(dim=0)  # (2, 3)
print(f"[eval] 기준 손 위치(cm, 골반 기준 x 앞·y 왼쪽·z 위): 왼손 {(hand_ref[0] * 100).tolist()}, 오른손 {(hand_ref[1] * 100).tolist()}")

cmd_term.vel_command_b[:] = target_cmd
obs = env.get_observations()

with torch.inference_mode():
    for t in range(T):
        actions = policy(obs)
        if arm_sched is not None:
            robot.set_joint_position_target(arm_sched[t], joint_ids=arm_target_ids)
        obs, _, dones, _ = env.step(actions)
        terminated = unwrapped.termination_manager.terminated
        falls += terminated.long()
        new_fall = terminated & (fall_step < 0)
        fall_step = torch.where(new_fall, prev_len + 1, fall_step)
        fall_t = torch.where(new_fall, torch.full_like(fall_t, t), fall_t)
        fall_pitch = torch.where(new_fall, prev_pitch, fall_pitch)
        fall_roll = torch.where(new_fall, prev_roll, fall_roll)
        # 명령을 시나리오 값으로 되돌리고 관측을 다시 계산 (리셋된 환경은 명령이 다시 뽑혔으므로)
        cmd_term.vel_command_b[:] = target_cmd
        obs = env.get_observations()

        valid = unwrapped.episode_length_buf >= args_cli.warmup
        mask_hist[t] = valid
        dev_hist[t] = robot.data.joint_pos[:, joint_ids] - default_q
        hand_hist[t] = hand_pos_in_pelvis()
        v = valid.float()
        lin_err_sum += v * torch.norm(robot.data.root_lin_vel_b[:, :2] - target_cmd[:, :2], dim=1)
        yaw_err_sum += v * (robot.data.root_ang_vel_b[:, 2] - target_cmd[:, 2]).abs()
        vx_sum += v * robot.data.root_lin_vel_b[:, 0]
        wz_sum += v * robot.data.root_ang_vel_b[:, 2]

        # ---- 걸음 지표 ----
        contact = contact_sensor.data.current_contact_time[:, foot_sensor_ids] > 0.0  # (N, 2)
        cf = contact.float()
        foot_z = robot.data.body_pos_w[:, foot_body_ids, 2]  # (N, 2)
        g_stance_knee += robot.data.joint_pos[:, knee_ids] * cf * v[:, None]
        g_stance_cnt += cf * v[:, None]
        g_stance_footz += foot_z * cf * v[:, None]
        g_pelvis_z += robot.data.root_pos_w[:, 2] * v
        n_contact = contact.sum(dim=1)
        g_single += (n_contact == 1).float() * v
        g_double += (n_contact == 2).float() * v
        g_flight += (n_contact == 0).float() * v
        # 발이 닿는 순간: 들어 올린 동안의 최고 높이 − 닿는 순간 높이
        td = contact & ~prev_contact & valid[:, None]
        contact_hist[t] = contact
        footz_hist[t] = foot_z
        lift_hist[t] =torch.where(td, swing_max - foot_z, torch.full_like(foot_z, float("nan")))
        g_swing_peak += ((swing_max - foot_z) * td.float()).sum(dim=1)
        g_touchdowns += td.float().sum(dim=1)
        for f in range(2):
            ev = td[:, f]
            has_last = last_td_foot >= 0
            g_alt_total += (ev & has_last).float()
            g_alt_hits += (ev & has_last & (last_td_foot != f)).float()
            last_td_foot = torch.where(ev, torch.full_like(last_td_foot, f), last_td_foot)
        swing_max = torch.where(contact, foot_z, torch.maximum(swing_max, foot_z))
        prev_contact = contact
        # 양발 좌우 간격 (골반 기준 y)
        rel = (robot.data.body_pos_w[:, foot_body_ids[0]] - robot.data.body_pos_w[:, foot_body_ids[1]])
        rel_b = quat_apply_inverse(robot.data.root_quat_w, rel)
        g_feet_lat += rel_b[:, 1].abs() * v

        # ---- 자세·보폭·항목별 보상 진단 ----
        grav = robot.data.projected_gravity_b
        pitch = torch.atan2(grav[:, 0], -grav[:, 2])  # + = 앞으로 숙임
        roll = torch.atan2(-grav[:, 1], -grav[:, 2])  # + = 오른쪽으로 기욺
        pitch_sum += pitch * v
        tilt_sum += torch.acos(torch.clamp(-grav[:, 2], -1.0, 1.0)) * v
        vx_sq_sum += v * robot.data.root_lin_vel_b[:, 0] ** 2
        wz_sq_sum += v * robot.data.root_ang_vel_b[:, 2] ** 2
        pz_sq_sum += v * robot.data.root_pos_w[:, 2] ** 2
        waist = robot.data.joint_pos[:, torso_joint_id]
        waist_sum += waist * v
        waist_sq_sum += waist**2 * v
        knee_hist[t] = robot.data.joint_pos[:, knee_ids]
        sepx_hist[t] = rel_b[:, 0]
        footxy_hist[t] = robot.data.body_pos_w[:, foot_body_ids, :2]
        yaw = robot.data.heading_w
        heading_hist[t] = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        com = (robot.data.body_com_pos_w * body_mass[..., None]).sum(dim=1) / body_mass.sum(dim=1, keepdim=True)
        feet_mid = robot.data.body_pos_w[:, foot_body_ids, :2].mean(dim=1)
        com_hist[t] = ((com[:, :2] - feet_mid) * heading_hist[t]).sum(dim=1)
        rew_sum += reward_manager._step_reward * v[:, None]
        prev_len = unwrapped.episode_length_buf.clone()
        prev_pitch, prev_roll = pitch, roll

# ---- 통계 (환경·관절별, 유효 스텝만) ----
m = mask_hist.float()
cnt = m.sum(0).clamp(min=1)  # (N,)


def masked_stats(x: torch.Tensor):
    """x: (T, N, ...) → 평균, 표준편차, 최대 |x-평균|, 최대-최소 (유효 스텝만)."""
    mm = m.view(T, N, *([1] * (x.dim() - 2)))
    c = cnt.view(N, *([1] * (x.dim() - 2)))
    mean = (x * mm).sum(0) / c
    std = (((x - mean) ** 2 * mm).sum(0) / c).sqrt()
    peak = torch.where(mm.bool(), (x - mean).abs(), torch.zeros_like(x)).amax(0)
    big = torch.finfo(x.dtype).max
    p2p = torch.where(mm.bool(), x, torch.full_like(x, -big)).amax(0) - torch.where(mm.bool(), x, torch.full_like(x, big)).amin(0)
    return mean, std, peak, p2p


j_mean, j_std, j_peak, _ = masked_stats(dev_hist)  # (N, J) rad
h_mean, h_std, _, h_p2p = masked_stats(hand_hist)  # (N, 2, 3) m
h_shift = h_mean - hand_ref  # 기준 대비 평균 이동

lin_err = (lin_err_sum / cnt).cpu()
yaw_err = (yaw_err_sum / cnt).cpu()
vx_mean = (vx_sum / cnt).cpu()
wz_mean = (wz_sum / cnt).cpu()
falls = falls.cpu()
valid_frac = (m.sum(0) / T).cpu()

# 걸음 지표 (환경별)
valid_steps = m.sum(0)  # (N,)
gait = {
    "knee": ((g_stance_knee.sum(1) / g_stance_cnt.sum(1).clamp(min=1)) * RAD2DEG).cpu(),  # 딛는 발 무릎 (deg)
    "pelvis": (g_pelvis_z / cnt * 100).cpu(),  # cm
    "cadence": (g_touchdowns / (valid_steps * dt).clamp(min=1e-6)).cpu(),  # 걸음/초
    "lift": (g_swing_peak / g_touchdowns.clamp(min=1) * 100).cpu(),  # cm
    "single": (g_single / cnt * 100).cpu(),
    "double": (g_double / cnt * 100).cpu(),
    "flight": (g_flight / cnt * 100).cpu(),
    "alt": (g_alt_hits / g_alt_total.clamp(min=1) * 100).cpu(),
    "feet_lat": (g_feet_lat / cnt * 100).cpu(),  # cm
    "stance_footz": ((g_stance_footz.sum(1) / g_stance_cnt.sum(1).clamp(min=1)) * 100).cpu(),  # cm
}


def jcol(side: int, name: str) -> int:
    return side * len(ARM_JOINTS) + ARM_JOINTS.index(name)


# 팔 하나(환경 × 좌우)를 한 표본으로: (N, 2) 형태로 정리
def per_arm(stat: torch.Tensor, name: str) -> torch.Tensor:
    return torch.stack([stat[:, jcol(0, name)], stat[:, jcol(1, name)]], dim=1).cpu()


arm = {
    "sp_mean": per_arm(j_mean, "shoulder_pitch") * RAD2DEG,
    "sp_std": per_arm(j_std, "shoulder_pitch") * RAD2DEG,
    "sp_peak": per_arm(j_peak, "shoulder_pitch") * RAD2DEG,
    "sr_std": per_arm(j_std, "shoulder_roll") * RAD2DEG,
    "ep_mean": per_arm(j_mean, "elbow_pitch") * RAD2DEG,
    "ep_std": per_arm(j_std, "elbow_pitch") * RAD2DEG,
    "ep_peak": per_arm(j_peak, "elbow_pitch") * RAD2DEG,
    "hx_shift": h_shift[:, :, 0].cpu() * 100,
    "hz_shift": h_shift[:, :, 2].cpu() * 100,
    "hx_std": h_std[:, :, 0].cpu() * 100,
    "hz_std": h_std[:, :, 2].cpu() * 100,
    "hx_p2p": h_p2p[:, :, 0].cpu() * 100,
    "hz_p2p": h_p2p[:, :, 2].cpu() * 100,
}


def summarize(sel: torch.Tensor) -> dict:
    """sel: (N,) bool 환경 선택 → 표에 쓸 값."""
    out = {"envs": int(sel.sum()), "fell_envs": int((falls[sel] > 0).sum()), "falls": int(falls[sel].sum())}
    for k, v in arm.items():
        vals = v[sel].flatten()
        out[k] = vals.mean().item()
        out[k + "_max"] = vals.max().item()
    out["lin_err"] = lin_err[sel].mean().item()
    out["yaw_err"] = yaw_err[sel].mean().item()
    out["vx"] = vx_mean[sel].mean().item()
    out["wz"] = wz_mean[sel].mean().item()
    return out


def print_table(title: str, rows: list[tuple[str, dict]]):
    print(f"\n========== {title} ==========")
    print("팔 각도는 도(deg), 손 위치는 cm. 평균 = 표본 평균, [ ] = 표본 중 최대")
    print(f"{'구분':16s} {'환경':>4s} {'넘어진환경':>9s} {'어깨pitch 처짐':>13s} {'흔들림std':>10s} {'최대':>12s} "
          f"{'팔꿈치 처짐':>11s} {'std':>10s} {'손z이동':>7s} {'손z std':>12s} {'손z폭':>12s} {'손x std':>12s}")
    for name, s in rows:
        print(
            f"{name:16s} {s['envs']:4d} {s['fell_envs']:4d} ({s['fell_envs'] / max(s['envs'], 1) * 100:4.1f}%) "
            f"{s['sp_mean']:+13.2f} {s['sp_std']:5.2f}[{s['sp_std_max']:4.2f}] {s['sp_peak']:5.2f}[{s['sp_peak_max']:5.2f}] "
            f"{s['ep_mean']:+11.2f} {s['ep_std']:5.2f}[{s['ep_std_max']:4.2f}] {s['hz_shift']:+7.2f} "
            f"{s['hz_std']:5.2f}[{s['hz_std_max']:5.2f}] {s['hz_p2p']:5.2f}[{s['hz_p2p_max']:5.2f}] {s['hx_std']:5.2f}[{s['hx_std_max']:5.2f}]"
        )


def bin_name(b: int) -> str:
    lo, hi = MASS_BINS[b]
    return f"+{lo:.0f}~{hi:.0f}kg"


print(f"\n[eval] 환경 {num_envs}개 (시나리오 {n_scen} × 무게 구간 {n_bins} × {per_cell}), {T}스텝({T / 50:.0f}초), "
      f"흔들림 통계는 에피소드 시작 후 {args_cli.warmup}스텝 제외 (유효 스텝 비율 평균 {valid_frac.mean().item() * 100:.1f}%)")
print(f"[eval] 전체 넘어짐: {int(falls.sum())}회, 넘어진 환경 {int((falls > 0).sum())}/{num_envs}")

print_table("무게 구간별 (모든 시나리오)", [(bin_name(b), summarize(bin_of_env == b)) for b in range(n_bins)])
print_table("명령 시나리오별 (모든 무게)", [(SCENARIOS[s][0], summarize(scen_of_env == s)) for s in range(n_scen)])

print("\n========== 시나리오별 속도 따라가기 (유효 스텝 평균) ==========")
print(f"{'시나리오':16s} {'명령 vx':>8s} {'실제 vx':>8s} {'명령 wz':>8s} {'실제 wz':>8s} {'선속도 오차':>10s} {'회전 오차':>9s}")
for s, (name, c) in enumerate(SCENARIOS):
    r = summarize(scen_of_env == s)
    print(f"{name:16s} {c[0]:8.2f} {r['vx']:8.3f} {c[2]:8.2f} {r['wz']:8.3f} {r['lin_err']:10.3f} {r['yaw_err']:9.3f}")

def print_gait_table(title: str, rows: list[tuple[str, torch.Tensor]]):
    print(f"\n========== 걸음 지표: {title} ==========")
    print(f"{'구분':16s} {'딛는발무릎':>9s} {'골반높이':>8s} {'걸음/초':>7s} {'발높이':>7s} "
          f"{'한발지지':>8s} {'양발지지':>8s} {'양발공중':>8s} {'번갈아':>7s} {'발간격':>7s}")
    for name, sel in rows:
        g = {k: v[sel].mean().item() for k, v in gait.items()}
        print(f"{name:16s} {g['knee']:8.1f}° {g['pelvis']:6.1f}cm {g['cadence']:7.2f} {g['lift']:5.1f}cm "
              f"{g['single']:7.1f}% {g['double']:7.1f}% {g['flight']:7.1f}% {g['alt']:6.1f}% {g['feet_lat']:5.1f}cm")


print_gait_table("명령 시나리오별 (모든 무게)", [(SCENARIOS[s][0], scen_of_env == s) for s in range(n_scen)])
print_gait_table("무게 구간별 (모든 시나리오)", [(bin_name(b), bin_of_env == b) for b in range(n_bins)])
walk_sel = scen_of_env != 0
print_gait_table("걷는 시나리오(천천히·빠르게·회전) × 무게",
                 [(bin_name(b), walk_sel & (bin_of_env == b)) for b in range(n_bins)])
print(f"[eval] 딛고 있는 발(ankle_roll_link 원점) 높이 평균: {gait['stance_footz'].mean().item():.2f} cm")


def phase_durations(c: torch.Tensor, value: bool) -> list[int]:
    """1차원 접촉 기록에서 value(True=접촉, False=공중)가 이어진 구간 길이(스텝) 목록. 처음·끝 구간은 뺌."""
    out, run, started = [], 0, False
    for x in c.tolist():
        if x == value:
            run += 1
        else:
            if run > 0 and started:
                out.append(run)
            run, started = 0, True
    return out


# ---- 떨림을 걸러낸 걸음 지표 ----
# 0.02~0.04초짜리 접촉 떨림(딛는 발이 잠깐 떨어지거나, 옮기는 발이 잠깐 스침)을 걸음으로 세지 않도록
# 접촉 기록에서 2스텝 이하 공중 틈은 접촉으로 메우고(closing), 2스텝 이하 접촉은 공중으로 지운다(opening).
def _dilate(x: torch.Tensor, r: int) -> torch.Tensor:  # x: (B, T) float, 시간축 최대 필터
    return torch.nn.functional.max_pool1d(x[:, None], 2 * r + 1, stride=1, padding=r)[:, 0]


def _erode(x: torch.Tensor, r: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - x, r)


DEBOUNCE_R = 1  # 반지름 1 → 길이 2 이하 구간을 걸러냄
c_seq = contact_hist.permute(1, 2, 0).reshape(N * 2, T).float()
c_seq = _erode(_dilate(c_seq, DEBOUNCE_R), DEBOUNCE_R)  # closing: 짧은 공중 틈 메우기
c_seq = _dilate(_erode(c_seq, DEBOUNCE_R), DEBOUNCE_R)  # opening: 짧은 접촉 지우기
dc = c_seq.reshape(N, 2, T).permute(2, 0, 1) > 0.5  # (T, N, 2)
fz = footz_hist
vmask = mask_hist  # (T, N)
d_td = torch.zeros(N, device=device)
d_lift_sum = torch.zeros(N, device=device)
d_single = torch.zeros(N, device=device)
d_double = torch.zeros(N, device=device)
d_flight = torch.zeros(N, device=device)
d_alt_hits = torch.zeros(N, device=device)
d_alt_total = torch.zeros(N, device=device)
d_last = torch.full((N,), -1, dtype=torch.long, device=device)
d_swing_max = fz[0].clone()
d_step_sum = torch.zeros(N, device=device)  # 보폭: 발이 닿는 순간 그 발이 반대 발보다 앞에 있는 거리
d_slide = torch.zeros(N, device=device)  # 닿은 채 발이 수평으로 움직인 거리 합
d_slide_fwd = torch.zeros(N, device=device)  # 그중 진행 방향 성분 (+ = 앞으로 미끄러짐)
d_slide_early = torch.zeros(N, device=device)  # 닿은 뒤 0.1초(5스텝) 안에 움직인 거리
since_td = torch.zeros(N, 2, device=device)
for t in range(1, T):
    _w = (dc[t] & dc[t - 1]).float() * vmask[t].float()[:, None]  # 두 스텝 연속 닿아 있던 발
    _d = footxy_hist[t] - footxy_hist[t - 1]
    _dist = _d.norm(dim=-1)
    since_td = torch.where(dc[t], since_td + 1, torch.zeros_like(since_td))
    d_slide += (_dist * _w).sum(dim=1)
    d_slide_fwd += ((_d * heading_hist[t][:, None, :]).sum(dim=-1) * _w).sum(dim=1)
    d_slide_early += (_dist * _w * (since_td <= 5).float()).sum(dim=1)
    c, pc, v = dc[t], dc[t - 1], vmask[t]
    vf = v.float()
    nc = c.sum(dim=1)
    d_single += (nc == 1).float() * vf
    d_double += (nc == 2).float() * vf
    d_flight += (nc == 0).float() * vf
    td_d = c & ~pc & v[:, None]
    d_lift_sum += ((d_swing_max - fz[t]) * td_d.float()).sum(dim=1)
    d_td += td_d.float().sum(dim=1)
    d_step_sum += sepx_hist[t] * (td_d[:, 0].float() - td_d[:, 1].float())
    for f in range(2):
        ev = td_d[:, f]
        has_last = d_last >= 0
        d_alt_total += (ev & has_last).float()
        d_alt_hits += (ev & has_last & (d_last != f)).float()
        d_last = torch.where(ev, torch.full_like(d_last, f), d_last)
    d_swing_max = torch.where(c, fz[t], torch.maximum(d_swing_max, fz[t]))
gait_d = {
    "cadence": (d_td / (valid_steps * dt).clamp(min=1e-6)).cpu(),
    "lift": (d_lift_sum / d_td.clamp(min=1) * 100).cpu(),
    "single": (d_single / cnt * 100).cpu(),
    "double": (d_double / cnt * 100).cpu(),
    "flight": (d_flight / cnt * 100).cpu(),
    "alt": (d_alt_hits / d_alt_total.clamp(min=1) * 100).cpu(),
}
print("\n========== 떨림을 걸러낸 걸음 지표 (2스텝 이하 공중 틈·접촉 제거) ==========")
print(f"{'구분':16s} {'걸음/초':>7s} {'발높이':>7s} {'한발지지':>8s} {'양발지지':>8s} {'양발공중':>8s} {'번갈아':>7s}")
for s in range(n_scen):
    sel = scen_of_env == s
    g = {k: v[sel].mean().item() for k, v in gait_d.items()}
    print(f"{SCENARIOS[s][0]:16s} {g['cadence']:7.2f} {g['lift']:5.1f}cm {g['single']:7.1f}% {g['double']:7.1f}% "
          f"{g['flight']:7.1f}% {g['alt']:6.1f}%")
for s in range(n_scen):
    e = int((scen_of_env == s).nonzero().flatten()[0])
    c = dc[args_cli.warmup:args_cli.warmup + 100, e].cpu()
    chars = "".join("B" if l and r else "L" if l else "R" if r else "." for l, r in c.tolist())
    print(f"{SCENARIOS[s][0]:16s} {chars}")

# ---- 자세·보폭 진단 (사람 걸음과 비교할 값) ----
def _std(sq_sum: torch.Tensor, s_sum: torch.Tensor) -> torch.Tensor:
    return (sq_sum / cnt - (s_sum / cnt) ** 2).clamp(min=0).sqrt()


knee_valid = torch.where(mask_hist[:, :, None], knee_hist, torch.full_like(knee_hist, float("nan")))
posture = {
    "pitch": (pitch_sum / cnt * RAD2DEG).cpu(),  # 골반 앞뒤 기울기 평균 (deg, + 앞)
    "tilt": (tilt_sum / cnt * RAD2DEG).cpu(),  # 골반 전체 기울기 평균 (deg)
    "step_len": (d_step_sum / d_td.clamp(min=1) * 100).cpu(),  # 보폭 (cm)
    "swing_knee": (torch.nanquantile(knee_valid, 0.95, dim=0).mean(dim=1) * RAD2DEG).cpu(),  # 무릎 각도 상위 5% (옮길 때 굽힘)
    "waist_mean": (waist_sum / cnt * RAD2DEG).cpu(),
    "waist_std": (_std(waist_sq_sum, waist_sum) * RAD2DEG).cpu(),
    "vx_std": _std(vx_sq_sum, vx_sum).cpu(),
    "wz_std": _std(wz_sq_sum, wz_sum).cpu(),
    "pz_std": (_std(pz_sq_sum, g_pelvis_z) * 100).cpu(),  # 골반 높이 흔들림 (cm)
    # 발 미끄러짐: 한 걸음(발이 한 번 닿았다 떨어질 때까지) 동안 닿은 채 수평으로 움직인 거리 (cm)
    "slide": (d_slide / d_td.clamp(min=1) * 100).cpu(),
    "slide_fwd": (d_slide_fwd / d_td.clamp(min=1) * 100).cpu(),
    "slide_early": (d_slide_early / d_slide.clamp(min=1e-6) * 100).cpu(),  # 닿은 뒤 0.1초 안의 몫 (%)
}
print("\n========== 자세·보폭 진단 (떨림을 걸러낸 접촉 기준, 유효 스텝) ==========")
print("보폭 = 닿는 순간 앞뒤 발 간격, 미끄러짐 = 한 걸음 동안 닿은 채 움직인 거리 (앞 = 진행 방향 성분, 초반 = 닿은 뒤 0.1초 안의 몫)")
print(f"{'구분':16s} {'골반기울기':>9s} {'전체기울기':>9s} {'보폭':>7s} {'미끄러짐(앞/초반)':>20s} {'옮길때무릎':>9s} {'허리회전 평균/std':>16s} "
      f"{'vx std':>7s} {'wz std':>7s} {'골반높이std':>10s}")


def print_posture(name: str, sel: torch.Tensor):
    p = {k: v[sel].mean().item() for k, v in posture.items()}
    print(f"{name:16s} {p['pitch']:+8.1f}° {p['tilt']:8.1f}° {p['step_len']:5.1f}cm {p['slide']:5.1f}cm ({p['slide_fwd']:+5.1f} / {p['slide_early']:3.0f}%) "
          f"{p['swing_knee']:8.1f}° {p['waist_mean']:+7.1f}° / {p['waist_std']:4.1f}° {p['vx_std']:7.3f} {p['wz_std']:7.3f} {p['pz_std']:8.2f}cm")


for s in range(n_scen):
    print_posture(SCENARIOS[s][0], scen_of_env == s)
for s in range(n_scen):
    for b in (0, n_bins - 1):
        print_posture(f"{SCENARIOS[s][0][:6]} {bin_name(b)}", (scen_of_env == s) & (bin_of_env == b))

# ---- 넘어진 시점·방향 ----
fall_step_c, fall_pitch_c, fall_roll_c = fall_step.cpu(), fall_pitch.cpu() * RAD2DEG, fall_roll.cpu() * RAD2DEG
print("\n========== 처음 넘어진 시점 (에피소드 시작 후 초) · 넘어지기 직전 기울기 ==========")
for s in range(n_scen):
    sel = (scen_of_env == s) & (fall_step_c >= 0)
    if int(sel.sum()) == 0:
        continue
    ts = fall_step_c[sel].float() * dt
    edges = [0, 2, 5, 10, 15, 1e9]
    hist = [int(((ts >= lo) & (ts < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
    print(f"{SCENARIOS[s][0]:16s} {int(sel.sum()):3d}개 | 0~2초 {hist[0]}, 2~5초 {hist[1]}, 5~10초 {hist[2]}, 10~15초 {hist[3]}, 15초~ {hist[4]} "
          f"| 직전 앞뒤 기울기 평균 {fall_pitch_c[sel].mean():+.1f}° (앞으로 {int((fall_pitch_c[sel] > 0).sum())}, 뒤로 {int((fall_pitch_c[sel] < 0).sum())}), "
          f"좌우 |기울기| 평균 {fall_roll_c[sel].abs().mean():.1f}°")

# ---- 몸 전체 무게중심 (발목 가운데 기준 앞뒤, + 앞) ----
print("\n========== 몸 전체 무게중심: 양 발목(ankle_roll_link) 가운데 기준 진행 방향 앞뒤 거리 (cm, + 앞, 유효 스텝) ==========")
print(f"{'구분':16s} {'평균':>7s} {'std':>6s} {'하위 5%':>8s} {'상위 5%':>8s}")
com_valid = torch.where(mask_hist, com_hist, torch.full_like(com_hist, float("nan"))) * 100


def print_com(name: str, sel: torch.Tensor):
    x = com_valid[:, sel.to(device)].flatten()
    x = x[~torch.isnan(x)]
    if len(x) == 0:
        return
    print(f"{name:16s} {x.mean():+7.2f} {x.std():6.2f} {x.quantile(0.05):+8.2f} {x.quantile(0.95):+8.2f}")


for s in range(n_scen):
    print_com(SCENARIOS[s][0], scen_of_env == s)
for b in range(n_bins):
    print_com(f"제자리 {bin_name(b)}", (scen_of_env == 0) & (bin_of_env == b))

# 넘어지기 전 무게중심·팔 목표 (처음 넘어진 환경, 시나리오별)
fall_t_c = fall_t.cpu()
for s in range(n_scen):
    ids = ((scen_of_env == s) & (fall_t_c >= 0)).nonzero().flatten()
    if len(ids) == 0:
        continue
    ft = fall_t_c[ids]
    pre = torch.clamp(ft - 10, min=0)  # 0.2초 전
    com_pre = com_hist[pre.to(device), ids.to(device)].cpu() * 100
    line = f"{SCENARIOS[s][0]:16s} {len(ids):3d}개 | 넘어지기 0.2초 전 무게중심 평균 {com_pre.mean():+.1f}cm (뒤쪽 {int((com_pre < 0).sum())}개)"
    if arm_sched is not None:
        at = arm_sched_pitch[ft.to(device), ids.to(device)].cpu()  # (n, 4) [왼 어깨, 오른 어깨, 왼 팔꿈치, 오른 팔꿈치]
        before = arm_sched_pitch[torch.clamp(ft - 50, min=0).to(device), ids.to(device)].cpu()
        sh = at[:, :2].mean(dim=1)
        dsh = (at[:, :2] - before[:, :2]).mean(dim=1)
        line += (f" | 넘어질 때 어깨 목표 평균 {sh.mean():+.2f} (양팔 −0.2 보다 뒤/아래 {int(((at[:, 0] > -0.2) & (at[:, 1] > -0.2)).sum())}개, "
                 f"짐 든 자세 −0.5 근처 {int(((at[:, :2] + 0.5).abs().max(dim=1).values < 0.05).sum())}개), "
                 f"1초 동안 어깨 목표 변화 평균 {dsh.mean():+.2f} (뒤로 간 경우 {int((dsh > 0.1).sum())}개, 앞으로 {int((dsh < -0.1).sum())}개)")
    print(line)

# ---- 항목별 보상 (가중치를 곱한 값의 스텝 평균) ----
rew_mean = (rew_sum / cnt[:, None]).cpu()
print("\n========== 항목별 보상: 시나리오별 스텝 평균 (가중치 곱한 값, 유효 스텝) ==========")
print(f"{'항목':28s} " + " ".join(f"{SCENARIOS[s][0][:8]:>10s}" for s in range(n_scen)) + f" {'빠르게 +0~1':>12s} {'빠르게 +4~5':>12s}")
for k, name in enumerate(reward_names):
    cols = [rew_mean[scen_of_env == s, k].mean().item() for s in range(n_scen)]
    extra = [rew_mean[(scen_of_env == 2) & (bin_of_env == b), k].mean().item() for b in (0, n_bins - 1)]
    print(f"{name:28s} " + " ".join(f"{c:10.4f}" for c in cols) + " " + " ".join(f"{c:12.4f}" for c in extra))
print(f"{'합계':28s} " + " ".join(f"{rew_mean[scen_of_env == s].sum(dim=1).mean().item():10.4f}" for s in range(n_scen)))

# 진단: 공중·접촉 구간 길이 분포 (유효 구간, 시나리오별 앞 20개 환경)
print("\n========== 진단: 한 발의 공중·접촉 구간 길이 분포 (스텝, 1스텝 = 0.02초) ==========")
w0 = args_cli.warmup
for s in range(n_scen):
    envs = (scen_of_env == s).nonzero().flatten()[:20].tolist()
    air, stance = [], []
    for e in envs:
        for f in range(2):
            c = contact_hist[w0:, e, f].cpu()
            air += phase_durations(c, False)
            stance += phase_durations(c, True)
    lifts = lift_hist[w0:, envs].flatten()
    lifts = lifts[~torch.isnan(lifts)].cpu() * 100
    if len(air) == 0 or len(stance) == 0 or len(lifts) == 0:  # 발을 떼지 않음 (정지 명령에 양발로 서 있는 경우)
        print(f"{SCENARIOS[s][0]:16s} 공중 {len(air)}개, 접촉 {len(stance)}개, 닿는 순간 {len(lifts)}개 — 분포 계산 생략")
        continue
    a, st = torch.tensor(air, dtype=torch.float), torch.tensor(stance, dtype=torch.float)
    print(f"{SCENARIOS[s][0]:16s} 공중 {len(air):5d}개: 1~2스텝 {(a <= 2).float().mean() * 100:5.1f}%, 중앙값 {a.median():.0f}, "
          f"평균 {a.mean():.1f} | 접촉 {len(stance):5d}개: 1~2스텝 {(st <= 2).float().mean() * 100:5.1f}%, 중앙값 {st.median():.0f}, 평균 {st.mean():.1f}")
    print(f"{'':16s} 닿는 순간 들어 올린 높이(cm): 2cm 미만 {(lifts < 2).float().mean() * 100:5.1f}%, "
          f"중앙값 {lifts.median():.1f}, 상위 10% {lifts.quantile(0.9):.1f}")

# 진단: 접촉 패턴 (L=왼발만, R=오른발만, B=양발, .=공중), 시나리오별 첫 환경, 유효 구간 처음 100스텝(2초)
print("\n========== 진단: 접촉 패턴 (L 왼발만, R 오른발만, B 양발, . 공중 / 1글자 = 0.02초) ==========")
for s in range(n_scen):
    e = int((scen_of_env == s).nonzero().flatten()[0])
    c = contact_hist[w0:w0 + 100, e].cpu()
    chars = "".join("B" if l and r else "L" if l else "R" if r else "." for l, r in c.tolist())
    print(f"{SCENARIOS[s][0]:16s} {chars}")

print("\n========== 시나리오 × 무게: 어깨 pitch 흔들림 std 평균(deg) / 손 z 폭 평균(cm) / 넘어진 환경 ==========")
print(f"{'시나리오':16s} " + " ".join(f"{bin_name(b):>20s}" for b in range(n_bins)))
for s in range(n_scen):
    cells = []
    for b in range(n_bins):
        r = summarize((scen_of_env == s) & (bin_of_env == b))
        cells.append(f"{r['sp_std']:5.2f} / {r['hz_p2p']:5.2f} / {r['fell_envs']:2d}")
    print(f"{SCENARIOS[s][0]:16s} " + " ".join(f"{c:>20s}" for c in cells))

# 환경별 원자료 저장 (학습 폴더, git 제외)
csv_name = {None: "eval_carry", "narrow": "eval_carry_armmotion", "wide": "eval_carry_armmotion_wide"}[args_cli.arm_motion]
out_csv = os.path.join(os.path.dirname(resume_path), csv_name + ("_back" if args_cli.scenarios == "back" else "") + ".csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["env", "scenario", "added_mass_kg", "falls", "lin_err", "yaw_err", "vx", "wz"]
               + [f"{k}_{s}" for k in arm for s in SIDES] + [f"gait_{k}" for k in gait])
    for i in range(N):
        w.writerow([i, SCENARIOS[int(scen_of_env[i])][0], round(added_mass[i].item(), 3), int(falls[i]),
                    round(lin_err[i].item(), 4), round(yaw_err[i].item(), 4), round(vx_mean[i].item(), 4), round(wz_mean[i].item(), 4)]
                   + [round(arm[k][i, si].item(), 4) for k in arm for si in range(2)]
                   + [round(gait[k][i].item(), 4) for k in gait])
print(f"\n[eval] 환경별 원자료: {out_csv}")

env.close()
simulation_app.close()
