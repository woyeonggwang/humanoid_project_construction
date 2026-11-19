"""짐 든 채 걷기 정책 평가 (영상 없이 숫자만).

학습된 정책을 명령 속도 시나리오 × 손 무게 구간으로 나눈 환경에서 돌리고 기록한다.
- 넘어짐: 환경별 base_contact 종료 횟수
- 팔 흔들림: 매 스텝 어깨·팔꿈치 관절의 (실제값 - 기본값) → 평균(꾸준한 처짐), 표준편차·최대(흔들림 폭)
- 손 위치: 골반 기준 손 위치(cm)의 평균 이동, 표준편차, 최대-최소
- 속도 따라가기: 명령과 실제 속도의 차이

설정
- 랜덤 밀기·외력 끔, 관측 노이즈 끔 (Play 설정)
- 명령은 시나리오 값으로 고정: 방향 명령·정지 환경을 끄고, 매 스텝 덮어쓴 뒤 관측을 다시 계산
  (넘어져 리셋된 환경은 명령이 다시 뽑히므로 매 스텝 덮어씀)
- 손 무게: 시나리오 × 구간마다 같은 수의 환경. 구간 안에서 무작위, 양손 같은 무게 (상자를 드는 상황)
  학습 때의 무게 무작위화(add_hand_mass)는 끄고 같은 방식(질량 + 관성 비율)으로 직접 넣음
- 흔들림·손 위치 통계는 에피소드 시작 후 --warmup 스텝(처음 처지는 구간)을 뺀다. 넘어짐은 전체 구간

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
cmd_cfg.debug_vis = False

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
        obs, _, dones, _ = env.step(actions)
        falls += unwrapped.termination_manager.terminated.long()
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

print("\n========== 시나리오 × 무게: 어깨 pitch 흔들림 std 평균(deg) / 손 z 폭 평균(cm) / 넘어진 환경 ==========")
print(f"{'시나리오':16s} " + " ".join(f"{bin_name(b):>20s}" for b in range(n_bins)))
for s in range(n_scen):
    cells = []
    for b in range(n_bins):
        r = summarize((scen_of_env == s) & (bin_of_env == b))
        cells.append(f"{r['sp_std']:5.2f} / {r['hz_p2p']:5.2f} / {r['fell_envs']:2d}")
    print(f"{SCENARIOS[s][0]:16s} " + " ".join(f"{c:>20s}" for c in cells))

# 환경별 원자료 저장 (학습 폴더, git 제외)
out_csv = os.path.join(os.path.dirname(resume_path), "eval_carry.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["env", "scenario", "added_mass_kg", "falls", "lin_err", "yaw_err", "vx", "wz"]
               + [f"{k}_{s}" for k in arm for s in SIDES])
    for i in range(N):
        w.writerow([i, SCENARIOS[int(scen_of_env[i])][0], round(added_mass[i].item(), 3), int(falls[i]),
                    round(lin_err[i].item(), 4), round(yaw_err[i].item(), 4), round(vx_mean[i].item(), 4), round(wz_mean[i].item(), 4)]
                   + [round(arm[k][i, si].item(), 4) for k in arm for si in range(2)])
print(f"\n[eval] 환경별 원자료: {out_csv}")

env.close()
simulation_app.close()
