"""걷다가 멈추기 평가: 걸음 박자(위상)와 상관없이 정지 명령을 줬을 때, 멈추는 동안 얼마나 돌고 움직이는지와 넘어짐을 잰다.

- 환경마다 정지 명령을 주는 시각을 한 스텝씩 다르게 해서 걸음 주기(0.8초 = 40스텝)의 모든 위상을 고르게 본다
- 손 무게는 --hand_mass (양손 같은 무게), 팔은 짐 든 자세 그대로
- 정지 명령 뒤 6초 동안의 변화를 정지 순간의 방향 기준으로 잰다: 회전(°), 앞·옆 이동(cm), 멈추기까지 걸린 시간

예) python -u scripts\\eval_stop.py --task Humanoid-G1-Loco-V2-Flat-Play-v0 --load_run ".*_loco_v2" --headless
    비교 기준: python -u scripts\\eval_stop.py --task Humanoid-G1-Carry-Gait-ArmRand-Flat-Play-v0
               --load_run 2026-11-20_20-54-06_armrand_scratch --checkpoint model_3999.pt --headless
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate stopping from walking at every gait phase.")
parser.add_argument("--task", type=str, default="Humanoid-G1-Loco-V2-Flat-Play-v0")
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--cmd", type=str, default="0.3,0,0", help="멈추기 전에 걷는 명령 'vx,vy,wz'")
parser.add_argument("--hand_mass", type=float, default=0.0, help="양손에 더하는 무게 (kg, 손 하나당)")
parser.add_argument("--repeat", type=int, default=4, help="위상마다 환경 수 (환경 수 = 40 × repeat)")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

PERIOD_STEPS = 40  # 걸음 주기 0.8초
RAD2DEG = 57.29578
N = PERIOD_STEPS * args_cli.repeat
walk_cmd = [float(v) for v in args_cli.cmd.split(",")]

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
env_cfg.events.add_hand_mass = None
if hasattr(env_cfg.events, "arm_motion"):
    env_cfg.events.arm_motion = None
env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cmd_cfg = env_cfg.commands.base_velocity
cmd_cfg.heading_command = False
cmd_cfg.rel_standing_envs = 0.0
cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)
if hasattr(cmd_cfg, "rel_turn_envs"):
    cmd_cfg.rel_turn_envs = 0.0
cmd_cfg.debug_vis = False

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
robot = u.scene["robot"]
device = u.device
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
print(f"[stop] 체크포인트: {os.path.join(os.path.basename(os.path.dirname(resume_path)), os.path.basename(resume_path))}")

if args_cli.hand_mass > 0:  # eval_carry.py 와 같은 방식: 기본 질량 + 추가, 관성은 질량 비율만큼
    palm = [robot.body_names.index(f"{s}_palm_link") for s in ("left", "right")]
    view = robot.root_physx_view
    dm, di = robot.data.default_mass.cpu(), robot.data.default_inertia.cpu()
    masses = view.get_masses().clone()
    masses[:, palm] = dm[:, palm] + args_cli.hand_mass
    view.set_masses(masses, torch.arange(N))
    inertias = view.get_inertias().clone()
    inertias[:, palm] = di[:, palm] * (masses[:, palm] / dm[:, palm])[..., None]
    view.set_inertias(inertias, torch.arange(N))

term = u.command_manager.get_term("base_velocity")
T_WALK, T_AFTER = 150, 300  # 3초 걷고, 정지 명령 뒤 6초 관찰
stop_at = T_WALK + torch.arange(N, device=device) % PERIOD_STEPS
walk = torch.tensor(walk_cmd, device=device)
term.vel_command_b[:] = walk
obs = env.get_observations()
yaw0, pos0, phase0 = torch.zeros(N, device=device), torch.zeros(N, 2, device=device), torch.zeros(N, device=device)
falls = torch.zeros(N, dtype=torch.bool, device=device)
falls_walk = torch.zeros(N, dtype=torch.bool, device=device)
still_at = torch.full((N,), -1, dtype=torch.long, device=device)
slow_count = torch.zeros(N, dtype=torch.long, device=device)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


with torch.inference_mode():
    for t in range(T_WALK + PERIOD_STEPS + T_AFTER):
        obs, _, _, _ = env.step(policy(obs))
        stopped = t >= stop_at
        term_now = u.termination_manager.terminated
        falls |= term_now & stopped
        falls_walk |= term_now & ~stopped
        term.vel_command_b[:] = torch.where((t + 1 >= stop_at)[:, None], torch.zeros(N, 3, device=device), walk[None].expand(N, -1))
        obs = env.get_observations()
        now = stop_at == t + 1  # 다음 스텝부터 정지 명령
        if bool(now.any()):
            yaw0[now], pos0[now] = robot.data.heading_w[now], robot.data.root_pos_w[now, :2]
            phase0[now] = torch.remainder(u.episode_length_buf[now].float() * u.step_dt / 0.8, 1.0)
        # 멈추기까지 걸린 시간: 정지 명령 뒤 골반 수평 속도 0.05m/s·회전 0.1rad/s 미만이 0.5초 이어진 첫 시점
        slow = (torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1) < 0.05) & (robot.data.root_ang_vel_w[:, 2].abs() < 0.1) & stopped
        slow_count = torch.where(slow, slow_count + 1, torch.zeros_like(slow_count))
        newly = (slow_count >= 25) & (still_at < 0)
        still_at[newly] = (t - stop_at - 24)[newly]
    ok = ~falls & ~falls_walk
    dyaw = wrap(robot.data.heading_w - yaw0) * RAD2DEG
    d = robot.data.root_pos_w[:, :2] - pos0
    c, s = torch.cos(yaw0), torch.sin(yaw0)
    fwd, lat = (c * d[:, 0] + s * d[:, 1]) * 100, (-s * d[:, 0] + c * d[:, 1]) * 100

print(f"\n========== 걷다가 멈추기: 걷는 명령 {tuple(walk_cmd)}, 손 +{args_cli.hand_mass}kg, 환경 {N}개 (위상마다 {args_cli.repeat}개) ==========")
print(f"넘어짐: 걷는 중 {int(falls_walk.sum())}, 정지 명령 뒤 {int(falls.sum())} (넘어진 환경은 아래 통계에서 뺌)")
print("정지 명령 뒤 6초 동안의 변화 (정지 순간의 방향 기준). 위상 = 정지 명령을 준 순간의 걸음 주기 위치")
print(f"{'위상':>8s} {'회전(°) 평균':>12s} {'최소':>7s} {'최대':>7s} {'앞(cm)':>8s} {'옆(cm)':>8s} {'|옆| 최대':>9s}")
for b in range(8):
    sel = ok & (phase0 >= b / 8) & (phase0 < (b + 1) / 8)
    if int(sel.sum()) == 0:
        continue
    print(f"{b / 8:5.3f}~ {dyaw[sel].mean().item():+12.1f} {dyaw[sel].min().item():+7.1f} {dyaw[sel].max().item():+7.1f} "
          f"{fwd[sel].mean().item():+8.1f} {lat[sel].mean().item():+8.1f} {lat[sel].abs().max().item():9.1f}")
st = still_at[ok & (still_at >= 0)].float() * u.step_dt
print(f"전체: 회전 평균 {dyaw[ok].mean().item():+.1f}°, |회전| 평균 {dyaw[ok].abs().mean().item():.1f}°·최대 {dyaw[ok].abs().max().item():.1f}° | "
      f"앞 평균 {fwd[ok].mean().item():+.1f}cm (std {fwd[ok].std().item():.1f}) | 옆 평균 {lat[ok].mean().item():+.1f}cm·|옆| 최대 {lat[ok].abs().max().item():.1f}cm | "
      f"멈추기까지 평균 {st.mean().item():.2f}초·최대 {st.max().item():.2f}초 ({len(st)}/{int(ok.sum())}개가 6초 안에 멈춤)")
env.close()
simulation_app.close()
