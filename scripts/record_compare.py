"""근접 비교 영상 녹화: 한 대를 옆에서 따라가며 같은 명령 순서로 찍는다 (영상 저장 위치를 따로 정해 play 영상을 덮어쓰지 않음).

- 설정: *-Record-v0 과제 (Play 설정 상속, 한 대, 골반을 따라가는 옆 카메라, 앞을 보고 시작, 양손 +2.5kg)
- 명령: --schedule 의 순서대로 고정 (기본: 앞으로 0.5m/s 6초 → 정지 4초). 매 스텝 명령을 덮어쓰고 관측을 다시 계산
- 영상: <학습 폴더>/videos/compare/<name>-step-0.mp4 (1 스텝 = 0.02초, 50fps)
- RTX 렌더러가 필요하므로 --kit_args "--/app/vulkan=false --/renderer/active=d3d12" 와 함께 실행 (CLAUDE.md 시행착오)

예) python -u scripts\\record_compare.py --task Humanoid-G1-Carry-Flat-Record-v0 --load_run 2026-11-19_17-33-52 --headless
    --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a side-view comparison video of one robot.")
parser.add_argument("--task", type=str, required=True, help="*-Record-v0 과제")
parser.add_argument("--load_run", type=str, required=True, help="experiment_name 폴더 아래 학습 폴더 이름")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt", help="체크포인트 파일 패턴 (기본: 마지막)")
parser.add_argument("--schedule", type=str, default="0.5,0,0:6;0,0,0:4",
                    help="명령 순서 'vx,vy,wz:초;...' (기본: 앞으로 0.5m/s 6초 → 정지 4초)")
parser.add_argument("--name", type=str, default="compare", help="영상 파일 이름 앞부분")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # 영상 녹화에 필요 (원본 play.py --video 와 같음)

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

# 명령 순서 → 스텝별 명령
segments = []
for part in args_cli.schedule.split(";"):
    vel, sec = part.split(":")
    segments.append((tuple(float(x) for x in vel.split(",")), float(sec)))

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.seed = args_cli.seed
cmd_cfg = env_cfg.commands.base_velocity
cmd_cfg.heading_command = False
cmd_cfg.rel_standing_envs = 0.0
cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)
if hasattr(cmd_cfg, "rel_turn_envs"):
    cmd_cfg.rel_turn_envs = 0.0
if hasattr(env_cfg.events, "arm_motion"):  # 팔 무작위 움직임판도 비교 영상에서는 짐 든 자세로
    env_cfg.events.arm_motion = None

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
video_dir = os.path.join(os.path.dirname(resume_path), "videos", "compare")

step_dt = env_cfg.sim.dt * env_cfg.decimation
schedule = []
for vel, sec in segments:
    schedule += [vel] * int(round(sec / step_dt))
T = len(schedule)

env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env, video_folder=video_dir, step_trigger=lambda step: step == 0, video_length=T,
    name_prefix=args_cli.name, disable_logger=True,
)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
unwrapped = env.unwrapped
robot = unwrapped.scene["robot"]
device = unwrapped.device

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
print(f"[record] 체크포인트: {resume_path}")
print(f"[record] 명령 순서: {segments}, {T}스텝")

cmd_term = unwrapped.command_manager.get_term("base_velocity")
sched = torch.tensor(schedule, device=device)
cmd_term.vel_command_b[:] = sched[0]
obs = env.get_observations()
falls = 0
seg_vx = {}
with torch.inference_mode():
    for t in range(T):
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        falls += int(unwrapped.termination_manager.terminated.sum())
        nxt = sched[min(t + 1, T - 1)]
        cmd_term.vel_command_b[:] = nxt
        obs = env.get_observations()
        key = tuple(sched[t].tolist())
        seg_vx.setdefault(key, []).append(robot.data.root_lin_vel_b[0, 0].item())

for key, vals in seg_vx.items():
    tail = vals[len(vals) // 3:]  # 명령이 바뀐 직후 1/3 은 빼고 평균
    print(f"[record] 명령 {key}: 실제 vx 평균 {sum(tail) / len(tail):.3f} m/s (구간 뒤쪽 2/3)")
print(f"[record] 넘어짐: {falls}회")
print(f"[record] 영상: {os.path.join(video_dir, args_cli.name + '-step-0.mp4')}")

env.close()
simulation_app.close()
