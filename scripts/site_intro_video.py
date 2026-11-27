"""10단계 공사장 장면 소개 영상: 위에서 내려다본 장면 → 로봇 옆모습. 로봇 한 대가 걷기 정책으로 뒤돌아 철골 쪽으로 걸어간다.

카메라는 스크립트가 매 스텝 옮긴다 (위에서 본 구간은 천천히 내려오고, 옆모습 구간은 로봇을 따라감).
명령 순서: 서 있기 → 제자리에서 뒤로 돌기 → 철골 앞까지 걷기 → 서 있기. 걷기 정책은 v2 (9단계와 같음).

예) python -u scripts\\site_intro_video.py --headless --task Humanoid-G1-Site-A-v0 --out portfolio\\g1_site_intro.mp4 --gif portfolio\\g1_site_intro.gif --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import math
import os
import random
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a short intro video of the construction site scene.")
parser.add_argument("--task", type=str, default="Humanoid-G1-Site-A-v0")
parser.add_argument("--out", type=str, required=True, help="결과 mp4")
parser.add_argument("--gif", type=str, default=None)
parser.add_argument("--gif_width", type=int, default=720)
parser.add_argument("--gif_fps", type=int, default=10)
parser.add_argument("--gif_speed", type=float, default=1.5)
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--seed", type=int, default=21)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from humanoid_tasks.g1_carry import site_layout as sl  # noqa: E402
from humanoid_tasks.g1_carry import vla_full as vf  # noqa: E402
from humanoid_tasks.g1_carry.vlm_scene import set_lighting  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.seed = args_cli.seed
env_cfg.viewer.resolution = (1280, 720)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
resume_path = get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)), args_cli.load_run, args_cli.checkpoint)
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=u.device)
step_dt = env_cfg.sim.dt * env_cfg.decimation
fps = int(round(1 / step_dt))
cmd_term = u.command_manager.get_term("base_velocity")
robot = u.scene["robot"]
origin = u.scene.env_origins[0].tolist()

scene = vf.sample_scene(random.Random(args_cli.seed), n_boxes=2, on_pad=False)
# 걷는 목표: 철골 앞 약 1.3m (철골의 로봇 쪽 면에서)
goal_x = sl.STEEL_FRAME_POS[0] + 1.3

T_TOP, T_SIDE = 7.0, 9.0  # 위에서 본 구간, 옆모습 구간 (초)
total = int((T_TOP + T_SIDE) / step_dt)
frames = []
with torch.inference_mode():
    env.reset()
    vf.apply_scenes(u, [scene])
    set_lighting(dome_intensity=1000.0, sun_intensity=2000.0, sun_elevation_deg=55.0, sun_azimuth_deg=40.0)
    cmd_term.vel_command_b[:] = 0.0
    obs = env.get_observations()
    u.sim.set_camera_view(eye=(origin[0] - 2.5, origin[1] - 2.0, 26.0), target=(origin[0] - 2.5, origin[1] + 0.2, 0.0))
    for _ in range(30):  # 렌더가 자리 잡을 때까지 (처음 몇 장은 검게 나옴). 녹화하지 않음
        obs, _, _, _ = env.step(policy(obs))
        cmd_term.vel_command_b[:] = 0.0
        obs = env.get_observations()
        u.render()
    for k in range(total):
        t = k * step_dt
        pos = robot.data.root_pos_w[0]
        px, py = float(pos[0]) - origin[0], float(pos[1]) - origin[1]
        yaw = float(robot.data.heading_w[0])
        # ---- 명령: 1.5초 서 있기 → 뒤(-x)를 볼 때까지 돌기 → 철골 앞까지 걷기 → 서 있기 ----
        err = math.atan2(math.sin(math.pi - yaw), math.cos(math.pi - yaw))
        if t < 1.5 or px <= goal_x:
            cmd = (0.0, 0.0, 0.0)
        elif abs(err) > 0.5:
            cmd = (0.0, 0.0, 0.9 * math.copysign(1.0, err))
        else:
            cmd = (0.5, 0.0, max(-0.6, min(0.6, 1.5 * err)))
        # ---- 카메라 ----
        if t < T_TOP:  # 위에서: 높이 26m → 19m 로 천천히 내려오며 장면 전체
            a = t / T_TOP
            eye = (origin[0] - 2.5, origin[1] - 2.0 - 1.0 * a, 26.0 - 7.0 * a)
            target = (origin[0] - 2.5, origin[1] + 0.2, 0.0)
        else:  # 옆모습: 로봇을 따라가며 공사장을 배경으로
            eye = (origin[0] + px + 0.8, origin[1] + py - 3.6, 1.5)
            target = (origin[0] + px - 0.3, origin[1] + py, 0.85)
        u.sim.set_camera_view(eye=eye, target=target)
        obs, _, _, _ = env.step(policy(obs))
        cmd_term.vel_command_b[0] = torch.tensor(cmd, device=u.device)
        obs = env.get_observations()
        frames.append(u.render().copy())

os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
iio.imwrite(args_cli.out, np.stack(frames), fps=fps)
print(f"[site] mp4 {os.path.getsize(args_cli.out) / 1e6:.2f}MB, {len(frames) / fps:.1f}초 → {args_cli.out}")
if args_cli.gif:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf_arg = (f"setpts=PTS/{args_cli.gif_speed},fps={args_cli.gif_fps},scale={args_cli.gif_width}:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", args_cli.out, "-filter_complex", vf_arg, args_cli.gif], check=True)
    print(f"[site] gif {os.path.getsize(args_cli.gif) / 1e6:.2f}MB → {args_cli.gif}")

env.close()
simulation_app.close()
