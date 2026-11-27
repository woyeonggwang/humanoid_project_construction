"""11단계: 용접 동작(스크립트 IK) 실행·평가·녹화.

한 번의 시도 = 철골 앞 설 자리에서 조금 떨어진 곳에서 시작 → 준비 자세 → 설 자리로 걷기 → 멈춤 → 토치를 띄운 점으로 →
용접선에서 약 1cm 를 유지하며 일정한 속도로 이동 → 빼기 → 토치 거두기 → 뒤로 물러나 팔 내리기.
- 설 자리는 site_layout.WELD_STANCE 의 값에 ±scatter_xy·±scatter_yaw 를 더해 흩는다 (환경마다 무작위).
- 성공 = 끝까지 진행 + 토치·손이 철골에 닿지 않음 + 넘어지지 않음.
- 잴 것: 경로 오차(토치 끝과 그 순간의 목표점 사이), 띄운 거리, 속도, 용접선을 덮은 비율, 토치 각도, 걸린 시간.

예) python -u scripts\\weld_eval.py --headless --trials 20 --num_envs 24 --out data\\weld\\eval.json
    python -u scripts\\weld_eval.py --headless --video --seams column_B_seam_4 --name weld_b4 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run / evaluate / record the scripted welding motion.")
parser.add_argument("--task", type=str, default=None, help="기본: 평가는 카메라 없는 장면, 녹화는 머리 카메라가 있는 장면")
parser.add_argument("--seams", type=str, default=None, help="용접선 이름 (쉼표로). 기본 = site_layout.WELD_STANCE 에 있는 것 전부")
parser.add_argument("--trials", type=int, default=20, help="용접선마다 시도 수")
parser.add_argument("--num_envs", type=int, default=24)
parser.add_argument("--scatter_xy", type=float, default=0.05, help="설 자리를 흩는 범위 (m, 앞뒤·좌우 각각 ±)")
parser.add_argument("--scatter_yaw", type=float, default=5.0, help="설 방향을 흩는 범위 (도, ±)")
parser.add_argument("--start_back", type=float, default=1.2, help="설 자리에서 이만큼 뒤에서 시작해 걸어 들어감 (m)")
parser.add_argument("--stance", type=str, default=None, help="설 자리 바꿔 보기: '용접선:D,L,yaw;용접선:D,L,yaw'")
parser.add_argument("--speed", type=float, default=None, help="용접 속도 (m/s). 기본 = weld.WELD_SPEED")
parser.add_argument("--fx", action="store_true", help="불꽃·비드 효과를 켠다 (녹화에서는 항상 켬)")
parser.add_argument("--no_fx", action="store_true", help="녹화에서도 효과를 끈다 (속도·메모리 비교용)")
parser.add_argument("--no_stow", action="store_true", help="용접 뒤 물러나 팔을 내리는 구간을 뺀다")
parser.add_argument("--no_obs_clamp", action="store_true", help="걷기 정책에 팔 관절 각도를 자르지 않고 그대로 넣는다 (비교용)")
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--seed", type=int, default=11000)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--video", action="store_true", help="환경 0 을 녹화 (환경 1개, 시도 1번)")
parser.add_argument("--eye", type=str, default=None, help="녹화: 카메라 위치 (용접선 가운데 기준 'x,y,z')")
parser.add_argument("--lookat", type=str, default="0.12,0.0,-0.03")
parser.add_argument("--video_dir", type=str, default=os.path.join("data", "weld", "videos"))
parser.add_argument("--name", type=str, default="weld")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.num_envs, args_cli.trials = 1, 1
    args_cli.enable_cameras = True
if args_cli.task is None:
    args_cli.task = "Humanoid-G1-Weld-v0" if args_cli.video else "Humanoid-G1-Weld-NoCam-v0"
if "NoCam" not in args_cli.task and "Probe" not in args_cli.task:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from humanoid_tasks.g1_carry import pick_place_machine as ppm  # noqa: E402
from humanoid_tasks.g1_carry import site_layout as sl  # noqa: E402
from humanoid_tasks.g1_carry import weld as wd  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

N = args_cli.num_envs


def gpu_used_mb() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


stances = dict(sl.WELD_STANCE)
if args_cli.stance:
    for item in args_cli.stance.split(";"):
        name, vals = item.split(":")
        stances[name.strip()] = tuple(float(v) for v in vals.split(","))
seam_names = [s.strip() for s in args_cli.seams.split(",")] if args_cli.seams else list(sl.WELD_STANCE)
rng = random.Random(args_cli.seed)
trials = []
for nm in seam_names:
    for _ in range(args_cli.trials):
        trials.append({"seam": nm, "dx": rng.uniform(-1, 1) * args_cli.scatter_xy, "dy": rng.uniform(-1, 1) * args_cli.scatter_xy,
                       "dyaw": math.radians(rng.uniform(-1, 1) * args_cli.scatter_yaw),
                       "start_lat": rng.uniform(-0.10, 0.10), "start_yaw": rng.uniform(-0.2, 0.2)})
rng.shuffle(trials)
rounds = [trials[k:k + N] for k in range(0, len(trials), N)]
print(f"[weld] 용접선 {len(seam_names)}개 × {args_cli.trials}번 = {len(trials)}번, 묶음 {len(rounds)}개 (환경 {N}개), "
      f"설 자리 흩기 ±{args_cli.scatter_xy * 100:.0f}cm·±{args_cli.scatter_yaw:.0f}°")

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
if args_cli.video:
    sm = sl.weld_seam_frames()[seam_names[0]]
    mid = [0.5 * (a + b) for a, b in zip(sm["start"], sm["end"])]
    side = 1.0 if sm["normal"][1] >= 0 else -1.0
    eye = [float(v) for v in args_cli.eye.split(",")] if args_cli.eye else [0.45, side * 1.25, 0.22]
    look = [float(v) for v in args_cli.lookat.split(",")]
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(m + e for m, e in zip(mid, eye))
    env_cfg.viewer.lookat = tuple(m + e for m, e in zip(mid, look))
    env_cfg.viewer.resolution = (1280, 720)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
resume_path = get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)), args_cli.load_run, args_cli.checkpoint)
step_dt = env_cfg.sim.dt * env_cfg.decimation
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
if args_cli.video:
    os.makedirs(args_cli.video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_folder=args_cli.video_dir, step_trigger=lambda step: step == 0,
                                   video_length=9000, name_prefix=args_cli.name, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
device = u.device
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
robot = u.scene["robot"]
cmd_term = u.command_manager.get_term("base_velocity")
use_fx = (args_cli.video and not args_cli.no_fx) or args_cli.fx
fx = wd.WeldEffects(N, device) if use_fx else None
machine = ppm.PickPlaceMachine(u, until="stand")
welder = wd.WeldMachine(u, machine, effects=fx, speed=args_cli.speed or wd.WELD_SPEED, hide_torch=False)
obs_clamp = (lambda o: o) if args_cli.no_obs_clamp else wd.ArmObsClamp(u, robot)
has_cam = "vlm_cam" in u.scene.sensors
mem = {"start": gpu_used_mb()}
mem["max"] = mem["start"]

records, head_frames, events = [], [], []
n_steps, wall_step, wall_weld = 0, [], []
t_start = time.time()


def step_once(obs):
    global n_steps
    t0 = time.time()
    welder.pre_step()
    machine.pre_step()
    obs, _, _, _ = env.step(policy(obs_clamp(obs)))
    term = u.termination_manager.terminated.clone()
    cmd = welder.post_step(machine.post_step(term), term)
    cmd_term.vel_command_b[:] = cmd
    obs = env.get_observations()
    n_steps += 1
    (wall_weld if int(welder.phase[0]) == wd.W_WELD else wall_step).append(time.time() - t0)
    if args_cli.video and has_cam:
        head_frames.append(u.scene.sensors["vlm_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().copy())
    if n_steps % 500 == 0:
        mem["max"] = max(mem["max"], gpu_used_mb())
    return obs, term


with torch.inference_mode():
    for r_i, group in enumerate(rounds):
        real = len(group)
        group = group + [group[-1]] * (N - real)
        env.reset()
        machine.reset()
        welder.reset()
        origins = u.scene.env_origins
        target = torch.zeros(N, 3, device=device)
        root = robot.data.default_root_state[:, :7].clone()
        for e, tr in enumerate(group):
            x, y, yaw = sl.weld_stance(tr["seam"], origins[e].tolist(), stances[tr["seam"]])
            c, s = math.cos(yaw), math.sin(yaw)
            tx, ty, tyaw = x + c * tr["dx"] - s * tr["dy"], y + s * tr["dx"] + c * tr["dy"], yaw + tr["dyaw"]
            target[e] = torch.tensor([tx, ty, tyaw], device=device)
            # 시작 자리: 설 자리에서 로봇의 처음 자리(환경 원점) 쪽으로 start_back 만큼 떨어진 곳, 설 자리를 보고 섬
            # (설 방향의 바로 뒤에 두면 A 기둥 쪽에서는 안전 고깔 옆에서 시작해 걸려 넘어졌음)
            ox, oy = float(origins[e, 0]), float(origins[e, 1])
            hx, hy = ox - tx, oy - ty
            hn = math.hypot(hx, hy)
            hx, hy = hx / hn, hy / hn
            sx = tx + hx * args_cli.start_back - hy * tr["start_lat"]
            sy = ty + hy * args_cli.start_back + hx * tr["start_lat"]
            syaw = math.atan2(-hy, -hx) + tr["start_yaw"]
            root[e, 0], root[e, 1] = sx, sy
            root[e, 3:7] = torch.tensor([math.cos(syaw / 2), 0.0, 0.0, math.sin(syaw / 2)], device=device)
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=device))
        cmd_term.vel_command_b[:] = 0.0
        obs = env.get_observations()
        for _ in range(25):
            obs, _ = step_once(obs)
        all_mask = torch.ones(N, dtype=torch.bool, device=device)
        welder.begin(all_mask, [tr["seam"] for tr in group], target)
        result = [None] * N
        state = ["weld"] * N  # weld → stow → ok / fail
        last_phase = -1
        for _ in range(6000):
            obs, term = step_once(obs)
            if args_cli.video and int(welder.phase[0]) != last_phase:
                last_phase = int(welder.phase[0])
                events.append({"step": n_steps, "text": wd.W_NAMES[last_phase]})
            for e in range(N):
                ph = int(welder.phase[e])
                if state[e] == "weld" and ph in (wd.W_DONE, wd.W_FAIL):
                    result[e] = welder.result(e)
                    if ph == wd.W_FAIL:
                        state[e] = "fail"
                    elif args_cli.no_stow:
                        state[e] = "ok"
                    else:
                        state[e] = "stow"
                        m = torch.zeros(N, dtype=torch.bool, device=device)
                        m[e] = True
                        welder.stow(m)
                elif state[e] == "stow" and ph in (wd.W_IDLE, wd.W_FAIL):
                    if ph == wd.W_FAIL:
                        result[e]["fail"], result[e]["fail_phase"] = welder.fail_reason[e], welder.fail_phase[e]
                        state[e] = "fail"
                    else:
                        state[e] = "ok"
                    result[e]["t_all_s"] = (machine.step - float(welder.rec["t_begin"][e])) * step_dt
            if all(s in ("ok", "fail") for s in state):
                break
        for e in range(real):
            row = result[e] if result[e] is not None else {"seam": group[e]["seam"], "fail": "시간 초과", "fail_phase": wd.W_NAMES[int(welder.phase[e])]}
            row["success"] = state[e] == "ok"
            row["scatter"] = [round(group[e]["dx"] * 100, 1), round(group[e]["dy"] * 100, 1), round(math.degrees(group[e]["dyaw"]), 1)]
            records.append(row)
        if args_cli.video:
            for _ in range(60):
                obs, _ = step_once(obs)
        print(f"[weld] 묶음 {r_i + 1}/{len(rounds)}: 누적 성공 {sum(r['success'] for r in records)}/{len(records)}, {time.time() - t_start:.0f}초", flush=True)
mem["max"] = max(mem["max"], gpu_used_mb())

# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
print(f"\n========== 용접 평가: 용접 속도 {welder.speed * 100:.1f}cm/s, 설 자리 흩기 ±{args_cli.scatter_xy * 100:.0f}cm·±{args_cli.scatter_yaw:.0f}°, 시도 {len(records)}번 ==========")
summary = {}
for nm in seam_names:
    rows = [r for r in records if r["seam"] == nm]
    ok = [r for r in rows if r["success"]]
    meas = [r for r in rows if r.get("err_max_mm") is not None and r.get("t_weld_s", 0) > 0]  # 용접 구간을 끝까지 간 것
    fails = {}
    for r in rows:
        if not r["success"]:
            key = f"{r.get('fail')} ({r.get('fail_phase')})"
            fails[key] = fails.get(key, 0) + 1
    falls = sum(1 for r in rows if r.get("fail") == "넘어짐")
    line = f"[{nm}] 설 자리 D {stances[nm][0]:.2f} L {stances[nm][1]:+.2f} yaw {stances[nm][2]:+.0f}° | 성공 {len(ok)}/{len(rows)} | 넘어짐 {falls}"
    item = {"n": len(rows), "success": len(ok), "falls": falls, "fails": fails, "stance": stances[nm]}
    if meas:
        f = lambda key, fn=np.mean: float(fn([r[key] for r in meas]))  # noqa: E731
        item.update({
            "err_mean_mm": f("err_mean_mm"), "err_max_mm": f("err_max_mm", np.max), "cross_mean_mm": f("cross_mean_mm"),
            "cross_max_mm": f("cross_max_mm", np.max), "gap_mean_mm": f("gap_mean_mm"), "gap_dev_max_mm": f("gap_dev_max_mm", np.max),
            "speed_mean_mm_s": f("speed_mean_mm_s"), "speed_std_mm_s": f("speed_std_mm_s"), "speed_min_mm_s": f("speed_min_mm_s", np.min),
            "speed_max_mm_s": f("speed_max_mm_s", np.max), "coverage_mean": f("coverage"), "coverage_min": f("coverage", np.min),
            "angle_mean_deg": f("angle_mean_deg"), "angle_max_deg": f("angle_max_deg", np.max),
            "angle_err_mean_deg": f("angle_err_mean_deg"), "angle_err_max_deg": f("angle_err_max_deg", np.max),
            "torch_clear_min_mm": f("torch_clear_mm", np.min), "hand_clear_min_mm": f("hand_clear_mm", np.min),
            "t_weld_s": f("t_weld_s"), "t_total_s": f("t_total_s"), "tilt_max_deg": f("tilt_max_deg", np.max),
            "stance_err_cm": [float(np.mean([abs(r["stance_x_cm"]) for r in meas])), float(np.mean([abs(r["stance_y_cm"]) for r in meas]))],
            "stance_err_yaw_deg": float(np.mean([abs(r["stance_yaw_deg"]) for r in meas])),
        })
        t_all = [r["t_all_s"] for r in ok if "t_all_s" in r]
        item["t_all_s"] = float(np.mean(t_all)) if t_all else None
        line += (f"\n    경로 오차 평균 {item['err_mean_mm']:.1f} 최대 {item['err_max_mm']:.1f}mm (경로에서 옆으로 벗어난 거리 평균 {item['cross_mean_mm']:.1f} 최대 {item['cross_max_mm']:.1f}mm)"
                 f" | 띄운 거리 평균 {item['gap_mean_mm']:.1f}mm, 10mm 에서 최대 {item['gap_dev_max_mm']:.1f}mm 벗어남"
                 f"\n    속도 평균 {item['speed_mean_mm_s']:.1f}mm/s, 흔들림(std) {item['speed_std_mm_s']:.1f}, 범위 {item['speed_min_mm_s']:.1f}~{item['speed_max_mm_s']:.1f}"
                 f" | 덮은 비율 평균 {item['coverage_mean']:.1%} 최소 {item['coverage_min']:.1%}"
                 f" | 토치 각도(바깥 방향의 반대에서) 평균 {item['angle_mean_deg']:.1f}° 최대 {item['angle_max_deg']:.1f}°, 의도한 방향과의 차이 평균 {item['angle_err_mean_deg']:.1f}° 최대 {item['angle_err_max_deg']:.1f}°"
                 f"\n    철골과의 거리 최소: 토치 {item['torch_clear_min_mm']:.1f}mm, 손 {item['hand_clear_min_mm']:.1f}mm | 기울기 최대 {item['tilt_max_deg']:.1f}°"
                 f" | 멈춘 자리 오차 앞뒤 {item['stance_err_cm'][0]:.1f}·좌우 {item['stance_err_cm'][1]:.1f}cm·방향 {item['stance_err_yaw_deg']:.1f}°"
                 f" | 시간: 용접 {item['t_weld_s']:.1f}초, 걷기~토치 거두기 {item['t_total_s']:.1f}초"
                 + (f", 팔 내리기까지 {item['t_all_s']:.1f}초" if item["t_all_s"] else ""))
    if fails:
        line += "\n    실패: " + ", ".join(f"{k} {v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1]))
    print(line)
    summary[nm] = item
total_ok = sum(r["success"] for r in records)
print(f"전체: 성공 {total_ok}/{len(records)}, 넘어짐 {sum(1 for r in records if r.get('fail') == '넘어짐')}")
print(f"한 스텝 평균 {np.mean(wall_step) * 1000:.1f}ms, 환경 0 이 용접하는 동안 {np.mean(wall_weld) * 1000 if wall_weld else float('nan'):.1f}ms "
      f"(환경 {N}개, 효과 {'켬' if use_fx else '끔'}) | GPU 전체 사용량 (MB): 시작 {mem['start']}, 최대 {mem['max']} | 전체 {time.time() - t_start:.0f}초")
if args_cli.out:
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w", encoding="utf-8") as fh:
        json.dump({"seed": args_cli.seed, "speed": welder.speed, "scatter_xy": args_cli.scatter_xy, "scatter_yaw_deg": args_cli.scatter_yaw,
                   "summary": summary, "step_ms": float(np.mean(wall_step) * 1000), "step_ms_welding": float(np.mean(wall_weld) * 1000) if wall_weld else None, "fx": use_fx, "gpu_mb": mem, "records": records},
                  fh, ensure_ascii=False, indent=1)
    print(f"결과 저장: {args_cli.out}")
if args_cli.video:
    info = {"name": args_cli.name, "seam": seam_names[0], "success": bool(records[0]["success"]), "fps": int(round(1 / step_dt)),
            "total_steps": n_steps, "events": events, "result": records[0]}
    with open(os.path.join(args_cli.video_dir, f"{args_cli.name}_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    if head_frames:
        iio.imwrite(os.path.join(args_cli.video_dir, f"{args_cli.name}_head.mp4"), np.stack(head_frames), fps=info["fps"])
    print(f"[weld] 영상: {os.path.join(args_cli.video_dir, args_cli.name + '-step-0.mp4')}")

env.close()
simulation_app.close()
