"""상자 앞에 선 상태에서 '뻗기 → 잡기 → 들기' 를 VLA(SmolVLA) 또는 스크립트 IK 로 실행하고 평가한다.

- 받침대 앞까지 걷고 멈추는 것은 두 방식 모두 스크립트(상태 머신)가 한다. 그 뒤의 팔 동작과 잡기 신호만 다르다
  - --mode vla: VLA 작업자(vla/policy_worker.py, 별도 환경의 파이썬)에 사진·상태·명령을 보내 행동 묶음(50스텝)을 받고,
    그중 앞의 --n_action_steps 스텝을 실행한 뒤 다시 요청한다. 응답을 기다리는 동안 시뮬레이션은 멈춰 있다
  - --mode script: 수집 때와 같은 스크립트 IK (비교 기준)
- 성공: 상자가 붙은 채 처음 높이보다 5cm 넘게 들려 0.5초 유지 (두 방식 공통, vla_pick.LiftTracker). 제한 시간 --max_time 초
- 실패 단계 (vla): 못 닿음 / 잡기 신호 시점 틀림 / 들다 놓침 / 붙였지만 못 듦 / 넘어짐

VLA 를 띄울 파이썬은 --vla_python 또는 환경 변수 VLA_PYTHON 으로 준다.
예) python -u scripts\\vla_pick_eval.py --headless --mode vla --ckpt <체크포인트>/pretrained_model --episodes 100 --num_envs 25 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate reach-grasp-lift with a VLA policy or the scripted IK.")
parser.add_argument("--mode", choices=["vla", "script"], default="vla")
parser.add_argument("--ckpt", type=str, default=None, help="VLA 체크포인트의 pretrained_model 폴더")
parser.add_argument("--vla_python", type=str, default=os.environ.get("VLA_PYTHON"), help="VLA 작업자를 띄울 파이썬 실행 파일")
parser.add_argument("--n_action_steps", type=int, default=25, help="행동 묶음 50스텝 중 몇 스텝을 실행하고 다시 요청할지")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--pos_range", type=float, default=0.10)
parser.add_argument("--yaw_range", type=float, default=15.0)
parser.add_argument("--mass_range", type=str, default="2.0,3.0")
parser.add_argument("--color", type=str, default=None, help="상자 색을 하나로 고정. 'cycle' 이면 묶음마다 red, blue, green, yellow 를 차례로 (기본: 묶음마다 무작위)")
parser.add_argument("--two_box", action="store_true",
                    help="2차: 넓은 받침대에 색이 다른 작은 상자 두 개, 명령한 색의 상자를 집음 (과제 Humanoid-G1-VLA-Pick2-v0)")
parser.add_argument("--side", choices=["left", "right"], default=None, help="2차: 집을 상자의 자리를 고정 (녹화용)")
parser.add_argument("--other_color", type=str, default=None, help="2차: 다른 상자의 색을 고정 (녹화용)")
parser.add_argument("--stance_y", type=str, default=None,
                    help="로봇이 서는 자리를 상자 정면에서 옆으로 벗어나게: 크기 범위 '최소,최대' m (방향은 무작위). 학습 범위 밖 시험용")
parser.add_argument("--stance_yaw", type=str, default=None, help="로봇이 상자를 비스듬히 보게: 크기 범위 '최소,최대' 도 (방향은 무작위)")
parser.add_argument("--max_time", type=float, default=12.0, help="팔 동작 제한 시간 (초)")
parser.add_argument("--keep_steps", type=int, default=0, help="VLA: 성공한 뒤에도 이 스텝 수만큼 더 실행 (녹화용)")
parser.add_argument("--task", type=str, default="Humanoid-G1-VLA-Pick-v0")
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--out", type=str, default=None, help="결과 JSON 을 저장할 파일")
parser.add_argument("--video", action="store_true", help="환경 0 을 녹화 (num_envs 1 권장)")
parser.add_argument("--view", choices=["side", "front"], default="side")
parser.add_argument("--video_dir", type=str, default=os.path.join("data", "vla_pick", "videos"))
parser.add_argument("--name", type=str, default="vla_pick")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from humanoid_tasks.g1_carry import pick_place_machine as ppm  # noqa: E402
from humanoid_tasks.g1_carry import vla_pick as vp  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

MARK = "@@VLA@@"
N = args_cli.num_envs
if args_cli.two_box and args_cli.task == "Humanoid-G1-VLA-Pick-v0":
    args_cli.task = "Humanoid-G1-VLA-Pick2-v0"
mass_range = tuple(float(v) for v in args_cli.mass_range.split(","))
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
if args_cli.video:
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.15, -2.3, 0.35) if args_cli.view == "side" else (2.2, -1.2, 0.75)
    env_cfg.viewer.lookat = (0.15, 0.0, 0.05)
    env_cfg.viewer.resolution = (1280, 720)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
step_dt = env_cfg.sim.dt * env_cfg.decimation
T_ARM = int(args_cli.max_time / step_dt)
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
if args_cli.video:
    os.makedirs(args_cli.video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_folder=args_cli.video_dir, step_trigger=lambda step: step == 0,
                                   video_length=4000, name_prefix=args_cli.name, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
device = u.device
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)

machine = ppm.PickPlaceMachine(u, until="lift" if args_cli.mode == "script" and not args_cli.two_box else "stand")
seq = vp.SidePickScript(machine)  # 2차의 스크립트 IK
tracker = vp.LiftTracker(machine)
executor = vp.VlaArmExecutor(machine)
cmd_term = u.command_manager.get_term("base_velocity")
rng = random.Random(args_cli.seed)

# ---------------------------------------------------------------------------
# VLA 작업자
# ---------------------------------------------------------------------------
worker, worker_info, tmp_dir = None, {}, None
latencies = []
if args_cli.mode == "vla":
    if not args_cli.vla_python or not args_cli.ckpt:
        raise SystemExit("--mode vla 에는 --ckpt 와 --vla_python(또는 환경 변수 VLA_PYTHON)이 필요합니다")
    tmp_dir = tempfile.mkdtemp(prefix="vla_obs_")
    worker = subprocess.Popen([args_cli.vla_python, "-X", "utf8", str(ROOT / "vla" / "policy_worker.py"), "--ckpt", os.path.abspath(args_cli.ckpt)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(tmp_dir, "worker.log"), "w", encoding="utf-8"),
                              text=True, encoding="utf-8")


def worker_read():
    while True:
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"VLA 작업자가 끝났습니다 ({os.path.join(tmp_dir, 'worker.log')} 참고)")
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])


def vla_infer(img: torch.Tensor, state: torch.Tensor, tasks: list[str]) -> torch.Tensor:
    """사진 (B, H, W, 3)·상태 (B, 11)·명령 → 행동 묶음 (B, 50, 11)."""
    path = os.path.join(tmp_dir, "obs.npz")
    np.savez(path, images=img.cpu().numpy(), state=state.cpu().numpy().astype(np.float32), tasks=np.array(tasks))
    worker.stdin.write(json.dumps({"obs": path}) + "\n")
    worker.stdin.flush()
    ans = worker_read()
    latencies.append((ans["latency_s"], ans["batch"]))
    return torch.from_numpy(np.load(ans["act"])).to(device)


if worker is not None:
    worker_info = worker_read()
    print(f"[eval] VLA 작업자 준비: 불러오기 {worker_info['load_s']}초, GPU {worker_info['gpu_mb']}MB, 행동 묶음 {worker_info['chunk_size']}스텝")

obs = None
n_steps = 0  # env.step 을 부른 횟수 (녹화한 영상의 프레임 번호와 맞추려고)
head_frames, arm_span = [], [None, None]  # --video: 팔을 움직이는 동안의 머리 카메라 사진, 그 구간의 스텝 번호


def grab_head():
    if args_cli.video:
        if arm_span[0] is None:
            arm_span[0] = n_steps
        arm_span[1] = n_steps
        head_frames.append(u.scene.sensors["vla_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().copy())


def tick_machine():
    global obs, n_steps
    n_steps += 1
    obs, _, _, _ = env.step(policy(obs))
    term = u.termination_manager.terminated.clone()
    cmd_term.vel_command_b[:] = machine.post_step(term)
    obs = env.get_observations()
    return term


records = []
t_start = time.time()
n_round = 0
with torch.inference_mode():
    while len(records) < args_cli.episodes:
        n_round += 1
        env.reset()
        machine.reset()
        other, rounds = None, None
        round_color = ("red", "blue", "green", "yellow")[(n_round - 1) % 4] if args_cli.color == "cycle" else args_cli.color
        if args_cli.two_box:
            seq.reset()
            color, rounds = vp.sample_round2(rng, N, args_cli.pos_range, args_cli.yaw_range, mass_range, color=round_color,
                                            side={"left": 1, "right": -1, None: None}[args_cli.side], other=args_cli.other_color)
            box, stand_xy, stand_yaw, side = vp.setup_round2(u, machine, color, rounds)
            other = vp.OtherBox(u, machine, rounds, side)
            masses = [r["mass"] for r in rounds]
            tasks = [vp.task_sentence(rng, color) for _ in range(N)]
        else:
            color, layouts, masses = vp.sample_round(rng, N, args_cli.pos_range, args_cli.yaw_range, mass_range, color=round_color)
            box, stand_xy, stand_yaw = vp.setup_round(u, machine, color, layouts, masses)
            tasks = [vp.task_sentence(rng) for _ in range(N)]
        vp.random_lighting(rng)
        cmd_term.vel_command_b[:] = 0.0
        obs = env.get_observations()
        for _ in range(25):
            machine.pre_step()
            tick_machine()
        # 서는 자리를 일부러 벗어나게 할 때: 상태 머신에 '옆으로 옮기고 돌린 받침대'를 목표로 준다 (상자는 그대로, 잡는 위치는 실제 상자 기준)
        aim_xy, aim_yaw = stand_xy.clone(), stand_yaw.clone()
        off_y, off_yaw = [0.0] * N, [0.0] * N
        if args_cli.stance_y:
            lo, hi = (float(v) for v in args_cli.stance_y.split(","))
            off_y = [rng.choice([-1, 1]) * rng.uniform(lo, hi) for _ in range(N)]
            left = torch.stack([-torch.sin(stand_yaw), torch.cos(stand_yaw)], dim=1)
            aim_xy = aim_xy + torch.tensor(off_y, device=device)[:, None] * left
        if args_cli.stance_yaw:
            lo, hi = (float(v) for v in args_cli.stance_yaw.split(","))
            off_yaw = [rng.choice([-1, 1]) * rng.uniform(lo, hi) for _ in range(N)]
            aim_yaw = aim_yaw + torch.deg2rad(torch.tensor(off_yaw, device=device))
        machine.start(box, aim_xy, aim_yaw, aim_xy, aim_yaw, obstacles=stand_xy[None])
        fell = torch.zeros(N, dtype=torch.bool, device=device)
        started = torch.zeros(N, dtype=torch.bool, device=device)
        stage = [""] * N

        if args_cli.mode == "script" and args_cli.two_box:
            for _ in range(2500):
                machine.pre_step()
                term = tick_machine()
                fell |= term
                seq.fail(term & seq.active)
                seq.update()
                tracker.update(seq.active)
                if bool(started[0]):
                    grab_head()
                new = (machine.state == ppm.S_DONE) & (seq.phase == seq.IDLE) & ~fell & ~started
                if bool(new.any()):
                    seq.begin(new)
                    tracker.begin(new)
                    started |= new
                if not bool(machine.active.any()) and not bool(seq.active.any()):
                    break
            ready = started
            success = tracker.success & ~fell
            for e in range(N):
                if not bool(success[e]):
                    stage[e] = "넘어짐" if bool(fell[e]) else f"팔 동작({seq.NAMES[int(seq.phase[e])]})"
        elif args_cli.mode == "script":
            for _ in range(2500):
                machine.pre_step()
                new = (machine.state == ppm.S_CLOSE) & ~started
                if bool(new.any()):
                    tracker.begin(new)
                    started |= new
                fell |= tick_machine()
                tracker.update(started & machine.active)
                if bool(started[0]):
                    grab_head()
                if not bool(machine.active.any()):
                    break
            ready = started
            success = tracker.success & ~fell
            for e in range(N):
                if not bool(success[e]):
                    stage[e] = "넘어짐" if bool(fell[e]) else (ppm.STATE_NAMES[int(machine.fail_state[e])] if int(machine.fail_state[e]) >= 0 else "기준 미달")
        else:
            for _ in range(2500):  # 받침대 앞까지 걷고 멈춤
                machine.pre_step()
                fell |= tick_machine()
                if not bool(machine.active.any()):
                    break
            ready = (machine.state == ppm.S_DONE) & ~fell
            executor.begin(ready)
            tracker.begin(ready)
            active = ready.clone()
            chunk = torch.zeros(N, 50, vp.ACTION_DIM, device=device)
            k_in_chunk = args_cli.n_action_steps
            for k in range(T_ARM):
                if k_in_chunk >= args_cli.n_action_steps:  # 새 행동 묶음을 받음 (진행 중인 환경만)
                    img, state = vp.read_obs(u, machine)
                    ids = active.nonzero().flatten()
                    if len(ids) == 0:
                        break
                    chunk[ids] = vla_infer(img[ids], state[ids], [tasks[i] for i in ids.tolist()])
                    k_in_chunk = 0
                action = chunk[:, k_in_chunk].clone()
                k_in_chunk += 1
                action[~active, :10] = machine.ik.q_cmd[~active]  # 끝난 환경은 지금 자세·잡기 상태를 유지
                action[~active, 10] = machine.attached[~active].float()
                executor.pre(action, active)
                term = tick_machine()
                grab_head()
                executor.post(action, active, term, other=other)
                tracker.update(active & ~term)
                # 성공한 환경은 --keep_steps 스텝을 더 실행한 뒤 멈춤 (영상에서 든 자세까지 보이게. 평가에서는 0)
                finished = tracker.success & (tracker.t - tracker.t_success >= args_cli.keep_steps)
                active &= ~(finished | term | executor.wrong_pick)
                if not bool(active.any()):
                    break
            success = ready & tracker.success & ~executor.fell & ~executor.wrong_pick
            for e in range(N):
                if bool(ready[e]) and not bool(success[e]):
                    stage[e] = vp.classify_failure(executor, tracker, e)
        for e in range(N):
            if len(records) >= args_cli.episodes:
                break
            if not bool(ready[e]):  # 받침대 앞에 서지 못한 환경은 평가에서 뺌 (걷기 쪽의 실패)
                continue
            rec = {"success": bool(success[e]), "stage": stage[e], "color": color, "task": tasks[e], "box_mass": round(masses[e], 3),
                   "time_s": round(float(tracker.t_success[e]) * step_dt, 2) if bool(success[e]) else None,
                   "max_lift": round(float(tracker.max_lift[e]), 4),
                   "stance_error": [round(float(machine.rec[kk][e]), 4) for kk in ("stance_a_x", "stance_a_y", "stance_a_yaw")],
                   "stance_offset": [round(off_y[e], 4), round(off_yaw[e], 2)]}
            if rounds is not None:
                rec["side"] = "left" if rounds[e]["side"] > 0 else "right"
                rec["other_color"] = rounds[e]["other"]
                rec["wrong_box"] = bool(executor.wrong_pick[e]) if args_cli.mode == "vla" else False
            if args_cli.mode == "vla":
                rec["min_hand_err"] = round(float(executor.min_err[e]), 4)
                rec["min_clearance"] = round(float(executor.min_clear[e]), 4) if float(executor.min_clear[e]) < 1.0e8 else None
            records.append(rec)
        ok = sum(r["success"] for r in records)
        print(f"[eval] 묶음 {n_round}: 색 {color}, 받침대 앞에 선 환경 {int(ready.sum())}/{N}, 누적 성공 {ok}/{len(records)}, {time.time() - t_start:.0f}초", flush=True)

if worker is not None:
    worker.stdin.write(json.dumps({"quit": True}) + "\n")
    worker.stdin.flush()
    worker.wait(timeout=30)

# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
n = len(records)
ok = [r for r in records if r["success"]]
stages = {}
for r in records:
    if not r["success"]:
        stages[r["stage"]] = stages.get(r["stage"], 0) + 1
times = [r["time_s"] for r in ok]
label = f"VLA (행동 묶음 중 {args_cli.n_action_steps}스텝 실행)" if args_cli.mode == "vla" else "스크립트 IK"
extra = (f", 서는 자리 옆으로 {args_cli.stance_y}m" if args_cli.stance_y else "") + (f", 비스듬히 {args_cli.stance_yaw}°" if args_cli.stance_yaw else "")
print(f"\n========== 뻗기 → 잡기 → 들기: {label}, 위치 ±{args_cli.pos_range * 100:.0f}cm·방향 ±{args_cli.yaw_range:.0f}°{extra}, 에피소드 {n}개 ==========")
print(f"성공: {len(ok)}/{n} ({len(ok) / n:.0%})")
print("실패 단계: " + (", ".join(f"{k} {v}" for k, v in sorted(stages.items(), key=lambda kv: -kv[1])) if stages else "없음"))
if times:
    print(f"성공까지 걸린 시간 (초): 평균 {np.mean(times):.2f}, 중앙값 {np.median(times):.2f}, 최대 {np.max(times):.2f}")
by_color = {}
for r in records:
    c = by_color.setdefault(r["color"], [0, 0])
    c[0] += r["success"]
    c[1] += 1
print("색별 성공: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in by_color.items()))
if args_cli.two_box:
    wrong = sum(r.get("wrong_box", False) for r in records)
    by_side = {}
    for r in records:
        c = by_side.setdefault(r["side"], [0, 0, 0])
        c[0] += r["success"]
        c[1] += 1
        c[2] += r.get("wrong_box", False)
    print(f"다른 상자를 집은 비율: {wrong}/{n} ({wrong / n:.0%})")
    print("집을 상자가 놓인 쪽별 (성공/전체, 다른 상자): " + ", ".join(f"{k} {v[0]}/{v[1]}, {v[2]}" for k, v in by_side.items()))
if args_cli.mode == "vla":
    lat = [v[0] for v in latencies]
    errs = [r["min_hand_err"] for r in records if r["min_hand_err"] < 1.0e8]
    clear = [r["min_clearance"] for r in records if r.get("min_clearance") is not None]
    print(f"VLA 응답 시간 (초, 묶음 한 번): 평균 {np.mean(lat):.3f}, 최대 {np.max(lat):.3f}, 요청 {len(lat)}번, 요청당 환경 수 평균 {np.mean([v[1] for v in latencies]):.1f}")
    print(f"붙이기 전 손이 잡는 위치에 가장 가까웠던 거리 (cm): 평균 {np.mean(errs) * 100:.2f}, 최대 {np.max(errs) * 100:.2f}")
    if clear:
        print(f"들고 있는 동안 손과 상자 옆면 사이의 최소 틈 (cm, 음수 = 파고듦): 평균 {np.mean(clear) * 100:.2f}, 최소 {np.min(clear) * 100:.2f}")
print(f"전체 걸린 시간 {time.time() - t_start:.0f}초 (묶음 {n_round}개)")
if args_cli.out:
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    summary = {"mode": args_cli.mode, "two_box": args_cli.two_box, "n_action_steps": args_cli.n_action_steps, "pos_range": args_cli.pos_range, "yaw_range": args_cli.yaw_range,
               "stance_y": args_cli.stance_y, "stance_yaw": args_cli.stance_yaw, "episodes": n, "success": len(ok), "stages": stages, "time_mean_s": float(np.mean(times)) if times else None,
               "seed": args_cli.seed, "records": records}
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"결과 저장: {args_cli.out}")

if args_cli.video:
    import imageio.v3 as iio  # noqa: E402

    info = {"name": args_cli.name, "mode": args_cli.mode, "label": label, "task": records[0]["task"], "success": records[0]["success"],
            "fps": int(round(1 / step_dt)), "arm_start_step": arm_span[0], "arm_end_step": arm_span[1], "total_steps": n_steps}
    with open(os.path.join(args_cli.video_dir, f"{args_cli.name}_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    if head_frames:
        iio.imwrite(os.path.join(args_cli.video_dir, f"{args_cli.name}_head.mp4"), np.stack(head_frames), fps=info["fps"])
    print(f"[eval] 영상: {os.path.join(args_cli.video_dir, args_cli.name + '-step-0.mp4')} (팔 동작 구간 스텝 {arm_span[0]}~{arm_span[1]})")

env.close()
simulation_app.close()
