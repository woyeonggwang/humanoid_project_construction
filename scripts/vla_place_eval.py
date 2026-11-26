"""든 상자를 명령한 색의 판 위에 내려놓기를 VLA(SmolVLA) 또는 스크립트 IK 로 실행하고 평가한다.

- 팔을 든 자세로 → 상자를 손에 붙임 → 받침대 앞까지 걷고 멈춤: 두 방식 모두 스크립트가 한다. 그 뒤의 팔 동작과 잡기 신호(놓기)만 다르다
  - --mode vla: VLA 작업자에 사진·상태·명령을 보내 행동 묶음(50스텝)을 받고, 앞의 --n_action_steps 스텝을 실행한 뒤 다시 요청한다
  - --mode script: 수집 때와 같은 스크립트 IK (vla_place.SidePlaceScript, 비교 기준)
- 환경을 둘씩 쌍으로 뽑는다 (같은 장면, 놓을 판만 다름 = 명령의 색만 다름). --n_single 만큼은 다른 판에 상자가 이미 놓인 장면
- 성공 (두 방식 공통, vla_place.PlaceTracker): 손을 뗀 상자가 놓을 판 가운데에서 5cm 안에 바로 놓이고 손이 떨어진 채 0.5초 유지. 제한 시간 --max_time 초
- 결과 이름: 성공 / 엉뚱한 판에 놓음 / 판에서 벗어나게 놓음 / 상자가 떨어지거나 넘어짐 / 놓지 않음 / 손을 떼지 않음 / 놓았다가 다시 잡음 / 넘어짐

VLA 를 띄울 파이썬은 --vla_python 또는 환경 변수 VLA_PYTHON 으로 준다.
예) python -u scripts\\vla_place_eval.py --headless --mode vla --ckpt <체크포인트>/pretrained_model --episodes 96 --num_envs 24 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate place-on-pad with a VLA policy or the scripted IK.")
parser.add_argument("--mode", choices=["vla", "script"], default="vla")
parser.add_argument("--ckpt", type=str, default=None, help="VLA 체크포인트의 pretrained_model 폴더")
parser.add_argument("--vla_python", type=str, default=os.environ.get("VLA_PYTHON"), help="VLA 작업자를 띄울 파이썬 실행 파일")
parser.add_argument("--n_action_steps", type=int, default=50, help="행동 묶음 50스텝 중 몇 스텝을 실행하고 다시 요청할지")
parser.add_argument("--episodes", type=int, default=96)
parser.add_argument("--num_envs", type=int, default=24)
parser.add_argument("--n_single", type=int, default=0, help="한 묶음의 환경 중 '다른 판에 상자가 이미 놓인 장면' 의 수")
parser.add_argument("--pos_range", type=float, default=0.10)
parser.add_argument("--yaw_range", type=float, default=15.0)
parser.add_argument("--mass_range", type=str, default="2.0,3.0")
parser.add_argument("--stance_y", type=str, default=None, help="서는 자리를 받침대 정면에서 옆으로 벗어나게: 크기 범위 '최소,최대' m (부호는 무작위)")
parser.add_argument("--stance_yaw", type=str, default=None, help="받침대를 비스듬히 보게: 크기 범위 '최소,최대' 도 (부호는 무작위)")
parser.add_argument("--max_time", type=float, default=12.0, help="팔 동작 제한 시간 (초)")
parser.add_argument("--keep_steps", type=int, default=0, help="VLA: 성공한 뒤에도 이 스텝 수만큼 더 실행 (녹화용)")
parser.add_argument("--pads", type=str, default=None, help="녹화용: 판의 색을 '왼쪽,오른쪽' 으로 고정 (예: red,blue)")
parser.add_argument("--side", choices=["left", "right"], default=None, help="녹화용: 놓을 판의 자리를 고정")
parser.add_argument("--held_color", type=str, default=None, help="녹화용: 든 상자의 색을 고정")
parser.add_argument("--task", type=str, default="Humanoid-G1-VLA-Place-v0")
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--out", type=str, default=None, help="결과 JSON 을 저장할 파일")
parser.add_argument("--video", action="store_true", help="환경 0 을 녹화 (num_envs 2 권장: 쌍 하나)")
parser.add_argument("--view", choices=["side", "front"], default="side")
parser.add_argument("--video_dir", type=str, default=os.path.join("data", "vla_place", "videos"))
parser.add_argument("--name", type=str, default="vla_place")
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
from humanoid_tasks.g1_carry import vla_place as vpl  # noqa: E402
from humanoid_tasks.g1_carry.vla_worker import VlaWorker  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

N = args_cli.num_envs
if (N - args_cli.n_single) % 2:
    raise SystemExit("쌍으로 뽑는 환경 수(num_envs - n_single)가 짝수여야 합니다")
mass_range = tuple(float(v) for v in args_cli.mass_range.split(","))
stance_y = tuple(float(v) for v in args_cli.stance_y.split(",")) if args_cli.stance_y else None
stance_yaw = tuple(float(v) for v in args_cli.stance_yaw.split(",")) if args_cli.stance_yaw else None
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
if args_cli.video:
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.15, -2.3, 0.45) if args_cli.view == "side" else (2.2, -1.2, 0.85)
    env_cfg.viewer.lookat = (0.15, 0.0, 0.10)
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

machine = ppm.PickPlaceMachine(u, until="stand")
seq = vpl.SidePlaceScript(machine)
tracker = vpl.PlaceTracker(machine)
executor = vp.VlaArmExecutor(machine)
cmd_term = u.command_manager.get_term("base_velocity")
rng = random.Random(args_cli.seed)
deck = []
GIVE_UP_STEPS = 150  # 손을 뗀 뒤 이만큼(3초) 지나도 성공하지 못하면 그 환경은 멈춤

worker = None
if args_cli.mode == "vla":
    if not args_cli.vla_python or not args_cli.ckpt:
        raise SystemExit("--mode vla 에는 --ckpt 와 --vla_python(또는 환경 변수 VLA_PYTHON)이 필요합니다")
    worker = VlaWorker(args_cli.vla_python, args_cli.ckpt)
    print(f"[eval] VLA 작업자 준비: 불러오기 {worker.info['load_s']}초, GPU {worker.info['gpu_mb']}MB, 행동 묶음 {worker.info['chunk_size']}스텝")

obs = None
n_steps = 0  # env.step 을 부른 횟수 (녹화한 영상의 프레임 번호와 맞추려고)
head_frames, arm_span = [], [None, None]


def grab_head():
    if args_cli.video:
        if arm_span[0] is None:
            arm_span[0] = n_steps
        arm_span[1] = n_steps
        head_frames.append(u.scene.sensors["vla_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().copy())


def tick():
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
all_envs = torch.ones(N, dtype=torch.bool, device=device)
with torch.inference_mode():
    while len(records) < args_cli.episodes:
        n_round += 1
        env.reset()
        machine.reset()
        seq.reset()
        rounds = vpl.sample_place_round(rng, N, args_cli.pos_range, args_cli.yaw_range, mass_range, stance_y, stance_yaw, deck,
                                        n_single=args_cli.n_single)
        for r in rounds:  # 녹화용: 난수를 뽑은 뒤에 덮어써서, 같은 시드면 받침대 위치가 같게 한다
            if args_cli.pads:
                left, right = args_cli.pads.split(",")
                r["pads"] = {1: left, -1: right}
            if args_cli.side:
                r["side"] = 1 if args_cli.side == "left" else -1
            if args_cli.held_color:
                r["held"] = args_cli.held_color
            r["color"], r["other"] = r["pads"][r["side"]], r["pads"][-r["side"]]
        scene = vpl.setup_place_round(u, machine, rounds)
        tasks = vpl.place_sentences(rng, rounds)
        vp.random_lighting(rng)
        cmd_term.vel_command_b[:] = 0.0
        obs = env.get_observations()
        fell = torch.zeros(N, dtype=torch.bool, device=device)
        for _ in range(25):
            machine.pre_step()
            tick()
        machine._start_move(all_envs, vpl.carry_goal(machine), 60)  # 팔을 든 자세로
        for _ in range(75):
            machine.pre_step()
            fell |= tick()
        vpl.give_box(machine, scene.held)
        machine.arm_hold[:] = False
        for _ in range(10):
            machine.pre_step()
            fell |= tick()
        occ_start = vpl.occupied_boxes(u, rounds)[0].clone()
        aim_xy, aim_yaw = vp.stance_aim(scene.stand_xy, scene.stand_yaw, [r["stance"][0] for r in rounds], [r["stance"][1] for r in rounds])
        machine.start_walk(aim_xy, aim_yaw, obstacles=scene.stand_xy[None], keep_box=True)
        for _ in range(2500):  # 받침대 앞까지 걷고 멈춤
            machine.pre_step()
            fell |= tick()
            if not bool(machine.active.any()):
                break
        ready = (machine.state == ppm.S_DONE) & ~fell
        tracker.begin(ready, scene)

        if args_cli.mode == "script":
            seq.begin(ready, scene)
            for _ in range(T_ARM):
                machine.pre_step()
                term = tick()
                grab_head()
                fell |= term
                seq.fail(term & seq.active)
                seq.update()
                tracker.update(ready & ~fell)
                if not bool(seq.active.any()):
                    break
        else:
            executor.begin(ready)
            active = ready.clone()
            chunk = torch.zeros(N, 50, vp.ACTION_DIM, device=device)
            k_in_chunk = args_cli.n_action_steps
            for k in range(T_ARM):
                if k_in_chunk >= args_cli.n_action_steps:  # 새 행동 묶음을 받음 (진행 중인 환경만)
                    img, state = vp.read_obs(u, machine)
                    ids = active.nonzero().flatten()
                    if len(ids) == 0:
                        break
                    out = worker.infer(img[ids].cpu().numpy(), state[ids].cpu().numpy(), [tasks[i] for i in ids.tolist()])
                    chunk[ids] = torch.from_numpy(out).to(device)
                    k_in_chunk = 0
                action = chunk[:, k_in_chunk].clone()
                k_in_chunk += 1
                action[~active, :10] = machine.ik.q_cmd[~active]  # 끝난 환경은 지금 자세·잡기 상태를 유지
                action[~active, 10] = machine.attached[~active].float()
                executor.pre(action, active)
                term = tick()
                grab_head()
                fell |= term
                executor.post(action, active, term)
                tracker.update(active & ~term)
                finished = tracker.success & (tracker.t - tracker.t_success >= args_cli.keep_steps)
                gave_up = tracker.released & ~tracker.success & (tracker.t - tracker.t_release > GIVE_UP_STEPS)
                active &= ~(finished | term | gave_up)
                if not bool(active.any()):
                    break
        occ_pos, _, occ_has = vpl.occupied_boxes(u, rounds)
        occ_moved = torch.norm(occ_pos - occ_start, dim=1)
        box_xy = machine.box.data.root_pos_w[:, :2]
        d_target, d_other = torch.norm(box_xy - scene.target_xy, dim=1), torch.norm(box_xy - scene.other_xy, dim=1)
        for e in range(N):
            if len(records) >= args_cli.episodes:
                break
            if not bool(ready[e]):  # 받침대 앞에 서지 못한 환경은 평가에서 뺌 (걷기 쪽의 실패)
                continue
            name, err = tracker.outcome(e, bool(fell[e]))
            r = rounds[e]
            chosen = None  # 어느 판에 놓았는지 (손을 뗀 상자가 그 판 가운데 가까이에 있음)
            if bool(tracker.released[e]):
                chosen = "target" if float(d_target[e]) <= vpl.WRONG_PAD_XY else ("other" if float(d_other[e]) <= vpl.WRONG_PAD_XY else None)
            records.append({
                "success": name == "성공", "outcome": name, "place_err": None if err != err else round(err, 4), "chosen": chosen,
                "color": r["color"], "other_color": r["other"], "side": "left" if r["side"] > 0 else "right", "held_color": r["held"],
                "occupied": r["occupied"], "pair": f"{n_round}-{r['pair']}" if r["pair"] is not None else None, "task": tasks[e],
                "time_s": round(float(tracker.t_success[e]) * step_dt, 2) if name == "성공" else None,
                "occupied_moved": round(float(occ_moved[e]), 4) if bool(occ_has[e]) else None,
                "stance_error": [round(float(machine.rec[kk][e]), 4) for kk in ("stance_b_x", "stance_b_y", "stance_b_yaw")],
                "stance_offset": [round(r["stance"][0], 4), round(r["stance"][1], 2)],
            })
        ok = sum(r["success"] for r in records)
        print(f"[eval] 묶음 {n_round}: 받침대 앞에 선 환경 {int(ready.sum())}/{N}, 누적 성공 {ok}/{len(records)}, {time.time() - t_start:.0f}초", flush=True)

if worker is not None:
    worker.close()

# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
n = len(records)
ok = [r for r in records if r["success"]]
outcomes = {}
for r in records:
    outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
label = f"VLA (행동 묶음 중 {args_cli.n_action_steps}스텝 실행)" if args_cli.mode == "vla" else "스크립트 IK"
extra = (f", 서는 자리 옆으로 {args_cli.stance_y}m" if args_cli.stance_y else "") + (f", 비스듬히 {args_cli.stance_yaw}°" if args_cli.stance_yaw else "")
print(f"\n========== 판 위에 내려놓기: {label}, 위치 ±{args_cli.pos_range * 100:.0f}cm·방향 ±{args_cli.yaw_range:.0f}°{extra}, 에피소드 {n}개 ==========")
print(f"성공: {len(ok)}/{n} ({len(ok) / n:.0%})")
print("결과: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1])))
wrong = outcomes.get("엉뚱한 판에 놓음", 0)
fallen = outcomes.get("상자가 떨어지거나 넘어짐", 0)
print(f"엉뚱한 판에 놓은 비율: {wrong}/{n} ({wrong / n:.0%}) | 상자가 떨어지거나 넘어진 비율: {fallen}/{n} ({fallen / n:.0%})")
errs = [r["place_err"] for r in ok]
times = [r["time_s"] for r in ok]
if ok:
    print(f"놓은 위치 오차 (cm, 성공한 것): 평균 {np.mean(errs) * 100:.2f}, 최대 {np.max(errs) * 100:.2f} | 성공까지 걸린 시간 (초): 평균 {np.mean(times):.2f}, 최대 {np.max(times):.2f}")


def tally(key):
    out = {}
    for r in records:
        c = out.setdefault(key(r), [0, 0])
        c[0] += r["success"]
        c[1] += 1
    return ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in out.items())


print("놓을 판 색별 성공: " + tally(lambda r: r["color"]))
print("놓을 판 자리별 성공: " + tally(lambda r: r["side"]) + " | 종류별: " + tally(lambda r: "다른 판 참" if r["pair"] is None else "쌍"))
moved = [r["occupied_moved"] for r in records if r["occupied_moved"] is not None]
if moved:
    print(f"다른 판에 놓여 있던 상자가 밀린 거리 (cm): 평균 {np.mean(moved) * 100:.2f}, 최대 {np.max(moved) * 100:.2f}, 1cm 넘게 밀린 것 {sum(v > 0.01 for v in moved)}/{len(moved)}")
by_pair = {}
for r in records:
    if r["pair"] is not None:
        by_pair.setdefault(r["pair"], []).append(r)
full = [v for v in by_pair.values() if len(v) == 2]
pair_summary = None
if full:
    both_ok = sum(all(r["success"] for r in v) for v in full)
    chose = [v for v in full if all(r["chosen"] for r in v)]
    follow = sum(all(r["chosen"] == "target" for r in v) for v in chose)
    same = sum(len({r["chosen"] for r in v}) == 2 for v in chose)  # 하나는 놓을 판, 하나는 다른 판 = 실제로는 같은 판
    opposite = sum(all(r["chosen"] == "other" for r in v) for v in chose)
    pair_summary = {"pairs": len(full), "both_success": both_ok, "both_chose": len(chose), "follow_command": follow, "same_pad": same, "both_wrong": opposite}
    print(f"명령 바꿔치기 (같은 장면에 명령의 색만 다른 쌍 {len(full)}개): 둘 다 성공 {both_ok}/{len(full)}")
    if chose:
        print(f"  두 환경 모두 어느 판에든 놓은 쌍 {len(chose)}개 중: 명령대로 서로 다른 판 {follow} ({follow / len(chose):.0%}), "
              f"명령이 달라도 같은 판 {same} ({same / len(chose):.0%}), 둘 다 반대 판 {opposite}")
if worker is not None:
    lat = [v[0] for v in worker.latencies]
    print(f"VLA 응답 시간 (초, 묶음 한 번): 평균 {np.mean(lat):.3f}, 최대 {np.max(lat):.3f}, 요청 {len(lat)}번, 요청당 환경 수 평균 {np.mean([v[1] for v in worker.latencies]):.1f}")
print(f"전체 걸린 시간 {time.time() - t_start:.0f}초 (묶음 {n_round}개)")
if args_cli.out:
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    summary = {"mode": args_cli.mode, "n_action_steps": args_cli.n_action_steps, "pos_range": args_cli.pos_range, "yaw_range": args_cli.yaw_range,
               "stance_y": args_cli.stance_y, "stance_yaw": args_cli.stance_yaw, "n_single": args_cli.n_single, "episodes": n, "success": len(ok),
               "outcomes": outcomes, "pairs": pair_summary, "seed": args_cli.seed, "records": records}
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
