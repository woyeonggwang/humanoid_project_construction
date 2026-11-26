"""VLA 학습용 내려놓기 시범 수집: 작은 상자를 든 로봇이 받침대 앞에 선 뒤, 명령한 색의 판 위에 내려놓는 구간을 기록한다.

- 한 에피소드: (기록 안 함) 팔을 든 자세로 → 상자를 손에 붙임 → 받침대 앞까지 걷고 멈춤
              (기록) 자세 잡기 → 판 위로 → 내리기 → 놓기 → 손 벌리기 → 팔 올리기 → 0.5초 유지   (humanoid_tasks/g1_carry/vla_place.py 의 SidePlaceScript)
- 환경을 둘씩 쌍으로 묶는다: 한 쌍은 장면(받침대·두 판의 색과 자리·든 상자·서는 자리)이 같고 놓을 판만 다르다. 쌍이 둘 다 성공한 것만 저장한다
- --n_single: 한 묶음의 환경 중 이만큼은 '다른 판에 상자가 이미 놓인 장면' (놓을 판은 빈 판. 하나씩 저장하고, 놓여 있던 상자를 1cm 넘게 밀면 버림)
- 성공 기준은 평가와 같다 (vla_place.PlaceTracker): 상자가 놓을 판 가운데에서 5cm 안에 놓이고 손이 떨어진 채 0.5초 유지, 넘어지지 않음
- 저장 형식은 집기 시범과 같다: <out>/ep_000000.npz (images, state (T, 11), action (T, 11), meta). meta 의 "skill" 이 "place"

예) python -u scripts\\collect_vla_place.py --headless --episodes 440 --num_envs 24 --n_single 6 --stance_y 0,0.10 --stance_yaw 0,10 --out data\\vla_place\\raw_v1 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect scripted place-on-pad demonstrations for VLA fine-tuning.")
parser.add_argument("--out", type=str, required=True, help="원자료를 저장할 폴더")
parser.add_argument("--episodes", type=int, default=16, help="모을 성공 에피소드 수")
parser.add_argument("--num_envs", type=int, default=24)
parser.add_argument("--n_single", type=int, default=0, help="한 묶음의 환경 중 '다른 판에 상자가 이미 놓인 장면' 의 수")
parser.add_argument("--pos_range", type=float, default=0.10, help="받침대 위치 무작위 범위 (m)")
parser.add_argument("--yaw_range", type=float, default=15.0, help="받침대 방향 무작위 범위 (도)")
parser.add_argument("--mass_range", type=str, default="2.0,3.0", help="상자 무게 범위 kg")
parser.add_argument("--stance_y", type=str, default=None, help="서는 자리를 받침대 정면에서 옆으로 벗어나게 할 크기 범위 '최소,최대' m (부호는 무작위)")
parser.add_argument("--stance_yaw", type=str, default=None, help="받침대를 비스듬히 보게 할 크기 범위 '최소,최대' 도 (부호는 무작위)")
parser.add_argument("--hold_steps", type=int, default=25, help="팔을 올린 뒤 몇 스텝까지 기록할지")
parser.add_argument("--task", type=str, default="Humanoid-G1-VLA-Place-v0")
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--seed", type=int, default=42)
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
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

N = args_cli.num_envs
assert (N - args_cli.n_single) % 2 == 0, "쌍으로 뽑는 환경 수(num_envs - n_single)가 짝수여야 합니다"
mass_range = tuple(float(v) for v in args_cli.mass_range.split(","))
stance_y = tuple(float(v) for v in args_cli.stance_y.split(",")) if args_cli.stance_y else None
stance_yaw = tuple(float(v) for v in args_cli.stance_yaw.split(",")) if args_cli.stance_yaw else None
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
device = u.device
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
print(f"[collect] 걷기 정책: {os.path.join(os.path.basename(os.path.dirname(resume_path)), os.path.basename(resume_path))}")

machine = ppm.PickPlaceMachine(u, until="stand")
seq = vpl.SidePlaceScript(machine)
tracker = vpl.PlaceTracker(machine)
cmd_term = u.command_manager.get_term("base_velocity")
rng = random.Random(args_cli.seed)
out_dir = Path(args_cli.out)
out_dir.mkdir(parents=True, exist_ok=True)
MAX_STEPS = 2500
OCC_MOVE_OK = 0.01  # 놓여 있던 상자가 이만큼(m) 넘게 밀리면 그 에피소드는 버림
REC = (seq.PREP, seq.OVER, seq.DOWN, seq.ADJUST, seq.RELEASE, seq.OPEN, seq.RETRACT)
obs = None


def tick():
    """한 스텝. 돌려주는 값: 넘어짐 여부 (N,)"""
    global obs
    obs, _, _, _ = env.step(policy(obs))
    term = u.termination_manager.terminated.clone()
    cmd_term.vel_command_b[:] = machine.post_step(term)
    obs = env.get_observations()
    return term


with torch.inference_mode():
    saved = len(list(out_dir.glob("ep_*.npz")))
    target = saved + args_cli.episodes
    n_round, n_try, n_ok, n_pair_ok, t0 = 0, 0, 0, 0, time.time()
    fail_counts, place_errs, clear_h, clear_o, occ_moves = {}, [], [], [], []
    count_color, count_side, count_kind, deck = {}, {}, {}, []
    all_envs = torch.ones(N, dtype=torch.bool, device=device)
    while saved < target:
        env.reset()
        machine.reset()
        seq.reset()
        rounds = vpl.sample_place_round(rng, N, args_cli.pos_range, args_cli.yaw_range, mass_range, stance_y, stance_yaw, deck,
                                        n_single=args_cli.n_single)
        scene = vpl.setup_place_round(u, machine, rounds)
        vp.random_lighting(rng)
        cmd_term.vel_command_b[:] = 0.0
        obs = env.get_observations()
        fell = torch.zeros(N, dtype=torch.bool, device=device)
        for _ in range(25):  # 자리 잡기
            machine.pre_step()
            tick()
        machine._start_move(all_envs, vpl.carry_goal(machine), 60)  # 팔을 든 자세로
        for _ in range(75):
            machine.pre_step()
            fell |= tick()
        vpl.give_box(machine, scene.held)  # 양손 사이에 상자를 놓고 붙임
        machine.arm_hold[:] = False  # 걷는 동안은 팔 관절 목표 고정
        for _ in range(10):
            machine.pre_step()
            fell |= tick()
        occ_start = vpl.occupied_boxes(u, rounds)[0].clone()
        aim_xy, aim_yaw = vp.stance_aim(scene.stand_xy, scene.stand_yaw, [r["stance"][0] for r in rounds], [r["stance"][1] for r in rounds])
        machine.start_walk(aim_xy, aim_yaw, obstacles=scene.stand_xy[None], keep_box=True)
        tasks = vpl.place_sentences(rng, rounds)
        buf = [{"images": [], "state": [], "action": []} for _ in range(N)]
        started = torch.zeros(N, dtype=torch.bool, device=device)
        min_clear_h = torch.full((N,), 1.0e9, device=device)
        min_clear_o = torch.full((N,), 1.0e9, device=device)
        for step in range(MAX_STEPS):
            machine.pre_step()
            ph = seq.phase.clone()
            rec = (ph == seq.HOLD) & (seq.t < args_cli.hold_steps)
            for p_ in REC:
                rec |= ph == p_
            img, state = vp.read_obs(u, machine)  # 스텝 직전의 사진·상태
            q_cmd = machine.ik.q_cmd.clone()
            term = tick()
            fell |= term
            seq.fail(term & seq.active)
            seq.update()
            grasp = machine.attached.float()  # 이 스텝이 끝났을 때의 잡기 신호
            tracker.update(started & ~fell)
            if step % 5 == 0 and bool(seq.active.any()):
                act = seq.active
                o_pos, o_yaw, o_has = vpl.occupied_boxes(u, rounds)
                min_clear_o = torch.where(act & o_has, torch.minimum(min_clear_o, machine.hand_box_clearance(o_pos, o_yaw)), min_clear_o)
                held = act & machine.attached
                min_clear_h = torch.where(held, torch.minimum(min_clear_h, machine.hand_box_clearance()), min_clear_h)
            if bool(rec.any()):
                img_np, state_np = img.cpu().numpy(), state.cpu().numpy()
                act_np = torch.cat([q_cmd, grasp[:, None]], dim=1).cpu().numpy()
                for e in rec.nonzero().flatten().tolist():
                    buf[e]["images"].append(img_np[e].copy())
                    buf[e]["state"].append(state_np[e])
                    buf[e]["action"].append(act_np[e])
            ready = (machine.state == ppm.S_DONE) & (seq.phase == seq.IDLE) & ~fell & ~started
            if bool(ready.any()):  # 받침대 앞에 선 환경은 다음 스텝부터 팔을 움직임
                seq.begin(ready, scene)
                tracker.begin(ready, scene)
                started |= ready
            if not bool(machine.active.any()) and not bool(seq.active.any()):
                break
        n_round += 1
        occ_pos, _, occ_has = vpl.occupied_boxes(u, rounds)
        occ_moved = torch.where(occ_has, torch.norm(occ_pos - occ_start, dim=1), torch.zeros(N, device=device))
        ok = (seq.phase == seq.DONE) & tracker.success & ~fell & (occ_moved <= OCC_MOVE_OK)
        n_try += N
        n_ok += int(ok.sum())
        outcome = [tracker.outcome(e, bool(fell[e])) for e in range(N)]
        for e in range(N):
            if not bool(ok[e]):
                if not bool(started[e]):
                    key = "서지 못함"
                elif bool(fell[e]):
                    key = "넘어짐"
                elif float(occ_moved[e]) > OCC_MOVE_OK:
                    key = "놓여 있던 상자를 밂"
                elif outcome[e][0] == "성공":
                    key = "팔 동작을 끝내지 못함"
                else:
                    key = outcome[e][0]
                fail_counts[key] = fail_counts.get(key, 0) + 1
        groups = [[e for e in range(N) if rounds[e]["pair"] == k] for k in range((N - args_cli.n_single) // 2)]
        groups += [[e] for e in range(N) if rounds[e]["pair"] is None]
        for group in groups:
            if not all(bool(ok[e]) for e in group):
                continue
            n_pair_ok += len(group) == 2
            for e in group:
                place_errs.append(outcome[e][1])
                clear_h.append(float(min_clear_h[e]))
                if bool(occ_has[e]):
                    clear_o.append(float(min_clear_o[e]))
                    occ_moves.append(float(occ_moved[e]))
            if saved >= target:
                continue
            for e in group:
                r = rounds[e]
                side_name = "left" if r["side"] > 0 else "right"
                meta = {
                    "task": tasks[e], "skill": "place", "color": r["color"], "side": side_name, "other_color": r["other"],
                    "held_color": r["held"], "occupied": r["occupied"],
                    "pair": f"{args_cli.seed}-{n_round}-{r['pair']}" if r["pair"] is not None else None,
                    "stance_offset": [round(r["stance"][0], 4), round(r["stance"][1], 2)],
                    "box_mass": round(r["mass"], 3), "fps": vp.CONTROL_HZ, "stand_offset": [round(v, 4) for v in r["stand"]],
                    "stance_error": [round(float(machine.rec[kk][e]), 4) for kk in ("stance_b_x", "stance_b_y", "stance_b_yaw")],
                    "place_err": round(outcome[e][1], 4), "release_err": round(float(tracker.release_err[e]), 4),
                    "min_clearance_held": round(float(min_clear_h[e]), 4),
                    "min_clearance_occupied": round(float(min_clear_o[e]), 4) if bool(occ_has[e]) else None,
                    "occupied_moved": round(float(occ_moved[e]), 4) if bool(occ_has[e]) else None,
                    "frames": len(buf[e]["images"]), "seed": args_cli.seed, "round": n_round, "env": e,
                }
                np.savez_compressed(out_dir / f"ep_{saved:06d}.npz", images=np.stack(buf[e]["images"]),
                                    state=np.stack(buf[e]["state"]).astype(np.float32),
                                    action=np.stack(buf[e]["action"]).astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))
                saved += 1
                kind = "쌍" if r["pair"] is not None else "다른 판 참"
                count_color[r["color"]] = count_color.get(r["color"], 0) + 1
                count_side[side_name] = count_side.get(side_name, 0) + 1
                count_kind[kind] = count_kind.get(kind, 0) + 1
        lens = [len(b["images"]) for b in buf if b["images"]]
        print(f"[collect] 묶음 {n_round}: 성공 {int(ok.sum())}/{N}, 둘 다 성공한 쌍 누적 {n_pair_ok}/{n_round * (N - args_cli.n_single) // 2}, 저장 {saved}/{target}, "
              f"에피소드 길이 {min(lens) if lens else 0}~{max(lens) if lens else 0} 스텝, {time.time() - t0:.0f}초", flush=True)
    print(f"\n========== 시범 수집 (내려놓기): 저장 {saved}개 ({out_dir}), 시도 {n_try}개 중 성공 {n_ok}개, "
          f"둘 다 성공한 쌍 {n_pair_ok}/{n_round * (N - args_cli.n_single) // 2}, 묶음 {n_round}개, {time.time() - t0:.0f}초 ==========")
    print("이번에 저장한 것 — 놓을 판 색: " + ", ".join(f"{k} {v}" for k, v in sorted(count_color.items()))
          + " | 자리: " + ", ".join(f"{k} {v}" for k, v in sorted(count_side.items()))
          + " | 종류: " + ", ".join(f"{k} {v}" for k, v in sorted(count_kind.items())))
    if fail_counts:
        print("실패: " + ", ".join(f"{k} {v}" for k, v in fail_counts.items()))
    if place_errs:
        print(f"놓은 위치 오차 (cm, 판 가운데까지): 평균 {np.mean(place_errs) * 100:.2f}, 최대 {np.max(place_errs) * 100:.2f}")
        clear_h = [v for v in clear_h if v < 1.0e8]
        print(f"손과 든 상자 옆면 사이의 최소 틈 (cm, 음수 = 파고듦): 평균 {np.mean(clear_h) * 100:.2f}, 최소 {np.min(clear_h) * 100:.2f}")
    clear_o = [v for v in clear_o if v < 1.0e8]
    if clear_o:
        print(f"손과 놓여 있던 상자 사이의 최소 틈 (cm): 평균 {np.mean(clear_o) * 100:.2f}, 최소 {np.min(clear_o) * 100:.2f} | "
              f"놓여 있던 상자가 밀린 거리 (cm): 평균 {np.mean(occ_moves) * 100:.2f}, 최대 {np.max(occ_moves) * 100:.2f}")

env.close()
simulation_app.close()
