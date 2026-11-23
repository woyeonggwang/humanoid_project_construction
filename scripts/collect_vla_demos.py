"""VLA 학습용 시범 데이터 수집: 상자 집어 옮기기 스크립트(상태 머신 + IK)를 돌리며 '뻗기 → 붙이기 → 들기' 구간을 기록한다.

- 걷기(받침대 앞까지 가서 멈춤)는 스크립트가 하고 기록하지 않는다. 기록은 손을 뻗기 시작하는 스텝부터, 든 자세에서 1초 머문 데까지
- 성공한 에피소드만 저장한다 (상자가 붙은 채 5cm 넘게 들려 0.5초 유지, 넘어지지 않음)
- 무작위화: 받침대·상자 위치 ±10cm·방향 ±15°, 상자 무게 2~3kg, 상자 색(묶음마다), 조명(묶음마다), 명령 문장
- 저장: <out>/ep_000000.npz (images (T, 256, 256, 3) uint8, state (T, 11), action (T, 11), meta = JSON 문자열)
  관측·행동의 뜻과 시점은 humanoid_tasks/g1_carry/vla_pick.py 설명 참고
- --sync_check: 사진이 그 스텝의 상태를 보여 주는지 확인 (상자를 순간 이동시킨 스텝의 앞뒤 사진을 비교)

예) python -u scripts\\collect_vla_demos.py --headless --episodes 200 --num_envs 16 --out data\\vla_pick\\raw_v1 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect scripted reach-grasp-lift demonstrations for VLA fine-tuning.")
parser.add_argument("--out", type=str, required=True, help="원자료를 저장할 폴더")
parser.add_argument("--episodes", type=int, default=10, help="모을 성공 에피소드 수")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--pos_range", type=float, default=0.10, help="받침대·상자 위치 무작위 범위 (m)")
parser.add_argument("--yaw_range", type=float, default=15.0, help="받침대·상자 방향 무작위 범위 (도)")
parser.add_argument("--mass_range", type=str, default="2.0,3.0", help="상자 무게 범위 kg")
parser.add_argument("--hold_steps", type=int, default=50, help="든 자세에서 몇 스텝까지 기록할지")
parser.add_argument("--task", type=str, default="Humanoid-G1-VLA-Pick-v0")
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--color", type=str, default=None, help="집을 상자의 색을 하나로 고정 (기본: 묶음마다 무작위)")
parser.add_argument("--sync_check", action="store_true", help="사진과 상태의 시점이 맞는지 확인하고 끝")
parser.add_argument("--two_box", action="store_true",
                    help="2차: 넓은 받침대에 색이 다른 작은 상자 두 개, 명령한 색의 상자를 집음 (과제 Humanoid-G1-VLA-Pick2-v0)")
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

N = args_cli.num_envs
if args_cli.two_box and args_cli.task == "Humanoid-G1-VLA-Pick-v0":
    args_cli.task = "Humanoid-G1-VLA-Pick2-v0"
mass_range = tuple(float(v) for v in args_cli.mass_range.split(","))
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

machine = ppm.PickPlaceMachine(u, until="stand" if args_cli.two_box else "lift")
tracker = vp.LiftTracker(machine)
seq = vp.SidePickScript(machine)  # 2차에서만 씀
cmd_term = u.command_manager.get_term("base_velocity")
rng = random.Random(args_cli.seed)
out_dir = Path(args_cli.out)
out_dir.mkdir(parents=True, exist_ok=True)
REC_STATES = (ppm.S_CLOSE, ppm.S_CLOSE2, ppm.S_LIFT1, ppm.S_LIFT2)
MAX_STEPS = 2500  # 한 묶음의 최대 스텝 (걷기 + 집기)

obs = None


def tick():
    """한 스텝: 상태 머신이 팔과 걷기 명령을 정한다. 돌려주는 값: 넘어짐 여부 (N,)"""
    global obs
    obs, _, _, _ = env.step(policy(obs))
    term = u.termination_manager.terminated.clone()
    cmd_term.vel_command_b[:] = machine.post_step(term)
    obs = env.get_observations()
    return term


def start_round():
    """모든 환경을 리셋하고 새 배치로 시작한다. 돌려주는 값: (색, 상자 무게, 받침대 이동·방향, 명령 문장들)"""
    global obs
    env.reset()
    machine.reset()
    color, layouts, masses = vp.sample_round(rng, N, args_cli.pos_range, args_cli.yaw_range, mass_range, color=args_cli.color)
    box, stand_xy, stand_yaw = vp.setup_round(u, machine, color, layouts, masses)
    vp.random_lighting(rng)
    cmd_term.vel_command_b[:] = 0.0
    obs = env.get_observations()
    for _ in range(25):  # 자리 잡기
        machine.pre_step()
        tick()
    machine.start(box, stand_xy, stand_yaw, stand_xy, stand_yaw, obstacles=stand_xy[None])
    tasks = [vp.task_sentence(rng) for _ in range(N)]
    return color, masses, layouts, tasks


with torch.inference_mode():
    if args_cli.sync_check:
        # 손을 뻗는 도중의 한 스텝 K 에서, env.step 직전에 상자를 20cm 위로 순간 이동시킨다 (그 스텝 동안 다시 떨어지지 않게 매번 다시 놓음).
        # 시점이 맞으면: K 번째 사진(스텝 직전)에는 원래 자리, K+1 번째 사진에는 옮긴 자리의 상자가 보여야 한다.
        color, masses, layouts, tasks = start_round()
        box = machine.box
        frames, moved_at = [], None
        for step in range(MAX_STEPS):
            machine.pre_step()
            st = machine.state.clone()
            if moved_at is None and int(st[0]) == ppm.S_CLOSE and int(machine.t_state[0]) >= 20:
                moved_at = step
            img, _ = vp.read_obs(u, machine)
            if moved_at is not None:
                frames.append(img[0].cpu().numpy().copy())
                if step >= moved_at:  # K 번째 스텝부터 상자를 위에 둠
                    ids0 = torch.zeros(1, dtype=torch.long, device=device)
                    pose = box.data.root_state_w[:1, :7].clone()
                    pose[:, 2] = ppm.BOX_Z_ON_TABLE + 0.20
                    box.write_root_pose_to_sim(pose, env_ids=ids0)
                    box.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=ids0)
                if step >= moved_at + 3:
                    break
            tick()
        import imageio.v3 as iio  # noqa: E402

        diffs = [float(np.abs(frames[i + 1].astype(np.int16) - frames[i].astype(np.int16)).mean()) for i in range(len(frames) - 1)]
        iio.imwrite(out_dir / "sync_check.png", np.concatenate(frames, axis=1))
        print(f"[sync] 상자를 옮긴 스텝 K = {moved_at}. 사진 K, K+1, K+2, K+3 을 나란히 저장: {out_dir / 'sync_check.png'}")
        print(f"[sync] 이웃한 사진의 평균 밝기 차이: K→K+1 {diffs[0]:.2f}, K+1→K+2 {diffs[1]:.2f}, K+2→K+3 {diffs[2]:.2f}")
        print("[sync] K→K+1 에서만 크게 바뀌면 사진이 한 스텝도 늦지 않는 것 (K+1→K+2 에서 바뀌면 한 스텝 늦음)")
    elif args_cli.two_box:
        # 2차: 상태 머신은 받침대 앞에 서는 데까지만, 그 뒤 팔 동작은 SidePickScript (위로 → 상자 위로 → 내려서 잡기 → 똑바로 들기)
        saved = len(list(out_dir.glob("ep_*.npz")))
        target = saved + args_cli.episodes
        n_round, n_try, n_ok, t0 = 0, 0, 0, time.time()
        fail_counts, clear_t, clear_o, other_moved = {}, [], [], []
        REC = (seq.UP, seq.OVER, seq.DOWN, seq.GRASP, seq.LIFT)
        while saved < target:
            env.reset()
            machine.reset()
            seq.reset()
            color, rounds = vp.sample_round2(rng, N, args_cli.pos_range, args_cli.yaw_range, mass_range, color=args_cli.color)
            box, stand_xy, stand_yaw, side = vp.setup_round2(u, machine, color, rounds)
            other = vp.OtherBox(u, machine, rounds, side)
            vp.random_lighting(rng)
            cmd_term.vel_command_b[:] = 0.0
            obs = env.get_observations()
            for _ in range(25):
                machine.pre_step()
                tick()
            other_start = other.pose()[0].clone()
            machine.start(box, stand_xy, stand_yaw, stand_xy, stand_yaw, obstacles=stand_xy[None])
            tasks = [vp.task_sentence(rng, color) for _ in range(N)]
            buf = [{"images": [], "state": [], "action": []} for _ in range(N)]
            started = torch.zeros(N, dtype=torch.bool, device=device)
            fell = torch.zeros(N, dtype=torch.bool, device=device)
            min_clear_t = torch.full((N,), 1.0e9, device=device)
            min_clear_o = torch.full((N,), 1.0e9, device=device)
            for step in range(MAX_STEPS):
                machine.pre_step()
                ph = seq.phase.clone()
                rec = (ph == seq.HOLD) & (seq.t < args_cli.hold_steps)
                for p_ in REC:
                    rec |= ph == p_
                img, state = vp.read_obs(u, machine)
                q_cmd = machine.ik.q_cmd.clone()
                term = tick()
                fell |= term
                seq.fail(term & seq.active)
                seq.update()
                grasp = machine.attached.float()
                tracker.update(seq.active)
                if step % 5 == 0 and bool(seq.active.any()):
                    act = seq.active
                    min_clear_o = torch.where(act, torch.minimum(min_clear_o, other.clearance()), min_clear_o)
                    held = act & machine.attached
                    min_clear_t = torch.where(held, torch.minimum(min_clear_t, machine.hand_box_clearance()), min_clear_t)
                if bool(rec.any()):
                    img_np, state_np = img.cpu().numpy(), state.cpu().numpy()
                    act_np = torch.cat([q_cmd, grasp[:, None]], dim=1).cpu().numpy()
                    for e in rec.nonzero().flatten().tolist():
                        buf[e]["images"].append(img_np[e].copy())
                        buf[e]["state"].append(state_np[e])
                        buf[e]["action"].append(act_np[e])
                ready = (machine.state == ppm.S_DONE) & (seq.phase == seq.IDLE) & ~fell & ~started
                if bool(ready.any()):  # 받침대 앞에 선 환경은 다음 스텝부터 팔을 움직임
                    seq.begin(ready)
                    tracker.begin(ready)
                    started |= ready
                if not bool(machine.active.any()) and not bool(seq.active.any()):
                    break
            n_round += 1
            ok = (seq.phase == seq.DONE) & tracker.success & ~fell
            moved = torch.norm(other.pose()[0] - other_start, dim=1)
            n_try += N
            n_ok += int(ok.sum())
            for e in range(N):
                if not bool(ok[e]):
                    key = "넘어짐" if bool(fell[e]) else ("서지 못함" if not bool(started[e]) else f"팔 동작({seq.NAMES[int(seq.phase[e])]})")
                    fail_counts[key] = fail_counts.get(key, 0) + 1
                    continue
                clear_t.append(float(min_clear_t[e]))
                clear_o.append(float(min_clear_o[e]))
                other_moved.append(float(moved[e]))
                if saved >= target:
                    continue
                r = rounds[e]
                meta = {
                    "task": tasks[e], "color": color, "side": "left" if r["side"] > 0 else "right", "other_color": r["other"],
                    "box_mass": round(r["mass"], 3), "fps": vp.CONTROL_HZ, "stand_offset": [round(v, 4) for v in r["stand"]],
                    "stance_error": [round(float(machine.rec[k][e]), 4) for k in ("stance_a_x", "stance_a_y", "stance_a_yaw")],
                    "grasp_err": round(float(seq.grasp_err[e]), 4), "max_lift": round(float(tracker.max_lift[e]), 4),
                    "min_clearance_target": round(float(min_clear_t[e]), 4), "min_clearance_other": round(float(min_clear_o[e]), 4),
                    "other_box_moved": round(float(moved[e]), 4),
                    "frames": len(buf[e]["images"]), "seed": args_cli.seed, "round": n_round, "env": e,
                }
                np.savez_compressed(out_dir / f"ep_{saved:06d}.npz", images=np.stack(buf[e]["images"]),
                                    state=np.stack(buf[e]["state"]).astype(np.float32),
                                    action=np.stack(buf[e]["action"]).astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))
                saved += 1
            lens = [len(b["images"]) for b in buf if b["images"]]
            print(f"[collect] 묶음 {n_round}: 집을 색 {color}, 성공 {int(ok.sum())}/{N}, 저장 {saved}/{target}, "
                  f"에피소드 길이 {min(lens) if lens else 0}~{max(lens) if lens else 0} 스텝, {time.time() - t0:.0f}초", flush=True)
        print(f"\n========== 시범 수집 (2차, 상자 두 개): 저장 {saved}개 ({out_dir}), 시도 {n_try}개 중 성공 {n_ok}개, 묶음 {n_round}개, {time.time() - t0:.0f}초 ==========")
        if fail_counts:
            print("실패: " + ", ".join(f"{k} {v}" for k, v in fail_counts.items()))
        clear_t, clear_o = [v for v in clear_t if v < 1.0e8], [v for v in clear_o if v < 1.0e8]
        if clear_t and clear_o:
            print(f"손과 집은 상자 옆면 사이의 최소 틈 (cm, 들고 있는 동안, 음수 = 파고듦): 평균 {np.mean(clear_t) * 100:.2f}, 최소 {np.min(clear_t) * 100:.2f}")
            print(f"손과 옆 상자 사이의 최소 틈 (cm, 팔을 움직이는 동안): 평균 {np.mean(clear_o) * 100:.2f}, 최소 {np.min(clear_o) * 100:.2f}")
            print(f"옆 상자가 밀린 거리 (cm): 평균 {np.mean(other_moved) * 100:.2f}, 최대 {np.max(other_moved) * 100:.2f}")
    else:
        saved = len(list(out_dir.glob("ep_*.npz")))
        target = saved + args_cli.episodes
        n_round, n_try, t0 = 0, 0, time.time()
        fail_counts = {}
        while saved < target:
            color, masses, layouts, tasks = start_round()
            buf = [{"images": [], "state": [], "action": []} for _ in range(N)]
            started = torch.zeros(N, dtype=torch.bool, device=device)
            fell = torch.zeros(N, dtype=torch.bool, device=device)
            for step in range(MAX_STEPS):
                machine.pre_step()  # 이 스텝의 팔 목표값
                st = machine.state.clone()
                rec = torch.zeros(N, dtype=torch.bool, device=device)
                for s in REC_STATES:
                    rec |= st == s
                rec |= (st == ppm.S_HOLD) & (machine.t_state < args_cli.hold_steps)
                new = rec & ~started
                if bool(new.any()):
                    tracker.begin(new)
                    started |= new
                img, state = vp.read_obs(u, machine)  # 스텝 직전의 사진·상태
                q_cmd = machine.ik.q_cmd.clone()
                term = tick()
                fell |= term
                grasp = machine.attached.float()  # 이 스텝이 끝났을 때의 잡기 신호
                tracker.update(started & machine.active)
                if bool(rec.any()):
                    img_np, state_np = img.cpu().numpy(), state.cpu().numpy()
                    act_np = torch.cat([q_cmd, grasp[:, None]], dim=1).cpu().numpy()
                    for e in rec.nonzero().flatten().tolist():
                        buf[e]["images"].append(img_np[e].copy())
                        buf[e]["state"].append(state_np[e])
                        buf[e]["action"].append(act_np[e])
                if not bool(machine.active.any()):
                    break
            n_round += 1
            ok = (machine.state == ppm.S_DONE) & tracker.success & ~fell
            n_try += N
            for e in range(N):
                if not bool(ok[e]):
                    key = ppm.STATE_NAMES[int(machine.fail_state[e])] if int(machine.fail_state[e]) >= 0 else "기준 미달"
                    fail_counts[key] = fail_counts.get(key, 0) + 1
                    continue
                if saved >= target:
                    break
                meta = {
                    "task": tasks[e], "color": color, "box_mass": round(masses[e], 3), "fps": vp.CONTROL_HZ,
                    "stand_offset": [round(v, 4) for v in layouts[e]["stands"]["stand_C"]],
                    "stance_error": [round(float(machine.rec[k][e]), 4) for k in ("stance_a_x", "stance_a_y", "stance_a_yaw")],
                    "grasp_err": round(float(machine.rec["grasp_err"][e]), 4), "max_lift": round(float(tracker.max_lift[e]), 4),
                    "frames": len(buf[e]["images"]), "seed": args_cli.seed, "round": n_round, "env": e,
                }
                np.savez_compressed(out_dir / f"ep_{saved:06d}.npz", images=np.stack(buf[e]["images"]),
                                    state=np.stack(buf[e]["state"]).astype(np.float32),
                                    action=np.stack(buf[e]["action"]).astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))
                saved += 1
            print(f"[collect] 묶음 {n_round}: 색 {color}, 성공 {int(ok.sum())}/{N}, 저장 {saved}/{target}, "
                  f"에피소드 길이 {min(len(b['images']) for b in buf)}~{max(len(b['images']) for b in buf)} 스텝, {time.time() - t0:.0f}초", flush=True)
        print(f"\n========== 시범 수집: 저장 {saved}개 ({out_dir}), 시도 {n_try}개 중 성공 {n_try - sum(fail_counts.values())}개, "
              f"묶음 {n_round}개, {time.time() - t0:.0f}초 ==========")
        if fail_counts:
            print("실패: " + ", ".join(f"{k} {v}" for k, v in fail_counts.items()))

env.close()
simulation_app.close()
