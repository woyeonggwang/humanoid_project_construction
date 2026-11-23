"""상자 집어 옮기기 전체 순서 실행·평가·녹화 (학습 없이 스크립트 상태 머신, humanoid_tasks/g1_carry/pick_place_machine.py).

순서 (환경마다 따로 진행)
  팔 벌리기 → 받침대 A 앞까지 걷기 → 멈춤 → 손을 상자 옆으로 → 붙이기 → 들기 → 뒤로 물러나기
  → 뒤로 돌아 받침대 B 앞까지 걷기 → 멈춤 → 내려놓기 → 떼기 → 손 벌리기 → 팔 거두기

- 기본: 걷기 정책 v2 (6단계). 스크립트 보정 없이 실행한다
- --legacy: 5단계의 걷기 정책(팔 무작위판 처음부터 학습)과 그때의 보정 세 가지
  (정책 관측 거르기, 정지 명령을 정해진 걸음 박자에서만 주기, 뒤로 물러나는 대신 옆으로 비켜서기)
- 붙이기: 양손이 상자 옆 잡는 위치에 오면 매 스텝 상자 위치·자세를 두 손 기준으로 옮기고(중력은 같은 크기의 힘으로 상쇄),
  손(palm) 질량을 상자 무게의 절반씩 늘림. 뗄 때 원래대로

예) 평가 (상자 위치 무작위): python -u scripts\\pick_place_demo.py --num_envs 100 --randomize --headless
    들기까지만:              python -u scripts\\pick_place_demo.py --until lift --num_envs 64 --headless
    녹화:                    python -u scripts\\pick_place_demo.py --video --headless --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
    5단계 재현:              python -u scripts\\pick_place_demo.py --legacy --num_envs 100 --randomize --headless
"""

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pick-and-place demo with a scripted state machine.")
parser.add_argument("--legacy", action="store_true", help="5단계의 걷기 정책과 스크립트 보정으로 실행")
parser.add_argument("--task", type=str, default=None, help="기본: v2 는 Humanoid-G1-PickPlace-V2-v0, legacy 는 Humanoid-G1-PickPlace-v0")
parser.add_argument("--load_run", type=str, default=None, help="기본: v2 는 가장 최근 *_loco_v2, legacy 는 5단계에서 쓴 학습")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--until", type=str, default="full", choices=["lift", "full"], help="lift: 들어 올린 뒤 2초 유지하고 끝")
parser.add_argument("--randomize", action="store_true", help="상자(와 받침대 A) 시작 위치를 좌우·앞뒤 ±10cm, 방향 ±15° 로 바꿈")
parser.add_argument("--box_mass", type=float, default=None, help="상자 무게 kg (기본: 설정값)")
parser.add_argument("--grasp_gap", type=float, default=None, help="상자 옆면에서 palm 원점까지의 거리 m (기본: pick_place_machine.GRASP_GAP)")
parser.add_argument("--max_time", type=float, default=75.0, help="최대 진행 시간 (초)")
parser.add_argument("--video", action="store_true", help="환경 0 을 따라가며 녹화 (num_envs 1 권장)")
parser.add_argument("--view", type=str, default="side", choices=["side", "front"], help="녹화 시점: 옆모습 / 앞에서 비스듬히")
parser.add_argument("--name", type=str, default="pick_place", help="영상 파일 이름 앞부분")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--debug", action="store_true", help="환경 0 의 상태 변화를 출력")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.task is None:
    args_cli.task = "Humanoid-G1-PickPlace-v0" if args_cli.legacy else "Humanoid-G1-PickPlace-V2-v0"
if args_cli.load_run is None:
    args_cli.load_run = "2026-11-20_20-54-06_armrand_scratch" if args_cli.legacy else ".*_loco_v2"
if args_cli.checkpoint is None:
    args_cli.checkpoint = "model_3999.pt" if args_cli.legacy else "model_.*.pt"

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
from humanoid_tasks.g1_carry import pick_place_machine as ppm  # noqa: E402
from humanoid_tasks.g1_carry.arm_ik import PolicyObsFilter  # noqa: E402
from humanoid_tasks.g1_carry.pick_place_cfg import BOX_SIZE, BOX_Z_ON_TABLE, TABLE_A_XY, TABLE_B_XY, TABLE_SIZE  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

RAD2DEG = ppm.RAD2DEG

# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------
N = args_cli.num_envs
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
if args_cli.box_mass is not None:
    env_cfg.scene.box.spawn.mass_props.mass = args_cli.box_mass
if args_cli.video:
    # 골반을 따라가는 카메라. side: 옆(-y)에서 (걷기·잡기·내려놓기가 모두 옆모습), front: 받침대 A 쪽 앞에서 비스듬히
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.0, -3.4, 0.25) if args_cli.view == "side" else (2.6, -2.2, 0.7)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.05)
    env_cfg.viewer.resolution = (1280, 720)

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
step_dt = env_cfg.sim.dt * env_cfg.decimation
max_steps = int(args_cli.max_time / step_dt)

env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
video_dir = os.path.join(os.path.dirname(resume_path), "videos", "pick_place")
if args_cli.video:
    env = gym.wrappers.RecordVideo(env, video_folder=video_dir, step_trigger=lambda step: step == 0,
                                   video_length=max_steps, name_prefix=args_cli.name, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
device = u.device
robot, box, table_a = u.scene["robot"], u.scene["box"], u.scene["table_a"]
origins = u.scene.env_origins

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
run_name = os.path.join(os.path.basename(os.path.dirname(resume_path)), os.path.basename(resume_path))
print(f"[pick] 체크포인트: {run_name} ({'5단계 보정 사용' if args_cli.legacy else '스크립트 보정 없음'})")

grasp_gap = args_cli.grasp_gap if args_cli.grasp_gap is not None else (ppm.LEGACY_GRASP_GAP if args_cli.legacy else ppm.GRASP_GAP)
machine = ppm.PickPlaceMachine(u, legacy=args_cli.legacy, grasp_gap=grasp_gap, until=args_cli.until, debug=args_cli.debug)
obs_filter = PolicyObsFilter(u, robot) if args_cli.legacy else (lambda o: o)
cmd_term = u.command_manager.get_term("base_velocity")
box_mass = float(box.root_physx_view.get_masses()[0, 0])
print(f"[pick] 환경 {N}개, 상자 {BOX_SIZE} m·{box_mass:.2f}kg, 받침대 {TABLE_SIZE} m, 손과 상자 옆면 사이 {grasp_gap * 100:.1f}cm")

# ---------------------------------------------------------------------------
# 시작 배치: 받침대 A·상자 (무작위면 좌우·앞뒤 ±10cm, 방향 ±15°), 받침대 B 는 고정
# ---------------------------------------------------------------------------
gen = torch.Generator().manual_seed(args_cli.seed)
if args_cli.randomize:
    off = (torch.rand(N, 2, generator=gen) * 2 - 1) * 0.10
    box_yaw0 = (torch.rand(N, generator=gen) * 2 - 1) * math.radians(15.0)
else:
    off, box_yaw0 = torch.zeros(N, 2), torch.zeros(N)
off, box_yaw0 = off.to(device), box_yaw0.to(device)
a_xy = origins[:, :2] + torch.tensor(TABLE_A_XY, device=device) + off
b_xy = origins[:, :2] + torch.tensor(TABLE_B_XY, device=device)
table_a.write_root_pose_to_sim(torch.cat([a_xy, torch.full((N, 1), TABLE_SIZE[2] / 2, device=device), ppm.quat_from_yaw(box_yaw0)], dim=1))
box.write_root_pose_to_sim(torch.cat([a_xy, torch.full((N, 1), BOX_Z_ON_TABLE, device=device), ppm.quat_from_yaw(box_yaw0)], dim=1))
box.write_root_velocity_to_sim(torch.zeros(N, 6, device=device))
b_yaw = torch.full((N,), math.pi, device=device)  # 받침대 B 는 -x 를 보고 섬

# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
cmd_term.vel_command_b[:] = 0.0
obs = env.get_observations()
step = 0
with torch.inference_mode():
    machine.start(box, a_xy, box_yaw0, b_xy, b_yaw)
    for step in range(max_steps):
        machine.pre_step()
        obs, _, _, _ = env.step(policy(obs_filter(obs)))
        cmd = machine.post_step(u.termination_manager.terminated)
        cmd_term.vel_command_b[:] = cmd
        obs = env.get_observations()
        if args_cli.debug and step % 25 == 0:
            p = robot.data.root_pos_w[0] - origins[0]
            bp = box.data.root_pos_w[0] - origins[0]
            print(f"[dbg] t={step:5d} {ppm.STATE_NAMES[int(machine.state[0])]:12s} 골반 ({p[0].item():+.2f}, {p[1].item():+.2f}) "
                  f"방향 {robot.data.heading_w[0].item() * RAD2DEG:+6.1f}° 명령 ({cmd[0, 0].item():+.2f}, {cmd[0, 1].item():+.2f}, {cmd[0, 2].item():+.2f}) "
                  f"기울기 {machine.tilt()[0].item():4.1f}° 상자 ({bp[0].item():+.2f}, {bp[1].item():+.2f}, {bp[2].item():.3f}) 붙음 {int(machine.attached[0])}")
        if bool(machine.finished.all()):
            break

# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------
rec, state = machine.rec, machine.state
finished = state == ppm.S_DONE
if args_cli.until == "full":
    # 성공: 끝까지 진행 + 상자가 받침대 B 위에 있고(높이 차 2cm 이내) 가운데에서 5cm 이내
    success = finished & (rec["place_xy"] <= 0.05) & (rec["place_z"].abs() <= 0.02)
else:
    success = finished
fail_state = machine.fail_state.clone()
unfinished = machine.active
fail_state[unfinished] = state[unfinished]


def stat(key: str, scale: float = 100.0, absolute: bool = False):
    v = rec[key][~torch.isnan(rec[key])] * scale
    if len(v) == 0:
        return "—"
    if absolute:
        v = v.abs()
    return f"평균 {v.mean().item():.2f}, 최대 {v.max().item():.2f} ({len(v)}개)"


print(f"\n========== 상자 집어 옮기기 ({'들기까지' if args_cli.until == 'lift' else '전체 순서'}, 환경 {N}개, "
      f"상자 위치 {'무작위 (±10cm, ±15°)' if args_cli.randomize else '고정'}, 상자 {box_mass:.1f}kg, "
      f"{'5단계 정책 + 보정' if args_cli.legacy else 'v2 정책, 보정 없음'}) ==========")
print(f"성공: {int(success.sum())}/{N} ({int(success.sum()) / N * 100:.0f}%), 넘어짐: {int(machine.fell.sum())}, 진행 {step + 1}스텝 ({(step + 1) * step_dt:.1f}초)")
if bool((~success).any()):
    print("실패한 곳 (상태별 환경 수):")
    for s_code in range(len(ppm.STATE_NAMES)):
        sel = (~success) & (fail_state == s_code)
        if bool(sel.any()):
            print(f"  [{ppm.STATE_NAMES[s_code]}] {int(sel.sum())}개 (그중 넘어짐 {int((sel & machine.fell).sum())}) — 환경 {sel.nonzero().flatten().tolist()[:12]}")
    for e, info in sorted(machine.fall_info.items()):
        print(f"  넘어짐 환경 {e}: {info}")
    sel = (~success) & finished
    if bool(sel.any()):
        print(f"  [놓인 위치가 기준 밖] {int(sel.sum())}개 — 환경 {sel.nonzero().flatten().tolist()[:12]}")
print(f"받침대 A 앞에 멈춘 위치 오차 (cm): 앞뒤 |{stat('stance_a_x', absolute=True)}|, 좌우 |{stat('stance_a_y', absolute=True)}|, "
      f"방향(°) |{stat('stance_a_yaw', RAD2DEG, True)}|")
print(f"정지 명령 뒤 멈추는 동안 돈 각도 (°): A 앞 |{stat('stop_turn_a', RAD2DEG, True)}|, B 앞 |{stat('stop_turn_b', RAD2DEG, True)}|")
print(f"멈춘 자리가 범위 밖이라 다시 다가간 횟수: A 평균 {machine.reapproach_a.float().mean().item():.2f}·최대 {int(machine.reapproach_a.max())}, "
      f"B 평균 {machine.reapproach_b.float().mean().item():.2f}·최대 {int(machine.reapproach_b.max())}")
print(f"붙이기 직전 손 위치 오차 (cm, 양손 중 큰 값): {stat('grasp_err')}")
print(f"붙이기 직전까지 상자가 처음 자리에서 밀린 거리 (cm): {stat('box_shift')}")
for key, label in (("hand_clear", "붙이는 순간"), ("hand_clear_min", "들고 있는 동안 최소")):
    v = rec[key][torch.isfinite(rec[key])] * 100
    if len(v):
        print(f"손(손바닥·손가락 형상)과 상자 옆면 사이의 틈, {label} (cm, 음수 = 상자 안으로 들어감): "
              f"평균 {v.mean().item():.2f}, 최소 {v.min().item():.2f}, 최대 {v.max().item():.2f} ({len(v)}개)")
print(f"붙은 상자가 두 손 기준 자리에서 벗어난 거리 (cm, 한 스텝 뒤, 환경별 최대값): 공중 평균 {machine.box_dev_air.mean().item() * 100:.3f}·"
      f"최대 {machine.box_dev_air.max().item() * 100:.3f}, 받침대에 닿은 동안 평균 {machine.box_dev_table.mean().item() * 100:.2f}·최대 {machine.box_dev_table.max().item() * 100:.2f}")
print(f"들어 올리는 동안 골반 기울기 최대 (°): {stat('lift_tilt', 1.0)}")
print(f"들기까지 걸린 시간 (초): {stat('t_pick', 1.0)}")
if args_cli.until == "full":
    print(f"받침대 B 앞에 멈춘 위치 오차 (cm): 앞뒤 |{stat('stance_b_x', absolute=True)}|, 좌우 |{stat('stance_b_y', absolute=True)}|, "
          f"방향(°) |{stat('stance_b_yaw', RAD2DEG, True)}|")
    print(f"떼기 직전 상자 위치 오차 (cm): {stat('place_err_before')}")
    print(f"내려놓은 위치 오차 (cm, 받침대 B 가운데 기준 수평): {stat('place_xy')}")
    print(f"내려놓은 상자 높이 차 (cm): {stat('place_z', absolute=True)}")
    print(f"전체 걸린 시간 (초): {stat('t_total', 1.0)}")
print("IK 가 쓴 팔 관절 목표값 범위 (rad, 진행 중인 환경 전체)")
for j, jid in enumerate(machine.ik.all_joint_ids):
    print(f"  {robot.joint_names[jid]:28s} {machine.q_cmd_min[j].item():+.2f} ~ {machine.q_cmd_max[j].item():+.2f}"
          f"   (최소: {ppm.STATE_NAMES[int(machine.q_cmd_min_state[j])]}, 최대: {ppm.STATE_NAMES[int(machine.q_cmd_max_state[j])]})")
masses = robot.root_physx_view.get_masses()[:, machine.palm_ids]
print(f"끝난 뒤 손 질량 (kg): 최소 {masses.min().item():.3f}, 최대 {masses.max().item():.3f} (기본 {machine.default_mass[0, machine.palm_ids[0]].item():.3f})")
if args_cli.video:
    print(f"[pick] 영상: {os.path.join('logs', 'rsl_rl', agent_cfg.experiment_name, os.path.basename(os.path.dirname(resume_path)), 'videos', 'pick_place', args_cli.name + '-step-0.mp4')}")

env.close()
simulation_app.close()
