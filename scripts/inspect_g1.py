"""G1 짐 든 채 걷기 과제 점검 (학습 전 1회).

확인하는 것
1. 로봇 링크(body) 이름 → 손 링크 이름(HAND_LINKS) 확정
2. 관절 이름과 동작(정책 출력) 크기 → 다리+허리 13개인지
3. 정책이 0 을 내도 팔이 짐 든 자세를 유지하는지 (팔 관절 목표값·실제 값)
4. 손 무게 무작위화가 적용됐는지 (환경별 palm 링크 질량, 원래 0.281kg)
5. 팔 자세 방향: 리셋 직후 팔꿈치·손이 골반 기준으로 앞(x>0)/뒤 어디에 있는지
6. 구동기 그룹별 PD 이득, 손 질량별 어깨·팔꿈치 처짐, 가장 무거운 손의 시간 변화(출렁임)
예) python -u scripts\\inspect_g1.py --headless
   (-u: 출력을 파일로 저장할 때 끝부분이 잘리지 않게 함. simulation_app.close() 가 버퍼를 비우기 전에 프로세스를 끝냄)
   원래 G1 과 비교: --task Isaac-Velocity-Flat-G1-v0
   팔 PD 이득 비교: --fix_base --num_envs 16 --steps 50 [--arm_stiffness 200 --arm_damping 20]
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect G1 carry task.")
parser.add_argument("--task", type=str, default="Humanoid-G1-Carry-Flat-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=100, help="0 동작으로 진행할 스텝 수 (50스텝 = 1초)")
parser.add_argument("--seed", type=int, default=42, help="같은 시드면 환경별 손 질량이 같아 설정끼리 비교 가능")
parser.add_argument("--arm_stiffness", type=float, default=None, help="어깨·팔꿈치 강성 덮어쓰기 (비교용, 짐 과제만)")
parser.add_argument("--arm_damping", type=float, default=None, help="어깨·팔꿈치 감쇠 덮어쓰기 (비교용, 짐 과제만)")
parser.add_argument("--fix_base", action="store_true",
                    help="골반을 공중에 고정: 넘어지는 몸 움직임 없이 팔 PD 의 처짐·출렁임만 봄")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = args_cli.task

env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
if args_cli.arm_stiffness is not None:
    env_cfg.scene.robot.actuators["arms"].stiffness = args_cli.arm_stiffness
if args_cli.arm_damping is not None:
    env_cfg.scene.robot.actuators["arms"].damping = args_cli.arm_damping
if args_cli.fix_base:
    spawn = env_cfg.scene.robot.spawn
    env_cfg.scene.robot.spawn = spawn.replace(articulation_props=spawn.articulation_props.replace(fix_root_link=True))
env = gym.make(TASK, cfg=env_cfg)
env.reset()

unwrapped = env.unwrapped
robot = unwrapped.scene["robot"]

print(f"\n========== 팔 자세 방향 ({TASK}, 리셋 직후, 환경 0) ==========")
print("골반(pelvis) 기준 위치 m — x: 앞(+)/뒤(-), y: 왼쪽(+), z: 위(+)")
root_pos = robot.data.root_pos_w[:1]
root_quat = robot.data.root_quat_w[:1]
for side in ("left", "right"):
    for link in ("shoulder_pitch_link", "elbow_pitch_link", "palm_link"):
        name = f"{side}_{link}"
        p_w = robot.data.body_pos_w[:1, robot.body_names.index(name)] - root_pos
        p_b = quat_apply_inverse(root_quat, p_w)[0]
        print(f"{name:28s} x={p_b[0].item():+.3f}  y={p_b[1].item():+.3f}  z={p_b[2].item():+.3f}")

print("\n========== 링크(body) 목록 ==========")
masses = robot.root_physx_view.get_masses()[0]
for i, name in enumerate(robot.body_names):
    print(f"{i:3d}  {name:35s} {masses[i].item():7.3f} kg")

all_masses = robot.root_physx_view.get_masses()
print("\n========== 환경별 손 질량 (kg) ==========")
for name in ("left_palm_link", "right_palm_link"):
    i = robot.body_names.index(name)
    print(f"{name:20s}", [round(m, 3) for m in all_masses[:, i].tolist()])

print("\n========== 관절 목록 ==========")
for i, name in enumerate(robot.joint_names):
    print(f"{i:3d}  {name}")

act_dim = unwrapped.action_manager.total_action_dim
print(f"\n동작 크기: {act_dim}")
print("동작 관절:", getattr(unwrapped.action_manager.get_term("joint_pos"), "_joint_names", "(확인 불가)"))
print("관측 크기:", unwrapped.observation_manager.group_obs_dim)

print("\n========== 구동기 그룹과 시뮬레이션에 들어간 PD 이득 (환경 0) ==========")
sim_kp = robot.root_physx_view.get_dof_stiffnesses()[0]
sim_kd = robot.root_physx_view.get_dof_dampings()[0]
for act_name, act in robot.actuators.items():
    j = robot.joint_names.index(act.joint_names[0])
    print(f"{act_name:10s} 관절 {len(act.joint_names):2d}개  강성 {sim_kp[j].item():7.1f}  감쇠 {sim_kd[j].item():6.1f}"
          f"  (예: {act.joint_names[0]})")

arm_ids = [i for i, n in enumerate(robot.joint_names) if "shoulder" in n or "elbow" in n or n.startswith(("left_one", "left_two"))]
zero_action = torch.zeros(args_cli.num_envs, act_dim, device=unwrapped.device)
# 넘어져 리셋되면 실제값이 기본값으로 다시 쓰이므로, 리셋 횟수와 진행 중 최대 벗어남도 기록한다
reset_count = torch.zeros(args_cli.num_envs, dtype=torch.long, device=unwrapped.device)
max_dev = torch.zeros(args_cli.num_envs, len(arm_ids), device=unwrapped.device)
history = []  # 스텝별 (실제 - 기본), 출렁임 확인용
for _ in range(args_cli.steps):
    _, _, terminated, truncated, _ = env.step(zero_action)
    reset_count += (terminated | truncated).long()
    dev = robot.data.joint_pos[:, arm_ids] - robot.data.default_joint_pos[:, arm_ids]
    history.append(dev.clone())
    max_dev = torch.maximum(max_dev, dev.abs())
history = torch.stack(history).cpu()  # (스텝, 환경, 팔 관절)

print(f"\n========== 진행 중 리셋 ({args_cli.steps}스텝) ==========")
print("환경별 리셋 횟수:", reset_count.tolist())
print("환경별 현재 에피소드 길이(스텝):", unwrapped.episode_length_buf.tolist())

# 리셋 직후 환경은 실제값이 기본값 그대로이므로, 현재 에피소드가 가장 긴 환경을 보여준다
e = int(torch.argmax(unwrapped.episode_length_buf).item())
hand_ids = [robot.body_names.index(n) for n in ("left_palm_link", "right_palm_link")]
print(f"\n========== 팔 관절 ({args_cli.steps}스텝 후, 환경 {e}, "
      f"에피소드 {int(unwrapped.episode_length_buf[e])}스텝, "
      f"손 질량 {[round(all_masses[e, h].item(), 3) for h in hand_ids]} kg) ==========")
print(f"{'관절':30s} {'기본값':>8s} {'목표값':>8s} {'실제값':>8s} {'최대벗어남':>10s}")
for k, i in enumerate(arm_ids):
    print(
        f"{robot.joint_names[i]:30s} "
        f"{robot.data.default_joint_pos[e, i].item():8.3f} "
        f"{robot.data.joint_pos_target[e, i].item():8.3f} "
        f"{robot.data.joint_pos[e, i].item():8.3f} "
        f"{max_dev[e, k].item():10.3f}"
    )

# 손 질량별 처짐: 리셋이 없었던 환경만 (리셋되면 실제값이 기본값으로 다시 쓰임)
ok_envs = [i for i in range(args_cli.num_envs) if reset_count[i].item() == 0]
rows = []
for env_i in ok_envs:
    for s, side in enumerate(("left", "right")):
        sp = arm_ids.index(robot.joint_names.index(f"{side}_shoulder_pitch_joint"))
        el = arm_ids.index(robot.joint_names.index(f"{side}_elbow_pitch_joint"))
        rows.append((all_masses[env_i, hand_ids[s]].item(), env_i, side, sp, el))
rows.sort(reverse=True)
print(f"\n========== 손 질량별 처짐 ({args_cli.steps}스텝 후, 리셋 없는 환경 {len(ok_envs)}개, 단위 rad, 실제-기본) ==========")
print(f"{'손 질량':>8s} {'환경':>4s} {'쪽':>6s} {'어깨pitch':>10s} {'최대':>7s} {'팔꿈치':>8s} {'최대':>7s}")
for mass, env_i, side, sp, el in rows:
    print(f"{mass:8.3f} {env_i:4d} {side:>6s} {history[-1, env_i, sp].item():+10.3f} {max_dev[env_i, sp].item():7.3f} "
          f"{history[-1, env_i, el].item():+8.3f} {max_dev[env_i, el].item():7.3f}")

# 출렁임: 가장 무거운 손의 어깨 pitch 가 시간에 따라 어떻게 움직이는지
if rows:
    mass, env_i, side, sp, el = rows[0]
    print(f"\n========== 시간 변화 (환경 {env_i}, {side}, 손 {mass:.3f} kg, 2스텝 = 0.04초 간격, 실제-기본 rad) ==========")
    print("어깨pitch:", " ".join(f"{v:+.3f}" for v in history[1::2, env_i, sp].tolist()))
    print("팔꿈치  :", " ".join(f"{v:+.3f}" for v in history[1::2, env_i, el].tolist()))

env.close()
simulation_app.close()
