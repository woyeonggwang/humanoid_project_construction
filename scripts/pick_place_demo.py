"""5단계: 상자 집어 옮기기 전체 순서 (학습 없이 스크립트 상태 머신).

순서 (환경마다 따로 진행)
  팔 벌리기 → 받침대 A 앞까지 걷기 → 멈춤 → 손을 상자 옆으로 → 붙이기 → 들기 → 옆으로 비켜서기
  → 뒤로 돌아 받침대 B 앞까지 걷기 → 멈춤 → 내려놓기 → 떼기 → 손 벌리기 → 팔 거두기

- 걷기: 걷기 정책(팔 무작위판 처음부터 학습)에 속도·회전 명령을 줌. 목표 지점까지의 방향·거리로 명령을 정하는 간단한 제어기.
  정지 명령은 걸음 박자의 정해진 구간에서만 주고, 멈추는 동안 움직이는 만큼 미리 보정함. 멈춘 자리가 범위 밖이면 다시 다가감
- 팔: IK (humanoid_tasks/g1_carry/arm_ik.py). 손에 무게가 있을 때는 이동을 시작할 때 목표를 몸통 좌표에 고정하고,
  빈손으로 상자에 손을 뻗을 때만 월드 목표를 계속 따라감. 걷는 동안은 관절 목표를 그대로 둠
- 정책에 넣는 관측에서 학습 때 고정이던 팔 관절(어깨 roll·yaw, 팔꿈치 roll)은 학습 때 값으로 바꿈 (PolicyObsFilter)
- 붙이기: 양손이 상자 옆 잡는 위치에 오면 그때부터 매 스텝 상자 위치·자세를 두 손 기준으로 옮기고(중력은 같은 크기의 힘으로 상쇄),
  손(palm) 질량을 상자 무게의 절반씩 늘림. 뗄 때 원래대로

예) 들기까지 확인 (3단계):  python -u scripts\\pick_place_demo.py --until lift --num_envs 16 --headless
    전체 순서 + 녹화:      python -u scripts\\pick_place_demo.py --video --headless --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
    평가 (상자 위치 무작위): python -u scripts\\pick_place_demo.py --num_envs 100 --randomize --headless
"""

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pick-and-place demo with a scripted state machine.")
parser.add_argument("--task", type=str, default="Humanoid-G1-PickPlace-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--until", type=str, default="full", choices=["lift", "full"], help="lift: 들어 올린 뒤 2초 유지하고 끝")
parser.add_argument("--randomize", action="store_true", help="상자(와 받침대 A) 시작 위치를 좌우·앞뒤 ±10cm, 방향 ±15° 로 바꿈")
parser.add_argument("--box_mass", type=float, default=None, help="상자 무게 kg (기본: 설정값)")
parser.add_argument("--max_time", type=float, default=75.0, help="최대 진행 시간 (초)")
parser.add_argument("--video", action="store_true", help="환경 0 을 옆에서 따라가며 녹화 (num_envs 1 권장)")
parser.add_argument("--name", type=str, default="pick_place", help="영상 파일 이름 앞부분")
parser.add_argument("--load_run", type=str, default="2026-11-20_20-54-06_armrand_scratch")
parser.add_argument("--checkpoint", type=str, default="model_3999.pt")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--debug", action="store_true", help="환경 0 의 상태 변화를 출력")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

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
from humanoid_tasks.g1_carry.arm_ik import DualArmIK, PolicyObsFilter, TorsoFrame  # noqa: E402
from humanoid_tasks.g1_carry.pick_place_cfg import BOX_SIZE, BOX_Z_ON_TABLE, TABLE_A_XY, TABLE_B_XY, TABLE_SIZE  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

RAD2DEG = 57.29578

# ---------------------------------------------------------------------------
# 동작에 쓰는 값 (m, 골반 위치 + 골반이 향한 방향 기준: x 앞, y 왼쪽, z 위)
# ---------------------------------------------------------------------------
GRASP_IN_BOX = (-0.08, 0.24, 0.04)  # 상자 좌표에서 본 왼손(palm 원점) 잡는 위치. 오른손은 y 부호 반대
#   y: 상자 옆면(15cm)에서 9cm 바깥 — 엄지 끝이 palm 원점보다 안쪽으로 약 7cm 나오므로 (CLAUDE.md 1단계 '손 크기')
#   x·z: 손가락이 palm 원점보다 앞 16cm·아래 9cm 까지 내려가므로 손 전체가 옆면 가운데에 오게 뒤·위로 둠
STANCE_DIST = 0.36  # 멈출 때 골반에서 상자(받침대) 중심까지의 거리. 잡는 손 위치가 골반 앞 0.28m = 닿는 범위(0.20~0.35m)의 가운데
OPEN_POSE = (0.30, 0.31, 0.125)  # 벌린 자세 (좌우 ±). 이 자세로 받침대에 다가가면 손이 상자 양옆을 지나감
CARRY_BOX_POSE = (0.28, 0.24, 0.20)  # 상자를 든 자세. 상자 아랫면이 받침대 윗면보다 약 9cm 높음
RETRACT_POSE = (0.20, 0.31, 0.125)  # 내려놓은 뒤 거둔 자세 (벌린 채 뒤로)
LIFT_UP = 0.06  # 붙인 뒤 먼저 똑바로 들어 올리는 높이
PLACE_GAP = 0.005  # 내려놓을 때 상자 아랫면과 받침대 윗면 사이에 남기는 틈
ATTACH_TOL = 0.03  # 양손이 잡는 위치에서 이 거리 안이면 붙임
STILL_STEPS, STILL_POS, STILL_YAW = 50, 0.01, 0.0175  # 멈춤 판정: 1초 동안 골반이 1cm·1° 넘게 움직이지 않음
#   속도만 보면(0.5초 동안 0.05m/s 미만) 아직 천천히 도는 중에도 통과해서, 그 뒤 팔을 뻗는 동안 몸이 더 돌았음
ATTACH_HOLD = 10  # 양손이 ATTACH_TOL 안에 이 스텝만큼 머물면 붙임
# 멈춘 자리가 이 범위 밖이면 다시 다가감 (최대 MAX_REAPPROACH 번). 5cm 넘게 덜 간 채 팔을 뻗으면 닿는 범위 끝이라
# 로봇이 발을 떼며 돌았음 (32개 중 10개, 모두 앞 오차 +4.8~+6.7cm)
STANCE_TOL_X = (-0.08, 0.03)  # 앞 오차 (+ = 덜 감)
STANCE_TOL_Y, STANCE_TOL_YAW = 0.05, 0.10
MAX_REAPPROACH = 3
# 정지 명령을 주는 걸음 박자: 오른발이 떠 있는 구간의 가운데 (위상 0.22~0.30).
# 2026-11-22 정지 실험 (환경 160개, 정지 명령 뒤 6초 동안의 변화): 이 구간에서 멈추면 회전 +4~8°·옆 이동 4cm 이내로 일정하고,
# 다른 박자에서는 평균 10~50°(최대 170°) 돌거나 옆으로 20~50cm 움직임. 빈손/짐 든 상태, 걷다 멈춤/제자리걸음 후 멈춤 모두 같은 경향
GAIT_PERIOD = 0.8
STOP_PHASE = (0.22, 0.30)
# 멈추는 동안 움직이는 만큼 미리 보정 (빈손, 짐 든 상태). 실험의 그 구간 평균값
STOP_YAW_DRIFT = (0.087, 0.131)  # rad: 왼쪽으로 5°, 7.5° 돎 → 그만큼 덜 돈 방향에서 멈춤
STOP_LAT_DRIFT = (0.03, 0.0)  # m: 왼쪽으로 3cm, 0cm 밀림 → 그만큼 오른쪽에서 멈춤
STOP_FWD_DRIFT = (-0.03, -0.02)  # m: 뒤로 3cm, 2cm 밀림 → 그만큼 더 가서 멈춤
# 상자를 든 뒤 돌기 전에 왼쪽으로 비켜서는 거리. 받침대 바로 앞에서 제자리 회전하면 도는 동안 받침대 쪽으로 15~20cm 밀려가
# 몸통이 받침대에 닿았음 (64개 중 8개). 뒤로 걷기는 학습하지 않았으므로 옆으로 비킴
SIDESTEP_DIST, SIDESTEP_SPEED = 0.55, 0.3
G = 9.81

# 상태
S_OPEN, S_WALK_A, S_SETTLE_A, S_CLOSE, S_CLOSE2, S_LIFT1, S_LIFT2, S_HOLD = range(8)
S_WALK_B, S_SETTLE_B, S_PLACE, S_PLACE2, S_RELEASE, S_OPEN_B, S_RETRACT, S_END_HOLD, S_DONE, S_FAIL = range(8, 18)
S_SIDE = 18  # 든 채 유지 → 옆으로 비켜서기 → B 로 걷기
STATE_NAMES = ["팔 벌리기", "A 로 걷기", "A 앞에서 멈춤", "손 뻗기", "손 위치 맞추기", "들어 올리기", "든 자세로", "든 채 유지",
               "B 로 걷기", "B 앞에서 멈춤", "내려놓기", "내려놓기 보정", "떼기", "손 벌리기", "팔 거두기", "마무리", "완료", "실패",
               "옆으로 비켜서기"]
# 상태별 제한 시간 (스텝). 넘으면 그 상태에서 실패로 기록
TIMEOUT = {S_OPEN: 200, S_WALK_A: 1000, S_SETTLE_A: 400, S_CLOSE: 300, S_CLOSE2: 150, S_LIFT1: 150, S_LIFT2: 250,
           S_HOLD: 200, S_WALK_B: 1500, S_SETTLE_B: 400, S_PLACE: 300, S_PLACE2: 150, S_RELEASE: 100, S_OPEN_B: 150,
           S_RETRACT: 250, S_END_HOLD: 200, S_SIDE: 400}

# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------
N = args_cli.num_envs
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
if args_cli.box_mass is not None:
    env_cfg.scene.box.spawn.mass_props.mass = args_cli.box_mass
if args_cli.video:
    # 옆(-y)에서 골반을 따라가는 카메라. 로봇이 +x·-x 로 걷고 받침대를 마주 보므로 걷기·잡기가 모두 옆모습으로 보임
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.0, -3.4, 0.25)
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
robot, box, table_a, table_b = u.scene["robot"], u.scene["box"], u.scene["table_a"], u.scene["table_b"]
origins = u.scene.env_origins

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
print(f"[pick] 체크포인트: {os.path.join(os.path.basename(os.path.dirname(resume_path)), os.path.basename(resume_path))}")

ik = DualArmIK(robot)
torso = TorsoFrame(robot)
obs_filter = PolicyObsFilter(u, robot)
cmd_term = u.command_manager.get_term("base_velocity")
palm_ids = ik.body_ids
box_mass = float(box.root_physx_view.get_masses()[0, 0])
print(f"[pick] 환경 {N}개, 상자 {BOX_SIZE} m·{box_mass:.2f}kg, 받침대 {TABLE_SIZE} m, 붙일 때 손 하나에 더하는 질량 {box_mass / 2:.2f}kg")


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------
def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_z(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """벡터 (..., 3) 를 z 축으로 yaw 만큼 돌린다. yaw 는 앞쪽 차원과 같은 크기."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], dim=-1)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1)


def sym_points(spec) -> torch.Tensor:
    """(x, y, z) → 왼손·오른손 (2, 3)."""
    return torch.tensor([[spec[0], spec[1], spec[2]], [spec[0], -spec[1], spec[2]]], device=device)


def pelvis_to_world(points: torch.Tensor) -> torch.Tensor:
    """골반 yaw 좌표의 점 (2, 3) → 환경별 월드 (N, 2, 3)."""
    yaw = robot.data.heading_w[:, None].expand(-1, points.shape[0])
    return robot.data.root_pos_w[:, None, :] + rot_z(yaw, points[None].expand(N, -1, -1))


def hand_frame():
    """두 손 기준 좌표: 원점 = 양손 가운데, 방향 = 오른손 → 왼손 선에 수직인 앞쪽."""
    hp = ik.hand_pos_w()
    mid = hp.mean(dim=1)
    v = hp[:, 0, :2] - hp[:, 1, :2]
    return mid, torch.atan2(-v[:, 0], v[:, 1])


cpu_all = torch.arange(N)
default_mass, default_inertia = robot.data.default_mass.cpu(), robot.data.default_inertia.cpu()


def set_palm_extra_mass(env_ids: torch.Tensor, extra: float):
    """손(palm) 질량을 기본값 + extra 로 바꾼다. 관성은 질량 비율만큼 (원본 randomize_rigid_body_mass 와 같은 방식).
    실행 중에도 root_physx_view.set_masses / set_inertias 로 바꿀 수 있다 (원본 이벤트가 reset·interval 모드에서 같은 호출을 씀)."""
    ids = env_ids.cpu()
    view = robot.root_physx_view
    masses = view.get_masses().clone()
    masses[ids[:, None], torch.tensor(palm_ids)] = default_mass[ids[:, None], torch.tensor(palm_ids)] + extra
    view.set_masses(masses, ids)
    inertias = view.get_inertias().clone()
    ratio = masses[ids[:, None], torch.tensor(palm_ids)] / default_mass[ids[:, None], torch.tensor(palm_ids)]
    inertias[ids[:, None], torch.tensor(palm_ids)] = default_inertia[ids[:, None], torch.tensor(palm_ids)] * ratio[..., None]
    view.set_inertias(inertias, ids)


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
zeros6 = torch.zeros(N, 6, device=device)
table_a.write_root_pose_to_sim(torch.cat([a_xy, torch.full((N, 1), TABLE_SIZE[2] / 2, device=device), quat_from_yaw(box_yaw0)], dim=1))
box.write_root_pose_to_sim(torch.cat([a_xy, torch.full((N, 1), BOX_Z_ON_TABLE, device=device), quat_from_yaw(box_yaw0)], dim=1))
box.write_root_velocity_to_sim(zeros6)

# 멈출 자리: 상자를 마주 보고 STANCE_DIST 떨어진 곳. B 는 -x 를 보고 섬
dir_a = torch.stack([torch.cos(box_yaw0), torch.sin(box_yaw0)], dim=1)
stance_a_xy, stance_a_yaw = a_xy - STANCE_DIST * dir_a, box_yaw0.clone()
stance_b_yaw = torch.full((N,), math.pi, device=device)
stance_b_xy = b_xy - STANCE_DIST * torch.stack([torch.cos(stance_b_yaw), torch.sin(stance_b_yaw)], dim=1)
place_target = torch.cat([b_xy, torch.full((N, 1), BOX_Z_ON_TABLE + PLACE_GAP, device=device)], dim=1)  # 내려놓을 상자 중심

# ---------------------------------------------------------------------------
# 상태 변수
# ---------------------------------------------------------------------------
state = torch.full((N,), S_OPEN, dtype=torch.long, device=device)
t_state = torch.zeros(N, dtype=torch.long, device=device)
still_count = torch.zeros(N, dtype=torch.long, device=device)
in_tol_count = torch.zeros(N, dtype=torch.long, device=device)
reapproach = torch.zeros(N, dtype=torch.long, device=device)  # 멈춘 자리가 범위 밖이라 다시 다가간 횟수 (받침대 A)
reapproach_b = torch.zeros(N, dtype=torch.long, device=device)  # (받침대 B)
pos_hist = torch.zeros(STILL_STEPS, N, 2, device=device)  # 멈춤 판정용: 최근 1초의 골반 위치·방향
yaw_hist = torch.zeros(STILL_STEPS, N, device=device)
near_mode = torch.zeros(N, dtype=torch.bool, device=device)
fail_state = torch.full((N,), -1, dtype=torch.long, device=device)
fell = torch.zeros(N, dtype=torch.bool, device=device)
done_step = torch.full((N,), -1, dtype=torch.long, device=device)
# 팔 이동
arm_start = torch.zeros(N, 2, 3, device=device)
arm_goal = torch.zeros(N, 2, 3, device=device)  # 몸통 좌표
arm_goal_w0 = torch.zeros(N, 2, 3, device=device)  # 이동을 시작할 때의 월드 목표 (오차 기록용)
arm_t = torch.zeros(N, dtype=torch.long, device=device)
arm_T = torch.ones(N, dtype=torch.long, device=device)
arm_busy = torch.zeros(N, dtype=torch.bool, device=device)
arm_hold = torch.zeros(N, dtype=torch.bool, device=device)  # 이동이 끝난 뒤에도 목표를 계속 따라갈지 (서서 팔을 쓰는 동안)
arm_world = torch.zeros(N, dtype=torch.bool, device=device)  # 목표를 월드에 둘지 (빈손일 때만). False 면 몸통 좌표에 고정
# 붙이기
attached = torch.zeros(N, dtype=torch.bool, device=device)
att_off = torch.zeros(N, 3, device=device)
att_yaw = torch.zeros(N, device=device)
att_floor = torch.full((N,), -1.0e9, device=device)  # 붙인 직후 상자를 이 높이 아래로는 내리지 않음 (받침대에 눌리지 않게)
box_prev_pos = torch.zeros(N, 3, device=device)  # 직전 스텝에 놓은 자리
box_pred_pos = torch.zeros(N, 3, device=device)  # 그 속도대로 한 스텝 뒤에 있을 자리
box_prev_yaw = torch.zeros(N, device=device)
box_prev_valid = torch.zeros(N, dtype=torch.bool, device=device)
# 기록
rec = {k: torch.full((N,), float("nan"), device=device) for k in (
    "stance_a_x", "stance_a_y", "stance_a_yaw", "grasp_err", "stance_b_x", "stance_b_y", "stance_b_yaw",
    "place_err_before", "place_xy", "place_z", "lift_tilt", "t_pick", "t_total", "box_shift")}
box_dev_air = torch.zeros(N, device=device)  # 붙은 상자가 공중에 있을 때 '두 손 기준 자리'에서 벗어난 최대 거리 (한 스텝 뒤)
box_dev_table = torch.zeros(N, device=device)  # 받침대에 닿아 있을 때 (붙인 직후 팔이 처지며 받침대에 눌림)


def start_move(mask: torch.Tensor, goal_w: torch.Tensor, steps: int, world: bool = False):
    """mask 환경의 팔 이동을 시작: 지금 손 위치 → goal_w (N, 2, 3 월드).

    world=False: 목표를 지금의 몸통 좌표에 고정 (몸이 움직여도 팔 관절 목표가 바뀌지 않음). 손에 무게가 있을 때는 이쪽만 씀.
    world=True: 목표를 월드에 두고 매 스텝 몸통 좌표로 다시 바꿔 따라감. 빈손일 때만 (2단계에서 손 0kg 은 안정적이었음).
    """
    if not bool(mask.any()):
        return
    cur = torso.to_frame(ik.hand_pos_w())
    goal = torso.to_frame(goal_w)
    arm_start[mask], arm_goal[mask], arm_goal_w0[mask] = cur[mask], goal[mask], goal_w[mask]
    arm_t[mask], arm_T[mask] = 0, steps
    arm_busy[mask], arm_hold[mask] = True, True
    arm_world[mask] = world


def set_state(mask: torch.Tensor, new_state: int):
    if not bool(mask.any()):
        return
    if args_cli.debug and bool(mask[0]):
        print(f"[dbg] t={step:5d} ({step * step_dt:5.1f}s) 환경 0: {STATE_NAMES[int(state[0])]} → {STATE_NAMES[new_state]}")
    state[mask] = new_state
    t_state[mask] = 0
    still_count[mask] = 0
    in_tol_count[mask] = 0
    near_mode[mask] = False


def fail(mask: torch.Tensor):
    if not bool(mask.any()):
        return
    if args_cli.debug:
        torso_id = robot.body_names.index("torso_link")
        coms = robot.root_physx_view.get_coms()[:, torso_id, :3]
        for e in mask.nonzero().flatten().tolist():
            d = robot.data.root_pos_w[e, :2] - stance_a_xy[e]
            print(f"[dbg] t={step:5d} 환경 {e}: 실패 ({STATE_NAMES[int(state[e])]}, {t_state[e].item() * step_dt:.1f}초째), "
                  f"A 멈출 자리에서 ({d[0].item() * 100:+.1f}, {d[1].item() * 100:+.1f})cm, 방향 {robot.data.heading_w[e].item() * RAD2DEG:+.1f}°, "
                  f"기울기 {tilt()[e].item():.1f}°, 손 오차 {rec['grasp_err'][e].item() * 100:.1f}cm, "
                  f"멈췄을 때 오차 ({rec['stance_a_x'][e].item() * 100:+.1f}, {rec['stance_a_y'][e].item() * 100:+.1f})cm, "
                  f"몸통 무게중심 ({coms[e, 0].item() * 100:+.1f}, {coms[e, 1].item() * 100:+.1f})cm")
    fail_state[mask] = state[mask]
    detach(mask & attached)
    set_state(mask, S_FAIL)
    arm_busy[mask], arm_hold[mask], arm_world[mask] = False, False, False


def box_grasp_points() -> torch.Tensor:
    """상자의 지금 위치·방향 기준 양손 잡는 위치 (N, 2, 3 월드)."""
    yaw = yaw_of(box.data.root_quat_w)[:, None].expand(-1, 2)
    return box.data.root_pos_w[:, None, :] + rot_z(yaw, sym_points(GRASP_IN_BOX)[None].expand(N, -1, -1))


def update_gravity_cancel():
    """붙어 있는 상자에는 중력과 같은 크기의 위쪽 힘을 걸어 둔다 (무게는 손 질량으로 대신하므로)."""
    forces = torch.zeros(N, 1, 3, device=device)
    forces[attached, 0, 2] = box_mass * G
    box.set_external_force_and_torque(forces, torch.zeros(N, 1, 3, device=device), is_global=True)


def attach(mask: torch.Tensor):
    if not bool(mask.any()):
        return
    mid, psi = hand_frame()
    att_off[mask] = rot_z(-psi, box.data.root_pos_w - mid)[mask]
    att_yaw[mask] = wrap(yaw_of(box.data.root_quat_w) - psi)[mask]
    attached[mask] = True
    box_prev_valid[mask] = False
    # 손에 무게가 실리면 팔이 1~2cm 처지는데, 상자를 그대로 따라 내리면 받침대에 눌림 → 들어 올릴 때까지 지금 높이를 바닥으로
    att_floor[mask] = box.data.root_pos_w[mask, 2]
    set_palm_extra_mass(mask.nonzero().flatten(), box_mass / 2)
    update_gravity_cancel()


def detach(mask: torch.Tensor):
    if not bool(mask.any()):
        return
    attached[mask] = False
    ids = mask.nonzero().flatten()
    set_palm_extra_mass(ids, 0.0)
    update_gravity_cancel()
    box.write_root_velocity_to_sim(zeros6[: len(ids)], env_ids=ids)


def move_attached_box():
    """붙어 있는 상자를 두 손 기준 자리로 옮긴다 (위치·자세 + 다음 스텝 동안 따라갈 속도)."""
    if not bool(attached.any()):
        return
    mid, psi = hand_frame()
    pos = mid + rot_z(psi, att_off)
    pos[:, 2] = torch.maximum(pos[:, 2], att_floor)
    yaw = psi + att_yaw
    ids = attached.nonzero().flatten()
    # 직전 스텝에 놓은 자리에서 물리 계산 뒤 얼마나 벗어났는지 (튀거나 떨면 커짐)
    dev = torch.norm(box.data.root_pos_w - box_pred_pos, dim=1)
    on = attached & box_prev_valid
    in_air = box.data.root_pos_w[:, 2] > BOX_Z_ON_TABLE + 0.02
    box_dev_air[on & in_air] = torch.maximum(box_dev_air, dev)[on & in_air]
    box_dev_table[on & ~in_air] = torch.maximum(box_dev_table, dev)[on & ~in_air]
    vel = torch.zeros(N, 6, device=device)
    moving = attached & box_prev_valid
    vel[moving, :3] = ((pos - box_prev_pos) / step_dt)[moving]
    vel[moving, 5] = (wrap(yaw - box_prev_yaw) / step_dt)[moving]
    box.write_root_pose_to_sim(torch.cat([pos, quat_from_yaw(yaw)], dim=1)[ids], env_ids=ids)
    box.write_root_velocity_to_sim(vel[ids], env_ids=ids)
    box_prev_pos[attached] = pos[attached]
    box_pred_pos[attached] = (pos + vel[:, :3] * step_dt)[attached]
    box_prev_yaw[attached] = yaw[attached]
    box_prev_valid[attached] = True


def walk_command(target_xy: torch.Tensor, target_yaw: torch.Tensor, loaded: bool):
    """목표 지점·방향까지의 속도 명령 (N, 3) 과 도착 여부, 오차(앞, 옆, 방향)를 돌려준다.

    멀 때: 목표 쪽을 보게 돈 뒤(방향 차이 > 0.6 rad 이면 제자리 회전) 앞으로 걸음.
    가까울 때(0.3m 안): 목표 방향을 보도록 돌면서 앞·옆으로 조금씩 맞춤. 뒤로 걷기는 학습하지 않았으므로 앞으로만.
    정책은 명령이 0.1 미만이면 '정지'로 보고 발을 떼지 않으므로, 움직여야 할 때는 0.15 이상으로 준다.
    loaded: 상자를 든 상태인지 (멈추는 동안 움직이는 양이 달라 보정값이 다름).
    """
    yaw = robot.data.heading_w
    d = target_xy - robot.data.root_pos_w[:, :2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    ex, ey = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    dist = torch.hypot(ex, ey)
    bearing = torch.atan2(ey, ex)
    yaw_err = wrap(target_yaw - yaw)
    near_mode.copy_(torch.where(near_mode, dist < 0.5, dist < 0.3))

    turn_only = bearing.abs() > 0.6
    wz_far = torch.clamp(1.5 * bearing, -0.8, 0.8)
    wz_far = torch.where(turn_only & (wz_far.abs() < 0.4), 0.4 * torch.sign(bearing), wz_far)
    vx_far = torch.where(turn_only, torch.zeros_like(dist), torch.clamp(1.0 * dist, 0.25, 0.5))

    # 멈추는 동안 움직이는 만큼 미리 보정한 오차로 맞춘다
    k = 1 if loaded else 0
    yaw_aim = yaw_err - STOP_YAW_DRIFT[k]
    ey_aim = ey - STOP_LAT_DRIFT[k]
    ex_aim = ex - STOP_FWD_DRIFT[k]
    need_x, need_y, need_yaw = ex_aim > 0.02, ey_aim.abs() > 0.03, yaw_aim.abs() > 0.06
    vx_near = torch.where(need_x, torch.clamp(1.5 * ex_aim, 0.15, 0.3), torch.zeros_like(ex))
    vy_near = torch.where(need_y, torch.sign(ey_aim) * torch.clamp(1.5 * ey_aim.abs(), 0.15, 0.3), torch.zeros_like(ey))
    wz_near = torch.where(need_yaw, torch.sign(yaw_aim) * torch.clamp(1.5 * yaw_aim.abs(), 0.15, 0.5), torch.zeros_like(ex))
    # 자리는 맞았지만 정지할 박자가 아니면, 아주 느린 회전 명령으로 제자리걸음을 이어 가며 기다린다 (0.1 이상이어야 정지로 보지 않음)
    in_place = ~need_x & ~need_y & ~need_yaw
    phase = torch.remainder(u.episode_length_buf.float() * step_dt / GAIT_PERIOD, 1.0)
    stop_now = in_place & (phase >= STOP_PHASE[0]) & (phase < STOP_PHASE[1])
    wait_sign = torch.where(yaw_aim >= 0, torch.ones_like(yaw_aim), -torch.ones_like(yaw_aim))
    wz_near = torch.where(in_place & ~stop_now, 0.12 * wait_sign, wz_near)

    cmd = torch.stack([torch.where(near_mode, vx_near, vx_far), torch.where(near_mode, vy_near, torch.zeros_like(ey)),
                       torch.where(near_mode, wz_near, wz_far)], dim=1)
    arrived = near_mode & stop_now
    cmd[arrived] = 0.0
    return cmd, arrived, ex, ey, yaw_err


def is_still() -> torch.Tensor:
    """최근 1초 동안 골반이 STILL_POS·STILL_YAW 넘게 움직이지 않았는지. 상태가 바뀐 뒤 1초는 지나야 함."""
    k = step % STILL_STEPS  # 이 칸에는 1초 전 값이 들어 있음 (이번 스텝 값으로 덮어쓰기 전)
    moved = torch.norm(robot.data.root_pos_w[:, :2] - pos_hist[k], dim=1)
    turned = wrap(robot.data.heading_w - yaw_hist[k]).abs()
    return (t_state >= STILL_STEPS) & (moved < STILL_POS) & (turned < STILL_YAW)


def tilt() -> torch.Tensor:
    return torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)) * RAD2DEG


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
cmd = torch.zeros(N, 3, device=device)
cmd_term.vel_command_b[:] = 0.0
obs = env.get_observations()
step = 0
pick_tilt_max = torch.zeros(N, device=device)
last_pos = robot.data.root_pos_w.clone()
last_yaw = robot.data.heading_w.clone()
last_grav = robot.data.projected_gravity_b.clone()
fall_info = {}  # 환경 번호 → 넘어지기 직전 상태 (문자열)
foot_ids = [robot.body_names.index(f"{s_}_ankle_roll_link") for s_ in ("left", "right")]
last_feet = robot.data.body_pos_w[:, foot_ids].clone()
last_box_t = torch.zeros(N, 3, device=device)
last_box_z = torch.zeros(N, device=device)
last_torso_force = torch.zeros(N, device=device)
contact_sensor = u.scene.sensors["contact_forces"]
torso_sensor_id = contact_sensor.body_names.index("torso_link")

with torch.inference_mode():
    start_move(state == S_OPEN, pelvis_to_world(sym_points(OPEN_POSE)), 75)
    for step in range(max_steps):
        # ---- 팔: 이동 중이면 목표 궤적을 따라가고, 걷는 동안은 관절 목표를 그대로 둠 ----
        if bool(arm_world.any()):  # 월드에 둔 목표는 매 스텝 지금의 몸통 좌표로 다시 바꿈
            arm_goal[arm_world] = torso.to_frame(arm_goal_w0)[arm_world]
        a = torch.clamp((arm_t + 1).float() / arm_T.float(), 0.0, 1.0)
        a = (10 * a**3 - 15 * a**4 + 6 * a**5)[:, None, None]
        ik.step(torso.to_world(arm_start + a * (arm_goal - arm_start)), mask=arm_busy | arm_hold)
        ik.apply()
        arm_t += arm_busy.long()
        move_done = arm_busy & (arm_t >= arm_T)
        arm_busy &= ~move_done
        move_attached_box()

        # ---- 한 스텝 ----
        obs, _, _, _ = env.step(policy(obs_filter(obs)))
        t_state += 1
        terminated = u.termination_manager.terminated
        last_torso_force = contact_sensor.data.net_forces_w_history[:, :, torso_sensor_id].norm(dim=-1).max(dim=1).values
        new_fall = terminated & (state != S_DONE) & (state != S_FAIL)
        if bool(new_fall.any()):
            fell |= new_fall
            for e in new_fall.nonzero().flatten().tolist():
                parts = []
                for name, xy in (("A", a_xy[e]), ("B", b_xy[e])):
                    rel = rot_z(-last_yaw[e], torch.cat([xy - last_pos[e, :2], torch.zeros(1, device=device)]))
                    feet = min(torch.norm(last_feet[e, k, :2] - xy).item() for k in range(2)) * 100
                    parts.append(f"받침대 {name}: 골반 기준 앞 {rel[0].item() * 100:+.0f}·왼쪽 {rel[1].item() * 100:+.0f}cm (가까운 발목까지 {feet:.0f}cm)")
                bt = last_box_t[e] * 100
                fall_info[e] = (f"{STATE_NAMES[int(state[e])]} {t_state[e].item() * step_dt:.1f}초째, {', '.join(parts)}, "
                                f"기울기 앞뒤 {torch.atan2(last_grav[e, 0], -last_grav[e, 2]).item() * RAD2DEG:+.0f}° "
                                f"좌우 {torch.atan2(-last_grav[e, 1], -last_grav[e, 2]).item() * RAD2DEG:+.0f}°, "
                                f"상자(몸통 좌표) ({bt[0].item():+.0f}, {bt[1].item():+.0f}, {bt[2].item():+.0f})cm·높이 {last_box_z[e].item() * 100:.0f}cm, "
                                f"몸통 접촉 힘 {last_torso_force[e].item():.0f}N")
            ik.reset(new_fall.nonzero().flatten())
            fail(new_fall)
        active = (state != S_DONE) & (state != S_FAIL)
        timed_out = torch.zeros_like(active)
        for s_code, limit in TIMEOUT.items():
            timed_out |= (state == s_code) & (t_state > limit)
        fail(timed_out)

        cmd.zero_()
        hand_w = ik.hand_pos_w()
        st = state.clone()  # 한 스텝에 상태가 한 번만 바뀌게, 조건은 이 값으로만 본다

        # 팔 벌리기 → A 로 걷기
        m = (st == S_OPEN) & move_done
        arm_hold[m] = False
        set_state(m, S_WALK_A)

        # A 로 걷기
        m = st == S_WALK_A
        if bool(m.any()):
            c, arrived, ex, ey, yerr = walk_command(stance_a_xy, stance_a_yaw, loaded=False)
            cmd[m] = c[m]
            if args_cli.debug and bool((m & arrived).any()):
                for e in (m & arrived).nonzero().flatten().tolist():
                    print(f"[dbg] t={step:5d} 환경 {e}: A 도착 판정, 오차 앞 {ex[e].item() * 100:+.1f} 옆 {ey[e].item() * 100:+.1f}cm 방향 {yerr[e].item() * RAD2DEG:+.1f}°")
            set_state(m & arrived, S_SETTLE_A)

        # A 앞에서 멈춤 → (자리가 범위 밖이면 다시 다가감) → 손 뻗기
        m = st == S_SETTLE_A
        if bool(m.any()):
            still = m & is_still()
            if bool(still.any()):
                _, _, ex, ey, yerr = walk_command(stance_a_xy, stance_a_yaw, loaded=False)
                in_range = (ex >= STANCE_TOL_X[0]) & (ex <= STANCE_TOL_X[1]) & (ey.abs() <= STANCE_TOL_Y) & (yerr.abs() <= STANCE_TOL_YAW)
                retry = still & ~in_range & (reapproach < MAX_REAPPROACH)
                reapproach[retry] += 1
                set_state(retry, S_WALK_A)
                go = still & ~retry
                rec["stance_a_x"][go], rec["stance_a_y"][go], rec["stance_a_yaw"][go] = ex[go], ey[go], yerr[go]
                start_move(go, box_grasp_points(), 100, world=True)
                set_state(go, S_CLOSE)

        # 손 뻗기 (빈손이라 월드 목표를 계속 따라감) → 손 위치 맞추기
        m = (st == S_CLOSE) & move_done
        set_state(m, S_CLOSE2)

        # 양손이 잡는 위치에 ATTACH_HOLD 스텝 머물면 붙이고 들어 올리기. 제한 시간 안에 못 맞추면 실패
        m = st == S_CLOSE2
        if bool(m.any()):
            err2 = torch.norm(hand_w - box_grasp_points(), dim=-1)
            err = err2.max(dim=1).values
            rec["grasp_err"][m] = err[m]
            rec["box_shift"][m] = torch.norm(box.data.root_pos_w[:, :2] - a_xy, dim=1)[m]
            in_tol_count[m] = torch.where(err <= ATTACH_TOL, in_tol_count + 1, torch.zeros_like(in_tol_count))[m]
            ok = m & (in_tol_count >= ATTACH_HOLD)
            if args_cli.debug:
                torso_id = robot.body_names.index("torso_link")
                coms = robot.root_physx_view.get_coms()[:, torso_id, :3]
                for e in ok.nonzero().flatten().tolist():
                    print(f"[dbg]   (몸통 무게중심 ({coms[e, 0].item() * 100:+.1f}, {coms[e, 1].item() * 100:+.1f})cm)")
                    print(f"[dbg] t={step:5d} 환경 {e}: 붙임, 손 오차 L {err2[e, 0].item() * 100:.1f} R {err2[e, 1].item() * 100:.1f}cm, "
                          f"맞추는 데 {t_state[e].item() * step_dt:.1f}초, 기울기 {tilt()[e].item():.1f}°")
            attach(ok)
            up = hand_w.clone()
            up[..., 2] += LIFT_UP
            start_move(ok, up, 50)  # 여기서부터 손에 무게가 있으므로 목표는 몸통 좌표에 고정
            set_state(ok, S_LIFT1)

        # 들어 올리기 → 든 자세로
        m = (st == S_LIFT1) & move_done
        att_floor[m] = -1.0e9
        start_move(m, pelvis_to_world(sym_points(CARRY_BOX_POSE)), 100)
        set_state(m, S_LIFT2)

        # 든 자세 → 잠깐 유지
        m = (st == S_LIFT2) & move_done
        set_state(m, S_HOLD)
        m = st == S_HOLD
        pick_tilt_max[m | (st == S_LIFT1) | (st == S_LIFT2)] = torch.maximum(pick_tilt_max, tilt())[
            m | (st == S_LIFT1) | (st == S_LIFT2)]
        hold_steps = 100 if args_cli.until == "lift" else 25
        m = m & (t_state >= hold_steps)
        if bool(m.any()):
            rec["t_pick"][m] = step * step_dt
            rec["lift_tilt"][m] = pick_tilt_max[m]
            if args_cli.until == "lift":
                done_step[m] = step
                set_state(m, S_DONE)
            else:
                arm_hold[m] = False  # 걷는 동안은 관절 목표 고정
                set_state(m, S_SIDE)

        # 옆으로 비켜서기: A 를 마주 본 방향을 유지하며 왼쪽으로 SIDESTEP_DIST 만큼
        m = st == S_SIDE
        if bool(m.any()):
            left = torch.stack([-torch.sin(stance_a_yaw), torch.cos(stance_a_yaw)], dim=1)
            moved = ((robot.data.root_pos_w[:, :2] - stance_a_xy) * left).sum(dim=1)
            yaw_err = wrap(stance_a_yaw - robot.data.heading_w)
            cmd[m, 1] = SIDESTEP_SPEED
            cmd[m, 2] = torch.clamp(1.5 * yaw_err, -0.4, 0.4)[m]
            set_state(m & (moved >= SIDESTEP_DIST), S_WALK_B)

        # B 로 걷기
        m = st == S_WALK_B
        if bool(m.any()):
            c, arrived, ex, ey, yerr = walk_command(stance_b_xy, stance_b_yaw, loaded=True)
            cmd[m] = c[m]
            set_state(m & arrived, S_SETTLE_B)

        # B 앞에서 멈춤 → (자리가 범위 밖이면 다시 다가감) → 내려놓기 (상자 중심이 받침대 B 가운데 위로 가도록 양손을 같은 만큼 옮김)
        m = st == S_SETTLE_B
        if bool(m.any()):
            still = m & is_still()
            if bool(still.any()):
                _, _, ex, ey, yerr = walk_command(stance_b_xy, stance_b_yaw, loaded=True)
                in_range = (ex >= STANCE_TOL_X[0]) & (ex <= STANCE_TOL_X[1]) & (ey.abs() <= STANCE_TOL_Y) & (yerr.abs() <= STANCE_TOL_YAW)
                retry = still & ~in_range & (reapproach_b < MAX_REAPPROACH)
                reapproach_b[retry] += 1
                set_state(retry, S_WALK_B)
                go = still & ~retry
                rec["stance_b_x"][go], rec["stance_b_y"][go], rec["stance_b_yaw"][go] = ex[go], ey[go], yerr[go]
                start_move(go, hand_w + (place_target - box.data.root_pos_w)[:, None, :], 125)
                set_state(go, S_PLACE)

        m = (st == S_PLACE) & move_done
        start_move(m, hand_w + (place_target - box.data.root_pos_w)[:, None, :], 40)
        set_state(m, S_PLACE2)

        # 떼기
        m = (st == S_PLACE2) & move_done
        if bool(m.any()):
            rec["place_err_before"][m] = torch.norm(box.data.root_pos_w - place_target, dim=1)[m]
            detach(m)
            set_state(m, S_RELEASE)

        # 떼고 0.3초 뒤 손 벌리기 (양손을 바깥으로 7cm)
        m = (st == S_RELEASE) & (t_state >= 15)
        if bool(m.any()):
            mid, psi = hand_frame()
            out = rot_z(psi[:, None].expand(-1, 2), torch.tensor([[0.0, 0.07, 0.0], [0.0, -0.07, 0.0]], device=device)[None].expand(N, -1, -1))
            start_move(m, hand_w + out, 40)
            set_state(m, S_OPEN_B)

        # 팔 거두기
        m = (st == S_OPEN_B) & move_done
        start_move(m, pelvis_to_world(sym_points(RETRACT_POSE)), 75)
        set_state(m, S_RETRACT)
        m = (st == S_RETRACT) & move_done
        set_state(m, S_END_HOLD)
        m = (st == S_END_HOLD) & (t_state >= 50)
        if bool(m.any()):
            rec["place_xy"][m] = torch.norm(box.data.root_pos_w[:, :2] - b_xy, dim=1)[m]
            rec["place_z"][m] = (box.data.root_pos_w[:, 2] - BOX_Z_ON_TABLE)[m]
            rec["t_total"][m] = step * step_dt
            done_step[m] = step
            set_state(m, S_DONE)

        ok_env = (state != S_DONE) & (state != S_FAIL)
        last_pos[ok_env], last_yaw[ok_env] = robot.data.root_pos_w[ok_env], robot.data.heading_w[ok_env]
        last_grav[ok_env], last_feet[ok_env] = robot.data.projected_gravity_b[ok_env], robot.data.body_pos_w[:, foot_ids][ok_env]
        last_box_t[ok_env] = torso.to_frame(box.data.root_pos_w[:, None, :])[:, 0][ok_env]
        last_box_z[ok_env] = box.data.root_pos_w[ok_env, 2]

        # 멈춤 판정용 기록 (1초 전 값을 읽은 뒤에 덮어씀)
        pos_hist[step % STILL_STEPS] = robot.data.root_pos_w[:, :2]
        yaw_hist[step % STILL_STEPS] = robot.data.heading_w

        # ---- 다음 스텝의 명령 ----
        cmd_term.vel_command_b[:] = cmd
        obs = env.get_observations()

        if args_cli.debug and step % 25 == 0:
            g = robot.data.projected_gravity_b[0]
            p = robot.data.root_pos_w[0] - origins[0]
            bp = box.data.root_pos_w[0] - origins[0]
            print(f"[dbg] t={step:5d} {STATE_NAMES[int(state[0])]:12s} 골반 ({p[0].item():+.2f}, {p[1].item():+.2f}) 방향 {robot.data.heading_w[0].item() * RAD2DEG:+6.1f}° "
                  f"명령 ({cmd[0, 0].item():+.2f}, {cmd[0, 1].item():+.2f}, {cmd[0, 2].item():+.2f}) 기울기 {tilt()[0].item():4.1f}° "
                  f"상자 ({bp[0].item():+.2f}, {bp[1].item():+.2f}, {bp[2].item():.3f}) 붙음 {int(attached[0])}")
        if bool(((state == S_DONE) | (state == S_FAIL)).all()):
            break

# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------
finished = state == S_DONE
if args_cli.until == "full":
    # 성공: 끝까지 진행 + 상자가 받침대 B 위에 있고(높이 차 2cm 이내) 가운데에서 5cm 이내
    success = finished & (rec["place_xy"] <= 0.05) & (rec["place_z"].abs() <= 0.02)
else:
    success = finished
unfinished = (state != S_DONE) & (state != S_FAIL)
fail_state[unfinished] = state[unfinished]


def stat(key: str, scale: float = 100.0, absolute: bool = False):
    v = rec[key][~torch.isnan(rec[key])] * scale
    if len(v) == 0:
        return "—"
    if absolute:
        v = v.abs()
    return f"평균 {v.mean().item():.2f}, 최대 {v.max().item():.2f} ({len(v)}개)"


print(f"\n========== 상자 집어 옮기기 ({'들기까지' if args_cli.until == 'lift' else '전체 순서'}, 환경 {N}개, "
      f"상자 위치 {'무작위 (±10cm, ±15°)' if args_cli.randomize else '고정'}, 상자 {box_mass:.1f}kg) ==========")
print(f"성공: {int(success.sum())}/{N} ({int(success.sum()) / N * 100:.0f}%), 넘어짐: {int(fell.sum())}, 진행 {step + 1}스텝 ({(step + 1) * step_dt:.1f}초)")
if bool((~success).any()):
    print("실패한 곳 (상태별 환경 수):")
    for s_code in range(len(STATE_NAMES)):
        sel = (~success) & (fail_state == s_code)
        if bool(sel.any()):
            print(f"  {STATE_NAMES[s_code]}: {int(sel.sum())}개 (그중 넘어짐 {int((sel & fell).sum())}) — 환경 {sel.nonzero().flatten().tolist()[:12]}")
    for e, info in sorted(fall_info.items()):
        print(f"  넘어짐 환경 {e}: {info}")
    sel = (~success) & finished
    if bool(sel.any()):
        print(f"  끝까지 갔지만 놓인 위치가 기준 밖: {int(sel.sum())}개 — 환경 {sel.nonzero().flatten().tolist()[:12]}")
print(f"받침대 A 앞에 멈춘 위치 오차 (cm): 앞뒤 |{stat('stance_a_x', absolute=True)}|, 좌우 |{stat('stance_a_y', absolute=True)}|, "
      f"방향(°) |{stat('stance_a_yaw', RAD2DEG, True)}|")
print(f"멈춘 자리가 범위 밖이라 다시 다가간 횟수: A 평균 {reapproach.float().mean().item():.2f}·최대 {int(reapproach.max())}, "
      f"B 평균 {reapproach_b.float().mean().item():.2f}·최대 {int(reapproach_b.max())}")
print(f"붙이기 직전 손 위치 오차 (cm, 양손 중 큰 값): {stat('grasp_err')}")
print(f"붙이기 직전까지 상자가 처음 자리에서 밀린 거리 (cm): {stat('box_shift')}")
print(f"붙은 상자가 두 손 기준 자리에서 벗어난 거리 (cm, 한 스텝 뒤, 환경별 최대값): 공중 평균 {box_dev_air.mean().item() * 100:.3f}·최대 {box_dev_air.max().item() * 100:.3f}, "
      f"받침대에 닿은 동안 평균 {box_dev_table.mean().item() * 100:.2f}·최대 {box_dev_table.max().item() * 100:.2f}")
print(f"들어 올리는 동안 골반 기울기 최대 (°): {stat('lift_tilt', 1.0)}")
print(f"들기까지 걸린 시간 (초): {stat('t_pick', 1.0)}")
if args_cli.until == "full":
    print(f"받침대 B 앞에 멈춘 위치 오차 (cm): 앞뒤 |{stat('stance_b_x', absolute=True)}|, 좌우 |{stat('stance_b_y', absolute=True)}|, "
          f"방향(°) |{stat('stance_b_yaw', RAD2DEG, True)}|")
    print(f"떼기 직전 상자 위치 오차 (cm): {stat('place_err_before')}")
    print(f"내려놓은 위치 오차 (cm, 받침대 B 가운데 기준 수평): {stat('place_xy')}")
    print(f"내려놓은 상자 높이 차 (cm): {stat('place_z', absolute=True)}")
    print(f"전체 걸린 시간 (초): {stat('t_total', 1.0)}")
masses = robot.root_physx_view.get_masses()[:, palm_ids]
print(f"끝난 뒤 손 질량 (kg): 최소 {masses.min().item():.3f}, 최대 {masses.max().item():.3f} (기본 {default_mass[0, palm_ids[0]].item():.3f})")
if args_cli.video:
    print(f"[pick] 영상: {os.path.join('logs', 'rsl_rl', agent_cfg.experiment_name, os.path.basename(os.path.dirname(resume_path)), 'videos', 'pick_place', args_cli.name + '-step-0.mp4')}")

env.close()
simulation_app.close()
