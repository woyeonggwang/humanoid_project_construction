"""5단계 확인용: IK 로 양손을 목표 위치로 보내기.

--mode grid  (1단계) 몸을 공중에 고정하고, 골반 기준 목표 위치 격자마다 환경 하나씩 배정해 닿는 범위를 잰다.
             다리는 0 동작(기본 자세). 손 주변 링크 배치도 함께 출력한다.
--mode stand (2단계) 걷기 정책에 정지 명령을 주고 서 있는 상태에서 짐 든 자세 ↔ 뻗은 자세를 반복한다.
             넘어짐과 손 위치 오차를 잰다. --hand_mass 로 양손에 무게를 더한다.

목표 위치는 '골반 위치 + 골반이 향한 방향(yaw)만 반영한 좌표' 기준 (x 앞, y 왼쪽, z 위, m).
예) python -u scripts\\ik_check.py --mode grid --headless
    python -u scripts\\ik_check.py --mode stand --hand_mass 3.0 --headless
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check arm IK reach (fixed base grid / standing with the walking policy).")
parser.add_argument("--mode", type=str, default="grid", choices=["grid", "stand"])
parser.add_argument("--task", type=str, default="Humanoid-G1-ArmIK-v0")
parser.add_argument("--command_type", type=str, default="position", choices=["position", "pose"],
                    help="position: 위치만, pose: 위치 + 처음 손 방향 유지 (grid 모드에서 비교용)")
parser.add_argument("--lambda_val", type=float, default=0.05)
parser.add_argument("--gain", type=float, default=0.3)
# stand 모드
parser.add_argument("--num_envs", type=int, default=32, help="stand 모드의 환경 수")
parser.add_argument("--hand_mass", type=float, default=0.0, help="stand 모드: 양손에 더하는 무게 (kg, 손 하나당)")
parser.add_argument("--reach", type=str, default="0.30,0.24,0.125",
                    help="stand 모드: 뻗을 위치 'x,y,z' (y 는 좌우 대칭 거리). 기본값 = 상자 잡는 위치")
parser.add_argument("--cycles", type=int, default=3)
parser.add_argument("--debug", action="store_true", help="stand 모드: 환경 0·1 의 상태를 10스텝마다 출력")
parser.add_argument("--frame", type=str, default="torso", choices=["torso", "yaw"],
                    help="stand 모드: 이동을 시작할 때 목표를 어디에 고정할지. torso = 몸통 좌표(몸이 기울어도 팔 목표 유지), "
                         "yaw = 수평을 유지하는 골반 좌표(비교용)")
parser.add_argument("--prewalk", type=float, default=2.0,
                    help="stand 모드: 팔을 움직이기 전에 앞으로 0.3m/s 로 걷는 시간(초). 걸어와서 멈춘 뒤 팔을 뻗는 실제 순서를 흉내 냄. "
                         "0 이면 리셋 직후의 발 위치에서 바로 시작")
parser.add_argument("--no_obs_filter", action="store_true",
                    help="stand 모드: 학습 때 고정이던 팔 관절(어깨 roll·yaw, 팔꿈치 roll)을 정책에 그대로 보여 줌 (비교용)")
parser.add_argument("--load_run", type=str, default="2026-11-20_20-54-06_armrand_scratch")
parser.add_argument("--checkpoint", type=str, default="model_3999.pt")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_error_magnitude, yaw_quat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from humanoid_tasks.g1_carry.arm_ik import ARM_JOINTS, SIDES, DualArmIK, PolicyObsFilter, TorsoFrame  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

RAD2DEG = 57.29578


def min_jerk(s: float) -> float:
    """0 → 1 로 부드럽게 (처음·끝 속도 0)."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def to_world(robot, pos_b: torch.Tensor) -> torch.Tensor:
    """골반 yaw 좌표 (N, K, 3) → 월드."""
    quat = yaw_quat(robot.data.root_quat_w)[:, None, :].expand(-1, pos_b.shape[1], -1)
    return robot.data.root_pos_w[:, None, :] + quat_apply(quat.reshape(-1, 4), pos_b.reshape(-1, 3)).reshape(pos_b.shape)


def to_base(robot, pos_w: torch.Tensor) -> torch.Tensor:
    """월드 (N, K, 3) → 골반 yaw 좌표."""
    quat = yaw_quat(robot.data.root_quat_w)[:, None, :].expand(-1, pos_w.shape[1], -1)
    rel = pos_w - robot.data.root_pos_w[:, None, :]
    return quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(pos_w.shape)


# ===========================================================================
# 1단계: 몸 고정, 격자
# ===========================================================================
def run_grid():
    xs = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    zs = [0.25, 0.20, 0.15, 0.10, 0.05, 0.00, -0.05, -0.10]
    ys = [0.16, 0.20, 0.24]
    grid = [(x, y, z) for y in ys for z in zs for x in xs]
    n = len(grid)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    env_cfg.seed = args_cli.seed
    spawn = env_cfg.scene.robot.spawn
    env_cfg.scene.robot.spawn = spawn.replace(articulation_props=spawn.articulation_props.replace(fix_root_link=True))
    # 앞(+x)을 보고 시작 → 월드 좌표 경계 상자를 골반 좌표로 바로 읽을 수 있음
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    device = u.device
    print(f"[ik] 몸 고정: {robot.is_fixed_base}, 환경 {n}개 (격자 x {len(xs)} × z {len(zs)} × y {len(ys)})")
    print(f"[ik] 자코비안 크기 {tuple(robot.root_physx_view.get_jacobians().shape)} (환경, 링크, 6, 자유도), "
          f"링크 {robot.num_bodies}개, 관절 {robot.num_joints}개")

    ik = DualArmIK(robot, command_type=args_cli.command_type, lambda_val=args_cli.lambda_val, gain=args_cli.gain)
    print(f"[ik] 팔 관절 번호: 왼쪽 {ik.joint_ids[0]}, 오른쪽 {ik.joint_ids[1]}, 손 링크 번호 {ik.body_ids}")

    # ---- 손 주변 링크 배치 (환경 0, 짐 든 자세) ----
    print("\n========== 손 주변 링크 (환경 0, 짐 든 자세, 골반 yaw 좌표 cm) ==========")
    names = [nm for nm in robot.body_names if nm.startswith("left_") and any(
        k in nm for k in ("palm", "zero", "one", "two", "three", "four", "five", "six", "elbow"))]
    ids = [robot.body_names.index(nm) for nm in names]
    pos_b = to_base(robot, robot.data.body_pos_w[:, ids])[0] * 100
    for nm, p in zip(names, pos_b.tolist()):
        print(f"  {nm:28s} x {p[0]:+6.1f}  y {p[1]:+6.1f}  z {p[2]:+6.1f}")
    rot = matrix_from_quat(robot.data.body_quat_w[0, ik.body_ids])  # (2, 3, 3) 손 좌표축을 월드로
    base_rot = matrix_from_quat(yaw_quat(robot.data.root_quat_w[:1]))[0]
    for k, side in enumerate(SIDES):
        axes = base_rot.T @ rot[k]
        print(f"  {side} palm 좌표축 (골반 yaw 좌표에서 본 방향): "
              f"x축 {[round(v, 2) for v in axes[:, 0].tolist()]}, y축 {[round(v, 2) for v in axes[:, 1].tolist()]}, "
              f"z축 {[round(v, 2) for v in axes[:, 2].tolist()]}")

    # 손(손바닥 + 손가락) 크기: 링크별 경계 상자(보이는 형상 기준)를 골반 좌표로. 상자 옆면에서 얼마나 띄울지 정하는 데 씀
    # USD 의 링크 자세는 시뮬레이션 중 갱신되지 않으므로(관절 0 자세 그대로) 월드 좌표가 아니라 palm 링크 좌표로 잰다.
    # 손가락은 관절 0 자세 기준이라 실제(기본 손가락 자세)와 조금 다를 수 있음
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    print("\n========== 손 링크 경계 상자 (palm 링크 좌표 cm, 보이는 형상 기준, 손가락은 관절 0 자세) ==========")
    for side in SIDES:
        palm_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{side}_palm_link")
        lo_all, hi_all = [1e9] * 3, [-1e9] * 3
        for nm in [x.replace("left_", f"{side}_") for x in names if "elbow" not in x]:
            rng = cache.ComputeRelativeBound(stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{nm}"), palm_prim).ComputeAlignedRange()
            lo, hi = [v * 100 for v in rng.GetMin()], [v * 100 for v in rng.GetMax()]
            if side == "left":
                print(f"  {nm:18s} x {lo[0]:+6.1f}~{hi[0]:+6.1f}  y {lo[1]:+6.1f}~{hi[1]:+6.1f}  z {lo[2]:+6.1f}~{hi[2]:+6.1f}")
            lo_all = [min(a, b) for a, b in zip(lo_all, lo)]
            hi_all = [max(a, b) for a, b in zip(hi_all, hi)]
        print(f"  {side} 손 전체       x {lo_all[0]:+6.1f}~{hi_all[0]:+6.1f}  y {lo_all[1]:+6.1f}~{hi_all[1]:+6.1f}  z {lo_all[2]:+6.1f}~{hi_all[2]:+6.1f}")

    g = torch.tensor(grid, device=device)
    target_b = torch.stack([g, g * torch.tensor([1.0, -1.0, 1.0], device=device)], dim=1)  # (N, 2, 3) 왼손·오른손
    start_b = to_base(robot, ik.hand_pos_w())
    quat0 = ik.hand_quat_w().clone()
    q0 = robot.data.joint_pos[:, ik.all_joint_ids].clone()
    print(f"\n[ik] 시작 손 위치 (환경 0, cm): 왼손 {[round(v * 100, 1) for v in start_b[0, 0].tolist()]}, "
          f"오른손 {[round(v * 100, 1) for v in start_b[0, 1].tolist()]}")

    zero_action = torch.zeros(n, u.action_manager.total_action_dim, device=device)
    t_move, t_hold = 125, 75
    with torch.inference_mode():
        for t in range(t_move + t_hold):
            a = min_jerk((t + 1) / t_move)
            tgt_w = to_world(robot, start_b + a * (target_b - start_b))
            ik.step(tgt_w, target_quat_w=quat0)
            ik.apply()
            env.step(zero_action)
        hand_b = to_base(robot, ik.hand_pos_w())
        err = torch.norm(hand_b - target_b, dim=-1) * 100  # (N, 2) cm
        rot_change = quat_error_magnitude(ik.hand_quat_w().reshape(-1, 4), quat0.reshape(-1, 4)).reshape(n, 2) * RAD2DEG
        q = robot.data.joint_pos[:, ik.all_joint_ids]
        lim = robot.data.soft_joint_pos_limits[:, ik.all_joint_ids]
        at_limit = ((ik.q_cmd - lim[..., 0]).abs() < 1e-3) | ((ik.q_cmd - lim[..., 1]).abs() < 1e-3)

    err_max = err.max(dim=1).values.cpu()
    rot_max = rot_change.max(dim=1).values.cpu()
    lim_any = at_limit.any(dim=1).cpu()
    print(f"\n========== 닿는 범위 ({args_cli.command_type} 모드, 몸 고정, {t_move}스텝 이동 + {t_hold}스텝 유지 뒤) ==========")
    print("칸 = 손 위치 오차 cm (양손 중 큰 값). * = 관절 목표값이 한계에 닿음. 목표는 골반 yaw 좌표, 좌우 대칭")
    i = 0
    for y in ys:
        print(f"\n  y = ±{y * 100:.0f}cm        " + "".join(f"x={x * 100:4.0f} " for x in xs))
        for z in zs:
            cells = []
            for _ in xs:
                cells.append(f"{err_max[i].item():5.1f}{'*' if lim_any[i] else ' '} ")
                i += 1
            print(f"    z = {z * 100:+5.0f}cm   " + "".join(cells))
    ok = err_max <= 2.0
    print(f"\n[ik] 오차 2cm 이내: {int(ok.sum())}/{n}칸")
    for y in ys:
        sel = [(gx, gz) for (gx, gy, gz), o in zip(grid, ok.tolist()) if gy == y and o]
        if sel:
            print(f"  y=±{y * 100:.0f}cm: x {min(s[0] for s in sel) * 100:.0f}~{max(s[0] for s in sel) * 100:.0f}cm, "
                  f"z {min(s[1] for s in sel) * 100:+.0f}~{max(s[1] for s in sel) * 100:+.0f}cm 안의 {len(sel)}칸")
    print(f"[ik] 손 방향 변화 (처음 대비, 오차 2cm 이내 칸): 평균 {rot_max[ok].mean():.1f}°, 최대 {rot_max[ok].max():.1f}°")
    print("\n[ik] 관절 목표값 범위 (오차 2cm 이내 칸, rad) / 부드러운 한계")
    for j, nm in enumerate(ARM_JOINTS):
        qc = ik.q_cmd[ok.to(device)][:, j]
        print(f"  left_{nm:15s} {qc.min().item():+.2f} ~ {qc.max().item():+.2f}   한계 {lim[0, j, 0].item():+.2f} ~ {lim[0, j, 1].item():+.2f}"
              f"   시작 {q0[0, j].item():+.2f}")
    del q
    env.close()


# ===========================================================================
# 2단계: 서 있는 상태 (걷기 정책 + 정지 명령)
# ===========================================================================
def run_stand():
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    n = args_cli.num_envs
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped
    robot = u.scene["robot"]
    device = u.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)
    print(f"[ik] 체크포인트: {os.path.join(os.path.basename(os.path.dirname(resume_path)), os.path.basename(resume_path))}")

    # 손 무게 (eval_carry.py 와 같은 방식: 기본 질량 + 추가, 관성은 질량 비율만큼)
    ik = DualArmIK(robot, command_type="position", lambda_val=args_cli.lambda_val, gain=args_cli.gain)
    view = robot.root_physx_view
    cpu_ids = torch.arange(n)
    default_mass, default_inertia = robot.data.default_mass.cpu(), robot.data.default_inertia.cpu()
    masses = view.get_masses().clone()
    masses[:, ik.body_ids] = default_mass[:, ik.body_ids] + args_cli.hand_mass
    view.set_masses(masses, cpu_ids)
    inertias = view.get_inertias().clone()
    inertias[:, ik.body_ids] = default_inertia[:, ik.body_ids] * (masses[:, ik.body_ids] / default_mass[:, ik.body_ids])[..., None]
    view.set_inertias(inertias, cpu_ids)
    print(f"[ik] 손 질량: {view.get_masses()[0, ik.body_ids].tolist()} kg (더한 무게 {args_cli.hand_mass})")

    obs_filter = (lambda o: o) if args_cli.no_obs_filter else PolicyObsFilter(u, robot)
    print(f"[ik] 정책 관측 거르기: {'끔' if args_cli.no_obs_filter else '켬 (어깨 roll·yaw, 팔꿈치 roll 을 기본 자세 값으로)'}")
    rx, ry, rz = (float(v) for v in args_cli.reach.split(","))
    reach_b = torch.tensor([[rx, ry, rz], [rx, -ry, rz]], device=device).expand(n, 2, 3)
    cmd_term = u.command_manager.get_term("base_velocity")
    cmd_term.vel_command_b[:] = 0.0
    obs = env.get_observations()

    settle, t_carry, t_move, t_hold = 100, 100, 125, 75
    falls = torch.zeros(n, dtype=torch.long, device=device)
    hold_err, move_err, tilt_max = [], [], torch.zeros(n, device=device)
    root0 = None

    step_count = 0
    cmd_vx = 0.0  # 걷기 정책에 주는 앞뒤 속도 명령 (팔을 움직이는 동안은 0 = 정지)
    torso = TorsoFrame(robot)
    use_torso = args_cli.frame == "torso"

    def fix_target(pos_b):
        """골반 yaw 좌표의 위치를 지금 자세 기준으로 몸통 좌표(또는 그대로)에 고정한다."""
        return torso.to_frame(to_world(robot, pos_b)).clone() if use_torso else pos_b.clone()

    def target_world(pos_f):
        return torso.to_world(pos_f) if use_torso else to_world(robot, pos_f)

    def hand_in_frame():
        return torso.to_frame(ik.hand_pos_w()) if use_torso else to_base(robot, ik.hand_pos_w())

    def step_once(target_f):
        nonlocal obs, step_count
        if target_f is not None:
            ik.step(target_world(target_f))
        ik.apply()
        obs, _, _, _ = env.step(policy(obs_filter(obs)))
        done = u.termination_manager.terminated
        falls.add_(done.long())
        step_count += 1
        if args_cli.debug and (step_count % 10 == 0 or bool(done[:4].any())):
            hf = hand_in_frame()
            for e in range(2):
                err = (torch.norm(hf[e] - target_f[e], dim=-1) * 100).tolist() if target_f is not None else [0.0, 0.0]
                g = robot.data.projected_gravity_b[e]
                print(f"[dbg] t={step_count:4d} env{e} 넘어짐={int(done[e])} 오차 {err[0]:5.1f} {err[1]:5.1f}cm "
                      f"기울기 앞뒤 {torch.atan2(g[0], -g[2]).item() * RAD2DEG:+5.1f}° 좌우 {torch.atan2(-g[1], -g[2]).item() * RAD2DEG:+5.1f}° "
                      f"목표값 L {[round(v, 2) for v in ik.q_cmd[e, :5].tolist()]} R {[round(v, 2) for v in ik.q_cmd[e, 5:].tolist()]} "
                      f"실제 L {[round(v, 2) for v in robot.data.joint_pos[e, ik.joint_ids[0]].tolist()]} "
                      f"허리 {robot.data.joint_pos[e, robot.joint_names.index('torso_joint')].item():+.2f} "
                      f"골반 z {robot.data.root_pos_w[e, 2].item():.3f}")
        reset_ids = (u.episode_length_buf == 0).nonzero().flatten()
        if len(reset_ids) > 0:
            ik.reset(reset_ids)
        cmd_term.vel_command_b[:] = 0.0
        cmd_term.vel_command_b[:, 0] = cmd_vx
        obs = env.get_observations()
        grav = robot.data.projected_gravity_b
        tilt_max.copy_(torch.maximum(tilt_max, torch.acos(torch.clamp(-grav[:, 2], -1.0, 1.0))))

    world_err = []  # 이동을 시작할 때 정한 월드 목표와 유지 구간 끝의 손 위치 차이 (몸이 움직인 몫 포함)

    def move_and_hold(name, c, goal_b, t_hold_steps):
        """지금 손 위치 → goal_b(골반 yaw 좌표) 로 이동한 뒤 유지. 목표는 이동을 시작할 때 고정."""
        goal_w0 = to_world(robot, goal_b).clone()
        start_f, goal_f = hand_in_frame().clone(), fix_target(goal_b)
        worst = torch.zeros(n, 2, device=device)
        for t in range(t_move):
            tgt = start_f + min_jerk((t + 1) / t_move) * (goal_f - start_f)
            step_once(tgt)
            worst = torch.maximum(worst, torch.norm(hand_in_frame() - tgt, dim=-1))
        move_err.append(worst)
        for t in range(t_hold_steps):
            step_once(goal_f)
        hold_err.append((name, c, torch.norm(hand_in_frame() - goal_f, dim=-1)))
        world_err.append((name, c, torch.norm(ik.hand_pos_w() - goal_w0, dim=-1)))

    with torch.inference_mode():
        for _ in range(settle):  # 처음 자리 잡기 (팔은 짐 든 자세 목표값 그대로)
            step_once(None)
        if args_cli.prewalk > 0:  # 걸어와서 멈춘 뒤 시작 (발 위치를 정책이 스스로 정한 상태)
            cmd_vx = 0.3
            for _ in range(int(args_cli.prewalk / u.step_dt)):
                step_once(None)
            cmd_vx = 0.0
            # 완전히 멈출 때까지 기다림: 모든 환경의 골반 수평 속도가 0.5초 동안 0.05m/s 미만 (최대 8초)
            still = 0
            for waited in range(400):
                step_once(None)
                slow = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1) < 0.05
                still = still + 1 if bool(slow.all()) else 0
                if still >= 25:
                    break
            print(f"[ik] 걷기를 멈춘 뒤 {(waited + 1) * u.step_dt:.1f}초 만에 모든 환경이 멈춤 (0.5초 동안 0.05m/s 미만)")
            falls.zero_()
            tilt_max.zero_()
        carry_b = to_base(robot, ik.hand_pos_w()).clone()
        root0 = robot.data.root_pos_w.clone()
        print(f"[ik] 짐 든 자세 손 위치 (평균, cm): {[round(v * 100, 1) for v in carry_b[:, 0].mean(0).tolist()]}, "
              f"뻗을 위치: {[rx * 100, ry * 100, rz * 100]}, 목표 고정: {'몸통 좌표' if use_torso else '골반 yaw 좌표'}, "
              f"시작 전 걷기: {args_cli.prewalk}초")
        for c in range(args_cli.cycles):
            move_and_hold("뻗은 자세", c, reach_b, t_hold)
            move_and_hold("짐 든 자세", c, carry_b, t_carry)
        drift = torch.norm(robot.data.root_pos_w[:, :2] - root0[:, :2], dim=1)

    total_steps = args_cli.cycles * (t_carry + 2 * t_move + t_hold)
    print(f"\n========== 서 있는 상태에서 팔 뻗기 (환경 {n}개, 손 +{args_cli.hand_mass}kg, {total_steps}스텝 = {total_steps * u.step_dt:.0f}초) ==========")
    print(f"넘어짐: {int(falls.sum())}회, 넘어진 환경 {int((falls > 0).sum())}/{n}")
    print("유지 구간 끝의 손 위치 오차 (cm): 평균 / 최대 (환경 × 양손) | 이동 시작 때 정한 월드 목표와의 차이: 평균 / 최대")
    for (name, c, e), (_, _, w) in zip(hold_err, world_err):
        print(f"  {c + 1}회차 {name}: {e.mean().item() * 100:.2f} / {e.max().item() * 100:.2f}"
              f" | {w.mean().item() * 100:.2f} / {w.max().item() * 100:.2f}")
    me = torch.stack(move_err)
    print(f"이동 중 목표 궤적과의 차이 (cm): 평균 {me.mean().item() * 100:.2f}, 최대 {me.max().item() * 100:.2f}")
    print(f"골반 기울기 최대: 평균 {tilt_max.mean().item() * RAD2DEG:.1f}°, 가장 큰 환경 {tilt_max.max().item() * RAD2DEG:.1f}°")
    print(f"처음 대비 골반 수평 이동: 평균 {drift.mean().item() * 100:.1f}cm, 최대 {drift.max().item() * 100:.1f}cm")
    worst_hold = max(e.max().item() for _, _, e in hold_err) * 100
    worst_world = max(w.max().item() for _, _, w in world_err) * 100
    print(f"[ik] 통과 기준 (넘어짐 0, 손 위치 오차 3cm 이내): 넘어짐 {int(falls.sum())}, 유지 구간 최대 오차 {worst_hold:.2f}cm, "
          f"월드 목표 기준 최대 {worst_world:.2f}cm → {'통과' if int(falls.sum()) == 0 and worst_world <= 3.0 else '미통과'}")
    env.close()


if args_cli.mode == "grid":
    run_grid()
else:
    run_stand()
simulation_app.close()
