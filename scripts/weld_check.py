"""11단계 점검: 용접 토치와 닿는 범위.

--mode reach  몸을 공중에 고정한 로봇을 (용접선 × 서는 자리 후보)마다 하나씩 두고, 토치 끝을 용접선 따라 움직여 본다.
              철골은 좌표로만 다룬다 (로봇마다 '그 자리에 섰을 때 철골이 있을 곳'을 계산). 위치 오차·토치 각도·철골과의 거리를 잰다.
              서는 자리: 철골 가운데 면에서의 거리 D, 용접선에서 옆으로 L(월드 y), 철골 정면에서 돌아선 각도.
--mode hand   토치를 쥔 손을 가까이에서 찍는다 (토치 위치·손가락 자세 확인용 사진).

예) python -u scripts\\weld_check.py --mode reach --headless
    python -u scripts\\weld_check.py --mode hand --out data\\weld\\hand --headless --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the welding torch and its reach.")
parser.add_argument("--mode", choices=["reach", "hand", "pose"], default="reach")
parser.add_argument("--poses", type=str, default=None, help="pose: 오른팔 관절값 후보 'a,b,c,d,e;a,b,c,d,e' (어깨 pitch·roll·yaw, 팔꿈치 pitch·roll)")
parser.add_argument("--task", type=str, default="Humanoid-G1-Weld-Probe-v0")
parser.add_argument("--stand_height", type=float, default=0.743, help="reach: 서 있을 때의 골반 높이 (몸을 고정한 높이와의 차이만큼 철골을 옮김). weld_eval.py 로 잰 값")
parser.add_argument("--stand_pitch", type=float, default=-5.0, help="reach: 서 있을 때의 골반 앞뒤 기울기 (도, - = 뒤로 젖힘). weld_eval.py 로 잰 값")
parser.add_argument("--dists", type=str, default="0.30,0.34,0.38,0.42,0.46,0.50")
parser.add_argument("--lats", type=str, default="-0.40,-0.30,-0.20,-0.10,0,0.10,0.20,0.30,0.40")
parser.add_argument("--yaws", type=str, default="-40,-20,0,20,40")
parser.add_argument("--seams", type=str, default=None, help="reach: 볼 용접선 (쉼표로. 기본 = 전부)")
parser.add_argument("--top", type=int, default=6, help="reach: 용접선마다 출력할 후보 수")
parser.add_argument("--no_dir", action="store_true", help="reach: 토치 방향은 맞추지 않고 위치만")
parser.add_argument("--out", type=str, default=os.path.join("data", "weld", "hand"))
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.mode == "hand":
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from humanoid_tasks.g1_carry import site_layout as sl  # noqa: E402
from humanoid_tasks.g1_carry import weld as wd  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

RAD2DEG = 57.29578


def make_env(num_envs: int, render: bool = False):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed
    spawn = env_cfg.scene.robot.spawn
    env_cfg.scene.robot.spawn = spawn.replace(articulation_props=spawn.articulation_props.replace(fix_root_link=True))
    pitch = math.radians(args_cli.stand_pitch) if args_cli.mode == "reach" else 0.0
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0), "pitch": (pitch, pitch)}
    if render:
        env_cfg.viewer.resolution = (960, 720)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if render else None)
    env.reset()
    return env


def run_reach():
    seams = sl.weld_seam_frames()
    names = [s.strip() for s in args_cli.seams.split(",")] if args_cli.seams else list(seams)
    dists = [float(v) for v in args_cli.dists.split(",")]
    lats = [float(v) for v in args_cli.lats.split(",")]
    yaws = [float(v) for v in args_cli.yaws.split(",")]
    combos = [(nm, d, lat, yw) for nm in names for d in dists for lat in lats for yw in yaws]
    n = len(combos)
    env = make_env(n)
    u = env.unwrapped
    robot = u.scene["robot"]
    dev = u.device
    grav = robot.data.projected_gravity_b[0]
    print(f"[weld] 몸 고정 {robot.is_fixed_base}, 환경 {n}개 = 용접선 {len(names)} × 거리 {len(dists)} × 옆 {len(lats)} × 방향 {len(yaws)}, "
          f"골반 기울기 {math.degrees(math.atan2(float(grav[0]), -float(grav[2]))):+.1f}° (- = 뒤로), 서 있을 때 높이 {args_cli.stand_height}m")

    # ---- 환경마다: 로봇은 환경 원점에서 +x 를 보고 고정. 그 자리에 섰을 때의 철골 자리·용접선을 계산 ----
    root = robot.data.root_pos_w.clone()
    f_pos, f_yaw = torch.zeros(n, 3, device=dev), torch.zeros(n, device=dev)
    p_start, p_end, normal = torch.zeros(n, 3, device=dev), torch.zeros(n, 3, device=dev), torch.zeros(n, 3, device=dev)
    for i, (nm, d, lat, yw) in enumerate(combos):
        sm = seams[nm]
        mid_y = 0.5 * (sm["start"][1] + sm["end"][1])
        yaw_st = math.pi + math.radians(yw)  # 철골 정면(-x)을 보는 방향에서 yw 만큼 돌아섬
        px, py = sl.STEEL_FRAME_POS[0] + d, mid_y + lat
        c, s = math.cos(-yaw_st), math.sin(-yaw_st)

        def to_env(p, point=True, c=c, s=s, px=px, py=py):
            x, y = (p[0] - px, p[1] - py) if point else (p[0], p[1])
            return [c * x - s * y, s * x + c * y, p[2]]

        z_shift = float(root[i, 2]) - args_cli.stand_height  # 고정한 골반 높이와 서 있을 때 높이의 차이
        base = torch.tensor([float(root[i, 0]), float(root[i, 1]), z_shift], device=dev)
        f_pos[i] = base + torch.tensor(to_env(sl.STEEL_FRAME_POS), device=dev)
        f_yaw[i] = math.radians(sl.STEEL_FRAME_YAW_DEG) - yaw_st
        p_start[i] = base + torch.tensor(to_env(sm["start"]), device=dev)
        p_end[i] = base + torch.tensor(to_env(sm["end"]), device=dev)
        normal[i] = torch.tensor(to_env(sm["normal"], point=False), device=dev)

    ik = wd.TorchIK(robot)
    clear = wd.FrameClearance(robot, f_pos, f_yaw)
    r_ids = ik.joint_ids
    l_ids = [robot.joint_names.index(j) for j in wd.LEFT_ARM_JOINTS]
    grip_ids, grip_q = wd.grip_targets(robot)
    q_cmd = robot.data.default_joint_pos[:, r_ids].clone()
    q_left = torch.tensor([wd.LEFT_SIDE_POSE], device=dev).repeat(n, 1)
    act = torch.zeros(n, u.action_manager.total_action_dim, device=dev)

    # 세로선은 양 끝을 조금 줄여 위·아래 플랜지와 닿지 않게 한다
    is_v_flat = torch.tensor([seams[c[0]]["kind"] == "v" for c in combos], device=dev)
    seam_dir = p_end - p_start
    seam_len = torch.norm(seam_dir, dim=1, keepdim=True)
    seam_dir = seam_dir / seam_len
    is_v = is_v_flat[:, None]
    trim = torch.where(is_v, torch.full_like(seam_len, wd.WELD_END_TRIM_V), torch.zeros_like(seam_len))
    w_start, w_end = p_start + trim * seam_dir, p_end - trim * seam_dir
    off, gap = wd.WELD_OFFSET, wd.WELD_GAP
    tilt = torch.where(is_v_flat, torch.full((n,), wd.WELD_TILT_V, device=dev), torch.zeros(n, device=dev))

    def new_stats():
        big = torch.full((n,), 1.0e3, device=dev)
        return {"err": torch.zeros(n, device=dev), "ang": torch.zeros(n, device=dev), "torch": big.clone(), "hand": big.clone(),
                "body": big.clone(), "lim": torch.zeros(n, dtype=torch.bool, device=dev)}

    def observe(stats, target=None, axis_des=None):
        if target is not None:
            stats["err"] = torch.maximum(stats["err"], torch.norm(ik.tip_pos_w() - target, dim=1))
        if axis_des is not None:
            ang = torch.acos(torch.clamp((ik.axis_w() * axis_des).sum(dim=1), -1.0, 1.0)) * RAD2DEG
            stats["ang"] = torch.maximum(stats["ang"], ang)
        stats["torch"] = torch.minimum(stats["torch"], clear.torch_clearance())
        stats["hand"] = torch.minimum(stats["hand"], clear.hand_clearance())
        stats["body"] = torch.minimum(stats["body"], clear.body_clearance())

    def apply():
        robot.set_joint_position_target(q_cmd, joint_ids=r_ids)
        robot.set_joint_position_target(q_left, joint_ids=l_ids)
        robot.set_joint_position_target(grip_q.expand(n, -1), joint_ids=grip_ids)
        env.step(act)

    q_tab = None  # 기준 자세 (거치는 점, 가운데, 시작, 끝). 1차에서 구해 2차에서 쓴다

    def nominal(target):
        if q_tab is None:
            return None
        half = (0.5 * torch.norm(w_end - w_start, dim=1)).clamp(min=1.0e-6)
        rel = target - (mid + gap * normal)
        sc = torch.clamp((rel * seam_dir).sum(dim=1) / half, -1.0, 1.0)[:, None]
        out = torch.clamp((rel * normal).sum(dim=1) / (wd.WELD_FAR - gap), 0.0, 1.0)[:, None]
        far, q_mid, q_start, q_end_ = q_tab
        on_seam = torch.where(sc < 0, q_mid + (-sc) * (q_start - q_mid), q_mid + sc * (q_end_ - q_mid))
        return on_seam + out * (far - q_mid)

    def track(p_from, p_to, steps, stats=None, smooth=True, err=True):
        """토치 끝을 p_from → p_to 로 (IK). stats 를 주면 오차·각도·거리를 기록한다."""
        nonlocal q_cmd
        axis_des = None if args_cli.no_dir else -normal
        prev = p_from
        for k in range(steps):
            a = (k + 1) / steps
            if smooth:
                a = 10 * a**3 - 15 * a**4 + 6 * a**5
            target = p_from + a * (p_to - p_from)
            q_cmd = ik.solve(q_cmd, target, axis_des, feed=target - prev, q_nom=nominal(target))
            prev = target
            if stats is not None:
                stats["lim"] |= ((q_cmd <= ik.q_min + 1e-4) | (q_cmd >= ik.q_max - 1e-4)).any(dim=1)
            apply()
            if stats is not None:
                observe(stats, target if err else None, axis_des if err else None)

    def joint_move(q_to, steps, stats=None):
        nonlocal q_cmd
        q_from = q_cmd.clone()
        for k in range(steps):
            a = (k + 1) / steps
            a = 10 * a**3 - 15 * a**4 + 6 * a**5
            q_cmd = q_from + a * (q_to - q_from)
            apply()
            if stats is not None:
                observe(stats)

    ready = torch.tensor([wd.READY_Q], device=dev).repeat(n, 1)
    mid = 0.5 * (w_start + w_end)
    p_far = 0.5 * (p_start + p_end) + wd.WELD_FAR * normal
    S = wd.W_STEPS
    with torch.inference_mode():
        for _ in range(25):
            apply()
        # ---- 1차: 기본 자세에서 용접선 가운데로 바로 가서 기준 자세를 구한다 (가운데 → 시작 → 끝 → 가운데 → 거치는 점) ----
        track(ik.tip_pos_w().clone(), mid + off * normal, 120)
        track(mid + off * normal, mid + gap * normal, 60)
        track(mid + gap * normal, mid + gap * normal, 20)
        tab_mid = q_cmd.clone()
        track(mid + gap * normal, w_start + gap * normal, 100)
        track(w_start + gap * normal, w_start + gap * normal, 20)
        tab_start = q_cmd.clone()
        track(w_start + gap * normal, w_end + gap * normal, 200)
        track(w_end + gap * normal, w_end + gap * normal, 20)
        tab_end = q_cmd.clone()
        track(w_end + gap * normal, mid + gap * normal, 100)
        track(mid + gap * normal, mid + off * normal, 40)
        track(mid + off * normal, p_far, 80)
        track(p_far, p_far, 30)
        seed = q_cmd.clone()
        q_tab = (seed, tab_mid, tab_start, tab_end)
        # ---- 2차: 용접 상태 머신과 같은 순서 (준비 자세 → 팔 뻗기 → 다가가기 → 띄운 점 → 가까이 → 시작점 → 용접 → 가운데 → 빼기 → 물러나기 → 거두기) ----
        joint_move(ready, 100)
        st_move, st_near, st_weld = new_stats(), new_stats(), new_stats()
        joint_move(ready + wd.SEED_FRAC * (seed - ready), S[wd.W_SEED], st_move)
        track(ik.tip_pos_w().clone(), p_far, S[wd.W_FAR_IN], st_move, err=False)
        track(p_far, mid + off * normal, S[wd.W_OFF_IN], st_near)
        track(mid + off * normal, mid + gap * normal, S[wd.W_IN], st_near)
        track(mid + gap * normal, w_start + gap * normal, S[wd.W_TO_START], st_near)
        track(w_start + gap * normal, w_start + gap * normal, S[wd.W_DWELL], st_near)
        steps = int(float(seam_len.max()) / wd.WELD_SPEED / u.step_dt)
        track(w_start + gap * normal, w_end + gap * normal, steps, st_weld, smooth=False)
        q_end = q_cmd.clone()
        track(w_end + gap * normal, mid + gap * normal, S[wd.W_TO_MID], st_near)
        track(mid + gap * normal, mid + off * normal, S[wd.W_OUT], st_near)
        track(mid + off * normal, p_far, S[wd.W_FAR_OUT], st_near)
        joint_move(ready, S[wd.W_RETRACT], st_move)

    err_max = torch.maximum(st_weld["err"], st_near["err"]) * 1000
    err_weld = st_weld["err"] * 1000
    ang_max = torch.maximum(st_weld["ang"], st_near["ang"])
    t_cl = torch.minimum(st_weld["torch"], st_near["torch"]) * 1000
    h_cl = torch.minimum(st_weld["hand"], st_near["hand"]) * 1000
    b_cl = torch.stack([x["body"] for x in (st_weld, st_near, st_move)]).min(dim=0).values * 1000
    jm = torch.minimum(st_move["torch"], st_move["hand"]) * 1000
    lim = st_weld["lim"] | st_near["lim"]
    ok = (err_max < 5.0) & (t_cl > 3.0) & (h_cl > 10.0) & (b_cl > 30.0) & (jm > 20.0)
    # 이웃 후보(거리·옆·방향이 한 칸씩 다른 자리) 중 통과한 수: 서는 자리가 조금 어긋나도 되는 후보를 고르려는 것
    index = {c: i for i, c in enumerate(combos)}
    near_ok = torch.zeros(n, dtype=torch.long)
    for i, (nm, d, lat, yw) in enumerate(combos):
        di, li, yi = dists.index(d), lats.index(lat), yaws.index(yw)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if 0 <= di + a < len(dists) and 0 <= li + b < len(lats) and 0 <= yi + c < len(yaws):
                        near_ok[i] += int(ok[index[(nm, dists[di + a], lats[li + b], yaws[yi + c])]])

    print("\n========== 용접선별 서는 자리 후보 (몸 고정, 용접 순서 그대로. 오차·거리 mm, 각도 = 토치 방향과 '용접선 바깥 방향의 반대' 사이) ==========")
    print("D = 철골 가운데 면에서 골반까지, L = 용접선에서 옆으로, yaw = 철골 정면에서 돌아선 각도(+ = 왼쪽으로), 이웃 = 둘레 27칸 중 통과한 수")
    print("통과 = 용접선 가까이에서 위치 오차 5mm 미만, 토치와 철골 3mm 넘게, 손 10mm·몸 30mm 넘게, 팔을 뻗고 거둘 때 20mm 넘게")
    for nm in names:
        idx = [i for i, c in enumerate(combos) if c[0] == nm]
        n_ok = sum(bool(ok[i]) for i in idx)
        print(f"\n[{nm}] 종류 {seams[nm]['kind']}, 통과한 후보 {n_ok}/{len(idx)}")
        order = sorted(idx, key=lambda i: (-int(near_ok[i]) if ok[i] else 1, float(ang_max[i]) + (0.0 if ok[i] else 1000.0 + float(err_max[i]))))
        for i in order[: args_cli.top]:
            _, d, lat, yw = combos[i]
            print(f"  {'✔' if ok[i] else '✘'} D {d:.2f} L {lat:+.2f} yaw {yw:+4.0f}° 이웃 {int(near_ok[i]):2d} | 위치 오차 최대 {err_max[i]:5.1f} (용접 중 {err_weld[i]:4.1f}) | "
                  f"각도 최대 {ang_max[i]:5.1f}° | 거리: 토치 {t_cl[i]:6.1f} 손 {h_cl[i]:6.1f} 몸 {b_cl[i]:6.1f} 팔 뻗을 때 {jm[i]:6.1f} | "
                  f"관절 한계 {'닿음' if lim[i] else '-'}\n      기준 자세 " + ", ".join(
                      f'"{key}": ({", ".join(f"{v:.2f}" for v in tab[i].tolist())})' for key, tab in zip(("far", "mid", "start", "end"), q_tab)))
    env.close()


def run_hand():
    import imageio.v3 as iio

    env = make_env(1, render=True)
    u = env.unwrapped
    robot = u.scene["robot"]
    grip_ids, grip_q = wd.grip_targets(robot)
    act = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
    palm = robot.body_names.index("right_palm_link")
    os.makedirs(args_cli.out, exist_ok=True)
    views = {"side": (0.0, -0.55, 0.12), "front": (0.60, -0.05, 0.15), "top": (0.05, -0.05, 0.60), "inner": (0.0, 0.55, 0.10)}
    with torch.inference_mode():
        for _ in range(40):
            robot.set_joint_position_target(grip_q, joint_ids=grip_ids)
            env.step(act)
            u.render()
        p = robot.data.body_pos_w[0, palm].tolist()
        for name, eye in views.items():
            u.sim.set_camera_view(eye=(p[0] + eye[0], p[1] + eye[1], p[2] + eye[2]), target=(p[0] + 0.05, p[1], p[2] + 0.02))
            for _ in range(8):
                env.step(act)
                frame = u.render()
            iio.imwrite(os.path.join(args_cli.out, f"hand_{name}.png"), frame)
    print(f"[weld] 사진 {len(views)}장 → {args_cli.out}")
    env.close()


def run_pose():
    """오른팔 관절값 후보마다: 토치 끝·손의 위치(골반 기준 cm)와, 토치·손이 골반 앞으로 가장 멀리 나간 거리."""
    poses = [[float(v) for v in item.split(",")] for item in args_cli.poses.split(";")]
    n = len(poses)
    env = make_env(n)
    u = env.unwrapped
    robot = u.scene["robot"]
    dev = u.device
    ik = wd.TorchIK(robot)
    clear = wd.FrameClearance(robot, robot.data.root_pos_w.clone(), torch.zeros(n, device=dev))
    grip_ids, grip_q = wd.grip_targets(robot)
    act = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
    q = torch.tensor(poses, device=dev)
    with torch.inference_mode():
        for _ in range(80):
            robot.set_joint_position_target(q, joint_ids=ik.joint_ids)
            robot.set_joint_position_target(grip_q.expand(n, -1), joint_ids=grip_ids)
            env.step(act)
    from isaaclab.utils.math import matrix_from_quat
    root = robot.data.root_pos_w
    rot = matrix_from_quat(robot.data.body_quat_w[:, clear.palm_id])
    pts = robot.data.body_pos_w[:, clear.palm_id][:, None, :] + (rot[:, None] @ clear.torch_pts[None, :, :, None])[..., 0]
    hand = robot.data.body_pos_w[:, clear.hand_ids]
    for i, pose in enumerate(poses):
        tip = ((ik.tip_pos_w()[i] - root[i]) * 100).tolist()
        palm = ((robot.data.body_pos_w[i, clear.palm_id] - root[i]) * 100).tolist()
        fwd = max(float((pts[i, :, 0] - root[i, 0]).max()), float((hand[i, :, 0] - root[i, 0]).max()) + 0.03) * 100
        print(f"  {pose} → 토치 끝 {[round(v, 1) for v in tip]}, 손 {[round(v, 1) for v in palm]}, 앞으로 가장 멀리 {fwd:.1f}cm, "
              f"토치 방향 {[round(v, 2) for v in ik.axis_w()[i].tolist()]}")
    env.close()


if __name__ == "__main__":
    {"reach": run_reach, "hand": run_hand, "pose": run_pose}[args_cli.mode]()
    simulation_app.close()
