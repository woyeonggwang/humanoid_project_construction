"""10단계 공사장 장면 점검: 모델이 제 크기·방향·색으로 보이는지, 로봇의 두 카메라 화면이 9단계 장면과 같은지 본다.

--mode overview   바깥에서 본 사진 몇 장을 <out>/ 에 저장 (모델의 크기·방향·색을 눈으로 확인. 확인한 뒤 지운다)
--mode cameras    로봇 몸을 고정하고 정해진 자세(처음 자리, 집는 받침대 앞, 내려놓을 받침대 앞)·조명마다
                  VLM 카메라와 VLA 카메라 사진을 <out>.npz 로 저장. 과제만 바꿔 두 번 찍어 --compare 로 비교한다
--mode collide    충돌 상자 확인: 로봇 두 대를 블록 더미 위에 놓고 0.5초 뒤 골반 높이를 본다 (충돌이 없으면 바닥까지 떨어짐)
--compare A B     두 npz 의 사진 차이를 출력 (Isaac 을 켜지 않음)

예) python -u scripts\\site_check.py --headless --mode cameras --task Humanoid-G1-VLA-Full-v0 --out data\\site\\cam_full --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
    python -u scripts\\site_check.py --headless --mode cameras --task Humanoid-G1-Site-A-v0 --out data\\site\\cam_site_a --kit_args "..."
    python scripts\\site_check.py --compare data\\site\\cam_full.npz data\\site\\cam_site_a.npz
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Check the construction site scene: overview renders and robot camera comparison.")
parser.add_argument("--mode", choices=["overview", "cameras", "collide"], default="cameras")
parser.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
parser.add_argument("--task", type=str, default="Humanoid-G1-Site-A-v0")
parser.add_argument("--out", type=str, default=os.path.join("data", "site", "check"))
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--save_png", action="store_true", help="cameras 모드에서 사진을 PNG 로도 저장")
parser.add_argument("--settle", type=int, default=6, help="조명을 바꾼 뒤 사진을 찍기까지 진행하는 스텝 수")
parser.add_argument("--poses", type=str, default="", help="원인 찾기용: 이 번호의 자세만 찍는다 (쉼표로 여러 개. 환경 수 = 자세 수)")
parser.add_argument("--drop", type=str, default="", help="원인 찾기용: 이 글자로 시작하는 공사장 모델을 빼고 장면을 만든다 (쉼표로 여러 개. 예: fence,scaffold. all 이면 모두 뺌 = 같은 배치의 빈 장면)")

if "--compare" in sys.argv:  # Isaac 없이 비교만
    args = parser.parse_args()
    a, b = (np.load(p if p.endswith(".npz") else p + ".npz") for p in args.compare)
    print(f"비교: {args.compare[0]}  ↔  {args.compare[1]}")
    worst = 0.0
    for cam in ("vlm", "vla"):
        x, y = a[cam].astype(np.int16), b[cam].astype(np.int16)
        d = np.abs(x - y)  # [조명, 자세, 높이, 폭, 3]
        per = d.reshape(d.shape[0], d.shape[1], -1)
        mean, mx = per.mean(-1), per.max(-1)
        frac = (d.max(-1) > 8).reshape(d.shape[0], d.shape[1], -1).mean(-1)
        print(f"[{cam}] 사진 {d.shape[0] * d.shape[1]}장: 평균 차이 {mean.mean():.3f}/255 (사진별 최대 {mean.max():.3f}), "
              f"가장 큰 화소 차이 {mx.max()}, 8/255 넘게 다른 화소 비율 평균 {frac.mean():.4%} (사진별 최대 {frac.max():.4%})")
        names = [str(n) for n in a["poses"]]
        for j, name in enumerate(names):
            print(f"    {name}: 평균 {mean[:, j].mean():.3f}, 8 넘는 화소 {frac[:, j].max():.4%}")
        worst = max(worst, float(frac.max()))
    print(f"가장 나쁜 사진의 다른 화소 비율 {worst:.4%}")
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from humanoid_tasks.g1_carry import site_layout as sl  # noqa: E402
from humanoid_tasks.g1_carry import vla_full as vf  # noqa: E402
from humanoid_tasks.g1_carry.vlm_scene import set_lighting  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

STANCE = 0.36  # 받침대 앞에 서는 거리 (pick_place_machine 과 같은 값)
# 조명: 해가 가장 낮을 때(고도 40°)를 여러 방향으로 + 중간·높은 고도 (학습·평가의 범위는 고도 40~85°)
LIGHTS = [(1000.0, 1800.0, 40.0, az) for az in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)] + [(700.0, 2400.0, 55.0, 30.0), (1300.0, 1000.0, 80.0, 200.0)]

rng = random.Random(args_cli.seed)


def stance(stand_xy, stand_yaw, side: float = 0.0, dyaw_deg: float = 0.0):
    """받침대 앞에 선 로봇의 자세 (x, y, z, roll, pitch, yaw). side = 옆으로 벗어난 거리, dyaw = 비스듬한 각도."""
    c, s = math.cos(stand_yaw), math.sin(stand_yaw)
    return (stand_xy[0] - STANCE * c - side * s, stand_xy[1] - STANCE * s + side * c, 0.74, 0.0, 0.0, stand_yaw + math.radians(dyaw_deg))


def make_scenes():
    """자세 8가지: 환경마다 하나. 장면(받침대 흩기·상자·판 색)은 시드로 정해지므로 과제가 달라도 같다."""
    scenes, names = [], []
    for name in ("처음 자리", "처음 자리 (조금 벗어남)", "집는 받침대 앞", "집는 받침대 앞 (벗어남)",
                 "내려놓을 받침대 앞", "내려놓을 받침대 앞 (벗어남)", "집는 받침대 앞 (다른 장면)", "내려놓을 받침대 앞 (다른 장면)"):
        sc = vf.sample_scene(rng, n_boxes=2, on_pad=False)
        pick = (vf.PICK_XY[0] + sc["pick"][0], vf.PICK_XY[1] + sc["pick"][1]), vf.PICK_YAW + sc["pick"][2]
        place = (vf.PLACE_XY[0] + sc["place"][0], vf.PLACE_XY[1] + sc["place"][1]), vf.PLACE_YAW + sc["place"][2]
        off = "벗어남" in name
        if name.startswith("처음"):
            sc["robot"] = (0.05, -0.05, 0.76, 0.0, math.radians(4.0), math.radians(3.0)) if off else (0.0, 0.0, 0.74, 0.0, 0.0, 0.0)
        elif name.startswith("집는"):
            sc["robot"] = stance(*pick, side=0.06 if off else 0.0, dyaw_deg=8.0 if off else 0.0)
        else:
            sc["robot"] = stance(*place, side=-0.10 if off else 0.0, dyaw_deg=-10.0 if off else 0.0)
        scenes.append(sc)
        names.append(name)
    return scenes, names


if args_cli.mode == "cameras":
    scenes, names = make_scenes()
    if args_cli.poses:
        keep = [int(v) for v in args_cli.poses.split(",")]
        scenes, names = [scenes[i] for i in keep], [names[i] for i in keep]
    N = len(scenes)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
    env_cfg.seed = args_cli.seed
    spawn = env_cfg.scene.robot.spawn
    env_cfg.scene.robot.spawn = spawn.replace(articulation_props=spawn.articulation_props.replace(fix_root_link=True))
    drops = [d for d in args_cli.drop.split(",") if d]
    if drops:
        terrain = env_cfg.scene.terrain
        terrain.site_layers = [([row for row in rows if not any(d == "all" or row[0].startswith(d) for d in drops)], z, collide)
                               for rows, z, collide in terrain.site_layers]
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    zero = torch.zeros(N, u.action_manager.total_action_dim, device=u.device)
    shots = {"vlm": [], "vla": []}
    for dome, sun, elev, az in LIGHTS:
        vf.apply_scenes(u, scenes)
        set_lighting(dome_intensity=dome, sun_intensity=sun, sun_elevation_deg=elev, sun_azimuth_deg=az)
        for _ in range(args_cli.settle):
            vf.apply_scenes(u, scenes)
            env.step(zero)
        shots["vlm"].append(vf.read_vlm_camera(u).cpu().numpy().copy())
        shots["vla"].append(u.scene.sensors["vla_cam"].data.output["rgb"][..., :3].cpu().numpy().copy())
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", vlm=np.stack(shots["vlm"]), vla=np.stack(shots["vla"]), poses=np.array(names))
    if args_cli.save_png:
        for cam in ("vlm", "vla"):
            for j in range(N):
                iio.imwrite(f"{out}_{cam}_{j}.png", shots[cam][0][j])
    print(f"[site] {args_cli.task}: 조명 {len(LIGHTS)}가지 x 자세 {N}가지, 저장 {out}.npz")
elif args_cli.mode == "collide":
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=2)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    name, model, xy, _ = next(row for row in sl.PLACEMENTS_A if row[1] == "SM_BlockPallet")
    top = sl.MODEL_SIZE[model][2]
    pose = robot.data.root_state_w[:, :7].clone()
    pose[:, 0] = u.scene.env_origins[:, 0] + xy[0]
    pose[:, 1] = u.scene.env_origins[:, 1] + xy[1]
    pose[:, 2] = top + 0.80
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(2, 6, device=u.device))
    zero = torch.zeros(2, u.action_manager.total_action_dim, device=u.device)
    for _ in range(25):
        env.step(zero)
    z = robot.data.root_pos_w[:, 2].tolist()
    print(f"[site] 환경 원점: {[[round(v, 1) for v in o] for o in u.scene.env_origins.tolist()]}")
    print(f"[site] {name}(윗면 {top:.2f}m) 위에 놓고 0.5초 뒤 골반 높이: {[round(v, 2) for v in z]} "
          f"→ {'충돌함' if min(z) > top + 0.3 else '충돌하지 않음 (바닥까지 떨어짐)'}")
else:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.resolution = (1280, 720)
    spawn = env_cfg.scene.robot.spawn
    env_cfg.scene.robot.spawn = spawn.replace(articulation_props=spawn.articulation_props.replace(fix_root_link=True))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    u = env.unwrapped
    zero = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
    sc = vf.sample_scene(rng, n_boxes=2, on_pad=False)
    sc["robot"] = (0.0, 0.0, 0.74, 0.0, 0.0, 0.0)
    set_lighting(dome_intensity=1000.0, sun_intensity=1800.0, sun_elevation_deg=55.0, sun_azimuth_deg=30.0)
    fx, fy = sl.STEEL_FRAME_POS[:2]
    views = {  # 이름: (눈, 보는 곳)
        "top": ((-4.0, -1.5, 19.0), (-4.0, 0.0, 0.0)),
        "front_high": ((7.5, -7.0, 6.5), (-3.5, 0.5, 0.8)),
        "back_high": ((-13.5, 8.5, 7.0), (-3.0, -0.5, 0.8)),
        "steel_frame": ((fx + 4.2, fy - 2.2, 1.8), (fx, fy, 1.0)),
        "materials_left": ((-1.0, 1.2, 1.7), (-4.5, 4.8, 0.3)),
        "materials_right": ((-0.5, -1.5, 1.7), (-3.0, -4.8, 0.4)),
        "office": ((-3.6, -1.0, 1.7), (-8.0, -2.6, 1.3)),
        "scaffold": ((-3.6, 0.5, 1.7), (-8.6, 3.6, 3.0)),
        "robot_eye_back": ((0.0, 0.0, 1.5), (-5.0, 0.0, 1.0)),
    }
    os.makedirs(args_cli.out, exist_ok=True)
    for name, (eye, target) in views.items():
        u.sim.set_camera_view(eye=eye, target=target)
        for _ in range(8):
            vf.apply_scenes(u, [sc])
            env.step(zero)
        iio.imwrite(os.path.join(args_cli.out, f"{name}.png"), env.render())
    print(f"[site] {args_cli.task}: 사진 {len(views)}장 → {args_cli.out}")

env.close()
simulation_app.close()
