"""VLM 학습 데이터 생성: 장면을 무작위로 만들어 머리 카메라 사진을 찍고, 장면의 실제 상태로 한국어 명령과 정답 JSON 을 만든다.

- 장면: 받침대 3개(왼쪽·가운데·오른쪽) 중 0~3개에 색 상자(red, blue, green, yellow)를 하나씩. 받침대 위치·방향,
  상자 위치·방향·크기(5가지), 카메라 위치·높이·기울기(= 로봇 골반 자세), 조명 세기·방향을 조금씩 무작위화
- 로봇은 몸을 공중에 고정하고 짐 든 자세로 둔다 (사진 아래쪽에 손이 조금 보임, 실제 실행 때와 같은 시점)
- 명령·정답: vlm/templates.py (조사 자동 선택, 템플릿마다 마지막 패턴은 test 전용)
- 결과: <out>/train.jsonl, val.jsonl, test.jsonl + <out>/images/<split>_<번호>.png (448x448)
  각 줄: {"image", "input", "output", "template", "heldout", "scene", (있으면) "color", "src", "dest"}
- test 는 절반이 학습 때 본 패턴, 절반이 처음 보는 패턴. 시드를 고정하므로 같은 명령이면 같은 데이터가 나온다

예) python -u scripts\\make_vlm_data.py --headless --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
    소량 시험: ... --n_train 32 --n_val 8 --n_test 8 --num_envs 8 --out vlm\\data_small
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate image + Korean command + plan JSON data for the VLM planner.")
parser.add_argument("--out", type=str, default=os.path.join("vlm", "data"))
parser.add_argument("--n_train", type=int, default=3600)
parser.add_argument("--n_val", type=int, default=300)
parser.add_argument("--n_test", type=int, default=400, help="절반은 본 패턴, 절반은 처음 보는 패턴")
parser.add_argument("--num_envs", type=int, default=16, help="한 번에 찍는 사진 수")
parser.add_argument("--seed", type=int, default=42)
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
sys.path.insert(0, str(ROOT / "vlm"))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
import isaaclab.sim as sim_utils  # noqa: E402
from humanoid_tasks.g1_carry.pick_place_cfg import BOX_SIZE, TABLE_SIZE  # noqa: E402
from humanoid_tasks.g1_carry.vlm_scene import apply_layouts, read_camera, set_lighting  # noqa: E402
from humanoid_tasks.g1_carry.vlm_scene_cfg import BOX_COLORS, STAND_NAMES  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from templates import Gen  # noqa: E402

TASK = "Humanoid-G1-VLM-Scene-v0"
# 상자 크기 (앞뒤, 좌우, 높이 m). 첫 번째가 기본 크기(실행 때 쓰는 크기). 환경 i 의 c 번째 색 상자는 BOX_SIZES[(i + c) % 5]
BOX_SIZES = [BOX_SIZE, (0.17, 0.26, 0.13), (0.22, 0.33, 0.18), (0.18, 0.32, 0.14), (0.23, 0.27, 0.17)]


def sample_layout(rng: random.Random) -> dict:
    """장면 하나: 상자 배치와 위치·방향·카메라 자세의 작은 흔들림."""
    k = rng.choices([0, 1, 2, 3], weights=[4, 26, 45, 25])[0]  # 상자 수 (빈 받침대가 자주 있게)
    stands = rng.sample(STAND_NAMES, k)
    colors = rng.sample(list(BOX_COLORS), k)
    boxes = {s: None for s in STAND_NAMES}
    boxes.update(dict(zip(stands, colors)))
    shift_x, spread = rng.uniform(-0.10, 0.10), rng.uniform(0.92, 1.08)
    stand_off = {}
    for name in STAND_NAMES:
        base_y = {"stand_L": 0.65, "stand_C": 0.0, "stand_R": -0.65}[name]
        stand_off[name] = (shift_x + rng.uniform(-0.03, 0.03), base_y * (spread - 1.0) + rng.uniform(-0.03, 0.03),
                           math.radians(rng.uniform(-8.0, 8.0)))
    box_off = {name: (rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), math.radians(rng.uniform(-15.0, 15.0)))
               for name in STAND_NAMES}
    # 로봇 자세: 걸어서 처음 자리로 돌아왔을 때의 오차(위치 수 cm, 방향 수 도)를 덮도록 넉넉히
    robot = (rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10), rng.uniform(0.72, 0.78),
             math.radians(rng.uniform(-2.0, 2.0)), math.radians(rng.uniform(-6.0, 5.0)), math.radians(rng.uniform(-8.0, 8.0)))
    return {"boxes": boxes, "stands": stand_off, "box_offset": box_off, "robot": robot}


# ---- 1. 줄(장면 + 명령 + 정답)을 먼저 모두 만든다 (렌더와 무관하게 시드로 정해짐) ----
rng = random.Random(args_cli.seed)
half = args_cli.n_test // 2
plan = [("train", args_cli.n_train, "train"), ("val", args_cli.n_val, "train"),
        ("test", half, "train"), ("test", args_cli.n_test - half, "heldout")]
rows = []
counts = {}
for split, n, mode in plan:
    gen = Gen(rng, mode)
    for _ in range(n):
        layout = sample_layout(rng)
        row = gen.sample(layout["boxes"])
        idx = counts.get(split, 0)
        counts[split] = idx + 1
        row = {"image": f"images/{split}_{idx:05d}.png", **row, "scene": layout["boxes"]}
        rows.append((split, row, layout))
print(f"[data] 만들 줄: " + ", ".join(f"{k} {v}" for k, v in counts.items()))

# ---- 2. 환경 ----
N = args_cli.num_envs
env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
spawn = env_cfg.scene.robot.spawn
env_cfg.scene.robot.spawn = spawn.replace(articulation_props=spawn.articulation_props.replace(fix_root_link=True))
# 상자 크기를 환경마다 다르게: 색마다 크기 목록을 한 칸씩 밀어 넣어, 한 환경 안에서도 색마다 크기가 다르게 함
env_cfg.scene.replicate_physics = False
for c_i, color in enumerate(BOX_COLORS):
    box_cfg = getattr(env_cfg.scene, f"{color}_box")
    base = box_cfg.spawn
    order = [BOX_SIZES[(k + c_i) % len(BOX_SIZES)] for k in range(len(BOX_SIZES))]
    box_cfg.spawn = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[sim_utils.CuboidCfg(size=s, visual_material=base.visual_material) for s in order],
        random_choice=False, rigid_props=base.rigid_props, mass_props=base.mass_props, collision_props=base.collision_props)


def box_z(env_i: int) -> dict:
    """환경 env_i 의 색별 상자 중심 높이 (받침대 윗면 + 상자 높이의 절반)."""
    return {color: TABLE_SIZE[2] + BOX_SIZES[(env_i + c_i) % len(BOX_SIZES)][2] / 2 for c_i, color in enumerate(BOX_COLORS)}


env = gym.make(TASK, cfg=env_cfg)
env.reset()
u = env.unwrapped
zero_action = torch.zeros(N, u.action_manager.total_action_dim, device=u.device)

out_dir = Path(args_cli.out)
(out_dir / "images").mkdir(parents=True, exist_ok=True)

# ---- 3. 묶음마다 배치 → 조명 → 몇 스텝 → 사진 저장 ----
light_rng = random.Random(args_cli.seed + 1)
done = 0
for start in range(0, len(rows), N):
    batch = rows[start:start + N]
    layouts = [lay for _, _, lay in batch]
    layouts += [layouts[-1]] * (N - len(layouts))  # 마지막 묶음의 남는 환경은 같은 배치로 채움
    layouts = [{**lay, "box_z": box_z(i)} for i, lay in enumerate(layouts)]
    apply_layouts(u, layouts)
    set_lighting(dome_intensity=light_rng.uniform(350.0, 1500.0), sun_intensity=light_rng.uniform(300.0, 3000.0),
                 sun_elevation_deg=light_rng.uniform(30.0, 85.0), sun_azimuth_deg=light_rng.uniform(0.0, 360.0))
    for _ in range(4):  # 옮긴 물체와 바뀐 조명이 사진에 반영되도록 몇 스텝 진행
        apply_layouts(u, layouts)  # 몸 고정 로봇·움직이지 않는 받침대는 그대로지만, 상자가 미끄러지지 않게 매번 다시 놓음
        env.step(zero_action)
    rgb = read_camera(u).cpu().numpy()
    for i, (split, row, _) in enumerate(batch):
        iio.imwrite(out_dir / row["image"], rgb[i])
    done += len(batch)
    if (start // N) % 20 == 0:
        print(f"[data] {done}/{len(rows)}")

for split in counts:
    with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for s, row, _ in rows:
            if s == split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

# 요약: 템플릿·정답 종류 분포 (train 기준)
train_rows = [r for s, r, _ in rows if s == "train"]
by_template, by_kind = {}, {}
for r in train_rows:
    by_template[r["template"]] = by_template.get(r["template"], 0) + 1
    kind = json.loads(r["output"]).get("error", "plan")
    by_kind[kind] = by_kind.get(kind, 0) + 1
print("[data] train 템플릿 분포:", dict(sorted(by_template.items(), key=lambda kv: -kv[1])))
print("[data] train 정답 종류:", by_kind)
print(f"[data] 저장: {out_dir} (사진 {len(rows)}장)")

env.close()
simulation_app.close()
