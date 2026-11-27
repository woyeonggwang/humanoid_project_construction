"""11단계 VLM 학습 데이터 생성: 옮기기 + 용접 명령, 배경을 섞은 사진.

9단계의 make_vlm_full_data.py 와 같은 방식이고 달라진 점은
- 명령·정답: vlm/templates_site.py (9단계 템플릿 + 용접, 옮기기 + 용접, 철골로 가기. 템플릿마다 마지막 패턴은 test 전용)
- 배경: 줄마다 다섯 가지 가운데 하나 (기본 장면 / 공사장 A / B1(받침대 뒤 펜스·자재) / B2(B1 + 콘크리트 바닥) / 흙 바닥(B1 + 흙 바닥)).
  배경마다 과제가 달라 한 번 실행에 한 배경의 사진만 찍는다 (--bg). 줄(jsonl)은 시드로 정해지므로 어느 실행이든 같은 내용을 쓴다
- 결과: <out>/train.jsonl, val.jsonl, test.jsonl + <out>/images/ (기본 vlm/data_site). 줄마다 "bg" 가 들어 있다

예) 배경 다섯 가지를 차례로:
    python -u scripts\\make_vlm_site_data.py --headless --bg base --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
    (… --bg site_a, site_b1, site_b2, site_dirt)
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

BG_TASKS = {"base": "Humanoid-G1-VLA-Full-v0", "site_a": "Humanoid-G1-Site-A-v0", "site_b1": "Humanoid-G1-Site-B1-v0",
            "site_b2": "Humanoid-G1-Site-B2-v0", "site_dirt": "Humanoid-G1-Site-B3-v0"}

parser = argparse.ArgumentParser(description="Generate image + Korean command + plan JSON data (carry + weld, mixed backgrounds).")
parser.add_argument("--out", type=str, default=os.path.join("vlm", "data_site"))
parser.add_argument("--bg", type=str, required=True, choices=list(BG_TASKS), help="이번 실행에서 사진을 찍을 배경")
parser.add_argument("--n_train", type=int, default=4800)
parser.add_argument("--n_val", type=int, default=300)
parser.add_argument("--n_test", type=int, default=600, help="절반은 본 패턴, 절반은 처음 보는 패턴")
parser.add_argument("--num_envs", type=int, default=16, help="한 번에 찍는 사진 수")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--splits", type=str, default="train,val,test")
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
from humanoid_tasks.g1_carry import site_layout as sl  # noqa: E402
from humanoid_tasks.g1_carry import vla_full as vf  # noqa: E402
from humanoid_tasks.g1_carry.vlm_scene import set_lighting  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from schema import WELD_ALL  # noqa: E402
from templates_site import GenSite  # noqa: E402

assert tuple(sorted(WELD_ALL)) == tuple(sorted(sl.WELD_REACHABLE)), "schema 의 용접선 이름이 site_layout 과 다름"
TASK = BG_TASKS[args_cli.bg]


def sample_robot(rng: random.Random):
    """사진을 찍는 로봇(= 카메라)의 자세: 처음 자리에서 조금씩 벗어나게 (x, y, z, roll, pitch, yaw)."""
    return (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.72, 0.78),
            math.radians(rng.uniform(-2.0, 2.0)), math.radians(rng.uniform(-6.0, 5.0)), math.radians(rng.uniform(-3.0, 3.0)))


# ---- 1. 줄(장면 + 명령 + 정답 + 배경)을 먼저 모두 만든다 (렌더와 무관하게 시드로 정해짐) ----
rng = random.Random(args_cli.seed)
bg_rng = random.Random(args_cli.seed + 7)
half = args_cli.n_test // 2
plan = [("train", args_cli.n_train, "train"), ("val", args_cli.n_val, "train"),
        ("test", half, "train"), ("test", args_cli.n_test - half, "heldout")]
rows, counts = [], {}
for split, n, mode in plan:
    gen = GenSite(rng, mode)
    for _ in range(n):
        scene = vf.sample_scene(rng)
        scene["robot"] = sample_robot(rng)
        row = gen.sample(vf.describe(scene))
        idx = counts.get(split, 0)
        counts[split] = idx + 1
        rows.append((split, {"image": f"images/{split}_{idx:05d}.png", "bg": bg_rng.choice(list(BG_TASKS)), **row,
                             "scene": vf.describe(scene)}, scene))
print("[data] 만들 줄: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
keep = set(args_cli.splits.split(","))
out_dir = Path(args_cli.out)
(out_dir / "images").mkdir(parents=True, exist_ok=True)
for split in counts:
    if split in keep:
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for s, row, _ in rows:
                if s == split:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
shots = [r for r in rows if r[0] in keep and r[1]["bg"] == args_cli.bg]
print(f"[data] 배경 {args_cli.bg} ({TASK}): 사진 {len(shots)}장")

# ---- 2. 환경 (로봇은 몸을 공중에 고정하고 짐 든 자세) ----
N = args_cli.num_envs
env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
spawn = env_cfg.scene.robot.spawn
env_cfg.scene.robot.spawn = spawn.replace(articulation_props=spawn.articulation_props.replace(fix_root_link=True))
env = gym.make(TASK, cfg=env_cfg)
env.reset()
u = env.unwrapped
zero_action = torch.zeros(N, u.action_manager.total_action_dim, device=u.device)

# ---- 3. 묶음마다 배치 → 조명 → 몇 스텝 → 사진 저장 ----
light_rng = random.Random(args_cli.seed + 1 + list(BG_TASKS).index(args_cli.bg))
done = 0
for start in range(0, len(shots), N):
    batch = shots[start:start + N]
    scenes = [sc for _, _, sc in batch]
    scenes += [scenes[-1]] * (N - len(scenes))  # 마지막 묶음의 남는 환경은 같은 장면으로 채움
    vf.apply_scenes(u, scenes)
    set_lighting(dome_intensity=light_rng.uniform(350.0, 1500.0), sun_intensity=light_rng.uniform(300.0, 3000.0),
                 sun_elevation_deg=light_rng.uniform(30.0, 85.0), sun_azimuth_deg=light_rng.uniform(0.0, 360.0))
    for _ in range(4):  # 옮긴 물체와 바뀐 조명이 사진에 반영되도록 몇 스텝 진행
        vf.apply_scenes(u, scenes)  # 상자가 미끄러지지 않게 매번 다시 놓음
        env.step(zero_action)
    rgb = vf.read_vlm_camera(u).cpu().numpy()
    for i, (split, row, _) in enumerate(batch):
        iio.imwrite(out_dir / row["image"], rgb[i])
    done += len(batch)
    if (start // N) % 20 == 0:
        print(f"[data] {done}/{len(shots)}", flush=True)

train_rows = [r for s, r, _ in rows if s == "train"]
by_template, by_kind, by_bg = {}, {}, {}
for r in train_rows:
    key = f"{r['template']}:{r.get('outcome')}"
    by_template[key] = by_template.get(key, 0) + 1
    kind = json.loads(r["output"]).get("error", "plan")
    by_kind[kind] = by_kind.get(kind, 0) + 1
    by_bg[r["bg"]] = by_bg.get(r["bg"], 0) + 1
print("[data] train 템플릿·결과 분포:", dict(sorted(by_template.items(), key=lambda kv: -kv[1])))
print("[data] train 정답 종류:", by_kind)
print("[data] train 배경:", by_bg)
print(f"[data] 저장: {out_dir} (이번 실행의 사진 {len(shots)}장)")

env.close()
simulation_app.close()
