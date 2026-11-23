"""Isaac 에서 모은 원자료(ep_*.npz)를 LeRobot 데이터셋 형식(v3.0)으로 바꾼다. 로컬 폴더에만 만든다 (Hub 에 올리지 않음).

원자료 한 에피소드: images (T, H, W, 3) uint8, state (T, 11), action (T, 11), meta(JSON: task, fps 등)
  — scripts/collect_vla_demos.py 가 만든 것. 상태·행동의 뜻은 humanoid_tasks/g1_carry/vla_pick.py 설명 참고
만드는 데이터셋의 항목: observation.images.head (영상), observation.state (11), action (11), 작업 문장(task)

LeRobot 을 설치한 별도 환경에서 실행한다 (Isaac 환경이 아님).
예) python vla/convert_to_lerobot.py --raw data/vla_pick/raw_v1 --root data/vla_pick/lerobot_v1
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]
JOINTS = [f"{side}_{j}" for side in ("left", "right") for j in ARM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, nargs="+", help="원자료 폴더 (여러 개 가능)")
    ap.add_argument("--root", required=True, help="만들 데이터셋 폴더 (있으면 지우고 다시 만듦)")
    ap.add_argument("--repo_id", default="local/g1_vla_pick", help="데이터셋 이름 (로컬에서만 씀)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서부터 N 에피소드만")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = sorted(f for r in args.raw for f in Path(r).glob("ep_*.npz"))
    if args.limit:
        files = files[: args.limit]
    first = np.load(files[0])
    h, w = first["images"].shape[1:3]
    fps = int(json.loads(str(first["meta"]))["fps"])
    features = {
        "observation.images.head": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (11,), "names": [f"{j}.pos" for j in JOINTS] + ["grasp.state"]},
        "action": {"dtype": "float32", "shape": (11,), "names": [f"{j}.target" for j in JOINTS] + ["grasp.command"]},
    }
    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(args.repo_id, fps=fps, features=features, root=root, robot_type="unitree_g1_arms", use_videos=True)
    t0, frames, tasks = time.time(), 0, {}
    for i, f in enumerate(files):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        images, state, action = d["images"], d["state"].astype(np.float32), d["action"].astype(np.float32)
        for t in range(images.shape[0]):
            ds.add_frame({"observation.images.head": images[t], "observation.state": state[t], "action": action[t], "task": meta["task"]})
        ds.save_episode()
        frames += images.shape[0]
        tasks[meta["task"]] = tasks.get(meta["task"], 0) + 1
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[convert] {i + 1}/{len(files)} 에피소드, {frames} 프레임, {time.time() - t0:.0f}초", flush=True)
    ds.finalize()
    print(f"[convert] 완료: {root} | 에피소드 {len(files)}, 프레임 {frames}, {fps}fps, 사진 {h}x{w}")
    print("[convert] 문장별 에피소드 수: " + ", ".join(f"{k!r} {v}" for k, v in tasks.items()))


if __name__ == "__main__":
    main()
