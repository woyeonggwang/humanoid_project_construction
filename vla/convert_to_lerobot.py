"""Isaac 에서 모은 원자료(ep_*.npz)를 LeRobot 데이터셋 형식(v3.0)으로 바꾼다. 로컬 폴더에만 만든다 (Hub 에 올리지 않음).

원자료 한 에피소드: images (T, H, W, 3) uint8, state (T, 11), action (T, 11), meta(JSON: task, fps 등)
  — scripts/collect_vla_demos.py 가 만든 것. 상태·행동의 뜻은 humanoid_tasks/g1_carry/vla_pick.py 설명 참고
만드는 데이터셋의 항목: observation.images.head (영상), observation.state (11), action (11), 작업 문장(task)

--cross_frames K (2차의 쌍 시범용): 쌍의 두 에피소드 A·B 는 장면이 거의 같고 집을 상자(명령)만 다르지만, 로봇이 선 자리가 조금 달라
사진이 완전히 같지는 않다 → 모델이 명령 대신 장면을 외워 답을 맞힐 수 있다. 그래서 '사진·상태가 완전히 같고 명령만 다른' 구간을 만들어 넣는다:
  에피소드마다 짧은 에피소드를 하나 더 만든다 = 앞 K 프레임의 사진·상태는 짝(A)의 것, 명령과 그 뒤의 행동은 자기(B) 것.
  앞 K 프레임(손을 위로 올리는 구간)은 어느 상자를 집든 같은 동작이라 사진만으로는 구분되지 않으므로, 같은 사진에 두 명령·두 가지 뒤 행동이 붙는다.
  행동은 K 프레임 직전의 --cross_blend 프레임 동안 A 의 것에서 B 의 것으로 서서히 바꾼다 (두 로봇의 팔 각도가 0.01~0.03 rad 다르므로)

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
    ap.add_argument("--task_key", default="task", help="원자료 meta 에서 작업 문장으로 쓸 항목 (task = 한국어, task_en = 영어)")
    ap.add_argument("--cross_frames", type=int, default=0,
                    help="0 보다 크면 쌍의 짝에서 앞 K 프레임의 사진·상태를 가져온 짧은 에피소드를 에피소드마다 하나 더 넣는다 (위 설명)")
    ap.add_argument("--cross_len", type=int, default=None, help="--cross_frames 로 만드는 짧은 에피소드의 길이 (기본 K + 50)")
    ap.add_argument("--cross_blend", type=int, default=15, help="K 프레임 직전 몇 프레임 동안 행동을 짝의 것에서 자기 것으로 바꿀지")
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
    ds = LeRobotDataset.create(args.repo_id, fps=fps, features=features, root=root, robot_type="unitree_g1_arms", use_videos=True,
                               image_writer_threads=8)
    partner = {}  # --cross_frames: 파일 → 쌍의 짝 파일
    if args.cross_frames > 0:
        by_pair = {}
        for f in files:
            with np.load(f) as d:
                by_pair.setdefault(json.loads(str(d["meta"]))["pair"], []).append(f)
        for v in by_pair.values():
            if len(v) == 2:
                partner[v[0]], partner[v[1]] = v[1], v[0]
        print(f"[convert] 쌍 {len(partner) // 2}개에 '같은 사진, 다른 명령' 짧은 에피소드를 넣음 (앞 {args.cross_frames} 프레임)", flush=True)
    t0, frames, tasks, n_ep = time.time(), 0, {}, 0
    for i, f in enumerate(files):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        images, state, action = d["images"], d["state"].astype(np.float32), d["action"].astype(np.float32)
        for t in range(images.shape[0]):
            ds.add_frame({"observation.images.head": images[t], "observation.state": state[t], "action": action[t], "task": meta[args.task_key]})
        ds.save_episode()
        frames += images.shape[0]
        n_ep += 1
        if f in partner:
            k = args.cross_frames
            length = min(args.cross_len or k + 50, images.shape[0])
            with np.load(partner[f]) as a:
                a_img, a_state, a_action = a["images"][:k], a["state"][:k].astype(np.float32), a["action"][:k].astype(np.float32)
            mix = np.clip((np.arange(k) - (k - args.cross_blend)) / max(args.cross_blend, 1), 0.0, 1.0)[:, None].astype(np.float32)
            head_action = a_action + mix * (action[:k] - a_action)
            for t in range(length):
                own = t >= k
                ds.add_frame({"observation.images.head": images[t] if own else a_img[t], "observation.state": state[t] if own else a_state[t],
                              "action": action[t] if own else head_action[t], "task": meta[args.task_key]})
            ds.save_episode()
            frames += length
            n_ep += 1
        tasks[meta[args.task_key]] = tasks.get(meta[args.task_key], 0) + 1
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"[convert] {i + 1}/{len(files)} 에피소드, {frames} 프레임, {time.time() - t0:.0f}초", flush=True)
    ds.finalize()
    print(f"[convert] 완료: {root} | 에피소드 {n_ep} (원자료 {len(files)}개), 프레임 {frames}, {fps}fps, 사진 {h}x{w}")
    print("[convert] 문장별 에피소드 수: " + ", ".join(f"{k!r} {v}" for k, v in tasks.items()))


if __name__ == "__main__":
    main()
