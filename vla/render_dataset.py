"""변환한 LeRobot 데이터셋을 LeRobot 으로 다시 읽어, 몇 에피소드를 영상으로 그려 확인한다.

왼쪽에 사진, 오른쪽에 그 프레임의 상태·행동(팔 관절 10개의 위치와 목표값 막대, 잡기 상태·신호)과 작업 문장의 색인을 그린다.
원자료(--raw)를 주면 같은 에피소드의 원자료와 숫자가 같은지도 비교한다 (영상은 압축되므로 사진은 평균 차이만 봄).

LeRobot 을 설치한 별도 환경에서 실행한다.
예) python vla/render_dataset.py --root data/vla_pick/lerobot_v1 --episodes 0,1,2 --out data/vla_pick/check_v1 --raw data/vla_pick/raw_v1
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo_id", default="local/g1_vla_pick")
    ap.add_argument("--episodes", default="0,1,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None, help="비교할 원자료 폴더 (에피소드 번호가 파일 번호와 같을 때)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = [int(v) for v in args.episodes.split(",")]
    ds = LeRobotDataset(args.repo_id, root=args.root)
    meta = ds.meta
    print(f"[check] 에피소드 {meta.total_episodes}, 프레임 {meta.total_frames}, fps {meta.fps}, 항목 "
          + ", ".join(f"{k} {v['shape']}" for k, v in meta.features.items() if k.startswith(("observation", "action"))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ep_index = np.array(ds.hf_dataset["episode_index"]).reshape(-1) if hasattr(ds, "hf_dataset") else None
    for ep in episodes:
        idx = np.nonzero(ep_index == ep)[0]
        raw = np.load(Path(args.raw) / f"ep_{ep:06d}.npz") if args.raw else None
        writer, img_diff, st_diff, ac_diff, task = None, [], [], [], None
        for k, i in enumerate(idx.tolist()):
            item = ds[i]
            img = (item["observation.images.head"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            st, ac, task = item["observation.state"].numpy(), item["action"].numpy(), item["task"]
            if raw is not None:
                img_diff.append(np.abs(img.astype(np.int16) - raw["images"][k].astype(np.int16)).mean())
                st_diff.append(np.abs(st - raw["state"][k]).max())
                ac_diff.append(np.abs(ac - raw["action"][k]).max())
            big = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
            panel = np.full((512, 420, 3), 24, np.uint8)
            cv2.putText(panel, f"episode {ep}  frame {k}/{len(idx)}  t={float(item['timestamp']):.2f}s", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            for j in range(10):  # 관절마다: 위치(흰색)와 목표값(하늘색), -2~2 rad 를 막대 길이로
                y = 60 + j * 34
                cv2.putText(panel, f"{'L' if j < 5 else 'R'}{j % 5}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                cv2.line(panel, (230, y - 10), (230, y + 10), (90, 90, 90), 1)
                cv2.line(panel, (230, y - 4), (int(230 + st[j] * 80), y - 4), (255, 255, 255), 5)
                cv2.line(panel, (230, y + 4), (int(230 + ac[j] * 80), y + 4), (60, 220, 255), 5)
            cv2.putText(panel, f"grasp state {st[10]:.0f}   command {ac[10]:.0f}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (80, 255, 120) if ac[10] > 0.5 else (200, 200, 200), 2)
            cv2.putText(panel, "white = joint pos, blue = target", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            frame = np.concatenate([big, panel], axis=1)
            if writer is None:
                writer = cv2.VideoWriter(str(out / f"episode_{ep:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (frame.shape[1], 512))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if k in (0, len(idx) // 3, 2 * len(idx) // 3, len(idx) - 1):
                cv2.imwrite(str(out / f"episode_{ep:03d}_f{k:03d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        line = f"[check] 에피소드 {ep}: {len(idx)} 프레임, 문장 {task!r}"
        if raw is not None:
            line += (f" | 원자료와 비교: 프레임 수 {raw['images'].shape[0]}, 원자료 문장 {json.loads(str(raw['meta']))['task']!r}, "
                     f"사진 평균 차이 {np.mean(img_diff):.2f}/255 (최대 {np.max(img_diff):.2f}), "
                     f"상태 최대 차이 {np.max(st_diff):.2e}, 행동 최대 차이 {np.max(ac_diff):.2e}")
        print(line)
    print(f"[check] 영상 저장: {out}")


if __name__ == "__main__":
    main()
