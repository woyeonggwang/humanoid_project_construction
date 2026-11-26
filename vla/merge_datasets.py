"""변환해 둔 LeRobot 데이터셋 여러 개를 하나로 합친다 (집기 + 내려놓기를 한 모델로 학습할 때). 로컬 폴더에만 만든다.

영상을 다시 압축하지 않고 이어 붙이므로 변환을 다시 하는 것보다 빠르다. 합친 뒤 에피소드·프레임 수가 원본의 합과 같은지 확인한다.

LeRobot 을 설치한 별도 환경에서 실행한다 (Isaac 환경이 아님).
예) python vla/merge_datasets.py --roots data/vla_pick/lerobot3x_ko data/vla_place/lerobot_v1 --repo_ids local/g1_vla_pick3x_ko local/g1_vla_place --out data/vla_place/lerobot_joint --repo_id local/g1_vla_joint
"""

import argparse
import shutil
import time
from pathlib import Path

from _win import no_error_dialogs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True, nargs="+", help="합칠 데이터셋 폴더들")
    ap.add_argument("--repo_ids", required=True, nargs="+", help="각 데이터셋의 이름 (변환할 때 준 --repo_id, 폴더와 같은 순서)")
    ap.add_argument("--out", required=True, help="만들 데이터셋 폴더 (있으면 지우고 다시 만듦)")
    ap.add_argument("--repo_id", default="local/g1_vla_joint")
    args = ap.parse_args()
    assert len(args.roots) == len(args.repo_ids), "--roots 와 --repo_ids 의 개수가 같아야 합니다"
    no_error_dialogs()

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    t0 = time.time()
    parts = [LeRobotDataset(repo_id, root=Path(root)) for repo_id, root in zip(args.repo_ids, args.roots)]
    for ds, root in zip(parts, args.roots):
        print(f"[merge] {root}: 에피소드 {ds.meta.total_episodes}, 프레임 {ds.meta.total_frames}, 문장 {len(ds.meta.tasks)}가지", flush=True)
    merged = merge_datasets(parts, output_repo_id=args.repo_id, output_dir=out)
    n_ep, n_fr = sum(d.meta.total_episodes for d in parts), sum(d.meta.total_frames for d in parts)
    print(f"[merge] 완료: {out} | 에피소드 {merged.meta.total_episodes} (원본 합 {n_ep}), 프레임 {merged.meta.total_frames} (원본 합 {n_fr}), "
          f"문장 {len(merged.meta.tasks)}가지, {time.time() - t0:.0f}초")
    assert merged.meta.total_episodes == n_ep and merged.meta.total_frames == n_fr, "합친 수가 원본의 합과 다릅니다"


if __name__ == "__main__":
    main()
