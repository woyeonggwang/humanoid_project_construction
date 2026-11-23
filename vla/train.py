"""SmolVLA 미세조정: 로컬 LeRobot 데이터셋(vla/convert_to_lerobot.py 가 만든 것)으로 lerobot-train 을 부른다.

- 기본 설정 그대로 학습한다 (행동 부분만 학습, 시각 인코더·VLM 고정). --train_vlm 이면 VLM 도 학습한다 (시각 인코더는 고정). 데이터의 머리 카메라를 smolvla_base 의 camera1 자리에 넣는다
- 이 프로세스가 쓸 GPU 메모리에 상한을 건다 (--cap_gb). 넘으면 시스템 메모리로 넘쳐 느려지는 대신 오류로 멈춘다
- Windows 에서는 모델 이름을 그대로 넘기면 경로로 바뀌어 실패하므로 먼저 내려받아 로컬 경로를 넘긴다 (CLAUDE.md 8단계 준비 시행착오)

LeRobot 을 설치한 별도 환경에서 실행한다 (Isaac 환경이 아님). 학습 중에는 Isaac 을 띄우지 않는다.
예) python vla/train.py --dataset_root data/vla_pick/lerobot_v1 --steps 2000 --batch 32 --out data/vla_pick/train/short_2k
"""

import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="LeRobot 데이터셋 폴더")
    ap.add_argument("--repo_id", default="local/g1_vla_pick")
    ap.add_argument("--rename_map", default='{"observation.images.head": "observation.images.camera1"}')
    ap.add_argument("--policy", default="lerobot/smolvla_base")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--save_freq", type=int, default=None, help="몇 스텝마다 체크포인트를 저장할지 (기본: 끝에 한 번)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_freq", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap_gb", type=float, default=10.5)
    ap.add_argument("--train_vlm", action="store_true",
                    help="VLM(언어·사진을 함께 읽는 부분)도 학습. 시각 인코더는 고정. 명령의 낱말을 구분해야 하는 과제용 (메모리가 더 듦)")
    args = ap.parse_args()

    import torch

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(args.cap_gb / total, 1.0))
    policy_path = args.policy
    if not os.path.isdir(policy_path):
        from huggingface_hub import snapshot_download

        policy_path = snapshot_download(policy_path)
    sys.argv = [
        "lerobot-train",
        f"--policy.path={policy_path}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={os.path.abspath(args.dataset_root)}",
        f"--rename_map={args.rename_map}",
        f"--batch_size={args.batch}",
        f"--steps={args.steps}",
        f"--num_workers={args.workers}",
        f"--output_dir={args.out}",
        f"--seed={args.seed}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.train_expert_only={'false' if args.train_vlm else 'true'}",
        "--policy.freeze_vision_encoder=true",
        "--trackio.enable=false",
        f"--log_freq={args.log_freq}",
        "--save_checkpoint=true",
        f"--save_freq={args.save_freq or args.steps}",
    ]
    print("[train] " + " ".join(a for a in sys.argv if not a.startswith(("--policy.path", "--dataset.root"))), flush=True)
    from lerobot.scripts.lerobot_train import main as train_main

    t0 = time.time()
    train_main()
    print(f"[train] 끝: {time.time() - t0:.0f}초, GPU 최대 {torch.cuda.max_memory_allocated() / 2**30:.2f}GB (잡은 값), "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f}GB (예약)", flush=True)


if __name__ == "__main__":
    main()
