"""SmolVLA 미세조정이 정해진 GPU 메모리 안에서 도는지 재는 실험 (LeRobot 의 lerobot-train 을 그대로 부른다).

- 이 프로세스가 쓸 수 있는 GPU 메모리에 상한을 건다 (--cap_gb). 넘으면 시스템 메모리로 넘쳐 느려지는 대신 오류로 멈춘다.
- 학습이 끝나면 PyTorch 가 잡은 최대 메모리(allocated / reserved)와, 실행 중 nvidia-smi 로 본 GPU 전체 사용량의 최대값을 출력한다.
- 스텝당 시간은 lerobot-train 이 로그에 찍는 값(updt_s = 갱신, data_s = 데이터 읽기)을 본다.

LeRobot 을 설치한 별도 환경에서 실행한다 (Isaac 환경이 아님).
예) python train_probe.py --batch 8 --steps 60 --out outputs/probe_b8
    python train_probe.py --batch 16 --steps 300 --save --out outputs/smolvla_300   (체크포인트 저장)
    python train_probe.py --batch 4 --steps 60 --train_vlm --out outputs/probe_vlm_b4  (VLM 까지 학습)
"""

import argparse
import json
import subprocess
import sys
import threading
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace", help="Hugging Face 의 LeRobot 데이터셋")
    ap.add_argument("--rename_map", default='{"observation.images.up": "observation.images.camera1", '
                                            '"observation.images.side": "observation.images.camera2"}',
                    help="데이터셋의 카메라 이름 → smolvla_base 가 기대하는 이름 (JSON)")
    ap.add_argument("--policy", default="lerobot/smolvla_base")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--unfreeze_vision", action="store_true", help="시각 인코더도 학습 (기본: 고정)")
    ap.add_argument("--train_vlm", action="store_true", help="VLM 전체도 학습 (기본: 행동 전문가 부분만)")
    ap.add_argument("--save", action="store_true", help="끝에 체크포인트 저장")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap_gb", type=float, default=10.5, help="이 프로세스의 GPU 메모리 상한 (GB)")
    ap.add_argument("--log_freq", type=int, default=10)
    args = ap.parse_args()

    import torch

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(args.cap_gb / total, 1.0))

    # Windows 에서는 --policy.path=lerobot/smolvla_base 가 경로로 바뀌며 'lerobot\smolvla_base' 가 되어 내려받기에 실패한다
    # (LeRobot 0.7.0). 먼저 내려받아 로컬 폴더 경로를 넘긴다.
    import os

    policy_path = args.policy
    if not os.path.isdir(policy_path):
        from huggingface_hub import snapshot_download

        policy_path = snapshot_download(policy_path)

    peak = {"smi": 0}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                peak["smi"] = max(peak["smi"], int(out.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass
            stop.wait(0.5)

    sys.argv = [
        "lerobot-train",
        f"--policy.path={policy_path}",
        f"--dataset.repo_id={args.dataset}",
        f"--rename_map={args.rename_map}",
        f"--batch_size={args.batch}",
        f"--steps={args.steps}",
        f"--num_workers={args.workers}",
        f"--output_dir={args.out}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.freeze_vision_encoder={'false' if args.unfreeze_vision else 'true'}",
        f"--policy.train_expert_only={'false' if args.train_vlm else 'true'}",
        "--trackio.enable=false",
        f"--log_freq={args.log_freq}",
        f"--save_checkpoint={'true' if args.save else 'false'}",
        f"--save_freq={args.steps}",
    ]
    print("[probe] " + " ".join(a for a in sys.argv if not a.startswith("--policy.path")), flush=True)
    from lerobot.scripts.lerobot_train import main as train_main

    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    status = "ok"
    try:
        train_main()
    except torch.OutOfMemoryError as e:
        status = "out_of_memory: " + str(e).splitlines()[0][:160]
    finally:
        stop.set()
    result = {
        "status": status, "batch": args.batch, "steps": args.steps,
        "freeze_vision_encoder": not args.unfreeze_vision, "train_expert_only": not args.train_vlm,
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "peak_nvidia_smi_gb": round(peak["smi"] / 1024, 2), "wall_s": round(time.time() - t0, 1),
    }
    print("[probe] RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
