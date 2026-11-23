"""학습한 SmolVLA 체크포인트로 행동 출력(action chunk)을 실행해 GPU 메모리와 지연 시간을 잰다.

데이터셋의 한 프레임(카메라 사진들 + 로봇 상태 + 작업 문장)을 넣고, 체크포인트에 저장된 전처리·후처리를 그대로 쓴다.
첫 실행은 준비 시간이 섞이므로 따로 적고, 그 뒤 --repeat 번의 중앙값을 낸다. 데이터셋에 기록된 행동과의 차이도 참고로 출력한다.

LeRobot 을 설치한 별도 환경에서 실행한다 (Isaac 환경이 아님).
예) python infer_probe.py --ckpt outputs/smolvla_300/checkpoints/last/pretrained_model
"""

import argparse
import json
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="체크포인트의 pretrained_model 폴더")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--frame", type=int, default=100, help="입력으로 쓸 프레임 번호")
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
    load_s = time.time() - t0
    cfg = policy.config
    print(f"[infer] 불러오기 {load_s:.1f}초, GPU {torch.cuda.memory_allocated() / 2**30:.2f}GB | "
          f"입력 {{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in cfg.input_features.items())}}} | "
          f"출력 {{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in cfg.output_features.items())}}} | "
          f"chunk_size {cfg.chunk_size}, n_action_steps {cfg.n_action_steps}, 상태·행동 상한 {cfg.max_state_dim}·{cfg.max_action_dim}, "
          f"denoise 단계 {cfg.num_steps}")

    ds = LeRobotDataset(args.dataset)
    item = ds[args.frame]
    obs = {k: v for k, v in item.items() if k.startswith("observation.")}
    obs["task"] = item["task"]
    print("[infer] 관측: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in obs.items() if hasattr(v, "shape")) + f" | 작업 문장: {obs['task']!r}")

    def run_once():
        batch = pre(dict(obs))
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        return chunk, time.perf_counter() - t

    torch.cuda.reset_peak_memory_stats()
    chunk, first = run_once()
    times = [run_once()[1] for _ in range(args.repeat)]
    action = post(chunk)
    action = action[0] if action.ndim == 3 else action  # (chunk, 행동 차원)
    truth = torch.stack([ds[args.frame + i]["action"] for i in range(min(10, action.shape[0]))])
    diff = (action[: truth.shape[0]].float().cpu() - truth).abs().mean().item()
    result = {
        "chunk_shape": list(chunk.shape), "first_s": round(first, 3), "median_s": round(statistics.median(times), 3),
        "min_s": round(min(times), 3), "max_s": round(max(times), 3),
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "mean_abs_diff_first10_vs_dataset": round(diff, 3),
    }
    print("[infer] 첫 행동 (후처리 뒤): " + ", ".join(f"{v:.2f}" for v in action[0].tolist()))
    print("[infer] 데이터의 행동:        " + ", ".join(f"{v:.2f}" for v in truth[0].tolist()))
    print("[infer] RESULT " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
