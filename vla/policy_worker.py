"""SmolVLA 행동 출력 작업자: 모델을 한 번 올려 두고, 표준 입력으로 관측 파일 경로를 받아 행동 묶음을 돌려준다.

Isaac 쪽 스크립트(scripts/vla_pick_eval.py)가 이 파일을 LeRobot 을 설치한 별도 환경의 파이썬으로 띄운다
(Isaac 환경에는 LeRobot 을 설치할 수 없어 두 프로세스로 나눔. 7단계 VLM 작업자와 같은 방식).

주고받는 형식 (한 줄에 JSON 하나. 답은 앞에 표시 문자열을 붙여 다른 출력과 구분)
  시작하면:  @@VLA@@{"ready": true, "load_s": ..., "gpu_mb": ..., "chunk_size": 50, "action_dim": 11}
  요청:      {"obs": "<npz 경로>"}   npz: images (B, H, W, 3) uint8, state (B, 11) float32, tasks (B,) 문자열
  답:        @@VLA@@{"act": "<npy 경로>", "latency_s": ..., "batch": B}   npy: (B, chunk_size, 11) float32 (후처리까지 한 값)
  끝:        {"quit": true}
"""

import argparse
import json
import sys
import time

MARK = "@@VLA@@"


def reply(obj):
    sys.stdout.write(MARK + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="체크포인트의 pretrained_model 폴더")
    ap.add_argument("--image_key", default="observation.images.head")
    args = ap.parse_args()
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    import numpy as np
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
    cfg = policy.config
    reply({"ready": True, "load_s": round(time.time() - t0, 2), "gpu_mb": round(torch.cuda.memory_allocated() / 2**20),
           "chunk_size": cfg.chunk_size, "action_dim": cfg.output_features["action"].shape[0]})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        t0 = time.time()
        data = np.load(req["obs"], allow_pickle=False)
        images = torch.from_numpy(data["images"]).permute(0, 3, 1, 2).float() / 255.0
        batch = {args.image_key: images, "observation.state": torch.from_numpy(data["state"]).float(),
                 "task": [str(t) for t in data["tasks"]]}
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(pre(batch))
            action = post(chunk)
        torch.cuda.synchronize()
        out = req["obs"][: -len(".npz")] + "_act.npy"
        np.save(out, action.float().cpu().numpy())
        reply({"act": out, "latency_s": round(time.time() - t0, 3), "batch": int(images.shape[0]),
               "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)})


if __name__ == "__main__":
    main()
