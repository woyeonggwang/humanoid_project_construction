"""VLM 추론 작업자: 모델을 한 번만 불러 두고, 요청을 한 줄씩 받아 답을 한 줄씩 낸다.

Isaac 환경과 라이브러리가 섞이지 않게 별도 파인튜닝 환경의 파이썬으로 따로 실행한다
(scripts/vlm_pick_place_demo.py 가 이 파일을 하위 프로세스로 띄우고 표준 입출력으로 주고받음).

요청 (표준 입력, JSON 한 줄): {"image": 사진 파일 경로, "text": 한국어 명령}   끝낼 때 {"quit": true}
응답 (표준 출력, '@@VLM@@' 로 시작하는 줄): {"raw": 모델 출력, "json": 꺼낸 JSON 또는 null, "latency_s": 생성 시간, "gpu_mb": 이 프로세스의 GPU 메모리 최대값}
처음 준비되면 {"ready": true, "load_s": ..., "gpu_mb": ...} 를 한 번 낸다.

직접 시험: python infer_worker.py --model Qwen/Qwen3.5-0.8B --adapter outputs/qwen35_08b/lora
"""

import argparse
import json
import sys
import time

import torch

from common import extract_json, generate, load_image, load_model

MARK = "@@VLM@@"


def reply(obj):
    print(MARK + json.dumps(obj, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    t0 = time.time()
    model, processor = load_model(args.model, args.adapter)
    reply({"ready": True, "load_s": round(time.time() - t0, 2), "gpu_mb": int(torch.cuda.max_memory_allocated() / 2**20)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        t0 = time.time()
        raw = generate(model, processor, [load_image(req["image"])], [req["text"]])[0]
        reply({"raw": raw, "json": extract_json(raw), "latency_s": round(time.time() - t0, 3),
               "gpu_mb": int(torch.cuda.max_memory_allocated() / 2**20)})


if __name__ == "__main__":
    main()
