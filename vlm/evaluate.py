"""test 세트로 VLM 정확도 측정 (파인튜닝 전 기준점 / 파인튜닝 후 비교 모두 이 스크립트로, 같은 프롬프트).

실행 (vlm 폴더, 별도 파인튜닝 환경):
  python evaluate.py --model Qwen/Qwen3.5-0.8B --name base_08b
  python evaluate.py --model Qwen/Qwen3.5-0.8B --adapter outputs/qwen35_08b/lora --name ft_08b

지표
  - JSON 유효율: 출력에서 JSON 을 꺼내 파싱할 수 있는 비율
  - 정확도: plan 과 error 가 정답과 완전히 같은 비율
  - 거부 정확도: 거부해야 하는 명령(없는 상자·놓을 곳에 상자 있음·할 수 없는 일)을 맞는 error 로 거부한 비율
  - '본 패턴' / '처음 보는 패턴' 으로 나눠 출력
  - 색별·받침대별 정확도: 명령에 나온 상자 색별, 그 상자가 실제로 놓인 받침대별 (사진에서 상자를 제대로 찾았는지)
결과: results/<이름>.jsonl 에 문장별 예측 저장
"""

import argparse
import json
import os
import time

import torch

from common import extract_json, generate, load_image, load_model, normalize


def summarize(rows) -> str:
    n = len(rows)
    if n == 0:
        return "0개"
    valid = sum(r["valid"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    rej = [r for r in rows if json.loads(r["output"]).get("error")]
    s = f"{n}개 | JSON 유효 {valid / n:.1%} | 정확도 {correct / n:.1%}"
    if rej:
        s += f" | 거부 정확도 {sum(r['correct'] for r in rej) / len(rej):.1%} ({len(rej)}개)"
    return s


def acc(rows) -> str:
    return f"{sum(r['correct'] for r in rows) / len(rows):.1%} ({len(rows)}개)" if rows else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face 모델 이름 또는 로컬 폴더")
    ap.add_argument("--adapter", default=None, help="LoRA 어댑터 폴더 (없으면 모델 그대로)")
    ap.add_argument("--name", default=None, help="결과 파일 이름")
    ap.add_argument("--data", default="data", help="test.jsonl 과 images/ 가 있는 폴더")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="앞뒤에서 반씩 N개만 평가")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(os.path.join(args.data, f"{args.split}.jsonl"), encoding="utf-8")]
    if args.limit:
        half = args.limit // 2
        rows = rows[:half] + rows[-(args.limit - half):]

    print(f"모델 불러오는 중: {args.model}" + (f" + {args.adapter}" if args.adapter else ""))
    model, processor = load_model(args.model, args.adapter)
    print(f"파라미터: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        images = [load_image(os.path.join(args.data, r["image"])) for r in batch]
        outs = generate(model, processor, images, [r["input"] for r in batch])
        for r, gen in zip(batch, outs):
            pred = extract_json(gen)
            r["raw"] = gen
            r["valid"] = pred is not None
            r["correct"] = pred is not None and normalize(pred) == normalize(json.loads(r["output"]))
        print(f"  {min(i + args.batch, len(rows))}/{len(rows)}", end="\r")
    elapsed = time.time() - t0

    name = args.name or (args.adapter or args.model).replace("/", "_").replace("\\", "_")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n\n=== {args.model}{' + ' + args.adapter if args.adapter else ''} | {args.data}/{args.split}.jsonl "
          f"({elapsed:.0f}초, 문장당 {elapsed / len(rows):.2f}초, GPU 최대 {torch.cuda.max_memory_allocated() / 2**20:.0f}MB) ===")
    print("전체          :", summarize(rows))
    print("본 패턴       :", summarize([r for r in rows if not r.get("heldout")]))
    print("처음 보는 패턴:", summarize([r for r in rows if r.get("heldout")]))
    need_image = {"move", "move_wait", "pick", "goto_box", "move_to_empty", "place"}
    print("사진을 봐야 하는 명령:", acc([r for r in rows if r["template"] in need_image]),
          "| 사진 없이 되는 명령:", acc([r for r in rows if r["template"] not in need_image]))
    print("거부 종류별   :", ", ".join(
        f"{e} {acc([r for r in rows if json.loads(r['output']).get('error') == e])}"
        for e in ("object_not_found", "target_occupied", "unsupported_command")))
    print("상자 색별     :", ", ".join(f"{c} {acc([r for r in rows if r.get('color') == c])}"
                                      for c in ("red", "blue", "green", "yellow")))
    print("상자가 놓인 받침대별:", ", ".join(f"{s} {acc([r for r in rows if r.get('src') == s])}"
                                            for s in ("stand_L", "stand_C", "stand_R")),
          "| 사진에 없는 상자", acc([r for r in rows if "color" in r and r.get("src") is None]))
    print("템플릿별      :", ", ".join(f"{t} {acc([r for r in rows if r['template'] == t])}"
                                      for t in sorted({r["template"] for r in rows})))
    wrong = [r for r in rows if not r["correct"]][:4]
    if wrong:
        print("\n틀린 예시:")
        for r in wrong:
            print("  사진:", r["image"], "| 장면:", r["scene"])
            print("  입력:", r["input"])
            print("  정답:", r["output"])
            print("  출력:", r["raw"][:200].replace("\n", " "))
    print(f"\n문장별 결과: results/{name}.jsonl")


if __name__ == "__main__":
    main()
