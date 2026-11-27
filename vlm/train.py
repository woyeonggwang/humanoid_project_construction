"""VLM LoRA 파인튜닝: 머리 카메라 사진 + 한국어 명령 → 계획 JSON.

실행 (vlm 폴더, 별도 파인튜닝 환경):
  python train.py                                             # 기본: unsloth/Qwen3.5-0.8B → outputs/qwen35_08b/lora
  python train.py --model unsloth/Qwen3.5-2B --out outputs/qwen35_2b
  python train.py --schema full --data data_full --out outputs/qwen35_08b_full   # 9단계 전체 연결용
평가:
  python evaluate.py --model Qwen/Qwen3.5-0.8B --adapter outputs/qwen35_08b/lora --name ft_08b

- 프롬프트는 evaluate.py 와 완전히 같은 형식 (common.build_prompt)
- 답변(JSON)만 학습: labels 에서 프롬프트(시스템·사진·명령) 부분은 -100 으로 가려 loss 에서 뺀다
- LoRA 는 언어 쪽 층과 시각 쪽 층 모두에 붙인다 (--no_vision 이면 언어 쪽만)
- 모델은 Unsloth(FastVisionModel)로 불러온다. transformers 로 그대로 불러와 학습하면 묶음 1 에 GPU 3.3GB(중간값을 다시 계산하는
  방식을 켜도), Unsloth 는 2.0GB 였다 (2026-11-22 측정, 사진 448x448 + 프롬프트 약 440 토큰). 일반 방식으로 묶음 4 를 돌렸을 때는
  다른 학습과 함께 GPU 메모리를 넘겨 시스템 전체가 느려졌음
- --mem_cap_gb: 이 프로세스의 GPU 메모리 상한. 넘으면 느려지는 대신 오류로 멈춘다
"""

import argparse
import json
import math
import os
import time

from unsloth import FastVisionModel  # unsloth 는 transformers 보다 먼저 import 해야 최적화가 적용됨
import torch
from transformers import Trainer, TrainingArguments

from common import build_prompt, load_image
from schema import SYSTEM_PROMPTS

class Rows(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str):
        self.dir = data_dir
        self.rows = [json.loads(line) for line in open(os.path.join(data_dir, f"{split}.jsonl"), encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"image": load_image(os.path.join(self.dir, r["image"])), "input": r["input"], "output": r["output"]}


class Collator:
    """묶음을 모델 입력으로 바꾼다. 프롬프트 부분의 labels 는 -100."""

    def __init__(self, processor, system: str):
        self.processor = processor
        self.system = system
        tok = processor.tokenizer
        self.end = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token

    def __call__(self, batch):
        proc = self.processor
        proc.tokenizer.padding_side = "right"
        images = [b["image"] for b in batch]
        prompts = [build_prompt(proc, b["input"], self.system) for b in batch]
        full = [p + b["output"] + self.end for p, b in zip(prompts, batch)]
        enc = proc(text=full, images=images, return_tensors="pt", padding=True)
        # 프롬프트 길이(사진 자리 토큰이 펼쳐진 뒤 기준)를 재서 그만큼 가린다
        prompt_len = proc(text=prompts, images=images, return_tensors="pt", padding=True)["attention_mask"].sum(dim=1)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, n in enumerate(prompt_len.tolist()):
            labels[i, :n] = -100
        enc["labels"] = labels
        return dict(enc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    ap.add_argument("--out", default="outputs/qwen35_08b")
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--batch", type=int, default=8, help="한 번에 GPU 에 올리는 묶음 크기")
    ap.add_argument("--accum", type=int, default=1, help="몇 묶음을 모아 한 번 갱신할지 (실제 묶음 = batch × accum)")
    ap.add_argument("--max_steps", type=int, default=-1, help="시험용: 이 스텝 수만 학습")
    ap.add_argument("--mem_cap_gb", type=float, default=10.0,
                    help="이 프로세스가 쓸 GPU 메모리 상한 (GB). 넘으면 느려지는 대신 오류로 멈춤")
    ap.add_argument("--no_vision", action="store_true", help="시각 쪽 층에는 LoRA 를 붙이지 않음")
    ap.add_argument("--schema", choices=list(SYSTEM_PROMPTS), default="stands",
                    help="명령 체계: stands = 7단계(받침대 3개), full = 9단계 전체 연결(상자 → 판). 시스템 프롬프트가 달라짐")
    args = ap.parse_args()

    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(args.mem_cap_gb / total_gb, 1.0))
    model, processor = FastVisionModel.from_pretrained(args.model, load_in_4bit=False, dtype=torch.bfloat16,
                                                       use_gradient_checkpointing="unsloth")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=not args.no_vision, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.r, lora_alpha=args.r, lora_dropout=0, bias="none", random_state=42)
    FastVisionModel.for_training(model)
    model.print_trainable_parameters()

    train, val = Rows(args.data, "train"), Rows(args.data, "val")
    collator = Collator(processor, SYSTEM_PROMPTS[args.schema])
    sample = collator([train[0]])
    n_ans = int((sample["labels"][0] != -100).sum())
    print(f"train {len(train)}, val {len(val)} | 예시: 전체 {sample['input_ids'].shape[1]} 토큰 중 답변 {n_ans} 토큰만 학습")
    print("  답변 부분:", repr(processor.tokenizer.decode(sample["input_ids"][0][sample["labels"][0] != -100])))

    total_steps = math.ceil(len(train) / (args.batch * args.accum)) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    print(f"총 {int(total_steps)} 스텝, warmup {warmup_steps} 스텝")

    trainer = Trainer(
        model=model,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        args=TrainingArguments(
            output_dir=f"{args.out}/checkpoints",
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            max_steps=args.max_steps,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=150,
            save_strategy="no",
            bf16=True,
            optim="adamw_torch",
            report_to="none",
            seed=42,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        ),
    )
    t0 = time.time()
    trainer.train()
    print(f"학습 시간: {(time.time() - t0) / 60:.1f}분, GPU 최대 {torch.cuda.max_memory_allocated() / 2**20:.0f}MB")

    model.save_pretrained(f"{args.out}/lora")
    processor.save_pretrained(f"{args.out}/lora")
    print(f"저장 완료: {args.out}/lora")


if __name__ == "__main__":
    main()
