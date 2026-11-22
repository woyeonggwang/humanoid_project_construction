"""VLM 평가·학습·실행이 같이 쓰는 부분: 모델 불러오기, 프롬프트 만들기, 생성, JSON 꺼내기.

별도 파인튜닝 환경(transformers, peft)에서 실행한다. Isaac 환경에서는 import 하지 않는다.
"""

import json
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from schema import SYSTEM_PROMPT


def load_model(model_id: str, adapter: str | None = None):
    """모델과 processor 를 불러온다. adapter 가 있으면 LoRA 어댑터를 얹는다."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, processor


def build_messages(text: str, system: str = SYSTEM_PROMPT):
    """사진 한 장 + 명령. 사진은 processor 에 따로 넘기므로 여기서는 자리만 표시한다."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
    ]


def build_prompt(processor, text: str, system: str = SYSTEM_PROMPT) -> str:
    # enable_thinking=False: 사고 모드를 끄고 바로 답하게 함 (학습·평가·실행이 모두 같은 형식)
    return processor.apply_chat_template(build_messages(text, system), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def generate(model, processor, images, texts, max_new_tokens: int = 128, system: str = SYSTEM_PROMPT):
    """사진·명령 묶음 → 생성한 문자열 묶음."""
    processor.tokenizer.padding_side = "left"  # 묶음 생성은 왼쪽을 채워야 문장 끝이 맞춰짐
    prompts = [build_prompt(processor, t, system) for t in texts]
    enc = processor(text=prompts, images=list(images), return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=processor.tokenizer.pad_token_id)
    return [processor.tokenizer.decode(ids[enc["input_ids"].shape[1]:], skip_special_tokens=True) for ids in out]


def extract_json(text: str):
    """출력에서 첫 '{' 부터 마지막 '}' 까지 잘라 파싱. 실패하면 None"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def normalize(obj):
    """비교용 정규화: plan 리스트와 error 만 본다"""
    if not isinstance(obj, dict):
        return None
    plan = obj.get("plan", [])
    if not isinstance(plan, list):
        return None
    return {"plan": plan, "error": obj.get("error")}
