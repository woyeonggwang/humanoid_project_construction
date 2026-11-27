"""VLA 작업자(vla/policy_worker.py)와 VLM 작업자(vlm/infer_worker.py)를 별도 환경의 파이썬으로 띄우고 주고받는 쪽.

Isaac 환경에는 LeRobot 을 설치할 수 없어 두 프로세스로 나눈다 (CLAUDE.md 8단계). 관측은 npz 파일로 쓰고 경로를 표준 입력으로 보낸다.
isaaclab 을 import 하지 않으므로 앱이 켜지기 전에도 import 할 수 있다.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

MARK = "@@VLA@@"
WORKER = Path(__file__).resolve().parents[2] / "vla" / "policy_worker.py"


class VlaWorker:
    def __init__(self, python: str, ckpt: str):
        """python: LeRobot 을 설치한 환경의 파이썬 실행 파일. ckpt: 체크포인트의 pretrained_model 폴더."""
        self.tmp_dir = tempfile.mkdtemp(prefix="vla_obs_")
        self.log = os.path.join(self.tmp_dir, "worker.log")
        self.proc = subprocess.Popen([python, "-X", "utf8", str(WORKER), "--ckpt", os.path.abspath(ckpt)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.log, "w", encoding="utf-8"),
                                     text=True, encoding="utf-8")
        self.latencies: list[tuple[float, int]] = []  # (응답 시간 초, 묶음 크기)
        self.info = self._read()  # {"ready": True, "load_s", "gpu_mb", "chunk_size", "action_dim"}

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"VLA 작업자가 끝났습니다 ({self.log} 참고)")
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])

    def infer(self, images: np.ndarray, state: np.ndarray, tasks: list[str]) -> np.ndarray:
        """사진 (B, H, W, 3) uint8·상태 (B, 11)·명령 → 행동 묶음 (B, 50, 11)."""
        path = os.path.join(self.tmp_dir, "obs.npz")
        np.savez(path, images=images, state=state.astype(np.float32), tasks=np.array(tasks))
        self.proc.stdin.write(json.dumps({"obs": path}) + "\n")
        self.proc.stdin.flush()
        ans = self._read()
        self.latencies.append((ans["latency_s"], ans["batch"]))
        self.gpu_peak_mb = ans.get("gpu_peak_mb")
        return np.load(ans["act"])

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write(json.dumps({"quit": True}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)


class VlmWorker:
    """VLM 작업자(vlm/infer_worker.py)를 별도 환경의 파이썬으로 띄우고, 사진 경로와 명령을 보내 계획 JSON 을 받는다."""

    MARK = "@@VLM@@"

    def __init__(self, python: str, adapter: str, schema: str = "full", max_new_tokens: int = 320, model: str = "Qwen/Qwen3.5-0.8B"):
        vlm_dir = Path(__file__).resolve().parents[2] / "vlm"
        self.log = os.path.join(tempfile.mkdtemp(prefix="vlm_"), "worker.log")
        self.proc = subprocess.Popen(
            [python, "-X", "utf8", "infer_worker.py", "--model", model, "--adapter", os.path.abspath(adapter),
             "--schema", schema, "--max_new_tokens", str(max_new_tokens)],
            cwd=str(vlm_dir), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.log, "w", encoding="utf-8"),
            text=True, encoding="utf-8")
        self.info = self._read()  # {"ready": True, "load_s", "gpu_mb"}
        self.latencies: list[float] = []
        self.gpu_mb = self.info.get("gpu_mb")

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"VLM 작업자가 끝났습니다 ({self.log} 참고)")
            if line.startswith(self.MARK):
                return json.loads(line[len(self.MARK):])

    def ask(self, image_path: str, text: str) -> dict:
        """사진 한 장 + 명령 → {"raw": 모델 출력, "json": 꺼낸 JSON 또는 None, "latency_s": 생성 시간}."""
        self.proc.stdin.write(json.dumps({"image": os.path.abspath(image_path), "text": text}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        ans = self._read()
        self.latencies.append(ans["latency_s"])
        self.gpu_mb = ans.get("gpu_mb", self.gpu_mb)
        return ans

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write(json.dumps({"quit": True}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
