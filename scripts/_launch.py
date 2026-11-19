"""train.py / play.py 공통: 과제를 등록한 뒤 Isaac Lab 원본 스크립트를 그대로 실행한다.

명령줄 인자(--task, --headless 등)는 손대지 않고 원본 스크립트로 넘어간다.
"""

import importlib.util
import os
import runpy
import sys
from pathlib import Path


def _find_isaaclab() -> Path:
    """Isaac Lab 소스 폴더 찾기: 환경 변수 ISAACLAB_PATH, 없으면 설치된 isaaclab 패키지 위치에서 계산.

    pip install -e source/isaaclab 로 설치했다면 패키지 위치는
    <IsaacLab>/source/isaaclab/isaaclab/__init__.py 이다.
    """
    if os.environ.get("ISAACLAB_PATH"):
        return Path(os.environ["ISAACLAB_PATH"])
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        raise RuntimeError("isaaclab 패키지를 찾을 수 없습니다. Isaac Lab 환경을 켰는지 확인하세요.")
    return Path(spec.origin).resolve().parents[3]


ISAACLAB_PATH = _find_isaaclab()
RSL_RL_DIR = ISAACLAB_PATH / "scripts" / "reinforcement_learning" / "rsl_rl"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(script_name: str) -> None:
    # humanoid_tasks 를 import 할 수 있게 프로젝트 루트를 검색 경로에 추가
    sys.path.insert(0, str(PROJECT_ROOT))
    # 원본 스크립트의 `import cli_args` 가 같은 폴더의 cli_args.py 를 찾게 함
    sys.path.insert(0, str(RSL_RL_DIR))

    import humanoid_tasks.g1_carry  # noqa: F401  ← 여기서 gym.register 실행

    runpy.run_path(str(RSL_RL_DIR / script_name), run_name="__main__")
