"""VLM 판단 층의 명령 체계 (받침대, 상자, 기술, 거부 코드, 시스템 프롬프트).

데이터 생성(scripts/make_vlm_data.py)·학습(train.py)·평가(evaluate.py)·실행(infer_worker.py)이 모두 이 파일을 기준으로 한다.
다른 라이브러리를 쓰지 않으므로 Isaac 환경과 별도 파인튜닝 환경 양쪽에서 import 할 수 있다.
"""

# 받침대: JSON 에 쓰는 이름 → 한국어 표현들. 왼쪽·오른쪽은 로봇이 볼 때(= 머리 카메라 사진에서) 기준
STANDS = {
    "stand_L": ["왼쪽 받침대", "왼편 받침대", "맨 왼쪽 받침대", "왼쪽 대"],
    "stand_C": ["가운데 받침대", "중앙 받침대", "가운데 대"],
    "stand_R": ["오른쪽 받침대", "오른편 받침대", "맨 오른쪽 받침대", "오른쪽 대"],
}

# 상자 색: 이름 → 한국어 표현들
COLORS = {
    "red": ["빨간", "빨간색", "붉은"],
    "blue": ["파란", "파란색", "푸른"],
    "green": ["초록", "초록색", "녹색"],
    "yellow": ["노란", "노란색"],
}
BOX_WORDS = ["상자", "박스"]


def box_name(color: str) -> str:
    return f"{color}_box"


OBJECTS = [box_name(c) for c in COLORS]

# 기술 목록 (pick·place 는 IK + 붙이기, go_to 는 걷기 정책, wait·stop 은 스크립트)
SKILLS = ["pick", "place", "go_to", "wait", "stop"]

# 거부할 때 쓰는 에러 코드
ERRORS = ["object_not_found", "target_occupied", "unsupported_command"]

SYSTEM_PROMPT = (
    "너는 상자를 옮기는 로봇의 명령 해석기다. 로봇 머리 카메라 사진과 한국어 명령을 보고 JSON 계획으로만 답한다.\n"
    "받침대: stand_L(왼쪽), stand_C(가운데), stand_R(오른쪽) — 사진에서 보이는 위치 기준\n"
    "물건: red_box(빨간 상자), blue_box(파란 상자), green_box(초록 상자), yellow_box(노란 상자)\n"
    "기술: pick(object, from), place(to), go_to(target), wait(sec), stop\n"
    '형식: {"plan": [{"skill": ..., ...}]}\n'
    '사진에 없는 상자면 {"plan": [], "error": "object_not_found"}, '
    '놓을 받침대에 이미 상자가 있으면 {"plan": [], "error": "target_occupied"}, '
    '할 수 없는 일이면 {"plan": [], "error": "unsupported_command"}'
)
