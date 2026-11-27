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

# ---------------------------------------------------------------------------
# 9단계 전체 연결용 (schema 이름 "full"): 집는 받침대의 색 상자를 내려놓을 받침대의 색 판으로 옮긴다
# ---------------------------------------------------------------------------
# 장소: 사진에서 왼쪽이 상자가 놓인 받침대, 오른쪽이 색 판이 깔린 받침대
PLACES = ["pick_stand", "place_stand"]
PAD_WORDS = ["판"]
SKILLS_FULL = ["go_to", "pick", "place"]
# pick·place 는 팔을 움직이는 모델(VLA)에 그대로 넘길 문장을 담는다. VLA 학습 때 쓴 표현 가운데 하나로 통일한다
VLA_COLOR = {"red": "빨간", "blue": "파란", "green": "초록", "yellow": "노란"}


def pick_instruction(color: str) -> str:
    return f"{VLA_COLOR[color]} 상자를 집어"


def place_instruction(color: str) -> str:
    return f"{VLA_COLOR[color]} 판 위에 내려놔"


def instruction_color(text: str):
    """pick·place 문장의 색 (못 찾으면 None)."""
    for color, word in VLA_COLOR.items():
        if text.strip().startswith(word):
            return color
    return None


SYSTEM_PROMPT_FULL = (
    "너는 상자를 옮기는 로봇의 명령 해석기다. 로봇 머리 카메라 사진과 한국어 명령을 보고 JSON 계획으로만 답한다.\n"
    "장소: pick_stand(색 상자가 놓인 받침대, 사진 왼쪽), place_stand(색 판이 깔린 받침대, 사진 오른쪽)\n"
    "색: 빨간, 파란, 초록, 노란 (상자와 판)\n"
    "기술: go_to(target), pick(instruction), place(instruction)\n"
    'pick 의 instruction 은 "<색> 상자를 집어", place 의 instruction 은 "<색> 판 위에 내려놔" 로 쓴다. '
    "상자 하나를 옮길 때마다 go_to pick_stand, pick, go_to place_stand, place 순서로 적는다\n"
    '형식: {"plan": [{"skill": ..., ...}]}\n'
    '사진에 없는 색의 상자나 판이면 {"plan": [], "error": "object_not_found"}, '
    '놓을 판에 이미 상자가 있으면 {"plan": [], "error": "target_occupied"}, '
    '할 수 없는 일이면 {"plan": [], "error": "unsupported_command"}'
)

# ---------------------------------------------------------------------------
# 11단계 (schema 이름 "site"): 9단계의 옮기기 + 철골 용접
# ---------------------------------------------------------------------------
PLACES_SITE = ["pick_stand", "place_stand", "steel_frame"]
SKILLS_SITE = ["go_to", "pick", "place", "weld"]
ERRORS_SITE = ERRORS + ["unknown_target"]
# 용접선 이름은 humanoid_tasks/g1_carry/site_layout.py (assets/weld_seams.json) 의 이름과 같다.
# 철골 기둥은 A·B·C (철골을 마주 보고 왼쪽부터). 세로 이음부만 손이 닿는다: A 기둥 1개, B 기둥 2개(왼쪽·오른쪽), C 기둥 1개
WELD_VERTICAL = {"A": ["column_A_seam_2"], "B": ["column_B_seam_2", "column_B_seam_4"], "C": ["column_C_seam_2"]}
WELD_B_SIDE = {"left": "column_B_seam_2", "right": "column_B_seam_4"}
WELD_ALL = [s for c in ("A", "B", "C") for s in WELD_VERTICAL[c]]

SYSTEM_PROMPT_SITE = (
    "너는 공사장에서 상자를 옮기고 철골을 용접하는 로봇의 명령 해석기다. 로봇 머리 카메라 사진과 한국어 명령을 보고 JSON 계획으로만 답한다.\n"
    "장소: pick_stand(색 상자가 놓인 받침대, 사진 왼쪽), place_stand(색 판이 깔린 받침대, 사진 오른쪽), steel_frame(철골)\n"
    "색: 빨간, 파란, 초록, 노란 (상자와 판)\n"
    "기술: go_to(target), pick(instruction), place(instruction), weld(seam)\n"
    'pick 의 instruction 은 "<색> 상자를 집어", place 의 instruction 은 "<색> 판 위에 내려놔" 로 쓴다. '
    "상자 하나를 옮길 때마다 go_to pick_stand, pick, go_to place_stand, place 순서로 적는다\n"
    "철골 기둥은 A, B, C. 용접할 수 있는 세로 이음부: column_A_seam_2(A 기둥), column_B_seam_2(B 기둥 왼쪽), "
    "column_B_seam_4(B 기둥 오른쪽), column_C_seam_2(C 기둥). 가로 이음부는 손이 닿지 않는다. "
    "용접은 go_to steel_frame 한 번 뒤에 weld 를 용접선마다 차례로 적는다. 기둥만 말하면 그 기둥의 세로 이음부를 모두 적는다\n"
    '형식: {"plan": [{"skill": ..., ...}]}\n'
    '사진에 없는 색의 상자나 판이면 {"plan": [], "error": "object_not_found"}, '
    '놓을 판에 이미 상자가 있으면 {"plan": [], "error": "target_occupied"}, '
    '없는 기둥이면 {"plan": [], "error": "unknown_target"}, '
    '손이 닿지 않는 이음부이거나 할 수 없는 일이면 {"plan": [], "error": "unsupported_command"}'
)

SYSTEM_PROMPTS = {"stands": SYSTEM_PROMPT, "full": SYSTEM_PROMPT_FULL, "site": SYSTEM_PROMPT_SITE}
