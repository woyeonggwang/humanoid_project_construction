"""9단계 전체 연결용: 한국어 명령과 정답 계획을 만드는 템플릿 (집는 받침대의 색 상자 → 내려놓을 받침대의 색 판).

장면의 실제 상태를 받아 (명령 문장, 정답 JSON, 부가 정보)를 만든다. 7단계의 templates.py 와 같은 방식:
- 템플릿마다 문장 패턴이 여러 개 있고, 마지막 패턴 하나는 test 전용(처음 보는 패턴)이다
- 계획의 pick·place 는 VLA 에 그대로 넘길 문장(instruction)을 담는다. 문장은 VLA 학습 때 쓴 표현 하나로 통일한다 (schema.pick_instruction·place_instruction)

장면(scene) 형식: {"boxes": {"L": 색 또는 None, "R": …},    집는 받침대의 왼쪽·오른쪽 자리에 놓인 상자
                  "pads": {"L": 색, "R": 색},              내려놓을 받침대의 왼쪽·오른쪽 판
                  "on_pad": {"L": 색 또는 None, "R": …}}   판 위에 이미 놓여 있는 상자
다른 라이브러리를 쓰지 않는다 (random, json 만).
"""

import json
import random

from schema import BOX_WORDS, COLORS, PAD_WORDS, pick_instruction, place_instruction
from templates import _last_info, eul, ga, ro


def eun(word: str) -> str:  # 은/는
    return word + ("은" if _last_info(word)[0] else "는")


def p_go(place: str):
    return {"skill": "go_to", "target": place}


def plan_move(pairs) -> list:
    """(상자 색, 판 색) 쌍들을 차례로 옮기는 계획."""
    steps = []
    for box, pad in pairs:
        steps += [p_go("pick_stand"), {"skill": "pick", "instruction": pick_instruction(box)},
                  p_go("place_stand"), {"skill": "place", "instruction": place_instruction(pad)}]
    return steps


def answer(plan=None, error=None) -> str:
    if error:
        return json.dumps({"plan": [], "error": error}, ensure_ascii=False)
    return json.dumps({"plan": plan}, ensure_ascii=False)


def facts(scene: dict) -> dict:
    boxes = [c for c in scene["boxes"].values() if c]
    pads = list(scene["pads"].values())
    full = [scene["pads"][s] for s in ("L", "R") if scene["on_pad"].get(s)]
    seen = set(boxes) | {c for c in scene["on_pad"].values() if c}  # 상자로 보이는 색 (같은 색의 판만 있는 경우는 '없는 상자')
    return {"boxes": boxes, "pads": pads, "empty_pads": [p for p in pads if p not in full], "full_pads": full,
            "absent_boxes": [c for c in COLORS if c not in seen], "absent_pads": [c for c in COLORS if c not in pads]}


class Gen:
    """템플릿 하나를 고르고 문장을 만든다. mode: 'train'(마지막 패턴 제외) / 'heldout'(마지막 패턴만) / 'all'(모든 패턴)."""

    WEIGHTS = [("move1", 34), ("move2", 30), ("pick", 11), ("goto", 9), ("unsupported", 16)]

    def __init__(self, rng: random.Random, mode: str):
        self.rng = rng
        self.mode = mode

    def pool(self, pats):
        if self.mode == "train":
            return self.rng.choice(pats[:-1])
        if self.mode == "all":
            return self.rng.choice(pats)
        return pats[-1]

    def box(self, color: str) -> str:
        return f"{self.rng.choice(COLORS[color])} {self.rng.choice(BOX_WORDS)}"

    def pad(self, color: str) -> str:
        return f"{self.rng.choice(COLORS[color])} {self.rng.choice(PAD_WORDS)}"

    def _outcome(self, options, outcome):
        ok = [(n, w) for n, w, feasible in options if feasible and (outcome is None or n == outcome)]
        if not ok:
            return None
        return self.rng.choices([n for n, _ in ok], weights=[w for _, w in ok])[0]

    # ── 옮기기 (한 쌍) ──────────────────────────────────────────
    def t_move1(self, scene, outcome=None):
        f = facts(scene)
        outcome = self._outcome([
            ("ok", 60, bool(f["boxes"]) and bool(f["empty_pads"])),
            ("not_found", 22, bool(f["absent_boxes"]) or (bool(f["boxes"]) and bool(f["absent_pads"]))),
            ("occupied", 30, bool(f["boxes"]) and bool(f["full_pads"])),
        ], outcome)
        if outcome is None:
            return None
        if outcome == "ok":
            box, pad = self.rng.choice(f["boxes"]), self.rng.choice(f["empty_pads"])
        elif outcome == "occupied":
            box, pad = self.rng.choice(f["boxes"]), self.rng.choice(f["full_pads"])
        else:  # 없는 상자 또는 없는 판
            no_box = bool(f["absent_boxes"]) and (not f["boxes"] or self.rng.random() < 0.6)
            box = self.rng.choice(f["absent_boxes"] if no_box else f["boxes"])
            pad = self.rng.choice(f["pads"] if no_box else f["absent_pads"])
        o, p = self.box(box), self.pad(pad)
        text = self.pool([
            lambda: f"{eul(o)} {p}에 옮겨",
            lambda: f"{o} {ro(p)} 옮겨 줘",
            lambda: f"{o} 집어서 {p}에 놔",
            lambda: f"{eul(o)} {p} 위에 올려놔",
            lambda: f"{p}에 {o} 갖다 놔",
            lambda: f"{eul(o)} 들어서 {p} 위에 내려놔",
            lambda: f"{o} 좀 {ro(p)} 옮겨 줄래?",
            lambda: f"{ro(p)} {eul(o)} 옮겨",
            lambda: f"{o}, {p}에 가져다 놓아 줘",
        ])()
        meta = {"pairs": [[box, pad]], "outcome": outcome}
        if outcome == "ok":
            return text, answer(plan_move([(box, pad)])), meta
        return text, answer(error="object_not_found" if outcome == "not_found" else "target_occupied"), meta

    # ── 옮기기 (두 쌍, 연속 명령) ────────────────────────────────
    def t_move2(self, scene, outcome=None):
        f = facts(scene)
        two = len(f["boxes"]) == 2
        outcome = self._outcome([
            ("ok", 62, two and len(f["empty_pads"]) == 2),
            ("not_found", 20, bool(f["boxes"]) and (bool(f["absent_boxes"]) or bool(f["absent_pads"]))),
            ("occupied", 30, two and len(f["full_pads"]) == 1),
        ], outcome)
        if outcome is None:
            return None
        if outcome == "ok":
            boxes = self.rng.sample(f["boxes"], 2)
            pads = self.rng.sample(f["empty_pads"], 2)
        elif outcome == "occupied":  # 한 상자는 찬 판으로, 다른 상자는 빈 판으로
            boxes = self.rng.sample(f["boxes"], 2)
            pads = [f["full_pads"][0], f["empty_pads"][0]]
            self.rng.shuffle(pads)
        else:  # 두 상자 중 하나가 없거나, 두 판 중 하나가 없음
            no_box = bool(f["absent_boxes"]) and (not f["absent_pads"] or self.rng.random() < 0.6)
            if no_box:
                boxes = [self.rng.choice(f["boxes"]), self.rng.choice(f["absent_boxes"])]
                pads = self.rng.sample(f["pads"], 2)
            else:
                boxes = self.rng.sample(f["boxes"], 2) if two else [f["boxes"][0], self.rng.choice([c for c in COLORS if c != f["boxes"][0]])]
                pads = [self.rng.choice(f["pads"]), self.rng.choice(f["absent_pads"])]
            order = self.rng.sample([0, 1], 2)
            boxes, pads = [boxes[i] for i in order], [pads[i] for i in order]
        o1, o2, p1, p2 = self.box(boxes[0]), self.box(boxes[1]), self.pad(pads[0]), self.pad(pads[1])
        text = self.pool([
            lambda: f"{eun(o1)} {p1}에, {eun(o2)} {p2}에 옮겨",
            lambda: f"{eul(o1)} {p1}에 놓고 {eul(o2)} {p2}에 놔",
            lambda: f"{o1} {ro(p1)} 옮기고 {o2} {ro(p2)} 옮겨 줘",
            lambda: f"먼저 {eul(o1)} {p1}에 옮긴 다음 {eul(o2)} {p2}에 옮겨",
            lambda: f"{eul(o1)} {p1} 위에, {eul(o2)} {p2} 위에 올려놔",
            lambda: f"{o1} {p1}에 갖다 놓고 나서 {o2} {p2}에 갖다 놔",
        ])()
        meta = {"pairs": [[boxes[0], pads[0]], [boxes[1], pads[1]]], "outcome": outcome}
        if outcome == "ok":
            return text, answer(plan_move(list(zip(boxes, pads)))), meta
        return text, answer(error="object_not_found" if outcome == "not_found" else "target_occupied"), meta

    # ── 집기만 ──────────────────────────────────────────────────
    def t_pick(self, scene, outcome=None):
        f = facts(scene)
        outcome = self._outcome([("ok", 75, bool(f["boxes"])), ("not_found", 25, bool(f["absent_boxes"]))], outcome)
        if outcome is None:
            return None
        box = self.rng.choice(f["boxes"] if outcome == "ok" else f["absent_boxes"])
        o = self.box(box)
        text = self.pool([
            lambda: f"{eul(o)} 집어",
            lambda: f"{o} 들어",
            lambda: f"{o} 좀 집어 줘",
            lambda: f"{eul(o)} 들어 올려",
            lambda: f"{o} 잡아",
            lambda: f"{o} 하나 집어 올려 줄래?",
        ])()
        meta = {"pairs": [[box, None]], "outcome": outcome}
        if outcome == "not_found":
            return text, answer(error="object_not_found"), meta
        return text, answer([p_go("pick_stand"), {"skill": "pick", "instruction": pick_instruction(box)}]), meta

    # ── 사진 없이도 답할 수 있는 템플릿 ────────────────────────────
    def t_goto(self, scene, outcome=None):
        place = self.rng.choice(["pick_stand", "place_stand"])
        w = "상자" if place == "pick_stand" else "판"
        text = self.pool([
            lambda: f"{w} 있는 받침대로 가",
            lambda: f"{ga(w)} 놓인 곳으로 가 줘",
            lambda: f"{w} 쪽으로 이동해",
            lambda: f"{w} 있는 받침대 앞으로 가",
            lambda: f"{ga(w)} 있는 데까지 가 봐",
        ])()
        return text, answer([p_go(place)]), {"outcome": "ok"}

    def t_unsupported(self, scene, outcome=None):
        color = self.rng.choice(list(COLORS))
        o, p = self.box(color), self.pad(color)
        # 할 수 없는 일: 동사를 여러 가지로 (집기·옮기기·가기가 아닌 일은 모두 거부). 마지막 패턴은 test 전용
        text = self.pool([
            lambda: f"{eul(o)} 던져",
            lambda: f"{eul(o)} 발로 차",
            lambda: f"{o} 뚜껑 열어",
            lambda: f"{eul(o)} 바닥에 내려놔",
            lambda: f"{p} 위에 올라가",
            lambda: f"{eul(p)} 뒤집어",
            lambda: f"{o} 무게 재 줘",
            lambda: f"{eul(o)} 굴려",
            lambda: f"{o} 위에 다른 상자를 쌓아",
            lambda: f"{eul(o)} 색칠해 줘",
            lambda: f"{eul(o)} 밀어서 떨어뜨려",
            lambda: f"{eul(o)} 돌려 봐",
            lambda: f"{eul(p)} 치워",
            lambda: f"{o} 안을 열어서 보여 줘",
            lambda: f"{ga(o)} 몇 개인지 세어 봐",
            lambda: f"{eul(o)} 반으로 잘라",
            lambda: f"{eul(p)} 닦아 줘",
            lambda: f"{eul(o)} 두드려 봐",
            lambda: f"{eul(o)} 숨겨",
            lambda: "받침대를 옮겨",
            lambda: "노래 불러 줘",
            lambda: "창문 닦아",
            lambda: "춤춰 봐",
            lambda: "불 꺼 줘",
            lambda: "문 열어 줘",
            lambda: f"{eul(o)} 흔들어 봐",
        ])()
        return text, answer(error="unsupported_command"), {"outcome": "unsupported"}

    def make(self, name: str, scene: dict, outcome=None):
        """템플릿 name 으로 한 줄을 만든다 (그 장면에서 만들 수 없으면 None)."""
        made = getattr(self, f"t_{name}")(scene, outcome)
        if made is None:
            return None
        text, output, meta = made
        return {"input": text, "output": output, "template": name, "heldout": self.mode == "heldout", **meta}

    def sample(self, scene: dict) -> dict:
        """장면에 맞는 템플릿을 하나 골라 한 줄을 만든다."""
        names = [n for n, _ in self.WEIGHTS]
        weights = [w for _, w in self.WEIGHTS]
        while True:
            row = self.make(self.rng.choices(names, weights=weights)[0], scene)
            if row is not None:
                return row
