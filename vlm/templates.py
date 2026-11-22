"""한국어 명령과 정답 계획을 만드는 템플릿.

장면의 실제 상태(어느 받침대에 어떤 색 상자가 있는지)를 받아 (명령 문장, 정답 JSON, 템플릿 이름, 부가 정보)를 만든다.
- 템플릿마다 문장 패턴이 여러 개 있고, 마지막 패턴 하나는 test 전용(처음 보는 패턴)이다.
- 상자가 어느 받침대에 있는지는 명령에 넣지 않는다 → 사진을 봐야 답할 수 있는 문제가 대부분이 되게 한다.
다른 라이브러리를 쓰지 않는다 (random, json 만).
"""

import json
import random

from schema import BOX_WORDS, COLORS, STANDS, box_name

# ── 한국어 조사 처리 ─────────────────────────────────────────────
_DIGIT_BATCHIM = {"0": True, "1": True, "2": False, "3": True, "4": False,
                  "5": False, "6": True, "7": True, "8": True, "9": False}
_DIGIT_RIEUL = {"1", "7", "8"}  # 일·칠·팔 → ㄹ 받침 (으로 대신 로)


def _last_info(word: str):
    """(받침 있음, ㄹ 받침)"""
    ch = word.strip()[-1]
    if "가" <= ch <= "힣":
        jong = (ord(ch) - 0xAC00) % 28
        return jong != 0, jong == 8
    if ch.isdigit():
        return _DIGIT_BATCHIM[ch], ch in _DIGIT_RIEUL
    return False, False


def ro(word: str) -> str:  # (으)로
    has, rieul = _last_info(word)
    return word + ("으로" if has and not rieul else "로")


def eul(word: str) -> str:  # 을/를
    return word + ("을" if _last_info(word)[0] else "를")


def ga(word: str) -> str:  # 이/가
    return word + ("이" if _last_info(word)[0] else "가")


# ── 계획(JSON) 조각 ──────────────────────────────────────────────
def p_pick(color: str, stand: str):
    return {"skill": "pick", "object": box_name(color), "from": stand}


def p_place(stand: str):
    return {"skill": "place", "to": stand}


def p_go(stand: str):
    return {"skill": "go_to", "target": stand}


def answer(plan=None, error=None) -> str:
    if error:
        return json.dumps({"plan": [], "error": error}, ensure_ascii=False)
    return json.dumps({"plan": plan}, ensure_ascii=False)


# ── 장면 도우미 ─────────────────────────────────────────────────
def present(scene: dict) -> dict:
    """색 → 받침대 (사진에 있는 상자만)."""
    return {c: s for s, c in scene.items() if c}


def empty_stands(scene: dict) -> list:
    return [s for s, c in scene.items() if not c]


class Gen:
    """템플릿 하나를 고르고 문장을 만든다. mode: 'train'(마지막 패턴 제외) / 'heldout'(마지막 패턴만)."""

    # (템플릿, 뽑힐 비중). 위 여섯 개는 사진을 봐야 답할 수 있음
    WEIGHTS = [("move", 34), ("pick", 12), ("goto_box", 10), ("move_to_empty", 8), ("place", 6), ("move_wait", 6),
               ("goto_stand", 7), ("wait", 4), ("stop", 3), ("unsupported", 10)]

    def __init__(self, rng: random.Random, mode: str):
        self.rng = rng
        self.mode = mode

    def pool(self, pats):
        if self.mode == "train":
            return self.rng.choice(pats[:-1])
        return pats[-1]

    def obj(self, color: str) -> str:
        return f"{self.rng.choice(COLORS[color])} {self.rng.choice(BOX_WORDS)}"

    def stand(self, name: str) -> str:
        return self.rng.choice(STANDS[name])

    def pick_outcome(self, options):
        """[(결과 이름, 비중, 가능한지)] 중 가능한 것에서 비중대로 하나."""
        ok = [(n, w) for n, w, feasible in options if feasible]
        if not ok:
            return None
        return self.rng.choices([n for n, _ in ok], weights=[w for _, w in ok])[0]

    # ── 사진을 봐야 하는 템플릿 ──────────────────────────────────
    def _move_target(self, scene, need_wait=False):
        """옮기기: (색, 놓을 받침대, 결과) 를 고른다. 결과 = ok / not_found / occupied."""
        here = present(scene)
        absent = [c for c in COLORS if c not in here]
        empties = empty_stands(scene)
        outcome = self.pick_outcome([
            ("ok", 60, bool(here) and bool(empties)),
            ("not_found", 20, bool(absent)),
            ("occupied", 20, len(here) >= 2),
        ])
        if outcome is None:
            return None
        if outcome == "ok":
            color = self.rng.choice(list(here))
            dest = self.rng.choice(empties)
        elif outcome == "not_found":
            color = self.rng.choice(absent)
            dest = self.rng.choice(list(STANDS))
        else:
            color = self.rng.choice(list(here))
            dest = self.rng.choice([s for c, s in here.items() if c != color])
        return color, dest, outcome

    def t_move(self, scene):
        picked = self._move_target(scene)
        if picked is None:
            return None
        color, dest, outcome = picked
        o, d = self.obj(color), self.stand(dest)
        text = self.pool([
            lambda: f"{eul(o)} {ro(d)} 옮겨",
            lambda: f"{o} {ro(d)} 옮겨 줘",
            lambda: f"{o} 집어서 {d}에 놔",
            lambda: f"{eul(o)} {d}에 올려놔",
            lambda: f"{d}에 {o} 갖다 놔",
            lambda: f"{eul(o)} 들어서 {ro(d)} 가져가",
            lambda: f"{o} 좀 {ro(d)} 옮겨 줄래?",
            lambda: f"{ro(d)} {eul(o)} 옮겨",
            lambda: f"{eul(o)} {d} 위에 둬",
            lambda: f"{o}, {d}에 가져다 놓아 줘",
        ])()
        here = present(scene)
        meta = {"color": color, "src": here.get(color), "dest": dest}
        if outcome == "not_found":
            return text, answer(error="object_not_found"), meta
        if outcome == "occupied":
            return text, answer(error="target_occupied"), meta
        return text, answer([p_pick(color, here[color]), p_place(dest)]), meta

    def t_move_wait(self, scene):
        picked = self._move_target(scene)
        if picked is None:
            return None
        color, dest, outcome = picked
        o, d, n = self.obj(color), self.stand(dest), self.rng.choice([2, 3, 5, 10])
        text = self.pool([
            lambda: f"{eul(o)} {ro(d)} 옮기고 {n}초 기다려",
            lambda: f"{o} {d}에 놓고 {n}초 대기해",
            lambda: f"{eul(o)} {ro(d)} 옮긴 다음 {n}초 쉬어",
            lambda: f"{o} {d}에 올려놓고 {n}초 멈춰 있어",
            lambda: f"{o} {d}에 갖다 놓은 뒤에 {n}초 기다려 줘",
        ])()
        here = present(scene)
        meta = {"color": color, "src": here.get(color), "dest": dest}
        if outcome == "not_found":
            return text, answer(error="object_not_found"), meta
        if outcome == "occupied":
            return text, answer(error="target_occupied"), meta
        return text, answer([p_pick(color, here[color]), p_place(dest), {"skill": "wait", "sec": n}]), meta

    def _one_box(self, scene):
        here = present(scene)
        absent = [c for c in COLORS if c not in here]
        outcome = self.pick_outcome([("ok", 75, bool(here)), ("not_found", 25, bool(absent))])
        if outcome is None:
            return None
        color = self.rng.choice(list(here) if outcome == "ok" else absent)
        return color, outcome, here

    def t_pick(self, scene):
        picked = self._one_box(scene)
        if picked is None:
            return None
        color, outcome, here = picked
        o = self.obj(color)
        text = self.pool([
            lambda: f"{eul(o)} 집어",
            lambda: f"{o} 들어",
            lambda: f"{o} 좀 집어 줘",
            lambda: f"{eul(o)} 들어 올려",
            lambda: f"{o} 잡아",
            lambda: f"{eul(o)} 손에 들어",
            lambda: f"{o} 하나 집어 올려 줄래?",
        ])()
        meta = {"color": color, "src": here.get(color)}
        if outcome == "not_found":
            return text, answer(error="object_not_found"), meta
        return text, answer([p_pick(color, here[color])]), meta

    def t_goto_box(self, scene):
        picked = self._one_box(scene)
        if picked is None:
            return None
        color, outcome, here = picked
        o = self.obj(color)
        text = self.pool([
            lambda: f"{o} 앞으로 가",
            lambda: f"{o} 있는 데로 가",
            lambda: f"{o} 쪽으로 이동해",
            lambda: f"{ga(o)} 있는 받침대로 가",
            lambda: f"{o} 있는 곳으로 가 줘",
            lambda: f"{ga(o)} 놓인 곳까지 가 봐",
        ])()
        meta = {"color": color, "src": here.get(color)}
        if outcome == "not_found":
            return text, answer(error="object_not_found"), meta
        return text, answer([p_go(here[color])]), meta

    def t_move_to_empty(self, scene):
        here = present(scene)
        empties = empty_stands(scene)
        if len(empties) > 1:  # 빈 받침대가 둘 이상이면 어느 쪽인지 정할 수 없으므로 만들지 않음
            return None
        absent = [c for c in COLORS if c not in here]
        outcome = self.pick_outcome([
            ("ok", 65, bool(here) and len(empties) == 1),
            ("not_found", 15, bool(absent)),
            ("occupied", 20, bool(here) and len(empties) == 0),
        ])
        if outcome is None:
            return None
        color = self.rng.choice(absent if outcome == "not_found" else list(here))
        o = self.obj(color)
        text = self.pool([
            lambda: f"{eul(o)} 빈 받침대로 옮겨",
            lambda: f"{o} 비어 있는 받침대에 놔",
            lambda: f"{eul(o)} 빈 곳으로 옮겨 줘",
            lambda: f"{o} 집어서 빈 받침대에 올려놔",
            lambda: f"아무것도 없는 받침대로 {eul(o)} 옮겨",
        ])()
        meta = {"color": color, "src": here.get(color), "dest": empties[0] if empties else None}
        if outcome == "not_found":
            return text, answer(error="object_not_found"), meta
        if outcome == "occupied":
            return text, answer(error="target_occupied"), meta
        return text, answer([p_pick(color, here[color]), p_place(empties[0])]), meta

    def t_place(self, scene):
        here = present(scene)
        empties = empty_stands(scene)
        outcome = self.pick_outcome([("ok", 65, bool(empties)), ("occupied", 35, bool(here))])
        if outcome is None:
            return None
        dest = self.rng.choice(empties if outcome == "ok" else list(here.values()))
        d = self.stand(dest)
        text = self.pool([
            lambda: f"들고 있는 상자를 {d}에 내려놔",
            lambda: f"지금 든 상자 {d}에 놔",
            lambda: f"{d}에 상자 내려놔",
            lambda: f"상자를 {ro(d)} 내려놓아",
            lambda: f"손에 든 걸 {d} 위에 올려 줘",
        ])()
        meta = {"dest": dest}
        if outcome == "occupied":
            return text, answer(error="target_occupied"), meta
        return text, answer([p_place(dest)]), meta

    # ── 사진 없이도 답할 수 있는 템플릿 ────────────────────────────
    def t_goto_stand(self, scene):
        dest = self.rng.choice(list(STANDS))
        d = self.stand(dest)
        text = self.pool([
            lambda: f"{ro(d)} 가",
            lambda: f"{d} 앞으로 이동해",
            lambda: f"{d} 쪽으로 가 줘",
            lambda: f"{ro(d)} 가 있어",
            lambda: f"{d}에 가",
            lambda: f"{d}까지 가 줄래?",
        ])()
        return text, answer([p_go(dest)]), {"dest": dest}

    def t_wait(self, scene):
        n = self.rng.choice([1, 2, 3, 5, 10, 20, 30])
        text = self.pool([
            lambda: f"{n}초 기다려",
            lambda: f"{n}초 대기",
            lambda: f"{n}초만 기다려 줘",
            lambda: f"잠깐 {n}초 멈춰 있어",
            lambda: f"{n}초 동안 가만히 있어",
        ])()
        return text, answer([{"skill": "wait", "sec": n}]), {}

    def t_stop(self, scene):
        text = self.pool([lambda: "멈춰", lambda: "정지", lambda: "그만", lambda: "거기 서", lambda: "스톱",
                          lambda: "하던 거 멈춰"])()
        return text, answer([{"skill": "stop"}]), {}

    def t_unsupported(self, scene):
        color = self.rng.choice(list(COLORS))
        o, d = self.obj(color), self.stand(self.rng.choice(list(STANDS)))
        text = self.pool([
            lambda: f"{eul(o)} 던져",
            lambda: f"{eul(o)} 발로 차",
            lambda: f"{o} 뚜껑 열어",
            lambda: f"{eul(o)} 바닥에 내려놔",
            lambda: f"{d} 위에 올라가",
            lambda: f"{eul(d)} 옮겨",
            lambda: f"{o} 무게 재 줘",
            lambda: "노래 불러 줘",
            lambda: "창문 닦아",
            lambda: "춤춰 봐",
            lambda: "불 꺼 줘",
            lambda: f"{eul(o)} 흔들어 봐",
        ])()
        return text, answer(error="unsupported_command"), {}

    def sample(self, scene: dict) -> dict:
        """장면에 맞는 템플릿을 하나 골라 한 줄을 만든다."""
        names = [n for n, _ in self.WEIGHTS]
        weights = [w for _, w in self.WEIGHTS]
        while True:
            name = self.rng.choices(names, weights=weights)[0]
            made = getattr(self, f"t_{name}")(scene)
            if made is not None:
                text, output, meta = made
                return {"input": text, "output": output, "template": name, "heldout": self.mode == "heldout", **meta}
