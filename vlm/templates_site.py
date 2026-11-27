"""11단계용: 옮기기 + 철골 용접 명령과 정답 계획을 만드는 템플릿 (schema 이름 "site").

9단계의 templates_full.py(한 쌍·두 쌍 옮기기, 집기, 가기, 할 수 없는 일)를 그대로 물려받고 다음을 더한다.
- weld: 용접만. 용접선 하나("A 기둥 세로 이음부"), 기둥 하나("B 기둥 용접 다 해" → 그 기둥의 닿는 용접선 전부), 철골 전체
- move_weld: 상자 한 쌍 옮기기 + 용접 ("빨간 상자를 파란 판에 옮기고 B 기둥 용접해")
- goto: 철골 앞으로 가기 추가
- 거부: 없는 기둥(D 기둥 등) → unknown_target, 손이 닿지 않는 가로 이음부 → unsupported_command. 옮기기 쪽의 거부는 그대로
계획에서 용접은 go_to steel_frame 한 번 뒤에 weld 를 차례로 적는다. 템플릿마다 마지막 문장 패턴은 test 전용이다.
"""

import random

from schema import WELD_ALL, WELD_B_SIDE, WELD_VERTICAL
from templates import eul, ro
from templates_full import Gen, answer, facts, p_go, plan_move

SEAM_WORDS = ["이음부", "용접선", "이음매"]
VERT_WORDS = ["세로", "수직"]
HORI_WORDS = ["가로", "수평"]
SIDE_WORDS = {"left": ["왼쪽", "왼편"], "right": ["오른쪽", "오른편"]}
UNKNOWN_COLUMNS = ["D", "E", "F", "G", "H"]


def plan_weld(seams) -> list:
    return [p_go("steel_frame")] + [{"skill": "weld", "seam": s} for s in seams]


class GenSite(Gen):
    WEIGHTS = [("move1", 20), ("move2", 14), ("pick", 6), ("goto", 8), ("unsupported", 12), ("weld", 22), ("move_weld", 18)]

    def column(self, letter: str) -> str:
        return self.rng.choice([f"{letter} 기둥", f"{letter}번 기둥"])

    def weld_target(self, kind: str):
        """용접 대상을 가리키는 말과 정답: (말, 용접선 목록 또는 None, 거부 코드 또는 None)."""
        rng = self.rng
        seam = rng.choice(SEAM_WORDS)
        if kind == "one":  # 세로 이음부 하나 (B 기둥은 왼쪽·오른쪽을 말하지 않으면 둘 다)
            col = rng.choice(["A", "B", "B", "C"])
            if col == "B" and rng.random() < 0.65:
                side = rng.choice(["left", "right"])
                return f"{self.column(col)} {rng.choice(SIDE_WORDS[side])} {rng.choice(VERT_WORDS)} {seam}", [WELD_B_SIDE[side]], None
            return f"{self.column(col)} {rng.choice(VERT_WORDS)} {seam}", list(WELD_VERTICAL[col]), None
        if kind == "col":  # 기둥 하나: 그 기둥의 닿는 용접선 전부
            col = rng.choice(["A", "B", "B", "C"])
            word = rng.choice([self.column(col), f"{self.column(col)} {seam} 전부", f"{self.column(col)} 쪽 {seam}"])
            return word, list(WELD_VERTICAL[col]), None
        if kind == "all":
            return rng.choice(["철골", "철골 전체", "철골 기둥 전부", f"철골의 {seam} 전부"]), list(WELD_ALL), None
        if kind == "unknown":  # 없는 기둥
            col = rng.choice(UNKNOWN_COLUMNS)
            word = rng.choice([self.column(col), f"{self.column(col)} {rng.choice(VERT_WORDS + HORI_WORDS)} {seam}"])
            return word, None, "unknown_target"
        # 손이 닿지 않는 가로 이음부
        col = rng.choice(["A", "B", "C"])
        side = f"{rng.choice(SIDE_WORDS[rng.choice(['left', 'right'])])} " if col == "B" and rng.random() < 0.5 else ""
        return f"{self.column(col)} {side}{rng.choice(HORI_WORDS)} {seam}", None, "unsupported_command"

    # ── 용접만 ──────────────────────────────────────────────────
    def t_weld(self, scene, outcome=None):
        kind = self._outcome([("one", 34, True), ("col", 26, True), ("all", 10, True), ("unknown", 14, True), ("unreachable", 16, True)], outcome)
        t, seams, error = self.weld_target(kind)
        text = self.pool([
            lambda: f"{t} 용접해",
            lambda: f"{t} 용접해 줘",
            lambda: f"{eul(t)} 용접해",
            lambda: f"{t} 용접 작업 해",
            lambda: f"가서 {t} 용접해 줘",
            lambda: f"{t} 용접 좀 부탁해",
            lambda: f"{t} 용접 다 해",
            lambda: f"{t} 용접 작업을 시작해",
        ])()
        meta = {"seams": seams, "outcome": kind}
        return text, (answer(error=error) if error else answer(plan_weld(seams))), meta

    # ── 옮기기 한 쌍 + 용접 ──────────────────────────────────────
    def t_move_weld(self, scene, outcome=None):
        f = facts(scene)
        outcome = self._outcome([
            ("ok", 58, bool(f["boxes"]) and bool(f["empty_pads"])),
            ("not_found", 10, bool(f["absent_boxes"]) or (bool(f["boxes"]) and bool(f["absent_pads"]))),
            ("occupied", 10, bool(f["boxes"]) and bool(f["full_pads"])),
            ("unknown", 10, bool(f["boxes"]) and bool(f["empty_pads"])),
            ("unreachable", 12, bool(f["boxes"]) and bool(f["empty_pads"])),
        ], outcome)
        if outcome is None:
            return None
        if outcome == "occupied":
            box, pad = self.rng.choice(f["boxes"]), self.rng.choice(f["full_pads"])
        elif outcome == "not_found":
            no_box = bool(f["absent_boxes"]) and (not f["boxes"] or self.rng.random() < 0.6)
            box = self.rng.choice(f["absent_boxes"] if no_box else f["boxes"])
            pad = self.rng.choice(f["pads"] if no_box else f["absent_pads"])
        else:
            box, pad = self.rng.choice(f["boxes"]), self.rng.choice(f["empty_pads"])
        kind = outcome if outcome in ("unknown", "unreachable") else self.rng.choice(["one", "one", "col"])
        t, seams, weld_error = self.weld_target(kind)
        o, p = self.box(box), self.pad(pad)
        w = self.rng.choice([f"{t} 용접해", f"{t} 용접해 줘", f"{eul(t)} 용접해", f"{t} 용접 작업 해"])
        text = self.pool([
            lambda: f"{eul(o)} {p}에 옮기고 {w}",
            lambda: f"{o} {ro(p)} 옮긴 다음 {w}",
            lambda: f"{eul(o)} {p}에 놓고 나서 {w}",
            lambda: f"먼저 {eul(o)} {p} 위에 올려놓고 {w}",
            lambda: f"{o} {p}에 갖다 놓은 뒤에 {w}",
            lambda: f"{eul(o)} {p}에 옮겨. 그러고 나서 {w}",
        ])()
        meta = {"pairs": [[box, pad]], "seams": seams, "outcome": outcome}
        if outcome == "ok":
            return text, answer(plan_move([(box, pad)]) + plan_weld(seams)), meta
        error = {"not_found": "object_not_found", "occupied": "target_occupied"}.get(outcome, weld_error)
        return text, answer(error=error), meta

    # ── 가기: 철골 추가 ─────────────────────────────────────────
    def t_goto(self, scene, outcome=None):
        if self.rng.random() < 0.4:
            text = self.pool([
                lambda: "철골 앞으로 가",
                lambda: "철골 쪽으로 이동해",
                lambda: "철골 있는 데로 가 줘",
                lambda: "용접할 철골 앞으로 가",
                lambda: "철골 구조물 쪽으로 가 봐",
            ])()
            return text, answer([p_go("steel_frame")]), {"outcome": "ok"}
        return super().t_goto(scene, outcome)


    def sample(self, scene: dict) -> dict:
        """장면에 맞는 템플릿을 하나 골라 한 줄을 만든다.

        판 위에 상자가 이미 놓인 장면에서 옮기기 명령을 만들 때는 '그 판에 놓으라'는 명령(거부: target_occupied)을 자주 고른다.
        이 결과의 학습 예가 적으면 판이 찼는지를 잘 가려내지 못했다 (9단계: 3.4% 일 때 80%, 8% 로 늘리자 100%).
        """
        names = [n for n, _ in self.WEIGHTS]
        weights = [w for _, w in self.WEIGHTS]
        while True:
            name = self.rng.choices(names, weights=weights)[0]
            row = None
            if name in ("move1", "move2", "move_weld") and facts(scene)["full_pads"] and self.rng.random() < 0.4:
                row = self.make(name, scene, "occupied")
            row = row or self.make(name, scene)
            if row is not None:
                return row


if __name__ == "__main__":  # 문장 예시 보기
    rng = random.Random(0)
    scene = {"boxes": {"L": "red", "R": "blue"}, "pads": {"L": "green", "R": "yellow"}, "on_pad": {"L": None, "R": None}}
    for mode in ("train", "heldout"):
        gen = GenSite(rng, mode)
        for name in ("weld", "move_weld", "goto"):
            for _ in range(6):
                row = gen.make(name, scene)
                print(mode, name, "|", row["input"], "→", row["output"])
