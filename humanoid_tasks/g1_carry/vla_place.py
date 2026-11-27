"""9단계 공용 코드: 든 상자를 색 판 위에 내려놓는 과제의 장면 배치, 시범 스크립트, 성공 판정.

scripts/collect_vla_demos.py (--place 시범 수집)와 scripts/vla_pick_eval.py (--place 평가)가 같이 쓴다.
isaaclab 을 import 하므로 Isaac Sim 앱이 켜진 뒤에만 import 한다.

장면: 넓은 받침대(`place_stand`) 위에 색이 다른 판 두 개를 좌우로 깐다. 로봇은 작은 상자를 양손 가운데에 든 채(든 자세) 받침대 앞에 선다.
동작: 자세 잡기 → 판 위로 옮기기 → 내리기 → 놓기(잡기 신호 0) → 손 벌리기 → 팔 올리기
관측·행동 형식은 집기(vla_pick)와 같다 (사진 + 상태 11 → 행동 11). 시작할 때 잡기 상태가 1 이다.
"""

from __future__ import annotations

import math
import random

import torch

from . import pick_place_machine as ppm
from . import vla_pick as vp
from .vla_pick_cfg import SMALL_BOX_SIZE, WIDE_STAND_SIZE
from .vla_place_cfg import PAD_SLOT_Y, PAD_TOP_Z
from .vlm_scene import apply_layouts
from .vlm_scene_cfg import BOX_COLORS, HIDDEN_POS, STAND_NAMES, STAND_X

# 든 자세 (골반 기준 m): 상자를 양손 가운데에 두고, 옆면에서 palm 원점까지 좌우 같은 거리 CARRY_GAP.
# 높이는 손가락 끝(palm 원점보다 약 9cm 아래)이 판 위에 이미 놓인 상자의 윗면을 넘도록 정한다 (다른 판에 상자가 있는 채로 받침대에 다가가므로)
CARRY_GAP = 0.10
CARRY_POSE = (0.28, SMALL_BOX_SIZE[1] / 2 + CARRY_GAP, 0.38)
# 든 자세의 팔 관절 목표값 (왼팔 어깨 pitch·roll·yaw, 팔꿈치 pitch·roll, 오른팔 같은 순서): 내려놓기 시범 200개의 첫 스텝 목표값 평균.
# 시범에서는 기본 자세에서 IK 로 CARRY_POSE 에 가므로 이 값이 거의 한 점에 모인다 (관절별 표준편차 0.01 rad 이하).
# 다른 자세에서 IK 로 같은 손 위치에 가면 관절값이 달라져 모델이 처음 보는 상태가 되므로, 집은 뒤에는 이 관절값으로 직접 옮긴다
CARRY_Q = (-1.075, 0.361, 0.029, 0.272, 0.009, -1.053, -0.384, -0.035, 0.259, -0.010)
BOX_REST_Z = PAD_TOP_Z + SMALL_BOX_SIZE[2] / 2  # 판 위에 놓인 상자 중심 높이
PLACE_OK_XY = 0.05  # 성공: 상자 중심이 판 가운데에서 수평으로 이 거리(m) 안
PLACE_OK_Z = 0.015  # 높이가 놓인 높이에서 이만큼 안
PLACE_OK_TILT = 10.0  # 기울기 (도)
PLACE_HOLD_STEPS = 25  # 위 조건과 '손이 상자에서 떨어져 있음' 을 이 스텝 수(0.5초) 동안 유지
HANDS_OFF = 0.05  # 손이 잡던 자리에서 이만큼(m) 넘게 떨어지면 손을 뗀 것으로 봄
WRONG_PAD_XY = 0.08  # 놓은 상자가 다른 판 가운데에서 이 거리 안이면 '엉뚱한 판'
FAR = (-25.0, 0.0, 0.0)

PLACE_SENTENCES = ["{c} 판 위에 내려놔", "{c} 판에 놓아 줘", "{c} 판 위에 올려놔", "상자를 {c} 판에 내려놓아", "{c} 판으로 내려 줘"]


def place_sentence(color: str, i: int, j: int) -> str:
    """내려놓기 명령: i 번째 문장 틀, j 번째 색 낱말."""
    return PLACE_SENTENCES[i].format(c=vp.COLOR_WORDS[color][j])


def place_sentences(rng: random.Random, rounds: list[dict]) -> list[str]:
    """환경별 내려놓기 명령. 한 쌍의 두 환경은 문장 틀이 같고 색 낱말만 다르다."""
    out, picked = [], {}
    for e, r in enumerate(rounds):
        key = r["pair"] if r["pair"] is not None else ("single", e)
        if key not in picked:
            picked[key] = (rng.randrange(len(PLACE_SENTENCES)), rng.randrange(2))
        out.append(place_sentence(r["color"], *picked[key]))
    return out


def sample_place_round(rng: random.Random, n: int, pos_range: float = 0.10, yaw_range_deg: float = 15.0,
                       mass_range: tuple[float, float] = (2.0, 3.0), stance_y=None, stance_yaw_deg=None,
                       deck: list | None = None, n_single: int = 0):
    """내려놓기 한 묶음. 앞의 n - n_single 개(짝수)는 '같은 장면, 다른 명령'의 쌍, 뒤의 n_single 개는 다른 판에 상자가 이미 놓인 장면.

    쌍: 받침대 위치·두 판의 색과 자리·든 상자의 색과 무게·서는 자리가 같고, 놓을 판만 다르다 (하나는 왼쪽 판, 하나는 오른쪽 판).
    다른 판이 찬 장면: 놓을 판은 빈 판 하나뿐 (계획을 세우는 쪽이 빈 판만 고르므로). 두 번째 상자를 옮길 때의 모습.
    (왼쪽 판 색, 오른쪽 판 색) 12가지를 deck 에서 차례로 꺼낸다.
    돌려주는 값: 환경별 배치 {"stand", "pads": {+1: 왼쪽 색, -1: 오른쪽 색}, "pad_offset", "side": 놓을 판 (+1 왼쪽), "color": 놓을 판 색,
                 "other": 다른 판 색, "held": 든 상자 색, "mass", "stance": (옆 m, 방향 도), "pair": 쌍 번호 또는 None,
                 "occupied": 다른 판 위 상자의 색 또는 None, "occ_offset", "occ_mass"}
    """
    n_pair = n - n_single
    assert n_pair % 2 == 0, "쌍으로 뽑는 환경 수가 짝수여야 합니다"
    colors = list(BOX_COLORS)
    deck = [] if deck is None else deck

    def scene():
        if not deck:
            deck.extend((a, b) for a in colors for b in colors if a != b)
            rng.shuffle(deck)
        c_left, c_right = deck.pop()
        return {
            "stand": (rng.uniform(-pos_range, pos_range), rng.uniform(-pos_range, pos_range), math.radians(rng.uniform(-yaw_range_deg, yaw_range_deg))),
            "pads": {1: c_left, -1: c_right},
            "pad_offset": {s: (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01)) for s in (1, -1)},
            "held": rng.choice(colors), "mass": rng.uniform(*mass_range),
            "stance": (vp._signed(rng, stance_y), vp._signed(rng, stance_yaw_deg)),
            "occupied": None, "occ_offset": (0.0, 0.0, 0.0), "occ_mass": 2.5,
        }

    rounds = [None] * n
    for k in range(n_pair // 2):
        base = scene()
        first = rng.choice([0, 1])
        for j, side in ((first, 1), (1 - first, -1)):
            rounds[2 * k + j] = dict(base, side=side, color=base["pads"][side], other=base["pads"][-side], pair=k)
    for e in range(n_pair, n):
        base = scene()
        side = rng.choice([1, -1])
        base["occupied"] = rng.choice([c for c in colors if c != base["held"]])
        base["occ_offset"] = (rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), math.radians(rng.uniform(-8.0, 8.0)))
        base["occ_mass"] = rng.uniform(*mass_range)
        rounds[e] = dict(base, side=side, color=base["pads"][side], other=base["pads"][-side], pair=None)
    return rounds


def _local_to_world(pos, yaw: float, lx: float, ly: float):
    return pos[0] + math.cos(yaw) * lx - math.sin(yaw) * ly, pos[1] + math.sin(yaw) * lx + math.cos(yaw) * ly


class PlaceScene:
    """setup_place_round 가 돌려주는 것: 받침대·판의 자리와 든 상자 묶음."""

    def __init__(self, held, stand_xy, stand_yaw, pad_xy, side):
        self.held = held  # 든 상자 (vp.BoxGroup)
        self.stand_xy, self.stand_yaw = stand_xy, stand_yaw  # (N, 2), (N,)
        self.pad_xy = pad_xy  # (N, 2, 2): [환경, 0 = 왼쪽 판 / 1 = 오른쪽 판, xy]
        self.side = side  # (N,) 놓을 판이 왼쪽이면 +1
        idx = (side < 0).long()
        ar = torch.arange(len(side), device=side.device)
        self.target_xy = pad_xy[ar, idx]  # (N, 2)
        self.other_xy = pad_xy[ar, 1 - idx]


def setup_place_round(env, machine: ppm.PickPlaceMachine, rounds: list[dict], stand_name: str = "place_stand",
                      stand_xy0: tuple[float, float] = (STAND_X, 0.0), stand_yaw0: float = 0.0, hide_others: bool = True) -> PlaceScene:
    """내려놓기 배치를 적용한다: 받침대, 판 두 개, (있으면) 다른 판 위의 상자. 든 상자는 give_box 가 손에 붙일 때 놓는다.

    hide_others: 집는 받침대와 7단계 장면의 받침대·큰 상자, 쓰지 않는 작은 상자를 멀리 치운다 (전체 장면에서는 False 로 두고 따로 배치).
    """
    n, device = len(rounds), env.device
    origins = env.scene.env_origins[:n]
    ids = torch.arange(n, device=device)
    zeros6 = torch.zeros(n, 6, device=device)
    if hide_others:
        apply_layouts(env, [{"boxes": {s: None for s in STAND_NAMES}, "stands": {s: FAR for s in STAND_NAMES}}] * n)
        far = torch.zeros(n, 7, device=device)
        far[:, :3] = origins + torch.tensor([STAND_X + FAR[0], 3.0, WIDE_STAND_SIZE[2] / 2], device=device)
        far[:, 3] = 1.0
        env.scene["wide_stand"].write_root_pose_to_sim(far, env_ids=ids)
    stand_pose = torch.zeros(n, 7, device=device)
    stand_yaw = torch.zeros(n, device=device)
    pad_pose = {c: torch.zeros(n, 7, device=device) for c in BOX_COLORS}
    box_pose = {c: torch.zeros(n, 7, device=device) for c in BOX_COLORS}
    masses = {c: torch.full((n,), 2.5) for c in BOX_COLORS}
    for c_i, c in enumerate(BOX_COLORS):  # 기본은 숨김
        pad_pose[c][:, :3] = origins + torch.tensor([HIDDEN_POS[0] - 4.0, HIDDEN_POS[1] + 0.6 * c_i, HIDDEN_POS[2]], device=device)
        pad_pose[c][:, 3] = 1.0
        box_pose[c][:, :3] = origins + torch.tensor([HIDDEN_POS[0] - 2.0, HIDDEN_POS[1] + 0.6 * c_i, HIDDEN_POS[2]], device=device)
        box_pose[c][:, 3] = 1.0
    pad_xy = torch.zeros(n, 2, 2, device=device)
    for i, r in enumerate(rounds):
        dx, dy, yaw = r["stand"]
        yaw = yaw + stand_yaw0
        ox, oy = float(origins[i, 0]), float(origins[i, 1])
        sx, sy = ox + stand_xy0[0] + dx, oy + stand_xy0[1] + dy
        stand_pose[i, 0], stand_pose[i, 1], stand_pose[i, 2] = sx, sy, WIDE_STAND_SIZE[2] / 2
        stand_pose[i, 3], stand_pose[i, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
        stand_yaw[i] = yaw
        for k, s in enumerate((1, -1)):
            px, py = r["pad_offset"][s]
            wx, wy = _local_to_world((sx, sy), yaw, px, s * PAD_SLOT_Y + py)
            c = r["pads"][s]
            pad_pose[c][i, 0], pad_pose[c][i, 1], pad_pose[c][i, 2] = wx, wy, PAD_TOP_Z - (PAD_TOP_Z - WIDE_STAND_SIZE[2]) / 2
            pad_pose[c][i, 3], pad_pose[c][i, 4:6], pad_pose[c][i, 6] = math.cos(yaw / 2), 0.0, math.sin(yaw / 2)
            pad_xy[i, k, 0], pad_xy[i, k, 1] = wx, wy
        masses[r["held"]][i] = r["mass"]
        if r["occupied"]:
            c = r["occupied"]
            bx, by, byaw = r["occ_offset"]
            k = 0 if -r["side"] > 0 else 1
            wx, wy = _local_to_world((float(pad_xy[i, k, 0]), float(pad_xy[i, k, 1])), yaw, bx, by)
            box_pose[c][i, 0], box_pose[c][i, 1], box_pose[c][i, 2] = wx, wy, BOX_REST_Z
            box_pose[c][i, 3], box_pose[c][i, 4:6], box_pose[c][i, 6] = math.cos((yaw + byaw) / 2), 0.0, math.sin((yaw + byaw) / 2)
            masses[c][i] = r["occ_mass"]
    env.scene[stand_name].write_root_pose_to_sim(stand_pose, env_ids=ids)
    for c in BOX_COLORS:
        env.scene[f"pad_{c}"].write_root_pose_to_sim(pad_pose[c], env_ids=ids)
        box = env.scene[f"small_{c}_box"]
        box.write_root_pose_to_sim(box_pose[c], env_ids=ids)
        box.write_root_velocity_to_sim(zeros6, env_ids=ids)
        vp.set_box_mass(box, masses[c])
    machine.set_grasp_geometry(SMALL_BOX_SIZE, CARRY_GAP, CARRY_GAP)
    side = torch.tensor([r["side"] for r in rounds], device=device, dtype=torch.float)
    return PlaceScene(vp.BoxGroup(env, [r["held"] for r in rounds]), stand_pose[:, :2].clone(), stand_yaw, pad_xy, side)


def occupied_boxes(env, rounds: list[dict]):
    """다른 판 위에 놓여 있는 상자의 지금 위치 (N, 3)·방향 (N,) 과, 그런 상자가 있는 환경 (N,)."""
    n, device = len(rounds), env.device
    pos, yaw = torch.zeros(n, 3, device=device), torch.zeros(n, device=device)
    has = torch.zeros(n, dtype=torch.bool, device=device)
    for i, r in enumerate(rounds):
        if r["occupied"]:
            box = env.scene[f"small_{r['occupied']}_box"]
            pos[i], yaw[i], has[i] = box.data.root_pos_w[i], ppm.yaw_of(box.data.root_quat_w)[i], True
    return pos, yaw, has


def carry_goal(machine: ppm.PickPlaceMachine) -> torch.Tensor:
    """든 자세의 양손 위치 (N, 2, 3 월드)."""
    return machine._pelvis_to_world(*CARRY_POSE)


def give_box(machine: ppm.PickPlaceMachine, held, mask: torch.Tensor | None = None):
    """든 자세로 벌린 양손 사이에 상자를 놓고 붙인다 (집는 과정 없이 '든 상태'에서 시작). 손은 먼저 든 자세에 가 있어야 한다."""
    m = torch.ones(machine.n, dtype=torch.bool, device=machine.device) if mask is None else mask
    mid, psi = machine.hand_frame()
    pos = mid + ppm.rot_z(psi, torch.tensor([-ppm.GRASP_X, 0.0, -ppm.GRASP_Z], device=machine.device).expand(machine.n, -1))
    ids = m.nonzero().flatten()
    held.write_root_pose_to_sim(torch.cat([pos, ppm.quat_from_yaw(psi)], dim=1)[ids], env_ids=ids)
    held.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=machine.device), env_ids=ids)
    machine.take_box(held, m)


def box_tilt_deg(quat: torch.Tensor) -> torch.Tensor:
    """상자가 기운 각도 (도): 상자의 위쪽 축과 연직 사이."""
    x, y = quat[:, 1], quat[:, 2]
    return torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


class PlaceTracker:
    """내려놓기의 결과를 환경별로 센다 (스크립트·VLA 공통 기준).

    성공: 손을 뗀 상자가 놓을 판 위에 있고(가운데에서 PLACE_OK_XY 안, 높이·기울기 정상) 양손이 잡던 자리에서 HANDS_OFF 넘게 떨어진 채 PLACE_HOLD_STEPS 유지.
    """

    def __init__(self, machine: ppm.PickPlaceMachine):
        self.m = machine
        n, dev = machine.n, machine.device
        z = lambda dtype=torch.bool: torch.zeros(n, dtype=dtype, device=dev)  # noqa: E731
        self.success, self.released, self.was_attached = z(), z(), z()
        self.count, self.t, self.t_success, self.t_release = z(torch.long), z(torch.long), z(torch.long), z(torch.long)
        self.release_err = torch.full((n,), float("nan"), device=dev)  # 손을 뗀 순간 상자와 놓을 판 가운데의 수평 거리
        self.scene: PlaceScene | None = None

    def begin(self, mask: torch.Tensor, scene: PlaceScene):
        self.scene = scene
        for v in (self.success, self.released):
            v[mask] = False
        self.was_attached[mask] = self.m.attached[mask]
        self.count[mask], self.t[mask] = 0, 0
        self.t_success[mask], self.t_release[mask] = -1, -1
        self.release_err[mask] = float("nan")

    def _box(self):
        box = self.m.box
        return box.data.root_pos_w, box.data.root_quat_w

    def update(self, mask: torch.Tensor):
        """mask 환경을 한 스텝 진행 (env.step 뒤에 부름)."""
        m, sc = self.m, self.scene
        if sc is None or not bool(mask.any()):
            return
        pos, quat = self._box()
        att = m.attached
        self.t[mask] += 1
        new_rel = mask & self.was_attached & ~att & ~self.released
        self.release_err[new_rel] = torch.norm(pos[:, :2] - sc.target_xy, dim=1)[new_rel]
        self.t_release[new_rel] = self.t[new_rel]
        self.released |= new_rel
        self.was_attached = torch.where(mask, att, self.was_attached)
        on_pad = (torch.norm(pos[:, :2] - sc.target_xy, dim=1) <= PLACE_OK_XY) & ((pos[:, 2] - BOX_REST_Z).abs() <= PLACE_OK_Z) \
            & (box_tilt_deg(quat) <= PLACE_OK_TILT)
        hands_off = torch.norm(m.ik.hand_pos_w() - m.box_grasp_points(), dim=-1).min(dim=1).values > HANDS_OFF
        ok = mask & ~att & self.released & on_pad & hands_off
        self.count = torch.where(ok, self.count + 1, torch.where(mask, torch.zeros_like(self.count), self.count))
        new = mask & ~self.success & (self.count >= PLACE_HOLD_STEPS)
        self.t_success[new] = self.t[new]
        self.success |= new

    def outcome(self, e: int, fell: bool = False) -> tuple[str, float]:
        """환경 e 의 결과 이름과 놓은 위치 오차 (m, 놓을 판 가운데까지의 수평 거리. 놓지 않았으면 nan)."""
        sc = self.scene
        pos, quat = self._box()
        err = float(torch.norm(pos[e, :2] - sc.target_xy[e]))
        if bool(self.success[e]):
            return "성공", err
        if fell:
            return "넘어짐", float("nan")
        if not bool(self.released[e]):
            return "놓지 않음", float("nan")
        if float(pos[e, 2]) < WIDE_STAND_SIZE[2] - 0.05 or float(box_tilt_deg(quat)[e]) > 30.0:
            return "상자가 떨어지거나 넘어짐", err
        if float(torch.norm(pos[e, :2] - sc.other_xy[e])) <= WRONG_PAD_XY:
            return "엉뚱한 판에 놓음", err
        if bool(self.m.attached[e]):
            return "놓았다가 다시 잡음", err
        if err <= PLACE_OK_XY:
            return "손을 떼지 않음", err
        return "판에서 벗어나게 놓음", err


class SidePlaceScript:
    """내려놓기 시범용 스크립트 IK: 든 자세로 받침대 앞에 선 상태에서 자세 잡기 → 판 위로 → 내리기 → 놓기 → 손 벌리기 → 팔 올리기.

    상자가 손에 붙어 있는 동안은 손에 무게가 있으므로 목표를 몸통 좌표에 고정해 옮기고(pick_place_machine 설명), 단계가 바뀔 때마다 지금 상자 위치로 다시 계산한다.
    처음의 '자세 잡기'는 어느 판에 놓든 같은 동작이다 → 이 구간의 사진·상태만으로는 놓을 판을 알 수 없다 (명령을 봐야 함).
    매 스텝: machine.pre_step() → env.step → machine.post_step → update()
    """

    IDLE, PREP, OVER, DOWN, ADJUST, RELEASE, OPEN, RETRACT, HOLD, DONE, FAIL = range(11)
    NAMES = ["대기", "자세 잡기", "판 위로", "내리기", "위치 맞추기", "놓기", "손 벌리기", "팔 올리기", "유지", "완료", "실패"]
    PREP_UP = 0.02  # 자세 잡기: 똑바로 조금 올림
    OPEN_OUT = 0.07  # 손 벌리기: 양손을 바깥으로
    STEPS = {PREP: 40, OVER: 75, DOWN: 75, ADJUST: 30, OPEN: 40, RETRACT: 60}
    RELEASE_WAIT = 10
    HOLD_STEPS = 25

    def __init__(self, machine: ppm.PickPlaceMachine):
        self.m = machine
        n, dev = machine.n, machine.device
        self.phase = torch.full((n,), self.IDLE, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.place_err = torch.full((n,), float("nan"), device=dev)
        self.scene: PlaceScene | None = None

    @property
    def active(self) -> torch.Tensor:
        return (self.phase != self.IDLE) & (self.phase != self.DONE) & (self.phase != self.FAIL)

    def _enter(self, mask: torch.Tensor, phase: int):
        self.phase[mask] = phase
        self.t[mask] = 0

    def reset(self):
        self.phase[:] = self.IDLE
        self.t[:] = 0
        self.place_err[:] = float("nan")

    def _move_box_to(self, mask: torch.Tensor, xy: torch.Tensor | None, z, steps: int):
        """양손을 같은 만큼 옮겨 상자 중심을 (xy, z) 로 보낸다. xy 가 None 이면 수평 위치는 그대로."""
        if not bool(mask.any()):
            return
        m = self.m
        pos = m.box.data.root_pos_w
        delta = torch.zeros_like(pos)
        if xy is not None:
            delta[:, :2] = xy - pos[:, :2]
        delta[:, 2] = z - pos[:, 2]
        m._start_move(mask, m.ik.hand_pos_w() + delta[:, None, :], steps)

    def begin(self, mask: torch.Tensor, scene: PlaceScene):
        self.scene = scene
        z = self.m.box.data.root_pos_w[:, 2] + self.PREP_UP
        self._move_box_to(mask, None, z, self.STEPS[self.PREP])
        self._enter(mask, self.PREP)

    def fail(self, mask: torch.Tensor):
        """넘어진(리셋된) 환경 등: 상자를 떼고 팔을 기본 자세로."""
        m = self.m
        if not bool(mask.any()):
            return
        m._detach(mask & m.attached)
        m.arm_busy[mask], m.arm_hold[mask], m.arm_world[mask] = False, False, False
        m.ik.reset(mask.nonzero().flatten())
        self._enter(mask, self.FAIL)

    def update(self):
        m, sc = self.m, self.scene
        if sc is None:  # 아직 시작한 환경이 없음
            return
        ph = self.phase.clone()
        self.t += 1
        done = m.move_done
        low_z = torch.full((m.n,), BOX_REST_Z + ppm.PLACE_GAP, device=m.device)
        go = (ph == self.PREP) & done
        if bool(go.any()):
            self._move_box_to(go, sc.target_xy, m.box.data.root_pos_w[:, 2], self.STEPS[self.OVER])
            self._enter(go, self.OVER)
        go = (ph == self.OVER) & done
        self._move_box_to(go, sc.target_xy, low_z, self.STEPS[self.DOWN])
        self._enter(go, self.DOWN)
        go = (ph == self.DOWN) & done
        self._move_box_to(go, sc.target_xy, low_z, self.STEPS[self.ADJUST])
        self._enter(go, self.ADJUST)
        go = (ph == self.ADJUST) & done
        if bool(go.any()):
            self.place_err[go] = torch.norm(m.box.data.root_pos_w[:, :2] - sc.target_xy, dim=1)[go]
            m._detach(go)
            self._enter(go, self.RELEASE)
        go = (ph == self.RELEASE) & (self.t >= self.RELEASE_WAIT)
        if bool(go.any()):
            _, psi = m.hand_frame()
            out = ppm.rot_z(psi[:, None].expand(-1, 2), m._sym(0.0, self.OPEN_OUT, 0.0)[None].expand(m.n, -1, -1))
            m._start_move(go, m.ik.hand_pos_w() + out, self.STEPS[self.OPEN])
            self._enter(go, self.OPEN)
        go = (ph == self.OPEN) & done
        if bool(go.any()):
            up = m.ik.hand_pos_w().clone()
            up[..., 2] = (m.robot.data.root_pos_w[:, 2] + CARRY_POSE[2])[:, None]
            m._start_move(go, up, self.STEPS[self.RETRACT])
            self._enter(go, self.RETRACT)
        self._enter((ph == self.RETRACT) & done, self.HOLD)
        self._enter((ph == self.HOLD) & (self.t >= self.HOLD_STEPS), self.DONE)
