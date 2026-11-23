"""8단계 공용 코드: VLA 로 팔을 움직여 상자를 집는 과제의 관측·행동 정의, 장면 무작위화, 붙이기 규칙, 성공 판정.

scripts/collect_vla_demos.py (시범 수집)와 scripts/vla_pick_eval.py (VLA·스크립트 평가)가 같이 쓴다.
isaaclab 을 import 하므로 Isaac Sim 앱이 켜진 뒤에만 import 한다.

관측과 행동 (제어 주기 50Hz, 한 스텝 = 0.02초)
- 사진: 아래를 보는 머리 카메라 한 대 (vla_pick_cfg.vla_cam, 256x256 RGB)
- 상태 (11): 팔 10관절 위치 (왼팔 어깨 pitch·roll·yaw, 팔꿈치 pitch·roll, 오른팔 같은 순서) + 잡기 상태 (0 빈손 / 1 상자가 붙어 있음)
- 행동 (11): 팔 10관절 목표값 + 잡기 신호 (0 놓기 / 1 잡기)
  잡기 신호가 1 이고 양손이 잡는 위치에서 ATTACH_TOL 안이면 상자를 붙이고, 0 이 되면 뗀다 (붙이기는 pick_place_machine 의 것을 그대로 씀)
- k 번째 스텝의 한 줄: 사진·상태 = env.step 을 부르기 직전의 것, 행동 = 그 스텝에 넣은 팔 목표값과 그 스텝이 끝났을 때의 잡기 신호
"""

from __future__ import annotations

import math
import random

import torch

from . import pick_place_machine as ppm
from .vlm_scene import apply_layouts, set_lighting
from .vlm_scene_cfg import BOX_COLORS, STAND_NAMES

STATE_DIM = 11
ACTION_DIM = 11
CONTROL_HZ = 50
ATTACH_TOL = 0.04  # 잡기 신호가 1 일 때 붙이는 조건: 양손이 잡는 위치에서 이 거리(m) 안 (스크립트는 0.03 에서 0.2초 머문 뒤 붙임)
LIFT_OK = 0.05  # 성공: 상자가 처음 높이보다 이만큼(m) 넘게 들린 채
LIFT_HOLD_STEPS = 25  # 이 스텝 수(0.5초) 동안 붙어 있음
MAX_JOINT_STEP = 0.04  # 팔 목표값이 한 스텝에 바뀔 수 있는 최대 크기 (rad). 스크립트 IK·걷기 정책 학습과 같은 2 rad/s
FAR = (-25.0, 0.0, 0.0)  # 쓰지 않는 받침대를 치워 두는 이동량 (카메라가 그리는 거리 밖)

# 명령 문장. 색을 정하지 않는 문장(1차)과 색을 넣는 문장(2차)
TASK_SENTENCES = ["상자를 집어", "상자 들어 줘", "상자를 들어 올려", "앞에 있는 상자를 집어 줘", "상자 잡아서 들어"]
COLOR_WORDS = {"red": ["빨간", "빨간색"], "blue": ["파란", "파란색"], "green": ["초록", "초록색"], "yellow": ["노란", "노란색"]}
COLOR_SENTENCES = ["{c} 상자를 집어", "{c} 상자 들어 줘", "{c} 상자를 들어 올려", "{c} 상자를 집어 줘", "{c} 상자 잡아서 들어"]


def task_sentence(rng: random.Random, color: str | None = None) -> str:
    if color is None:
        return rng.choice(TASK_SENTENCES)
    return rng.choice(COLOR_SENTENCES).format(c=rng.choice(COLOR_WORDS[color]))


def sample_round(rng: random.Random, n: int, pos_range: float = 0.10, yaw_range_deg: float = 15.0,
                 mass_range: tuple[float, float] = (2.0, 3.0), color: str | None = None):
    """환경 n 개의 한 묶음: 가운데 받침대에 상자 하나. 받침대와 상자를 함께 앞뒤·좌우 ±pos_range, 방향 ±yaw_range_deg.

    상자 색은 묶음마다 하나 (상태 머신이 한 번에 한 종류의 상자만 다루므로). 돌려주는 값: (색, 배치 목록, 상자 무게 목록)
    """
    color = rng.choice(list(BOX_COLORS)) if color is None else color
    layouts, masses = [], []
    for _ in range(n):
        stand = (rng.uniform(-pos_range, pos_range), rng.uniform(-pos_range, pos_range),
                 math.radians(rng.uniform(-yaw_range_deg, yaw_range_deg)))
        layouts.append({
            "boxes": {"stand_L": None, "stand_C": color, "stand_R": None},
            "stands": {"stand_L": FAR, "stand_C": stand, "stand_R": FAR},
            "box_offset": {"stand_C": (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), math.radians(rng.uniform(-3.0, 3.0)))},
        })
        masses.append(rng.uniform(*mass_range))
    return color, layouts, masses


def random_lighting(rng: random.Random):
    """조명을 조금 바꾼다 (모든 환경이 함께 쓰므로 묶음마다 한 번)."""
    set_lighting(dome_intensity=rng.uniform(600.0, 1400.0), sun_intensity=rng.uniform(800.0, 2500.0),
                 sun_elevation_deg=rng.uniform(40.0, 85.0), sun_azimuth_deg=rng.uniform(0.0, 360.0))


def set_box_mass(box, masses) -> None:
    """상자 질량을 환경별로 정한다 (관성은 질량 비율만큼)."""
    view = box.root_physx_view
    old = view.get_masses().clone()
    new = old.clone()
    new[:, 0] = torch.as_tensor(masses, dtype=old.dtype)
    ids = torch.arange(new.shape[0])
    view.set_masses(new, ids)
    inertias = view.get_inertias().clone()
    view.set_inertias(inertias * (new / old), ids)


def setup_round(env, machine: ppm.PickPlaceMachine, color: str, layouts: list[dict], masses: list[float]):
    """배치를 적용하고 상자 무게를 정한다. 돌려주는 값: (상자, 받침대 중심 (N, 2), 받침대 방향 (N,))"""
    scene = apply_layouts(env, layouts)
    box = env.scene[f"{color}_box"]
    set_box_mass(box, masses)
    k = STAND_NAMES.index("stand_C")
    return box, scene["stand_pos"][k, :, :2].clone(), scene["stand_yaw"][k].clone()


def read_obs(env, machine: ppm.PickPlaceMachine):
    """지금의 사진 (N, H, W, 3) uint8 과 상태 (N, 11)."""
    img = env.scene.sensors["vla_cam"].data.output["rgb"][..., :3]
    q = machine.robot.data.joint_pos[:, machine.ik.all_joint_ids]
    return img, torch.cat([q, machine.attached.float()[:, None]], dim=1)


def hand_error(machine: ppm.PickPlaceMachine) -> torch.Tensor:
    """양손이 잡는 위치에서 떨어진 거리 (N,) m, 두 손 중 큰 값."""
    return torch.norm(machine.ik.hand_pos_w() - machine.box_grasp_points(), dim=-1).max(dim=1).values


class LiftTracker:
    """'상자가 붙은 채 처음 높이보다 LIFT_OK 넘게 들려 LIFT_HOLD_STEPS 동안 유지' 를 환경별로 센다 (스크립트·VLA 공통 성공 기준)."""

    def __init__(self, machine: ppm.PickPlaceMachine):
        self.m = machine
        n, dev = machine.n, machine.device
        self.z0 = torch.zeros(n, device=dev)
        self.count = torch.zeros(n, dtype=torch.long, device=dev)
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_success = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.ever_attached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.max_lift = torch.zeros(n, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)

    def begin(self, mask: torch.Tensor):
        self.z0[mask] = self.m.box.data.root_pos_w[mask, 2]
        self.count[mask], self.t[mask] = 0, 0
        self.success[mask], self.ever_attached[mask] = False, False
        self.t_success[mask] = -1
        self.max_lift[mask] = 0.0

    def update(self, mask: torch.Tensor):
        """mask 환경을 한 스텝 진행 (env.step 뒤에 부름)."""
        lift = self.m.box.data.root_pos_w[:, 2] - self.z0
        att = self.m.attached
        self.t[mask] += 1
        self.ever_attached |= mask & att
        self.max_lift = torch.where(mask & att, torch.maximum(self.max_lift, lift), self.max_lift)
        ok = mask & att & (lift >= LIFT_OK)
        self.count = torch.where(ok, self.count + 1, torch.where(mask, torch.zeros_like(self.count), self.count))
        new = mask & ~self.success & (self.count >= LIFT_HOLD_STEPS)
        self.t_success[new] = self.t[new]
        self.success |= new


class VlaArmExecutor:
    """VLA 가 낸 행동(팔 목표값 10 + 잡기 신호 1)을 로봇에 적용한다. 붙이기·떼기는 상태 머신의 것을 그대로 쓴다.

    매 스텝: pre(action, mask) → env.step → machine.post_step → post(action, mask, terminated)
    """

    def __init__(self, machine: ppm.PickPlaceMachine):
        self.m = machine
        n, dev = machine.n, machine.device
        z = lambda dtype=torch.bool: torch.zeros(n, dtype=dtype, device=dev)  # noqa: E731
        self.ever_signal, self.dropped, self.fell = z(), z(), z()
        self.signal_out_of_range = z()  # 잡기 신호가 1 인데 손이 멀어 붙지 못한 스텝이 있었음
        self.in_range_no_signal = z()  # 손이 닿는 자리에 있었는데 신호가 0 이던 스텝이 있었음
        self.wrong_pick = z()  # (2차) 잡기 신호를 낸 자리가 다른 상자의 잡는 위치였음
        self.min_err = torch.full((n,), 1.0e9, device=dev)
        self.min_clear = torch.full((n,), 1.0e9, device=dev)
        self.q_lo, self.q_hi = machine.ik._q_min, machine.ik._q_max

    def begin(self, mask: torch.Tensor):
        for v in (self.ever_signal, self.dropped, self.fell, self.signal_out_of_range, self.in_range_no_signal, self.wrong_pick):
            v[mask] = False
        self.min_err[mask], self.min_clear[mask] = 1.0e9, 1.0e9

    def pre(self, action: torch.Tensor, mask: torch.Tensor):
        """팔 목표값을 넣는다 (관절 한계로 자르고, 한 스텝에 MAX_JOINT_STEP 넘게 바뀌지 않게)."""
        q_prev = self.m.ik.q_cmd
        q = torch.clamp(action[:, :10], self.q_lo, self.q_hi)
        q = q_prev + torch.clamp(q - q_prev, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        self.m.ik.q_cmd[mask] = q[mask]
        self.m.pre_step()

    def post(self, action: torch.Tensor, mask: torch.Tensor, terminated: torch.Tensor, other=None):
        """잡기 신호를 처리한다. 넘어진(리셋된) 환경은 상자를 떼고 팔 목표값을 기본 자세로 되돌린다.

        other (2차, OtherBox): 잡기 신호가 1 일 때 손이 '다른 상자'의 잡는 위치에 있으면 wrong_pick 으로 기록한다 (붙이지는 않음).
        """
        m = self.m
        fell = mask & terminated
        if bool(fell.any()):
            m._detach(fell & m.attached)
            m.ik.reset(fell.nonzero().flatten())
            self.fell |= fell
        live = mask & ~terminated
        err = hand_error(m)
        signal = action[:, 10] > 0.5
        self.min_err = torch.where(live & ~m.attached, torch.minimum(self.min_err, err), self.min_err)
        self.ever_signal |= live & signal
        self.signal_out_of_range |= live & signal & ~m.attached & (err > ATTACH_TOL)
        self.in_range_no_signal |= live & ~signal & ~m.attached & (err <= ATTACH_TOL)
        m._attach(live & signal & ~m.attached & (err <= ATTACH_TOL))
        if other is not None:
            self.wrong_pick |= live & signal & ~m.attached & (other.hand_error() <= ATTACH_TOL)
        drop = live & ~signal & m.attached
        if bool(drop.any()):
            m._detach(drop)
            self.dropped |= drop
        held = live & m.attached
        if bool(held.any()) and m.step % 10 == 0:
            self.min_clear = torch.where(held, torch.minimum(self.min_clear, m.hand_box_clearance()), self.min_clear)


def classify_failure(ex: VlaArmExecutor, tr: LiftTracker, e: int) -> str:
    """실패한 환경 e 가 어디서 실패했는지 (VLA 실행용)."""
    if bool(ex.fell[e]):
        return "넘어짐"
    if bool(ex.wrong_pick[e]):
        return "다른 상자를 집음"
    if bool(tr.ever_attached[e]):
        return "들다 놓침" if bool(ex.dropped[e]) else "붙였지만 못 듦"
    if float(ex.min_err[e]) > ATTACH_TOL:
        return "못 닿음"
    return "잡기 신호 시점 틀림"


# ===========================================================================
# 2차: 넓은 받침대에 색이 다른 작은 상자 두 개. 명령한 색의 상자를 집는다
# ===========================================================================
from .pick_place_cfg import BOX_Z_ON_TABLE  # noqa: E402
from .vla_pick_cfg import SLOT_Y, SMALL_BOX_SIZE, WIDE_STAND_SIZE  # noqa: E402
from .vlm_scene_cfg import HIDDEN_POS, STAND_X  # noqa: E402

GAP_OUTER = 0.07  # 바깥 손: 상자 옆면에서 palm 원점까지 (1차와 같음)
GAP_INNER = 0.13  # 안쪽 손(몸 가운데 쪽으로 넘어온 손)은 비스듬해져 손가락이 상자 쪽을 향하므로 더 띄움


def sample_round2(rng: random.Random, n: int, pos_range: float = 0.10, yaw_range_deg: float = 15.0,
                  mass_range: tuple[float, float] = (2.0, 3.0), color: str | None = None,
                  side: int | None = None, other: str | None = None):
    """2차 한 묶음: 집을 상자의 색은 묶음마다 하나, 환경마다 그 상자가 왼쪽·오른쪽 자리 중 어디에 놓일지와 다른 상자의 색을 무작위로.

    돌려주는 값: (집을 색, 환경별 배치 목록). 배치 = {"stand": (dx, dy, yaw), "side": +1 왼쪽 / -1 오른쪽, "other": 다른 상자 색,
                 "offsets": {"target": (bx, by, byaw), "other": (...)}, "mass": 집을 상자 무게, "other_mass": 다른 상자 무게}
    """
    colors = list(BOX_COLORS)
    color = rng.choice(colors) if color is None else color
    rounds = []
    for _ in range(n):
        rounds.append({
            "stand": (rng.uniform(-pos_range, pos_range), rng.uniform(-pos_range, pos_range),
                      math.radians(rng.uniform(-yaw_range_deg, yaw_range_deg))),
            "side": rng.choice([1, -1]),
            "other": rng.choice([c for c in colors if c != color]),
            "offsets": {k: (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), math.radians(rng.uniform(-3.0, 3.0))) for k in ("target", "other")},
            "mass": rng.uniform(*mass_range), "other_mass": rng.uniform(*mass_range),
        })
        # 자리·다른 상자 색을 정해 줄 때 (녹화용): 난수를 뽑은 뒤에 덮어써서, 같은 시드면 받침대 위치가 같게 한다
        if side is not None:
            rounds[-1]["side"] = side
        if other is not None:
            rounds[-1]["other"] = other
    return color, rounds


def setup_round2(env, machine: ppm.PickPlaceMachine, color: str, rounds: list[dict]):
    """2차 배치를 적용한다. 돌려주는 값: (집을 상자, 받침대 중심 (N, 2), 받침대 방향 (N,), 집을 상자가 왼쪽이면 +1 (N,))

    1차 장면의 받침대 세 개와 큰 상자들은 멀리 치우고, 넓은 받침대와 작은 상자 두 개(나머지 두 색은 숨김)를 놓는다.
    손 잡는 위치는 바깥 손 GAP_OUTER, 안쪽 손 GAP_INNER 로 정한다.
    """
    n, device = len(rounds), env.device
    apply_layouts(env, [{"boxes": {s: None for s in STAND_NAMES}, "stands": {s: FAR for s in STAND_NAMES}}] * n)
    origins = env.scene.env_origins[:n]
    ids = torch.arange(n, device=device)
    stand_pose = torch.zeros(n, 7, device=device)
    stand_yaw = torch.zeros(n, device=device)
    box_pose = {c: torch.zeros(n, 7, device=device) for c in BOX_COLORS}
    masses = {c: torch.full((n,), 2.5) for c in BOX_COLORS}
    for c_i, c in enumerate(BOX_COLORS):  # 기본은 숨김
        box_pose[c][:, :3] = origins + torch.tensor([HIDDEN_POS[0] - 2.0, HIDDEN_POS[1] + 0.6 * c_i, HIDDEN_POS[2]], device=device)
        box_pose[c][:, 3] = 1.0
    for i, r in enumerate(rounds):
        dx, dy, yaw = r["stand"]
        pos = origins[i] + torch.tensor([STAND_X + dx, dy, WIDE_STAND_SIZE[2] / 2], device=device)
        stand_pose[i, :3], stand_yaw[i] = pos, yaw
        stand_pose[i, 3], stand_pose[i, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
        for key, c, side in (("target", color, r["side"]), ("other", r["other"], -r["side"])):
            bx, by, byaw = r["offsets"][key]
            lx, ly = bx, side * SLOT_Y + by  # 받침대 좌표
            box_pose[c][i, 0] = pos[0] + math.cos(yaw) * lx - math.sin(yaw) * ly
            box_pose[c][i, 1] = pos[1] + math.sin(yaw) * lx + math.cos(yaw) * ly
            box_pose[c][i, 2] = BOX_Z_ON_TABLE
            box_pose[c][i, 3], box_pose[c][i, 4:6], box_pose[c][i, 6] = math.cos((yaw + byaw) / 2), 0.0, math.sin((yaw + byaw) / 2)
            masses[c][i] = r["mass"] if key == "target" else r["other_mass"]
    env.scene["wide_stand"].write_root_pose_to_sim(stand_pose, env_ids=ids)
    zeros6 = torch.zeros(n, 6, device=device)
    for c in BOX_COLORS:
        box = env.scene[f"small_{c}_box"]
        box.write_root_pose_to_sim(box_pose[c], env_ids=ids)
        box.write_root_velocity_to_sim(zeros6, env_ids=ids)
        set_box_mass(box, masses[c])
    side = torch.tensor([r["side"] for r in rounds], device=device, dtype=torch.float)
    outer, inner = torch.full_like(side, GAP_OUTER), torch.full_like(side, GAP_INNER)
    machine.set_grasp_geometry(SMALL_BOX_SIZE, torch.where(side > 0, outer, inner), torch.where(side > 0, inner, outer))
    return env.scene[f"small_{color}_box"], stand_pose[:, :2].clone(), stand_yaw, side


class OtherBox:
    """2차에서 '집으면 안 되는 상자'의 지금 위치·잡는 위치를 환경별로 모아 준다 (환경마다 색이 다르므로)."""

    def __init__(self, env, machine: ppm.PickPlaceMachine, rounds: list[dict], side: torch.Tensor):
        self.env, self.m = env, machine
        self.masks = {c: torch.tensor([r["other"] == c for r in rounds], device=env.device) for c in BOX_COLORS}
        half = SMALL_BOX_SIZE[1] / 2
        outer, inner = torch.full_like(side, GAP_OUTER), torch.full_like(side, GAP_INNER)
        off = torch.zeros(len(rounds), 2, 3, device=env.device)
        off[..., 0], off[..., 2] = ppm.GRASP_X, ppm.GRASP_Z
        # 다른 상자는 반대쪽 자리에 있으므로 안쪽·바깥 손이 바뀐다
        off[:, 0, 1] = half + torch.where(side > 0, inner, outer)
        off[:, 1, 1] = -(half + torch.where(side > 0, outer, inner))
        self.off = off

    def pose(self):
        pos = torch.zeros(self.m.n, 3, device=self.env.device)
        yaw = torch.zeros(self.m.n, device=self.env.device)
        for c, mask in self.masks.items():
            if bool(mask.any()):
                box = self.env.scene[f"small_{c}_box"]
                pos[mask] = box.data.root_pos_w[mask]
                yaw[mask] = ppm.yaw_of(box.data.root_quat_w)[mask]
        return pos, yaw

    def hand_error(self) -> torch.Tensor:
        """양손이 '다른 상자'의 잡는 위치에서 떨어진 거리 (N,), 두 손 중 큰 값."""
        pos, yaw = self.pose()
        return torch.norm(self.m.ik.hand_pos_w() - self.m.grasp_points_of(pos, yaw, self.off), dim=-1).max(dim=1).values

    def clearance(self) -> torch.Tensor:
        pos, yaw = self.pose()
        return self.m.hand_box_clearance(pos, yaw)


class SidePickScript:
    """2차 시범용 스크립트 IK: 받침대 앞에 선 상태에서 손을 위로 → 상자 위로 → 내려서 잡기 → 똑바로 들기.

    옆의 상자를 건드리지 않으려고 위에서 내려온다. 상태 머신이 '서기'까지 끝낸 뒤(until="stand") 그 팔 이동 기능만 빌려 쓴다.
    매 스텝: machine.pre_step() → env.step → machine.post_step → update()
    """

    IDLE, UP, OVER, DOWN, GRASP, LIFT, HOLD, DONE, FAIL = range(9)
    NAMES = ["대기", "손 위로", "상자 위로", "내리기", "잡기", "들기", "유지", "완료", "실패"]
    UP_Z = 0.29  # 손을 올리는 높이 (골반 기준 m): 손가락이 상자 윗면에 닿지 않을 만큼 위
    OVER_DZ = 0.165  # 잡는 위치보다 이만큼 위에서 내려옴
    LIFT_UP = 0.09
    STEPS = {UP: 50, OVER: 75, DOWN: 75, LIFT: 75}
    HOLD_STEPS = 50
    GRASP_TIMEOUT = 150

    def __init__(self, machine: ppm.PickPlaceMachine):
        self.m = machine
        n, dev = machine.n, machine.device
        self.phase = torch.full((n,), self.IDLE, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.in_tol = torch.zeros(n, dtype=torch.long, device=dev)
        self.grasp_err = torch.full((n,), float("nan"), device=dev)

    @property
    def active(self) -> torch.Tensor:
        return (self.phase != self.IDLE) & (self.phase != self.DONE) & (self.phase != self.FAIL)

    def _enter(self, mask: torch.Tensor, phase: int):
        self.phase[mask] = phase
        self.t[mask] = 0

    def reset(self):
        self.phase[:] = self.IDLE
        self.t[:], self.in_tol[:] = 0, 0
        self.grasp_err[:] = float("nan")

    def begin(self, mask: torch.Tensor):
        m = self.m
        self.in_tol[mask] = 0
        m._start_move(mask, m._pelvis_to_world(ppm.OPEN_POSE[0], ppm.OPEN_POSE[1], self.UP_Z), self.STEPS[self.UP], world=True)
        self._enter(mask, self.UP)

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
        m = self.m
        ph = self.phase.clone()
        self.t += 1
        done = m.move_done
        go = (ph == self.UP) & done
        if bool(go.any()):
            above = m.box_grasp_points()
            above[..., 2] += self.OVER_DZ
            m._start_move(go, above, self.STEPS[self.OVER], world=True)
            self._enter(go, self.OVER)
        go = (ph == self.OVER) & done
        m._start_move(go, m.box_grasp_points(), self.STEPS[self.DOWN], world=True)
        self._enter(go, self.DOWN)
        self._enter((ph == self.DOWN) & done, self.GRASP)
        g = ph == self.GRASP
        if bool(g.any()):
            err = hand_error(m)
            self.grasp_err[g] = err[g]
            self.in_tol[g] = torch.where(err <= ppm.ATTACH_TOL, self.in_tol + 1, torch.zeros_like(self.in_tol))[g]
            ok = g & (self.in_tol >= ppm.ATTACH_HOLD)
            if bool(ok.any()):
                m._attach(ok)
                up = m.ik.hand_pos_w().clone()
                up[..., 2] += self.LIFT_UP
                m._start_move(ok, up, self.STEPS[self.LIFT])  # 손에 무게가 있으므로 목표는 몸통 좌표에 고정
                self._enter(ok, self.LIFT)
            self._enter(g & ~ok & (self.t > self.GRASP_TIMEOUT), self.FAIL)
        go = (ph == self.LIFT) & done
        m.att_floor[go] = -1.0e9
        self._enter(go, self.HOLD)
        self._enter((ph == self.HOLD) & (self.t >= self.HOLD_STEPS), self.DONE)
