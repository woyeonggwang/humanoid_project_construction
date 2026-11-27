"""상자 집어 옮기기 상태 머신 (환경 여러 개를 한꺼번에, 환경마다 따로 진행).

scripts/pick_place_demo.py (평가·녹화)와 scripts/vlm_pick_place_demo.py (VLM 연결 데모)가 같이 쓴다.
isaaclab 을 import 하므로 Isaac Sim 앱이 켜진 뒤에만 import 한다.

쓰는 법 (매 스텝):
    machine.pre_step()                 # 팔 목표값(IK)과 붙은 상자를 갱신
    obs = env.step(policy(obs))
    cmd = machine.post_step(terminated)  # 상태 전환, 다음 스텝의 속도 명령 (N, 3)
    cmd_term.vel_command_b[:] = cmd

순서: 팔 벌리기 → (받침대 가까이 있으면 뒤로 물러나기) → A 로 걷기 → 멈춤 → 손 뻗기 → 붙이기 → 들기
      → 뒤로 물러나기 → B 로 걷기 → 멈춤 → 내려놓기 → 떼기 → 손 벌리기 → 팔 거두기

legacy=True 는 5단계의 걷기 정책(팔 무작위판)용: 정지 명령을 정해진 걸음 박자에서만 주고, 멈추는 동안 움직이는 만큼
미리 보정하고, 뒤로 물러나는 대신 옆으로 비켜선다 (그 정책은 뒤로 걷기와 아무 박자에서나 멈추기를 배우지 않았음).
"""

from __future__ import annotations

import math

import torch

from .arm_ik import DualArmIK, TorsoFrame
from .pick_place_cfg import BOX_SIZE, BOX_Z_ON_TABLE

RAD2DEG = 57.29578
G = 9.81

# ---------------------------------------------------------------------------
# 동작에 쓰는 값 (m, 골반 위치 + 골반이 향한 방향 기준: x 앞, y 왼쪽, z 위)
# ---------------------------------------------------------------------------
GRASP_X, GRASP_Z = -0.08, 0.04  # 상자 좌표에서 본 손(palm 원점) 잡는 위치의 앞뒤·높이
#   손가락이 palm 원점보다 앞 16cm·아래 9cm 까지 내려가므로 손 전체가 옆면 가운데에 오게 뒤·위로 둠
GRASP_GAP = 0.07  # 상자 옆면에서 palm 원점까지의 거리
#   손바닥을 옆면에 붙일 수는 없다: 엄지 뿌리가 손바닥 안쪽 면에서 나와 있어, 엄지 관절을 어떻게 움직여도 palm 원점보다
#   안쪽으로 5.4cm 이상 나온다 (2026-11-22 측정: 엄지 관절 3개를 한계 안에서 5×5×5 로 바꿔 가며 엄지 링크 형상의 가장 안쪽 점을 잼)
#   0.07 = 엄지가 옆면에 거의 닿는 거리 (2026-11-23 측정, hand_box_clearance: 손 형상과 옆면 사이 틈이 붙이는 순간 평균 0.8cm,
#   들고 있는 동안 최소 0.1cm. 0.065 는 들고 있는 동안 최대 1.5cm 파고듦, 5단계의 0.09 는 틈 3.1cm)
LEGACY_GRASP_GAP = 0.09  # 5단계 값 (--legacy 로 5단계 동작을 다시 볼 때)
STANCE_DIST = 0.36  # 멈출 때 골반에서 상자(받침대) 중심까지의 거리. 잡는 손 위치가 골반 앞 0.28m = 닿는 범위(0.20~0.35m)의 가운데
OPEN_POSE = (0.30, 0.31, 0.125)  # 벌린 자세 (좌우 ±). 이 자세로 받침대에 다가가면 손이 상자 양옆을 지나감
CARRY_BOX_X, CARRY_BOX_Z = 0.28, 0.20  # 상자를 든 자세의 손 위치 (좌우는 잡은 간격 그대로). 상자 아랫면이 받침대 윗면보다 약 9cm 높음
RETRACT_POSE = (0.20, 0.31, 0.125)  # 내려놓은 뒤 거둔 자세 (벌린 채 뒤로)
LIFT_UP = 0.06  # 붙인 뒤 먼저 똑바로 들어 올리는 높이
PLACE_GAP = 0.005  # 내려놓을 때 상자 아랫면과 받침대 윗면 사이에 남기는 틈
ATTACH_TOL, ATTACH_HOLD = 0.03, 10  # 양손이 잡는 위치에서 3cm 안에 10스텝 머물면 붙임
STILL_STEPS, STILL_POS, STILL_YAW = 50, 0.01, 0.0175  # 멈춤 판정: 1초 동안 골반이 1cm·1° 넘게 움직이지 않음
STANCE_TOL_X = (-0.08, 0.03)  # 멈춘 자리 허용 범위: 앞 오차 (+ = 덜 감). 밖이면 다시 다가감
STANCE_TOL_Y, STANCE_TOL_YAW = 0.05, 0.10
MAX_REAPPROACH = 3
GAIT_PERIOD = 0.8
BACKOFF_DIST, BACKOFF_SPEED = 0.40, 0.3  # 받침대 앞에서 뒤로 물러나는 거리·속도
BACKOFF_NEAR = 0.75  # 받침대 중심이 이 거리 안에 있으면 걷기 전에 먼저 물러남
HOLONOMIC_DIST, HOLONOMIC_YAW = 2.0, 0.6  # 목표가 이 거리·방향 차이 안이면 돌지 않고 옆·앞으로만 움직여 다가감
ALIGN_Y = 0.10  # 옆으로 이만큼 맞춘 뒤에야 앞으로 다가감 (받침대 모서리에 걸리지 않게)

# legacy(5단계 정책)용: 정지 명령을 주는 걸음 박자와 멈추는 동안 움직이는 양의 보정 (빈손, 짐 든 상태)
STOP_PHASE = (0.22, 0.30)
STOP_YAW_DRIFT = (0.087, 0.131)
STOP_LAT_DRIFT = (0.03, 0.0)
STOP_FWD_DRIFT = (-0.03, -0.02)
SIDESTEP_DIST, SIDESTEP_SPEED = 0.55, 0.3

# 상태
(S_OPEN, S_BACK_A, S_WALK_A, S_SETTLE_A, S_CLOSE, S_CLOSE2, S_LIFT1, S_LIFT2, S_HOLD, S_BACK_B, S_WALK_B, S_SETTLE_B,
 S_PLACE, S_PLACE2, S_RELEASE, S_OPEN_B, S_BACK_END, S_RETRACT, S_END_HOLD, S_DONE, S_FAIL, S_IDLE) = range(22)
STATE_NAMES = ["팔 벌리기", "물러나기(A 전)", "A 로 걷기", "A 앞에서 멈춤", "손 뻗기", "손 위치 맞추기", "들어 올리기", "든 자세로",
               "든 채 유지", "물러나기(B 전)", "B 로 걷기", "B 앞에서 멈춤", "내려놓기", "내려놓기 보정", "떼기", "손 벌리기",
               "물러나기(끝)", "팔 거두기", "마무리", "완료", "실패", "대기"]
TIMEOUT = {S_OPEN: 200, S_BACK_A: 400, S_WALK_A: 1500, S_SETTLE_A: 400, S_CLOSE: 300, S_CLOSE2: 150, S_LIFT1: 150,
           S_LIFT2: 250, S_HOLD: 200, S_BACK_B: 400, S_WALK_B: 1500, S_SETTLE_B: 400, S_PLACE: 300, S_PLACE2: 150,
           S_RELEASE: 100, S_OPEN_B: 150, S_BACK_END: 400, S_RETRACT: 250, S_END_HOLD: 200}


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_z(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """벡터 (..., 3) 를 z 축으로 yaw 만큼 돌린다. yaw 는 앞쪽 차원과 같은 크기."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], dim=-1)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1)


class PickPlaceMachine:
    def __init__(self, env, legacy: bool = False, grasp_gap: float = GRASP_GAP, until: str = "full", debug: bool = False):
        """
        Args:
            env: ManagerBasedRLEnv (unwrapped).
            legacy: 5단계 정책용 보정을 쓸지 (모듈 설명 참고).
            grasp_gap: 상자 옆면에서 palm 원점까지의 거리 (m).
            until: "full", "lift"(들어 올린 뒤 2초 유지하고 끝) 또는 "stand"(상자 앞에 멈춰 선 데서 끝. 팔은 벌린 자세).
        """
        self.env = env
        self.robot = env.scene["robot"]
        self.device = env.device
        self.n = env.num_envs
        self.dt = env.step_dt
        self.legacy = legacy
        self.until = until
        self.debug = debug
        self.grasp_y = BOX_SIZE[1] / 2 + grasp_gap
        self.box_size = BOX_SIZE  # 지금 다루는 상자의 크기 (set_grasp_geometry 로 바꿈)
        self.ik = DualArmIK(self.robot)
        self.torso = TorsoFrame(self.robot)
        self.palm_ids = self.ik.body_ids
        self.foot_ids = [self.robot.body_names.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
        self.default_mass = self.robot.data.default_mass.cpu()
        self.default_inertia = self.robot.data.default_inertia.cpu()
        n, dev = self.n, self.device
        z = lambda *shape, dtype=torch.float: torch.zeros(*shape, dtype=dtype, device=dev)  # noqa: E731
        self.step = 0
        self.state = torch.full((n,), S_IDLE, dtype=torch.long, device=dev)
        self.t_state = z(n, dtype=torch.long)
        self.in_tol_count = z(n, dtype=torch.long)
        self.near_mode = z(n, dtype=torch.bool)
        self.fail_state = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.fell = z(n, dtype=torch.bool)
        self.reapproach_a, self.reapproach_b = z(n, dtype=torch.long), z(n, dtype=torch.long)
        self.pos_hist, self.yaw_hist = z(STILL_STEPS, n, 2), z(STILL_STEPS, n)
        self.back_start, self.back_yaw = z(n, 2), z(n)
        # 팔 이동
        self.arm_start, self.arm_goal, self.arm_goal_w0 = z(n, 2, 3), z(n, 2, 3), z(n, 2, 3)
        self.arm_t, self.arm_T = z(n, dtype=torch.long), torch.ones(n, dtype=torch.long, device=dev)
        self.arm_busy, self.arm_hold, self.arm_world = z(n, dtype=torch.bool), z(n, dtype=torch.bool), z(n, dtype=torch.bool)
        self.move_done = z(n, dtype=torch.bool)
        # 잡는 위치: 상자 좌표에서 본 양손(palm 원점) 자리 (N, 2, 3). 기본은 좌우 대칭
        self.grasp_off = torch.tensor([[GRASP_X, self.grasp_y, GRASP_Z], [GRASP_X, -self.grasp_y, GRASP_Z]], device=dev).repeat(n, 1, 1)
        # 붙이기
        self.box = None
        self.box_mass = z(n)  # 환경별 상자 질량 (start 에서 읽음)
        self.attached = z(n, dtype=torch.bool)
        self.att_off, self.att_yaw = z(n, 3), z(n)
        self.att_floor = torch.full((n,), -1.0e9, device=dev)
        self.box_prev_pos, self.box_pred_pos, self.box_prev_yaw = z(n, 3), z(n, 3), z(n)
        self.box_prev_valid = z(n, dtype=torch.bool)
        # 목표
        self.src_xy, self.src_yaw, self.dst_xy, self.dst_yaw = z(n, 2), z(n), z(n, 2), z(n)
        self.stance_a_xy, self.stance_b_xy = z(n, 2), z(n, 2)
        self.place_target = z(n, 3)
        self.obstacles = z(0, n, 2)  # (받침대 수, N, 2) 걷기 전에 가까이 있으면 물러날 받침대 중심들
        # 기록
        keys = ("stance_a_x", "stance_a_y", "stance_a_yaw", "grasp_err", "stance_b_x", "stance_b_y", "stance_b_yaw",
                "place_err_before", "place_xy", "place_z", "lift_tilt", "t_pick", "t_total", "box_shift",
                "stop_turn_a", "stop_turn_b", "hand_clear", "hand_clear_min")
        self.rec = {k: torch.full((n,), float("nan"), device=dev) for k in keys}
        self.box_dev_air, self.box_dev_table = z(n), z(n)
        self.pick_tilt_max = z(n)
        self.arrive_yaw = z(n)
        self.fall_info = {}
        self.last_pos = self.robot.data.root_pos_w.clone()
        self.last_yaw = self.robot.data.heading_w.clone()
        self.last_grav = self.robot.data.projected_gravity_b.clone()
        self.j_move = z(n, dtype=torch.bool)  # 관절 공간에서 짐 든 자세로 되돌리는 중
        self.j_from = z(n, 10)
        self.j_t, self.j_T = z(n, dtype=torch.long), torch.ones(n, dtype=torch.long, device=dev)
        self.q_cmd_min = torch.full((10,), 1.0e9, device=dev)
        self.q_cmd_max = torch.full((10,), -1.0e9, device=dev)
        self.q_cmd_min_state = torch.full((10,), S_IDLE, dtype=torch.long, device=dev)
        self.q_cmd_max_state = torch.full((10,), S_IDLE, dtype=torch.long, device=dev)
        self.t_start = z(n, dtype=torch.long)
        # 멈춘 자리 허용 범위 (환경마다: 앞 오차 하한·상한, 좌우, 방향). 용접처럼 더 정확히 서야 하는 작업이 그 환경의 값만 바꿔 쓴다
        self.stance_tol = torch.tensor([STANCE_TOL_X[0], STANCE_TOL_X[1], STANCE_TOL_Y, STANCE_TOL_YAW], device=dev).repeat(n, 1)

    # ------------------------------------------------------------------ 시작
    def start(self, box, src_xy: torch.Tensor, src_yaw: torch.Tensor, dst_xy: torch.Tensor, dst_yaw: torch.Tensor,
              obstacles: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        """mask 환경(기본: 전부)의 옮기기를 시작한다.

        box: 옮길 상자 (RigidObject). src_xy/dst_xy: 상자가 놓인/놓을 받침대 중심 (N, 2 월드).
        src_yaw/dst_yaw: 로봇이 그 받침대를 마주 볼 때의 방향 (N,). obstacles: 받침대 중심들 (K, N, 2).
        """
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        self.walk_only = False
        self.box = box
        self.box_mass = box.root_physx_view.get_masses()[:, 0].to(self.device).clone()
        self.src_xy[m], self.src_yaw[m], self.dst_xy[m], self.dst_yaw[m] = src_xy[m], src_yaw[m], dst_xy[m], dst_yaw[m]
        dir_a = torch.stack([torch.cos(src_yaw), torch.sin(src_yaw)], dim=1)
        dir_b = torch.stack([torch.cos(dst_yaw), torch.sin(dst_yaw)], dim=1)
        self.stance_a_xy[m] = (src_xy - STANCE_DIST * dir_a)[m]
        self.stance_b_xy[m] = (dst_xy - STANCE_DIST * dir_b)[m]
        self.place_target[m] = torch.cat([dst_xy, torch.full((self.n, 1), BOX_Z_ON_TABLE + PLACE_GAP, device=self.device)], dim=1)[m]
        self.obstacles = torch.stack([src_xy, dst_xy]) if obstacles is None else obstacles
        self.fail_state[m] = -1
        self.reapproach_a[m], self.reapproach_b[m] = 0, 0
        self.t_start[m] = self.step
        self._set_state(m, S_OPEN)
        self._start_move(m, self._pelvis_to_world(OPEN_POSE[0], OPEN_POSE[1], OPEN_POSE[2]), 75)

    def start_walk(self, target_xy: torch.Tensor, target_yaw: torch.Tensor, obstacles: torch.Tensor | None = None,
                   keep_box: bool = False, mask: torch.Tensor | None = None):
        """go_to 기술: mask 환경(기본: 전부)이 목표 받침대 앞까지 걸어가 멈춘다 (상태는 B 쪽 걷기를 그대로 씀).

        keep_box=True 면 지금 다루는 상자(붙어 있으면 든 채)를 그대로 둔다. 팔 관절 목표는 건드리지 않는다.
        """
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        if not keep_box:
            self.box = None
        self.dst_xy[m], self.dst_yaw[m] = target_xy[m], target_yaw[m]
        dir_b = torch.stack([torch.cos(target_yaw), torch.sin(target_yaw)], dim=1)
        self.stance_b_xy[m] = (target_xy - STANCE_DIST * dir_b)[m]
        if obstacles is not None:
            self.obstacles = obstacles
        self.fail_state[m] = -1
        self.reapproach_b[m] = 0
        self.t_start[m] = self.step
        self.walk_only = True
        self._enter_backoff(m, S_BACK_B, S_WALK_B)

    walk_only = False

    def take_box(self, box, mask: torch.Tensor | None = None):
        """상자를 지금 자리에서 손에 붙인다 (집는 과정 없이 '든 상태'로 시작할 때). 상자는 미리 양손 사이에 놓아 둔다."""
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        self.box = box
        self.box_mass = box.root_physx_view.get_masses()[:, 0].to(self.device).clone()
        self._attach(m)
        self.att_floor[m] = -1.0e9

    def arms_to_carry(self, mask: torch.Tensor | None = None, steps: int = 75):
        """mask 환경(기본: 전부)의 팔 관절 목표값을 짐 든 자세(기본 자세)로 부드럽게 되돌린다 (IK 없이 관절 공간에서 steps 스텝).

        닿는 범위 끝에서 IK 로 팔을 거두면 팔꿈치를 끝까지 접는 자세로 빠질 수 있어, 받침대에서 물러난 뒤에는 이 방법을 쓴다.
        """
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        if not bool(m.any()):
            return
        self.j_from[m] = self.ik.q_cmd[m]
        self.j_t[m], self.j_T[m] = 0, steps
        self.j_move[m] = True
        self.arm_busy[m], self.arm_hold[m], self.arm_world[m] = False, False, False

    def reset(self):
        """모든 환경을 처음 상태(대기)로: 붙은 상자를 떼고, 팔 목표값·기록·상태를 비운다. env.reset() 직후에 부른다."""
        if self.box is not None and bool(self.attached.any()):
            self._detach(self.attached.clone())
        self.state[:] = S_IDLE
        self.t_state[:] = 0
        self.in_tol_count[:] = 0
        self.near_mode[:] = False
        self.fail_state[:] = -1
        self.fell[:] = False
        self.fall_info = {}
        self.reapproach_a[:], self.reapproach_b[:] = 0, 0
        self.arm_busy[:], self.arm_hold[:], self.arm_world[:], self.j_move[:] = False, False, False, False
        self.move_done[:] = False
        self.att_floor[:] = -1.0e9
        self.box_prev_valid[:] = False
        self.box_dev_air[:], self.box_dev_table[:], self.pick_tilt_max[:] = 0.0, 0.0, 0.0
        for v in self.rec.values():
            v[:] = float("nan")
        self.ik.reset()

    # ------------------------------------------------------------------ 도우미
    def _sym(self, x: float, y: float, z: float) -> torch.Tensor:
        return torch.tensor([[x, y, z], [x, -y, z]], device=self.device)

    def _pelvis_to_world(self, x: float, y: float, z: float) -> torch.Tensor:
        """골반 yaw 좌표의 좌우 대칭 점 → 환경별 월드 (N, 2, 3)."""
        pts = self._sym(x, y, z)
        yaw = self.robot.data.heading_w[:, None].expand(-1, 2)
        return self.robot.data.root_pos_w[:, None, :] + rot_z(yaw, pts[None].expand(self.n, -1, -1))

    def hand_frame(self):
        """두 손 기준 좌표: 원점 = 양손 가운데, 방향 = 오른손 → 왼손 선에 수직인 앞쪽."""
        hp = self.ik.hand_pos_w()
        v = hp[:, 0, :2] - hp[:, 1, :2]
        return hp.mean(dim=1), torch.atan2(-v[:, 0], v[:, 1])

    def tilt(self) -> torch.Tensor:
        return torch.acos(torch.clamp(-self.robot.data.projected_gravity_b[:, 2], -1.0, 1.0)) * RAD2DEG

    def set_grasp_geometry(self, box_size, gap_left, gap_right):
        """다른 크기의 상자나 좌우가 다른 손 간격을 쓸 때: 상자 크기 (앞뒤, 좌우, 높이)와 손별 '옆면에서 palm 원점까지 거리' (숫자 또는 (N,))."""
        self.box_size = tuple(box_size)
        self.grasp_off[:, 0, 1] = self.box_size[1] / 2 + torch.as_tensor(gap_left, device=self.device)
        self.grasp_off[:, 1, 1] = -(self.box_size[1] / 2 + torch.as_tensor(gap_right, device=self.device))

    def grasp_points_of(self, box_pos: torch.Tensor, box_yaw: torch.Tensor, grasp_off: torch.Tensor) -> torch.Tensor:
        """상자 위치 (N, 3)·방향 (N,) 과 상자 좌표의 손 자리 (N, 2, 3) → 양손 잡는 위치 (N, 2, 3 월드)."""
        return box_pos[:, None, :] + rot_z(box_yaw[:, None].expand(-1, 2), grasp_off)

    def box_grasp_points(self) -> torch.Tensor:
        """상자의 지금 위치·방향 기준 양손 잡는 위치 (N, 2, 3 월드)."""
        return self.grasp_points_of(self.box.data.root_pos_w, yaw_of(self.box.data.root_quat_w), self.grasp_off)

    def _set_palm_extra_mass(self, env_ids: torch.Tensor, extra):
        """손(palm) 질량을 기본값 + extra 로 (extra: 숫자 하나 또는 환경별 값). 관성은 질량 비율만큼 (원본 randomize_rigid_body_mass 와 같은 방식)."""
        ids = env_ids.cpu()
        palm = torch.tensor(self.palm_ids)
        view = self.robot.root_physx_view
        masses = view.get_masses().clone()
        extra = torch.as_tensor(extra, dtype=masses.dtype).cpu().reshape(-1, 1) if torch.is_tensor(extra) else extra
        masses[ids[:, None], palm] = self.default_mass[ids[:, None], palm] + extra
        view.set_masses(masses, ids)
        inertias = view.get_inertias().clone()
        ratio = masses[ids[:, None], palm] / self.default_mass[ids[:, None], palm]
        inertias[ids[:, None], palm] = self.default_inertia[ids[:, None], palm] * ratio[..., None]
        view.set_inertias(inertias, ids)

    def _update_gravity_cancel(self):
        """붙어 있는 상자에는 중력과 같은 크기의 위쪽 힘을 걸어 둔다 (무게는 손 질량으로 대신하므로)."""
        forces = torch.zeros(self.n, 1, 3, device=self.device)
        forces[self.attached, 0, 2] = self.box_mass[self.attached] * G
        self.box.set_external_force_and_torque(forces, torch.zeros(self.n, 1, 3, device=self.device), is_global=True)

    HAND_LINKS = ("palm", "zero", "one", "two", "three", "four", "five", "six")  # 손바닥, 엄지 3마디, 나머지 손가락

    def hand_box_clearance(self, box_pos: torch.Tensor | None = None, box_yaw: torch.Tensor | None = None) -> torch.Tensor:
        """손(손바닥·손가락 링크의 형상)과 상자 옆면 사이의 좌우 틈 (N,) m. 양손 중 작은 값, 음수 = 상자 안으로 들어감.

        box_pos·box_yaw 를 주면 그 자리의 (같은 크기) 상자와의 틈을 잰다. 없으면 지금 다루는 상자.

        링크마다 형상의 경계 상자(링크 좌표)를 한 번 읽어 3×3×3 점으로 나누고, 지금 자세로 옮겨 상자 좌표에서 본다.
        상자의 앞뒤·높이 범위 안에 있는 점만 보고 |좌우 위치| − 상자 폭의 절반 을 잰다 (경계 상자라 실제 형상보다 조금 크게 잡힘).
        """
        from isaaclab.utils.math import quat_apply

        if not hasattr(self, "_hand_pts"):
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            self._hand_pts, self._hand_ids = [], []
            for side in ("left", "right"):
                for part in self.HAND_LINKS:
                    name = f"{side}_{part}_link"
                    rng = cache.ComputeUntransformedBound(stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")).ComputeAlignedRange()
                    lo, hi = rng.GetMin(), rng.GetMax()
                    axes = [[lo[k] + (hi[k] - lo[k]) * f for f in (0.0, 0.5, 1.0)] for k in range(3)]
                    self._hand_pts.append(torch.tensor([[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]], device=self.device))
                    self._hand_ids.append(self.robot.body_names.index(name))
        clear = torch.full((self.n,), float("inf"), device=self.device)
        if box_pos is None:
            box_pos, box_yaw = self.box.data.root_pos_w, yaw_of(self.box.data.root_quat_w)
        size = self.box_size
        for pts, bid in zip(self._hand_pts, self._hand_ids):
            k = pts.shape[0]
            quat = self.robot.data.body_quat_w[:, bid][:, None, :].expand(-1, k, -1).reshape(-1, 4)
            w = self.robot.data.body_pos_w[:, bid][:, None, :] + quat_apply(quat, pts[None].expand(self.n, -1, -1).reshape(-1, 3)).reshape(self.n, k, 3)
            rel = rot_z(-box_yaw[:, None].expand(-1, k), w - box_pos[:, None, :])
            near = (rel[..., 0].abs() < size[0] / 2) & (rel[..., 2].abs() < size[2] / 2)
            gap = torch.where(near, rel[..., 1].abs() - size[1] / 2, torch.full_like(rel[..., 1], float("inf")))
            clear = torch.minimum(clear, gap.min(dim=1).values)
        return clear

    def _attach(self, mask: torch.Tensor):
        if not bool(mask.any()):
            return
        self.rec["hand_clear"][mask] = self.hand_box_clearance()[mask]
        self.rec["hand_clear_min"][mask] = self.rec["hand_clear"][mask]
        mid, psi = self.hand_frame()
        self.att_off[mask] = rot_z(-psi, self.box.data.root_pos_w - mid)[mask]
        self.att_yaw[mask] = wrap(yaw_of(self.box.data.root_quat_w) - psi)[mask]
        self.attached[mask] = True
        self.box_prev_valid[mask] = False
        # 손에 무게가 실리면 팔이 1~2cm 처지는데, 상자를 그대로 따라 내리면 받침대에 눌림 → 들어 올릴 때까지 지금 높이를 바닥으로
        self.att_floor[mask] = self.box.data.root_pos_w[mask, 2]
        ids = mask.nonzero().flatten()
        self._set_palm_extra_mass(ids, self.box_mass[ids] / 2)
        self._update_gravity_cancel()

    def _detach(self, mask: torch.Tensor):
        if not bool(mask.any()):
            return
        self.attached[mask] = False
        ids = mask.nonzero().flatten()
        self._set_palm_extra_mass(ids, 0.0)
        self._update_gravity_cancel()
        self.box.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    def _move_attached_box(self):
        """붙어 있는 상자를 두 손 기준 자리로 옮긴다 (위치·자세 + 다음 스텝 동안 따라갈 속도)."""
        att = self.attached
        if not bool(att.any()):
            return
        if self.step % 10 == 0:  # 들고 있는 동안 손이 상자 안으로 들어가는지 (0.2초마다)
            self.rec["hand_clear_min"][att] = torch.minimum(self.rec["hand_clear_min"], self.hand_box_clearance())[att]
        mid, psi = self.hand_frame()
        pos = mid + rot_z(psi, self.att_off)
        pos[:, 2] = torch.maximum(pos[:, 2], self.att_floor)
        yaw = psi + self.att_yaw
        ids = att.nonzero().flatten()
        # 직전 스텝에 놓은 자리에서 물리 계산 뒤 얼마나 벗어났는지 (튀거나 떨면 커짐)
        dev = torch.norm(self.box.data.root_pos_w - self.box_pred_pos, dim=1)
        on = att & self.box_prev_valid
        in_air = self.box.data.root_pos_w[:, 2] > BOX_Z_ON_TABLE + 0.02
        self.box_dev_air[on & in_air] = torch.maximum(self.box_dev_air, dev)[on & in_air]
        self.box_dev_table[on & ~in_air] = torch.maximum(self.box_dev_table, dev)[on & ~in_air]
        vel = torch.zeros(self.n, 6, device=self.device)
        vel[on, :3] = ((pos - self.box_prev_pos) / self.dt)[on]
        vel[on, 5] = (wrap(yaw - self.box_prev_yaw) / self.dt)[on]
        self.box.write_root_pose_to_sim(torch.cat([pos, quat_from_yaw(yaw)], dim=1)[ids], env_ids=ids)
        self.box.write_root_velocity_to_sim(vel[ids], env_ids=ids)
        self.box_prev_pos[att] = pos[att]
        self.box_pred_pos[att] = (pos + vel[:, :3] * self.dt)[att]
        self.box_prev_yaw[att] = yaw[att]
        self.box_prev_valid[att] = True

    def _start_move(self, mask: torch.Tensor, goal_w: torch.Tensor, steps: int, world: bool = False):
        """mask 환경의 팔 이동을 시작: 지금 손 위치 → goal_w (N, 2, 3 월드).

        world=False: 목표를 지금의 몸통 좌표에 고정 (몸이 움직여도 팔 관절 목표가 바뀌지 않음). 손에 무게가 있을 때는 이쪽만 씀.
        world=True: 목표를 월드에 두고 매 스텝 몸통 좌표로 다시 바꿔 따라감. 빈손일 때만.
        """
        if not bool(mask.any()):
            return
        cur = self.torso.to_frame(self.ik.hand_pos_w())
        goal = self.torso.to_frame(goal_w)
        self.arm_start[mask], self.arm_goal[mask], self.arm_goal_w0[mask] = cur[mask], goal[mask], goal_w[mask]
        self.arm_t[mask], self.arm_T[mask] = 0, steps
        self.arm_busy[mask], self.arm_hold[mask] = True, True
        self.arm_world[mask] = world

    def _set_state(self, mask: torch.Tensor, new_state: int):
        if not bool(mask.any()):
            return
        if self.debug and bool(mask[0]):
            print(f"[dbg] t={self.step:5d} ({self.step * self.dt:5.1f}s) 환경 0: {STATE_NAMES[int(self.state[0])]} → {STATE_NAMES[new_state]}")
        self.state[mask] = new_state
        self.t_state[mask] = 0
        self.in_tol_count[mask] = 0
        self.near_mode[mask] = False

    def _fail(self, mask: torch.Tensor):
        if not bool(mask.any()):
            return
        self.fail_state[mask] = self.state[mask]
        if self.box is not None:
            self._detach(mask & self.attached)
        self._set_state(mask, S_FAIL)
        self.arm_busy[mask], self.arm_hold[mask], self.arm_world[mask] = False, False, False
        self.j_move[mask] = False

    def _is_still(self) -> torch.Tensor:
        """최근 1초 동안 골반이 STILL_POS·STILL_YAW 넘게 움직이지 않았는지. 상태가 바뀐 뒤 1초는 지나야 함."""
        k = self.step % STILL_STEPS  # 이 칸에는 1초 전 값이 들어 있음 (이번 스텝 값으로 덮어쓰기 전)
        moved = torch.norm(self.robot.data.root_pos_w[:, :2] - self.pos_hist[k], dim=1)
        turned = wrap(self.robot.data.heading_w - self.yaw_hist[k]).abs()
        return (self.t_state >= STILL_STEPS) & (moved < STILL_POS) & (turned < STILL_YAW)

    def _enter_backoff(self, mask: torch.Tensor, back_state: int, walk_state: int):
        """걷기 전에: 받침대 중심이 BACKOFF_NEAR 안에 있으면 먼저 뒤로 물러난다 (legacy 는 물러나지 않음)."""
        if not bool(mask.any()):
            return
        pos = self.robot.data.root_pos_w[:, :2]
        near = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if self.obstacles.shape[0] > 0 and not self.legacy:
            near = (torch.norm(self.obstacles - pos[None], dim=-1) < BACKOFF_NEAR).any(dim=0)
        back = mask & near
        self.back_start[back], self.back_yaw[back] = pos[back], self.robot.data.heading_w[back]
        self._set_state(back, back_state)
        self._set_state(mask & ~near, walk_state)

    def _backoff_command(self):
        """뒤로 물러나기: 물러나기 시작한 방향을 유지하며 뒤로 걷는다. (명령, 다 물러났는지)"""
        yaw = self.robot.data.heading_w
        d = self.robot.data.root_pos_w[:, :2] - self.back_start
        moved_back = -(d[:, 0] * torch.cos(self.back_yaw) + d[:, 1] * torch.sin(self.back_yaw))
        cmd = torch.zeros(self.n, 3, device=self.device)
        cmd[:, 0] = -BACKOFF_SPEED
        cmd[:, 2] = torch.clamp(1.5 * wrap(self.back_yaw - yaw), -0.4, 0.4)
        return cmd, moved_back >= BACKOFF_DIST

    def _walk_command(self, target_xy: torch.Tensor, target_yaw: torch.Tensor, loaded: bool):
        """목표 지점·방향까지의 속도 명령 (N, 3) 과 도착 여부, 오차(앞, 옆, 방향)를 돌려준다.

        - 목표가 가깝고(2m 안) 방향 차이가 작으면(0.6 rad 안) 돌지 않고 옆으로 먼저 맞춘 뒤 앞으로 다가감 (legacy 는 0.3m 안에서만)
        - 그 밖에는 목표 쪽을 보게 돈 뒤(방향 차이 > 0.6 rad 이면 제자리 회전) 앞으로 걸음
        - 정책은 명령이 0.1 미만이면 '정지'로 보므로 움직여야 할 때는 0.15 이상으로 줌
        - legacy: 정지 명령은 STOP_PHASE 박자에서만 주고(그때까지 회전 0.12 rad/s 로 제자리걸음), 멈추는 동안 움직이는 만큼 미리 보정
        """
        yaw = self.robot.data.heading_w
        d = target_xy - self.robot.data.root_pos_w[:, :2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        ex, ey = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        dist = torch.hypot(ex, ey)
        bearing = torch.atan2(ey, ex)
        yaw_err = wrap(target_yaw - yaw)
        if self.legacy:
            self.near_mode.copy_(torch.where(self.near_mode, dist < 0.5, dist < 0.3))
        else:
            close = (dist < HOLONOMIC_DIST) & (yaw_err.abs() < HOLONOMIC_YAW)
            self.near_mode.copy_(torch.where(self.near_mode, dist < HOLONOMIC_DIST + 0.3, close))
        near = self.near_mode

        turn_only = bearing.abs() > 0.6
        wz_far = torch.clamp(1.5 * bearing, -0.8, 0.8)
        wz_far = torch.where(turn_only & (wz_far.abs() < 0.4), 0.4 * torch.sign(bearing), wz_far)
        vx_far = torch.where(turn_only, torch.zeros_like(dist), torch.clamp(1.0 * dist, 0.25, 0.5))

        k = 1 if loaded else 0
        if self.legacy:
            yaw_aim, ey_aim, ex_aim = yaw_err - STOP_YAW_DRIFT[k], ey - STOP_LAT_DRIFT[k], ex - STOP_FWD_DRIFT[k]
        else:
            yaw_aim, ey_aim, ex_aim = yaw_err, ey, ex
        zero = torch.zeros_like(ex)
        need_y, need_yaw = ey_aim.abs() > 0.03, yaw_aim.abs() > 0.06
        if self.legacy:  # 뒤로 걷기를 배우지 않은 정책: 앞으로만
            need_x = ex_aim > 0.02
            vx_near = torch.where(need_x, torch.clamp(1.5 * ex_aim, 0.15, 0.4), zero)
        else:  # 지나쳤으면(3cm 넘게) 뒤로 되돌아오고, 옆으로 덜 맞았으면 앞으로 가지 않음
            need_x = (ex_aim > 0.02) | (ex_aim < -0.03)
            vx_near = torch.where(ex_aim > 0, torch.clamp(1.5 * ex_aim, 0.15, 0.4), -torch.clamp(-1.5 * ex_aim, 0.15, 0.3))
            vx_near = torch.where(need_x & ~((ex_aim > 0) & (ey_aim.abs() > ALIGN_Y)), vx_near, zero)
            need_x = need_x & ~((ex_aim > 0) & (ex_aim <= 0.02))
        vy_near = torch.where(need_y, torch.sign(ey_aim) * torch.clamp(1.5 * ey_aim.abs(), 0.15, 0.35), zero)
        wz_near = torch.where(need_yaw, torch.sign(yaw_aim) * torch.clamp(1.5 * yaw_aim.abs(), 0.15, 0.5), zero)
        in_place = ~need_x & ~need_y & ~need_yaw
        if self.legacy:
            phase = torch.remainder(self.env.episode_length_buf.float() * self.dt / GAIT_PERIOD, 1.0)
            stop_now = in_place & (phase >= STOP_PHASE[0]) & (phase < STOP_PHASE[1])
            wait_sign = torch.where(yaw_aim >= 0, torch.ones_like(yaw_aim), -torch.ones_like(yaw_aim))
            wz_near = torch.where(in_place & ~stop_now, 0.12 * wait_sign, wz_near)
        else:
            stop_now = in_place

        cmd = torch.stack([torch.where(near, vx_near, vx_far), torch.where(near, vy_near, zero),
                           torch.where(near, wz_near, wz_far)], dim=1)
        arrived = near & stop_now
        cmd[arrived] = 0.0
        return cmd, arrived, ex, ey, yaw_err

    def _stance_ok(self, ex, ey, yerr):
        tol = self.stance_tol
        return (ex >= tol[:, 0]) & (ex <= tol[:, 1]) & (ey.abs() <= tol[:, 2]) & (yerr.abs() <= tol[:, 3])

    # ------------------------------------------------------------------ 매 스텝
    @property
    def active(self) -> torch.Tensor:
        return (self.state != S_DONE) & (self.state != S_FAIL) & (self.state != S_IDLE)

    @property
    def finished(self) -> torch.Tensor:
        return ~self.active

    def pre_step(self):
        """env.step 전에: 팔 목표값과 붙은 상자를 갱신한다."""
        if bool(self.arm_world.any()):  # 월드에 둔 목표는 매 스텝 지금의 몸통 좌표로 다시 바꿈
            self.arm_goal[self.arm_world] = self.torso.to_frame(self.arm_goal_w0)[self.arm_world]
        a = torch.clamp((self.arm_t + 1).float() / self.arm_T.float(), 0.0, 1.0)
        a = (10 * a**3 - 15 * a**4 + 6 * a**5)[:, None, None]
        self.ik.step(self.torso.to_world(self.arm_start + a * (self.arm_goal - self.arm_start)),
                     mask=self.arm_busy | self.arm_hold)
        j_done = torch.zeros_like(self.j_move)
        if bool(self.j_move.any()):  # 관절 공간에서 짐 든 자세로 (IK 를 쓰지 않는 환경)
            jm = self.j_move
            self.j_t[jm] += 1
            a = torch.clamp(self.j_t.float() / self.j_T.float(), 0.0, 1.0)
            a = (10 * a**3 - 15 * a**4 + 6 * a**5)[:, None]
            default = self.robot.data.default_joint_pos[:, self.ik.all_joint_ids]
            self.ik.q_cmd[jm] = (self.j_from + a * (default - self.j_from))[jm]
            j_done = jm & (self.j_t >= self.j_T)
            self.j_move &= ~j_done
        self.ik.apply()
        self.arm_t += self.arm_busy.long()
        self.move_done = (self.arm_busy & (self.arm_t >= self.arm_T)) | j_done
        self.arm_busy &= ~self.move_done
        if self.box is not None:
            self._move_attached_box()

    def post_step(self, terminated: torch.Tensor) -> torch.Tensor:
        """env.step 뒤에: 넘어짐·제한 시간·상태 전환을 처리하고 다음 스텝의 속도 명령 (N, 3) 을 돌려준다."""
        n, dev, robot = self.n, self.device, self.robot
        self.t_state += 1
        new_fall = terminated & self.active
        if bool(new_fall.any()):
            self.fell |= new_fall
            for e in new_fall.nonzero().flatten().tolist():
                parts = []
                for k in range(self.obstacles.shape[0]):
                    xy = self.obstacles[k, e]
                    rel = rot_z(-self.last_yaw[e], torch.cat([xy - self.last_pos[e, :2], torch.zeros(1, device=dev)]))
                    parts.append(f"받침대 {k}: 골반 기준 앞 {rel[0].item() * 100:+.0f}·왼쪽 {rel[1].item() * 100:+.0f}cm")
                self.fall_info[e] = (f"{STATE_NAMES[int(self.state[e])]} {self.t_state[e].item() * self.dt:.1f}초째, {', '.join(parts)}, "
                                     f"기울기 앞뒤 {torch.atan2(self.last_grav[e, 0], -self.last_grav[e, 2]).item() * RAD2DEG:+.0f}° "
                                     f"좌우 {torch.atan2(-self.last_grav[e, 1], -self.last_grav[e, 2]).item() * RAD2DEG:+.0f}°")
            self.ik.reset(new_fall.nonzero().flatten())
            self._fail(new_fall)
        timed_out = torch.zeros(n, dtype=torch.bool, device=dev)
        for s_code, limit in TIMEOUT.items():
            timed_out |= (self.state == s_code) & (self.t_state > limit)
        self._fail(timed_out)

        cmd = torch.zeros(n, 3, device=dev)
        hand_w = self.ik.hand_pos_w()
        st = self.state.clone()  # 한 스텝에 상태가 한 번만 바뀌게, 조건은 이 값으로만 본다
        move_done = self.move_done

        # 팔 벌리기 → (물러나기) → A 로 걷기
        m = (st == S_OPEN) & move_done
        self.arm_hold[m] = False
        self._enter_backoff(m, S_BACK_A, S_WALK_A)

        for back_state, walk_state in ((S_BACK_A, S_WALK_A), (S_BACK_B, S_WALK_B)):
            if self.legacy and back_state == S_BACK_B:
                continue  # legacy 는 아래에서 옆으로 비켜서기로 처리
            m = st == back_state
            if bool(m.any()):
                c, done = self._backoff_command()
                cmd[m] = c[m]
                self._set_state(m & done, walk_state)

        # A 로 걷기 → 멈춤 → (자리가 범위 밖이면 다시 다가감) → 손 뻗기
        m = st == S_WALK_A
        if bool(m.any()):
            c, arrived, ex, ey, yerr = self._walk_command(self.stance_a_xy, self.src_yaw, loaded=False)
            cmd[m] = c[m]
            self.arrive_yaw[m & arrived] = robot.data.heading_w[m & arrived]
            self._set_state(m & arrived, S_SETTLE_A)
        m = st == S_SETTLE_A
        if bool(m.any()):
            still = m & self._is_still()
            if bool(still.any()):
                _, _, ex, ey, yerr = self._walk_command(self.stance_a_xy, self.src_yaw, loaded=False)
                retry = still & ~self._stance_ok(ex, ey, yerr) & (self.reapproach_a < MAX_REAPPROACH)
                self.reapproach_a[retry] += 1
                self._set_state(retry, S_WALK_A)
                go = still & ~retry
                self.rec["stance_a_x"][go], self.rec["stance_a_y"][go], self.rec["stance_a_yaw"][go] = ex[go], ey[go], yerr[go]
                self.rec["stop_turn_a"][go] = wrap(robot.data.heading_w - self.arrive_yaw)[go]
                if self.until == "stand":  # 여기서부터는 밖에서 팔을 움직임 (VLA 등)
                    self.arm_hold[go] = False
                    self._set_state(go, S_DONE)
                else:
                    self._start_move(go, self.box_grasp_points(), 100, world=True)
                    self._set_state(go, S_CLOSE)

        # 손 뻗기 (빈손이라 월드 목표를 계속 따라감) → 손 위치 맞추기 → 붙이고 들어 올리기
        m = (st == S_CLOSE) & move_done
        self._set_state(m, S_CLOSE2)
        m = st == S_CLOSE2
        if bool(m.any()):
            err2 = torch.norm(hand_w - self.box_grasp_points(), dim=-1)
            err = err2.max(dim=1).values
            self.rec["grasp_err"][m] = err[m]
            self.rec["box_shift"][m] = torch.norm(self.box.data.root_pos_w[:, :2] - self.src_xy, dim=1)[m]
            self.in_tol_count[m] = torch.where(err <= ATTACH_TOL, self.in_tol_count + 1, torch.zeros_like(self.in_tol_count))[m]
            ok = m & (self.in_tol_count >= ATTACH_HOLD)
            self._attach(ok)
            up = hand_w.clone()
            up[..., 2] += LIFT_UP
            self._start_move(ok, up, 50)  # 여기서부터 손에 무게가 있으므로 목표는 몸통 좌표에 고정
            self._set_state(ok, S_LIFT1)

        # 들어 올리기 → 든 자세로 → 잠깐 유지
        m = (st == S_LIFT1) & move_done
        self.att_floor[m] = -1.0e9
        self._start_move(m, self._pelvis_to_world(CARRY_BOX_X, self.grasp_y, CARRY_BOX_Z), 100)
        self._set_state(m, S_LIFT2)
        m = (st == S_LIFT2) & move_done
        self._set_state(m, S_HOLD)
        lifting = (st == S_HOLD) | (st == S_LIFT1) | (st == S_LIFT2)
        self.pick_tilt_max[lifting] = torch.maximum(self.pick_tilt_max, self.tilt())[lifting]
        m = (st == S_HOLD) & (self.t_state >= (100 if self.until == "lift" else 25))
        if bool(m.any()):
            self.rec["t_pick"][m] = ((self.step - self.t_start) * self.dt)[m]
            self.rec["lift_tilt"][m] = self.pick_tilt_max[m]
            if self.until == "lift":
                self._set_state(m, S_DONE)
            else:
                self.arm_hold[m] = False  # 걷는 동안은 관절 목표 고정
                if self.legacy:
                    self.back_start[m] = robot.data.root_pos_w[m, :2]
                    self._set_state(m, S_BACK_B)
                else:
                    self._enter_backoff(m, S_BACK_B, S_WALK_B)

        # legacy: 뒤로 물러나는 대신 A 를 마주 본 채 왼쪽으로 비켜섬
        if self.legacy:
            m = st == S_BACK_B
            if bool(m.any()):
                left = torch.stack([-torch.sin(self.src_yaw), torch.cos(self.src_yaw)], dim=1)
                moved = ((robot.data.root_pos_w[:, :2] - self.back_start) * left).sum(dim=1)
                cmd[m, 1] = SIDESTEP_SPEED
                cmd[m, 2] = torch.clamp(1.5 * wrap(self.src_yaw - robot.data.heading_w), -0.4, 0.4)[m]
                self._set_state(m & (moved >= SIDESTEP_DIST), S_WALK_B)

        # B 로 걷기 → 멈춤 → (다시 다가감) → 내려놓기
        m = st == S_WALK_B
        if bool(m.any()):
            c, arrived, ex, ey, yerr = self._walk_command(self.stance_b_xy, self.dst_yaw, loaded=not self.walk_only)
            cmd[m] = c[m]
            self.arrive_yaw[m & arrived] = robot.data.heading_w[m & arrived]
            self._set_state(m & arrived, S_SETTLE_B)
        m = st == S_SETTLE_B
        if bool(m.any()):
            still = m & self._is_still()
            if bool(still.any()):
                _, _, ex, ey, yerr = self._walk_command(self.stance_b_xy, self.dst_yaw, loaded=not self.walk_only)
                retry = still & ~self._stance_ok(ex, ey, yerr) & (self.reapproach_b < MAX_REAPPROACH)
                self.reapproach_b[retry] += 1
                self._set_state(retry, S_WALK_B)
                go = still & ~retry
                self.rec["stance_b_x"][go], self.rec["stance_b_y"][go], self.rec["stance_b_yaw"][go] = ex[go], ey[go], yerr[go]
                self.rec["stop_turn_b"][go] = wrap(robot.data.heading_w - self.arrive_yaw)[go]
                if self.walk_only or self.until == "stand":  # 받침대 앞에 서는 데까지만 (그 뒤는 밖에서 팔을 움직임)
                    self.rec["t_total"][go] = ((self.step - self.t_start) * self.dt)[go]
                    self._set_state(go, S_DONE)
                else:
                    # 상자 중심이 받침대 B 가운데 위로 가도록 양손을 같은 만큼 옮김
                    self._start_move(go, hand_w + (self.place_target - self.box.data.root_pos_w)[:, None, :], 125)
                    self._set_state(go, S_PLACE)

        if self.box is not None:
            m = (st == S_PLACE) & move_done
            self._start_move(m, hand_w + (self.place_target - self.box.data.root_pos_w)[:, None, :], 40)
            self._set_state(m, S_PLACE2)

            # 떼기 → 0.3초 뒤 손 벌리기 (양손을 바깥으로 7cm) → 팔 거두기
            m = (st == S_PLACE2) & move_done
            if bool(m.any()):
                self.rec["place_err_before"][m] = torch.norm(self.box.data.root_pos_w - self.place_target, dim=1)[m]
                self._detach(m)
                self._set_state(m, S_RELEASE)
            m = (st == S_RELEASE) & (self.t_state >= 15)
            if bool(m.any()):
                _, psi = self.hand_frame()
                out = rot_z(psi[:, None].expand(-1, 2), self._sym(0.0, 0.07, 0.0)[None].expand(n, -1, -1))
                self._start_move(m, hand_w + out, 40)
                self._set_state(m, S_OPEN_B)
            m = (st == S_OPEN_B) & move_done
            if self.legacy:  # 제자리에서 IK 로 팔을 몸 쪽으로 거둠
                self._start_move(m, self._pelvis_to_world(RETRACT_POSE[0], RETRACT_POSE[1], RETRACT_POSE[2]), 75)
                self._set_state(m, S_RETRACT)
            elif bool(m.any()):  # 손을 벌린 채 뒤로 물러난 뒤, 관절 공간에서 짐 든 자세로
                self.arm_hold[m] = False
                self.back_start[m], self.back_yaw[m] = robot.data.root_pos_w[m, :2], robot.data.heading_w[m]
                self._set_state(m, S_BACK_END)
            m = st == S_BACK_END
            if bool(m.any()):
                c, done = self._backoff_command()
                cmd[m] = c[m]
                self.arms_to_carry(m & done)
                self._set_state(m & done, S_RETRACT)
            m = (st == S_RETRACT) & move_done
            self._set_state(m, S_END_HOLD)
            m = (st == S_END_HOLD) & (self.t_state >= 50)
            if bool(m.any()):
                self.rec["place_xy"][m] = torch.norm(self.box.data.root_pos_w[:, :2] - self.dst_xy, dim=1)[m]
                self.rec["place_z"][m] = (self.box.data.root_pos_w[:, 2] - BOX_Z_ON_TABLE)[m]
                self.rec["t_total"][m] = ((self.step - self.t_start) * self.dt)[m]
                self.arm_hold[m] = False
                self._set_state(m, S_DONE)

        # 기록
        ok_env = self.active
        self.last_pos[ok_env], self.last_yaw[ok_env] = robot.data.root_pos_w[ok_env], robot.data.heading_w[ok_env]
        self.last_grav[ok_env] = robot.data.projected_gravity_b[ok_env]
        if bool(ok_env.any()):
            qc, st_ok = self.ik.q_cmd[ok_env], self.state[ok_env]
            lo, hi = qc.min(dim=0), qc.max(dim=0)
            new_lo, new_hi = lo.values < self.q_cmd_min, hi.values > self.q_cmd_max
            self.q_cmd_min_state[new_lo] = st_ok[lo.indices][new_lo]  # 그 값이 나온 상태
            self.q_cmd_max_state[new_hi] = st_ok[hi.indices][new_hi]
            self.q_cmd_min = torch.minimum(self.q_cmd_min, lo.values)
            self.q_cmd_max = torch.maximum(self.q_cmd_max, hi.values)
        self.pos_hist[self.step % STILL_STEPS] = robot.data.root_pos_w[:, :2]
        self.yaw_hist[self.step % STILL_STEPS] = robot.data.heading_w
        self.step += 1
        return cmd
