"""11단계: 용접 동작 — 토치 끝을 용접선 따라 움직이는 IK, 철골과의 거리 판정, 용접 상태 머신, 겉모습 효과.

- TorchIK: 오른팔 5관절로 토치 끝(tip)의 위치를 맞추고, 남는 자유도로 토치 방향을 원하는 쪽에 가깝게 돌린다.
  위치가 먼저, 방향은 위치를 해치지 않는 범위(영공간)에서만. 팔 관절이 5개라 방향(2자유도)까지 항상 맞출 수는 없다.
- FrameClearance: 토치·오른손과 철골 판(site_layout.STEEL_FRAME_PLATES) 사이의 거리. 0 이하면 닿은 것으로 본다.
- WeldMachine: 준비 자세 → 철골 앞으로 걷기 → 띄운 점 → 용접선 따라 이동 → 빼기 → 팔 거두기.
- WeldEffects: 불꽃(깜빡이는 점광원 + 빛나는 작은 구)과 용접 비드(작은 구를 이어 놓음). 겉모습만.

이 모듈은 isaaclab 을 import 하므로 Isaac Sim 앱이 켜진 뒤에만 import 한다.
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import matrix_from_quat

from . import site_layout as sl
from .arm_ik import ARM_JOINTS
from .weld_cfg import TORCH, TORCH_GRIP

RAD2DEG = 57.29578
RIGHT_ARM_JOINTS = [f"right_{j}_joint" for j in ARM_JOINTS]
LEFT_ARM_JOINTS = [f"left_{j}_joint" for j in ARM_JOINTS]
# 용접하는 동안의 왼팔: 몸 옆으로 내린 자세 (어깨 pitch·roll·yaw, 팔꿈치 pitch·roll. 걷기 정책의 팔 무작위 범위 안)
LEFT_SIDE_POSE = (0.30, 0.20, 0.0, 0.90, 0.0)
WELD_OFFSET = 0.05  # 띄운 점: 용접선에서 바깥 방향으로 5cm
WELD_GAP = 0.01  # 용접하는 동안 토치 끝과 용접선 사이의 거리
WELD_END_TRIM_V = 0.015  # 세로선은 양 끝을 이만큼 줄인다: 세로선은 위·아래 플랜지 사이(높이 18cm)에 있어, 끝까지 가면 노즐(반지름 6mm)이 플랜지에 닿는다
WELD_TILT_V = 0.0  # 세로선의 양 끝에서 토치를 끝 쪽으로 기울이는 각도 (rad). 기울이면 끝을 덜 줄여도 되지만 팔이 그 방향을 따라가지 못해 쓰지 않는다


def _skew(v: torch.Tensor) -> torch.Tensor:
    """(N, 3) → (N, 3, 3) 외적 행렬."""
    z = torch.zeros_like(v[:, 0])
    return torch.stack([torch.stack([z, -v[:, 2], v[:, 1]], dim=1), torch.stack([v[:, 2], z, -v[:, 0]], dim=1),
                        torch.stack([-v[:, 1], v[:, 0], z], dim=1)], dim=1)


class TorchIK:
    """토치 끝 위치(먼저) + 토치 방향(남는 자유도로) → 오른팔 관절 목표값."""

    def __init__(self, robot: Articulation, torch_geom: dict = TORCH, lambda_pos: float = 0.02, gain: float = 0.3,
                 dir_gain: float = 0.15, max_joint_step: float = 0.04, max_dir_step: float = 0.03, max_dir_joint_step: float = 0.02):
        """
        Args:
            lambda_pos: 위치 풀이의 감쇠 계수 (팔을 다 편 자세 근처에서 관절이 튀지 않게).
            gain: 한 스텝에 위치 보정량의 몇 배를 더할지.
            dir_gain: 한 스텝에 방향 오차(rad)의 몇 배만큼 돌릴지.
            max_joint_step: 한 스텝에 위치 보정으로 관절 목표값이 바뀔 수 있는 최대 크기 (rad). 0.04 = 2 rad/s.
            max_dir_step: 한 스텝에 토치 방향을 돌리는 최대 각도 (rad). max_dir_joint_step: 방향 보정으로 관절이 바뀌는 최대 크기 (rad).
        """
        self.robot = robot
        dev = robot.device
        self.joint_ids = [robot.joint_names.index(j) for j in RIGHT_ARM_JOINTS]
        self.body_id = robot.body_names.index("right_palm_link")
        body_offset, joint_offset = (-1, 0) if robot.is_fixed_base else (0, 6)
        self._jac_body = self.body_id + body_offset
        self._jac_joint = [j + joint_offset for j in self.joint_ids]
        self.tip_local = torch.tensor(torch_geom["tip"], device=dev)
        self.axis_local = torch.tensor(torch_geom["axis"], device=dev)
        self.lambda_pos, self.gain, self.dir_gain, self.max_joint_step = lambda_pos, gain, dir_gain, max_joint_step
        self.max_dir_step, self.max_dir_joint_step = max_dir_step, max_dir_joint_step
        self.nom_gain = 0.10  # 한 스텝에 기준 자세와의 차이의 몇 배를 당길지
        self.dir_damp_nom = 0.15  # 기준 자세를 줄 때의 방향 풀이 감쇠: 클수록 기준 자세를, 작을수록 방향을 따른다
        limits = robot.data.soft_joint_pos_limits[:, self.joint_ids]
        self.q_min, self.q_max = limits[..., 0].clone(), limits[..., 1].clone()

    def palm_rot(self) -> torch.Tensor:
        return matrix_from_quat(self.robot.data.body_quat_w[:, self.body_id])

    def tip_pos_w(self) -> torch.Tensor:
        """토치 끝 위치 (N, 3)."""
        return self.robot.data.body_pos_w[:, self.body_id] + (self.palm_rot() @ self.tip_local)

    def axis_w(self) -> torch.Tensor:
        """토치가 향하는 방향 (N, 3)."""
        return self.palm_rot() @ self.axis_local

    def solve(self, q_cmd: torch.Tensor, target_pos_w: torch.Tensor, axis_des_w: torch.Tensor | None = None,
              feed: torch.Tensor | None = None, q_nom: torch.Tensor | None = None) -> torch.Tensor:
        """직전 관절 목표값 q_cmd (N, 5) 에 보정을 더한 새 목표값을 돌려준다. 오차는 지금 측정한 손 자세로 계산한다.

        feed (N, 3): 이번 스텝에 목표점이 움직인 만큼. 주면 그만큼을 그대로 더해(앞먹임) 움직이는 목표를 뒤처지지 않고 따라간다.
        q_nom (N, 5): 이 목표점에서의 기준 자세. 주면 토치 끝을 움직이지 않는 범위에서 팔을 이 자세 쪽으로 당기고, 그 위에서 방향을 맞춘다.
            방향만 보고 풀면 서는 자리가 조금 어긋났을 때 팔이 다른 꼴(팔꿈치를 편 꼴 등)로 넘어가 토치가 철골에 닿았다 (2026-11-27).
        """
        rot = self.palm_rot()
        r = rot @ self.tip_local  # 손 → 토치 끝 (월드)
        tip = self.robot.data.body_pos_w[:, self.body_id] + r
        jac = self.robot.root_physx_view.get_jacobians()[:, self._jac_body][:, :, self._jac_joint]  # (N, 6, 5)
        jv, jw_all = jac[:, :3], jac[:, 3:]
        j1_all = jv - _skew(r) @ jw_all  # 토치 끝의 속도 = 손 속도 + 각속도 × r
        eye3 = torch.eye(3, device=tip.device).expand(tip.shape[0], 3, 3)
        eye5 = torch.eye(5, device=tip.device).expand(tip.shape[0], 5, 5)
        e1 = self.gain * (target_pos_w - tip)
        if feed is not None:
            e1 = e1 + feed
        if axis_des_w is not None:
            a = rot @ self.axis_local
            # 방향: a 를 axis_des 쪽으로 돌리는 각속도. a 둘레의 회전은 상관없으므로 a 에 수직인 성분만 본다.
            # 한 스텝에 돌리는 양을 작게 묶는다: 방향이 많이 틀어져 있을 때 이 보정이 커지면 위치까지 흐트러진다
            w_des = self.dir_gain * torch.linalg.cross(a, axis_des_w)
            w_norm = w_des.norm(dim=1, keepdim=True).clamp(min=1.0e-9)
            w_des = (w_des * torch.clamp(self.max_dir_step / w_norm, max=1.0))[:, :, None]
            proj = eye3 - a[:, :, None] @ a[:, None, :]
        # 관절 한계: 풀어 보고 한계를 넘는 관절이 있으면 그 관절을 빼고(자코비안의 그 열을 0 으로) 다시 푼다.
        # 한계에서 잘라 내기만 하면 위치 오차가 남고, 그 오차를 다른 관절이 메우려다 팔이 엉뚱한 꼴로 넘어간다
        active = torch.ones(tip.shape[0], 5, dtype=torch.bool, device=tip.device)
        for _ in range(3):
            mask = active.float()[:, None, :]
            j1, jw = j1_all * mask, jw_all * mask
            j1t = j1.transpose(1, 2)
            dq = j1t @ torch.linalg.solve(j1 @ j1t + self.lambda_pos**2 * eye3, e1[:, :, None])
            dq = torch.clamp(dq, -self.max_joint_step, self.max_joint_step)
            if axis_des_w is not None:
                null = eye5 - j1t @ torch.linalg.solve(j1 @ j1t + 1.0e-5 * eye3, j1)  # 토치 끝을 움직이지 않는 관절 움직임
                j2 = proj @ jw @ null
                j2t = j2.transpose(1, 2)
                z0 = torch.zeros_like(dq) if q_nom is None else (self.nom_gain * (q_nom - q_cmd) * active.float())[:, :, None]
                damp = self.dir_damp_nom if q_nom is not None else 0.05
                dq2 = null @ (z0 + j2t @ torch.linalg.solve(j2 @ j2t + damp**2 * eye3, w_des - proj @ jw @ dq - j2 @ z0))
                dq = dq + torch.clamp(dq2, -self.max_dir_joint_step, self.max_dir_joint_step)
            q_new = q_cmd + dq[:, :, 0] * active.float()
            over = ((q_new < self.q_min) | (q_new > self.q_max)) & active
            if not bool(over.any()):
                break
            active = active & ~over
        return torch.clamp(q_new, self.q_min, self.q_max)


class FrameClearance:
    """토치·오른손·몸과 철골 판 사이의 거리 (m). 판은 환경마다 철골 자리로 옮겨 놓은 상자들."""

    # (링크 이름, 반지름): 링크 원점을 가운데로 한 공으로 어림한다
    HAND_SPHERES = [("right_palm_link", 0.025), ("right_three_link", 0.02), ("right_four_link", 0.02), ("right_five_link", 0.02),
                    ("right_six_link", 0.02), ("right_one_link", 0.02), ("right_two_link", 0.02), ("right_zero_link", 0.02)]
    BODY_SPHERES = [("right_elbow_pitch_link", 0.04), ("right_elbow_roll_link", 0.04), ("torso_link", 0.13), ("pelvis", 0.12),
                    ("left_knee_link", 0.06), ("right_knee_link", 0.06), ("left_ankle_roll_link", 0.10), ("right_ankle_roll_link", 0.10),
                    ("left_palm_link", 0.05)]

    def __init__(self, robot: Articulation, frame_pos_w: torch.Tensor, frame_yaw: torch.Tensor, torch_geom: dict = TORCH):
        """frame_pos_w (N, 3): 환경별 철골 원점의 월드 좌표, frame_yaw (N,): 철골 방향 (rad)."""
        self.robot = robot
        dev = robot.device
        self.frame_pos, self.frame_yaw = frame_pos_w, frame_yaw
        self.centers = torch.tensor([c for c, _ in sl.STEEL_FRAME_PLATES], device=dev)
        self.half = torch.tensor([s for _, s in sl.STEEL_FRAME_PLATES], device=dev) / 2
        self.palm_id = robot.body_names.index("right_palm_link")
        pts, rad = [], []
        for _, p0, p1, radius, _ in torch_geom["parts"]:  # 토치: 원통 축 위에 1cm 간격의 점
            a, b = torch.tensor(p0), torch.tensor(p1)
            k = max(2, int(math.ceil(float(torch.norm(b - a)) / 0.01)) + 1)
            for s in torch.linspace(0, 1, k):
                pts.append(a + s * (b - a))
                rad.append(radius)
        self.torch_pts = torch.stack(pts).to(dev)
        self.torch_rad = torch.tensor(rad, device=dev)
        self.hand_ids = [robot.body_names.index(n) for n, _ in self.HAND_SPHERES]
        self.hand_rad = torch.tensor([r for _, r in self.HAND_SPHERES], device=dev)
        self.body_ids = [robot.body_names.index(n) for n, _ in self.BODY_SPHERES]
        self.body_rad = torch.tensor([r for _, r in self.BODY_SPHERES], device=dev)

    def _dist(self, pts_w: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """월드의 점들 (N, K, 3) 과 철골 판 사이 거리에서 반지름을 뺀 값의 최소 (N,)."""
        rel = pts_w - self.frame_pos[:, None, :]
        c, s = torch.cos(self.frame_yaw)[:, None], torch.sin(self.frame_yaw)[:, None]
        local = torch.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1], rel[..., 2]], dim=-1)
        q = (local[:, :, None, :] - self.centers[None, None]).abs() - self.half[None, None]  # (N, K, 판, 3)
        d = torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)
        return (d.min(dim=2).values - radius[None]).min(dim=1).values

    def torch_clearance(self) -> torch.Tensor:
        rot = matrix_from_quat(self.robot.data.body_quat_w[:, self.palm_id])
        pts = self.robot.data.body_pos_w[:, self.palm_id][:, None, :] + (rot[:, None] @ self.torch_pts[None, :, :, None])[..., 0]
        return self._dist(pts, self.torch_rad)

    def hand_clearance(self) -> torch.Tensor:
        return self._dist(self.robot.data.body_pos_w[:, self.hand_ids], self.hand_rad)

    def body_clearance(self) -> torch.Tensor:
        return self._dist(self.robot.data.body_pos_w[:, self.body_ids], self.body_rad)


class ArmObsClamp:
    """걷기 정책에 넣는 관측에서 팔 관절 각도를 학습 때 본 범위 안으로 자른다 (실제 팔은 그대로 움직임).

    용접 자세는 걷기 정책(v2)이 학습 때 본 팔 범위(어깨 pitch -1.6~0.4, roll 0~0.9(바깥쪽), yaw ±0.3, 팔꿈치 pitch 0~1.8, roll ±0.2)를
    크게 벗어난다 (어깨를 몸 안쪽으로 1.2 rad, 어깨 yaw 1 rad 등). 본 적 없는 값이 들어가면 정책의 반응이 정해져 있지 않아,
    오른팔을 몸 앞으로 가로지르는 자세에서 서 있다가 넘어졌다 (2026-11-27, 12번 중 10번). 범위 끝 값으로 잘라 넣으면 학습 때 본 입력이 된다.
    """

    # (어깨 pitch, roll, yaw, 팔꿈치 pitch, roll) 의 학습 범위. roll 은 왼팔 기준이고 오른팔은 부호가 반대
    LOW = (-1.6, 0.0, -0.3, 0.0, -0.2)
    HIGH = (0.4, 0.9, 0.3, 1.8, 0.2)

    def __init__(self, env, robot: Articulation, group: str = "policy"):
        manager = env.observation_manager
        names = manager.active_terms[group]
        dims = [d[0] for d in manager.group_obs_term_dim[group]]
        start = dict(zip(names, [sum(dims[:i]) for i in range(len(dims))]))["joint_pos"]
        dev = robot.device
        cols, lo, hi = [], [], []
        for side in ("left", "right"):
            for j, name in enumerate(ARM_JOINTS):
                jid = robot.joint_names.index(f"{side}_{name}_joint")
                a, b = self.LOW[j], self.HIGH[j]
                if side == "right" and name == "shoulder_roll":
                    a, b = -b, -a
                default = float(robot.data.default_joint_pos[0, jid])
                cols.append(start + jid)
                lo.append(a - default)  # 관측은 기본 자세와의 차이
                hi.append(b - default)
        self.group, self.cols = group, cols
        self.lo, self.hi = torch.tensor(lo, device=dev), torch.tensor(hi, device=dev)

    def __call__(self, obs, mask: torch.Tensor | None = None):
        """mask (N,) bool 를 주면 그 환경의 관측만 자른다 (용접 중인 환경만 자르고 나머지는 그대로 둘 때)."""
        def clamp(x):
            cut = torch.clamp(x[:, self.cols], self.lo, self.hi)
            x[:, self.cols] = cut if mask is None else torch.where(mask[:, None], cut, x[:, self.cols])
            return x

        if isinstance(obs, torch.Tensor):
            return clamp(obs.clone())
        out = obs.clone()
        out[self.group] = clamp(out[self.group].clone())
        return out


def grip_targets(robot: Articulation):
    """토치를 쥔 손가락 자세: (관절 번호, 목표값 (1, K))."""
    ids = [robot.joint_names.index(n) for n in TORCH_GRIP]
    return ids, torch.tensor([list(TORCH_GRIP.values())], device=robot.device)


def set_torch_visible(env_ids, visible: bool) -> None:
    """토치 겉모습을 보이게/안 보이게 한다 (토치를 집고 내려놓는 동작은 만들지 않아, 용접할 때만 손에 나타나게 함)."""
    from pxr import UsdGeom

    from isaaclab.sim.utils import get_current_stage

    from .weld_cfg import TORCH_PRIM

    stage = get_current_stage()
    for i in env_ids:
        prim = stage.GetPrimAtPath(f"/World/envs/env_{int(i)}/Robot/{TORCH_PRIM}")
        if prim.IsValid():
            img = UsdGeom.Imageable(prim)
            img.MakeVisible() if visible else img.MakeInvisible()


# ---------------------------------------------------------------------------
# 겉모습 효과
# ---------------------------------------------------------------------------
class WeldEffects:
    """용접 불꽃과 비드 (겉모습만, 충돌 없음).

    - 아크: 용접선 위의 빛나는 작은 구 + 깜빡이는 점광원 (환경마다 하나)
    - 불꽃: 빛나는 작은 구 여러 개가 아크에서 튀어나와 떨어진다 (간단한 포물선, 수명 0.1~0.35초)
    - 비드: 지나간 자리에 작은 구를 약 3.5mm 간격으로 남긴다. 막 놓은 것은 1.5초 동안 달아오른 색
    구들은 PointInstancer 하나(VisualizationMarkers)로 그린다.
    """

    BEAD_STEP = 0.0035
    HOT_STEPS = 75

    def __init__(self, num_envs: int, device, n_sparks: int = 16, light_intensity: float = 4.0e5):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        def ball(radius, color, emissive=None):
            mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=emissive or (0.0, 0.0, 0.0), roughness=0.5, metallic=0.6)
            return sim_utils.SphereCfg(radius=radius, visual_material=mat)

        self.markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/WeldFx", markers={
            "bead": ball(0.0045, (0.30, 0.31, 0.34)),
            "hot": ball(0.0045, (1.0, 0.40, 0.08), (6.0, 1.6, 0.2)),
            "spark": ball(0.0022, (1.0, 0.85, 0.4), (12.0, 8.0, 2.0)),
            "arc": ball(0.006, (0.85, 0.92, 1.0), (20.0, 24.0, 30.0)),
        }))
        self.n, self.dev, self.k = num_envs, device, n_sparks
        self.light_intensity = light_intensity
        self.lights = []
        from pxr import Gf, UsdGeom, UsdLux

        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        for i in range(num_envs):
            cfg = sim_utils.SphereLightCfg(intensity=0.0, radius=0.012, color=(0.75, 0.85, 1.0))
            cfg.func(f"/World/WeldFx/light_{i}", cfg, translation=(0.0, 0.0, -5.0))
            prim = stage.GetPrimAtPath(f"/World/WeldFx/light_{i}")
            ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
            self.lights.append((UsdLux.SphereLight(prim).GetIntensityAttr(), ops["xformOp:translate"], Gf.Vec3d))
        self.light_on = [False] * num_envs
        self.sp_pos = torch.zeros(num_envs, n_sparks, 3, device=device)
        self.sp_vel = torch.zeros(num_envs, n_sparks, 3, device=device)
        self.sp_left = torch.zeros(num_envs, n_sparks, device=device)  # 남은 수명 (초). 0 이하면 숨김
        self.bead_pos = torch.zeros(0, 3, device=device)
        self.bead_born = torch.zeros(0, dtype=torch.long, device=device)
        self.last_bead = torch.full((num_envs, 3), 1.0e6, device=device)
        self.step = 0
        self.hidden = torch.tensor([[0.0, 0.0, -5.0]], device=device)

    def clear(self):
        self.bead_pos, self.bead_born = self.bead_pos[:0], self.bead_born[:0]
        self.last_bead[:] = 1.0e6
        self.sp_left[:] = 0.0

    def update(self, welding: torch.Tensor, seam_point: torch.Tensor, normal: torch.Tensor, dt: float):
        """welding (N,) bool: 지금 용접 중인 환경. seam_point (N, 3): 토치 끝에서 가장 가까운 용접선 위의 점. normal (N, 3): 바깥 방향."""
        self.step += 1
        arc = seam_point + 0.004 * normal
        # 비드
        new = welding & (torch.norm(seam_point - self.last_bead, dim=1) >= self.BEAD_STEP)
        if bool(new.any()):
            self.bead_pos = torch.cat([self.bead_pos, seam_point[new] + 0.002 * normal[new]])
            self.bead_born = torch.cat([self.bead_born, torch.full((int(new.sum()),), self.step, dtype=torch.long, device=self.dev)])
            self.last_bead[new] = seam_point[new]
        # 불꽃
        self.sp_left -= dt
        self.sp_vel[..., 2] -= 9.81 * dt
        self.sp_pos += self.sp_vel * dt
        spawn = welding[:, None] & (self.sp_left <= 0.0) & (torch.rand(self.n, self.k, device=self.dev) < 0.5)
        if bool(spawn.any()):
            rnd = torch.randn(self.n, self.k, 3, device=self.dev)
            rnd = rnd / rnd.norm(dim=-1, keepdim=True)
            out = rnd + 1.2 * normal[:, None, :]  # 바깥 쪽으로 치우친 방향
            out = out / out.norm(dim=-1, keepdim=True)
            speed = 0.4 + 1.2 * torch.rand(self.n, self.k, 1, device=self.dev)
            self.sp_pos[spawn] = arc[:, None, :].expand(-1, self.k, -1)[spawn]
            self.sp_vel[spawn] = (out * speed)[spawn]
            self.sp_left[spawn] = (0.10 + 0.25 * torch.rand(self.n, self.k, device=self.dev))[spawn]
        alive = self.sp_left > 0.0
        # 점광원 (깜빡임)
        for i in range(self.n):
            on = bool(welding[i])
            if on:
                intensity, trans, vec = self.lights[i]
                p = (arc[i] + 0.01 * normal[i]).tolist()
                trans.Set(vec(*p))
                intensity.Set(self.light_intensity * (0.35 + 0.65 * float(torch.rand(1))))
            elif self.light_on[i]:
                self.lights[i][0].Set(0.0)
            self.light_on[i] = on
        # 그리기
        hot = (self.step - self.bead_born) < self.HOT_STEPS
        n_arc = int(welding.sum())
        pos = [self.bead_pos, self.sp_pos[alive], arc[welding]]
        idx = [torch.where(hot, 1, 0), torch.full((int(alive.sum()),), 2, device=self.dev), torch.full((n_arc,), 3, device=self.dev)]
        scale = [torch.ones(len(self.bead_pos), 3, device=self.dev), torch.ones(int(alive.sum()), 3, device=self.dev),
                 (0.6 + 0.8 * torch.rand(n_arc, 1, device=self.dev)).expand(-1, 3)]
        pos, idx, scale = torch.cat(pos), torch.cat(idx), torch.cat(scale)
        if len(pos) == 0:
            pos, idx, scale = self.hidden, torch.zeros(1, dtype=torch.long, device=self.dev), torch.ones(1, 3, device=self.dev)
        self.markers.visualize(translations=pos, scales=scale, marker_indices=idx)


# ---------------------------------------------------------------------------
# 용접 상태 머신
# ---------------------------------------------------------------------------
# 걸을 때·기다릴 때의 오른팔: 토치를 몸 앞에 세워 든 자세 (팔을 앞으로 들고 아래팔을 위로. 걷기 정책의 팔 무작위 범위 안)
READY_Q = (-1.45, -0.25, 0.0, 0.05, 0.0)
WELD_SPEED = 0.02  # 용접선을 따라가는 속도 (m/s)
WELD_FAR = 0.18  # 다가가고 빠질 때 거치는 점: 용접선 가운데에서 바깥 방향으로 이만큼 (두 면에서 약 13cm. 보의 플랜지 끝에서 5.6cm 밖)
SEED_FRAC = 0.70  # 팔 뻗기: 준비 자세에서 미리 구해 둔 자세 쪽으로 이만큼만 관절 공간에서 가고, 나머지는 IK 로 거치는 점까지 간다
#   관절 공간에서는 서는 자리가 어긋난 만큼 토치 끝도 어긋나므로, 철골 가까이까지는 가지 않게 한다
COVER_DIST = 0.015  # 토치 끝이 이 거리 안으로 지나간 용접선 위의 점을 '덮었다'고 센다
N_COVER = 200
BACK_DIST = 0.40

(W_IDLE, W_READY, W_REBACK, W_WALK, W_SEED, W_FAR_IN, W_OFF_IN, W_IN, W_TO_START, W_DWELL, W_WELD, W_TO_MID, W_OUT, W_FAR_OUT, W_RETRACT,
 W_DONE, W_BACK, W_HOME, W_FAIL) = range(19)
REBACK_DIST = 0.30  # 한 용접선을 마치고 다른 용접선으로 갈 때: 준비 자세로 이만큼 물러난 뒤에 옮겨 간다 (철골 앞에서 바로 돌지 않게)
W_NAMES = ["대기", "준비 자세", "물러나기(다음 용접선으로)", "철골 앞으로 걷기", "팔 뻗기", "다가가기", "띄운 점으로", "용접선 가까이", "시작점으로", "시작", "용접", "가운데로",
           "빼기", "물러나기(팔)", "토치 거두기", "용접 끝", "뒤로 물러나기", "팔 내리기", "실패"]
# 토치는 용접선의 가운데 높이에서 들어가고 나온다: 띄운 점(가운데, 5cm) → 가까이(가운데, 1cm) → 용접선을 따라 시작점으로 → 용접 → 가운데로 → 빼기.
# 세로선의 양 끝은 위·아래 플랜지에서 1.5cm 뿐이라, 끝에서 바로 들어가고 나오면 몸이 조금만 흔들려도 노즐이 플랜지에 닿는다
W_STEPS = {W_READY: 60, W_SEED: 75, W_FAR_IN: 75, W_OFF_IN: 100, W_IN: 60, W_TO_START: 100, W_DWELL: 15, W_TO_MID: 100, W_OUT: 60,
           W_FAR_OUT: 100, W_RETRACT: 75, W_HOME: 75}
# 걷다 멈출 때 목표에서 한쪽으로 치우치는 양 (설 방향 기준 앞 m, 왼쪽 m, 방향 rad). 걸어갈 목표를 이만큼 반대로 옮겨 준다.
# 준비 자세(오른팔을 든 자세)로 다가가 멈춘 96번의 평균: 앞으로 3.3cm 더 가고, 오른쪽으로 1.8cm, 오른쪽으로 4.5° 더 돎 (2026-11-27. 걷기 정책 v2 기준)
STOP_BIAS = (0.033, -0.018, -0.078)
STANCE_TOL = (-0.04, 0.03, 0.035, 0.07)
STANCE_HARD = (0.05, 0.05, 0.105)  # 멈춘 자리가 설 자리에서 이만큼(앞뒤 m, 좌우 m, 방향 rad) 넘게 어긋나 있으면 한 번 더 걸어가 맞춘다
MAX_REWALK = 2  # 용접할 때의 멈춘 자리 허용 범위 (앞 오차 하한·상한 m, 좌우 m, 방향 rad). 밖이면 다시 다가감
TRACK_LIMIT = 0.03  # 용접선 가까이에서 토치 끝이 목표점에서 이만큼 넘게 벗어나면 그만둔다


def weld_axis(normal: torch.Tensor, along: torch.Tensor, frac: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    """토치가 향할 방향 (N, 3): 바깥 방향의 반대에서, 용접선 방향(along)으로 tilt * (2 * frac - 1) 만큼 기울인 것.

    frac = 용접 구간에서 지나온 비율 (0 = 시작, 1 = 끝). 시작에서는 시작 끝 쪽을, 끝에서는 끝 쪽을 향한다.
    """
    ang = tilt * (2.0 * frac - 1.0)
    axis = -normal * torch.cos(ang)[:, None] + along * torch.sin(ang)[:, None]
    return axis / axis.norm(dim=1, keepdim=True)


def _smooth(a: torch.Tensor) -> torch.Tensor:
    a = torch.clamp(a, 0.0, 1.0)
    return 10 * a**3 - 15 * a**4 + 6 * a**5


class WeldMachine:
    """용접 한 번: 준비 자세 → 서는 자리로 걷기 → 다가가기 → 띄운 점 → 용접선 따라 이동 → 빼기 → 토치 거두기.

    걷기와 팔 관절 목표값을 넣는 일은 PickPlaceMachine(pm)이 맡는다: 이 클래스는 매 스텝 pm.ik.q_cmd(팔 10관절 목표값)를 고쳐 쓰고,
    걷기는 pm.start_walk 를 부른다. pre_step 은 pm.pre_step 보다 먼저, post_step 은 pm.post_step 이 돌려준 속도 명령을 받아 고쳐서 돌려준다.
    stow() = 용접을 다 마친 뒤: 뒤로 물러나 → 팔을 기본 자세로 → 토치를 숨김.
    """

    def __init__(self, env, pm, effects: WeldEffects | None = None, speed: float = WELD_SPEED, hide_torch: bool = True):
        self.env, self.pm, self.fx, self.speed = env, pm, effects, speed
        self.robot = pm.robot
        self.n, self.dev, self.dt = env.num_envs, env.device, env.step_dt
        n, dev = self.n, self.dev
        self.ik = TorchIK(self.robot)
        origins = env.scene.env_origins
        self.frame_pos = origins + torch.tensor(sl.STEEL_FRAME_POS, device=dev)
        self.clear = FrameClearance(self.robot, self.frame_pos, torch.full((n,), math.radians(sl.STEEL_FRAME_YAW_DEG), device=dev))
        info = sl.weld_seam_frames()
        self.names = list(info)
        self.kind = [info[k]["kind"] for k in self.names]
        self._start = torch.tensor([info[k]["start"] for k in self.names], device=dev)  # 환경 원점 기준
        self._end = torch.tensor([info[k]["end"] for k in self.names], device=dev)
        self._normal = torch.tensor([info[k]["normal"] for k in self.names], device=dev)
        self.grip_ids, self.grip_q = grip_targets(self.robot)
        self.open_q = self.robot.data.default_joint_pos[:, self.grip_ids].clone()
        self.ready_q = torch.tensor(LEFT_SIDE_POSE + READY_Q, device=dev)
        self.default_q = self.robot.data.default_joint_pos[:, pm.ik.all_joint_ids].clone()
        self.sensor = env.scene.sensors.get("contact_forces")
        self.sensor_ids = [i for i, b in enumerate(self.sensor.body_names) if b.startswith("right_") and any(
            k in b for k in ("palm", "zero", "one", "two", "three", "four", "five", "six", "elbow"))] if self.sensor is not None else []
        z = lambda *shape, dtype=torch.float: torch.zeros(*shape, dtype=dtype, device=dev)  # noqa: E731
        self.phase = torch.full((n,), W_IDLE, dtype=torch.long, device=dev)
        self.t = z(n, dtype=torch.long)
        self.seam = z(n, dtype=torch.long)
        self.p_start, self.p_end, self.normal = z(n, 3), z(n, 3), z(n, 3)  # 용접선 (월드)과 바깥 방향
        self.normal[:, 0] = 1.0
        self.w_start, self.w_end, self.p_far = z(n, 3), z(n, 3), z(n, 3)  # 실제로 따라가는 구간(끝을 줄인 것)과 거치는 점
        self.seg_from, self.seg_to = z(n, 3), z(n, 3)
        self.target = z(n, 3)
        self.stance = z(n, 3)
        self.along, self.tilt = z(n, 3), z(n)  # 용접선 방향(시작 → 끝)과 양 끝에서의 기울임
        self.along[:, 2] = 1.0
        self.seed = torch.tensor([READY_Q], device=dev).repeat(n, 1)
        self.q_tab = z(n, 4, 5)  # 기준 자세 (거치는 점, 가운데, 시작, 끝)
        self.weld_steps = torch.ones(n, dtype=torch.long, device=dev)
        self.j_from, self.j_to = z(n, 10), z(n, 10)
        self.torch_on = z(n, dtype=torch.bool)
        self.back_start, self.back_yaw = z(n, 2), z(n)
        self._tol_keep = pm.stance_tol.clone()
        self._tol_default = pm.stance_tol.clone()
        self.rewalk = z(n, dtype=torch.long)
        self.fail_reason = [None] * n
        self.fail_phase = [None] * n
        self.fail_q = [None] * n
        self.at_limit = z(n, dtype=torch.bool)  # IK 로 움직이는 동안 관절 목표값이 한계에 닿은 적이 있는지
        self.cover = z(n, N_COVER, dtype=torch.bool)
        self.s_hist = z(n, 1200)
        keys = ("err_sum", "err_max", "cross_sum", "cross_max", "gap_sum", "gap_dev_max", "ang_sum", "ang_max", "dang_sum", "dang_max", "k",
                "torch_clear", "hand_clear", "stance_x", "stance_y", "stance_yaw", "t_begin", "t_weld0", "t_weld1", "t_end", "tilt_max",
                "retries", "t_arrive", "pelvis_z", "pelvis_pitch")
        self.rec = {k: z(n) for k in keys}
        self.hide_torch = hide_torch
        if hide_torch:
            set_torch_visible(range(n), False)

    # ------------------------------------------------------------------ 시작·끝
    def begin(self, mask: torch.Tensor, seam_names: list, stance: torch.Tensor, postures: dict | None = None):
        """mask 환경의 용접을 시작한다. seam_names[e] = 용접선 이름, stance (N, 3) = 설 자리 (월드 x, y, 방향 rad).

        postures: {용접선: {"far", "mid", "start", "end"} 기준 자세(오른팔 5관절)}. 기본 = site_layout.WELD_POSTURE.
        """
        postures = sl.WELD_POSTURE if postures is None else postures
        ids = mask.nonzero().flatten().tolist()
        if not ids:
            return
        origins = self.env.scene.env_origins
        for e in ids:
            k = self.names.index(seam_names[e])
            self.seam[e] = k
            self.p_start[e], self.p_end[e], self.normal[e] = origins[e] + self._start[k], origins[e] + self._end[k], self._normal[k]
            d = self.p_end[e] - self.p_start[e]
            length = torch.norm(d)
            trim = WELD_END_TRIM_V if self.kind[k] == "v" else 0.0
            self.w_start[e] = self.p_start[e] + trim * d / length + WELD_GAP * self.normal[e]
            self.w_end[e] = self.p_end[e] - trim * d / length + WELD_GAP * self.normal[e]
            self.p_far[e] = 0.5 * (self.p_start[e] + self.p_end[e]) + WELD_FAR * self.normal[e]
            self.weld_steps[e] = max(1, int(round(float(length - 2 * trim) / self.speed / self.dt)))
            self.along[e], self.tilt[e] = d / length, WELD_TILT_V if self.kind[k] == "v" else 0.0
            self.fail_reason[e], self.fail_phase[e], self.fail_q[e] = None, None, None
            self.at_limit[e] = False
            post = postures[seam_names[e]]
            self.q_tab[e] = torch.tensor([post[key] for key in ("far", "mid", "start", "end")], device=self.dev)
            self.seed[e] = self.q_tab[e, 0]
        for v in self.rec.values():
            v[mask] = 0.0
        self.rec["torch_clear"][mask], self.rec["hand_clear"][mask] = 1.0e3, 1.0e3
        self.rec["t_begin"][mask] = float(self.pm.step)
        self.cover[mask] = False
        self.s_hist[mask] = 0.0
        self.stance[mask] = stance[mask]
        again = mask & self.torch_on & (self.phase == W_DONE)  # 방금 다른 용접선을 마친 환경: 준비 자세 그대로 조금 물러난 뒤 옮겨 간다
        show = [e for e in ids if not bool(self.torch_on[e])]
        if show:
            set_torch_visible(show, True)
            self.torch_on[show] = True
        self._joint_move(mask, self.ready_q.expand(self.n, -1))
        self._set(mask & ~again, W_READY)
        self.back_start[again], self.back_yaw[again] = self.robot.data.root_pos_w[again, :2], self.robot.data.heading_w[again]
        self._set(again, W_REBACK)

    def stow(self, mask: torch.Tensor):
        """용접을 마친 mask 환경: 뒤로 물러난 뒤 팔을 기본 자세로 내리고 토치를 숨긴다."""
        if not bool(mask.any()):
            return
        self.back_start[mask], self.back_yaw[mask] = self.robot.data.root_pos_w[mask, :2], self.robot.data.heading_w[mask]
        self._set(mask, W_BACK)

    def reset(self):
        self.phase[:] = W_IDLE
        self.t[:] = 0
        self.pm.stance_tol[:] = self._tol_default
        if self.hide_torch:
            set_torch_visible(range(self.n), False)
            self.torch_on[:] = False
        self.fail_reason, self.fail_phase = [None] * self.n, [None] * self.n
        if self.fx is not None:
            self.fx.clear()

    @property
    def active(self) -> torch.Tensor:
        return (self.phase != W_IDLE) & (self.phase != W_DONE) & (self.phase != W_FAIL)

    # ------------------------------------------------------------------ 도우미
    def _set(self, mask: torch.Tensor, phase: int):
        self.phase[mask] = phase
        self.t[mask] = 0

    def _joint_move(self, mask: torch.Tensor, q_to: torch.Tensor):
        self.j_from[mask] = self.pm.ik.q_cmd[mask]
        self.j_to[mask] = q_to[mask]

    def _segment(self, mask: torch.Tensor, p_to: torch.Tensor, from_tip: bool = False):
        self.seg_from[mask] = (self.ik.tip_pos_w() if from_tip else self.target)[mask]
        self.seg_to[mask] = p_to[mask]

    def _fail(self, mask: torch.Tensor, reason: str):
        for e in mask.nonzero().flatten().tolist():
            self.fail_reason[e] = reason
            self.fail_phase[e] = W_NAMES[int(self.phase[e])]
            self.fail_q[e] = [round(v, 2) for v in self.pm.ik.q_cmd[e, 5:].tolist()]  # 그 순간의 오른팔 관절 목표값
        self._set(mask, W_FAIL)

    def axis_des(self) -> torch.Tensor:
        """지금 단계에서 토치가 향할 방향 (N, 3)."""
        frac = torch.where(self.phase == W_WELD, torch.clamp(self.t.float() / self.weld_steps.float(), 0.0, 1.0),
                           (self.phase > W_WELD).float())
        return weld_axis(self.normal, self.along, frac, self.tilt)

    def q_nominal(self) -> torch.Tensor:
        """지금 목표점에서의 기준 자세 (N, 5): 용접선을 따라서는 시작·가운데·끝 자세 사이, 바깥 방향으로는 가운데 ↔ 거치는 점 자세 사이를 잇는다."""
        mid = 0.5 * (self.w_start + self.w_end)
        half = (0.5 * torch.norm(self.w_end - self.w_start, dim=1)).clamp(min=1.0e-6)
        rel = self.target - mid
        s = torch.clamp((rel * self.along).sum(dim=1) / half, -1.0, 1.0)[:, None]  # -1 = 시작, +1 = 끝
        out = torch.clamp((rel * self.normal).sum(dim=1) / (WELD_FAR - WELD_GAP), 0.0, 1.0)[:, None]  # 0 = 용접선 가까이, 1 = 거치는 점
        far, q_mid, q_start, q_end = self.q_tab.unbind(dim=1)
        on_seam = torch.where(s < 0, q_mid + (-s) * (q_start - q_mid), q_mid + s * (q_end - q_mid))
        return on_seam + out * (far - q_mid)

    def _start_walk(self, m: torch.Tensor):
        """mask 환경을 설 자리로 걸어가게 한다. 걸어갈 목표 = 설 자리에서 '멈출 때 치우치는 양'을 뺀 곳."""
        from .pick_place_machine import STANCE_DIST

        st = self.stance.clone()
        c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        st[:, 0] -= c * STOP_BIAS[0] - sn * STOP_BIAS[1]
        st[:, 1] -= sn * STOP_BIAS[0] + c * STOP_BIAS[1]
        st[:, 2] -= STOP_BIAS[2]
        dirs = torch.stack([torch.cos(st[:, 2]), torch.sin(st[:, 2])], dim=1)
        self.pm.start_walk(st[:, :2] + STANCE_DIST * dirs, st[:, 2], obstacles=torch.zeros(0, self.n, 2, device=self.dev), keep_box=True, mask=m)
        self.pm.stance_tol[m] = torch.tensor(STANCE_TOL, device=self.dev)
        self._set(m, W_WALK)

    def seam_point(self):
        """토치 끝에서 가장 가까운 용접선 위의 점 (N, 3) 과 용접선을 따라 잰 위치 (N,)."""
        d = self.p_end - self.p_start
        length = torch.norm(d, dim=1).clamp(min=1.0e-6)
        u = d / length[:, None]
        s = torch.clamp(((self.ik.tip_pos_w() - self.p_start) * u).sum(dim=1), torch.zeros_like(length), length)
        return self.p_start + s[:, None] * u, s

    # ------------------------------------------------------------------ 매 스텝
    def pre_step(self):
        """pm.pre_step (또는 팔 목표값을 넣는 다른 쪽) 보다 먼저 부른다: 용접 중인 환경의 팔 관절 목표값을 pm.ik.q_cmd 에 쓴다."""
        ph, q = self.phase, self.pm.ik.q_cmd
        steps = torch.tensor([W_STEPS.get(int(p), 1) for p in ph], device=self.dev)
        joint = (ph == W_READY) | (ph == W_SEED) | (ph == W_RETRACT) | (ph == W_HOME)
        if bool(joint.any()):
            a = _smooth((self.t + 1).float() / steps.float())[:, None]
            q[joint] = (self.j_from + a * (self.j_to - self.j_from))[joint]
        cart = (ph >= W_FAR_IN) & (ph <= W_FAR_OUT)
        if bool(cart.any()):
            steps = torch.where(ph == W_WELD, self.weld_steps, steps)
            a = (self.t + 1).float() / steps.float()
            a = torch.where(ph == W_WELD, torch.clamp(a, 0.0, 1.0), _smooth(a))[:, None]  # 용접 구간은 일정한 속도
            prev = self.target.clone()
            self.target[cart] = (self.seg_from + a * (self.seg_to - self.seg_from))[cart]
            q_new = self.ik.solve(q[:, 5:], self.target, self.axis_des(), feed=self.target - prev, q_nom=self.q_nominal())
            q[cart, 5:] = q_new[cart]
            self.at_limit |= cart & ((q_new <= self.ik.q_min + 1.0e-4) | (q_new >= self.ik.q_max - 1.0e-4)).any(dim=1)
        on = self.torch_on
        if bool(on.any()):
            ids = on.nonzero().flatten()
            self.robot.set_joint_position_target(self.grip_q.expand(len(ids), -1), joint_ids=self.grip_ids, env_ids=ids)

    def post_step(self, cmd: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """env.step 뒤에: 넘어짐·닿음·단계 전환을 처리하고 속도 명령을 돌려준다."""
        from .pick_place_machine import S_DONE, S_FAIL, STANCE_DIST

        ph = self.phase.clone()
        self.t += 1
        fell = terminated & (self.active | (ph == W_BACK) | (ph == W_HOME))
        if bool(fell.any()):
            self._fail(fell, "넘어짐")
            self.pm.ik.reset(fell.nonzero().flatten())
        # 철골에 닿았는지: 토치·손과 철골 판 사이의 거리, 오른팔·손의 접촉 힘
        near = (ph >= W_READY) & (ph <= W_RETRACT) & ~fell
        if bool(near.any()):
            tc, hc = self.clear.torch_clearance(), self.clear.hand_clearance()
            self.rec["torch_clear"][near] = torch.minimum(self.rec["torch_clear"], tc)[near]
            self.rec["hand_clear"][near] = torch.minimum(self.rec["hand_clear"], hc)[near]
            touch = near & ((tc <= 0.0) | (hc <= 0.0))
            self._fail(touch & (tc <= 0.0), "토치가 철골에 닿음")
            self._fail(touch & (tc > 0.0), "손이 철골에 닿음")
            if self.sensor is not None and self.sensor_ids:
                force = self.sensor.data.net_forces_w[:, self.sensor_ids].norm(dim=-1).max(dim=1).values
                self._fail(near & ~touch & (force > 1.0), "오른팔이 무엇에 부딪힘")
            self.rec["tilt_max"][near] = torch.maximum(self.rec["tilt_max"], self.pm.tilt())[near]
            close = (ph >= W_IN) & (ph <= W_OUT) & (self.phase != W_FAIL)
            self._fail(close & (torch.norm(self.ik.tip_pos_w() - self.target, dim=1) > TRACK_LIMIT), "토치 끝이 경로에서 3cm 넘게 벗어남")
        ph = torch.where(self.phase == W_FAIL, self.phase, ph)
        done = lambda code: (ph == code) & (self.t >= W_STEPS[code])  # noqa: E731

        # (이어서 용접할 때) 물러나기 → 걷기
        m = ph == W_REBACK
        if bool(m.any()):
            d = self.robot.data.root_pos_w[:, :2] - self.back_start
            moved = -(d[:, 0] * torch.cos(self.back_yaw) + d[:, 1] * torch.sin(self.back_yaw))
            dyaw = self.back_yaw - self.robot.data.heading_w
            cmd[m, 0], cmd[m, 1] = -0.3, 0.0
            cmd[m, 2] = torch.clamp(1.5 * torch.atan2(torch.sin(dyaw), torch.cos(dyaw)), -0.4, 0.4)[m]
            go = m & ((moved >= REBACK_DIST) | (self.t > 300))
            if bool(go.any()):
                cmd[go] = 0.0
                self._tol_keep[go] = self.pm.stance_tol[go]
                self.rewalk[go] = 0
                self._start_walk(go)
        # 준비 자세 → 걷기
        m = done(W_READY)
        if bool(m.any()):
            self._tol_keep[m] = self.pm.stance_tol[m]
            self.rewalk[m] = 0
            self._start_walk(m)
        # 걷기 → 다가가기
        m = ph == W_WALK
        if bool(m.any()):
            arrived = m & (self.pm.state == S_DONE) & (self.t > 1)
            if bool(arrived.any()):  # 걷기 쪽이 다시 다가가기를 다 쓰고도 많이 어긋난 채 멈춘 경우: 처음부터 다시 걸어가게 한다
                st = self.stance
                d = self.robot.data.root_pos_w[:, :2] - st[:, :2]
                c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
                dyaw = self.robot.data.heading_w - st[:, 2]
                off = ((c * d[:, 0] + sn * d[:, 1]).abs() > STANCE_HARD[0]) | ((-sn * d[:, 0] + c * d[:, 1]).abs() > STANCE_HARD[1]) \
                    | (torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs() > STANCE_HARD[2])
                again = arrived & off & (self.rewalk < MAX_REWALK)
                if bool(again.any()):
                    self.rewalk[again] += 1
                    self._start_walk(again)
                    arrived = arrived & ~again
            ended = m & ((self.pm.state == S_DONE) | (self.pm.state == S_FAIL)) & (self.t > 1) & (self.phase == W_WALK) & (self.t > 1)
            self.pm.stance_tol[ended] = self._tol_keep[ended]
            self._fail(m & (self.pm.state == S_FAIL) & (self.t > 1), "걷기 실패")
            if bool(arrived.any()):
                st = self.stance
                d = self.robot.data.root_pos_w[:, :2] - st[:, :2]
                c, s = torch.cos(st[:, 2]), torch.sin(st[:, 2])
                dyaw = self.robot.data.heading_w - st[:, 2]
                self.rec["stance_x"][arrived] = (c * d[:, 0] + s * d[:, 1])[arrived]
                self.rec["stance_y"][arrived] = (-s * d[:, 0] + c * d[:, 1])[arrived]
                self.rec["stance_yaw"][arrived] = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))[arrived]
                self.rec["retries"][arrived] = (self.pm.reapproach_b + 10 * self.rewalk).float()[arrived]
                self.rec["t_arrive"][arrived] = float(self.pm.step)
                grav = self.robot.data.projected_gravity_b
                self.rec["pelvis_z"][arrived] = (self.robot.data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2])[arrived]
                self.rec["pelvis_pitch"][arrived] = (torch.atan2(grav[:, 0], -grav[:, 2]) * RAD2DEG)[arrived]
                # 팔 뻗기: 그 용접선에 맞게 미리 구해 둔 자세(토치 끝이 거치는 점 근처)로 관절 공간에서 옮긴다.
                # 준비 자세에서 바로 IK 로 가면 다른 팔 꼴로 풀려 토치 방향을 맞추지 못한다
                q_to = self.pm.ik.q_cmd.clone()
                q_to[:, 5:] = q_to[:, 5:] + SEED_FRAC * (self.seed - q_to[:, 5:])
                self._joint_move(arrived, q_to)
                self._set(arrived, W_SEED)
        m = done(W_SEED)
        if bool(m.any()):
            self._segment(m, self.p_far, from_tip=True)
            self.target[m] = self.ik.tip_pos_w()[m]
            self._set(m, W_FAR_IN)
        # 다가가기 → 띄운 점(가운데) → 용접선 가까이(가운데) → 시작점으로 → 시작 → 용접 → 가운데로 → 빼기 → 물러나기 → 토치 거두기
        mid = 0.5 * (self.w_start + self.w_end)
        mid_off = mid + (WELD_OFFSET - WELD_GAP) * self.normal
        for code, nxt, goal in ((W_FAR_IN, W_OFF_IN, mid_off), (W_OFF_IN, W_IN, mid), (W_IN, W_TO_START, self.w_start),
                                (W_TO_START, W_DWELL, self.w_start), (W_DWELL, W_WELD, self.w_end), (W_TO_MID, W_OUT, mid_off),
                                (W_OUT, W_FAR_OUT, self.p_far)):
            m = done(code)
            if bool(m.any()):
                self._segment(m, goal)
                if nxt == W_WELD:
                    self.rec["t_weld0"][m] = float(self.pm.step)
                self._set(m, nxt)
        m = (ph == W_WELD) & (self.t >= self.weld_steps)
        if bool(m.any()):
            self.rec["t_weld1"][m] = float(self.pm.step)
            self._segment(m, mid)
            self._set(m, W_TO_MID)
        m = done(W_FAR_OUT)
        if bool(m.any()):
            self._joint_move(m, self.ready_q.expand(self.n, -1))
            self._set(m, W_RETRACT)
        m = done(W_RETRACT)
        if bool(m.any()):
            self.rec["t_end"][m] = float(self.pm.step)
            self._set(m, W_DONE)
        # 용접 중의 기록
        w = ph == W_WELD
        seam_pt, s = self.seam_point()
        if bool(w.any()):
            tip = self.ik.tip_pos_w()
            err = torch.norm(tip - self.target, dim=1)
            u = self.w_end - self.w_start
            u = u / torch.norm(u, dim=1, keepdim=True).clamp(min=1.0e-6)
            rel = tip - self.w_start
            cross = torch.norm(rel - (rel * u).sum(dim=1, keepdim=True) * u, dim=1)
            gap = torch.norm(tip - seam_pt, dim=1)
            ang = torch.acos(torch.clamp((self.ik.axis_w() * -self.normal).sum(dim=1), -1.0, 1.0)) * RAD2DEG
            frac = torch.clamp(self.rec["k"] / self.weld_steps.float(), 0.0, 1.0)
            want = weld_axis(self.normal, self.along, frac, self.tilt)
            dang = torch.acos(torch.clamp((self.ik.axis_w() * want).sum(dim=1), -1.0, 1.0)) * RAD2DEG
            r = self.rec
            idx = torch.clamp(r["k"].long(), 0, self.s_hist.shape[1] - 1)
            self.s_hist[w, idx[w]] = s[w]
            for key, val in (("err_sum", err), ("cross_sum", cross), ("gap_sum", gap), ("ang_sum", ang), ("dang_sum", dang)):
                r[key][w] += val[w]
            for key, val in (("err_max", err), ("cross_max", cross), ("gap_dev_max", (gap - WELD_GAP).abs()), ("ang_max", ang), ("dang_max", dang)):
                r[key][w] = torch.maximum(r[key], val)[w]
            r["k"][w] += 1
            frac = torch.linspace(0.0, 1.0, N_COVER, device=self.dev)[None, :, None]
            pts = self.p_start[:, None, :] + frac * (self.p_end - self.p_start)[:, None, :]
            self.cover[w] |= (torch.norm(pts - tip[:, None, :], dim=-1) < COVER_DIST)[w]
        if self.fx is not None:
            self.fx.update(w, seam_pt, self.normal, self.dt)
        # 뒤로 물러나기 → 팔 내리기 → 토치 숨김
        m = self.phase == W_BACK
        if bool(m.any()):
            d = self.robot.data.root_pos_w[:, :2] - self.back_start
            moved = -(d[:, 0] * torch.cos(self.back_yaw) + d[:, 1] * torch.sin(self.back_yaw))
            dyaw = self.back_yaw - self.robot.data.heading_w
            cmd[m, 0], cmd[m, 1] = -0.3, 0.0
            cmd[m, 2] = torch.clamp(1.5 * torch.atan2(torch.sin(dyaw), torch.cos(dyaw)), -0.4, 0.4)[m]
            far = m & ((moved >= BACK_DIST) | (self.t > 400))
            if bool(far.any()):
                cmd[far] = 0.0
                self._joint_move(far, self.default_q)
                self._set(far, W_HOME)
        m = (self.phase == W_HOME) & (self.t >= W_STEPS[W_HOME])
        if bool(m.any()):
            ids = m.nonzero().flatten()
            if self.hide_torch:
                set_torch_visible(ids.tolist(), False)
                self.torch_on[m] = False
                self.robot.set_joint_position_target(self.open_q[ids], joint_ids=self.grip_ids, env_ids=ids)
            self._set(m, W_IDLE)
        return cmd

    # ------------------------------------------------------------------ 결과
    def result(self, e: int) -> dict:
        """환경 e 의 마지막 용접 기록 (길이 mm, 각도 도, 시간 초)."""
        r, k = self.rec, max(float(self.rec["k"][e]), 1.0)
        n_w = int(self.rec["k"][e])
        s = self.s_hist[e, :n_w]
        speed = (s[5:] - s[:-5]) / (5 * self.dt) if n_w > 10 else torch.zeros(2, device=self.dev)
        g = lambda key: float(r[key][e])  # noqa: E731
        return {
            "seam": self.names[int(self.seam[e])], "welded": self.fail_reason[e] is None and g("t_end") > 0,
            "fail": self.fail_reason[e], "fail_phase": self.fail_phase[e], "fail_q": self.fail_q[e], "at_limit": bool(self.at_limit[e]),
            "err_mean_mm": g("err_sum") / k * 1000, "err_max_mm": g("err_max") * 1000,
            "cross_mean_mm": g("cross_sum") / k * 1000, "cross_max_mm": g("cross_max") * 1000,
            "gap_mean_mm": g("gap_sum") / k * 1000, "gap_dev_max_mm": g("gap_dev_max") * 1000,
            "angle_mean_deg": g("ang_sum") / k, "angle_max_deg": g("ang_max"),
            "angle_err_mean_deg": g("dang_sum") / k, "angle_err_max_deg": g("dang_max"),
            "speed_mean_mm_s": float(speed.mean()) * 1000, "speed_std_mm_s": float(speed.std()) * 1000,
            "speed_min_mm_s": float(speed.min()) * 1000, "speed_max_mm_s": float(speed.max()) * 1000,
            "coverage": float(self.cover[e].float().mean()),
            "torch_clear_mm": g("torch_clear") * 1000, "hand_clear_mm": g("hand_clear") * 1000,
            "stance_x_cm": g("stance_x") * 100, "stance_y_cm": g("stance_y") * 100, "stance_yaw_deg": g("stance_yaw") * RAD2DEG,
            "tilt_max_deg": g("tilt_max"), "retries": int(g("retries")), "t_walk_s": (g("t_arrive") - g("t_begin")) * self.dt,
            "pelvis_z_cm": g("pelvis_z") * 100, "pelvis_pitch_deg": g("pelvis_pitch"),
            "t_weld_s": (g("t_weld1") - g("t_weld0")) * self.dt, "t_total_s": (g("t_end") - g("t_begin")) * self.dt,
        }
