"""양팔 IK: 양손(palm 링크)을 월드 좌표의 목표 위치로 보내는 관절 목표값을 계산한다.

Isaac Lab 의 DifferentialIKController(감쇠 최소제곱, "dls")를 팔마다 하나씩 쓴다.
- 자코비안: `robot.root_physx_view.get_jacobians()` → (환경, 링크, 6, 자유도), 월드 좌표 기준.
  원본 DifferentialInverseKinematicsAction 과 같은 규칙으로 고른다:
  몸이 고정이면 링크 번호 −1·관절 번호 그대로, 떠 있는 몸이면 링크 번호 그대로·관절 번호 +6 (앞 6칸이 몸통 자유도).
- 팔 관절만 고르기: 관절 이름으로 번호를 찾아 자코비안의 그 열만 쓴다 (한쪽 5개: 어깨 pitch·roll·yaw, 팔꿈치 pitch·roll).
- 목표값은 '측정한 관절 각도 + 보정'이 아니라 '직전 목표값 + 보정'으로 쌓는다.
  PD 제어라 손에 무게가 있으면 목표값보다 처지는데, 측정값 기준이면 그 처짐만큼 오차가 남기 때문이다.

이 모듈은 isaaclab 을 import 하므로 Isaac Sim 앱이 켜진 뒤에만 import 한다.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll")
SIDES = ("left", "right")  # [왼쪽, 오른쪽] 순서
NUM_ARM_JOINTS = len(ARM_JOINTS)


# 걷기 정책이 학습 때 움직이는 것을 본 적이 없는 팔 관절 (항상 짐 든 자세 값이었음).
# 팔 무작위판은 어깨 pitch·팔꿈치 pitch 만 바꿨다.
UNTRAINED_ARM_JOINTS = ("shoulder_roll", "shoulder_yaw", "elbow_roll")


class PolicyObsFilter:
    """걷기 정책에 넣는 관측에서, 학습 때 고정이던 팔 관절의 각도·속도를 학습 때 값(기본 자세, 속도 0)으로 바꾼다.

    이유 (2026-11-22 ik_check.py --mode stand): IK 가 어깨 roll 을 0.2 → 0.35~0.45 rad 로 벌리자 정책이
    몸을 앞으로 기울이며 발을 떼고 넘어짐 (32개 중 25개). 이 관절들은 학습 내내 같은 값이라
    정책이 그 입력에 어떻게 반응할지 정해진 적이 없다. 실제 팔 자세가 바뀌어 생기는 무게중심 변화는 그대로 남는다.
    """

    def __init__(self, env, robot: Articulation, group: str = "policy"):
        manager = env.observation_manager
        names = manager.active_terms[group]
        dims = [d[0] for d in manager.group_obs_term_dim[group]]
        starts = {}
        offset = 0
        for name, dim in zip(names, dims):
            starts[name] = offset
            offset += dim
        joints = [robot.joint_names.index(f"{s}_{j}_joint") for s in SIDES for j in UNTRAINED_ARM_JOINTS]
        self.group = group
        self.columns = [starts["joint_pos"] + j for j in joints] + [starts["joint_vel"] + j for j in joints]

    def __call__(self, obs):
        """관측(TensorDict 또는 텐서)을 복사해 해당 칸을 0(= 기본 자세와의 차이 0, 속도 0)으로 만든다."""
        if isinstance(obs, torch.Tensor):
            out = obs.clone()
            out[:, self.columns] = 0.0
            return out
        out = obs.clone()
        policy_obs = out[self.group].clone()
        policy_obs[:, self.columns] = 0.0
        out[self.group] = policy_obs
        return out


class TorsoFrame:
    """몸통(torso_link) 좌표와 월드 좌표 사이의 변환. 손 목표 위치를 몸통에 고정해 둘 때 쓴다.

    목표를 월드나 '수평을 유지하는 골반 좌표'에 고정하면, 몸이 기울 때마다 IK 가 팔을 반대로 움직인다.
    걷기 정책은 팔 관절 각도를 보고 자세를 바꾸므로 둘이 서로를 밀어, 손에 무게가 있으면 서 있다가
    천천히 옆으로 기울며 발을 떼기 시작한다 (2026-11-22, 손 하나당 2kg: 뻗은 뒤 1.5초 동안 좌우 기울기 0 → 7.6°).
    몸통에 고정하면 몸이 움직여도 팔 관절 목표가 바뀌지 않아 학습 때(관절 목표 고정)와 같은 조건이 된다.
    """

    def __init__(self, robot: Articulation):
        self.robot = robot
        self.body_id = robot.body_names.index("torso_link")

    def _pose(self, num_points: int):
        pos = self.robot.data.body_pos_w[:, self.body_id]
        quat = self.robot.data.body_quat_w[:, self.body_id]
        return pos[:, None, :], quat[:, None, :].expand(-1, num_points, -1).reshape(-1, 4)

    def to_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        """월드 (N, K, 3) → 몸통 좌표."""
        pos, quat = self._pose(pos_w.shape[1])
        return quat_apply_inverse(quat, (pos_w - pos).reshape(-1, 3)).reshape(pos_w.shape)

    def to_world(self, pos_t: torch.Tensor) -> torch.Tensor:
        """몸통 좌표 (N, K, 3) → 월드."""
        pos, quat = self._pose(pos_t.shape[1])
        return pos + quat_apply(quat, pos_t.reshape(-1, 3)).reshape(pos_t.shape)


class DualArmIK:
    """양손 위치(또는 자세) 목표를 팔 관절 목표값으로 바꾼다. 목표·손 위치는 모두 월드 좌표."""

    def __init__(
        self,
        robot: Articulation,
        command_type: str = "position",
        lambda_val: float = 0.05,
        gain: float = 0.3,
        max_joint_step: float = 0.04,
    ):
        """
        Args:
            command_type: "position"(위치만) 또는 "pose"(위치 + 방향).
            lambda_val: 감쇠 최소제곱의 감쇠 계수. 팔을 거의 다 편 자세(특이 자세) 근처에서 관절이 튀지 않게 한다.
            gain: 한 스텝에 IK 보정량의 몇 배를 목표값에 더할지 (1 이면 한 번에 다 더함).
            max_joint_step: 한 스텝에 관절 목표값이 바뀔 수 있는 최대 크기 (rad). 0.04 = 2 rad/s (스텝 0.02초).
        """
        self.robot = robot
        self.command_type = command_type
        self.gain = gain
        self.max_joint_step = max_joint_step
        n, device = robot.num_instances, robot.device

        self.joint_ids = [[robot.joint_names.index(f"{s}_{j}_joint") for j in ARM_JOINTS] for s in SIDES]
        self.all_joint_ids = self.joint_ids[0] + self.joint_ids[1]
        self.body_ids = [robot.body_names.index(f"{s}_palm_link") for s in SIDES]
        body_offset, joint_offset = (-1, 0) if robot.is_fixed_base else (0, 6)
        self._jac_body = [b + body_offset for b in self.body_ids]
        self._jac_joint = [[j + joint_offset for j in ids] for ids in self.joint_ids]

        cfg = DifferentialIKControllerCfg(
            command_type=command_type, use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": lambda_val}
        )
        self._ctrl = [DifferentialIKController(cfg, n, device) for _ in SIDES]
        limits = robot.data.soft_joint_pos_limits[:, self.all_joint_ids]
        self._q_min, self._q_max = limits[..., 0].clone(), limits[..., 1].clone()
        self.q_cmd = robot.data.default_joint_pos[:, self.all_joint_ids].clone()

    def reset(self, env_ids=None):
        """목표값을 기본 자세(짐 든 자세)로 되돌린다. 리셋 이벤트 hold_arms 가 넣는 값과 같다."""
        if env_ids is None:
            env_ids = slice(None)
        self.q_cmd[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self.all_joint_ids]

    def hand_pos_w(self) -> torch.Tensor:
        """양손 위치 (N, 2, 3)."""
        return self.robot.data.body_pos_w[:, self.body_ids]

    def hand_quat_w(self) -> torch.Tensor:
        """양손 방향 (N, 2, 4), (w, x, y, z)."""
        return self.robot.data.body_quat_w[:, self.body_ids]

    def step(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        """목표값을 한 스텝 갱신한다.

        Args:
            target_pos_w: 양손 목표 위치 (N, 2, 3).
            target_quat_w: 양손 목표 방향 (N, 2, 4). command_type 이 "pose" 일 때만 쓴다.
            mask: (N,) bool. False 인 환경은 목표값을 그대로 둔다 (걷는 동안 팔 자세를 고정할 때).
        """
        jacobians = self.robot.root_physx_view.get_jacobians()
        new_cmd = self.q_cmd.clone()
        for k in range(len(SIDES)):
            ee_pos = self.robot.data.body_pos_w[:, self.body_ids[k]]
            ee_quat = self.robot.data.body_quat_w[:, self.body_ids[k]]
            jac = jacobians[:, self._jac_body[k]][:, :, self._jac_joint[k]]  # (N, 6, 5)
            if self.command_type == "position":
                command = target_pos_w[:, k]
            else:
                command = torch.cat([target_pos_w[:, k], target_quat_w[:, k]], dim=1)
            self._ctrl[k].set_command(command, ee_pos, ee_quat)
            cols = slice(k * NUM_ARM_JOINTS, (k + 1) * NUM_ARM_JOINTS)
            q_prev = self.q_cmd[:, cols]
            delta = (self._ctrl[k].compute(ee_pos, ee_quat, jac, q_prev) - q_prev) * self.gain
            new_cmd[:, cols] = q_prev + torch.clamp(delta, -self.max_joint_step, self.max_joint_step)
        new_cmd = torch.clamp(new_cmd, self._q_min, self._q_max)
        if mask is not None:
            new_cmd = torch.where(mask[:, None], new_cmd, self.q_cmd)
        self.q_cmd = new_cmd

    def apply(self):
        """팔 관절 목표값을 로봇에 넣는다. env.step 전에 부르면 그 스텝 동안 적용된다."""
        self.robot.set_joint_position_target(self.q_cmd, joint_ids=self.all_joint_ids)
