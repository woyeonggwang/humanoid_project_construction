"""짐 든 채 걷기 전용 명령 생성기 (Isaac Lab 원본 UniformVelocityCommand 상속, 원본은 수정하지 않음)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


class UniformVelocityCommandWithTurn(UniformVelocityCommand):
    """원본 명령에 '제자리 회전' 명령을 일정 비율 섞는다.

    다시 뽑을 때마다 환경별로 rel_turn_envs 확률로 이동 속도 0, 회전 속도 ±turn_speed_range 로 고정한다
    (방향 명령·정지 명령에서 빼서, 다시 뽑을 때까지 그대로 유지).
    원본 방식에서는 이동 속도를 무작위로 뽑으므로 '이동 0 + 큰 회전' 이 거의 나오지 않는다.
    정지 연습(rel_standing_envs)만 늘리면 '이동 0 = 서 있기' 로 일반화해 제자리 회전 중 넘어짐 (2026-11-20 시험 5차).
    """

    cfg: UniformVelocityCommandWithTurnCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if self.cfg.rel_turn_envs <= 0.0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        turn_ids = env_ids[torch.rand(len(env_ids), device=self.device) < self.cfg.rel_turn_envs]
        n = len(turn_ids)
        if n == 0:
            return
        speed = torch.empty(n, device=self.device).uniform_(*self.cfg.turn_speed_range)
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[turn_ids, :2] = 0.0
        self.vel_command_b[turn_ids, 2] = speed * sign
        self.is_heading_env[turn_ids] = False  # 방향 오차로 회전 속도를 덮어쓰지 않게
        self.is_standing_env[turn_ids] = False  # 정지 명령으로 0 이 되지 않게


@configclass
class UniformVelocityCommandWithTurnCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandWithTurn

    rel_turn_envs: float = 0.0
    """다시 뽑을 때 제자리 회전 명령을 받는 환경 비율."""

    turn_speed_range: tuple[float, float] = (0.5, 1.0)
    """제자리 회전 속도 크기 범위 (rad/s). 방향(±)은 반반."""
