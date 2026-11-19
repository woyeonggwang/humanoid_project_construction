"""짐 든 채 걷기 전용 이벤트 함수."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def hold_joints_at_default(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg):
    """정책이 움직이지 않는 관절의 모터 목표값을 기본 자세(init_state.joint_pos)로 맞춘다.

    동작(action)에 들어가지 않은 관절은 목표값을 설정하는 곳이 없어 0 으로 남는다
    (inspect_g1.py 로 확인, 2026-11-19). 리셋 때마다 목표값을 기본 자세로 넣어 팔이 자세를 유지하게 한다.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    target = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    asset.set_joint_position_target(target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
