
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class GammaTrajectoryDataset:
    obs: np.ndarray
    actions: np.ndarray
    policy_id: np.ndarray | None = None
    episode_returns: np.ndarray | None = None

    @classmethod
    def load_npz(cls, path: str | Path, split: str = "train") -> "GammaTrajectoryDataset":
        data = np.load(path, allow_pickle=False)
        obs_key = f"{split}_obs"
        actions_key = f"{split}_actions"
        policy_key = f"{split}_policy_id"
        returns_key = f"{split}_episode_returns"
        if obs_key not in data:
            obs_key = "obs"
            actions_key = "actions"
            policy_key = "policy_id"
            returns_key = "episode_returns"
        policy_id = data[policy_key] if policy_key in data else None
        episode_returns = data[returns_key] if returns_key in data else None
        return cls(
            obs=data[obs_key],
            actions=data[actions_key],
            policy_id=policy_id,
            episode_returns=episode_returns,
        )

    @property
    def num_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def episode_length(self) -> int:
        return int(self.obs.shape[1])

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return tuple(self.obs.shape[3:])

    def sample_batch(self, rng: np.random.Generator, batch_size: int, chunk_length: int):
        if chunk_length > self.episode_length:
            raise ValueError(f"chunk_length={chunk_length} exceeds episode_length={self.episode_length}")
        episode_idx = rng.integers(0, self.num_episodes, size=batch_size)
        agent_idx = rng.integers(0, self.obs.shape[2], size=batch_size)
        starts = rng.integers(0, self.episode_length - chunk_length + 1, size=batch_size)
        obs_batch = np.empty((chunk_length, batch_size, *self.obs_shape), dtype=np.float32)
        actions_batch = np.empty((chunk_length, batch_size), dtype=np.int32)
        for b, (ep, ag, st) in enumerate(zip(episode_idx, agent_idx, starts)):
            obs_batch[:, b] = self.obs[ep, st : st + chunk_length, ag]
            act = self.actions[ep, st : st + chunk_length, ag]
            actions_batch[:, b] = np.asarray(act).reshape(chunk_length)
        return obs_batch, actions_batch


def split_and_save_npz(
    output_path: str | Path,
    obs,
    actions,
    policy_id=None,
    episode_returns=None,
    validation_ratio=0.1,
    seed=42,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    perm = rng.permutation(n)
    val_n = max(1, int(round(n * validation_ratio))) if n > 1 else 0
    test_idx = perm[:val_n]
    train_idx = perm[val_n:] if val_n > 0 else perm
    if len(train_idx) == 0:
        train_idx = test_idx
    arrays = {
        "train_obs": obs[train_idx].astype(np.float32),
        "train_actions": actions[train_idx].astype(np.int32),
        "test_obs": obs[test_idx].astype(np.float32) if len(test_idx) else obs[train_idx].astype(np.float32),
        "test_actions": actions[test_idx].astype(np.int32) if len(test_idx) else actions[train_idx].astype(np.int32),
    }
    if policy_id is not None:
        arrays["train_policy_id"] = policy_id[train_idx].astype(np.int32)
        arrays["test_policy_id"] = policy_id[test_idx].astype(np.int32) if len(test_idx) else policy_id[train_idx].astype(np.int32)
    if episode_returns is not None:
        episode_returns = np.asarray(episode_returns, dtype=np.float32)
        arrays["train_episode_returns"] = episode_returns[train_idx]
        arrays["test_episode_returns"] = episode_returns[test_idx] if len(test_idx) else episode_returns[train_idx]
    np.savez_compressed(output_path, **arrays)
    return output_path
