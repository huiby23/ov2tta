from __future__ import annotations

from pathlib import Path
from typing import Any

import distrax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import core, struct

from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo.models.model import get_actor_critic


@struct.dataclass
class FrozenPopulationHState:
    policy_idx: jnp.ndarray


class FrozenCheckpointPopulationPolicy(AbstractPolicy):
    """Fixed checkpoint population sampled per actor/episode for TTACv2 training."""

    params: core.FrozenDict[str, Any]
    config: dict
    stochastic: bool = True
    uses_default_observation: bool = True

    def __init__(self, params, config, stochastic: bool = True):
        if config["model"]["TYPE"] != "CNN":
            raise ValueError("FrozenCheckpointPopulationPolicy currently supports CNN partners only.")
        self.params = params if isinstance(params, core.FrozenDict) else core.freeze(params)
        self.config = config
        self.stochastic = stochastic
        self.network = get_actor_critic(config)
        self.population_size = jax.tree_util.tree_leaves(self.params)[0].shape[0]

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, max_policies: int | None = 10, stochastic: bool = True):
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"population run dir does not exist: {run_dir}")
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_dirs = []
        for run_path in sorted(run_dir.glob("run_*"), key=lambda p: int(p.name.split("_")[1])):
            for partner_path in sorted(run_path.glob("partner_*"), key=lambda p: int(p.name.split("_")[1])):
                ckpt = partner_path / "ckpt_final"
                if ckpt.exists():
                    checkpoint_dirs.append(ckpt)
        if max_policies is not None:
            checkpoint_dirs = checkpoint_dirs[: int(max_policies)]
        if not checkpoint_dirs:
            raise FileNotFoundError(f"no partner_*/ckpt_final checkpoints found under {run_dir}")

        configs = []
        params = []
        for ckpt_dir in checkpoint_dirs:
            ckpt = checkpointer.restore(ckpt_dir, item=None)
            configs.append(ckpt["config"])
            params.append(ckpt["params"])
        first_config = configs[0]
        stacked_params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *params)
        print(f"Loaded frozen checkpoint population: {len(params)} policies from {run_dir}", flush=True)
        return cls(stacked_params, first_config, stochastic=stochastic)

    def init_hstate(self, batch_size, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        policy_idx = jax.random.randint(key, (batch_size,), 0, self.population_size)
        return FrozenPopulationHState(policy_idx=policy_idx)

    def _forward_all(self, obs, done):
        def _forward_one(params):
            _, pi, _ = self.network.apply(params, None, (obs[jnp.newaxis, ...], done[jnp.newaxis, ...]))
            return pi.logits[0]
        return jax.vmap(_forward_one)(self.params)

    def compute_action(self, obs, done, hstate: FrozenPopulationHState, key):
        unbatched = done.ndim == 0
        obs = jnp.expand_dims(obs, 0) if unbatched else obs
        done = jnp.expand_dims(done, 0) if unbatched else done
        if hstate is None:
            hstate = self.init_hstate(obs.shape[0], key)
        key_idx, key_action = jax.random.split(key)
        reset_idx = jax.random.randint(key_idx, hstate.policy_idx.shape, 0, self.population_size)
        done_bool = done.astype(jnp.bool_)
        policy_idx = jnp.where(done_bool, reset_idx, hstate.policy_idx)
        all_logits = self._forward_all(obs, done)
        logits = jnp.take_along_axis(all_logits, policy_idx[jnp.newaxis, :, jnp.newaxis], axis=0).squeeze(axis=0)
        dist = distrax.Categorical(logits=logits)
        action = dist.sample(seed=key_action) if self.stochastic else jnp.argmax(logits, axis=-1)
        action = action[0] if unbatched else action
        return action, FrozenPopulationHState(policy_idx=policy_idx)
