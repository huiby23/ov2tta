
from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path
from typing import Any

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import core

from overcooked_v2_experiments.eval.policy import AbstractPolicy
from .models import GammaVAE


@chex.dataclass
class GammaPartnerHState:
    decoder_carry: chex.Array
    z: chex.Array


class GammaGeneratedPartnerPolicy(AbstractPolicy):
    """Generated partner p(a|o,z) used as the fixed population side in IPPO."""

    network: GammaVAE
    params: core.FrozenDict[str, Any]
    z_dim: int
    hidden_dim: int
    action_dim: int
    z_mean: chex.Array
    z_std: chex.Array
    stochastic: bool = True
    uses_default_observation: bool = True

    def __init__(self, params, action_dim: int, z_dim: int = 16, hidden_dim: int = 64, activation: str = "relu", z_mean=None, z_std=None, stochastic: bool = True):
        self.network = GammaVAE(action_dim=action_dim, z_dim=z_dim, hidden_dim=hidden_dim, activation_name=activation)
        self.params = params
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.z_mean = jnp.zeros((z_dim,), dtype=jnp.float32) if z_mean is None else jnp.asarray(z_mean, dtype=jnp.float32)
        self.z_std = jnp.ones((z_dim,), dtype=jnp.float32) if z_std is None else jnp.asarray(z_std, dtype=jnp.float32)
        self.stochastic = stochastic

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, stochastic: bool = True):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        cfg = ckpt["config"]
        return cls(params=ckpt["params"], action_dim=int(cfg["action_dim"]), z_dim=int(cfg.get("z_dim", 16)), hidden_dim=int(cfg.get("hidden_dim", 64)), activation=cfg.get("activation", "relu"), z_mean=ckpt.get("z_mean", None), z_std=ckpt.get("z_std", None), stochastic=stochastic)

    def _sample_z(self, key, batch_size):
        eps = jax.random.normal(key, (batch_size, self.z_dim))
        return self.z_mean[jnp.newaxis, :] + self.z_std[jnp.newaxis, :] * eps

    def init_hstate(self, batch_size, key=None) -> GammaPartnerHState:
        if key is None:
            key = jax.random.PRNGKey(0)
        decoder_carry = jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32)
        return GammaPartnerHState(decoder_carry=decoder_carry, z=self._sample_z(key, batch_size))

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate: GammaPartnerHState, key):
        unbatched = done.ndim == 0
        obs = jnp.expand_dims(obs, 0) if unbatched else obs
        done = jnp.expand_dims(done, 0) if unbatched else done
        batch_size = obs.shape[0]
        key_z, key_action = jax.random.split(key)
        reset_z = self._sample_z(key_z, batch_size)
        reset_carry = jnp.zeros_like(hstate.decoder_carry)
        done = done.astype(jnp.bool_)
        carry = jnp.where(done[:, None], reset_carry, hstate.decoder_carry)
        z = jnp.where(done[:, None], reset_z, hstate.z)
        logits, new_carry = self.network.apply({"params": self.params}, obs, z, carry, method=GammaVAE.decode_step)
        dist = distrax.Categorical(logits=logits)
        if self.stochastic:
            action = dist.sample(seed=key_action)
        else:
            action = jnp.argmax(logits, axis=-1)
        action = action[0] if unbatched else action
        return action, GammaPartnerHState(decoder_carry=new_carry, z=z)
