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

from overcooked_v2_experiments.eval.policy import AbstractPolicy, PolicyPairing
from overcooked_v2_experiments.talents.models.abstract import ActorCriticBase
from overcooked_v2_experiments.talents.models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.gamma.models import GammaVAE


@chex.dataclass
class PPOParams:
    params: core.FrozenDict[str, Any]


@chex.dataclass
class TalentsPartnerHState:
    decoder_carry: chex.Array
    z: chex.Array
    cluster_id: chex.Array


@chex.dataclass
class TalentsBeliefHState:
    base_hstate: Any
    weights: chex.Array
    decoder_carry: chex.Array


def _load_vae_and_clusters(vae_checkpoint: str | Path, cluster_checkpoint: str | Path):
    with open(vae_checkpoint, "rb") as f:
        vae_ckpt = pickle.load(f)
    with open(cluster_checkpoint, "rb") as f:
        cluster_ckpt = pickle.load(f)
    cfg = vae_ckpt["config"]
    network = GammaVAE(
        action_dim=int(cfg["action_dim"]),
        z_dim=int(cfg.get("z_dim", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        activation_name=cfg.get("activation", "relu"),
    )
    return network, vae_ckpt, cluster_ckpt


class TalentsGeneratedPartnerPolicy(AbstractPolicy):
    """TALENTS generated partner p(a|o,z_c) for Algorithm 1 line 12."""

    uses_default_observation: bool = True

    def __init__(self, vae_checkpoint, cluster_checkpoint, stochastic=True, z_sample_scale=1.0):
        self.network, self.vae_ckpt, self.cluster_ckpt = _load_vae_and_clusters(vae_checkpoint, cluster_checkpoint)
        self.params = self.vae_ckpt["params"]
        self.cluster_mean = jnp.asarray(self.cluster_ckpt["cluster_mean"], dtype=jnp.float32)
        self.cluster_std = jnp.asarray(self.cluster_ckpt["cluster_std"], dtype=jnp.float32)
        self.num_clusters = int(self.cluster_mean.shape[0])
        self.z_dim = int(self.cluster_mean.shape[1])
        self.hidden_dim = int(self.vae_ckpt["config"].get("hidden_dim", 64))
        self.action_dim = int(self.vae_ckpt["config"].get("action_dim", 6))
        self.stochastic = stochastic
        self.z_sample_scale = float(z_sample_scale)

    @classmethod
    def from_checkpoints(cls, vae_checkpoint, cluster_checkpoint, stochastic=True, z_sample_scale=1.0):
        return cls(
            vae_checkpoint,
            cluster_checkpoint,
            stochastic=stochastic,
            z_sample_scale=z_sample_scale,
        )

    def _sample_z_for_cluster(self, cluster_id, key):
        cluster_id = jnp.clip(cluster_id.astype(jnp.int32), 0, self.num_clusters - 1)
        eps = jax.random.normal(key, cluster_id.shape + (self.z_dim,))
        return self.cluster_mean[cluster_id] + self.z_sample_scale * self.cluster_std[cluster_id] * eps

    def init_hstate(self, batch_size, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_cluster, key_z = jax.random.split(key)
        cluster_id = jax.random.randint(key_cluster, (batch_size,), 0, self.num_clusters, dtype=jnp.int32)
        z = self._sample_z_for_cluster(cluster_id, key_z)
        decoder_carry = jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32)
        return TalentsPartnerHState(decoder_carry=decoder_carry, z=z, cluster_id=cluster_id)

    def set_cluster_ids(self, hstate: TalentsPartnerHState, cluster_id, key):
        cluster_id = cluster_id.astype(jnp.int32)
        sampled_z = self._sample_z_for_cluster(cluster_id, key)
        same_cluster = cluster_id == hstate.cluster_id
        z = jnp.where(same_cluster[:, None], hstate.z, sampled_z)
        decoder_carry = jnp.where(
            same_cluster[:, None],
            hstate.decoder_carry,
            jnp.zeros_like(hstate.decoder_carry),
        )
        return TalentsPartnerHState(
            decoder_carry=decoder_carry,
            z=z,
            cluster_id=cluster_id,
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate: TalentsPartnerHState, key):
        unbatched = done.ndim == 0
        obs = obs[jnp.newaxis, ...] if unbatched else obs
        done = done[jnp.newaxis, ...] if unbatched else done
        done = done.astype(jnp.bool_)
        carry = jnp.where(done[:, None], jnp.zeros_like(hstate.decoder_carry), hstate.decoder_carry)
        logits, new_carry = self.network.apply({"params": self.params}, obs, hstate.z, carry, method=GammaVAE.decode_step)
        dist = distrax.Categorical(logits=logits)
        action = dist.sample(seed=key) if self.stochastic else jnp.argmax(logits, axis=-1)
        action = action[0] if unbatched else action
        return action, TalentsPartnerHState(decoder_carry=new_carry, z=hstate.z, cluster_id=hstate.cluster_id)


class TalentsPPOPolicy(AbstractPolicy):
    """Cluster-biased cooperator with Algorithm 2 fixed-share online belief update."""

    network: ActorCriticBase
    params: core.FrozenDict[str, Any]
    config: core.FrozenDict[str, Any]
    stochastic: bool = True
    with_batching: bool = False

    def __init__(self, params, config, stochastic=True, with_batching=False, eval_mode="fixed_share"):
        self.config = config
        self.stochastic = stochastic
        self.with_batching = with_batching
        self.eval_mode = eval_mode
        self.network = get_actor_critic(config)
        self.params = params
        tcfg = config.get("TALENTS", {})
        self.alpha = float(tcfg.get("FIXED_SHARE_ALPHA", 0.4))
        self.eta = float(tcfg.get("FIXED_SHARE_ETA", 0.2))
        self.regret_clip = float(tcfg.get("REGRET_CLIP", 1.0))
        self.strategy_network = None
        self.strategy_params = None
        self.cluster_mean = None
        self.cluster_std = None
        self.hidden_dim = 64
        if tcfg.get("VAE_CHECKPOINT") and tcfg.get("CLUSTER_CHECKPOINT"):
            self.strategy_network, vae_ckpt, cluster_ckpt = _load_vae_and_clusters(tcfg["VAE_CHECKPOINT"], tcfg["CLUSTER_CHECKPOINT"])
            self.strategy_params = vae_ckpt["params"]
            self.cluster_mean = jnp.asarray(cluster_ckpt["cluster_mean"], dtype=jnp.float32)
            self.cluster_std = jnp.asarray(cluster_ckpt["cluster_std"], dtype=jnp.float32)
            self.hidden_dim = int(vae_ckpt["config"].get("hidden_dim", 64))
        else:
            k = int(config["model"].get("NUM_STRATEGY_CLUSTERS", 1))
            self.cluster_mean = jnp.zeros((k, 1), dtype=jnp.float32)
            self.cluster_std = jnp.ones((k, 1), dtype=jnp.float32)
        self.num_clusters = int(self.cluster_mean.shape[0])

    def _format_network_input(self, obs, done, cluster_id):
        done = jnp.asarray(done)
        cluster_id = jnp.asarray(cluster_id, dtype=jnp.int32)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        ac_in = (obs, done, cluster_id)
        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)
        return ac_in

    def _format_network_input_batch(self, obs_batch, done_batch, cluster_id):
        obs_batch = jnp.asarray(obs_batch)
        done_batch = jnp.asarray(done_batch)
        cluster_id = jnp.asarray(cluster_id, dtype=jnp.int32)
        return obs_batch[jnp.newaxis, ...], done_batch[jnp.newaxis, ...], cluster_id[jnp.newaxis, ...]

    def _leading_cluster(self, hstate):
        return jnp.argmax(hstate.weights).astype(jnp.int32)

    def compute_action(self, obs, done, hstate: TalentsBeliefHState, key, params=None):
        if params is None:
            params = self.params
        cluster_id = self._leading_cluster(hstate)
        ac_in = self._format_network_input(obs, done, cluster_id)
        next_base_hstate, pi, _ = self.network.apply(params, hstate.base_hstate, ac_in)
        action = pi.sample(seed=key) if self.stochastic else jnp.argmax(pi.probs, axis=-1)
        action = action[0] if self.with_batching else action[0, 0]
        return action, TalentsBeliefHState(base_hstate=next_base_hstate, weights=hstate.weights, decoder_carry=hstate.decoder_carry)

    def init_hstate(self, batch_size, key=None):
        if batch_size != 1:
            # Evaluation rollout uses batch_size=1. Batch diagnostics can still use
            # forward_diagnostics_batch without belief state.
            pass
        base_hstate = initialize_carry(self.config, batch_size)
        weights = jnp.ones((self.num_clusters,), dtype=jnp.float32) / self.num_clusters
        decoder_carry = jnp.zeros((self.num_clusters, self.hidden_dim), dtype=jnp.float32)
        return TalentsBeliefHState(base_hstate=base_hstate, weights=weights, decoder_carry=decoder_carry)

    def update_after_step(self, hstate: TalentsBeliefHState, partner_obs, partner_action, done):
        if self.strategy_network is None or self.eval_mode == "static":
            return hstate
        obs = jnp.broadcast_to(partner_obs[jnp.newaxis, ...], (self.num_clusters,) + partner_obs.shape)
        z = self.cluster_mean
        logits, new_carry = self.strategy_network.apply({"params": self.strategy_params}, obs, z, hstate.decoder_carry, method=GammaVAE.decode_step)
        logp = jax.nn.log_softmax(logits, axis=-1)
        action = partner_action.astype(jnp.int32)
        loss = -logp[jnp.arange(self.num_clusters), action]
        loss = jnp.clip(loss, 0.0, self.regret_clip)
        pre = hstate.weights * jnp.exp(-self.eta * loss)
        pre = pre / jnp.maximum(pre.sum(), 1e-8)
        shared = pre.mean()
        weights = (1.0 - self.alpha) * pre + self.alpha * shared
        weights = weights / jnp.maximum(weights.sum(), 1e-8)
        done = jnp.asarray(done).astype(jnp.bool_)
        reset_weights = jnp.ones_like(weights) / self.num_clusters
        reset_carry = jnp.zeros_like(new_carry)
        weights = jnp.where(done, reset_weights, weights)
        new_carry = jnp.where(done, reset_carry, new_carry)
        return TalentsBeliefHState(base_hstate=hstate.base_hstate, weights=weights, decoder_carry=new_carry)

    def forward_diagnostics(self, obs, done, hstate, params=None):
        if params is None:
            params = self.params
        cluster_id = self._leading_cluster(hstate) if isinstance(hstate, TalentsBeliefHState) else jnp.array(0, dtype=jnp.int32)
        base_hstate = hstate.base_hstate if isinstance(hstate, TalentsBeliefHState) else hstate
        ac_in = self._format_network_input(obs, done, cluster_id)
        next_hstate, pi, value = self.network.apply(params, base_hstate, ac_in)
        probs = pi.probs[0] if self.with_batching else pi.probs[0, 0]
        value = value[0] if self.with_batching else value[0, 0]
        if isinstance(hstate, TalentsBeliefHState):
            next_hstate = TalentsBeliefHState(base_hstate=next_hstate, weights=hstate.weights, decoder_carry=hstate.decoder_carry)
        return probs, value, next_hstate

    def forward_diagnostics_batch(self, obs_batch, done_batch, hstate=None, params=None):
        if params is None:
            params = self.params
        cluster_id = jnp.zeros((obs_batch.shape[0],), dtype=jnp.int32)
        ac_in = self._format_network_input_batch(obs_batch, done_batch, cluster_id)
        next_hstate, pi, value = self.network.apply(params, hstate, ac_in)
        return pi.probs[0], value[0], next_hstate


# Backward-compatible names expected by copied store/eval utilities.
PPOPolicy = TalentsPPOPolicy


def policy_checkoints_to_policy_pairing(checkpoints: PPOParams, config, stochastic: bool = True):
    policies = [TalentsPPOPolicy(checkpoint.params, config, stochastic=stochastic) for checkpoint in checkpoints]
    return PolicyPairing(*policies)
