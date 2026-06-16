from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import core

from overcooked_v2_experiments.eval.policy import AbstractPolicy, PolicyPairing
from overcooked_v2_experiments.mappo.models.abstract import ActorCriticBase
from overcooked_v2_experiments.mappo.models.model import get_actor_critic, initialize_carry


@chex.dataclass
class MAPPOParams:
    params: Any


class MAPPOPolicy(AbstractPolicy):
    network: ActorCriticBase
    params: core.FrozenDict[str, Any]
    config: core.FrozenDict[str, Any]
    stochastic: bool = True
    with_batching: bool = False

    def __init__(self, params, config, stochastic=True, with_batching=False):
        self.config = config
        self.stochastic = stochastic
        self.with_batching = with_batching
        self.network = get_actor_critic(config)
        self.params = self._actor_params(params)

    @staticmethod
    def _actor_params(params):
        if isinstance(params, dict) and "actor" in params:
            return params["actor"]
        return params

    def _format_network_input(self, obs, done):
        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        ac_in = (obs, done)
        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)
        return ac_in

    def _format_network_input_batch(self, obs_batch, done_batch):
        obs_batch = jnp.array(obs_batch)
        done_batch = jnp.array(done_batch)
        return obs_batch[jnp.newaxis, ...], done_batch[jnp.newaxis, ...]

    def compute_action(self, obs, done, hstate, key, params=None):
        if params is None:
            params = self.params
        params = self._actor_params(params)

        if hstate is None:
            hstate = self.init_hstate(1)
        ac_in = self._format_network_input(obs, done)
        next_hstate, pi, _ = self.network.apply(params, hstate, ac_in)

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]

        return action, next_hstate

    def forward_diagnostics(self, obs, done, hstate, params=None):
        if params is None:
            params = self.params
        params = self._actor_params(params)

        if hstate is None:
            hstate = self.init_hstate(1)
        ac_in = self._format_network_input(obs, done)
        next_hstate, pi, value = self.network.apply(params, hstate, ac_in)

        if self.with_batching:
            probs = pi.probs[0]
            value = value[0]
        else:
            probs = pi.probs[0, 0]
            value = value[0, 0]

        return probs, value, next_hstate

    def forward_diagnostics_batch(self, obs_batch, done_batch, hstate=None, params=None):
        if params is None:
            params = self.params
        params = self._actor_params(params)

        if hstate is None:
            hstate = self.init_hstate(obs_batch.shape[0])
        ac_in = self._format_network_input_batch(obs_batch, done_batch)
        next_hstate, pi, value = self.network.apply(params, hstate, ac_in)
        return pi.probs[0], value[0], next_hstate

    def init_hstate(self, batch_size, key=None):
        return initialize_carry(self.config, batch_size)


def policy_checkoints_to_policy_pairing(checkpoints: MAPPOParams, config, stochastic: bool = True):
    policies = []
    for checkpoint in checkpoints:
        policies.append(MAPPOPolicy(checkpoint.params, config, stochastic=stochastic))
    return PolicyPairing(*policies)
