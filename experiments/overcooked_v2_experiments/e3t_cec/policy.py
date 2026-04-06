from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import core

from overcooked_v2_experiments.e3t_cec.models.abstract import ActorCriticBase
from overcooked_v2_experiments.e3t_cec.models.model import (
    get_actor_critic,
    initialize_carry,
)
from overcooked_v2_experiments.eval.policy import AbstractPolicy, PolicyPairing


@chex.dataclass
class E3TCECParams:
    params: core.FrozenDict[str, Any]


@chex.dataclass
class E3TCECHState:
    actor_hstate: Any


class E3TCECPolicy(AbstractPolicy):
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
        self.params = params

    def compute_action(self, obs, done, hstate, key, params=None):
        if params is None:
            params = self.params
        if hstate is None:
            hstate = self.init_hstate(1)

        obs = jnp.asarray(obs)
        done = jnp.asarray(done)

        if self.with_batching:
            flat_obs = obs.reshape((obs.shape[0], -1))
            batched_obs = flat_obs[jnp.newaxis, ...]
            batched_done = done[jnp.newaxis, ...]
            agent_positions = jnp.zeros((1, flat_obs.shape[0], 2), dtype=jnp.int32)
        else:
            flat_obs = obs.reshape((1, -1))
            batched_obs = flat_obs[jnp.newaxis, ...]
            batched_done = done[jnp.newaxis, jnp.newaxis, ...]
            agent_positions = jnp.zeros((1, 1, 2), dtype=jnp.int32)

        next_actor_hstate, pi, _, _ = self.network.apply(
            params,
            hstate.actor_hstate,
            (batched_obs, batched_done, agent_positions),
        )
        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        next_hstate = E3TCECHState(actor_hstate=next_actor_hstate)
        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]
        return action, next_hstate

    def init_hstate(self, batch_size, key=None):
        del key
        return E3TCECHState(
            actor_hstate=initialize_carry(self.config, batch_size),
        )


def policy_checkoints_to_policy_pairing(checkpoints: E3TCECParams, config):
    policies = []
    for checkpoint in checkpoints:
        policies.append(E3TCECPolicy(checkpoint.params, config))
    return PolicyPairing(*policies)
