from typing import Any
import copy

import chex
import jax
import jax.numpy as jnp
import jaxmarl
from flax import core

from overcooked_v2_experiments.eval.policy import AbstractPolicy, PolicyPairing
from overcooked_v2_experiments.e3t.models.abstract import ActorCriticBase
from overcooked_v2_experiments.e3t.models.model import get_actor_critic, initialize_carry


@chex.dataclass
class E3TParams:
    params: core.FrozenDict[str, Any]


@chex.dataclass
class E3THState:
    actor_hstate: Any
    history_obs: jnp.ndarray
    history_actions: jnp.ndarray
    pending_obs: jnp.ndarray
    initialized: jnp.ndarray


def _resolve_obs_shape(config):
    env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
    obs_shape = env_kwargs.get("obs_shape")
    if obs_shape is not None:
        return tuple(obs_shape)

    env = jaxmarl.make(config["env"]["ENV_NAME"], **env_kwargs)
    obs_shape = tuple(env.observation_space().shape)
    config["env"]["ENV_KWARGS"]["obs_shape"] = obs_shape
    return obs_shape


class E3TPolicy(AbstractPolicy):
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
        self.context_length = config["model"].get("CONTEXT_LENGTH", 5)
        self.obs_shape = _resolve_obs_shape(config)
        self.stay_action = min(4, self.network.action_dim - 1)

    def _bootstrap_history(self, obs_batch):
        history_obs = jnp.repeat(obs_batch[:, None, ...], self.context_length, axis=1)
        history_actions = jnp.full(
            (obs_batch.shape[0], self.context_length),
            self.stay_action,
            dtype=jnp.int32,
        )
        return history_obs, history_actions

    def compute_action(self, obs, done, hstate, key, params=None):
        if params is None:
            params = self.params
        if hstate is None:
            hstate = self.init_hstate(1)

        obs = jnp.asarray(obs)
        done = jnp.asarray(done)

        if self.with_batching:
            batched_obs = obs[jnp.newaxis, ...]
            batched_done = done[jnp.newaxis, ...]
            hist_obs = hstate.history_obs[jnp.newaxis, ...].astype(obs.dtype)
            hist_actions = hstate.history_actions[jnp.newaxis, ...]
        else:
            batched_obs = obs[jnp.newaxis, jnp.newaxis, ...]
            batched_done = done[jnp.newaxis, jnp.newaxis, ...]
            hist_obs = hstate.history_obs[jnp.newaxis, ...].astype(obs.dtype)
            hist_actions = hstate.history_actions[jnp.newaxis, ...]

        flat_obs = batched_obs.reshape((-1,) + batched_obs.shape[-3:])
        bootstrap_history_obs, bootstrap_history_actions = self._bootstrap_history(flat_obs.astype(jnp.float32))
        if self.with_batching:
            bootstrap_history_obs = bootstrap_history_obs.reshape((1, flat_obs.shape[0], self.context_length) + self.obs_shape)
            bootstrap_history_actions = bootstrap_history_actions.reshape((1, flat_obs.shape[0], self.context_length))
            init_mask = hstate.initialized.reshape((1, flat_obs.shape[0], 1, 1, 1, 1))
            init_act_mask = hstate.initialized.reshape((1, flat_obs.shape[0], 1))
        else:
            bootstrap_history_obs = bootstrap_history_obs.reshape((1, 1, self.context_length) + self.obs_shape)
            bootstrap_history_actions = bootstrap_history_actions.reshape((1, 1, self.context_length))
            init_mask = hstate.initialized.reshape((1, 1, 1, 1, 1, 1))
            init_act_mask = hstate.initialized.reshape((1, 1, 1))

        hist_obs = jnp.where(init_mask, hist_obs, bootstrap_history_obs).astype(obs.dtype)
        hist_actions = jnp.where(init_act_mask, hist_actions, bootstrap_history_actions)

        next_actor_hstate, pi, _, _ = self.network.apply(
            params,
            hstate.actor_hstate,
            (batched_obs, batched_done, hist_obs, hist_actions),
        )
        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        next_hstate = E3THState(
            actor_hstate=next_actor_hstate,
            history_obs=hist_obs[0].astype(jnp.float32),
            history_actions=hist_actions[0],
            pending_obs=flat_obs.astype(jnp.float32),
            initialized=jnp.ones_like(hstate.initialized, dtype=jnp.bool_),
        )
        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]
        return action, next_hstate

    def update_after_step(self, hstate, partner_obs, partner_action, done):
        del partner_obs
        partner_action = jnp.asarray(partner_action)
        done = jnp.asarray(done)

        if not self.with_batching:
            partner_action = partner_action[jnp.newaxis]
            done = done[jnp.newaxis]

        next_history_actions = jnp.concatenate(
            [hstate.history_actions[:, 1:], partner_action[:, None]],
            axis=1,
        )

        bootstrap_history_obs, bootstrap_history_actions = self._bootstrap_history(
            hstate.pending_obs.astype(jnp.float32)
        )
        next_history_obs = jnp.concatenate(
            [hstate.history_obs[:, 1:], hstate.pending_obs[:, None, ...].astype(hstate.history_obs.dtype)],
            axis=1,
        )
        done_obs_mask = done.reshape((done.shape[0],) + (1,) * (next_history_obs.ndim - 1))
        done_act_mask = done[:, None]
        next_history_obs = jnp.where(done_obs_mask, bootstrap_history_obs, next_history_obs)
        next_history_actions = jnp.where(
            done_act_mask,
            bootstrap_history_actions,
            next_history_actions,
        )

        return E3THState(
            actor_hstate=hstate.actor_hstate,
            history_obs=next_history_obs,
            history_actions=next_history_actions,
            pending_obs=jnp.zeros_like(hstate.pending_obs),
            initialized=jnp.logical_not(done),
        )

    def init_hstate(self, batch_size, key=None):
        return E3THState(
            actor_hstate=initialize_carry(self.config, batch_size),
            history_obs=jnp.zeros((batch_size, self.context_length) + self.obs_shape, dtype=jnp.float32),
            history_actions=jnp.full((batch_size, self.context_length), self.stay_action, dtype=jnp.int32),
            pending_obs=jnp.zeros((batch_size,) + self.obs_shape, dtype=jnp.float32),
            initialized=jnp.zeros((batch_size,), dtype=jnp.bool_),
        )


def policy_checkoints_to_policy_pairing(checkpoints: E3TParams, config):
    policies = []
    for checkpoint in checkpoints:
        policies.append(E3TPolicy(checkpoint.params, config))
    return PolicyPairing(*policies)
