from typing import Any
import copy

import chex
import jax
import jax.numpy as jnp
import jaxmarl
from flax import core

from overcooked_v2_experiments.eval.policy import AbstractPolicy, PolicyPairing
from overcooked_v2_experiments.e3t_ori.models.original import E3TContextModule, E3TPolicyModule


@chex.dataclass
class E3TParams:
    params: core.FrozenDict[str, Any]


@chex.dataclass
class E3THState:
    history_obs: jnp.ndarray
    history_actions: jnp.ndarray
    pending_obs: jnp.ndarray
    initialized: jnp.ndarray


def _resolve_obs_shape(config):
    env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
    env_kwargs.pop("obs_shape", None)
    obs_shape = env_kwargs.get("obs_shape")
    if obs_shape is not None:
        return tuple(obs_shape)

    env = jaxmarl.make(config["env"]["ENV_NAME"], **env_kwargs)
    obs_shape = tuple(env.observation_space().shape)
    config["env"]["ENV_KWARGS"]["obs_shape"] = obs_shape
    return obs_shape


class E3TPolicy(AbstractPolicy):
    def __init__(self, params, config, stochastic=True, with_batching=False):
        self.config = config
        self.stochastic = stochastic
        self.with_batching = with_batching
        self.params = params
        self.model_config = config["model"]
        self.obs_shape = _resolve_obs_shape(config)
        env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
        env_kwargs.pop("obs_shape", None)
        env = jaxmarl.make(config["env"]["ENV_NAME"], **env_kwargs)
        self.action_dim = env.action_space(env.agents[0]).n
        self.context_module = E3TContextModule(self.model_config, self.action_dim)
        self.policy_module = E3TPolicyModule(self.model_config, self.action_dim)
        self.context_length = self.model_config["CONTEXT_LENGTH"]
        self.stay_action = min(4, self.action_dim - 1)

    def _bootstrap_history(self, obs_batch):
        history_obs = jnp.repeat(obs_batch[:, None, ...], self.context_length, axis=1)
        history_actions = jnp.full(
            (obs_batch.shape[0], self.context_length),
            self.stay_action,
            dtype=jnp.int32,
        )
        return history_obs, history_actions

    def compute_action(self, obs, done, hstate, key, params=None):
        params = self.params if params is None else params
        if hstate is None:
            hstate = self.init_hstate(1)

        obs = jnp.asarray(obs)
        if self.with_batching:
            batched_obs = obs
        else:
            batched_obs = obs[jnp.newaxis, ...]

        bootstrap_history_obs, bootstrap_history_actions = self._bootstrap_history(batched_obs.astype(jnp.float32))
        init_obs_mask = hstate.initialized.reshape((hstate.initialized.shape[0],) + (1,) * (hstate.history_obs.ndim - 1))
        init_act_mask = hstate.initialized[:, None]
        history_obs = jnp.where(init_obs_mask, hstate.history_obs, bootstrap_history_obs)
        history_actions = jnp.where(init_act_mask, hstate.history_actions, bootstrap_history_actions)

        context_logits = self.context_module.apply(
            params["context_params"],
            batched_obs,
            history_obs.astype(batched_obs.dtype),
            history_actions,
        )
        context_probs = jax.nn.softmax(context_logits, axis=-1)
        pi, _ = self.policy_module.apply(params["policy_params"], batched_obs, context_probs)

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        next_hstate = E3THState(
            history_obs=history_obs,
            history_actions=history_actions,
            pending_obs=batched_obs.astype(jnp.float32),
            initialized=jnp.ones_like(hstate.initialized, dtype=jnp.bool_),
        )
        if self.with_batching:
            return action, next_hstate
        return action[0], next_hstate

    def update_after_step(self, hstate, partner_obs, partner_action, done):
        del partner_obs
        partner_action = jnp.asarray(partner_action)
        done = jnp.asarray(done)

        if not self.with_batching:
            partner_action = partner_action[jnp.newaxis]
            done = done[jnp.newaxis]

        next_history_obs = jnp.concatenate(
            [hstate.history_obs[:, 1:], hstate.pending_obs[:, None, ...]],
            axis=1,
        )
        next_history_actions = jnp.concatenate(
            [hstate.history_actions[:, 1:], partner_action[:, None]],
            axis=1,
        )

        done_obs_mask = done.reshape((done.shape[0],) + (1,) * (next_history_obs.ndim - 1))
        done_act_mask = done[:, None]
        next_history_obs = jnp.where(done_obs_mask, jnp.zeros_like(next_history_obs), next_history_obs)
        next_history_actions = jnp.where(
            done_act_mask,
            jnp.full_like(next_history_actions, self.stay_action),
            next_history_actions,
        )

        return E3THState(
            history_obs=next_history_obs,
            history_actions=next_history_actions,
            pending_obs=jnp.zeros_like(hstate.pending_obs),
            initialized=jnp.logical_not(done),
        )

    def init_hstate(self, batch_size, key=None):
        del key
        return E3THState(
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
