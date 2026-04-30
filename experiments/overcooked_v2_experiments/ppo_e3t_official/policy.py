from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo_e3t_official.models.abstract import ActorCriticBase
from overcooked_v2_experiments.ppo_e3t_official.models.model import (
    get_actor_critic,
    initialize_carry,
)
import jax.numpy as jnp
from overcooked_v2_experiments.eval.policy import PolicyPairing
import jax
import jaxmarl
from flax import core, struct
from typing import Any
import chex


@chex.dataclass
class PPOParams:
    params: core.FrozenDict[str, Any]


@struct.dataclass
class E3THistoryState:
    history_obs: chex.Array
    history_actions: chex.Array
    last_self_obs: chex.Array
    initialized: chex.Array


class PPOPolicy(AbstractPolicy):
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

    def _model_config(self):
        return self.config["model"] if "model" in self.config else self.config

    def _use_history_context(self):
        return self._model_config().get("USE_HISTORY_CONTEXT", True)

    def _context_length(self):
        return int(self._model_config().get("CONTEXT_LENGTH", 5))

    def _stay_action(self):
        return int(self._model_config().get("STAY_ACTION", 4))

    def _new_history_state(self, obs):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        history_obs = jnp.repeat(
            obs[jnp.newaxis, ...], self._context_length(), axis=0
        )
        history_actions = jnp.full(
            (self._context_length(),), self._stay_action(), dtype=jnp.int32
        )
        return E3THistoryState(
            history_obs=history_obs,
            history_actions=history_actions,
            last_self_obs=obs,
            initialized=jnp.asarray(True),
        )

    def _ensure_history_state(self, obs, hstate, done=None):
        if not self._use_history_context():
            return hstate
        obs = jnp.asarray(obs, dtype=jnp.float32)
        if not isinstance(hstate, E3THistoryState):
            return self._new_history_state(obs)
        if done is None:
            done = jnp.logical_not(hstate.initialized)
        else:
            done = jnp.logical_or(jnp.asarray(done), jnp.logical_not(hstate.initialized))

        reset_state = self._new_history_state(obs)
        done = jnp.asarray(done)
        obs_mask = done.reshape(done.shape + (1,) * (hstate.history_obs.ndim - done.ndim))
        action_mask = done.reshape(done.shape + (1,) * (hstate.history_actions.ndim - done.ndim))
        last_mask = done.reshape(done.shape + (1,) * (hstate.last_self_obs.ndim - done.ndim))
        return E3THistoryState(
            history_obs=jnp.where(obs_mask, reset_state.history_obs, hstate.history_obs),
            history_actions=jnp.where(
                action_mask, reset_state.history_actions, hstate.history_actions
            ),
            last_self_obs=jnp.where(last_mask, reset_state.last_self_obs, hstate.last_self_obs),
            initialized=jnp.ones_like(hstate.initialized, dtype=bool),
        )

    def _format_network_input(self, obs, done, hstate=None):
        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        if self._use_history_context():
            hstate = self._ensure_history_state(obs, hstate)
            ac_in = (obs, done, hstate.history_obs, hstate.history_actions)
        else:
            ac_in = (obs, done)
        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)
        return ac_in

    def _format_network_input_batch(self, obs_batch, done_batch, hstate=None):
        obs_batch = jnp.array(obs_batch)
        done_batch = jnp.array(done_batch)
        if self._use_history_context():
            if isinstance(hstate, E3THistoryState):
                history_obs = hstate.history_obs
                history_actions = hstate.history_actions
            else:
                history_obs = jnp.repeat(
                    obs_batch[:, jnp.newaxis, ...], self._context_length(), axis=1
                )
                history_actions = jnp.full(
                    (obs_batch.shape[0], self._context_length()),
                    self._stay_action(),
                    dtype=jnp.int32,
                )
            return (
                obs_batch[jnp.newaxis, ...],
                done_batch[jnp.newaxis, ...],
                history_obs[jnp.newaxis, ...],
                history_actions[jnp.newaxis, ...],
            )
        return obs_batch[jnp.newaxis, ...], done_batch[jnp.newaxis, ...]

    def _unpack_apply_output(self, output):
        if len(output) == 3:
            return output[0], output[1], output[2], None, None
        if len(output) == 4:
            return output[0], output[1], output[2], output[3], None
        return output

    def compute_action(self, obs, done, hstate, key, params=None):
        if params is None:
            params = self.params
        assert params is not None

        hstate = self._ensure_history_state(obs, hstate, done)
        ac_in = self._format_network_input(obs, done, hstate)
        _, pi, _, _, _ = self._unpack_apply_output(self.network.apply(params, None, ac_in))

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]
        if self._use_history_context():
            hstate = E3THistoryState(
                history_obs=hstate.history_obs,
                history_actions=hstate.history_actions,
                last_self_obs=jnp.asarray(obs, dtype=jnp.float32),
                initialized=jnp.ones_like(hstate.initialized, dtype=bool),
            )
        return action, hstate

    def update_after_step(self, hstate, partner_obs, partner_action, done):
        if not self._use_history_context():
            return hstate
        del partner_obs
        self_obs = hstate.last_self_obs if isinstance(hstate, E3THistoryState) else None
        if self_obs is None:
            return hstate
        hstate = self._ensure_history_state(self_obs, hstate)
        next_obs = jnp.concatenate(
            [hstate.history_obs[1:], jnp.asarray(self_obs, dtype=jnp.float32)[jnp.newaxis, ...]],
            axis=0,
        )
        next_actions = jnp.concatenate(
            [
                hstate.history_actions[1:],
                jnp.asarray(partner_action, dtype=jnp.int32).reshape((1,)),
            ],
            axis=0,
        )
        reset_state = self._new_history_state(self_obs)
        done = jnp.asarray(done)
        obs_mask = done.reshape((1,) * next_obs.ndim)
        action_mask = done.reshape((1,) * next_actions.ndim)
        last_mask = done.reshape((1,) * hstate.last_self_obs.ndim)
        return E3THistoryState(
            history_obs=jnp.where(obs_mask, reset_state.history_obs, next_obs),
            history_actions=jnp.where(action_mask, reset_state.history_actions, next_actions),
            last_self_obs=jnp.where(last_mask, reset_state.last_self_obs, hstate.last_self_obs),
            initialized=jnp.ones_like(hstate.initialized, dtype=bool),
        )

    def forward_diagnostics(self, obs, done, hstate, params=None):
        if params is None:
            params = self.params
        assert params is not None

        hstate = self._ensure_history_state(obs, hstate)
        ac_in = self._format_network_input(obs, done, hstate)
        _, pi, value, _, _ = self._unpack_apply_output(self.network.apply(params, None, ac_in))

        if self.with_batching:
            probs = pi.probs[0]
            value = value[0]
        else:
            probs = pi.probs[0, 0]
            value = value[0, 0]
        return probs, value, hstate

    def forward_diagnostics_batch(self, obs_batch, done_batch, hstate=None, params=None):
        if params is None:
            params = self.params
        assert params is not None

        ac_in = self._format_network_input_batch(obs_batch, done_batch, hstate)
        next_hstate, pi, value, _, _ = self._unpack_apply_output(
            self.network.apply(params, None, ac_in)
        )
        return pi.probs[0], value[0], next_hstate

    def init_hstate(self, batch_size, key=None):
        carry = initialize_carry(self.config, batch_size)
        if not self._use_history_context():
            return carry

        env_config = self.config.get("env", None)
        if env_config is None:
            return carry
        env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
        obs_shape = env.observation_space().shape
        context_length = self._context_length()
        stay_action = self._stay_action()
        if batch_size == 1 and not self.with_batching:
            history_obs = jnp.zeros((context_length, *obs_shape), dtype=jnp.float32)
            history_actions = jnp.full((context_length,), stay_action, dtype=jnp.int32)
            last_self_obs = jnp.zeros(obs_shape, dtype=jnp.float32)
            initialized = jnp.asarray(False)
        else:
            history_obs = jnp.zeros(
                (batch_size, context_length, *obs_shape), dtype=jnp.float32
            )
            history_actions = jnp.full(
                (batch_size, context_length), stay_action, dtype=jnp.int32
            )
            last_self_obs = jnp.zeros((batch_size, *obs_shape), dtype=jnp.float32)
            initialized = jnp.zeros((batch_size,), dtype=bool)
        return E3THistoryState(
            history_obs=history_obs,
            history_actions=history_actions,
            last_self_obs=last_self_obs,
            initialized=initialized,
        )


def policy_checkoints_to_policy_pairing(checkpoints: PPOParams, config):
    policies = []
    for checkpoint in checkpoints:
        policies.append(PPOPolicy(checkpoint.params, config))
    return PolicyPairing(*policies)
