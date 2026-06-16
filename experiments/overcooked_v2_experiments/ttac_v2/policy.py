from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
from flax import core, struct
from flax.traverse_util import flatten_dict, unflatten_dict
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

from overcooked_v2_experiments.eval.policy import AbstractPolicy, PolicyPairing
from overcooked_v2_experiments.ttac_v2.models.abstract import ActorCriticBase
from overcooked_v2_experiments.ttac_v2.models.model import get_actor_critic, initialize_carry


@chex.dataclass
class PPOParams:
    params: core.FrozenDict[str, Any]


@struct.dataclass
class TTACPolicyState:
    params: core.FrozenDict[str, Any]
    base_hstate: Any
    last_ego_obs: jnp.ndarray
    ego_obs_buffer: jnp.ndarray
    ego_action_buffer: jnp.ndarray
    partner_obs_buffer: jnp.ndarray
    partner_action_buffer: jnp.ndarray
    valid_mask: jnp.ndarray
    cursor: jnp.ndarray
    count: jnp.ndarray
    cached_self_action: jnp.ndarray
    prev_partner_action: jnp.ndarray


EVAL_MODES = (
    "base_no_test_adapt",
    "ttac_true_history",
    "ttac_projected_history",
    "ttac_confident_history",
    "ttac_projected_confident",
    "ttac_advantage_weighted",
    "ttac_value_gated_projected",
    "ttac_ego_advantage_weighted",
    "ttac_ego_aw_wrong_history",
    "ttac_ego_aw_random_history",
    "ttac_ego_aw_delayed_history",
    "ttac_contrastive_ego_aw",
    "ttac_contrastive_ego_aw_wrong_history",
    "ttac_contrastive_ego_aw_random_history",
    "ttac_contrastive_ego_aw_delayed_history",
    "ttac_semantic_ego_aw",
    "ttac_semantic_ego_aw_wrong_history",
    "ttac_semantic_ego_aw_random_history",
    "ttac_semantic_ego_aw_delayed_history",
    "ttac_wrong_history",
    "ttac_random_history",
    "ttac_delayed_history",
    "ttac_no_kl",
    "ttac_adapter_off",
)


def _apply_policy_network(network, params, hstate, ac_in):
    out = network.apply(params, hstate, ac_in)
    if isinstance(out, tuple) and len(out) == 4:
        return out
    next_hstate, pi, value = out
    return next_hstate, pi, value, {}


def _categorical_symmetric_kl(base_logits, adapted_logits):
    base_logits = jax.lax.stop_gradient(base_logits)
    base_log_probs = jax.nn.log_softmax(base_logits, axis=-1)
    adapted_log_probs = jax.nn.log_softmax(adapted_logits, axis=-1)
    base_probs = jax.nn.softmax(base_logits, axis=-1)
    adapted_probs = jax.nn.softmax(adapted_logits, axis=-1)
    kl_base_to_adapted = jnp.sum(
        base_probs * (base_log_probs - adapted_log_probs), axis=-1
    )
    kl_adapted_to_base = jnp.sum(
        adapted_probs * (adapted_log_probs - base_log_probs), axis=-1
    )
    return 0.5 * (kl_base_to_adapted + kl_adapted_to_base)


def _masked_mean(x, mask):
    mask = mask.astype(x.dtype)
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _make_adapter_mask(params):
    flat = flatten_dict(params, sep="/")
    mask_flat = {k: ("ttac_adapter_" in k) for k in flat}
    return core.freeze(unflatten_dict(mask_flat, sep="/"))


def _mask_grads(grads, mask):
    return jax.tree_util.tree_map(
        lambda g, m: jnp.where(jnp.asarray(m), g, jnp.zeros_like(g)), grads, mask
    )


def _resolve_history_action(
    eval_mode, partner_action, self_action, prev_partner_action, action_dim
):
    partner_action = partner_action.astype(jnp.int32)
    self_action = self_action.astype(jnp.int32)
    prev_partner_action = prev_partner_action.astype(jnp.int32)
    if eval_mode in (
        "ttac_wrong_history",
        "ttac_ego_aw_wrong_history",
        "ttac_contrastive_ego_aw_wrong_history",
        "ttac_semantic_ego_aw_wrong_history",
    ):
        return self_action, partner_action
    if eval_mode in (
        "ttac_random_history",
        "ttac_ego_aw_random_history",
        "ttac_contrastive_ego_aw_random_history",
        "ttac_semantic_ego_aw_random_history",
    ):
        return (partner_action + self_action + 1) % action_dim, partner_action
    if eval_mode in (
        "ttac_delayed_history",
        "ttac_ego_aw_delayed_history",
        "ttac_contrastive_ego_aw_delayed_history",
        "ttac_semantic_ego_aw_delayed_history",
    ):
        return prev_partner_action, partner_action
    return partner_action, partner_action


class PPOPolicy(AbstractPolicy):
    network: ActorCriticBase
    params: core.FrozenDict[str, Any]
    config: core.FrozenDict[str, Any]
    stochastic: bool = True
    with_batching: bool = False

    def __init__(
        self,
        params,
        config,
        stochastic=True,
        with_batching=False,
        eval_mode="base_no_test_adapt",
    ):
        if config["model"]["TYPE"] != "CNN":
            raise ValueError("TTAC policy currently supports the CNN model only.")
        if eval_mode not in EVAL_MODES:
            raise ValueError(f"Unknown TTAC eval_mode: {eval_mode}")
        self.config = config
        self.model_config = config["model"]
        self.stochastic = stochastic
        self.with_batching = with_batching
        self.eval_mode = eval_mode
        self.network = get_actor_critic(config)
        self.params = params if isinstance(params, core.FrozenDict) else core.freeze(params)
        self.adapter_mask = _make_adapter_mask(self.params)
        env = OvercookedV2(**config["env"]["ENV_KWARGS"])
        self.obs_shape = tuple(env.observation_space().shape)
        self.history_len = int(self.model_config.get("TTAC_HISTORY_LEN", 50))
        self.action_dim = int(env.action_space(env.agents[0]).n)

    def _format_network_input(self, obs, done, adapter_readout_scale=1.0):
        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        ac_in = (obs, done, jnp.asarray(adapter_readout_scale, dtype=jnp.float32))
        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)
        return ac_in

    def _format_network_input_batch(self, obs_batch, done_batch, adapter_readout_scale=1.0):
        obs_batch = jnp.array(obs_batch)
        done_batch = jnp.array(done_batch)
        return (
            obs_batch[jnp.newaxis, ...],
            done_batch[jnp.newaxis, ...],
            jnp.asarray(adapter_readout_scale, dtype=jnp.float32),
        )

    def _adapter_readout_scale(self):
        return jnp.array(0.0 if self.eval_mode == "ttac_adapter_off" else 1.0, dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate, key, params=None):
        if hstate is None:
            hstate = self.init_hstate(1)
        if params is None:
            params = hstate.params
        obs_f = jnp.asarray(obs, dtype=jnp.float32)
        ac_in = self._format_network_input(obs_f, done, self._adapter_readout_scale())
        next_hstate, pi, _, _ = _apply_policy_network(
            self.network, params, hstate.base_hstate, ac_in
        )
        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)
        action = action[0] if self.with_batching else action[0, 0]
        hstate = hstate.replace(
            params=params,
            base_hstate=next_hstate,
            last_ego_obs=obs_f,
            cached_self_action=action,
        )
        return action, hstate

    def _append_history(self, hstate, partner_obs, memory_action):
        idx = hstate.cursor % self.history_len
        ego_obs_buffer = hstate.ego_obs_buffer.at[idx].set(hstate.last_ego_obs)
        ego_action_buffer = hstate.ego_action_buffer.at[idx].set(hstate.cached_self_action)
        partner_obs_buffer = hstate.partner_obs_buffer.at[idx].set(partner_obs)
        partner_action_buffer = hstate.partner_action_buffer.at[idx].set(memory_action)
        valid_mask = hstate.valid_mask.at[idx].set(True)
        return hstate.replace(
            ego_obs_buffer=ego_obs_buffer,
            ego_action_buffer=ego_action_buffer,
            partner_obs_buffer=partner_obs_buffer,
            partner_action_buffer=partner_action_buffer,
            valid_mask=valid_mask,
            cursor=(hstate.cursor + 1) % self.history_len,
            count=jnp.minimum(hstate.count + 1, self.history_len),
        )

    def _apply_batch(self, params, obs_batch, adapter_readout_scale=1.0):
        done_batch = jnp.zeros((obs_batch.shape[0], 1), dtype=jnp.bool_)
        ac_in = (
            obs_batch[:, None, ...],
            done_batch,
            jnp.asarray(adapter_readout_scale, dtype=jnp.float32),
        )
        _, pi, value, aux = _apply_policy_network(self.network, params, None, ac_in)
        return pi.logits[:, 0, :], value[:, 0], aux

    def _ttac_loss(self, params, hstate, use_kl):
        hist_logits, hist_value, hist_aux = self._apply_batch(
            params, hstate.partner_obs_buffer, 1.0
        )
        ego_logits, _, ego_aux = self._apply_batch(params, hstate.ego_obs_buffer, 1.0)
        hist_actions = hstate.partner_action_buffer.astype(jnp.int32)
        hist_log_probs = jax.nn.log_softmax(hist_logits, axis=-1)
        hist_ce = -hist_log_probs[jnp.arange(self.history_len), hist_actions]

        base_logits = jax.lax.stop_gradient(hist_aux["base_logits"][:, 0, :])
        base_log_probs = jax.nn.log_softmax(base_logits, axis=-1)
        base_probs = jax.nn.softmax(base_logits, axis=-1)
        action_onehot = jax.nn.one_hot(hist_actions, self.action_dim)
        beta = jnp.asarray(self.model_config.get("TTAC_TEST_PROJECT_BETA", 2.0), dtype=jnp.float32)
        projected_target = jax.nn.softmax(base_log_probs + beta * action_onehot, axis=-1)
        projected_ce = -jnp.sum(jax.lax.stop_gradient(projected_target) * hist_log_probs, axis=-1)

        base_action_prob = jnp.sum(base_probs * action_onehot, axis=-1)
        base_entropy = -jnp.sum(base_probs * base_log_probs, axis=-1)
        min_prob = jnp.asarray(self.model_config.get("TTAC_TEST_SUPPORT_MIN_PROB", 0.05), dtype=jnp.float32)
        max_entropy = jnp.asarray(self.model_config.get("TTAC_TEST_SUPPORT_MAX_ENTROPY", 1.5), dtype=jnp.float32)
        confidence_weight = jnp.where(
            (base_action_prob >= min_prob) & (base_entropy <= max_entropy),
            jnp.ones_like(base_action_prob),
            jnp.zeros_like(base_action_prob),
        )
        confidence_mask = hstate.valid_mask.astype(jnp.float32) * confidence_weight
        support_advantage = jnp.clip(
            (base_action_prob - min_prob) / jnp.maximum(1.0 - min_prob, 1e-6),
            0.0,
            1.0,
        )
        advantage_power = jnp.asarray(
            self.model_config.get("TTAC_TEST_ADVANTAGE_POWER", 1.0), dtype=jnp.float32
        )
        advantage_weight = jnp.power(support_advantage + 1e-6, advantage_power)
        advantage_mask = hstate.valid_mask.astype(jnp.float32) * advantage_weight

        def _partner_support_weight(actions):
            onehot = jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim)
            action_prob = jnp.sum(base_probs * onehot, axis=-1)
            support = jnp.clip(
                (action_prob - min_prob) / jnp.maximum(1.0 - min_prob, 1e-6),
                0.0,
                1.0,
            )
            return jnp.power(support + 1e-6, advantage_power)

        wrong_partner_weight = _partner_support_weight(hstate.ego_action_buffer)
        delayed_actions = jnp.concatenate([hist_actions[:1], hist_actions[:-1]], axis=0)
        delayed_partner_weight = _partner_support_weight(delayed_actions)
        random_actions = (hist_actions + hstate.ego_action_buffer.astype(jnp.int32) + 1) % self.action_dim
        random_partner_weight = _partner_support_weight(random_actions)
        corrupt_partner_weight = (
            wrong_partner_weight + delayed_partner_weight + random_partner_weight
        ) / 3.0

        ego_actions = hstate.ego_action_buffer.astype(jnp.int32)
        ego_log_probs = jax.nn.log_softmax(ego_logits, axis=-1)
        ego_base_logits = jax.lax.stop_gradient(ego_aux["base_logits"][:, 0, :])
        ego_base_log_probs = jax.nn.log_softmax(ego_base_logits, axis=-1)
        ego_base_probs = jax.nn.softmax(ego_base_logits, axis=-1)
        ego_action_onehot = jax.nn.one_hot(ego_actions, self.action_dim)
        ego_projected_target = jax.nn.softmax(
            ego_base_log_probs + beta * ego_action_onehot, axis=-1
        )
        ego_projected_ce = -jnp.sum(
            jax.lax.stop_gradient(ego_projected_target) * ego_log_probs, axis=-1
        )
        ego_base_action_prob = jnp.sum(ego_base_probs * ego_action_onehot, axis=-1)
        ego_support_advantage = jnp.clip(
            (ego_base_action_prob - min_prob) / jnp.maximum(1.0 - min_prob, 1e-6),
            0.0,
            1.0,
        )
        ego_advantage_weight = jnp.power(ego_support_advantage + 1e-6, advantage_power)
        # Partner history decides which interaction steps are adaptation-worthy;
        # the actual supervised action is ego's own action on ego's own obs.
        # This avoids turning partner modeling into direct partner-action imitation.
        ego_aw_mask = hstate.valid_mask.astype(jnp.float32) * advantage_weight * ego_advantage_weight
        contrast_beta = jnp.asarray(
            self.model_config.get("TTAC_TEST_CONTRAST_BETA", 1.0), dtype=jnp.float32
        )
        contrast_floor = jnp.asarray(
            self.model_config.get("TTAC_TEST_CONTRAST_FLOOR", 0.0), dtype=jnp.float32
        )
        semantic_weight = jnp.maximum(
            advantage_weight - contrast_beta * corrupt_partner_weight,
            contrast_floor,
        )
        contrastive_ego_aw_mask = (
            hstate.valid_mask.astype(jnp.float32)
            * semantic_weight
            * ego_advantage_weight
        )
        semantic_lambda = jnp.asarray(
            self.model_config.get("TTAC_TEST_SEMANTIC_LAMBDA", 0.25), dtype=jnp.float32
        )

        # Use the learned state value only as a conservative update gate: low-value
        # history states are treated as more adaptation-worthy, but updates remain
        # projected into the base policy support above.
        hist_value = jax.lax.stop_gradient(hist_value)
        value_center = _masked_mean(hist_value, hstate.valid_mask)
        value_temp = jnp.asarray(
            self.model_config.get("TTAC_TEST_VALUE_GATE_TEMP", 1.0), dtype=jnp.float32
        )
        low_value_weight = jax.nn.sigmoid(
            (value_center - hist_value) / jnp.maximum(value_temp, 1e-6)
        )
        value_gate_mask = confidence_mask * low_value_weight

        if self.eval_mode == "ttac_projected_history":
            agreement_loss = _masked_mean(projected_ce, hstate.valid_mask)
        elif self.eval_mode == "ttac_confident_history":
            agreement_loss = _masked_mean(hist_ce, confidence_mask)
        elif self.eval_mode == "ttac_projected_confident":
            agreement_loss = _masked_mean(projected_ce, confidence_mask)
        elif self.eval_mode == "ttac_advantage_weighted":
            agreement_loss = _masked_mean(projected_ce, advantage_mask)
        elif self.eval_mode in (
            "ttac_ego_advantage_weighted",
            "ttac_ego_aw_wrong_history",
            "ttac_ego_aw_random_history",
            "ttac_ego_aw_delayed_history",
        ):
            agreement_loss = _masked_mean(ego_projected_ce, ego_aw_mask)
        elif self.eval_mode in (
            "ttac_contrastive_ego_aw",
            "ttac_contrastive_ego_aw_wrong_history",
            "ttac_contrastive_ego_aw_random_history",
            "ttac_contrastive_ego_aw_delayed_history",
        ):
            agreement_loss = _masked_mean(ego_projected_ce, contrastive_ego_aw_mask)
        elif self.eval_mode in (
            "ttac_semantic_ego_aw",
            "ttac_semantic_ego_aw_wrong_history",
            "ttac_semantic_ego_aw_random_history",
            "ttac_semantic_ego_aw_delayed_history",
        ):
            agreement_loss = _masked_mean(
                ego_projected_ce, contrastive_ego_aw_mask
            ) + semantic_lambda * _masked_mean(projected_ce, advantage_mask)
        elif self.eval_mode == "ttac_value_gated_projected":
            agreement_loss = _masked_mean(projected_ce, value_gate_mask)
        else:
            agreement_loss = _masked_mean(hist_ce, hstate.valid_mask)

        cur_logits, _, cur_aux = self._apply_batch(
            params, hstate.last_ego_obs[None, ...], 1.0
        )
        hist_kl = _masked_mean(
            _categorical_symmetric_kl(hist_aux["base_logits"][:, 0, :], hist_logits),
            hstate.valid_mask,
        )
        ego_kl = _masked_mean(
            _categorical_symmetric_kl(ego_aux["base_logits"][:, 0, :], ego_logits),
            hstate.valid_mask,
        )
        cur_kl = _categorical_symmetric_kl(cur_aux["base_logits"][:, 0, :], cur_logits).mean()
        entropy = -jnp.sum(
            jax.nn.softmax(cur_logits, axis=-1) * jax.nn.log_softmax(cur_logits, axis=-1),
            axis=-1,
        ).mean()
        hist_coef = jnp.asarray(self.model_config.get("TTAC_TEST_HIST_KL_COEF", 0.05))
        ego_coef = jnp.asarray(self.model_config.get("TTAC_TEST_EGO_KL_COEF", 0.05))
        cur_coef = jnp.asarray(self.model_config.get("TTAC_TEST_CUR_KL_COEF", 0.05))
        entropy_coef = jnp.asarray(self.model_config.get("TTAC_TEST_ENTROPY_COEF", 0.0))
        kl_mult = jnp.asarray(use_kl, dtype=jnp.float32)
        loss = (
            agreement_loss
            + kl_mult * (hist_coef * hist_kl + ego_coef * ego_kl + cur_coef * cur_kl)
            - entropy_coef * entropy
        )
        aux = {
            "agreement_loss": agreement_loss,
            "hist_kl": hist_kl,
            "ego_kl": ego_kl,
            "cur_kl": cur_kl,
            "entropy": entropy,
        }
        return loss, aux

    @partial(jax.jit, static_argnums=(0,))
    def update_after_step(self, hstate, partner_obs, partner_action, done):
        if hstate is None:
            hstate = self.init_hstate(1)
        memory_action, true_partner_action = _resolve_history_action(
            self.eval_mode,
            jnp.asarray(partner_action),
            hstate.cached_self_action,
            hstate.prev_partner_action,
            self.action_dim,
        )
        hstate = self._append_history(
            hstate, jnp.asarray(partner_obs, dtype=jnp.float32), memory_action
        )
        should_update = jnp.asarray(
            self.eval_mode
            in (
                "ttac_true_history",
                "ttac_projected_history",
                "ttac_confident_history",
                "ttac_projected_confident",
                "ttac_advantage_weighted",
                "ttac_value_gated_projected",
                "ttac_ego_advantage_weighted",
                "ttac_ego_aw_wrong_history",
                "ttac_ego_aw_random_history",
                "ttac_ego_aw_delayed_history",
                "ttac_contrastive_ego_aw",
                "ttac_contrastive_ego_aw_wrong_history",
                "ttac_contrastive_ego_aw_random_history",
                "ttac_contrastive_ego_aw_delayed_history",
                "ttac_semantic_ego_aw",
                "ttac_semantic_ego_aw_wrong_history",
                "ttac_semantic_ego_aw_random_history",
                "ttac_semantic_ego_aw_delayed_history",
                "ttac_wrong_history",
                "ttac_random_history",
                "ttac_delayed_history",
                "ttac_no_kl",
            )
        )
        use_kl = self.eval_mode != "ttac_no_kl"
        lr = float(self.model_config.get("TTAC_TEST_LR", 0.001))
        steps = int(self.model_config.get("TTAC_TEST_UPDATE_STEPS", 1))

        def _one_update(params, _):
            (loss, _aux), grads = jax.value_and_grad(self._ttac_loss, has_aux=True)(
                params, hstate, use_kl
            )
            del loss
            grads = _mask_grads(grads, self.adapter_mask)
            updates = jax.tree_util.tree_map(lambda g: -lr * g, grads)
            return optax.apply_updates(params, updates), None

        updated_params, _ = jax.lax.scan(_one_update, hstate.params, None, steps)
        next_params = jax.lax.cond(
            should_update,
            lambda _: updated_params,
            lambda _: hstate.params,
            operand=None,
        )
        valid_mask = jnp.where(done, jnp.zeros_like(hstate.valid_mask), hstate.valid_mask)
        return hstate.replace(
            params=next_params,
            valid_mask=valid_mask,
            cursor=jnp.where(done, jnp.array(0, dtype=jnp.int32), hstate.cursor),
            count=jnp.where(done, jnp.array(0, dtype=jnp.int32), hstate.count),
            prev_partner_action=true_partner_action,
        )

    def forward_diagnostics(self, obs, done, hstate, params=None):
        if params is None:
            params = self.params
        ac_in = self._format_network_input(obs, done, self._adapter_readout_scale())
        next_hstate, pi, value, _ = _apply_policy_network(
            self.network, params, hstate, ac_in
        )
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
        ac_in = self._format_network_input_batch(obs_batch, done_batch, self._adapter_readout_scale())
        next_hstate, pi, value, _ = _apply_policy_network(self.network, params, hstate, ac_in)
        return pi.probs[0], value[0], next_hstate

    def init_hstate(self, batch_size, key=None):
        del key
        if batch_size != 1:
            return initialize_carry(self.config, batch_size)
        return TTACPolicyState(
            params=self.params,
            base_hstate=initialize_carry(self.config, batch_size),
            last_ego_obs=jnp.zeros(self.obs_shape, dtype=jnp.float32),
            ego_obs_buffer=jnp.zeros((self.history_len,) + self.obs_shape, dtype=jnp.float32),
            ego_action_buffer=jnp.zeros((self.history_len,), dtype=jnp.int32),
            partner_obs_buffer=jnp.zeros((self.history_len,) + self.obs_shape, dtype=jnp.float32),
            partner_action_buffer=jnp.zeros((self.history_len,), dtype=jnp.int32),
            valid_mask=jnp.zeros((self.history_len,), dtype=jnp.bool_),
            cursor=jnp.array(0, dtype=jnp.int32),
            count=jnp.array(0, dtype=jnp.int32),
            cached_self_action=jnp.array(0, dtype=jnp.int32),
            prev_partner_action=jnp.array(0, dtype=jnp.int32),
        )


def policy_checkoints_to_policy_pairing(
    checkpoints: PPOParams,
    config,
    stochastic: bool = True,
    eval_mode: str = "base_no_test_adapt",
):
    policies = []
    for checkpoint in checkpoints:
        policies.append(
            PPOPolicy(
                checkpoint.params,
                config,
                stochastic=stochastic,
                eval_mode=eval_mode,
            )
        )
    return PolicyPairing(*policies)
