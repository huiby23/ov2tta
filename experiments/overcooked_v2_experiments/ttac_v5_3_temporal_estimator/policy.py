from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
from flax import core, struct
from flax.traverse_util import flatten_dict, unflatten_dict
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

from overcooked_v2_experiments.eval.policy import (
    AbstractPolicy,
    FunctionalPolicyPairing,
    PolicyPairing,
)
from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.models.abstract import ActorCriticBase
from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.utils.surrogate_heads import (
    apply_q_eta,
    expected_joint_q,
)
from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.utils.agreement_heads import (
    apply_agreement_estimator,
)


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
    prev_memory_action: jnp.ndarray
    prev_memory_action_buffer: jnp.ndarray
    time_buffer: jnp.ndarray
    global_step: jnp.ndarray


@struct.dataclass
class TTACFunctionalPolicySpec:
    params: core.FrozenDict[str, Any]
    eval_mode: str = struct.field(pytree_node=False, default="base_no_test_adapt")
    stochastic: bool = struct.field(pytree_node=False, default=True)


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
    "ttac_v4_support_aw",
    "ttac_v4_support_aw_wrong_history",
    "ttac_v4_support_aw_random_history",
    "ttac_v4_support_aw_delayed_history",
    "ttac_v4_direct_q",
    "ttac_v4_direct_q_wrong_history",
    "ttac_v4_direct_q_random_history",
    "ttac_v4_direct_q_delayed_history",
    "ttac_v4_direct_q_support",
    "ttac_v4_direct_q_support_wrong_history",
    "ttac_v4_direct_q_support_random_history",
    "ttac_v4_direct_q_support_delayed_history",
    "ttac_v4_gated_semantic",
    "ttac_v4_gated_semantic_wrong_history",
    "ttac_v4_gated_semantic_random_history",
    "ttac_v4_gated_semantic_delayed_history",
    "ttac_v4_gated_margin",
    "ttac_v4_gated_margin_wrong_history",
    "ttac_v4_gated_margin_random_history",
    "ttac_v4_gated_margin_delayed_history",
    "ttac_v4_margin_only",
    "ttac_v4_margin_only_wrong_history",
    "ttac_v4_margin_only_random_history",
    "ttac_v4_margin_only_delayed_history",
    "ttac_v5_support_aw",
    "ttac_v5_support_aw_wrong_history",
    "ttac_v5_support_aw_random_history",
    "ttac_v5_support_aw_delayed_history",
    "ttac_v5_agreement",
    "ttac_v5_agreement_wrong_history",
    "ttac_v5_agreement_random_history",
    "ttac_v5_agreement_delayed_history",
    "ttac_v5_agreement_support",
    "ttac_v5_agreement_support_wrong_history",
    "ttac_v5_agreement_support_random_history",
    "ttac_v5_agreement_support_delayed_history",
    "ttac_v5_1_multiquery",
    "ttac_v5_1_multiquery_wrong_history",
    "ttac_v5_1_multiquery_random_history",
    "ttac_v5_1_multiquery_delayed_history",
    "ttac_v5_1_multiquery_support",
    "ttac_v5_1_multiquery_support_wrong_history",
    "ttac_v5_1_multiquery_support_random_history",
    "ttac_v5_1_multiquery_support_delayed_history",
    "ttac_v5_1_confident",
    "ttac_v5_1_confident_wrong_history",
    "ttac_v5_1_confident_random_history",
    "ttac_v5_1_confident_delayed_history",
    "ttac_v5_1_confident_support",
    "ttac_v5_1_confident_support_wrong_history",
    "ttac_v5_1_confident_support_random_history",
    "ttac_v5_1_confident_support_delayed_history",
    "ttac_v5_2_latest",
    "ttac_v5_2_latest_wrong_history",
    "ttac_v5_2_latest_random_history",
    "ttac_v5_2_latest_delayed_history",
    "ttac_v5_2_prev_query",
    "ttac_v5_2_oldest_query",
    "ttac_v5_2_pseudorandom_query",
    "ttac_v5_2_tv_gate",
    "ttac_v5_2_tv_gate_wrong_history",
    "ttac_v5_2_tv_gate_random_history",
    "ttac_v5_2_tv_gate_delayed_history",
    "ttac_v5_2_value_tv_gate",
    "ttac_v5_2_value_tv_gate_wrong_history",
    "ttac_v5_2_value_tv_gate_random_history",
    "ttac_v5_2_value_tv_gate_delayed_history",
    "ttac_v5_2_change_tv_gate",
    "ttac_v5_2_change_tv_gate_wrong_history",
    "ttac_v5_2_change_tv_gate_random_history",
    "ttac_v5_2_change_tv_gate_delayed_history",
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
    ) or eval_mode.endswith("_wrong_history"):
        return self_action, partner_action
    if eval_mode in (
        "ttac_random_history",
        "ttac_ego_aw_random_history",
        "ttac_contrastive_ego_aw_random_history",
        "ttac_semantic_ego_aw_random_history",
    ) or eval_mode.endswith("_random_history"):
        return (partner_action + self_action + 1) % action_dim, partner_action
    if eval_mode in (
        "ttac_delayed_history",
        "ttac_ego_aw_delayed_history",
        "ttac_contrastive_ego_aw_delayed_history",
        "ttac_semantic_ego_aw_delayed_history",
    ) or eval_mode.endswith("_delayed_history"):
        return prev_partner_action, partner_action
    return partner_action, partner_action


def _select_v5_2_query_pos(eval_mode, history_len, count, global_step):
    latest_pos = jnp.asarray(history_len - 1, dtype=jnp.int32)
    valid_count = jnp.maximum(count.astype(jnp.int32), 1)
    earliest_pos = jnp.asarray(history_len, dtype=jnp.int32) - valid_count
    if eval_mode == "ttac_v5_2_prev_query":
        return jnp.maximum(latest_pos - 1, earliest_pos)
    if eval_mode == "ttac_v5_2_oldest_query":
        return earliest_pos
    if eval_mode == "ttac_v5_2_pseudorandom_query":
        offset = (global_step.astype(jnp.int32) * 9973 + 17) % valid_count
        return earliest_pos + offset
    return latest_pos


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
        memory_action = jnp.asarray(memory_action, dtype=jnp.int32)
        ego_obs_buffer = hstate.ego_obs_buffer.at[idx].set(hstate.last_ego_obs)
        ego_action_buffer = hstate.ego_action_buffer.at[idx].set(hstate.cached_self_action)
        partner_obs_buffer = hstate.partner_obs_buffer.at[idx].set(partner_obs)
        partner_action_buffer = hstate.partner_action_buffer.at[idx].set(memory_action)
        prev_memory_action_buffer = hstate.prev_memory_action_buffer.at[idx].set(
            hstate.prev_memory_action
        )
        time_buffer = hstate.time_buffer.at[idx].set(hstate.global_step)
        valid_mask = hstate.valid_mask.at[idx].set(True)
        return hstate.replace(
            ego_obs_buffer=ego_obs_buffer,
            ego_action_buffer=ego_action_buffer,
            partner_obs_buffer=partner_obs_buffer,
            partner_action_buffer=partner_action_buffer,
            prev_memory_action_buffer=prev_memory_action_buffer,
            time_buffer=time_buffer,
            valid_mask=valid_mask,
            cursor=(hstate.cursor + 1) % self.history_len,
            count=jnp.minimum(hstate.count + 1, self.history_len),
            prev_memory_action=memory_action,
            global_step=hstate.global_step + jnp.array(1, dtype=jnp.int32),
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
        ego_logits, ego_value, ego_aux = self._apply_batch(params, hstate.ego_obs_buffer, 1.0)
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

        def _projected_ce_for(actions):
            onehot = jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim)
            target = jax.nn.softmax(base_log_probs + beta * onehot, axis=-1)
            return -jnp.sum(jax.lax.stop_gradient(target) * hist_log_probs, axis=-1)

        wrong_projected_ce = _projected_ce_for(hstate.ego_action_buffer)
        delayed_projected_ce = _projected_ce_for(hstate.prev_memory_action_buffer)
        random_projected_ce = _projected_ce_for(random_actions)

        age = jnp.maximum(
            hstate.global_step - hstate.time_buffer - jnp.array(1, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        ).astype(jnp.float32)
        recency_tau = jnp.asarray(
            self.model_config.get("TTAC_V3_RECENCY_TAU", 10.0), dtype=jnp.float32
        )
        recency_weight = jnp.exp(-age / jnp.maximum(recency_tau, 1e-6))
        change_floor = jnp.asarray(
            self.model_config.get("TTAC_V3_CHANGE_GATE_FLOOR", 0.1), dtype=jnp.float32
        )
        use_change_gate = bool(self.model_config.get("TTAC_V3_USE_CHANGE_GATE", True))
        changed = (hist_actions != hstate.prev_memory_action_buffer).astype(jnp.float32)
        change_weight = jnp.where(changed > 0.0, 1.0, change_floor)
        if not use_change_gate:
            change_weight = jnp.ones_like(change_weight)
        differs_from_corrupt = (
            (hist_actions != hstate.ego_action_buffer.astype(jnp.int32))
            | (hist_actions != hstate.prev_memory_action_buffer.astype(jnp.int32))
        ).astype(jnp.float32)
        semantic_gate = (
            hstate.valid_mask.astype(jnp.float32)
            * recency_weight
            * change_weight
            * differs_from_corrupt
        )
        v3_semantic_coef = jnp.asarray(
            self.model_config.get("TTAC_V3_SEMANTIC_COEF", 0.25), dtype=jnp.float32
        )
        v3_margin_coef = jnp.asarray(
            self.model_config.get("TTAC_V3_MARGIN_COEF", 0.1), dtype=jnp.float32
        )
        v3_margin = jnp.asarray(
            self.model_config.get("TTAC_V3_MARGIN", 0.1), dtype=jnp.float32
        )
        v3_support_aw_loss = _masked_mean(ego_projected_ce, ego_aw_mask)
        v3_semantic_loss = _masked_mean(projected_ce, semantic_gate)
        v3_margin_terms = (
            jnp.maximum(0.0, v3_margin + projected_ce - wrong_projected_ce)
            + jnp.maximum(0.0, v3_margin + projected_ce - delayed_projected_ce)
            + jnp.maximum(0.0, v3_margin + projected_ce - random_projected_ce)
        )
        v3_margin_loss = _masked_mean(v3_margin_terms, semantic_gate)

        surrogate_params = self.model_config.get("TTAC_V4_SURROGATE", None)
        if surrogate_params is None:
            direct_q_loss = jnp.array(0.0, dtype=jnp.float32)
            direct_q_mean = jnp.array(0.0, dtype=jnp.float32)
            direct_q_entropy = jnp.array(0.0, dtype=jnp.float32)
        else:
            latest_idx = (hstate.cursor - 1) % self.history_len
            order = (hstate.cursor - self.history_len + jnp.arange(self.history_len)) % self.history_len
            chron_actions = hstate.partner_action_buffer[order]
            valid_chron = jnp.arange(self.history_len) >= (self.history_len - hstate.count)
            # Predict the next partner action from the history before the latest observed action.
            prev_hist_mask = valid_chron & (jnp.arange(self.history_len) < (self.history_len - 1))
            q_action_history = jnp.where(prev_hist_mask, chron_actions, 0)[None, :]
            latest_partner_obs = hstate.partner_obs_buffer[latest_idx][None, ...]
            latest_ego_obs = hstate.ego_obs_buffer[latest_idx][None, ...]
            latest_ego_logits = ego_logits[latest_idx][None, :]
            q_logits = apply_q_eta(
                surrogate_params,
                latest_partner_obs,
                q_action_history,
                self.action_dim,
            )
            q_probs = jax.nn.softmax(q_logits, axis=-1)
            direct_q_values = expected_joint_q(
                surrogate_params,
                latest_ego_obs,
                latest_partner_obs,
                latest_ego_logits,
                q_probs,
                self.action_dim,
            )
            direct_q_loss = -direct_q_values.mean()
            direct_q_mean = direct_q_values.mean()
            direct_q_entropy = (
                -jnp.sum(q_probs * jax.nn.log_softmax(q_logits, axis=-1), axis=-1)
            ).mean()
        v4_direct_coef = jnp.asarray(
            self.model_config.get("TTAC_V4_DIRECT_Q_COEF", 1.0), dtype=jnp.float32
        )
        v4_support_coef = jnp.asarray(
            self.model_config.get("TTAC_V4_SUPPORT_COEF", 1.0), dtype=jnp.float32
        )

        agreement_estimator_params = self.model_config.get("TTAC_V5_ESTIMATOR", None)
        if agreement_estimator_params is None:
            v5_agreement_loss = jnp.array(0.0, dtype=jnp.float32)
            v5_multiquery_loss = jnp.array(0.0, dtype=jnp.float32)
            v5_confident_loss = jnp.array(0.0, dtype=jnp.float32)
            v5_estimator_kl = jnp.array(0.0, dtype=jnp.float32)
            v5_estimator_tv = jnp.array(0.0, dtype=jnp.float32)
            v5_estimator_entropy = jnp.array(0.0, dtype=jnp.float32)
            v5_confidence_mass = jnp.array(0.0, dtype=jnp.float32)
            v5_target_base_tv = jnp.array(0.0, dtype=jnp.float32)
            v5_2_latest_target_base_tv = jnp.array(0.0, dtype=jnp.float32)
            v5_2_latest_entropy = jnp.array(0.0, dtype=jnp.float32)
            v5_2_latest_value = jnp.array(0.0, dtype=jnp.float32)
            v5_2_tv_gate = jnp.array(0.0, dtype=jnp.float32)
            v5_2_value_tv_gate = jnp.array(0.0, dtype=jnp.float32)
            v5_2_change_tv_gate = jnp.array(0.0, dtype=jnp.float32)
        else:
            latest_idx = (hstate.cursor - 1) % self.history_len
            order = (hstate.cursor - self.history_len + jnp.arange(self.history_len)) % self.history_len
            chron_actions = hstate.partner_action_buffer[order]
            valid_chron = jnp.arange(self.history_len) >= (self.history_len - hstate.count)
            q_action_history = jnp.where(valid_chron, chron_actions, 0)[None, :]

            query_obs = hstate.ego_obs_buffer[order]
            query_partner_obs = hstate.partner_obs_buffer[order]
            query_ego_logits = ego_logits[order]
            query_base_logits = ego_aux["base_logits"][order, 0, :]
            repeated_history = jnp.repeat(q_action_history, self.history_len, axis=0)
            chron_partner_obs = hstate.partner_obs_buffer[order]
            repeated_partner_obs_history = jnp.repeat(
                chron_partner_obs[None, ...], self.history_len, axis=0
            )
            estimator_logits = apply_agreement_estimator(
                agreement_estimator_params,
                query_obs,
                query_partner_obs,
                repeated_history,
                self.action_dim,
                partner_obs_history=repeated_partner_obs_history,
            )
            estimator_probs = jax.lax.stop_gradient(jax.nn.softmax(estimator_logits, axis=-1))
            estimator_log_probs = jnp.log(jnp.maximum(estimator_probs, 1e-6))
            query_ego_log_probs = jax.nn.log_softmax(query_ego_logits, axis=-1)
            query_ego_probs = jax.nn.softmax(query_ego_logits, axis=-1)
            query_base_probs = jax.nn.softmax(query_base_logits, axis=-1)
            per_query_ce = -jnp.sum(estimator_probs * query_ego_log_probs, axis=-1)
            per_query_kl = jnp.sum(
                estimator_probs * (estimator_log_probs - query_ego_log_probs), axis=-1
            )
            per_query_tv = 0.5 * jnp.sum(jnp.abs(estimator_probs - query_ego_probs), axis=-1)
            per_query_entropy = -jnp.sum(estimator_probs * estimator_log_probs, axis=-1)
            per_query_target_base_tv = 0.5 * jnp.sum(
                jnp.abs(estimator_probs - query_base_probs), axis=-1
            )
            valid_query_mask = valid_chron.astype(jnp.float32)
            v5_multiquery_loss = _masked_mean(per_query_ce, valid_query_mask)
            v5_estimator_kl = _masked_mean(per_query_kl, valid_query_mask)
            v5_estimator_tv = _masked_mean(per_query_tv, valid_query_mask)
            v5_estimator_entropy = _masked_mean(per_query_entropy, valid_query_mask)
            v5_target_base_tv = _masked_mean(per_query_target_base_tv, valid_query_mask)

            latest_pos = jnp.asarray(self.history_len - 1, dtype=jnp.int32)
            v5_2_query_pos = _select_v5_2_query_pos(
                self.eval_mode, self.history_len, hstate.count, hstate.global_step
            )
            v5_agreement_loss = jnp.take(per_query_ce, v5_2_query_pos, axis=0)
            query_ego_value = jax.lax.stop_gradient(ego_value[order])
            value_center_v5 = _masked_mean(query_ego_value, valid_query_mask)
            v5_2_latest_target_base_tv = jnp.take(
                per_query_target_base_tv, v5_2_query_pos, axis=0
            )
            v5_2_latest_entropy = jnp.take(per_query_entropy, v5_2_query_pos, axis=0)
            v5_2_latest_value = jnp.take(query_ego_value, v5_2_query_pos, axis=0)
            v5_2_tv_threshold = jnp.asarray(
                self.model_config.get("TTAC_V5_2_TV_THRESHOLD", 0.05), dtype=jnp.float32
            )
            latest_valid = jnp.take(valid_query_mask, v5_2_query_pos, axis=0)
            v5_2_tv_gate = latest_valid * (
                v5_2_latest_target_base_tv >= v5_2_tv_threshold
            ).astype(jnp.float32)
            v5_2_value_tv_gate = v5_2_tv_gate * (
                v5_2_latest_value <= value_center_v5
            ).astype(jnp.float32)
            prev_chron_actions = hstate.prev_memory_action_buffer[order].astype(jnp.int32)
            v5_2_change_tv_gate = v5_2_tv_gate * (
                jnp.take(chron_actions, v5_2_query_pos, axis=0)
                != jnp.take(prev_chron_actions, v5_2_query_pos, axis=0)
            ).astype(jnp.float32)

            conf_max_entropy = jnp.asarray(
                self.model_config.get("TTAC_V5_CONF_MAX_ENTROPY", 1.25), dtype=jnp.float32
            )
            conf_min_tv = jnp.asarray(
                self.model_config.get("TTAC_V5_CONF_MIN_TARGET_BASE_TV", 0.03), dtype=jnp.float32
            )
            entropy_gate = (per_query_entropy <= conf_max_entropy).astype(jnp.float32)
            tv_gate = (per_query_target_base_tv >= conf_min_tv).astype(jnp.float32)
            confidence_mask_v5 = valid_query_mask * entropy_gate * tv_gate
            v5_confident_loss = _masked_mean(per_query_ce, confidence_mask_v5)
            v5_confidence_mass = jnp.mean(confidence_mask_v5)
        v5_agreement_coef = jnp.asarray(
            self.model_config.get("TTAC_V5_AGREEMENT_COEF", 1.0), dtype=jnp.float32
        )
        v5_support_coef = jnp.asarray(
            self.model_config.get("TTAC_V5_SUPPORT_COEF", 1.0), dtype=jnp.float32
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

        mode_loss_gate = jnp.array(1.0, dtype=jnp.float32)
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
        elif self.eval_mode in (
            "ttac_v4_support_aw",
            "ttac_v4_support_aw_wrong_history",
            "ttac_v4_support_aw_random_history",
            "ttac_v4_support_aw_delayed_history",
        ):
            agreement_loss = v3_support_aw_loss
        elif self.eval_mode in (
            "ttac_v4_gated_semantic",
            "ttac_v4_gated_semantic_wrong_history",
            "ttac_v4_gated_semantic_random_history",
            "ttac_v4_gated_semantic_delayed_history",
        ):
            agreement_loss = v3_support_aw_loss + v3_semantic_coef * v3_semantic_loss
        elif self.eval_mode in (
            "ttac_v4_gated_margin",
            "ttac_v4_gated_margin_wrong_history",
            "ttac_v4_gated_margin_random_history",
            "ttac_v4_gated_margin_delayed_history",
        ):
            agreement_loss = (
                v3_support_aw_loss
                + v3_semantic_coef * v3_semantic_loss
                + v3_margin_coef * v3_margin_loss
            )
        elif self.eval_mode in (
            "ttac_v4_margin_only",
            "ttac_v4_margin_only_wrong_history",
            "ttac_v4_margin_only_random_history",
            "ttac_v4_margin_only_delayed_history",
        ):
            agreement_loss = v3_margin_coef * v3_margin_loss
        elif self.eval_mode in (
            "ttac_v4_direct_q",
            "ttac_v4_direct_q_wrong_history",
            "ttac_v4_direct_q_random_history",
            "ttac_v4_direct_q_delayed_history",
        ):
            agreement_loss = v4_direct_coef * direct_q_loss
        elif self.eval_mode in (
            "ttac_v4_direct_q_support",
            "ttac_v4_direct_q_support_wrong_history",
            "ttac_v4_direct_q_support_random_history",
            "ttac_v4_direct_q_support_delayed_history",
        ):
            agreement_loss = v4_direct_coef * direct_q_loss + v4_support_coef * v3_support_aw_loss
        elif self.eval_mode in (
            "ttac_v5_support_aw",
            "ttac_v5_support_aw_wrong_history",
            "ttac_v5_support_aw_random_history",
            "ttac_v5_support_aw_delayed_history",
        ):
            agreement_loss = v3_support_aw_loss
        elif self.eval_mode in (
            "ttac_v5_agreement",
            "ttac_v5_agreement_wrong_history",
            "ttac_v5_agreement_random_history",
            "ttac_v5_agreement_delayed_history",
        ):
            agreement_loss = v5_agreement_coef * v5_agreement_loss
        elif self.eval_mode in (
            "ttac_v5_agreement_support",
            "ttac_v5_agreement_support_wrong_history",
            "ttac_v5_agreement_support_random_history",
            "ttac_v5_agreement_support_delayed_history",
        ):
            agreement_loss = v5_agreement_coef * v5_agreement_loss + v5_support_coef * v3_support_aw_loss
        elif self.eval_mode in (
            "ttac_v5_1_multiquery",
            "ttac_v5_1_multiquery_wrong_history",
            "ttac_v5_1_multiquery_random_history",
            "ttac_v5_1_multiquery_delayed_history",
        ):
            agreement_loss = v5_agreement_coef * v5_multiquery_loss
        elif self.eval_mode in (
            "ttac_v5_1_multiquery_support",
            "ttac_v5_1_multiquery_support_wrong_history",
            "ttac_v5_1_multiquery_support_random_history",
            "ttac_v5_1_multiquery_support_delayed_history",
        ):
            agreement_loss = v5_agreement_coef * v5_multiquery_loss + v5_support_coef * v3_support_aw_loss
        elif self.eval_mode in (
            "ttac_v5_1_confident",
            "ttac_v5_1_confident_wrong_history",
            "ttac_v5_1_confident_random_history",
            "ttac_v5_1_confident_delayed_history",
        ):
            agreement_loss = v5_agreement_coef * v5_confident_loss
        elif self.eval_mode in (
            "ttac_v5_1_confident_support",
            "ttac_v5_1_confident_support_wrong_history",
            "ttac_v5_1_confident_support_random_history",
            "ttac_v5_1_confident_support_delayed_history",
        ):
            agreement_loss = v5_agreement_coef * v5_confident_loss + v5_support_coef * v3_support_aw_loss
        elif self.eval_mode in (
            "ttac_v5_2_latest",
            "ttac_v5_2_latest_wrong_history",
            "ttac_v5_2_latest_random_history",
            "ttac_v5_2_latest_delayed_history",
            "ttac_v5_2_prev_query",
            "ttac_v5_2_oldest_query",
            "ttac_v5_2_pseudorandom_query",
        ):
            agreement_loss = v5_agreement_coef * v5_agreement_loss
        elif self.eval_mode in (
            "ttac_v5_2_tv_gate",
            "ttac_v5_2_tv_gate_wrong_history",
            "ttac_v5_2_tv_gate_random_history",
            "ttac_v5_2_tv_gate_delayed_history",
        ):
            mode_loss_gate = v5_2_tv_gate
            agreement_loss = v5_agreement_coef * v5_agreement_loss
        elif self.eval_mode in (
            "ttac_v5_2_value_tv_gate",
            "ttac_v5_2_value_tv_gate_wrong_history",
            "ttac_v5_2_value_tv_gate_random_history",
            "ttac_v5_2_value_tv_gate_delayed_history",
        ):
            mode_loss_gate = v5_2_value_tv_gate
            agreement_loss = v5_agreement_coef * v5_agreement_loss
        elif self.eval_mode in (
            "ttac_v5_2_change_tv_gate",
            "ttac_v5_2_change_tv_gate_wrong_history",
            "ttac_v5_2_change_tv_gate_random_history",
            "ttac_v5_2_change_tv_gate_delayed_history",
        ):
            mode_loss_gate = v5_2_change_tv_gate
            agreement_loss = v5_agreement_coef * v5_agreement_loss
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
        raw_loss = (
            agreement_loss
            + kl_mult * (hist_coef * hist_kl + ego_coef * ego_kl + cur_coef * cur_kl)
            - entropy_coef * entropy
        )
        loss = mode_loss_gate * raw_loss
        aux = {
            "agreement_loss": agreement_loss,
            "hist_kl": hist_kl,
            "ego_kl": ego_kl,
            "cur_kl": cur_kl,
            "entropy": entropy,
            "v3_semantic_loss": v3_semantic_loss,
            "v3_margin_loss": v3_margin_loss,
            "v3_gate_mass": jnp.mean(semantic_gate),
            "v4_direct_q_loss": direct_q_loss,
            "v4_direct_q_mean": direct_q_mean,
            "v4_partner_q_entropy": direct_q_entropy,
            "v5_agreement_loss": v5_agreement_loss,
            "v5_multiquery_loss": v5_multiquery_loss,
            "v5_confident_loss": v5_confident_loss,
            "v5_estimator_kl": v5_estimator_kl,
            "v5_estimator_tv": v5_estimator_tv,
            "v5_estimator_entropy": v5_estimator_entropy,
            "v5_confidence_mass": v5_confidence_mass,
            "v5_target_base_tv": v5_target_base_tv,
            "v5_2_latest_target_base_tv": v5_2_latest_target_base_tv,
            "v5_2_latest_entropy": v5_2_latest_entropy,
            "v5_2_latest_value": v5_2_latest_value,
            "v5_2_tv_gate": v5_2_tv_gate,
            "v5_2_value_tv_gate": v5_2_value_tv_gate,
            "v5_2_change_tv_gate": v5_2_change_tv_gate,
            "v5_2_mode_loss_gate": mode_loss_gate,
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
                "ttac_v4_support_aw",
                "ttac_v4_support_aw_wrong_history",
                "ttac_v4_support_aw_random_history",
                "ttac_v4_support_aw_delayed_history",
                "ttac_v4_direct_q",
                "ttac_v4_direct_q_wrong_history",
                "ttac_v4_direct_q_random_history",
                "ttac_v4_direct_q_delayed_history",
                "ttac_v4_direct_q_support",
                "ttac_v4_direct_q_support_wrong_history",
                "ttac_v4_direct_q_support_random_history",
                "ttac_v4_direct_q_support_delayed_history",
                "ttac_v4_gated_semantic",
                "ttac_v4_gated_semantic_wrong_history",
                "ttac_v4_gated_semantic_random_history",
                "ttac_v4_gated_semantic_delayed_history",
                "ttac_v4_gated_margin",
                "ttac_v4_gated_margin_wrong_history",
                "ttac_v4_gated_margin_random_history",
                "ttac_v4_gated_margin_delayed_history",
                "ttac_v4_margin_only",
                "ttac_v4_margin_only_wrong_history",
                "ttac_v4_margin_only_random_history",
                "ttac_v4_margin_only_delayed_history",
                "ttac_wrong_history",
                "ttac_random_history",
                "ttac_delayed_history",
                "ttac_no_kl",
            )
            or self.eval_mode.startswith("ttac_v5_")
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
            global_step=jnp.where(done, jnp.array(0, dtype=jnp.int32), hstate.global_step),
            prev_partner_action=true_partner_action,
            prev_memory_action=jnp.where(
                done, jnp.array(0, dtype=jnp.int32), hstate.prev_memory_action
            ),
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
            prev_memory_action=jnp.array(0, dtype=jnp.int32),
            prev_memory_action_buffer=jnp.zeros((self.history_len,), dtype=jnp.int32),
            time_buffer=jnp.zeros((self.history_len,), dtype=jnp.int32),
            global_step=jnp.array(0, dtype=jnp.int32),
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


def policy_checkoints_to_functional_policy_pairing(
    checkpoints: PPOParams,
    config,
    stochastic: bool = True,
    eval_mode: str = "base_no_test_adapt",
):
    policies = []
    for checkpoint in checkpoints:
        params = (
            checkpoint.params
            if isinstance(checkpoint.params, core.FrozenDict)
            else core.freeze(checkpoint.params)
        )
        policies.append(
            TTACFunctionalPolicySpec(
                params=params,
                eval_mode=eval_mode,
                stochastic=stochastic,
            )
        )
    return FunctionalPolicyPairing(
        tuple(policies),
        backend="ttac_v5_3_temporal_estimator",
        config=config,
    )


def _functional_init_ttac_state(spec, config, obs_shape, history_len):
    return TTACPolicyState(
        params=spec.params,
        base_hstate=initialize_carry(config, 1),
        last_ego_obs=jnp.zeros(obs_shape, dtype=jnp.float32),
        ego_obs_buffer=jnp.zeros((history_len,) + obs_shape, dtype=jnp.float32),
        ego_action_buffer=jnp.zeros((history_len,), dtype=jnp.int32),
        partner_obs_buffer=jnp.zeros((history_len,) + obs_shape, dtype=jnp.float32),
        partner_action_buffer=jnp.zeros((history_len,), dtype=jnp.int32),
        valid_mask=jnp.zeros((history_len,), dtype=jnp.bool_),
        cursor=jnp.array(0, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
        cached_self_action=jnp.array(0, dtype=jnp.int32),
        prev_partner_action=jnp.array(0, dtype=jnp.int32),
        prev_memory_action=jnp.array(0, dtype=jnp.int32),
        prev_memory_action_buffer=jnp.zeros((history_len,), dtype=jnp.int32),
        time_buffer=jnp.zeros((history_len,), dtype=jnp.int32),
        global_step=jnp.array(0, dtype=jnp.int32),
    )


def get_functional_rollout(
    policies: FunctionalPolicyPairing, env, key, reward_only: bool = False
):
    """Functional TTAC rollout for fast evaluation.

    This backend intentionally supports the current v5.2 evaluation line first:
    base/no-adapt, latest-query agreement, and TV-gated agreement. It avoids the
    Python PolicyPairing/PPOPolicy object path inside rollout while preserving the
    same network, history buffers, adapter-only updates, and policy-level KL terms.
    """

    config = policies.config
    model_config = config["model"]
    network = get_actor_critic(config)
    history_len = int(model_config.get("TTAC_HISTORY_LEN", 50))
    action_dim = int(env.action_space(env.agents[0]).n)
    obs_shape = tuple(env.observation_space().shape)
    hist_kl_coef = float(model_config.get("TTAC_TEST_HIST_KL_COEF", 0.05))
    ego_kl_coef = float(model_config.get("TTAC_TEST_EGO_KL_COEF", 0.05))
    cur_kl_coef = float(model_config.get("TTAC_TEST_CUR_KL_COEF", 0.05))

    supported_modes = {
        "base_no_test_adapt",
        "ttac_v5_2_latest",
        "ttac_v5_2_prev_query",
        "ttac_v5_2_oldest_query",
        "ttac_v5_2_pseudorandom_query",
        "ttac_v5_2_tv_gate",
    }
    for spec in policies.policies:
        if spec.eval_mode not in supported_modes:
            raise ValueError(
                "Functional TTAC v5.2 eval currently supports only "
                f"{sorted(supported_modes)}, got {spec.eval_mode}"
            )

    adapter_masks = tuple(_make_adapter_mask(spec.params) for spec in policies.policies)

    init_hstate = {
        f"agent_{i}": _functional_init_ttac_state(
            spec, config, obs_shape, history_len
        )
        for i, spec in enumerate(policies.policies)
    }

    def _apply_batch_fn(params, obs_batch, adapter_readout_scale=1.0):
        done_batch = jnp.zeros((obs_batch.shape[0], 1), dtype=jnp.bool_)
        ac_in = (
            obs_batch[:, None, ...],
            done_batch,
            jnp.asarray(adapter_readout_scale, dtype=jnp.float32),
        )
        _, pi, value, aux = _apply_policy_network(network, params, None, ac_in)
        return pi.logits[:, 0, :], value[:, 0], aux

    def _compute_action_for_policy(spec, state, obs, done, sample_key):
        obs_f = jnp.asarray(obs, dtype=jnp.float32)
        ac_in = (
            obs_f[jnp.newaxis, jnp.newaxis, ...],
            jnp.asarray(done)[jnp.newaxis, jnp.newaxis],
            jnp.asarray(
                0.0 if spec.eval_mode == "ttac_adapter_off" else 1.0,
                dtype=jnp.float32,
            ),
        )
        next_base_hstate, pi, _, _ = _apply_policy_network(
            network, state.params, state.base_hstate, ac_in
        )
        if spec.stochastic:
            action = pi.sample(seed=sample_key)[0, 0]
        else:
            action = jnp.argmax(pi.probs, axis=-1)[0, 0]
        next_state = state.replace(
            base_hstate=next_base_hstate,
            last_ego_obs=obs_f,
            cached_self_action=action,
        )
        return action, next_state

    def _append_history_fn(state, partner_obs, memory_action):
        idx = state.cursor % history_len
        memory_action = jnp.asarray(memory_action, dtype=jnp.int32)
        return state.replace(
            ego_obs_buffer=state.ego_obs_buffer.at[idx].set(state.last_ego_obs),
            ego_action_buffer=state.ego_action_buffer.at[idx].set(
                state.cached_self_action
            ),
            partner_obs_buffer=state.partner_obs_buffer.at[idx].set(
                jnp.asarray(partner_obs, dtype=jnp.float32)
            ),
            partner_action_buffer=state.partner_action_buffer.at[idx].set(memory_action),
            prev_memory_action_buffer=state.prev_memory_action_buffer.at[idx].set(
                state.prev_memory_action
            ),
            time_buffer=state.time_buffer.at[idx].set(state.global_step),
            valid_mask=state.valid_mask.at[idx].set(True),
            cursor=(state.cursor + 1) % history_len,
            count=jnp.minimum(state.count + 1, history_len),
            prev_memory_action=memory_action,
            global_step=state.global_step + jnp.array(1, dtype=jnp.int32),
        )

    def _ttac_v5_2_loss(params, state, eval_mode):
        agreement_estimator_params = model_config.get("TTAC_V5_ESTIMATOR", None)
        if agreement_estimator_params is None:
            agreement_loss = jnp.array(0.0, dtype=jnp.float32)
            mode_loss_gate = jnp.array(1.0, dtype=jnp.float32)
        else:
            order = (
                state.cursor - history_len + jnp.arange(history_len)
            ) % history_len
            chron_actions = state.partner_action_buffer[order]
            valid_chron = jnp.arange(history_len) >= (history_len - state.count)
            q_action_history = jnp.where(valid_chron, chron_actions, 0)[None, :]
            query_pos = _select_v5_2_query_pos(
                eval_mode, history_len, state.count, state.global_step
            )
            selected_obs = jnp.take(state.ego_obs_buffer[order], query_pos, axis=0)
            selected_partner_obs = jnp.take(
                state.partner_obs_buffer[order], query_pos, axis=0
            )
            selected_logits, _, selected_aux = _apply_batch_fn(
                params, selected_obs[None, ...], 1.0
            )
            estimator_logits = apply_agreement_estimator(
                agreement_estimator_params,
                selected_obs[None, ...],
                selected_partner_obs[None, ...],
                q_action_history,
                action_dim,
            )
            estimator_probs = jax.lax.stop_gradient(
                jax.nn.softmax(estimator_logits, axis=-1)
            )
            selected_log_probs = jax.nn.log_softmax(selected_logits, axis=-1)
            selected_base_probs = jax.nn.softmax(
                selected_aux["base_logits"][:, 0, :], axis=-1
            )
            selected_ce = -jnp.sum(estimator_probs * selected_log_probs, axis=-1)[0]
            selected_tv = 0.5 * jnp.sum(
                jnp.abs(estimator_probs - selected_base_probs), axis=-1
            )[0]
            selected_valid = jnp.take(
                valid_chron.astype(jnp.float32), query_pos, axis=0
            )
            threshold = jnp.asarray(
                model_config.get("TTAC_V5_2_TV_THRESHOLD", 0.05), dtype=jnp.float32
            )
            tv_gate = selected_valid * (selected_tv >= threshold).astype(jnp.float32)
            if eval_mode == "ttac_v5_2_tv_gate":
                mode_loss_gate = tv_gate
            else:
                mode_loss_gate = jnp.array(1.0, dtype=jnp.float32)
            agreement_loss = (
                jnp.asarray(
                    model_config.get("TTAC_V5_AGREEMENT_COEF", 1.0),
                    dtype=jnp.float32,
                )
                * selected_ce
            )

        raw_loss = agreement_loss
        if hist_kl_coef != 0.0:
            hist_logits, _, hist_aux = _apply_batch_fn(
                params, state.partner_obs_buffer, 1.0
            )
            hist_kl = _masked_mean(
                _categorical_symmetric_kl(
                    hist_aux["base_logits"][:, 0, :], hist_logits
                ),
                state.valid_mask,
            )
            raw_loss = raw_loss + jnp.asarray(hist_kl_coef) * hist_kl
        if ego_kl_coef != 0.0:
            ego_logits, _, ego_aux = _apply_batch_fn(
                params, state.ego_obs_buffer, 1.0
            )
            ego_kl = _masked_mean(
                _categorical_symmetric_kl(
                    ego_aux["base_logits"][:, 0, :], ego_logits
                ),
                state.valid_mask,
            )
            raw_loss = raw_loss + jnp.asarray(ego_kl_coef) * ego_kl
        if cur_kl_coef != 0.0:
            cur_logits, _, cur_aux = _apply_batch_fn(
                params, state.last_ego_obs[None, ...], 1.0
            )
            cur_kl = _categorical_symmetric_kl(
                cur_aux["base_logits"][:, 0, :], cur_logits
            ).mean()
            raw_loss = raw_loss + jnp.asarray(cur_kl_coef) * cur_kl
        return mode_loss_gate * raw_loss

    def _update_policy_state(spec, adapter_mask, state, partner_obs, partner_action, done):
        memory_action, true_partner_action = _resolve_history_action(
            spec.eval_mode,
            jnp.asarray(partner_action),
            state.cached_self_action,
            state.prev_partner_action,
            action_dim,
        )
        state = _append_history_fn(state, partner_obs, memory_action)
        lr = float(model_config.get("TTAC_TEST_LR", 0.001))
        steps = int(model_config.get("TTAC_TEST_UPDATE_STEPS", 1))

        def _one_update(params, _):
            loss, grads = jax.value_and_grad(_ttac_v5_2_loss)(
                params, state, spec.eval_mode
            )
            del loss
            grads = _mask_grads(grads, adapter_mask)
            updates = jax.tree_util.tree_map(lambda g: -lr * g, grads)
            return optax.apply_updates(params, updates), None

        if spec.eval_mode == "base_no_test_adapt":
            next_params = state.params
        else:
            next_params, _ = jax.lax.scan(_one_update, state.params, None, steps)
        return state.replace(
            params=next_params,
            valid_mask=jnp.where(done, jnp.zeros_like(state.valid_mask), state.valid_mask),
            cursor=jnp.where(done, jnp.array(0, dtype=jnp.int32), state.cursor),
            count=jnp.where(done, jnp.array(0, dtype=jnp.int32), state.count),
            global_step=jnp.where(
                done, jnp.array(0, dtype=jnp.int32), state.global_step
            ),
            prev_partner_action=true_partner_action,
            prev_memory_action=jnp.where(
                done, jnp.array(0, dtype=jnp.int32), state.prev_memory_action
            ),
        )

    @jax.jit
    def _perform_step(carry, step_key):
        obs, env_state, done, total_reward, hstate = carry
        key_sample, key_step = jax.random.split(step_key, 2)
        sample_keys = jax.random.split(key_sample, env.num_agents)

        actions = {}
        next_hstate = {}
        for i, spec in enumerate(policies.policies):
            agent_id = f"agent_{i}"
            action, policy_state = _compute_action_for_policy(
                spec,
                hstate[agent_id],
                obs[agent_id],
                done[agent_id],
                sample_keys[i],
            )
            actions[agent_id] = action
            next_hstate[agent_id] = policy_state

        next_obs, next_env_state, reward, next_done, _ = env.step(
            key_step, env_state, actions
        )

        updated_hstate = {}
        for i, spec in enumerate(policies.policies):
            agent_id = f"agent_{i}"
            partner_id = f"agent_{1 - i}"
            updated_hstate[agent_id] = _update_policy_state(
                spec,
                adapter_masks[i],
                next_hstate[agent_id],
                obs[partner_id],
                actions[partner_id],
                next_done[agent_id],
            )

        carry = (
            next_obs,
            next_env_state,
            next_done,
            total_reward + reward["agent_0"],
            updated_hstate,
        )
        if reward_only:
            return carry, None
        return carry, (next_env_state, actions)

    key, key_r = jax.random.split(key, 2)
    obs, env_state = env.reset(key_r)
    init_done = {f"agent_{i}": False for i in range(env.num_agents)}
    init_done["__all__"] = False
    keys = jax.random.split(key, env.max_steps)
    carry = (obs, env_state, init_done, 0.0, init_hstate)
    carry, scan_output = jax.lax.scan(_perform_step, carry, keys)

    from overcooked_v2_experiments.eval.rollout import (
        PolicyRollout,
        RewardOnlyRollout,
    )

    if reward_only:
        return RewardOnlyRollout(total_reward=carry[-2])

    state_seq, actions_seq = scan_output

    return PolicyRollout(
        state_seq=state_seq,
        actions_seq=actions_seq,
        total_reward=carry[-2],
    )
