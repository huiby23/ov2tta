from functools import partial
from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp
from flax import core, struct

from overcooked_v2_experiments.eval.policy import (
    AbstractPolicy,
    FunctionalPolicyPairing,
    PolicyPairing,
)
from overcooked_v2_experiments.ttappo_v3_1_semantic_memory.models.abstract import ActorCriticBase
from overcooked_v2_experiments.ttappo_v3_1_semantic_memory.models.model import (
    get_actor_critic,
    initialize_carry,
)


@chex.dataclass
class PPOParams:
    params: core.FrozenDict[str, Any]
    ttt_source_stats: Dict[str, Any] | None = None


@chex.dataclass
class MemoryPolicyState:
    base_hstate: Any
    cached_feature_z: jnp.ndarray
    cached_temporal_feature: jnp.ndarray
    cached_partner_logits: jnp.ndarray
    cached_self_action: jnp.ndarray
    prev_partner_action: jnp.ndarray


@struct.dataclass
class MemoryFunctionalPolicySpec:
    params: core.FrozenDict[str, Any]
    eval_mode: str = struct.field(pytree_node=False, default="memory_off")


EVAL_MODES = (
    "memory_off",
    "state_adapt",
    "state_adapt_gated",
    "state_adapt_wrong_partner",
    "state_adapt_random_partner",
    "state_adapt_delayed_partner",
    "state_readout_off",
)


def summarize_ttt_source_stats(params, config, key):
    del params, config, key
    return {}


def _apply_policy_network(network, params, hstate, ac_in):
    out = network.apply(params, hstate, ac_in)
    if isinstance(out, tuple) and len(out) == 4:
        next_hstate, pi, value, aux = out
    else:
        next_hstate, pi, value = out
        aux = {}
    return next_hstate, pi, value, aux


def _zero_eval_cache(config):
    model_config = config["model"]
    return {
        "feature_z": jnp.zeros((model_config["FC_DIM_SIZE"],), dtype=jnp.float32),
        "temporal_feature": jnp.zeros(
            (
                model_config.get(
                    "TEMPORAL_HIDDEN_DIM",
                    model_config["FC_DIM_SIZE"],
                ),
            ),
            dtype=jnp.float32,
        ),
        "partner_logits": jnp.zeros((6,), dtype=jnp.float32),
        "self_action": jnp.array(0, dtype=jnp.int32),
        "prev_partner_action": jnp.array(0, dtype=jnp.int32),
    }


def _init_policy_state(config):
    zeros = _zero_eval_cache(config)
    return MemoryPolicyState(
        base_hstate=initialize_carry(config, 1),
        cached_feature_z=zeros["feature_z"],
        cached_temporal_feature=zeros["temporal_feature"],
        cached_partner_logits=zeros["partner_logits"],
        cached_self_action=zeros["self_action"],
        prev_partner_action=zeros["prev_partner_action"],
    )


def _compute_should_update(eval_mode, cached_partner_logits, partner_action, threshold):
    if eval_mode == "memory_off":
        return jnp.array([False], dtype=jnp.bool_)
    if eval_mode in (
        "state_adapt",
        "state_adapt_wrong_partner",
        "state_adapt_random_partner",
        "state_adapt_delayed_partner",
        "state_readout_off",
    ):
        return jnp.array([True], dtype=jnp.bool_)
    if eval_mode != "state_adapt_gated":
        raise ValueError(f"Unknown eval mode: {eval_mode}")

    log_probs = jax.nn.log_softmax(cached_partner_logits)
    ce = -log_probs[partner_action.astype(jnp.int32)]
    return jnp.array([ce > threshold], dtype=jnp.bool_)


def _readout_scale(eval_mode):
    return jnp.array(0.0 if eval_mode == "state_readout_off" else 1.0, dtype=jnp.float32)


def _resolve_memory_action(
    eval_mode,
    partner_action,
    self_action,
    prev_partner_action,
    key,
    action_dim,
):
    partner_action = partner_action.astype(jnp.int32)
    self_action = self_action.astype(jnp.int32)
    prev_partner_action = prev_partner_action.astype(jnp.int32)

    if eval_mode == "state_adapt_wrong_partner":
        memory_action = self_action
    elif eval_mode == "state_adapt_random_partner":
        if key is None:
            memory_action = (partner_action + self_action + 1) % action_dim
        else:
            memory_action = jax.random.randint(
                key,
                shape=partner_action.shape,
                minval=0,
                maxval=action_dim,
                dtype=jnp.int32,
            )
    elif eval_mode == "state_adapt_delayed_partner":
        memory_action = prev_partner_action
    else:
        memory_action = partner_action
    return memory_action, partner_action


def policy_checkoints_to_functional_policy_pairing(
    checkpoints: PPOParams,
    config,
    adaptive_index=0,
    eval_mode="state_adapt",
):
    policies = []
    for i, checkpoint in enumerate(checkpoints):
        params = (
            checkpoint.params
            if isinstance(checkpoint.params, core.FrozenDict)
            else core.freeze(checkpoint.params)
        )
        policy_mode = eval_mode if adaptive_index is not None and i == adaptive_index else "memory_off"
        policies.append(
            MemoryFunctionalPolicySpec(
                params=params,
                eval_mode=policy_mode,
            )
        )
    return FunctionalPolicyPairing(
        tuple(policies),
        backend="ttappo_v3_1_semantic_memory",
        config=config,
    )


def policy_checkoints_to_policy_pairing(
    checkpoints: PPOParams,
    config,
    adaptive_index=0,
    eval_mode="state_adapt",
    stochastic=True,
):
    policies = []
    for i, checkpoint in enumerate(checkpoints):
        if adaptive_index is not None and i == adaptive_index:
            policies.append(
                AdaptivePPOPolicy(
                    checkpoint,
                    config,
                    eval_mode=eval_mode,
                    stochastic=stochastic,
                )
            )
        else:
            policies.append(PPOPolicy(checkpoint.params, config, stochastic=stochastic))
    return PolicyPairing(*policies)


def get_functional_rollout(policies: FunctionalPolicyPairing, env, key):
    config = policies.config
    network = get_actor_critic(config)
    gate_threshold = float(config["model"].get("STATE_ADAPT_CE_THRESHOLD", 0.8))

    init_hstate = {
        f"agent_{i}": _init_policy_state(config) for i in range(env.num_agents)
    }

    def _add_dim(tree):
        return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

    def _compute_action_for_policy(spec, state, obs, done, key):
        ac_in = (obs, jnp.array(done))
        ac_in = _add_dim(ac_in)
        ac_in = _add_dim(ac_in)

        next_base_hstate, pi, _, aux = _apply_policy_network(
            network,
            spec.params,
            state.base_hstate,
            ac_in + (_readout_scale(spec.eval_mode),),
        )
        action = pi.sample(seed=key)[0, 0]
        next_state = MemoryPolicyState(
            base_hstate=next_base_hstate,
            cached_feature_z=aux["feature_z"][0, 0],
            cached_temporal_feature=aux["temporal_feature"][0, 0],
            cached_partner_logits=aux["partner_logits"][0, 0],
            cached_self_action=action,
            prev_partner_action=state.prev_partner_action,
        )
        return action, next_state

    def _update_policy_state(spec, state, self_action, partner_action, done, key):
        update_mask = _compute_should_update(
            spec.eval_mode,
            state.cached_partner_logits,
            partner_action,
            gate_threshold,
        )
        memory_action, next_prev_partner_action = _resolve_memory_action(
            spec.eval_mode,
            jnp.array([partner_action], dtype=jnp.int32),
            jnp.array([self_action], dtype=jnp.int32),
            state.prev_partner_action[jnp.newaxis, ...],
            key,
            state.cached_partner_logits.shape[-1],
        )
        next_base_hstate = jax.lax.cond(
            update_mask[0],
            lambda _: network.apply(
                spec.params,
                state.base_hstate,
                state.cached_feature_z[jnp.newaxis, ...],
                state.cached_temporal_feature[jnp.newaxis, ...],
                memory_action,
                jnp.array([done], dtype=jnp.bool_),
                update_mask,
                method=network.update_memory_state,
            ),
            lambda _: network.apply(
                spec.params,
                state.base_hstate,
                state.cached_feature_z[jnp.newaxis, ...],
                state.cached_temporal_feature[jnp.newaxis, ...],
                memory_action,
                jnp.array([done], dtype=jnp.bool_),
                jnp.array([False], dtype=jnp.bool_),
                method=network.update_memory_state,
            ),
            operand=None,
        )
        return MemoryPolicyState(
            base_hstate=next_base_hstate,
            cached_feature_z=state.cached_feature_z,
            cached_temporal_feature=state.cached_temporal_feature,
            cached_partner_logits=state.cached_partner_logits,
            cached_self_action=state.cached_self_action,
            prev_partner_action=next_prev_partner_action[0],
        )

    @jax.jit
    def _perform_step(carry, step_key):
        obs, state, done, total_reward, hstate = carry
        key_sample, key_step = jax.random.split(step_key, 2)
        sample_keys = jax.random.split(key_sample, env.num_agents)
        update_keys = jax.random.split(key_step, env.num_agents)

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

        next_obs, next_state, reward, next_done, _ = env.step(key_step, state, actions)

        updated_hstate = {}
        for i, spec in enumerate(policies.policies):
            agent_id = f"agent_{i}"
            partner_id = f"agent_{1 - i}"
            updated_hstate[agent_id] = _update_policy_state(
                spec,
                next_hstate[agent_id],
                actions[agent_id],
                actions[partner_id],
                next_done[agent_id],
                update_keys[i],
            )

        new_total_reward = total_reward + reward["agent_0"]
        carry = (next_obs, next_state, next_done, new_total_reward, updated_hstate)
        return carry, (next_state, actions)

    key, key_r = jax.random.split(key, 2)
    obs, state = env.reset(key_r)
    init_done = {f"agent_{i}": False for i in range(env.num_agents)}
    init_done["__all__"] = False
    keys = jax.random.split(key, env.max_steps)
    carry = (obs, state, init_done, 0.0, init_hstate)
    carry, (state_seq, actions_seq) = jax.lax.scan(_perform_step, carry, keys)
    total_reward = carry[-2]

    from overcooked_v2_experiments.eval.rollout import PolicyRollout

    return PolicyRollout(
        state_seq=state_seq,
        actions_seq=actions_seq,
        total_reward=total_reward,
    )


class PPOPolicy(AbstractPolicy):
    network: ActorCriticBase
    params: core.FrozenDict[str, Any]
    config: Dict[str, Any]
    stochastic: bool = True

    def __init__(self, params, config, stochastic=True):
        self.config = config
        self.stochastic = stochastic
        self.network = get_actor_critic(config)
        self.params = params if isinstance(params, core.FrozenDict) else core.freeze(params)

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate, key, params=None):
        if params is None:
            params = self.params
        eval_mode = getattr(self, "eval_mode", "memory_off")
        ac_in = (obs, jnp.array(done))
        ac_in = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], ac_in)
        ac_in = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], ac_in)
        ac_in = ac_in + (_readout_scale(eval_mode),)

        next_base_hstate, pi, _, aux = _apply_policy_network(
            self.network,
            params,
            hstate.base_hstate,
            ac_in,
        )
        if self.stochastic:
            action = pi.sample(seed=key)[0, 0]
        else:
            action = jnp.argmax(pi.probs, axis=-1)[0, 0]
        next_hstate = MemoryPolicyState(
            base_hstate=next_base_hstate,
            cached_feature_z=aux["feature_z"][0, 0],
            cached_temporal_feature=aux["temporal_feature"][0, 0],
            cached_partner_logits=aux["partner_logits"][0, 0],
            cached_self_action=action,
            prev_partner_action=hstate.prev_partner_action,
        )
        return action, next_hstate

    def init_hstate(self, batch_size, key=None):
        del key
        if batch_size != 1:
            base_hstate = initialize_carry(self.config, batch_size)
            zeros = _zero_eval_cache(self.config)
            return MemoryPolicyState(
                base_hstate=base_hstate,
                cached_feature_z=jnp.repeat(
                    zeros["feature_z"][jnp.newaxis, ...], batch_size, axis=0
                ),
                cached_temporal_feature=jnp.repeat(
                    zeros["temporal_feature"][jnp.newaxis, ...], batch_size, axis=0
                ),
                cached_partner_logits=jnp.repeat(
                    zeros["partner_logits"][jnp.newaxis, ...], batch_size, axis=0
                ),
                cached_self_action=jnp.repeat(
                    zeros["self_action"][jnp.newaxis, ...], batch_size, axis=0
                ),
                prev_partner_action=jnp.repeat(
                    zeros["prev_partner_action"][jnp.newaxis, ...], batch_size, axis=0
                ),
            )
        return _init_policy_state(self.config)


class AdaptivePPOPolicy(PPOPolicy):
    def __init__(self, checkpoint: PPOParams, config, eval_mode="state_adapt", stochastic=True):
        super().__init__(checkpoint.params, config, stochastic=stochastic)
        self.eval_mode = eval_mode
        self.gate_threshold = float(config["model"].get("STATE_ADAPT_CE_THRESHOLD", 0.8))

    @partial(jax.jit, static_argnums=(0,))
    def update_after_step(self, hstate, partner_obs, partner_action, done):
        del partner_obs
        update_mask = _compute_should_update(
            self.eval_mode,
            hstate.cached_partner_logits,
            partner_action,
            self.gate_threshold,
        )
        memory_action, next_prev_partner_action = _resolve_memory_action(
            self.eval_mode,
            jnp.array([partner_action], dtype=jnp.int32),
            jnp.array([hstate.cached_self_action], dtype=jnp.int32),
            jnp.array([hstate.prev_partner_action], dtype=jnp.int32),
            None,
            hstate.cached_partner_logits.shape[-1],
        )
        next_base_hstate = self.network.apply(
            self.params,
            hstate.base_hstate,
            hstate.cached_feature_z[jnp.newaxis, ...],
            hstate.cached_temporal_feature[jnp.newaxis, ...],
            memory_action,
            jnp.array([done], dtype=jnp.bool_),
            update_mask,
            method=self.network.update_memory_state,
        )
        return MemoryPolicyState(
            base_hstate=next_base_hstate,
            cached_feature_z=hstate.cached_feature_z,
            cached_temporal_feature=hstate.cached_temporal_feature,
            cached_partner_logits=hstate.cached_partner_logits,
            cached_self_action=hstate.cached_self_action,
            prev_partner_action=next_prev_partner_action[0],
        )
