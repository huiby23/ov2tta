from functools import partial
from typing import Any, Dict

import chex
from flax import core, traverse_util, struct
from overcooked_v2_experiments.eval.policy import (
    AbstractPolicy,
    PolicyPairing,
    FunctionalPolicyPairing,
)
from overcooked_v2_experiments.ttappo_v2_temporal.models.abstract import ActorCriticBase
from overcooked_v2_experiments.ttappo_v2_temporal.models.model import (
    get_actor_critic,
    initialize_carry,
)
import jax
import jax.numpy as jnp
import optax
import jaxmarl


@chex.dataclass
class PPOParams:
    params: core.FrozenDict[str, Any]
    ttt_source_stats: Dict[str, Any]


@chex.dataclass
class AdaptivePPOHState:
    base_hstate: Any
    adapt_params: core.FrozenDict[str, Any]
    opt_state: Any
    queue_z: jnp.ndarray
    queue_s: jnp.ndarray
    queue_count: jnp.ndarray
    queue_ptr: jnp.ndarray
    cached_base: jnp.ndarray
    cached_done: jnp.ndarray


@struct.dataclass
class TTTFunctionalPolicySpec:
    params: core.FrozenDict[str, Any]
    ttt_source_stats: Dict[str, Any]
    adaptive: bool = struct.field(pytree_node=False, default=False)


def _apply_policy_network(network, params, hstate, ac_in):
    out = network.apply(params, hstate, ac_in)
    if isinstance(out, tuple) and len(out) == 4:
        next_hstate, pi, value, aux = out
    else:
        next_hstate, pi, value = out
        aux = {}
    return next_hstate, pi, value, aux


def _build_ttt_mask(params):
    is_frozen = isinstance(params, core.FrozenDict)
    params_tree = core.unfreeze(params) if is_frozen else params
    flat = traverse_util.flatten_dict(params_tree)
    allowed = (
        "partner_temporal_proj",
        "partner_hidden",
        "partner_logits",
    )
    mask_flat = {}
    for key in flat:
        path = "/".join(key)
        mask_flat[key] = any(token in path for token in allowed)
    mask_tree = traverse_util.unflatten_dict(mask_flat)
    return core.freeze(mask_tree) if is_frozen else mask_tree


def _insert_queue(buffer, ptr, count, value):
    buffer = buffer.at[ptr].set(value)
    ptr = (ptr + 1) % buffer.shape[0]
    count = jnp.minimum(count + 1, buffer.shape[0])
    return buffer, ptr, count


def _compute_moments(buffer, count):
    mask = (jnp.arange(buffer.shape[0]) < count).astype(buffer.dtype)[:, None]
    denom = jnp.maximum(count.astype(buffer.dtype), 1.0)
    mean = jnp.sum(buffer * mask, axis=0) / denom
    centered = (buffer - mean) * mask
    cov = centered.T @ centered / jnp.maximum(denom - 1.0, 1.0)
    return mean, cov


def summarize_ttt_source_stats(params, config, key):
    env = jaxmarl.make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
    network = get_actor_critic(config)
    num_episodes = int(config["model"]["TTT_STATS_EPISODES"])
    max_steps = int(
        config["model"].get("TTT_STATS_MAX_STEPS", getattr(env, "max_steps", 400))
    )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    @jax.jit
    def _policy_step(params, hstate, obs, done, key):
        next_hstate, pi, _, aux = _apply_policy_network(
            network, params, hstate, (obs, done)
        )
        action = pi.sample(seed=key)
        return next_hstate, action, aux["feature_z"], aux["feature_s"]

    feature_z = []
    feature_s = []

    for _ in range(num_episodes):
        key, key_reset = jax.random.split(key)
        obs, env_state = reset_fn(key_reset)
        done = {agent: False for agent in env.agents}
        done["__all__"] = False
        hstate = {
            agent: initialize_carry(config, 1)
            for agent in env.agents
        }
        done_all = False
        steps = 0

        while not done_all and steps < max_steps:
            step_actions = {}
            for agent in env.agents:
                key, key_action = jax.random.split(key)
                agent_obs = obs[agent][jnp.newaxis, jnp.newaxis, ...]
                agent_done = jnp.array([[done[agent]]])
                next_hstate, action, f_z, f_s = _policy_step(
                    params,
                    hstate[agent],
                    agent_obs,
                    agent_done,
                    key_action,
                )
                hstate[agent] = next_hstate
                step_actions[agent] = action[0, 0]
                if agent == env.agents[0]:
                    feature_z.append(jax.device_get(f_z[0, 0]))
                    feature_s.append(jax.device_get(f_s[0, 0]))

            key, key_step = jax.random.split(key)
            obs, env_state, _, done, _ = step_fn(key_step, env_state, step_actions)
            done_all = bool(jax.device_get(done["__all__"]))
            steps += 1

    feature_z = jnp.stack(feature_z)
    feature_s = jnp.stack(feature_s)
    mu_z = jnp.mean(feature_z, axis=0)
    mu_s = jnp.mean(feature_s, axis=0)
    centered_z = feature_z - mu_z
    centered_s = feature_s - mu_s
    sigma_z = centered_z.T @ centered_z / jnp.maximum(feature_z.shape[0] - 1, 1)
    sigma_s = centered_s.T @ centered_s / jnp.maximum(feature_s.shape[0] - 1, 1)

    return {
        "mu_z": mu_z,
        "sigma_z": sigma_z,
        "mu_s": mu_s,
        "sigma_s": sigma_s,
        "num_episodes": jnp.array(num_episodes, dtype=jnp.int32),
        "max_steps": jnp.array(max_steps, dtype=jnp.int32),
    }


def _build_ttt_tx(config, mask):
    return optax.chain(
        optax.clip_by_global_norm(config["model"]["MAX_GRAD_NORM"]),
        optax.masked(
            optax.adam(float(config["model"]["TTT_EVAL_LR"]), eps=1e-5), mask
        ),
    )


def _init_functional_policy_state(spec: TTTFunctionalPolicySpec, config):
    if not spec.adaptive:
        return initialize_carry(config, 1)

    if not spec.ttt_source_stats:
        raise ValueError("Functional adaptive policy requires ttt_source_stats")

    z_dim = int(spec.ttt_source_stats["mu_z"].shape[-1])
    s_dim = int(spec.ttt_source_stats["mu_s"].shape[-1])
    cached_base_dim = int(config["model"]["FC_DIM_SIZE"]) + int(
        config["model"].get("TEMPORAL_HIDDEN_DIM", config["model"]["FC_DIM_SIZE"])
    )
    tx = _build_ttt_tx(config, _build_ttt_mask(spec.params))
    return AdaptivePPOHState(
        base_hstate=initialize_carry(config, 1),
        adapt_params=spec.params,
        opt_state=tx.init(spec.params),
        queue_z=jnp.zeros((int(config["model"]["TTT_QUEUE_SIZE"]), z_dim)),
        queue_s=jnp.zeros((int(config["model"]["TTT_QUEUE_SIZE"]), s_dim)),
        queue_count=jnp.array(0, dtype=jnp.int32),
        queue_ptr=jnp.array(0, dtype=jnp.int32),
        cached_base=jnp.zeros((cached_base_dim,)),
        cached_done=jnp.array(False),
    )


def policy_checkoints_to_functional_policy_pairing(
    checkpoints: PPOParams, config, adaptive_index=0
):
    policies = []
    for i, checkpoint in enumerate(checkpoints):
        params = (
            checkpoint.params
            if isinstance(checkpoint.params, core.FrozenDict)
            else core.freeze(checkpoint.params)
        )
        adaptive = adaptive_index is not None and i == adaptive_index
        policies.append(
            TTTFunctionalPolicySpec(
                params=params,
                ttt_source_stats=checkpoint.ttt_source_stats or {},
                adaptive=adaptive,
            )
        )
    return FunctionalPolicyPairing(
        tuple(policies), backend="ttappo_v2_temporal", config=config
    )


def get_functional_rollout(policies: FunctionalPolicyPairing, env, key):
    config = policies.config
    network = get_actor_critic(config)
    txs = [
        _build_ttt_tx(config, _build_ttt_mask(spec.params)) if spec.adaptive else None
        for spec in policies.policies
    ]

    init_hstate = {
        f"agent_{i}": _init_functional_policy_state(spec, config)
        for i, spec in enumerate(policies.policies)
    }

    def _add_dim(tree):
        return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

    def _compute_action_for_policy(spec, state, obs, done, key):
        done = jnp.array(done)
        ac_in = (obs, done)
        ac_in = _add_dim(ac_in)
        ac_in = _add_dim(ac_in)

        if spec.adaptive:
            next_base_hstate, pi, _, aux = _apply_policy_network(
                network, state.adapt_params, state.base_hstate, ac_in
            )
            next_state = AdaptivePPOHState(
                base_hstate=next_base_hstate,
                adapt_params=state.adapt_params,
                opt_state=state.opt_state,
                queue_z=state.queue_z,
                queue_s=state.queue_s,
                queue_count=state.queue_count,
                queue_ptr=state.queue_ptr,
                cached_base=aux["feature_base"][0, 0],
                cached_done=done,
            )
        else:
            next_state, pi, _, _ = _apply_policy_network(network, spec.params, state, ac_in)

        action = pi.sample(seed=key)[0, 0]
        return action, next_state

    def _update_adaptive_policy(spec, tx, state, partner_action, done):
        def _loss_fn(params):
            feature_z, feature_s, partner_logits = network.apply(
                params,
                state.cached_base,
                method=network._ttt_forward_from_base,
            )

            next_count = jnp.minimum(state.queue_count + 1, state.queue_z.shape[0])
            new_queue_z, _, _ = _insert_queue(
                state.queue_z, state.queue_ptr, state.queue_count, feature_z
            )
            new_queue_s, _, _ = _insert_queue(
                state.queue_s, state.queue_ptr, state.queue_count, feature_s
            )
            mu_q_z, sigma_q_z = _compute_moments(new_queue_z, next_count)
            mu_q_s, sigma_q_s = _compute_moments(new_queue_s, next_count)

            partner_loss = optax.softmax_cross_entropy_with_integer_labels(
                partner_logits, partner_action
            ).mean()
            align_z = jnp.mean((mu_q_z - spec.ttt_source_stats["mu_z"]) ** 2) + jnp.mean(
                (sigma_q_z - spec.ttt_source_stats["sigma_z"]) ** 2
            )
            align_s = jnp.mean((mu_q_s - spec.ttt_source_stats["mu_s"]) ** 2) + jnp.mean(
                (sigma_q_s - spec.ttt_source_stats["sigma_s"]) ** 2
            )
            align_total = align_z + align_s
            total = (
                float(config["model"]["TTT_SSL_COEF"]) * partner_loss
                + float(config["model"]["TTT_ALIGN_Z_COEF"]) * align_z
                + float(config["model"]["TTT_ALIGN_S_COEF"]) * align_s
            )
            return total, (feature_z, feature_s, partner_loss, align_total)

        (_, (feature_z, feature_s, partner_loss, align_total)), grads = (
            jax.value_and_grad(_loss_fn, has_aux=True)(state.adapt_params)
        )
        do_update = jnp.logical_or(
            jnp.logical_not(jnp.array(config["model"].get("TTT_GATED_ENABLED", False))),
            jnp.logical_or(
                partner_loss > float(config["model"].get("TTT_GATED_PARTNER_LOSS_THRESH", 0.0)),
                align_total > float(config["model"].get("TTT_GATED_ALIGN_THRESH", 0.0)),
            ),
        )
        updates, proposed_opt_state = tx.update(
            grads, state.opt_state, state.adapt_params
        )
        proposed_params = optax.apply_updates(state.adapt_params, updates)
        new_params = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_update, new, old),
            proposed_params,
            state.adapt_params,
        )
        new_opt_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_update, new, old),
            proposed_opt_state,
            state.opt_state,
        )
        queue_z, queue_ptr, queue_count = _insert_queue(
            state.queue_z, state.queue_ptr, state.queue_count, feature_z
        )
        queue_s, _, _ = _insert_queue(
            state.queue_s, state.queue_ptr, state.queue_count, feature_s
        )
        next_state = AdaptivePPOHState(
            base_hstate=state.base_hstate,
            adapt_params=new_params,
            opt_state=new_opt_state,
            queue_z=queue_z,
            queue_s=queue_s,
            queue_count=queue_count,
            queue_ptr=queue_ptr,
            cached_base=state.cached_base,
            cached_done=state.cached_done,
        )
        return jax.lax.cond(
            done,
            lambda _: _init_functional_policy_state(spec, config),
            lambda s: s,
            next_state,
        )

    @jax.jit
    def _perform_step(carry, key):
        obs, state, done, total_reward, hstate = carry
        key_sample, key_step = jax.random.split(key, 2)
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

        next_obs, next_state, reward, next_done, _ = env.step(key_step, state, actions)

        updated_hstate = {}
        for i, spec in enumerate(policies.policies):
            agent_id = f"agent_{i}"
            partner_id = f"agent_{1 - i}"
            if spec.adaptive:
                updated_hstate[agent_id] = _update_adaptive_policy(
                    spec,
                    txs[i],
                    next_hstate[agent_id],
                    actions[partner_id],
                    next_done[agent_id],
                )
            else:
                updated_hstate[agent_id] = next_hstate[agent_id]

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
        state_seq=state_seq, actions_seq=actions_seq, total_reward=total_reward
    )


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

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate, key, params=None):
        if params is None:
            params = self.params
        assert params is not None

        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        ac_in = (obs, done)
        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)

        next_hstate, pi, _, _ = _apply_policy_network(self.network, params, hstate, ac_in)

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]

        return action, next_hstate

    def init_hstate(self, batch_size, key=None):
        return initialize_carry(self.config, batch_size)


class AdaptivePPOPolicy(AbstractPolicy):
    def __init__(self, checkpoint: PPOParams, config, stochastic=True):
        self.config = config
        self.stochastic = stochastic
        self.network = get_actor_critic(config)
        self.params = (
            checkpoint.params
            if isinstance(checkpoint.params, core.FrozenDict)
            else core.freeze(checkpoint.params)
        )
        self.source_stats = checkpoint.ttt_source_stats or {}
        self.queue_size = int(config["model"]["TTT_QUEUE_SIZE"])
        self.lr = float(config["model"]["TTT_EVAL_LR"])
        self.ssl_coef = float(config["model"]["TTT_SSL_COEF"])
        self.align_z_coef = float(config["model"]["TTT_ALIGN_Z_COEF"])
        self.align_s_coef = float(config["model"]["TTT_ALIGN_S_COEF"])
        self.mask = _build_ttt_mask(self.params)
        self.tx = optax.chain(
            optax.clip_by_global_norm(config["model"]["MAX_GRAD_NORM"]),
            optax.masked(optax.adam(self.lr, eps=1e-5), self.mask),
        )

    def init_hstate(self, batch_size, key=None):
        if not self.source_stats:
            raise ValueError("AdaptivePPOPolicy requires ttt_source_stats in checkpoint")
        z_dim = int(self.source_stats["mu_z"].shape[-1])
        s_dim = int(self.source_stats["mu_s"].shape[-1])
        cached_base_dim = int(self.config["model"]["FC_DIM_SIZE"]) + int(
            self.config["model"].get(
                "TEMPORAL_HIDDEN_DIM", self.config["model"]["FC_DIM_SIZE"]
            )
        )
        base_hstate = initialize_carry(self.config, batch_size)
        return AdaptivePPOHState(
            base_hstate=base_hstate,
            adapt_params=self.params,
            opt_state=self.tx.init(self.params),
            queue_z=jnp.zeros((self.queue_size, z_dim)),
            queue_s=jnp.zeros((self.queue_size, s_dim)),
            queue_count=jnp.array(0, dtype=jnp.int32),
            queue_ptr=jnp.array(0, dtype=jnp.int32),
            cached_base=jnp.zeros((cached_base_dim,)),
            cached_done=jnp.array(False),
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate, key):
        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        ac_in = (obs, done)
        ac_in = _add_dim(ac_in)
        ac_in = _add_dim(ac_in)

        next_base_hstate, pi, _, aux = _apply_policy_network(
            self.network, hstate.adapt_params, hstate.base_hstate, ac_in
        )

        if self.stochastic:
            action = pi.sample(seed=key)[0, 0]
        else:
            action = jnp.argmax(pi.probs, axis=-1)[0, 0]

        next_hstate = AdaptivePPOHState(
            base_hstate=next_base_hstate,
            adapt_params=hstate.adapt_params,
            opt_state=hstate.opt_state,
            queue_z=hstate.queue_z,
            queue_s=hstate.queue_s,
            queue_count=hstate.queue_count,
            queue_ptr=hstate.queue_ptr,
            cached_base=aux["feature_base"][0, 0],
            cached_done=done,
        )
        return action, next_hstate

    @partial(jax.jit, static_argnums=(0,))
    def update_after_step(self, hstate, partner_obs, partner_action, done):
        del partner_obs

        def _loss_fn(params):
            feature_z, feature_s, partner_logits = self.network.apply(
                params,
                hstate.cached_base,
                method=self.network._ttt_forward_from_base,
            )

            new_queue_z, _, _ = _insert_queue(
                hstate.queue_z, hstate.queue_ptr, hstate.queue_count, feature_z
            )
            new_queue_s, _, _ = _insert_queue(
                hstate.queue_s, hstate.queue_ptr, hstate.queue_count, feature_s
            )
            mu_q_z, sigma_q_z = _compute_moments(
                new_queue_z, jnp.minimum(hstate.queue_count + 1, self.queue_size)
            )
            mu_q_s, sigma_q_s = _compute_moments(
                new_queue_s, jnp.minimum(hstate.queue_count + 1, self.queue_size)
            )

            partner_loss = optax.softmax_cross_entropy_with_integer_labels(
                partner_logits, partner_action
            ).mean()
            align_z = jnp.mean((mu_q_z - self.source_stats["mu_z"]) ** 2) + jnp.mean(
                (sigma_q_z - self.source_stats["sigma_z"]) ** 2
            )
            align_s = jnp.mean((mu_q_s - self.source_stats["mu_s"]) ** 2) + jnp.mean(
                (sigma_q_s - self.source_stats["sigma_s"]) ** 2
            )
            align_total = align_z + align_s
            total = (
                self.ssl_coef * partner_loss
                + self.align_z_coef * align_z
                + self.align_s_coef * align_s
            )
            aux = (feature_z, feature_s, partner_loss, align_total)
            return total, aux

        (loss, (feature_z, feature_s, partner_loss, align_total)), grads = jax.value_and_grad(
            _loss_fn, has_aux=True
        )(hstate.adapt_params)
        del loss
        do_update = jnp.logical_or(
            jnp.logical_not(jnp.array(self.config["model"].get("TTT_GATED_ENABLED", False))),
            jnp.logical_or(
                partner_loss > float(self.config["model"].get("TTT_GATED_PARTNER_LOSS_THRESH", 0.0)),
                align_total > float(self.config["model"].get("TTT_GATED_ALIGN_THRESH", 0.0)),
            ),
        )
        updates, proposed_opt_state = self.tx.update(
            grads, hstate.opt_state, hstate.adapt_params
        )
        proposed_params = optax.apply_updates(hstate.adapt_params, updates)
        new_params = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_update, new, old),
            proposed_params,
            hstate.adapt_params,
        )
        new_opt_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_update, new, old),
            proposed_opt_state,
            hstate.opt_state,
        )
        queue_z, queue_ptr, queue_count = _insert_queue(
            hstate.queue_z, hstate.queue_ptr, hstate.queue_count, feature_z
        )
        queue_s, _, _ = _insert_queue(
            hstate.queue_s, hstate.queue_ptr, hstate.queue_count, feature_s
        )
        next_state = AdaptivePPOHState(
            base_hstate=hstate.base_hstate,
            adapt_params=new_params,
            opt_state=new_opt_state,
            queue_z=queue_z,
            queue_s=queue_s,
            queue_count=queue_count,
            queue_ptr=queue_ptr,
            cached_base=hstate.cached_base,
            cached_done=hstate.cached_done,
        )

        def _reset_state():
            return self.init_hstate(1)

        return jax.lax.cond(done, _reset_state, lambda: next_state)


def policy_checkoints_to_policy_pairing(checkpoints: PPOParams, config):
    policies = []
    for i, checkpoint in enumerate(checkpoints):
        if i == 0:
            policies.append(AdaptivePPOPolicy(checkpoint, config))
        else:
            policies.append(PPOPolicy(checkpoint.params, config))
    return PolicyPairing(*policies)
