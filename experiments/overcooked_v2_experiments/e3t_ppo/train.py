from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import jaxmarl
import optax
import wandb
from flax import core
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from overcooked_v2_experiments.e3t_ppo.models.model import (
    get_actor_critic,
    initialize_carry,
)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    other_action: jnp.ndarray
    partner_mask: jnp.ndarray
    hist_obs: jnp.ndarray
    hist_action: jnp.ndarray
    train_mask: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def masked_mean(x, mask):
    mask = mask.astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)
    return (x.astype(jnp.float32) * mask).sum() / denom


def masked_std(x, mask):
    mean = masked_mean(x, mask)
    var = masked_mean(jnp.square(x - mean), mask)
    return jnp.sqrt(var + 1e-8)


def _scale_partner_logits(pi: distrax.Categorical, partner_mask, beta):
    scale = jnp.where(jnp.expand_dims(partner_mask, axis=-1), beta, 1.0)
    return distrax.Categorical(logits=pi.logits * scale)


def _partner_mask_from_env_idxs(partner_agent_idxs, env, num_actors):
    partner_mask_dict = {
        agent: partner_agent_idxs == idx for idx, agent in enumerate(env.agents)
    }
    return batchify(partner_mask_dict, env.agents, num_actors).squeeze().astype(jnp.bool_)


def _bootstrap_history(obs_batch, context_length, stay_action):
    history_obs = jnp.repeat(obs_batch[:, None, ...], context_length, axis=1)
    history_actions = jnp.full(
        (obs_batch.shape[0], context_length),
        stay_action,
        dtype=jnp.int32,
    )
    return history_obs, history_actions


def _update_history_buffers(
    history_obs,
    history_actions,
    ego_obs,
    partner_action,
    done_all_batch,
    stay_action,
):
    next_history_obs = jnp.concatenate(
        [history_obs[:, 1:], ego_obs[:, None, ...].astype(history_obs.dtype)],
        axis=1,
    )
    next_history_actions = jnp.concatenate(
        [history_actions[:, 1:], partner_action[:, None]],
        axis=1,
    )

    done_obs_mask = done_all_batch.reshape(
        (done_all_batch.shape[0],) + (1,) * (next_history_obs.ndim - 1)
    )
    done_act_mask = done_all_batch[:, None]
    reset_obs, reset_actions = _bootstrap_history(
        ego_obs.astype(next_history_obs.dtype),
        next_history_obs.shape[1],
        stay_action,
    )
    next_history_obs = jnp.where(done_obs_mask, reset_obs, next_history_obs)
    next_history_actions = jnp.where(done_act_mask, reset_actions, next_history_actions)
    return next_history_obs, next_history_actions


def _network_input(obs, done, hist_obs, hist_action, use_history_context):
    if use_history_context:
        return (obs, done, hist_obs.astype(obs.dtype), hist_action)
    return (obs, done)


def _strip_scan_axis(hstate):
    if hstate is None:
        return None
    return hstate.squeeze(axis=0)


def _add_scan_axis(hstate):
    if hstate is None:
        return None
    return hstate[jnp.newaxis, :]


def _take_axis1(x, indices):
    if x is None:
        return None
    return jnp.take(x, indices, axis=1)


def _build_official_param_masks(params, model_config):
    model_type = model_config["TYPE"]

    if model_type != "CNN":
        ones_mask = jax.tree_util.tree_map(lambda _: True, params)
        zeros_mask = jax.tree_util.tree_map(lambda _: False, params)
        return ones_mask, zeros_mask

    flat_params = flatten_dict(params)
    context_tokens = (
        "context_encoder",
        "context_obs_ln",
        "context_hist_ln",
        "context_proj_",
        "context_ln",
        "predictor_",
    )
    ppo_tokens = (
        "policy_encoder",
        "policy_ln",
        "actor_",
        "critic_",
    )

    ppo_flat = {}
    context_flat = {}
    unmatched = []
    overlap = []
    for key in flat_params:
        path = "/".join(key)
        is_context = any(token in path for token in context_tokens)
        is_ppo = any(token in path for token in ppo_tokens)
        if is_context and is_ppo:
            overlap.append(path)
        if not is_context and not is_ppo:
            unmatched.append(path)
        ppo_flat[key] = is_ppo
        context_flat[key] = is_context

    if overlap:
        raise ValueError(f"Overlapping PPO/context params: {overlap}")
    if unmatched:
        raise ValueError(f"Unmatched params for official split: {unmatched}")

    return (
        core.freeze(unflatten_dict(ppo_flat)),
        core.freeze(unflatten_dict(context_flat)),
    )


def _mask_grads(grads, mask):
    grads_is_frozen = isinstance(grads, core.FrozenDict)
    mask_is_frozen = isinstance(mask, core.FrozenDict)
    grads_tree = core.unfreeze(grads) if grads_is_frozen else grads
    mask_tree = core.unfreeze(mask) if mask_is_frozen else mask
    masked = jax.tree_util.tree_map(
        lambda g, m: g if m else jnp.zeros_like(g),
        grads_tree,
        mask_tree,
    )
    return core.freeze(masked) if grads_is_frozen else masked


def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
):
    if population_config is not None:
        raise NotImplementedError("E3T-PPO does not use partner populations.")

    env_config = config["env"]
    model_config = config["model"]

    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    if env.num_agents != 2:
        raise NotImplementedError("E3T-PPO currently assumes two-player environments.")

    obs_shape = env.observation_space().shape
    action_dim = env.action_space(env.agents[0]).n
    stay_action = min(4, action_dim - 1)
    context_length = int(model_config.get("CONTEXT_LENGTH", 5))
    use_history_context = model_config.get("USE_HISTORY_CONTEXT", False)
    use_partner_mix = model_config.get("USE_PARTNER_MIX", False)
    partner_mix_mode = model_config.get("PARTNER_MIX_MODE", "ppo_consistent")
    use_moa_aux = model_config.get("USE_MOA_AUX", True)
    separate_moa_update = model_config.get("SEPARATE_MOA_UPDATE", False)
    context_update_epochs = int(
        model_config.get("CONTEXT_UPDATE_EPOCHS", model_config["UPDATE_EPOCHS"])
    )
    official_param_split = model_config.get("OFFICIAL_SEPARATE_PARAM_SPLIT", True)
    moa_warmup_updates = int(model_config.get("MOA_WARMUP_UPDATES", 0))
    mix_warmup_updates = int(model_config.get("MIX_WARMUP_UPDATES", 0))

    model_config["NUM_ACTORS"] = env.num_agents * model_config["NUM_ENVS"]
    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"]
        // model_config["NUM_STEPS"]
        // model_config["NUM_ENVS"]
    )
    model_config["MINIBATCH_SIZE"] = (
        model_config["NUM_ACTORS"]
        * model_config["NUM_STEPS"]
        // model_config["NUM_MINIBATCHES"]
    )

    num_checkpoints = config["NUM_CHECKPOINTS"]
    checkpoint_steps = jnp.linspace(
        0,
        model_config["NUM_UPDATES"],
        num_checkpoints,
        endpoint=True,
        dtype=jnp.int32,
    )
    if num_checkpoints > 0:
        checkpoint_steps = checkpoint_steps.at[-1].set(model_config["NUM_UPDATES"])

    def _update_checkpoint(checkpoint_states, params, i):
        return jax.tree_util.tree_map(
            lambda x, y: x.at[i].set(y),
            checkpoint_states,
            params,
        )

    def create_learning_rate_fn():
        base_learning_rate = model_config["LR"]
        lr_warmup = model_config["LR_WARMUP"]
        update_steps = model_config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)
        steps_per_epoch = (
            model_config["NUM_MINIBATCHES"] * model_config["UPDATE_EPOCHS"]
        )

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch,
        )
        return optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=model_config["REW_SHAPING_HORIZON"],
    )

    def train(rng, population=None, initial_train_state=None):
        if population is not None:
            raise NotImplementedError("E3T-PPO does not use partner populations.")

        original_seed = rng[0]
        network = get_actor_critic(config)

        rng, _rng = jax.random.split(rng)
        if use_history_context:
            init_x = (
                jnp.zeros(
                    (1, model_config["NUM_ENVS"], *obs_shape),
                    dtype=jnp.float32,
                ),
                jnp.zeros((1, model_config["NUM_ENVS"]), dtype=jnp.bool_),
                jnp.zeros(
                    (1, model_config["NUM_ENVS"], context_length, *obs_shape),
                    dtype=jnp.float32,
                ),
                jnp.full(
                    (1, model_config["NUM_ENVS"], context_length),
                    stay_action,
                    dtype=jnp.int32,
                ),
            )
        else:
            init_x = (
                jnp.zeros(
                    (1, model_config["NUM_ENVS"], *obs_shape),
                    dtype=jnp.float32,
                ),
                jnp.zeros((1, model_config["NUM_ENVS"]), dtype=jnp.bool_),
            )
        init_hstate = initialize_carry(config, model_config["NUM_ENVS"])
        network_params = network.init(_rng, init_hstate, init_x)

        if model_config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(model_config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        if initial_train_state is not None:
            train_state = initial_train_state

        ppo_param_mask, context_param_mask = _build_official_param_masks(
            train_state.params,
            model_config,
        )
        use_official_param_split = separate_moa_update and official_param_split

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])

        init_obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape((-1,) + obs_shape)
        init_history_obs, init_history_actions = _bootstrap_history(
            init_obs_batch,
            context_length,
            stay_action,
        )

        def _sample_partner_agent_idxs(rng_key):
            return jax.random.randint(
                rng_key,
                (model_config["NUM_ENVS"],),
                0,
                env.num_agents,
            )

        rng, _rng = jax.random.split(rng)
        init_partner_agent_idxs = _sample_partner_agent_idxs(_rng)

        def _update_step(update_runner_state, unused):
            runner_state, update_step = update_runner_state

            def _env_step(env_step_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    history_obs,
                    history_actions,
                    partner_agent_idxs,
                    rng,
                    update_step,
                ) = env_step_state

                mix_active = use_partner_mix and (update_step >= mix_warmup_updates)
                mix_active_float = jnp.asarray(mix_active, dtype=jnp.float32)
                train_mask = jnp.ones((model_config["NUM_ACTORS"],), dtype=jnp.bool_)

                rng, policy_rng, step_rng = jax.random.split(rng, 3)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *obs_shape
                )
                ac_in = _network_input(
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                    history_obs[jnp.newaxis, :],
                    history_actions[jnp.newaxis, :],
                    use_history_context,
                )
                hstate, pi, value, other_pi = network.apply(
                    train_state.params,
                    hstate,
                    ac_in,
                )

                partner_mask = _partner_mask_from_env_idxs(
                    partner_agent_idxs,
                    env,
                    model_config["NUM_ACTORS"],
                )
                acting_partner_mask = jnp.where(
                    mix_active,
                    partner_mask,
                    jnp.zeros_like(partner_mask),
                )
                acting_pi = _scale_partner_logits(
                    pi,
                    acting_partner_mask,
                    model_config["PARTNER_MIX_EPS"],
                )
                action = acting_pi.sample(seed=policy_rng)
                log_prob = acting_pi.log_prob(action)

                env_act = unbatchify(
                    action,
                    env.agents,
                    model_config["NUM_ENVS"],
                    env.num_agents,
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}
                other_env_act = {
                    env.agents[0]: env_act[env.agents[1]],
                    env.agents[1]: env_act[env.agents[0]],
                }
                other_action = batchify(
                    other_env_act,
                    env.agents,
                    model_config["NUM_ACTORS"],
                ).squeeze()

                rng_step = jax.random.split(step_rng, model_config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                current_timestep = (
                    update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                original_reward = jnp.array([reward[a] for a in env.agents])
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor,
                    reward,
                    info["shaped_reward"],
                )
                shaped_reward = jnp.array([info["shaped_reward"][a] for a in env.agents])
                combined_reward = jnp.array([reward[a] for a in env.agents])

                mix_agent0_env = partner_agent_idxs == 0
                mix_agent1_env = partner_agent_idxs == 1
                mix_agent0_info = {
                    agent: mix_agent0_env.astype(jnp.float32) * mix_active_float
                    for agent in env.agents
                }
                mix_agent1_info = {
                    agent: mix_agent1_env.astype(jnp.float32) * mix_active_float
                    for agent in env.agents
                }
                mix_applied_info = {
                    agent: jnp.full(
                        (model_config["NUM_ENVS"],),
                        mix_active_float,
                        dtype=jnp.float32,
                    )
                    for agent in env.agents
                }
                partner_mask_info = unbatchify(
                    acting_partner_mask.astype(jnp.float32),
                    env.agents,
                    model_config["NUM_ENVS"],
                    env.num_agents,
                )

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward
                info["partner_mask"] = batchify(
                    partner_mask_info,
                    env.agents,
                    model_config["NUM_ACTORS"],
                ).squeeze()
                info["mix_applied_frac"] = batchify(
                    mix_applied_info,
                    env.agents,
                    model_config["NUM_ACTORS"],
                ).squeeze()
                info["mix_agent0_frac"] = batchify(
                    mix_agent0_info,
                    env.agents,
                    model_config["NUM_ACTORS"],
                ).squeeze()
                info["mix_agent1_frac"] = batchify(
                    mix_agent1_info,
                    env.agents,
                    model_config["NUM_ACTORS"],
                ).squeeze()
                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((model_config["NUM_ACTORS"])),
                    info,
                )

                done_batch = batchify(done, env.agents, model_config["NUM_ACTORS"]).squeeze()
                done_all_batch = jnp.tile(done["__all__"], env.num_agents)
                next_history_obs, next_history_actions = _update_history_buffers(
                    history_obs,
                    history_actions,
                    obs_batch,
                    other_action,
                    done_all_batch,
                    stay_action,
                )

                transition = Transition(
                    global_done=done_all_batch,
                    done=last_done,
                    action=action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(
                        reward, env.agents, model_config["NUM_ACTORS"]
                    ).squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    info=info,
                    other_action=other_action,
                    partner_mask=acting_partner_mask,
                    hist_obs=history_obs,
                    hist_action=history_actions,
                    train_mask=train_mask,
                )

                env_step_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    hstate,
                    next_history_obs,
                    next_history_actions,
                    partner_agent_idxs,
                    rng,
                    update_step,
                )
                return env_step_state, transition

            (
                train_state,
                env_state,
                obsv,
                done_batch,
                hstate,
                history_obs,
                history_actions,
                rng,
            ) = runner_state
            initial_hstate = hstate
            rng, _rng = jax.random.split(rng)
            partner_agent_idxs = _sample_partner_agent_idxs(_rng)
            env_step_state = (
                train_state,
                env_state,
                obsv,
                done_batch,
                hstate,
                history_obs,
                history_actions,
                partner_agent_idxs,
                rng,
                update_step,
            )
            env_step_state, traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, model_config["NUM_STEPS"]
            )
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                history_obs,
                history_actions,
                _partner_agent_idxs,
                rng,
                update_step,
            ) = env_step_state

            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *obs_shape
            )
            _, _, last_val, _ = network.apply(
                train_state.params,
                hstate,
                _network_input(
                    last_obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                    history_obs[jnp.newaxis, :],
                    history_actions[jnp.newaxis, :],
                    use_history_context,
                ),
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = (
                        transition.reward
                        + model_config["GAMMA"] * next_value * (1 - transition.global_done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + model_config["GAMMA"]
                        * model_config["GAE_LAMBDA"]
                        * (1 - transition.global_done)
                        * gae
                    )
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            moa_active = use_moa_aux and (update_step >= moa_warmup_updates)

            def _update_epoch(update_state, epoch_idx):
                def _context_loss_fn(params, init_hstate, traj_batch):
                    hstate = _strip_scan_axis(init_hstate)
                    train_mask = jax.lax.stop_gradient(traj_batch.train_mask)
                    _, _, _, other_pi = network.apply(
                        params,
                        hstate,
                        _network_input(
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.hist_obs,
                            traj_batch.hist_action,
                            use_history_context,
                        ),
                    )
                    other_log_prob = other_pi.log_prob(traj_batch.other_action)
                    context_loss = -other_log_prob
                    moa_pred = jnp.argmax(other_pi.logits, axis=-1)
                    moa_acc = (moa_pred == traj_batch.other_action).astype(jnp.float32)
                    return (
                        masked_mean(context_loss, train_mask),
                        masked_mean(moa_acc, train_mask),
                    )

                def _ppo_loss_fn(params, init_hstate, traj_batch, gae, targets):
                    hstate = _strip_scan_axis(init_hstate)
                    train_mask = jax.lax.stop_gradient(traj_batch.train_mask)
                    _, pi, value, other_pi = network.apply(
                        params,
                        hstate,
                        _network_input(
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.hist_obs,
                            traj_batch.hist_action,
                            use_history_context,
                        ),
                    )

                    if use_partner_mix and partner_mix_mode == "ppo_consistent":
                        behavior_pi = _scale_partner_logits(
                            pi,
                            traj_batch.partner_mask,
                            model_config["PARTNER_MIX_EPS"],
                        )
                    else:
                        behavior_pi = pi

                    log_prob = behavior_pi.log_prob(traj_batch.action)
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)

                    other_log_prob = other_pi.log_prob(traj_batch.other_action)
                    moa_loss_per_sample = -other_log_prob
                    moa_loss = masked_mean(moa_loss_per_sample, train_mask)
                    moa_pred = jnp.argmax(other_pi.logits, axis=-1)
                    moa_acc = masked_mean(
                        (moa_pred == traj_batch.other_action).astype(jnp.float32),
                        train_mask,
                    )

                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * masked_mean(
                        jnp.maximum(value_losses, value_losses_clipped),
                        train_mask,
                    )

                    gae = (gae - masked_mean(gae, train_mask)) / masked_std(gae, train_mask)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - model_config["CLIP_EPS"],
                            1.0 + model_config["CLIP_EPS"],
                        )
                        * gae
                    )
                    actor_loss = -masked_mean(
                        jnp.minimum(loss_actor1, loss_actor2),
                        train_mask,
                    )
                    entropy = masked_mean(pi.entropy(), train_mask)
                    approx_kl = masked_mean((ratio - 1.0) - logratio, train_mask)
                    clip_frac = masked_mean(
                        (jnp.abs(ratio - 1.0) > model_config["CLIP_EPS"]).astype(jnp.float32),
                        train_mask,
                    )

                    total_loss = (
                        actor_loss
                        + model_config["VF_COEF"] * value_loss
                        - model_config["ENT_COEF"] * entropy
                    )
                    if not separate_moa_update:
                        total_loss = total_loss + jnp.where(
                            moa_active,
                            model_config["MOA_COEF"] * moa_loss,
                            0.0,
                        )

                    aux = (
                        value_loss,
                        actor_loss,
                        entropy,
                        moa_loss,
                        moa_acc,
                        masked_mean(ratio, train_mask),
                        approx_kl,
                        clip_frac,
                    )
                    return total_loss, aux

                def _context_update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch = batch_info

                    def _perform_update():
                        grad_fn = jax.value_and_grad(_context_loss_fn, has_aux=True)
                        (context_loss, moa_acc), grads = grad_fn(
                            train_state.params,
                            init_hstate,
                            traj_batch,
                        )
                        if use_official_param_split:
                            grads = _mask_grads(grads, context_param_mask)
                        grad_norm = optax.global_norm(grads)
                        return (
                            train_state.apply_gradients(grads=grads),
                            (context_loss, moa_acc, grad_norm),
                        )

                    def _no_op():
                        zeros = jnp.array(0.0, dtype=jnp.float32)
                        return train_state, (zeros, zeros, zeros)

                    run_update = jnp.logical_and(moa_active, traj_batch.train_mask.any())
                    return jax.lax.cond(run_update, _perform_update, _no_op)

                def _ppo_update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, gae, targets = batch_info

                    def _perform_update():
                        grad_fn = jax.value_and_grad(_ppo_loss_fn, has_aux=True)
                        (total_loss, aux), grads = grad_fn(
                            train_state.params,
                            init_hstate,
                            traj_batch,
                            gae,
                            targets,
                        )
                        if use_official_param_split:
                            grads = _mask_grads(grads, ppo_param_mask)
                        grad_norm = optax.global_norm(grads)
                        return train_state.apply_gradients(grads=grads), (
                            total_loss,
                            aux,
                            grad_norm,
                        )

                    def _no_op():
                        zeros = jnp.array(0.0, dtype=jnp.float32)
                        aux = tuple(zeros for _ in range(8))
                        return train_state, (zeros, aux, zeros)

                    return jax.lax.cond(
                        traj_batch.train_mask.any(),
                        _perform_update,
                        _no_op,
                    )

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                hstate = _add_scan_axis(init_hstate)
                permutation = jax.random.permutation(_rng, model_config["NUM_ACTORS"])
                minibatch_indices = permutation.reshape(
                    model_config["NUM_MINIBATCHES"], -1
                )
                advantages = advantages.squeeze()
                targets = targets.squeeze()

                def _build_minibatch(indices):
                    return (
                        _take_axis1(hstate, indices),
                        jax.tree_util.tree_map(
                            lambda x: jnp.take(x, indices, axis=1),
                            traj_batch,
                        ),
                        jnp.take(advantages, indices, axis=1),
                        jnp.take(targets, indices, axis=1),
                    )

                run_context_epoch = separate_moa_update and (epoch_idx < context_update_epochs)

                def _run_context(train_state):
                    def _scan_context(train_state, indices):
                        init_hstate_mb, traj_batch_mb, _, _ = _build_minibatch(indices)
                        return _context_update_minbatch(
                            train_state,
                            (init_hstate_mb, traj_batch_mb),
                        )

                    return jax.lax.scan(
                        _scan_context,
                        train_state,
                        minibatch_indices,
                    )

                def _skip_context(train_state):
                    zeros = jnp.zeros((model_config["NUM_MINIBATCHES"],), dtype=jnp.float32)
                    return train_state, (zeros, zeros, zeros)

                train_state, context_info = jax.lax.cond(
                    run_context_epoch,
                    _run_context,
                    _skip_context,
                    train_state,
                )

                run_ppo_epoch = epoch_idx < model_config["UPDATE_EPOCHS"]

                def _run_ppo(train_state):
                    def _scan_ppo(train_state, indices):
                        return _ppo_update_minbatch(
                            train_state,
                            _build_minibatch(indices),
                        )

                    return jax.lax.scan(
                        _scan_ppo,
                        train_state,
                        minibatch_indices,
                    )

                def _skip_ppo(train_state):
                    zeros = jnp.zeros((model_config["NUM_MINIBATCHES"],), dtype=jnp.float32)
                    aux = tuple(zeros for _ in range(8))
                    return train_state, (zeros, aux, zeros)

                train_state, ppo_info = jax.lax.cond(
                    run_ppo_epoch,
                    _run_ppo,
                    _skip_ppo,
                    train_state,
                )

                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, (ppo_info, context_info)

            rng, _rng = jax.random.split(rng)
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                _rng,
            )
            total_update_epochs = max(
                model_config["UPDATE_EPOCHS"],
                context_update_epochs if separate_moa_update else 0,
            )
            total_update_epochs = max(total_update_epochs, 1)
            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                jnp.arange(total_update_epochs),
            )

            train_state = update_state[0]
            rng = update_state[-1]
            metric = traj_batch.info

            ppo_loss_info, context_loss_info = loss_info
            ppo_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), ppo_loss_info)
            context_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), context_loss_info)
            (
                total_loss,
                aux_data,
                ppo_grad_norm,
            ) = ppo_loss_info
            (
                value_loss,
                actor_loss,
                entropy,
                moa_loss,
                moa_acc,
                ratio,
                approx_kl,
                clip_frac,
            ) = aux_data
            (
                context_loss,
                context_acc,
                moa_grad_norm,
            ) = context_loss_info

            loss_metrics = {
                "total_loss": total_loss,
                "value_loss": value_loss,
                "actor_loss": actor_loss,
                "entropy": entropy,
                "moa_loss": moa_loss,
                "moa_acc": moa_acc,
                "ratio": ratio,
                "approx_kl": approx_kl,
                "clip_frac": clip_frac,
                "ppo_grad_norm": ppo_grad_norm,
                "moa_grad_norm": moa_grad_norm,
                "context_loss": context_loss,
                "context_acc": context_acc,
            }
            for key, value in loss_metrics.items():
                metric[key] = value
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

            update_step = update_step + 1
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
            )

            def callback(metric, original_seed):
                metric = {f"rng{int(original_seed)}/{k}": v for k, v in metric.items()}
                wandb.log(metric)

            jax.debug.callback(callback, metric, original_seed)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                history_obs,
                history_actions,
                rng,
            )
            return (runner_state, update_step), metric

        initial_update_step = update_step_offset or 0
        initial_checkpoints = jax.tree_util.tree_map(
            lambda p: jnp.zeros((num_checkpoints,) + p.shape, p.dtype),
            train_state.params,
        )
        if num_checkpoints > 0:
            initial_checkpoints = jax.lax.cond(
                (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                _update_checkpoint,
                lambda c, _p, _i: c,
                initial_checkpoints,
                train_state.params,
                0,
            )

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            initial_checkpoints,
            env_state,
            obsv,
            jnp.zeros((model_config["NUM_ACTORS"]), dtype=bool),
            initial_update_step,
            init_hstate,
            init_history_obs,
            init_history_actions,
            init_partner_agent_idxs,
            _rng,
        )

        def _scan_update(runner_state, unused):
            (
                train_state,
                checkpoint_states,
                env_state,
                obsv,
                done_batch,
                update_step,
                hstate,
                history_obs,
                history_actions,
                _partner_agent_idxs,
                rng,
            ) = runner_state
            (next_runner_state, next_update_step), metric = _update_step(
                (
                    (
                        train_state,
                        env_state,
                        obsv,
                        done_batch,
                        hstate,
                        history_obs,
                        history_actions,
                        rng,
                    ),
                    update_step,
                ),
                unused,
            )
            (
                train_state,
                env_state,
                obsv,
                done_batch,
                hstate,
                history_obs,
                history_actions,
                rng,
            ) = next_runner_state
            if num_checkpoints > 0:
                checkpoint_idx_selector = checkpoint_steps == next_update_step
                checkpoint_states = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    checkpoint_states,
                    train_state.params,
                    jnp.argmax(checkpoint_idx_selector),
                )
            runner_state = (
                train_state,
                checkpoint_states,
                env_state,
                obsv,
                done_batch,
                next_update_step,
                hstate,
                history_obs,
                history_actions,
                init_partner_agent_idxs,
                rng,
            )
            return runner_state, metric

        num_update_steps = update_step_num_overwrite or model_config["NUM_UPDATES"]
        runner_state, metric = jax.lax.scan(
            _scan_update,
            runner_state,
            None,
            num_update_steps,
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
