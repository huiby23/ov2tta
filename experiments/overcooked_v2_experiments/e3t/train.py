from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import jaxmarl
import optax
import wandb
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from overcooked_v2_experiments.e3t.models.model import get_actor_critic, initialize_carry


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    train_mask: jnp.ndarray
    other_action: jnp.ndarray
    hist_obs: jnp.ndarray
    hist_action: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _scale_partner_logits(pi: distrax.Categorical, partner_mask, beta):
    logits = pi.logits
    scaling = jnp.where(partner_mask[:, None], beta, 1.0)
    return distrax.Categorical(logits=logits * scaling)


def _partner_mask_from_env_idxs(partner_agent_idxs, env, num_actors):
    partner_mask_dict = {a: partner_agent_idxs == i for i, a in enumerate(env.agents)}
    return batchify(partner_mask_dict, env.agents, num_actors).squeeze().astype(jnp.bool_)


def _partner_obs_batch(last_obs, env, obs_shape):
    partner_obs = {
        env.agents[0]: last_obs[env.agents[1]],
        env.agents[1]: last_obs[env.agents[0]],
    }
    return jnp.stack([partner_obs[a] for a in env.agents]).reshape((-1,) + obs_shape)


def _bootstrap_history(obs_batch, context_length, stay_action):
    history_obs = jnp.repeat(obs_batch[:, None, ...], context_length, axis=1)
    history_actions = jnp.full((obs_batch.shape[0], context_length), stay_action, dtype=jnp.int32)
    return history_obs, history_actions


def _update_history_buffers(history_obs, history_actions, ego_obs, partner_action, done_all_batch, stay_action):
    next_history_obs = jnp.concatenate(
        [history_obs[:, 1:], ego_obs[:, None, ...].astype(history_obs.dtype)],
        axis=1,
    )
    next_history_actions = jnp.concatenate(
        [history_actions[:, 1:], partner_action[:, None]],
        axis=1,
    )
    done_obs_mask = done_all_batch.reshape((done_all_batch.shape[0],) + (1,) * (next_history_obs.ndim - 1))
    done_act_mask = done_all_batch[:, None]
    reset_obs, reset_actions = _bootstrap_history(
        ego_obs.astype(next_history_obs.dtype),
        next_history_obs.shape[1],
        stay_action,
    )
    next_history_obs = jnp.where(done_obs_mask, reset_obs, next_history_obs)
    next_history_actions = jnp.where(done_act_mask, reset_actions, next_history_actions)
    return next_history_obs, next_history_actions


def make_train(config, update_step_offset=None, update_step_num_overwrite=None, population_config=None):
    if population_config is not None:
        raise NotImplementedError("E3T does not use external partner populations.")

    env_config = config["env"]
    model_config = config["model"]
    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    if env.num_agents != 2:
        raise NotImplementedError("Current E3T implementation assumes two-player coordination.")

    obs_shape = env.observation_space().shape
    action_dim = env.action_space(env.agents[0]).n
    stay_action = min(4, action_dim - 1)
    context_length = model_config.get("CONTEXT_LENGTH", 5)
    env_shard_mode = config.get("ENV_SHARD_ACROSS_DEVICES", False)
    env_shard_axis_name = config.get("ENV_SHARD_AXIS_NAME", "env_shard")
    global_num_envs = config.get("ENV_SHARD_TOTAL_ENVS", model_config["NUM_ENVS"])
    env_config.setdefault("ENV_KWARGS", {})["obs_shape"] = obs_shape

    model_config["NUM_ACTORS"] = env.num_agents * model_config["NUM_ENVS"]
    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"] // model_config["NUM_STEPS"] // global_num_envs
    )
    model_config["MINIBATCH_SIZE"] = (
        model_config["NUM_ACTORS"] * model_config["NUM_STEPS"] // model_config["NUM_MINIBATCHES"]
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
        return jax.tree_util.tree_map(lambda x, y: x.at[i].set(y), checkpoint_states, params)

    def create_learning_rate_fn():
        base_learning_rate = model_config["LR"]
        warmup_steps = int(model_config["LR_WARMUP"] * model_config["NUM_UPDATES"])
        steps_per_epoch = model_config["NUM_MINIBATCHES"] * model_config["UPDATE_EPOCHS"]
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(model_config["NUM_UPDATES"] - warmup_steps, 1)
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

    separate_context_update = model_config.get("SEPARATE_CONTEXT_UPDATE", True)
    context_update_epochs = model_config.get("CONTEXT_UPDATE_EPOCHS", model_config["UPDATE_EPOCHS"])

    def train(rng, population=None, initial_train_state=None):
        if population is not None:
            raise NotImplementedError("E3T does not use external partner populations.")

        original_seed = rng[0]
        network = get_actor_critic(config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, model_config["NUM_ENVS"], *obs_shape), dtype=jnp.float32),
            jnp.zeros((1, model_config["NUM_ENVS"]), dtype=jnp.bool_),
            jnp.zeros((1, model_config["NUM_ENVS"], context_length, *obs_shape), dtype=jnp.float32),
            jnp.full((1, model_config["NUM_ENVS"], context_length), stay_action, dtype=jnp.int32),
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

        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        if initial_train_state is not None:
            train_state = initial_train_state

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])
        init_obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape((-1,) + obs_shape)
        init_history_obs, init_history_actions = _bootstrap_history(
            init_obs_batch.astype(jnp.float32),
            context_length,
            stay_action,
        )

        def _sample_partner_agent_idxs(rng_key):
            return jax.random.randint(rng_key, (model_config["NUM_ENVS"],), 0, env.num_agents)

        rng, _rng = jax.random.split(rng)
        init_partner_agent_idxs = _sample_partner_agent_idxs(_rng)

        def _update_step(runner_state, unused):
            (
                train_state,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                history_obs,
                history_actions,
                partner_agent_idxs,
                rng,
            ) = runner_state

            rng, _rng = jax.random.split(rng)
            partner_agent_idxs = _sample_partner_agent_idxs(_rng)

            def _env_step(env_step_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    history_obs,
                    history_actions,
                    partner_agent_idxs,
                    rng,
                ) = env_step_state

                partner_mask = _partner_mask_from_env_idxs(
                    partner_agent_idxs, env, model_config["NUM_ACTORS"]
                )
                train_mask = jnp.ones_like(partner_mask, dtype=jnp.bool_)

                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape((-1,) + obs_shape)
                ac_in = (
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                    history_obs[jnp.newaxis, :].astype(obs_batch.dtype),
                    history_actions[jnp.newaxis, :],
                )
                hstate, pi, value, other_pi = network.apply(train_state.params, hstate, ac_in)

                acting_pi = _scale_partner_logits(pi, partner_mask, model_config["PARTNER_MIX_EPS"])
                action = acting_pi.sample(seed=_rng)
                log_prob = acting_pi.log_prob(action)

                env_act = unbatchify(action, env.agents, model_config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}
                other_env_act = {
                    env.agents[0]: env_act[env.agents[1]],
                    env.agents[1]: env_act[env.agents[0]],
                }
                other_action = batchify(other_env_act, env.agents, model_config["NUM_ACTORS"]).squeeze()
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, model_config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                original_reward = jnp.array([reward[a] for a in env.agents])
                current_timestep = update_step * model_config["NUM_STEPS"] * global_num_envs
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )
                shaped_reward = jnp.array([info["shaped_reward"][a] for a in env.agents])
                combined_reward = jnp.array([reward[a] for a in env.agents])
                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward
                info["partner_mask"] = partner_mask.astype(jnp.float32)
                info = jax.tree_util.tree_map(lambda x: x.reshape((model_config["NUM_ACTORS"])), info)

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
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    train_mask,
                    other_action,
                    history_obs,
                    history_actions,
                )

                env_step_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    next_history_obs,
                    next_history_actions,
                    partner_agent_idxs,
                    rng,
                )
                return env_step_state, transition

            env_step_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                history_obs,
                history_actions,
                partner_agent_idxs,
                rng,
            )
            env_step_state, traj_batch = jax.lax.scan(_env_step, env_step_state, None, model_config["NUM_STEPS"])
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                history_obs,
                history_actions,
                partner_agent_idxs,
                rng,
            ) = env_step_state

            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape((-1,) + obs_shape)
            ac_in = (
                last_obs_batch[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
                history_obs[jnp.newaxis, :].astype(last_obs_batch.dtype),
                history_actions[jnp.newaxis, :],
            )
            _, _, last_val, _ = network.apply(train_state.params, next_initial_hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = transition.reward + model_config["GAMMA"] * next_value * (1 - transition.done) - transition.value
                    gae = delta + model_config["GAMMA"] * model_config["GAE_LAMBDA"] * (1 - transition.done) * gae
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

            def _update_epoch(update_state, epoch_idx):
                def _context_loss_fn(params, init_hstate, traj_batch):
                    hstate = init_hstate.squeeze(axis=0) if init_hstate is not None else None
                    train_mask = jax.lax.stop_gradient(traj_batch.train_mask)
                    _, _, _, other_pi = network.apply(
                        params,
                        hstate,
                        (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.hist_obs.astype(traj_batch.obs.dtype),
                            traj_batch.hist_action,
                        ),
                    )
                    other_log_prob = other_pi.log_prob(traj_batch.other_action)
                    return (-other_log_prob).mean(where=train_mask)

                def _ppo_loss_fn(params, init_hstate, traj_batch, gae, targets):
                    hstate = init_hstate.squeeze(axis=0) if init_hstate is not None else None
                    train_mask = jax.lax.stop_gradient(traj_batch.train_mask)

                    _, pi, value, other_pi = network.apply(
                        params,
                        hstate,
                        (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.hist_obs.astype(traj_batch.obs.dtype),
                            traj_batch.hist_action,
                        ),
                    )
                    log_prob = pi.log_prob(traj_batch.action)
                    other_log_prob = other_pi.log_prob(traj_batch.other_action)
                    context_loss = (-other_log_prob).mean(where=train_mask)

                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -model_config["CLIP_EPS"], model_config["CLIP_EPS"]
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=train_mask)

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean(where=train_mask)) / (gae.std(where=train_mask) + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(
                        ratio,
                        1.0 - model_config["CLIP_EPS"],
                        1.0 + model_config["CLIP_EPS"],
                    ) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean(where=train_mask)
                    entropy = pi.entropy().mean(where=train_mask)
                    ratio_mean = ratio.mean(where=train_mask)

                    total_loss = (
                        loss_actor
                        + model_config["VF_COEF"] * value_loss
                        - model_config["ENT_COEF"] * entropy
                    )
                    if not separate_context_update:
                        total_loss = total_loss + model_config["MOA_COEF"] * context_loss
                    return total_loss, (value_loss, loss_actor, entropy, ratio_mean, context_loss)

                def _context_update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch = batch_info

                    def _perform_update():
                        grad_fn = jax.value_and_grad(_context_loss_fn)
                        context_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch)
                        if env_shard_mode:
                            grads = jax.lax.pmean(grads, axis_name=env_shard_axis_name)
                            context_loss = jax.lax.pmean(context_loss, axis_name=env_shard_axis_name)
                        return train_state.apply_gradients(grads=grads), context_loss

                    def _no_op():
                        return train_state, 0.0

                    return jax.lax.cond(traj_batch.train_mask.any(), _perform_update, _no_op)

                def _ppo_update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _perform_update():
                        grad_fn = jax.value_and_grad(_ppo_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                        if env_shard_mode:
                            grads = jax.lax.pmean(grads, axis_name=env_shard_axis_name)
                            total_loss = jax.lax.pmean(total_loss, axis_name=env_shard_axis_name)
                        return train_state.apply_gradients(grads=grads), total_loss

                    def _no_op():
                        return train_state, (0.0, (0.0, 0.0, 0.0, 0.0, 0.0))

                    train_state, total_loss = jax.lax.cond(traj_batch.train_mask.any(), _perform_update, _no_op)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                hstate = init_hstate[jnp.newaxis, :] if init_hstate is not None else None
                batch = (hstate, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, model_config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], model_config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                run_context_epoch = separate_context_update and (epoch_idx < context_update_epochs)
                run_ppo_epoch = epoch_idx < model_config["UPDATE_EPOCHS"]

                def _run_context(train_state):
                    context_minibatches = (minibatches[0], minibatches[1])
                    return jax.lax.scan(_context_update_minbatch, train_state, context_minibatches)

                def _skip_context(train_state):
                    losses = jnp.zeros((model_config["NUM_MINIBATCHES"],), dtype=jnp.float32)
                    return train_state, losses

                train_state, context_losses = jax.lax.cond(
                    run_context_epoch,
                    _run_context,
                    _skip_context,
                    train_state,
                )

                def _run_ppo(train_state):
                    return jax.lax.scan(_ppo_update_minbatch, train_state, minibatches)

                def _skip_ppo(train_state):
                    zeros = (
                        jnp.zeros((model_config["NUM_MINIBATCHES"],), dtype=jnp.float32),
                        tuple(
                            jnp.zeros((model_config["NUM_MINIBATCHES"],), dtype=jnp.float32)
                            for _ in range(5)
                        ),
                    )
                    return train_state, zeros

                train_state, ppo_loss = jax.lax.cond(run_ppo_epoch, _run_ppo, _skip_ppo, train_state)
                return (train_state, init_hstate, traj_batch, advantages, targets, rng), (ppo_loss, context_losses)

            rng, _rng = jax.random.split(rng)
            update_state = (train_state, initial_hstate, traj_batch, advantages, targets, _rng)
            total_update_epochs = max(model_config["UPDATE_EPOCHS"], context_update_epochs)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, jnp.arange(total_update_epochs))
            train_state = update_state[0]
            metric = traj_batch.info

            ppo_loss_info, context_loss_info = loss_info
            total_loss, aux_data = ppo_loss_info
            value_loss, loss_actor, entropy, ratio, moa_loss = aux_data
            metric["total_loss"] = total_loss
            metric["value_loss"] = value_loss
            metric["loss_actor"] = loss_actor
            metric["entropy"] = entropy
            metric["ratio"] = ratio
            metric["moa_loss"] = moa_loss
            metric["context_loss"] = context_loss_info
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            if env_shard_mode:
                metric = jax.lax.pmean(metric, axis_name=env_shard_axis_name)

            update_step += 1
            metric["update_step"] = update_step
            metric["env_step"] = update_step * model_config["NUM_STEPS"] * global_num_envs

            def callback(metric, original_seed):
                metric.update({f"rng{int(original_seed)}/{k}": v for k, v in metric.items()})
                wandb.log(metric)

            if env_shard_mode:
                jax.lax.cond(
                    jax.lax.axis_index(env_shard_axis_name) == 0,
                    lambda _: jax.debug.callback(callback, metric, original_seed),
                    lambda _: None,
                    operand=None,
                )
            else:
                jax.debug.callback(callback, metric, original_seed)

            if num_checkpoints > 0:
                checkpoint_idx_selector = checkpoint_steps == update_step
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
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                history_obs,
                history_actions,
                partner_agent_idxs,
                rng,
            )
            return runner_state, metric

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
        num_update_steps = update_step_num_overwrite or model_config["NUM_UPDATES"]
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, num_update_steps)
        return {"runner_state": runner_state, "metrics": metric}

    return train
