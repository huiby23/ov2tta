from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import jaxmarl
import optax
import wandb
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from overcooked_v2_experiments.e3t_cec.models.model import (
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
    agent_positions: jnp.ndarray
    other_action: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _get_agent_positions(env_state, env, num_actors):
    positions = env_state.env_state.agents.pos.to_array()
    pos_dict = {agent: positions[:, i, :] for i, agent in enumerate(env.agents)}
    return batchify(pos_dict, env.agents, num_actors)


def make_train(config, update_step_offset=None, update_step_num_overwrite=None, population_config=None):
    if population_config is not None:
        raise NotImplementedError("E3T-CEC does not use external partner populations.")

    env_config = config["env"]
    model_config = config["model"]
    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    if env.num_agents != 2:
        raise NotImplementedError("E3T-CEC currently assumes two-player coordination.")

    obs_shape = tuple(env.observation_space().shape)
    flattened_obs_dim = int(jnp.prod(jnp.array(obs_shape)))
    env_config.setdefault("ENV_KWARGS", {})["obs_shape"] = obs_shape
    model_config["OBS_SHAPE"] = obs_shape
    model_config["ENV_NAME"] = env_config["ENV_NAME"]
    model_config["LAYOUT_NAME"] = env_config["ENV_KWARGS"]["layout"]

    model_config["NUM_ACTORS"] = env.num_agents * model_config["NUM_ENVS"]
    global_num_envs = config.get("ENV_SHARD_TOTAL_ENVS", model_config["NUM_ENVS"])
    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"] // model_config["NUM_STEPS"] // global_num_envs
    )
    model_config["MAX_TRAIN_UPDATES"] = (
        model_config.get("MAX_TRAIN_STEPS", model_config["TOTAL_TIMESTEPS"])
        // model_config["NUM_STEPS"]
        // global_num_envs
    )
    reward_shaping_horizon = model_config.get("REW_SHAPING_HORIZON")
    if reward_shaping_horizon is not None:
        model_config["NUM_REWARD_SHAPING_STEPS"] = max(
            int(reward_shaping_horizon) // (model_config["NUM_STEPS"] * global_num_envs),
            1,
        )
    else:
        model_config["NUM_REWARD_SHAPING_STEPS"] = max(
            model_config["MAX_TRAIN_UPDATES"] // 2, 1
        )
    model_config["MINIBATCH_SIZE"] = (
        model_config["NUM_ACTORS"] * model_config["NUM_STEPS"] // model_config["NUM_MINIBATCHES"]
    )
    model_config["CLIP_EPS"] = (
        model_config["CLIP_EPS"] / env.num_agents
        if model_config["SCALE_CLIP_EPS"]
        else model_config["CLIP_EPS"]
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

    resume_update_step = (update_step_offset or 0) * (
        model_config["NUM_MINIBATCHES"] * model_config["UPDATE_EPOCHS"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (model_config["NUM_MINIBATCHES"] * model_config["UPDATE_EPOCHS"]))
            / model_config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return model_config["LR"] * frac

    def train(rng, population=None, initial_train_state=None):
        if population is not None:
            raise NotImplementedError("E3T-CEC does not use external partner populations.")

        original_seed = rng[0]
        network = get_actor_critic(config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, model_config["NUM_ACTORS"], flattened_obs_dim), dtype=jnp.float32),
            jnp.zeros((1, model_config["NUM_ACTORS"]), dtype=jnp.bool_),
            jnp.zeros((1, model_config["NUM_ACTORS"], 2), dtype=jnp.int32),
        )
        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])
        network_params = network.init(_rng, init_hstate, init_x)

        if model_config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
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

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])

        def _update_step(update_runner_state, unused):
            runner_state, update_step = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step, beta_agent = runner_state

                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, model_config["NUM_ACTORS"])
                agent_positions = _get_agent_positions(env_state, env, model_config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                    agent_positions[jnp.newaxis, :],
                )
                hstate, pi, value, other_pi = network.apply(train_state.params, hstate, ac_in)

                unbatched_logits = unbatchify(
                    pi.logits, env.agents, model_config["NUM_ENVS"], env.num_agents
                )
                agent_0_scale = jnp.where(
                    beta_agent == 0,
                    model_config["PARTNER_MIX_EPS"],
                    1.0,
                )
                agent_1_scale = jnp.where(
                    beta_agent == 1,
                    model_config["PARTNER_MIX_EPS"],
                    1.0,
                )
                unbatched_logits["agent_0"] = jax.vmap(lambda x, y: x * y)(
                    unbatched_logits["agent_0"], agent_0_scale
                )
                unbatched_logits["agent_1"] = jax.vmap(lambda x, y: x * y)(
                    unbatched_logits["agent_1"], agent_1_scale
                )
                batched_logits = batchify(
                    unbatched_logits, env.agents, model_config["NUM_ACTORS"]
                )
                acting_pi = distrax.Categorical(logits=batched_logits)
                action = acting_pi.sample(seed=_rng)
                log_prob = acting_pi.log_prob(action)

                env_act = unbatchify(
                    action, env.agents, model_config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze(-1) for k, v in env_act.items()}
                other_env_act = {
                    env.agents[0]: env_act[env.agents[1]],
                    env.agents[1]: env_act[env.agents[0]],
                }
                other_action = batchify(
                    other_env_act, env.agents, model_config["NUM_ACTORS"]
                ).squeeze()

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, model_config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                reward_shaping_frac = jnp.maximum(
                    0.0,
                    1.0 - (update_step / model_config["NUM_REWARD_SHAPING_STEPS"]),
                )
                original_reward = jnp.array([reward[a] for a in env.agents])
                shaped_reward = jnp.array([info["shaped_reward"][a] for a in env.agents])
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * reward_shaping_frac,
                    reward,
                    info["shaped_reward"],
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, reward_shaping_frac)
                info["combined_reward"] = combined_reward
                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((model_config["NUM_ACTORS"])),
                    info,
                )
                done_batch = batchify(done, env.agents, model_config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    agent_positions,
                    other_action,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    hstate,
                    rng,
                    update_step,
                    beta_agent,
                )
                return runner_state, transition

            initial_hstate = runner_state[-2]
            train_state, env_state, obsv, done_batch, hstate, rng = runner_state
            beta_agent = jax.random.choice(
                rng,
                jnp.arange(env.num_agents),
                shape=(model_config["NUM_ENVS"],),
            )
            rng, _rng = jax.random.split(rng)
            runner_state = (
                train_state,
                env_state,
                obsv,
                done_batch,
                hstate,
                rng,
                update_step,
                beta_agent,
            )
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, model_config["NUM_STEPS"]
            )

            train_state, env_state, last_obs, last_done, hstate, rng, update_step, _ = runner_state
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            last_obs_batch = batchify(last_obs, env.agents, model_config["NUM_ACTORS"])
            agent_positions = _get_agent_positions(env_state, env, model_config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
                agent_positions[jnp.newaxis, :],
            )
            _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + model_config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + model_config["GAMMA"] * model_config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value, other_pi = network.apply(
                            params,
                            jax.tree_util.tree_map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        other_log_prob = other_pi.log_prob(traj_batch.other_action)
                        moa_nll_loss = -jnp.mean(other_log_prob)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - model_config["CLIP_EPS"],
                                1.0 + model_config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > model_config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + model_config["MOA_COEF"] * moa_nll_loss
                            + model_config["VF_COEF"] * value_loss
                            - model_config["ENT_COEF"] * entropy
                        )
                        aux = (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )
                        return total_loss, aux

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jax.tree_util.tree_map(
                    lambda h: jnp.reshape(h, (1, model_config["NUM_ACTORS"], -1)),
                    init_hstate,
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, model_config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1),
                    batch,
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], model_config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    jax.tree_util.tree_map(lambda h: h.squeeze(), init_hstate),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, model_config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metric = traj_batch.info
            loss_metrics = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": loss_info[1][3].at[0, 0].get(),
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            loss_metrics = jax.tree_util.tree_map(lambda x: x.mean(), loss_metrics)
            for key, value in loss_metrics.items():
                metric[key] = value
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            rng = update_state[-1]

            update_step = update_step + 1
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
            )

            def callback(metric, original_seed):
                metric = {f"rng{int(original_seed)}/{k}": v for k, v in metric.items()}
                wandb.log(metric)

            jax.debug.callback(callback, metric, original_seed)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
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
                rng,
            ) = runner_state
            (next_runner_state, next_update_step), metric = _update_step(
                ((train_state, env_state, obsv, done_batch, hstate, rng), update_step),
                unused,
            )
            train_state, env_state, obsv, done_batch, hstate, rng = next_runner_state
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
