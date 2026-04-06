from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import jaxmarl
import optax
import wandb
from flax.training.train_state import TrainState
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


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


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
        init_x = (
            jnp.zeros(
                (1, model_config["NUM_ENVS"], *env.observation_space().shape),
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

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])

        def _update_step(update_runner_state, unused):
            runner_state, update_step = update_runner_state

            def _env_step(env_step_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    beta_agent,
                    rng,
                    update_step,
                ) = env_step_state

                rng, policy_rng, mix_rng, step_rng = jax.random.split(rng, 4)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
                ac_in = (
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                )
                hstate, pi, value, other_pi = network.apply(train_state.params, hstate, ac_in)

                unbatched_logits = unbatchify(
                    pi.logits,
                    env.agents,
                    model_config["NUM_ENVS"],
                    env.num_agents,
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
                unbatched_logits[env.agents[0]] = jax.vmap(lambda x, y: x * y)(
                    unbatched_logits[env.agents[0]],
                    agent_0_scale,
                )
                unbatched_logits[env.agents[1]] = jax.vmap(lambda x, y: x * y)(
                    unbatched_logits[env.agents[1]],
                    agent_1_scale,
                )
                acting_logits = batchify(
                    unbatched_logits,
                    env.agents,
                    model_config["NUM_ACTORS"],
                )
                acting_pi = distrax.Categorical(logits=acting_logits)
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

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward
                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((model_config["NUM_ACTORS"])),
                    info,
                )

                done_batch = batchify(done, env.agents, model_config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
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
                )

                env_step_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    hstate,
                    beta_agent,
                    mix_rng,
                    update_step,
                )
                return env_step_state, transition

            train_state, env_state, obsv, done_batch, hstate, rng = runner_state
            initial_hstate = hstate
            rng, _rng = jax.random.split(rng)
            beta_agent = jax.random.choice(
                _rng,
                jnp.arange(env.num_agents),
                shape=(model_config["NUM_ENVS"],),
            )
            env_step_state = (
                train_state,
                env_state,
                obsv,
                done_batch,
                hstate,
                beta_agent,
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
                _beta_agent,
                rng,
                update_step,
            ) = env_step_state

            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            _, _, last_val, _ = network.apply(
                train_state.params,
                hstate,
                (last_obs_batch[jnp.newaxis, :], last_done[jnp.newaxis, :]),
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

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, gae, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        hstate = init_hstate.squeeze(axis=0)
                        _, pi, value, other_pi = network.apply(
                            params,
                            hstate,
                            (traj_batch.obs, traj_batch.done),
                        )

                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)

                        other_log_prob = other_pi.log_prob(traj_batch.other_action)
                        moa_loss = -other_log_prob.mean()

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

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
                        actor_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clip_frac = jnp.mean(
                            jnp.abs(ratio - 1.0) > model_config["CLIP_EPS"]
                        )

                        total_loss = (
                            actor_loss
                            + model_config["MOA_COEF"] * moa_loss
                            + model_config["VF_COEF"] * value_loss
                            - model_config["ENT_COEF"] * entropy
                        )
                        aux = (
                            value_loss,
                            actor_loss,
                            entropy,
                            moa_loss,
                            ratio.mean(),
                            approx_kl,
                            clip_frac,
                        )
                        return total_loss, aux

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        init_hstate,
                        traj_batch,
                        gae,
                        targets,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                hstate = init_hstate[jnp.newaxis, :]
                batch = (
                    hstate,
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
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            rng, _rng = jax.random.split(rng)
            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                _rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, model_config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            rng = update_state[-1]
            metric = traj_batch.info
            loss_metrics = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "moa_loss": loss_info[1][3],
                "ratio": loss_info[1][4],
                "approx_kl": loss_info[1][5],
                "clip_frac": loss_info[1][6],
            }
            loss_metrics = jax.tree_util.tree_map(lambda x: x.mean(), loss_metrics)
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
