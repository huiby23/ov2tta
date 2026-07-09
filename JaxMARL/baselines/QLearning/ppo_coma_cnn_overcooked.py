"""PPO-COMA hybrid baseline for Overcooked v2.

This migration keeps the native OV2 CNN actor-critic bottom and COMA
counterfactual critic, but stabilizes the actor update with PPO-style old
log-prob tracking, clipped importance ratios, minibatches, and update epochs.
The saved checkpoint layout remains compatible with the existing actor-critic
partner evaluators: policy evaluation loads ``params["actor"]`` and ignores the
counterfactual critic.
"""

from __future__ import annotations

import copy
import os
from typing import NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import LogWrapper


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return activation(x)


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNN(self.activation)(x)

        actor_hidden = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        actor_hidden = activation(actor_hidden)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_hidden)
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class CounterfactualCritic(nn.Module):
    action_dim: int
    num_agents: int
    hidden_size: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, state, other_actions, agent_id):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = jnp.concatenate([state, other_actions, agent_id], axis=-1)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    state: jnp.ndarray
    other_actions: jnp.ndarray
    agent_id: jnp.ndarray
    info: dict


def _action_dim(env) -> int:
    try:
        return int(env.action_space().n)
    except TypeError:
        return int(env.action_space(env.agents[0]).n)


def _obs_shape(env):
    try:
        return env.observation_space().shape
    except TypeError:
        return env.observation_space(env.agents[0]).shape


def batchify_scalar(x: dict, agent_list, num_actors: int):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors,))


def batchify_obs(x: dict, agent_list, obs_shape):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((-1, *obs_shape))


def batchify_state(x: dict, agent_list):
    state = jnp.concatenate(
        [x[a].reshape((x[a].shape[0], -1)) for a in agent_list],
        axis=-1,
    )
    return state


def unbatchify_actions(x: jnp.ndarray, agent_list, num_envs: int, num_agents: int):
    x = x.reshape((num_agents, num_envs))
    return {agent: x[i] for i, agent in enumerate(agent_list)}


def env_from_config(config):
    env_name = config["ENV_NAME"]
    env_kwargs = copy.deepcopy(config["ENV_KWARGS"])
    if "overcooked_v2" in env_name.lower():
        resolved_name = f"{env_name}_{env_kwargs['layout']}"
        env = jaxmarl.make(env_name, **env_kwargs)
    elif "overcooked" in env_name.lower():
        resolved_name = f"{env_name}_{env_kwargs['layout']}"
        env_kwargs["layout"] = overcooked_layouts[env_kwargs["layout"]]
        env = jaxmarl.make(env_name, **env_kwargs)
    else:
        resolved_name = env_name
        env = jaxmarl.make(env_name, **env_kwargs)
    return LogWrapper(env, replace_info=False), resolved_name


def apply_defaults(config):
    config = dict(config)
    if isinstance(config.get("alg"), dict):
        config = {**config, **config["alg"]}
    config.setdefault("TOTAL_TIMESTEPS", 10_000_000)
    config.setdefault("NUM_ENVS", 64)
    config.setdefault("NUM_STEPS", 128)
    config.setdefault("LR", 2.5e-4)
    config.setdefault("ANNEAL_LR", True)
    config.setdefault("MAX_GRAD_NORM", 0.5)
    config.setdefault("GAMMA", 0.99)
    config.setdefault("GAE_LAMBDA", 0.95)
    config.setdefault("VF_COEF", 0.5)
    config.setdefault("ENT_COEF", 0.01)
    config.setdefault("COMA_CRITIC_COEF", 0.5)
    config.setdefault("COMA_CRITIC_HIDDEN_SIZE", 128)
    config.setdefault("CLIP_EPS", 0.2)
    config.setdefault("UPDATE_EPOCHS", 4)
    config.setdefault("NUM_MINIBATCHES", 16)
    config.setdefault("ACTIVATION", "relu")
    config.setdefault("REW_SHAPING_HORIZON", 2_500_000.0)
    config.setdefault("NORMALIZE_ADVANTAGES", True)
    config.setdefault("NUM_SEEDS", 1)
    config.setdefault("SEED", 42)
    config.setdefault("ENTITY", "")
    config.setdefault("PROJECT", "ov2_ppo_coma")
    config.setdefault("WANDB_MODE", "disabled")
    config.setdefault("WANDB_LOG_ALL_SEEDS", False)
    config.setdefault("SAVE_PATH", None)
    return config


def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    if config["NUM_UPDATES"] <= 0:
        raise ValueError("TOTAL_TIMESTEPS must be at least NUM_STEPS * NUM_ENVS")

    obs_shape = _obs_shape(env)
    action_dim = _action_dim(env)
    state_dim = int(np.prod(obs_shape)) * env.num_agents
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=config["REW_SHAPING_HORIZON"],
    )
    batch_size = config["NUM_STEPS"] * config["NUM_ACTORS"]
    num_minibatches = int(config["NUM_MINIBATCHES"])
    if batch_size % num_minibatches != 0:
        raise ValueError("NUM_STEPS * NUM_ACTORS must be divisible by NUM_MINIBATCHES")
    minibatch_size = batch_size // num_minibatches

    def linear_schedule(count):
        frac = 1.0 - count / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCritic(action_dim, activation=config["ACTIVATION"])
        critic = CounterfactualCritic(
            action_dim=action_dim,
            num_agents=env.num_agents,
            hidden_size=int(config["COMA_CRITIC_HIDDEN_SIZE"]),
            activation=config["ACTIVATION"],
        )
        rng, init_rng, critic_rng = jax.random.split(rng, 3)
        init_x = jnp.zeros((1, *obs_shape))
        actor_params = network.init(init_rng, init_x)
        critic_params = critic.init(
            critic_rng,
            jnp.zeros((1, state_dim)),
            jnp.zeros((1, env.num_agents * action_dim)),
            jnp.zeros((1, env.num_agents)),
        )
        network_params = {"actor": actor_params, "critic": critic_params}

        lr = linear_schedule if config["ANNEAL_LR"] else config["LR"]
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(lr, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        def counterfactual_inputs(obs_dict, action_flat):
            state = batchify_state(obs_dict, env.agents)
            states = jnp.repeat(
                state[jnp.newaxis, ...],
                env.num_agents,
                axis=0,
            ).reshape((config["NUM_ACTORS"], state_dim))

            action_by_agent = action_flat.reshape((env.num_agents, config["NUM_ENVS"]))
            joint_onehot = jax.nn.one_hot(action_by_agent, action_dim)
            other_joint = jnp.stack(
                [
                    joint_onehot.at[i].set(jnp.zeros_like(joint_onehot[i]))
                    for i in range(env.num_agents)
                ],
                axis=0,
            )
            other_joint = other_joint.transpose((0, 2, 1, 3)).reshape(
                (config["NUM_ACTORS"], env.num_agents * action_dim)
            )
            agent_id = jax.nn.one_hot(
                jnp.repeat(jnp.arange(env.num_agents), config["NUM_ENVS"]),
                env.num_agents,
                dtype=jnp.float32,
            )
            return states, other_joint, agent_id

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

        def _update_step(runner_state, _):
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, update_step, rng = runner_state
                rng, action_rng, step_rng = jax.random.split(rng, 3)

                obs_batch = batchify_obs(last_obs, env.agents, obs_shape)
                pi, _ = network.apply(train_state.params["actor"], obs_batch)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)
                cf_state, cf_other_actions, cf_agent_id = counterfactual_inputs(
                    last_obs,
                    action,
                )
                q_all = critic.apply(
                    train_state.params["critic"],
                    cf_state,
                    cf_other_actions,
                    cf_agent_id,
                )
                value = jnp.sum(jax.nn.softmax(pi.logits, axis=-1) * q_all, axis=-1)
                env_act = unbatchify_actions(
                    action,
                    env.agents,
                    config["NUM_ENVS"],
                    env.num_agents,
                )

                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step,
                    in_axes=(0, 0, 0),
                )(step_rngs, env_state, env_act)

                shaped_reward = info.pop("shaped_reward")
                current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                reward = jax.tree.map(
                    lambda r, s: r + s * rew_shaping_anneal(current_timestep),
                    reward,
                    shaped_reward,
                )
                team_reward = jnp.stack([reward[a] for a in env.agents], axis=0).sum(axis=0)

                transition = Transition(
                    done=batchify_scalar(done, env.agents, config["NUM_ACTORS"]),
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=jnp.tile(team_reward, env.num_agents),
                    obs=obs_batch,
                    state=cf_state,
                    other_actions=cf_other_actions,
                    agent_id=cf_agent_id,
                    info=jax.tree.map(
                        lambda x: x.reshape((config["NUM_ACTORS"],)),
                        info,
                    ),
                )
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                config["NUM_STEPS"],
            )

            train_state, env_state, last_obs, update_step, rng = runner_state
            last_val = jnp.zeros((config["NUM_ACTORS"],), dtype=jnp.float32)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1 - transition.done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1 - transition.done)
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

            _, targets = _calculate_gae(traj_batch, last_val)

            flat_batch = (
                traj_batch.obs.reshape((batch_size, *obs_shape)),
                traj_batch.action.reshape((batch_size,)),
                traj_batch.log_prob.reshape((batch_size,)),
                traj_batch.state.reshape((batch_size, state_dim)),
                traj_batch.other_actions.reshape((batch_size, env.num_agents * action_dim)),
                traj_batch.agent_id.reshape((batch_size, env.num_agents)),
                targets.reshape((batch_size,)),
            )

            def _loss_fn(params, minibatch):
                (
                    obs_mb,
                    action_mb,
                    old_log_prob_mb,
                    state_mb,
                    other_actions_mb,
                    agent_id_mb,
                    targets_mb,
                ) = minibatch

                pi, _ = network.apply(params["actor"], obs_mb)
                q_all = critic.apply(
                    params["critic"],
                    state_mb,
                    other_actions_mb,
                    agent_id_mb,
                )
                chosen_q = jnp.take_along_axis(
                    q_all,
                    action_mb[:, jnp.newaxis],
                    axis=-1,
                ).squeeze(axis=-1)
                baseline = jnp.sum(jax.nn.softmax(pi.logits, axis=-1) * q_all, axis=-1)
                coma_advantages = jax.lax.stop_gradient(chosen_q - baseline)
                if config["NORMALIZE_ADVANTAGES"]:
                    coma_advantages = (
                        coma_advantages - coma_advantages.mean()
                    ) / (coma_advantages.std() + 1e-8)

                log_prob = pi.log_prob(action_mb)
                ratio = jnp.exp(log_prob - old_log_prob_mb)
                unclipped = ratio * coma_advantages
                clipped = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * coma_advantages
                )
                actor_loss = -jnp.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * jnp.square(chosen_q - targets_mb).mean()
                entropy = pi.entropy().mean()
                approx_kl = (old_log_prob_mb - log_prob).mean()
                clip_frac = (jnp.abs(ratio - 1.0) > config["CLIP_EPS"]).mean()
                total_loss = (
                    actor_loss
                    + config["COMA_CRITIC_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
                )
                return total_loss, (actor_loss, value_loss, entropy, approx_kl, clip_frac)

            def _update_minibatch(train_state, minibatch):
                (total_loss, aux), grads = jax.value_and_grad(
                    _loss_fn,
                    has_aux=True,
                )(train_state.params, minibatch)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (total_loss, *aux)

            def _update_epoch(carry, _):
                train_state, epoch_rng = carry
                epoch_rng, perm_rng = jax.random.split(epoch_rng)
                permutation = jax.random.permutation(perm_rng, batch_size)

                def _shuffle(x):
                    x = jnp.take(x, permutation, axis=0)
                    return x.reshape((num_minibatches, minibatch_size) + x.shape[1:])

                minibatches = jax.tree.map(_shuffle, flat_batch)
                train_state, minibatch_stats = jax.lax.scan(
                    _update_minibatch,
                    train_state,
                    minibatches,
                )
                epoch_stats = jax.tree.map(lambda x: x.mean(), minibatch_stats)
                return (train_state, epoch_rng), epoch_stats

            rng, ppo_rng = jax.random.split(rng)
            (train_state, _), epoch_stats = jax.lax.scan(
                _update_epoch,
                (train_state, ppo_rng),
                None,
                int(config["UPDATE_EPOCHS"]),
            )
            total_loss, actor_loss, value_loss, entropy, approx_kl, clip_frac = (
                jax.tree.map(lambda x: x.mean(), epoch_stats)
            )

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), traj_batch.info)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["loss"] = total_loss
            metric["actor_loss"] = actor_loss
            metric["value_loss"] = value_loss
            metric["entropy"] = entropy
            metric["approx_kl"] = approx_kl
            metric["clip_frac"] = clip_frac

            if config["WANDB_MODE"] != "disabled":

                def callback(metric):
                    wandb.log(metric, step=metric["update_step"])

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, loop_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.asarray(0), loop_rng)
        runner_state, metrics = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            config["NUM_UPDATES"],
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):
    config = apply_defaults(OmegaConf.to_container(config, resolve=True))
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = "ppo_coma_cnn"
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml',
            ),
        )

        for i, _ in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    raise NotImplementedError("PPO-COMA tuning is not wired; use explicit CLI overrides.")


@hydra.main(version_base=None, config_path="../QLearning/config", config_name="config")
def main(config):
    cfg = OmegaConf.to_container(config, resolve=True)
    if cfg.get("HYP_TUNE", False) or cfg.get("TUNE", False):
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
