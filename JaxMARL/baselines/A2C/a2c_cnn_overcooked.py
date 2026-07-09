"""
CNN A2C baseline for Overcooked v2.

This is intentionally a plain actor-critic implementation, not PPO:
there is no importance ratio, clipping objective, or PPO minibatch epoch loop.
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
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
    config.setdefault("ACTIVATION", "relu")
    config.setdefault("REW_SHAPING_HORIZON", 2_500_000.0)
    config.setdefault("NORMALIZE_ADVANTAGES", True)
    config.setdefault("NUM_SEEDS", 1)
    config.setdefault("SEED", 42)
    config.setdefault("ENTITY", "")
    config.setdefault("PROJECT", "ov2_a2c")
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
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=config["REW_SHAPING_HORIZON"],
    )

    def linear_schedule(count):
        frac = 1.0 - count / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCritic(action_dim, activation=config["ACTIVATION"])
        rng, init_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *obs_shape))
        network_params = network.init(init_rng, init_x)

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

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

        def _update_step(runner_state, _):
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, update_step, rng = runner_state
                rng, action_rng, step_rng = jax.random.split(rng, 3)

                obs_batch = batchify_obs(last_obs, env.agents, obs_shape)
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=action_rng)
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

                transition = Transition(
                    done=batchify_scalar(done, env.agents, config["NUM_ACTORS"]),
                    action=action,
                    value=value,
                    reward=batchify_scalar(reward, env.agents, config["NUM_ACTORS"]),
                    obs=obs_batch,
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
            last_obs_batch = batchify_obs(last_obs, env.agents, obs_shape)
            _, last_val = network.apply(train_state.params, last_obs_batch)

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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _loss_fn(params):
                obs_flat = traj_batch.obs.reshape((-1, *obs_shape))
                action_flat = traj_batch.action.reshape((-1,))
                advantages_flat = advantages.reshape((-1,))
                targets_flat = targets.reshape((-1,))
                if config["NORMALIZE_ADVANTAGES"]:
                    advantages_flat = (advantages_flat - advantages_flat.mean()) / (
                        advantages_flat.std() + 1e-8
                    )

                pi, value = network.apply(params, obs_flat)
                log_prob = pi.log_prob(action_flat)
                actor_loss = -(log_prob * advantages_flat).mean()
                value_loss = 0.5 * jnp.square(value - targets_flat).mean()
                entropy = pi.entropy().mean()
                total_loss = (
                    actor_loss
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
                )
                return total_loss, (actor_loss, value_loss, entropy)

            (total_loss, (actor_loss, value_loss, entropy)), grads = jax.value_and_grad(
                _loss_fn,
                has_aux=True,
            )(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), traj_batch.info)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["loss"] = total_loss
            metric["actor_loss"] = actor_loss
            metric["value_loss"] = value_loss
            metric["entropy"] = entropy

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

    alg_name = "a2c_cnn"
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
    raise NotImplementedError("A2C tuning is not wired; use explicit CLI overrides.")


@hydra.main(version_base=None, config_path="../QLearning/config", config_name="config")
def main(config):
    cfg = OmegaConf.to_container(config, resolve=True)
    if cfg.get("HYP_TUNE", False) or cfg.get("TUNE", False):
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
