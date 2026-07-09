
"""Method-level actor-critic migrations for OV2.

These trainers reuse the stable MAPPO actor model, checkpoint format, OV2
wrappers, reward shaping schedule, and evaluator contract, while swapping in
method-level actor-critic objectives/critics:

- coma_ppo: PPO actor update with COMA counterfactual Q advantage.
- maac: PPO actor update with an attention-style centralized critic.
- vtrace: IMPALA-style V-trace/A2C update with centralized critic.
- happo: HAPPO-lite sequential per-agent PPO updates with centralized critic.

The saved params are MAPPO-compatible: {"actor": actor_params, "critic": critic_params}.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
from pathlib import Path
from typing import Any, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
from overcooked_v2_experiments.mappo.models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.mappo.train import OvercookedV2WorldStateWrapper
from overcooked_v2_experiments.mappo.utils.store import store_checkpoint


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    joint_obs: jnp.ndarray
    world_state: jnp.ndarray
    other_actions: jnp.ndarray
    agent_id: jnp.ndarray
    info: Any
    train_mask: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_obs(x: dict, agent_list, obs_shape):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((-1, *obs_shape)).astype(jnp.float32)


def unbatchify_actions(x: jnp.ndarray, agent_list, num_envs: int, num_agents: int):
    x = x.reshape((num_agents, num_envs))
    return {agent: x[i] for i, agent in enumerate(agent_list)}


class CentralValueCritic(nn.Module):
    hidden_size: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state, agent_id):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = jnp.concatenate([world_state, agent_id], axis=-1)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x).squeeze(-1)


class CounterfactualQCritic(nn.Module):
    action_dim: int
    num_agents: int
    hidden_size: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state, other_actions, agent_id):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = jnp.concatenate([world_state, other_actions, agent_id], axis=-1)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class AttentionValueCritic(nn.Module):
    num_agents: int
    obs_dim: int
    hidden_size: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, joint_obs, agent_id):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        focal_idx = jnp.argmax(agent_id, axis=-1)
        focal_obs = jnp.take_along_axis(
            joint_obs,
            focal_idx[:, None, None],
            axis=1,
        ).squeeze(axis=1)
        batch_size = joint_obs.shape[0]
        flat_joint_obs = joint_obs.reshape((-1, self.obs_dim))
        tokens = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(flat_joint_obs)
        tokens = tokens.reshape((batch_size, self.num_agents, self.hidden_size))
        tokens = activation(tokens)
        query = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(focal_obs)
        flat_tokens = tokens.reshape((-1, self.hidden_size))
        key = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(flat_tokens)
        key = key.reshape((batch_size, self.num_agents, self.hidden_size))
        value = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(flat_tokens)
        value = value.reshape((batch_size, self.num_agents, self.hidden_size))
        logits = jnp.sum(query[:, None, :] * key, axis=-1) / jnp.sqrt(float(self.hidden_size))
        weights = jax.nn.softmax(logits, axis=-1)
        context = jnp.sum(weights[:, :, None] * value, axis=1)
        x = jnp.concatenate([query, context, agent_id], axis=-1)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x).squeeze(-1)


def default_config(args: argparse.Namespace) -> dict:
    return {
        "SEED": args.seed,
        "NUM_SEEDS": args.num_seeds,
        "NUM_CHECKPOINTS": 1,
        "VISUALIZE": False,
        "OPTIONAL_PREFIX": args.optional_prefix,
        "RUN_BASE_DIR": None,
        "env": {
            "ENV_NAME": "overcooked_v2",
            "ENV_KWARGS": {"layout": args.layout},
        },
        "model": {
            "TYPE": "CNN",
            "FC_DIM_SIZE": 64,
            "ACTIVATION": "relu",
            "ARCH": "cnn",
            "CNN_FEATURES": 32,
            "TOTAL_TIMESTEPS": args.total_timesteps,
            "REW_SHAPING_HORIZON": args.rew_shaping_horizon,
            "GAMMA": args.gamma,
            "GAE_LAMBDA": args.gae_lambda,
            "SCALE_CLIP_EPS": False,
            "VF_COEF": args.vf_coef,
            "MAX_GRAD_NORM": args.max_grad_norm,
            "ANNEAL_LR": True,
            "LR_WARMUP": args.lr_warmup,
            "LR": args.lr,
            "NUM_ENVS": args.num_envs,
            "NUM_STEPS": args.num_steps,
            "UPDATE_EPOCHS": args.update_epochs,
            "NUM_MINIBATCHES": args.num_minibatches,
            "CLIP_EPS": args.clip_eps,
            "ENT_COEF": args.ent_coef,
            "AC_METHOD": args.method,
            "CRITIC_HIDDEN_SIZE": args.critic_hidden_size,
            "VTRACE_RHO_CLIP": args.vtrace_rho_clip,
            "VTRACE_C_CLIP": args.vtrace_c_clip,
            "FULL_TRAIN_MASK": args.full_train_mask,
        },
        "wandb": {"WANDB_MODE": "disabled", "PROJECT": "ov2_ac_method_migrations", "ENTITY": ""},
    }


def make_train(config: dict):
    env_config = config["env"]
    model_config = config["model"]
    method = str(model_config.get("AC_METHOD", "maac"))

    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    env = OvercookedV2WorldStateWrapper(env)
    env = OvercookedV2LogWrapper(env, replace_info=False)

    obs_shape = env.observation_space().shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = int(env.action_space(env.agents[0]).n)
    num_agents = int(env.num_agents)
    num_envs = int(model_config["NUM_ENVS"])
    num_actors = num_agents * num_envs
    model_config["NUM_ACTORS"] = num_actors
    model_config["NUM_UPDATES"] = int(model_config["TOTAL_TIMESTEPS"] // model_config["NUM_STEPS"] // num_envs)
    batch_size = int(model_config["NUM_STEPS"] * num_actors)
    num_minibatches = int(model_config["NUM_MINIBATCHES"])
    if batch_size % num_minibatches != 0:
        raise ValueError("NUM_STEPS * NUM_ACTORS must be divisible by NUM_MINIBATCHES")
    minibatch_size = batch_size // num_minibatches

    checkpoint_steps = jnp.asarray([model_config["NUM_UPDATES"]], dtype=jnp.int32)
    rew_shaping_anneal = optax.linear_schedule(1.0, 0.0, model_config["REW_SHAPING_HORIZON"])

    train_idxs = jnp.linspace(0, num_agents, num_envs, dtype=jnp.int32, endpoint=False)
    train_mask_dict = {a: train_idxs == i for i, a in enumerate(env.agents)}
    train_mask_flat = batchify(train_mask_dict, env.agents, num_actors).squeeze()
    if bool(model_config.get("FULL_TRAIN_MASK", False)):
        train_mask_flat = jnp.ones((num_actors,), dtype=jnp.bool_)
    agent_id_flat = jax.nn.one_hot(
        jnp.repeat(jnp.arange(num_agents), num_envs),
        num_agents,
        dtype=jnp.float32,
    )

    def create_lr():
        steps_per_epoch = num_minibatches * int(model_config["UPDATE_EPOCHS"])
        warmup_steps = int(model_config["LR_WARMUP"] * model_config["NUM_UPDATES"])
        warmup = optax.linear_schedule(0.0, model_config["LR"], max(1, warmup_steps * steps_per_epoch))
        decay = optax.cosine_decay_schedule(model_config["LR"], max(1, (model_config["NUM_UPDATES"] - warmup_steps) * steps_per_epoch))
        return optax.join_schedules([warmup, decay], [warmup_steps * steps_per_epoch])

    def _world_state_features(obs):
        return obs["world_state"].swapaxes(0, 1).reshape((num_actors, -1)).astype(jnp.float32)

    def _joint_obs_features(obs_batch):
        obs_by_agent = obs_batch.astype(jnp.float32).reshape((num_agents, num_envs, *obs_shape))
        env_joint = obs_by_agent.swapaxes(0, 1).reshape((num_envs, num_agents, obs_dim))
        return jnp.repeat(env_joint[jnp.newaxis, ...], num_agents, axis=0).reshape((num_actors, num_agents, obs_dim))

    def _other_actions(action_flat):
        action_by_agent = action_flat.reshape((num_agents, num_envs))
        joint_onehot = jax.nn.one_hot(action_by_agent, action_dim, dtype=jnp.float32)
        other_joint = jnp.stack(
            [joint_onehot.at[i].set(jnp.zeros_like(joint_onehot[i])) for i in range(num_agents)],
            axis=0,
        )
        return other_joint.transpose((0, 2, 1, 3)).reshape((num_actors, num_agents * action_dim))

    def _flatten_info_leaf(x):
        x = jnp.asarray(x)
        if x.size == num_actors:
            return x.reshape((num_actors,))
        if x.size == num_envs:
            return jnp.tile(x.reshape((num_envs,)), num_agents)
        return jnp.zeros((num_actors,), dtype=x.dtype)

    def _critic_init_and_apply(method_name: str, critic_rng, world_dim):
        hidden = int(model_config["CRITIC_HIDDEN_SIZE"])
        activation = str(model_config["ACTIVATION"])
        if method_name == "coma_ppo":
            critic = CounterfactualQCritic(action_dim, num_agents, hidden, activation)
            params = critic.init(
                critic_rng,
                jnp.zeros((1, world_dim), dtype=jnp.float32),
                jnp.zeros((1, num_agents * action_dim), dtype=jnp.float32),
                jnp.zeros((1, num_agents), dtype=jnp.float32),
            )
        elif method_name == "maac":
            critic = AttentionValueCritic(num_agents, obs_dim, hidden, activation)
            params = critic.init(
                critic_rng,
                jnp.zeros((1, num_agents, obs_dim), dtype=jnp.float32),
                jnp.zeros((1, num_agents), dtype=jnp.float32),
            )
        else:
            critic = CentralValueCritic(hidden, activation)
            params = critic.init(
                critic_rng,
                jnp.zeros((1, world_dim), dtype=jnp.float32),
                jnp.zeros((1, num_agents), dtype=jnp.float32),
            )
        return critic, params

    def train(rng):
        actor = get_actor_critic(config)
        rng, actor_rng, critic_rng, reset_rng = jax.random.split(rng, 4)
        actor_init_x = (
            jnp.zeros((1, num_envs, *obs_shape), dtype=jnp.float32),
            jnp.zeros((1, num_envs), dtype=jnp.float32),
        )
        actor_init_hstate = initialize_carry(config, num_envs)
        actor_params = actor.init(actor_rng, actor_init_hstate, actor_init_x)

        reset_rngs = jax.random.split(reset_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rngs)
        world_dim = int(_world_state_features(obsv).shape[-1])
        critic, critic_params = _critic_init_and_apply(method, critic_rng, world_dim)

        tx = optax.chain(
            optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
            optax.adam(create_lr() if model_config["ANNEAL_LR"] else model_config["LR"], eps=1e-5),
        )
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=tx)
        critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=tx)
        actor_hstate = initialize_carry(config, num_actors)
        last_done = jnp.zeros((num_actors,), dtype=jnp.bool_)

        def critic_value(params, obs_batch, joint_obs, world_state, other_actions, agent_id, pi_probs=None, action=None):
            if method == "coma_ppo":
                q_all = critic.apply(params, world_state, other_actions, agent_id)
                if pi_probs is None:
                    value = jnp.max(q_all, axis=-1)
                else:
                    value = jnp.sum(pi_probs * q_all, axis=-1)
                return value, q_all
            if method == "maac":
                value = critic.apply(params, joint_obs, agent_id)
                return value, None
            value = critic.apply(params, world_state, agent_id)
            return value, None

        def _env_step(carry, _):
            actor_state, critic_state, env_state, last_obs, last_done, actor_hstate, update_step, rng = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            obs_batch = batchify_obs(last_obs, env.agents, obs_shape)
            actor_in = (obs_batch[jnp.newaxis, :], last_done[jnp.newaxis, :])
            actor_hstate, pi, _ = actor_state.apply_fn(actor_state.params, actor_hstate, actor_in)
            action = pi.sample(seed=action_rng).squeeze(axis=0)
            log_prob = pi.log_prob(action).squeeze(axis=0)
            pi_probs = pi.probs.squeeze(axis=0)
            world_state = _world_state_features(last_obs)
            joint_obs = _joint_obs_features(obs_batch)
            other_actions = _other_actions(action)
            value, _ = critic_value(
                critic_state.params,
                obs_batch,
                joint_obs,
                world_state,
                other_actions,
                agent_id_flat,
                pi_probs=pi_probs,
                action=action,
            )

            env_act = unbatchify_actions(action, env.agents, num_envs, num_agents)
            env_act = {k: v.flatten() for k, v in env_act.items()}
            step_rngs = jax.random.split(step_rng, num_envs)
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(step_rngs, env_state, env_act)
            current_timestep = update_step * model_config["NUM_STEPS"] * num_envs
            anneal = rew_shaping_anneal(current_timestep)
            reward = jax.tree_util.tree_map(lambda r, s: r + s * anneal, reward, info["shaped_reward"])
            info = jax.tree_util.tree_map(_flatten_info_leaf, info)
            done_batch = batchify(done, env.agents, num_actors).squeeze().astype(jnp.bool_)
            transition = Transition(
                global_done=jnp.tile(done["__all__"], num_agents).astype(jnp.bool_),
                done=last_done,
                action=action,
                value=value,
                reward=batchify(reward, env.agents, num_actors).squeeze(),
                log_prob=log_prob,
                obs=obs_batch,
                joint_obs=joint_obs,
                world_state=world_state,
                other_actions=other_actions,
                agent_id=agent_id_flat,
                info=info,
                train_mask=train_mask_flat,
            )
            return (actor_state, critic_state, env_state, obsv, done_batch, actor_hstate, update_step, rng), transition

        def _gae(traj, last_val):
            def step(carry, trans):
                gae, next_value = carry
                delta = trans.reward + model_config["GAMMA"] * next_value * (1 - trans.global_done) - trans.value
                gae = delta + model_config["GAMMA"] * model_config["GAE_LAMBDA"] * (1 - trans.global_done) * gae
                return (gae, trans.value), gae
            _, adv = jax.lax.scan(step, (jnp.zeros_like(last_val), last_val), traj, reverse=True, unroll=16)
            return adv, adv + traj.value

        def _vtrace(traj, values, log_probs, last_val):
            rhos = jnp.exp(log_probs - traj.log_prob)
            clipped_rhos = jnp.minimum(float(model_config["VTRACE_RHO_CLIP"]), rhos)
            cs = jnp.minimum(float(model_config["VTRACE_C_CLIP"]), rhos)
            values_tp1 = jnp.concatenate([values[1:], last_val[jnp.newaxis, :]], axis=0)
            deltas = clipped_rhos * (traj.reward + model_config["GAMMA"] * values_tp1 * (1 - traj.global_done) - values)
            def step(acc, xs):
                delta, c, done, value, value_tp1 = xs
                v = value + delta + model_config["GAMMA"] * c * (1 - done) * (acc - value_tp1)
                return v, v
            _, vs_rev = jax.lax.scan(step, last_val, (deltas, cs, traj.global_done, values, values_tp1), reverse=True)
            vs = vs_rev
            vs_tp1 = jnp.concatenate([vs[1:], last_val[jnp.newaxis, :]], axis=0)
            pg_adv = clipped_rhos * (traj.reward + model_config["GAMMA"] * vs_tp1 * (1 - traj.global_done) - values)
            return jax.lax.stop_gradient(pg_adv), jax.lax.stop_gradient(vs)

        def _update_step(carry, _):
            actor_state, critic_state, env_state, last_obs, last_done, actor_hstate, update_step, rng, checkpoint_states = carry
            initial_actor_hstate = actor_hstate
            env_carry = (actor_state, critic_state, env_state, last_obs, last_done, actor_hstate, update_step, rng)
            env_carry, traj = jax.lax.scan(_env_step, env_carry, None, model_config["NUM_STEPS"])
            actor_state, critic_state, env_state, last_obs, last_done, actor_hstate, update_step, rng = env_carry

            last_obs_batch = batchify_obs(last_obs, env.agents, obs_shape)
            actor_in = (last_obs_batch[jnp.newaxis, :], last_done[jnp.newaxis, :])
            _, last_pi, _ = actor_state.apply_fn(actor_state.params, actor_hstate, actor_in)
            last_world = _world_state_features(last_obs)
            last_joint = _joint_obs_features(last_obs_batch)
            zero_other = jnp.zeros((num_actors, num_agents * action_dim), dtype=jnp.float32)
            last_val, _ = critic_value(
                critic_state.params,
                last_obs_batch,
                last_joint,
                last_world,
                zero_other,
                agent_id_flat,
                pi_probs=last_pi.probs.squeeze(axis=0),
            )

            advantages, targets = _gae(traj, last_val)
            if method == "vtrace":
                _, current_pi, _ = actor_state.apply_fn(actor_state.params, initial_actor_hstate, (traj.obs, traj.done))
                current_logp = current_pi.log_prob(traj.action)
                values, _ = critic_value(
                    critic_state.params,
                    traj.obs.reshape((batch_size, *obs_shape)),
                    traj.joint_obs.reshape((batch_size, num_agents, obs_dim)),
                    traj.world_state.reshape((batch_size, -1)),
                    traj.other_actions.reshape((batch_size, num_agents * action_dim)),
                    traj.agent_id.reshape((batch_size, num_agents)),
                    pi_probs=None,
                )
                values = values.reshape((model_config["NUM_STEPS"], num_actors))
                advantages, targets = _vtrace(traj, values, current_logp, last_val)

            flat_batch = (
                traj.obs.reshape((batch_size, *obs_shape)),
                traj.joint_obs.reshape((batch_size, num_agents, obs_dim)),
                traj.world_state.reshape((batch_size, -1)),
                traj.other_actions.reshape((batch_size, num_agents * action_dim)),
                traj.agent_id.reshape((batch_size, num_agents)),
                traj.action.reshape((batch_size,)),
                traj.log_prob.reshape((batch_size,)),
                traj.value.reshape((batch_size,)),
                advantages.reshape((batch_size,)),
                targets.reshape((batch_size,)),
                traj.done.reshape((batch_size,)),
                traj.train_mask.reshape((batch_size,)),
            )

            def loss_fn(params, minibatch, agent_mask=None):
                actor_params, critic_params = params
                obs_mb, joint_mb, world_mb, other_mb, agent_id_mb, action_mb, old_logp_mb, old_value_mb, adv_mb, target_mb, done_mb, train_mask_mb = minibatch
                if agent_mask is not None:
                    train_mask_mb = jnp.logical_and(train_mask_mb, agent_mask)
                _, pi, _ = actor_state.apply_fn(actor_params, None, (obs_mb[jnp.newaxis, :], done_mb[jnp.newaxis, :]))
                logp = pi.log_prob(action_mb).squeeze(axis=0)
                probs = pi.probs.squeeze(axis=0)
                value, q_all = critic_value(critic_params, obs_mb, joint_mb, world_mb, other_mb, agent_id_mb, pi_probs=probs, action=action_mb)
                entropy = pi.entropy().squeeze(axis=0).mean(where=train_mask_mb)

                if method == "coma_ppo":
                    chosen_q = jnp.take_along_axis(q_all, action_mb[:, None], axis=-1).squeeze(-1)
                    baseline = jnp.sum(probs * q_all, axis=-1)
                    actor_adv = jax.lax.stop_gradient(chosen_q - baseline)
                    critic_target = target_mb
                    critic_pred = chosen_q
                else:
                    actor_adv = adv_mb
                    critic_target = target_mb
                    critic_pred = value

                actor_adv = (actor_adv - actor_adv.mean(where=train_mask_mb)) / (actor_adv.std(where=train_mask_mb) + 1e-8)
                ratio = jnp.exp(logp - old_logp_mb)
                if method == "vtrace":
                    actor_loss = -(logp * jax.lax.stop_gradient(actor_adv)).mean(where=train_mask_mb)
                else:
                    unclipped = ratio * actor_adv
                    clipped = jnp.clip(ratio, 1.0 - model_config["CLIP_EPS"], 1.0 + model_config["CLIP_EPS"]) * actor_adv
                    actor_loss = -jnp.minimum(unclipped, clipped).mean(where=train_mask_mb)
                critic_loss = 0.5 * jnp.square(critic_pred - critic_target).mean(where=train_mask_mb)
                total = actor_loss + model_config["VF_COEF"] * critic_loss - model_config["ENT_COEF"] * entropy
                return total, (actor_loss, critic_loss, entropy, ratio.mean(where=train_mask_mb))

            def update_minibatch(states, minibatch):
                actor_state, critic_state = states
                def apply_one(states, agent_idx):
                    actor_state, critic_state = states
                    agent_mask = None
                    if method == "happo":
                        agent_idx_arr = jnp.argmax(minibatch[4], axis=-1)
                        agent_mask = agent_idx_arr == agent_idx
                    grad_fn = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=True)
                    (loss, aux), grads = grad_fn((actor_state.params, critic_state.params), minibatch, agent_mask)
                    actor_grads, critic_grads = grads[0]
                    return (actor_state.apply_gradients(grads=actor_grads), critic_state.apply_gradients(grads=critic_grads)), (loss, *aux)
                if method == "happo":
                    (actor_state, critic_state), stats = jax.lax.scan(apply_one, (actor_state, critic_state), jnp.arange(num_agents))
                    stats = jax.tree_util.tree_map(lambda x: x.mean(axis=0), stats)
                    return (actor_state, critic_state), stats
                return apply_one((actor_state, critic_state), jnp.asarray(0, dtype=jnp.int32))

            def update_epoch(carry, _):
                states, rng = carry
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, batch_size)
                def shuffle(x):
                    x = jnp.take(x, perm, axis=0)
                    return x.reshape((num_minibatches, minibatch_size) + x.shape[1:])
                minibatches = jax.tree_util.tree_map(shuffle, flat_batch)
                states, stats = jax.lax.scan(update_minibatch, states, minibatches)
                stats = jax.tree_util.tree_map(lambda x: x.mean(axis=0), stats)
                return (states, rng), stats

            rng, upd_rng = jax.random.split(rng)
            (states, _), epoch_stats = jax.lax.scan(update_epoch, ((actor_state, critic_state), upd_rng), None, int(model_config["UPDATE_EPOCHS"]))
            actor_state, critic_state = states
            update_step += 1
            loss, actor_loss, critic_loss, entropy, ratio = jax.tree_util.tree_map(lambda x: x.mean(), epoch_stats)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), traj.info)
            metric["loss"] = loss
            metric["actor_loss"] = actor_loss
            metric["critic_loss"] = critic_loss
            metric["entropy"] = entropy
            metric["ratio"] = ratio
            metric["update_step"] = update_step
            metric["env_step"] = update_step * model_config["NUM_STEPS"] * num_envs

            checkpoint_states = jax.lax.cond(
                jnp.any(checkpoint_steps == update_step),
                lambda c: jax.tree_util.tree_map(lambda x, y: x.at[0].set(y), c, {"actor": actor_state.params, "critic": critic_state.params}),
                lambda c: c,
                checkpoint_states,
            )
            return (actor_state, critic_state, env_state, last_obs, last_done, actor_hstate, update_step, rng, checkpoint_states), metric

        init_ckpt = jax.tree_util.tree_map(lambda p: jnp.zeros((1,) + p.shape, p.dtype), {"actor": actor_state.params, "critic": critic_state.params})
        runner_state = (actor_state, critic_state, env_state, obsv, last_done, actor_hstate, jnp.asarray(0), rng, init_ckpt)
        runner_state, metrics = jax.lax.scan(update_step := _update_step, runner_state, None, model_config["NUM_UPDATES"])
        actor_state, critic_state, *_rest, checkpoint_states = runner_state
        return {"runner_state": ((actor_state, critic_state), checkpoint_states), "metrics": metrics}

    return train


def run_training(args: argparse.Namespace) -> Path:
    config = default_config(args)
    run_dir = Path(args.output_root) / args.optional_prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    config["RUN_BASE_DIR"] = run_dir
    OmegaConf.save(config, run_dir / "config.yaml")
    rngs = jax.random.split(jax.random.PRNGKey(args.seed), args.num_seeds)
    train_vmap = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vmap(rngs))
    checkpoints = outs["runner_state"][1]
    for i in range(args.num_seeds):
        params = jax.tree_util.tree_map(lambda x: x[i][0], checkpoints)
        store_checkpoint(config, params, i, 0, final=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("coma_ppo", "maac", "vtrace", "happo"))
    parser.add_argument("--output_root", default="runs/ac_method_migrations")
    parser.add_argument("--optional_prefix", required=True)
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=2)
    parser.add_argument("--total_timesteps", type=int, default=1024)
    parser.add_argument("--rew_shaping_horizon", type=int, default=512)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--update_epochs", type=int, default=1)
    parser.add_argument("--num_minibatches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--lr_warmup", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--critic_hidden_size", type=int, default=128)
    parser.add_argument("--vtrace_rho_clip", type=float, default=1.0)
    parser.add_argument("--vtrace_c_clip", type=float, default=1.0)
    parser.add_argument("--full_train_mask", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    out = run_training(parse_args())
    print(f"[ac-method] wrote {out}", flush=True)
