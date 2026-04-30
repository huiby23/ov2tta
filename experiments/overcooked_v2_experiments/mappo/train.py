"""
MAPPO for OvercookedV2.

The implementation keeps the existing PPO/IPPO experiment semantics where
possible, including reward shaping, checkpointing, multi-seed execution, and
population-based training hooks. The main architectural change is that the actor
remains decentralised while the critic consumes a centralised world state.
"""

from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax import core
from flax.training.train_state import TrainState
from jaxmarl.environments.overcooked_v2.overcooked import ObservationType
from jaxmarl.wrappers.baselines import JaxMARLWrapper, OvercookedV2LogWrapper

from overcooked_v2_experiments.eval.policy import AbstractPolicy
from .models.model import get_actor_critic, initialize_carry
from .models.rnn import ScannedRNN


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    train_mask: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _flatten_state_tree(tree) -> jnp.ndarray:
    leaves = []
    for leaf in jax.tree_util.tree_leaves(tree):
        if leaf is None:
            continue
        leaves.append(jnp.ravel(jnp.asarray(leaf, dtype=jnp.float32)))
    return jnp.concatenate(leaves, axis=0)


class OvercookedV2WorldStateWrapper(JaxMARLWrapper):
    """Adds a shared flattened environment state for each agent."""

    def __init__(self, env):
        super().__init__(env)
        _, init_state = env.reset(jax.random.PRNGKey(0))
        self._world_state_size = int(_flatten_state_tree(init_state).shape[0])

    @staticmethod
    def _repeat_world_state(env, state):
        world_state = _flatten_state_tree(state)
        return jnp.repeat(world_state[jnp.newaxis, :], env.num_agents, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self._repeat_world_state(self._env, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self._repeat_world_state(self._env, env_state)
        return obs, env_state, reward, done, info

    def world_state_size(self) -> int:
        return self._world_state_size


class CentralizedCriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x

        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        embedding = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = activation(embedding)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, jnp.squeeze(critic, axis=-1)


class CentralizedCriticCNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, _dones = x

        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        critic = activation(critic)
        critic = nn.LayerNorm()(critic)
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, jnp.squeeze(critic, axis=-1)


def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
):
    env_config = config["env"]
    model_config = config["model"]


    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    env = OvercookedV2WorldStateWrapper(env)
    env = OvercookedV2LogWrapper(env, replace_info=False)

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

    def _current_params(train_states):
        actor_train_state, critic_train_state = train_states
        return {
            "actor": actor_train_state.params,
            "critic": critic_train_state.params,
        }

    def _update_checkpoint(checkpoint_states, train_states, i):
        params = _current_params(train_states)
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

    train_idxs = jnp.linspace(
        0,
        env.num_agents,
        model_config["NUM_ENVS"],
        dtype=jnp.int32,
        endpoint=False,
    )
    train_mask_dict = {a: train_idxs == i for i, a in enumerate(env.agents)}
    train_mask_flat = batchify(train_mask_dict, env.agents, model_config["NUM_ACTORS"]).squeeze()

    use_population_annealing = False
    if "POPULATION_ANNEAL_HORIZON" in config:
        use_population_annealing = True
        transition_begin = config.get("POPULATION_ANNEAL_BEGIN", 0)
        anneal_horizon = config["POPULATION_ANNEAL_HORIZON"]
        if anneal_horizon == 0:
            population_annealing_schedule = optax.constant_schedule(1.0)
        else:
            population_annealing_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=1.0,
                transition_steps=anneal_horizon - transition_begin,
                transition_begin=transition_begin,
            )

    def train(
        rng,
        population: Optional[Union[AbstractPolicy, core.FrozenDict[str, Any]]] = None,
        initial_train_state=None,
    ):
        original_seed = rng[0]

        actor_network = get_actor_critic(config)
        critic_cls = CentralizedCriticRNN if model_config["TYPE"] == "RNN" else CentralizedCriticCNN
        critic_network = critic_cls(model_config)

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        actor_init_x = (
            jnp.zeros((1, model_config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, model_config["NUM_ENVS"])),
        )
        actor_init_hstate = initialize_carry(config, model_config["NUM_ENVS"])
        actor_params = actor_network.init(actor_rng, actor_init_hstate, actor_init_x)

        critic_init_x = (
            jnp.zeros((1, model_config["NUM_ENVS"], env.world_state_size())),
            jnp.zeros((1, model_config["NUM_ENVS"])),
        )
        critic_init_hstate = initialize_carry(config, model_config["NUM_ENVS"])
        critic_params = critic_network.init(critic_rng, critic_init_hstate, critic_init_x)

        if model_config["ANNEAL_LR"]:
            optimizer = lambda: optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            optimizer = lambda: optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(model_config["LR"], eps=1e-5),
            )

        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_params,
            tx=optimizer(),
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_params,
            tx=optimizer(),
        )
        train_states = (actor_train_state, critic_train_state)
        if initial_train_state is not None:
            train_states = initial_train_state

        rng, reset_rng = jax.random.split(rng)
        env_reset_rng = jax.random.split(reset_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(env_reset_rng)

        initial_actor_hstate = initialize_carry(config, model_config["NUM_ACTORS"])
        initial_critic_hstate = initialize_carry(config, model_config["NUM_ACTORS"])
        initial_population_hstate = None
        init_population_annealing_mask = None
        fcp_population_size = None
        population_network = None
        is_policy_population = False
        population_params = None

        if population is not None:
            is_policy_population = isinstance(population, AbstractPolicy)
            if is_policy_population:
                rng, pop_init_rng = jax.random.split(rng)
                initial_population_hstate = population.init_hstate(
                    model_config["NUM_ACTORS"], key=pop_init_rng
                )
            else:
                assert population_config is not None
                population_network = get_actor_critic(population_config)
                initial_population_hstate = initialize_carry(
                    population_config, model_config["NUM_ACTORS"]
                )
                if isinstance(population, dict) and "actor" in population:
                    population_params = population["actor"]
                else:
                    population_params = population
                fcp_population_size = jax.tree_util.tree_flatten(population_params)[0][0].shape[0]

            if use_population_annealing:

                def _sample_population_annealing_mask(step, rng):
                    return jax.random.uniform(rng, (model_config["NUM_ENVS"],)) < population_annealing_schedule(step)

                def _make_train_mask(annealing_mask):
                    full_anneal_mask = jnp.tile(annealing_mask, env.num_agents)
                    return jnp.where(full_anneal_mask, train_mask_flat, True)

                rng, mask_rng = jax.random.split(rng)
                init_population_annealing_mask = _sample_population_annealing_mask(0, mask_rng)

        def _update_step(runner_state, unused):
            (
                train_states,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_actor_hstate,
                initial_critic_hstate,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                rng,
            ) = runner_state

            def _env_step(env_step_state, unused):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    actor_hstate,
                    critic_hstate,
                    population_hstate,
                    population_annealing_mask,
                    fcp_pop_agent_idxs,
                    rng,
                ) = env_step_state
                actor_train_state, critic_train_state = train_states

                rng, action_rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
                actor_in = (
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                )
                actor_hstate, pi, _ = actor_train_state.apply_fn(
                    actor_train_state.params,
                    actor_hstate,
                    actor_in,
                )
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)
                action_pick_mask = jnp.ones((model_config["NUM_ACTORS"],), dtype=jnp.bool_)

                if population is not None:
                    obs_population = obs_batch
                    if isinstance(population, AbstractPolicy):
                        obs_featurized = jax.vmap(
                            env.get_obs_for_type, in_axes=(0, None)
                        )(env_state.env_state, ObservationType.FEATURIZED)
                        obs_population = batchify(
                            obs_featurized, env.agents, model_config["NUM_ACTORS"]
                        )

                    if is_policy_population:
                        rng, pop_action_rng = jax.random.split(rng)
                        pop_actions, population_hstate = population.compute_action(
                            obs_population, last_done, population_hstate, pop_action_rng
                        )
                    else:

                        def _compute_population_actions(policy_idx, obs_pop, obs_done, pop_hstate, key):
                            current_params = jax.tree_util.tree_map(
                                lambda x: x[policy_idx], population_params
                            )
                            current_ac_in = (
                                obs_pop[jnp.newaxis, jnp.newaxis, :],
                                jnp.array([obs_done])[jnp.newaxis, :],
                            )
                            new_pop_hstate, pop_pi, _ = population_network.apply(
                                current_params,
                                jax.tree_util.tree_map(lambda x: x[jnp.newaxis, :], pop_hstate),
                                current_ac_in,
                            )
                            pop_action = pop_pi.sample(seed=key)
                            return pop_action.squeeze(), jax.tree_util.tree_map(
                                lambda x: x.squeeze(axis=0), new_pop_hstate
                            )

                        pop_keys = jax.random.split(action_rng, model_config["NUM_ACTORS"])
                        pop_actions, population_hstate = jax.vmap(_compute_population_actions)(
                            fcp_pop_agent_idxs,
                            obs_population,
                            last_done,
                            population_hstate,
                            pop_keys,
                        )

                    action_pick_mask = train_mask_flat
                    if use_population_annealing:
                        action_pick_mask = _make_train_mask(population_annealing_mask)
                    action = jnp.where(action_pick_mask, action, pop_actions)

                world_state = last_obs["world_state"].swapaxes(0, 1).reshape(
                    (model_config["NUM_ACTORS"], -1)
                )
                critic_in = (
                    world_state[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                )
                critic_hstate, value = critic_train_state.apply_fn(
                    critic_train_state.params,
                    critic_hstate,
                    critic_in,
                )

                env_act = unbatchify(action, env.agents, model_config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, step_rng = jax.random.split(rng)
                rng_step = jax.random.split(step_rng, model_config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                original_reward = jnp.array([reward[a] for a in env.agents])
                current_timestep = (
                    update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                )
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
                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((model_config["NUM_ACTORS"])),
                    info,
                )

                done_batch = batchify(done, env.agents, model_config["NUM_ACTORS"]).squeeze()

                if use_population_annealing and population is not None:
                    env_steps = update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                    rng, anneal_rng = jax.random.split(rng)
                    new_population_annealing_mask = jnp.where(
                        done["__all__"],
                        _sample_population_annealing_mask(env_steps, anneal_rng),
                        population_annealing_mask,
                    )
                else:
                    new_population_annealing_mask = population_annealing_mask

                if population is not None and not is_policy_population:
                    rng, pop_idx_rng = jax.random.split(rng)
                    new_fcp_pop_agent_idxs = jnp.where(
                        jnp.tile(done["__all__"], env.num_agents),
                        jax.random.randint(
                            pop_idx_rng,
                            (model_config["NUM_ACTORS"],),
                            0,
                            fcp_population_size,
                        ),
                        fcp_pop_agent_idxs,
                    )
                else:
                    new_fcp_pop_agent_idxs = fcp_pop_agent_idxs

                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    done=last_done,
                    action=action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(reward, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    world_state=world_state,
                    info=info,
                    train_mask=action_pick_mask,
                )

                env_step_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    actor_hstate,
                    critic_hstate,
                    population_hstate,
                    new_population_annealing_mask,
                    new_fcp_pop_agent_idxs,
                    rng,
                )
                return env_step_state, transition

            env_step_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_actor_hstate,
                initial_critic_hstate,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                rng,
            )
            env_step_state, traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, model_config["NUM_STEPS"]
            )

            (
                train_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_actor_hstate,
                next_critic_hstate,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                rng,
            ) = env_step_state
            last_world_state = last_obs["world_state"].swapaxes(0, 1).reshape(
                (model_config["NUM_ACTORS"], -1)
            )
            critic_in = (
                last_world_state[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
            )
            _, last_val = critic_train_state.apply_fn(
                critic_train_state.params,
                next_critic_hstate,
                critic_in,
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
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    actor_init_hstate, critic_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        train_mask = jax.lax.stop_gradient(traj_batch.train_mask)
                        _, pi, _ = actor_train_state.apply_fn(
                            actor_params,
                            None if init_hstate is None else init_hstate.squeeze(axis=0),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean(where=train_mask)) / (gae.std(where=train_mask) + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(
                            ratio,
                            1.0 - model_config["CLIP_EPS"],
                            1.0 + model_config["CLIP_EPS"],
                        ) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean(where=train_mask)
                        entropy = pi.entropy().mean(where=train_mask)
                        approx_kl = ((ratio - 1) - logratio).mean(where=train_mask)
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > model_config["CLIP_EPS"], where=train_mask)

                        total_loss = loss_actor - model_config["ENT_COEF"] * entropy
                        return total_loss, (loss_actor, entropy, ratio.mean(where=train_mask), approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        train_mask = jax.lax.stop_gradient(traj_batch.train_mask)
                        _, value = critic_train_state.apply_fn(
                            critic_params,
                            None if init_hstate is None else init_hstate.squeeze(axis=0),
                            (traj_batch.world_state, traj_batch.done),
                        )
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=train_mask)
                        total_loss = model_config["VF_COEF"] * value_loss
                        return total_loss, value_loss

                    def _perform_update():
                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        actor_loss, actor_grads = actor_grad_fn(
                            actor_train_state.params,
                            actor_init_hstate,
                            traj_batch,
                            advantages,
                        )
                        critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                        critic_loss, critic_grads = critic_grad_fn(
                            critic_train_state.params,
                            critic_init_hstate,
                            traj_batch,
                            targets,
                        )

                        new_actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                        new_critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                        loss_info = {
                            "total_loss": actor_loss[0] + critic_loss[0],
                            "actor_loss": actor_loss[1][0],
                            "value_loss": critic_loss[1],
                            "entropy": actor_loss[1][1],
                            "ratio": actor_loss[1][2],
                            "approx_kl": actor_loss[1][3],
                            "clip_frac": actor_loss[1][4],
                        }
                        return (new_actor_train_state, new_critic_train_state), loss_info

                    def _no_op():
                        loss_info = {
                            "total_loss": 0.0,
                            "actor_loss": 0.0,
                            "value_loss": 0.0,
                            "entropy": 0.0,
                            "ratio": 0.0,
                            "approx_kl": 0.0,
                            "clip_frac": 0.0,
                        }
                        return (actor_train_state, critic_train_state), loss_info

                    return jax.lax.cond(
                        traj_batch.train_mask.any(),
                        _perform_update,
                        _no_op,
                    )

                (
                    train_states,
                    actor_init_hstate,
                    critic_init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, perm_rng = jax.random.split(rng)

                actor_batch_hstate = actor_init_hstate
                if actor_batch_hstate is not None:
                    actor_batch_hstate = actor_batch_hstate[jnp.newaxis, :]
                critic_batch_hstate = critic_init_hstate
                if critic_batch_hstate is not None:
                    critic_batch_hstate = critic_batch_hstate[jnp.newaxis, :]

                batch = (
                    actor_batch_hstate,
                    critic_batch_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(perm_rng, model_config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1),
                    batch,
                )
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

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (
                    train_states,
                    actor_init_hstate,
                    critic_init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            rng, update_rng = jax.random.split(rng)
            update_state = (
                train_states,
                initial_actor_hstate,
                initial_critic_hstate,
                traj_batch,
                advantages,
                targets,
                update_rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, model_config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]

            metric = traj_batch.info
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            for key, value in loss_info.items():
                metric[key] = value
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

            update_step += 1
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
            )

            def callback(metric, original_seed):
                metric = {f"rng{int(original_seed)}/{k}": v for k, v in metric.items()}
                wandb.log(metric)

            jax.debug.callback(callback, metric, original_seed)

            if num_checkpoints > 0:
                checkpoint_idx_selector = checkpoint_steps == update_step
                checkpoint_states = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _ts, _i: c,
                    checkpoint_states,
                    train_states,
                    jnp.argmax(checkpoint_idx_selector),
                )

            runner_state = (
                train_states,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_actor_hstate,
                next_critic_hstate,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                rng,
            )
            return runner_state, metric

        initial_update_step = 0
        if update_step_offset is not None:
            initial_update_step = update_step_offset

        initial_checkpoints = jax.tree_util.tree_map(
            lambda p: jnp.zeros((num_checkpoints,) + p.shape, p.dtype),
            _current_params(train_states),
        )
        if num_checkpoints > 0:
            initial_checkpoints = jax.lax.cond(
                (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                _update_checkpoint,
                lambda c, _ts, _i: c,
                initial_checkpoints,
                train_states,
                0,
            )

        init_fcp_pop_idxs = None
        if population is not None and not is_policy_population:
            rng, pop_idx_rng = jax.random.split(rng)
            init_fcp_pop_idxs = jax.random.randint(
                pop_idx_rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
            )

        rng, runner_rng = jax.random.split(rng)
        runner_state = (
            train_states,
            initial_checkpoints,
            env_state,
            obsv,
            jnp.zeros((model_config["NUM_ACTORS"]), dtype=bool),
            initial_update_step,
            initial_actor_hstate,
            initial_critic_hstate,
            initial_population_hstate,
            init_population_annealing_mask,
            init_fcp_pop_idxs,
            runner_rng,
        )
        num_update_steps = model_config["NUM_UPDATES"]
        if update_step_num_overwrite is not None:
            num_update_steps = update_step_num_overwrite
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_update_steps
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
