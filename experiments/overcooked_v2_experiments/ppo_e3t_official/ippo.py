""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Optional, Sequence, NamedTuple, Any, Dict, Union
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import functools
import math
import pickle
from models.rnn import ScannedRNN
import matplotlib.pyplot as plt
from jaxmarl.environments.overcooked_v2.overcooked import ObservationType
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from overcooked_v2_experiments.ppo_e3t_official.models.abstract import ActorCriticBase
from .models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.eval.policy import AbstractPolicy
from flax import core, struct, traverse_util


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    partner_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    history_obs: jnp.ndarray
    history_actions: jnp.ndarray
    info: jnp.ndarray
    train_mask: jnp.ndarray

@struct.dataclass
class E3TTrainState:
    step: jnp.ndarray
    apply_fn: Any = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    context_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    context_opt_state: optax.OptState
    ppo_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    ppo_opt_state: optax.OptState

    @classmethod
    def create(cls, *, apply_fn, params, context_tx, ppo_tx):
        return cls(
            step=jnp.asarray(0, dtype=jnp.int32),
            apply_fn=apply_fn,
            params=params,
            context_tx=context_tx,
            context_opt_state=context_tx.init(params),
            ppo_tx=ppo_tx,
            ppo_opt_state=ppo_tx.init(params),
        )

    def apply_context_gradients(self, *, grads):
        updates, new_opt_state = self.context_tx.update(
            grads, self.context_opt_state, self.params
        )
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            context_opt_state=new_opt_state,
        )

    def apply_ppo_gradients(self, *, grads):
        updates, new_opt_state = self.ppo_tx.update(
            grads, self.ppo_opt_state, self.params
        )
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ppo_opt_state=new_opt_state,
        )



def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}



def _unpack_apply_output(network_output):
    if len(network_output) == 3:
        next_hstate, pi, value = network_output
        partner_pi = None
        human_pi = None
    elif len(network_output) == 4:
        next_hstate, pi, value, partner_pi = network_output
        human_pi = None
    elif len(network_output) == 5:
        next_hstate, pi, value, partner_pi, human_pi = network_output
    else:
        raise ValueError(f"Unexpected actor-critic output length: {len(network_output)}")
    return next_hstate, pi, value, partner_pi, human_pi


def _network_input(
    obs,
    done,
    history_obs,
    history_actions,
    use_history_context,
    context_human=None,
):
    if use_history_context:
        if context_human is None:
            return obs, done, history_obs, history_actions
        return obs, done, history_obs, history_actions, context_human
    return obs, done


def _bootstrap_history(obs_batch, context_length, stay_action):
    history_obs = jnp.repeat(obs_batch[:, jnp.newaxis, ...], context_length, axis=1)
    history_actions = jnp.full(
        (obs_batch.shape[0], context_length), stay_action, dtype=jnp.int32
    )
    return history_obs, history_actions


def _partner_action_from_flat(action, num_envs, num_agents):
    action_by_agent = action.reshape((num_agents, num_envs))
    if num_agents == 2:
        partner_action = action_by_agent[::-1]
    else:
        partner_action = jnp.roll(action_by_agent, shift=-1, axis=0)
    return partner_action.reshape((num_agents * num_envs,))


def _partner_obs_from_flat(obs_batch, num_envs, num_agents):
    obs_by_agent = obs_batch.reshape((num_agents, num_envs) + obs_batch.shape[1:])
    if num_agents == 2:
        partner_obs = obs_by_agent[::-1]
    else:
        partner_obs = jnp.roll(obs_by_agent, shift=-1, axis=0)
    return partner_obs.reshape((num_agents * num_envs,) + obs_batch.shape[1:])


def _update_history_buffers(
    history_obs,
    history_actions,
    obs_batch,
    partner_action,
    done_all_batch,
    context_length,
    stay_action,
):
    del done_all_batch, context_length, stay_action
    # Official E3T resets obs/action history only at the start of Runner.run(),
    # not on episode termination inside the rollout.
    appended_obs = jnp.concatenate(
        [history_obs[:, 1:], obs_batch[:, jnp.newaxis, ...]], axis=1
    )
    appended_actions = jnp.concatenate(
        [history_actions[:, 1:], partner_action[:, jnp.newaxis]], axis=1
    )
    return appended_obs, appended_actions


def _mix_partner_probs(pi, partner_mask, eps):
    probs = pi.probs
    uniform = jnp.ones_like(probs) / probs.shape[-1]
    mixed_probs = (1.0 - eps) * probs + eps * uniform
    partner_mask = partner_mask.reshape(probs.shape[:-1] + (1,))
    probs = jnp.where(partner_mask, mixed_probs, probs)
    return distrax.Categorical(probs=probs)


def _merge_train_and_partner_behavior(train_pi, partner_pi, partner_mask, rand_eps):
    partner_behavior_pi = _mix_partner_probs(partner_pi, partner_mask, rand_eps)
    partner_mask = partner_mask.reshape(train_pi.probs.shape[:-1] + (1,))
    probs = jnp.where(partner_mask, partner_behavior_pi.probs, train_pi.probs)
    return distrax.Categorical(probs=probs)


def _path_tuple(path):
    return tuple(getattr(part, "key", part) for part in path)


def _path_has_marker(path, markers):
    path_str = "/".join(str(part) for part in _path_tuple(path))
    return any(marker in path_str for marker in markers)


def _mask_tree_to_markers(tree, markers):
    return jax.tree_util.tree_map_with_path(
        lambda path, value: value if _path_has_marker(path, markers) else jnp.zeros_like(value),
        tree,
    )


def _restore_tree_to_markers(updated_tree, old_tree, markers):
    return jax.tree_util.tree_map_with_path(
        lambda path, updated, old: updated if _path_has_marker(path, markers) else old,
        updated_tree,
        old_tree,
    )


CONTEXT_PARAM_MARKERS = (
    "context_conv_",
    "context_word_embd",
    "context_dense_",
    "context_temporal_dense_",
    "human_prob_pre_",
    "decoder_conv_",
    "decoder_dense_",
    "decoder_pi",
)

PPO_PARAM_MARKERS = (
    "policy_encoder",
    "policy_ln",
    "actor_0",
    "actor_out",
    "critic_0",
    "critic_out",
)

def _soft_update_human_policy(params, copy_coef):
    if copy_coef <= 0.0:
        return params

    module_pairs = (
        ("human_actor_out", "actor_out"),
    )
    source_lookup = traverse_util.flatten_dict(core.unfreeze(params))

    def _update(path, value):
        path_tuple = _path_tuple(path)
        for human_name, source_name in module_pairs:
            if human_name not in path_tuple:
                continue
            source_path = tuple(
                source_name if part == human_name else part for part in path_tuple
            )
            if source_path in source_lookup:
                return (1.0 - copy_coef) * value + copy_coef * source_lookup[source_path]
        return value

    return jax.tree_util.tree_map_with_path(_update, params)


def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
):
    env_config = config["env"]
    model_config = config["model"]

    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])

    model_config["NUM_ACTORS"] = env.num_agents * model_config["NUM_ENVS"]
    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"]
        // model_config["NUM_STEPS"]
        // model_config["NUM_ENVS"]
    )
    model_config["MINIBATCH_SIZE"] = (
        model_config["NUM_ENVS"]
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
        # make sure the last checkpoint is the last update step
        checkpoint_steps = checkpoint_steps.at[-1].set(model_config["NUM_UPDATES"])

    print("Checkpoint steps: ", checkpoint_steps)

    def _update_checkpoint(checkpoint_states, params, i):
        jax.debug.print("Saving checkpointing {i}", i=i)
        return jax.tree_util.tree_map(
            lambda x, y: x.at[i].set(y),
            checkpoint_states,
            params,
        )

    env = OvercookedV2LogWrapper(env, replace_info=False)

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

        print("Update steps: ", update_steps)
        print("Warmup epochs: ", warmup_steps)
        print("Cosine epochs: ", cosine_epochs)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

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
    train_mask_flat = batchify(
        train_mask_dict, env.agents, model_config["NUM_ACTORS"]
    ).squeeze()

    print("train_mask_flat", train_mask_flat.shape)
    print("train_mask_flat sum", train_mask_flat.sum())

    use_history_context = model_config.get("USE_HISTORY_CONTEXT", True)
    context_length = int(model_config.get("CONTEXT_LENGTH", 5))
    stay_action = int(model_config.get("STAY_ACTION", 4))
    use_partner_mix = model_config.get("USE_PARTNER_MIX", False)
    partner_mix_eps = float(model_config.get("PARTNER_MIX_EPS", 0.0))
    use_official_e3t_partner = model_config.get("USE_OFFICIAL_E3T_PARTNER", True)
    partner_source = str(
        model_config.get(
            "E3T_PARTNER_SOURCE",
            "human_branch" if use_official_e3t_partner else "main_policy",
        )
    ).lower()
    if partner_source not in ("human_branch", "main_policy"):
        raise ValueError(f"Unknown E3T_PARTNER_SOURCE={partner_source}")
    enable_ce = bool(model_config.get("E3T_ENABLE_CE", True))
    partner_rand = float(model_config.get("RAND", 0.7))
    copy_coef = float(model_config.get("COPY", 0.1))
    action_dim = int(model_config.get("ACTION_DIM", 6))
    partner_mask_flat = jnp.logical_not(train_mask_flat)
    train_actor_indices = jnp.nonzero(
        train_mask_flat, size=model_config["NUM_ENVS"], fill_value=0
    )[0]
    if partner_source == "human_branch":
        ppo_actor_indices = train_actor_indices
        ppo_train_actor_count = model_config["NUM_ENVS"]
    else:
        ppo_actor_indices = jnp.arange(model_config["NUM_ACTORS"])
        ppo_train_actor_count = model_config["NUM_ACTORS"]
    train_batch_size = model_config["NUM_STEPS"] * ppo_train_actor_count

    use_population_annealing = False
    if "POPULATION_ANNEAL_HORIZON" in config:
        print("Using population annealing")
        use_population_annealing = True
        transition_begin = 0
        if "POPULATION_ANNEAL_BEGIN" in config:
            transition_begin = config["POPULATION_ANNEAL_BEGIN"]

        anneal_horizon = config["POPULATION_ANNEAL_HORIZON"]
        if anneal_horizon == 0:
            population_annealing_schedule = optax.constant_schedule(1.0)
        else:
            population_annealing_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=1.0,
                transition_steps=config["POPULATION_ANNEAL_HORIZON"] - transition_begin,
                transition_begin=transition_begin,
            )

    def train(
        rng,
        run_index=None,
        population: Optional[Union[AbstractPolicy, core.FrozenDict[str, Any]]] = None,
        initial_train_state=None,
    ):
        original_seed = rng[0]

        jax.debug.print("original_seed {s}", s=rng)

        # INIT NETWORK
        network = get_actor_critic(config)

        rng, _rng = jax.random.split(rng)

        init_obs = jnp.zeros(
            (1, model_config["NUM_ENVS"], *env.observation_space().shape),
        )
        init_done = jnp.zeros((1, model_config["NUM_ENVS"]))
        if use_history_context:
            init_x = (
                init_obs,
                init_done,
                jnp.zeros(
                    (
                        1,
                        model_config["NUM_ENVS"],
                        context_length,
                        *env.observation_space().shape,
                    )
                ),
                jnp.full(
                    (1, model_config["NUM_ENVS"], context_length),
                    stay_action,
                    dtype=jnp.int32,
                ),
            )
        else:
            init_x = (init_obs, init_done)
        init_hstate = initialize_carry(config, model_config["NUM_ENVS"])

        if init_hstate is not None:
            print("init_hstate", init_hstate.shape)
        # jax.debug.print("check1 {x}", x=init_hstate.flatten()[0])

        print("init_x", init_x[0].shape, init_x[1].shape)

        network_params = network.init(_rng, init_hstate, init_x)
        if model_config["ANNEAL_LR"]:
            lr_or_schedule = create_learning_rate_fn()
        else:
            lr_or_schedule = model_config["LR"]

        context_tx = optax.adam(lr_or_schedule, eps=1e-5)
        ppo_tx = optax.chain(
            optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
            optax.adam(lr_or_schedule, eps=1e-5),
        )

        train_state = E3TTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            context_tx=context_tx,
            ppo_tx=ppo_tx,
        )

        if initial_train_state is not None:
            train_state = initial_train_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_hstate = initialize_carry(config, model_config["NUM_ACTORS"])
        init_obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape(
            -1, *env.observation_space().shape
        )
        init_history_obs, init_history_actions = _bootstrap_history(
            init_obs_batch, context_length, stay_action
        )
        # jax.debug.print("check2 {x}", x=init_hstate.flatten()[0])

        init_population_hstate = None
        init_population_annealing_mask = None
        if population is not None:
            is_policy_population = False
            if isinstance(population, AbstractPolicy):
                is_policy_population = True
                rng, _rng = jax.random.split(rng)
                init_population_hstate = population.init_hstate(
                    model_config["NUM_ACTORS"], key=_rng
                )
            else:
                assert (
                    population_config is not None
                ), "population_config cannot be None if population is not a policy"
                population_network = get_actor_critic(population_config)
                init_population_hstate = initialize_carry(
                    population_config, model_config["NUM_ACTORS"]
                )

                fcp_population_size = jax.tree_util.tree_flatten(population)[0][
                    0
                ].shape[0]
                print("FCP population size", fcp_population_size)

                # print(f"normal hstate {init_hstate.shape}")
                # print(f"population hstate {init_population_hstate.shape}")

            if use_population_annealing:

                def _sample_population_annealing_mask(step, rng):
                    return jax.random.uniform(
                        rng, (model_config["NUM_ENVS"],)
                    ) < population_annealing_schedule(step)

                def _make_train_mask(annealing_mask):
                    full_anneal_mask = jnp.tile(annealing_mask, env.num_agents)
                    return jnp.where(full_anneal_mask, train_mask_flat, True)

                rng, _rng = jax.random.split(rng)
                init_population_annealing_mask = _sample_population_annealing_mask(
                    0, _rng
                )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            (
                train_state,
                checkpoint_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                initial_history_obs,
                initial_history_actions,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                rng,
            ) = runner_state

            # jax.debug.print("check3 {x}", x=initial_hstate.flatten()[0])

            # COLLECT TRAJECTORIES
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
                    population_hstate,
                    population_annealing_mask,
                    fcp_pop_agent_idxs,
                    rng,
                ) = env_step_state

                # jax.debug.print("check4 {x}", x=hstate.flatten()[0])

                # SELECT ACTION
                rng, _rng_action, _rng_context = jax.random.split(rng, 3)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )

                context_human = None
                if partner_source == "human_branch":
                    context_human = jax.random.normal(
                        _rng_context,
                        (1, model_config["NUM_ACTORS"], action_dim),
                    )

                ac_in = _network_input(
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    history_obs[np.newaxis, :],
                    history_actions[np.newaxis, :],
                    use_history_context,
                    context_human,
                )

                hstate, pi, value, _partner_pi, human_pi = _unpack_apply_output(
                    network.apply(train_state.params, hstate, ac_in)
                )

                # jax.debug.print("check5 {x}", x=hstate.flatten()[0])

                behavior_pi = pi
                if partner_source == "human_branch":
                    behavior_pi = _merge_train_and_partner_behavior(
                        pi, human_pi, partner_mask_flat, partner_rand
                    )
                elif partner_source == "main_policy" and use_partner_mix:
                    behavior_pi = _mix_partner_probs(pi, partner_mask_flat, partner_mix_eps)
                elif use_partner_mix:
                    behavior_pi = _mix_partner_probs(pi, partner_mask_flat, partner_mix_eps)

                action = behavior_pi.sample(seed=_rng_action)
                log_prob = behavior_pi.log_prob(action)

                action_pick_mask = train_mask_flat
                if population is not None:
                    print("Using population")

                    obs_population = obs_batch
                    if isinstance(population, AbstractPolicy):
                        obs_featurized = jax.vmap(
                            env.get_obs_for_type, in_axes=(0, None)
                        )(env_state.env_state, ObservationType.FEATURIZED)
                        obs_population = batchify(
                            obs_featurized, env.agents, model_config["NUM_ACTORS"]
                        )

                    if is_policy_population:
                        rng, _rng = jax.random.split(rng)
                        pop_actions, population_hstate = population.compute_action(
                            obs_population, last_done, population_hstate, _rng
                        )
                    else:

                        def _compute_population_actions(
                            policy_idx, obs_pop, obs_ld, fcp_h_state
                        ):
                            current_p = jax.tree.map(
                                lambda x: x[policy_idx], population
                            )
                            current_ac_in = (
                                obs_pop[np.newaxis, np.newaxis, :],
                                jnp.array([obs_ld])[np.newaxis, :],
                            )
                            new_fcp_h_state, fcp_pi, _ = population_network.apply(
                                current_p,
                                jax.tree.map(lambda x: x[np.newaxis, :], fcp_h_state),
                                current_ac_in,
                            )
                            fcp_action = fcp_pi.sample(seed=_rng)
                            return fcp_action.squeeze(), jax.tree.map(
                                lambda x: x.squeeze(axis=0), new_fcp_h_state
                            )

                        pop_actions, population_hstate = jax.vmap(
                            _compute_population_actions
                        )(
                            fcp_pop_agent_idxs,
                            obs_population,
                            last_done,
                            population_hstate,
                        )

                    action_pick_mask = train_mask_flat
                    if use_population_annealing:
                        action_pick_mask = _make_train_mask(population_annealing_mask)

                    # use action_pick_mask to select the action from the population or the network
                    action = jnp.where(action_pick_mask, action, pop_actions)
                    log_prob = behavior_pi.log_prob(action)

                partner_action = _partner_action_from_flat(
                    action.squeeze(), model_config["NUM_ENVS"], env.num_agents
                )

                env_act = unbatchify(
                    action, env.agents, model_config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, model_config["NUM_ENVS"])

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

                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((model_config["NUM_ACTORS"])), info
                )

                done_batch = batchify(
                    done, env.agents, model_config["NUM_ACTORS"]
                ).squeeze()

                if use_population_annealing:
                    env_steps = (
                        update_step
                        * model_config["NUM_STEPS"]
                        * model_config["NUM_ENVS"]
                    )
                    rng, _rng = jax.random.split(rng)
                    new_population_annealing_mask = jnp.where(
                        done["__all__"],
                        _sample_population_annealing_mask(env_steps, _rng),
                        population_annealing_mask,
                    )
                else:
                    new_population_annealing_mask = population_annealing_mask

                if population is not None and not is_policy_population:
                    new_fcp_pop_agent_idxs = jnp.where(
                        jnp.tile(done["__all__"], env.num_agents),
                        jax.random.randint(
                            _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
                        ),
                        fcp_pop_agent_idxs,
                    )
                else:
                    new_fcp_pop_agent_idxs = fcp_pop_agent_idxs

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    partner_action,
                    value.squeeze(),
                    batchify(reward, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    history_obs,
                    history_actions,
                    info,
                    action_pick_mask,
                )

                # jax.debug.print("check6 {x}", x=hstate.flatten()[0])
                new_history_obs, new_history_actions = _update_history_buffers(
                    history_obs,
                    history_actions,
                    obs_batch,
                    partner_action,
                    jnp.tile(done["__all__"], env.num_agents),
                    context_length,
                    stay_action,
                )

                env_step_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    new_history_obs,
                    new_history_actions,
                    population_hstate,
                    new_population_annealing_mask,
                    new_fcp_pop_agent_idxs,
                    rng,
                )
                return env_step_state, transition

            rollout_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            rollout_history_obs, rollout_history_actions = _bootstrap_history(
                rollout_obs_batch, context_length, stay_action
            )

            env_step_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                initial_hstate,
                rollout_history_obs,
                rollout_history_actions,
                initial_population_hstate,
                last_population_annealing_mask,
                initial_fcp_pop_agent_idxs,
                rng,
            )
            env_step_state, traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, model_config["NUM_STEPS"]
            )
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                next_initial_hstate,
                next_history_obs,
                next_history_actions,
                next_population_hstate,
                last_population_annealing_mask,
                next_fcp_pop_agent_idxs,
                rng,
            ) = env_step_state

            # jax.debug.print("check7 {x}", x=next_initial_hstate)

            # print("Hilfeeeee", traj_batch.done.shape, traj_batch.action.shape)

            # CALCULATE ADVANTAGE
            # PPO+E3T migration preserves PPO bootstrap instead of official E3T zero bootstrap.
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            ac_in = _network_input(
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                next_history_obs[np.newaxis, :],
                next_history_actions[np.newaxis, :],
                use_history_context,
            )
            _, _, last_val, _, _ = _unpack_apply_output(
                network.apply(train_state.params, next_initial_hstate, ac_in)
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + model_config["GAMMA"] * next_value * (1 - done) - value
                    )
                    gae = (
                        delta
                        + model_config["GAMMA"]
                        * model_config["GAE_LAMBDA"]
                        * (1 - done)
                        * gae
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

            # UPDATE NETWORK
            def _select_train_side(tree):
                return jax.tree_util.tree_map(
                    lambda x: jnp.take(x, train_actor_indices, axis=1), tree
                )

            def _flatten_time_actor(tree):
                return jax.tree_util.tree_map(
                    lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
                    tree,
                )

            def _select_ppo_actors(tree):
                return jax.tree_util.tree_map(
                    lambda x: jnp.take(x, ppo_actor_indices, axis=1), tree
                )

            train_traj_batch = _flatten_time_actor(_select_ppo_actors(traj_batch))
            train_advantages = _flatten_time_actor(
                jnp.take(advantages, ppo_actor_indices, axis=1)
            )
            train_targets = _flatten_time_actor(
                jnp.take(targets, ppo_actor_indices, axis=1)
            )

            def _context_loss_fn(params, traj_batch):
                _, _, _, partner_pi, _ = _unpack_apply_output(
                    network.apply(
                        params,
                        initial_hstate,
                        _network_input(
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.history_obs,
                            traj_batch.history_actions,
                            use_history_context,
                        ),
                    )
                )
                classification_loss = optax.softmax_cross_entropy_with_integer_labels(
                    partner_pi.logits, traj_batch.partner_action
                ).mean()
                classification_pred = jnp.argmax(partner_pi.logits, axis=-1)
                classification_acc = (
                    classification_pred == traj_batch.partner_action
                ).mean()
                return classification_loss, (classification_loss, classification_acc)

            def _ppo_loss_fn(params, traj_batch, gae, targets):
                _, pi, value, _partner_pi, _ = _unpack_apply_output(
                    network.apply(
                        params,
                        initial_hstate,
                        _network_input(
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.history_obs,
                            traj_batch.history_actions,
                            use_history_context,
                        ),
                    )
                )

                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(
                    value_losses, value_losses_clipped
                ).mean()

                ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                entropy = pi.entropy().mean()
                ratio = ratio.mean()

                total_loss = (
                    loss_actor
                    + model_config["VF_COEF"] * value_loss
                    - model_config["ENT_COEF"] * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy, ratio)

            def _make_minibatches(batch_tree, rng_key):
                permutation = jax.random.permutation(rng_key, train_batch_size)
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch_tree
                )
                return jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x,
                        (model_config["NUM_MINIBATCHES"], -1) + x.shape[1:],
                    ),
                    shuffled,
                )

            def _update_context_minibatch(train_state, minibatch):
                (loss, aux), grads = jax.value_and_grad(
                    _context_loss_fn, has_aux=True
                )(train_state.params, minibatch)
                grads = _mask_tree_to_markers(grads, CONTEXT_PARAM_MARKERS)
                old_params = train_state.params
                new_train_state = train_state.apply_context_gradients(grads=grads)
                new_train_state = new_train_state.replace(
                    params=_restore_tree_to_markers(
                        new_train_state.params,
                        old_params,
                        CONTEXT_PARAM_MARKERS,
                    )
                )
                return new_train_state, aux

            def _update_ppo_minibatch(train_state, minibatch):
                traj_mb, adv_mb, target_mb = minibatch
                (loss, aux), grads = jax.value_and_grad(
                    _ppo_loss_fn, has_aux=True
                )(train_state.params, traj_mb, adv_mb, target_mb)
                grads = _mask_tree_to_markers(grads, PPO_PARAM_MARKERS)
                old_params = train_state.params
                new_train_state = train_state.apply_ppo_gradients(grads=grads)
                new_train_state = new_train_state.replace(
                    params=_restore_tree_to_markers(
                        new_train_state.params,
                        old_params,
                        PPO_PARAM_MARKERS,
                    )
                )
                return new_train_state, (loss, aux)

            def _update_context_epoch(update_state, unused):
                train_state, rng = update_state
                rng, _rng = jax.random.split(rng)
                minibatches = _make_minibatches(train_traj_batch, _rng)
                train_state, context_info = jax.lax.scan(
                    _update_context_minibatch, train_state, minibatches
                )
                return (train_state, rng), context_info

            def _update_ppo_epoch(update_state, unused):
                train_state, rng = update_state
                rng, _rng = jax.random.split(rng)
                minibatches = _make_minibatches(
                    (train_traj_batch, train_advantages, train_targets), _rng
                )
                train_state, ppo_info = jax.lax.scan(
                    _update_ppo_minibatch, train_state, minibatches
                )
                return (train_state, rng), ppo_info

            rng, _rng_context = jax.random.split(rng)
            if enable_ce:
                (train_state, rng), context_info = jax.lax.scan(
                    _update_context_epoch,
                    (train_state, _rng_context),
                    None,
                    int(model_config.get("CONTEXT_UPDATE_EPOCHS", 8)),
                )
            else:
                context_info = (
                    jnp.zeros((1,), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.float32),
                )

            rng, _rng_ppo = jax.random.split(rng)
            (train_state, rng), ppo_info = jax.lax.scan(
                _update_ppo_epoch,
                (train_state, _rng_ppo),
                None,
                model_config["UPDATE_EPOCHS"],
            )

            ppo_loss, ppo_aux = ppo_info
            value_loss, loss_actor, entropy, ratio = ppo_aux
            moa_loss, moa_acc = context_info
            total_loss = ppo_loss
            train_state = train_state.replace(
                params=_soft_update_human_policy(train_state.params, copy_coef)
            )
            metric = traj_batch.info

            metric["total_loss"] = total_loss
            metric["value_loss"] = value_loss
            metric["loss_actor"] = loss_actor
            metric["entropy"] = entropy
            metric["ratio"] = ratio
            metric["moa_loss"] = moa_loss
            metric["moa_acc"] = moa_acc

            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

            update_step += 1
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
            )

            def callback(metric, original_seed, run_index):
                if run_index is None:
                    prefix = f"rng{int(original_seed)}"
                else:
                    prefix = f"run_{int(run_index)}"
                metric.update(
                    {f"{prefix}/{k}": v for k, v in metric.items()}
                )
                wandb.log(metric)

            jax.debug.callback(callback, metric, original_seed, run_index)

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
                next_history_obs,
                next_history_actions,
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

        init_fcp_pop_idxs = None
        if population is not None and not is_policy_population:
            init_fcp_pop_idxs = jax.random.randint(
                _rng, (model_config["NUM_ACTORS"],), 0, fcp_population_size
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
            init_population_hstate,
            init_population_annealing_mask,
            init_fcp_pop_idxs,
            _rng,
        )
        num_update_steps = model_config["NUM_UPDATES"]
        if update_step_num_overwrite is not None:
            num_update_steps = update_step_num_overwrite
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_update_steps
        )

        # jax.debug.print("Runner state {x}", x=runner_state)
        # jax.debug.print("neg5 {x}", x=runner_state[-5])
        return {"runner_state": runner_state, "metrics": metric}

    return train
