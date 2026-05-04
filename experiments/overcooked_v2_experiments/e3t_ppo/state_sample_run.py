import copy
import itertools
import math
from datetime import datetime

import jax
import jax.numpy as jnp
import jaxmarl
import wandb
from omegaconf import OmegaConf
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

from overcooked_v2_experiments.e3t_ppo.models.model import (
    get_actor_critic,
    initialize_carry,
)
from overcooked_v2_experiments.e3t_ppo.policy import (
    E3TPPOParams,
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.e3t_ppo.train import make_train
from overcooked_v2_experiments.e3t_ppo.utils.store import store_checkpoint
from overcooked_v2_experiments.e3t_ppo.utils.utils import (
    combine_first_two_tree_dim,
    get_num_devices,
    get_run_base_dir,
)
from overcooked_v2_experiments.e3t_ppo.utils.visualize_ppo import visualize_ppo_policy
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.utils.utils import mini_batch_pmap, scanned_mini_batch_map


def _network_init_input(config, env, num_envs, context_length, stay_action):
    model_config = config["model"]
    obs_shape = env.observation_space().shape
    obs = jnp.zeros((1, num_envs, *obs_shape), dtype=jnp.float32)
    done = jnp.zeros((1, num_envs), dtype=jnp.bool_)
    if model_config.get("USE_HISTORY_CONTEXT", False):
        history_obs = jnp.zeros(
            (1, num_envs, context_length, *obs_shape),
            dtype=jnp.float32,
        )
        history_actions = jnp.full(
            (1, num_envs, context_length),
            stay_action,
            dtype=jnp.int32,
        )
        return obs, done, history_obs, history_actions
    return obs, done


def state_sample_run(config):
    print("Running E3T-PPO state sample run", flush=True)
    config = OmegaConf.to_container(config)

    if "FCP" in config or "BC" in config:
        raise NotImplementedError("E3T-PPO state augmentation supports self-play only.")

    num_seeds = config["NUM_SEEDS"]
    num_iterations = config["NUM_ITERATIONS"]
    assert num_iterations > 0

    model_name = config["model"]["TYPE"]
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    optional_prefix = config.get("OPTIONAL_PREFIX", "")
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    run_name = f"e3t_ppo_{model_name.lower()}_ov2_{layout_name}_{avs_str}_sa-{num_iterations}"
    if optional_prefix:
        run_name = f"{optional_prefix}_{run_name}"

    with wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=["E3T-PPO", model_name, "OvercookedV2", "state-aug"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    ) as run:
        config["RUN_BASE_DIR"] = get_run_base_dir(run.id, config)
        print("Run base dir: ", config["RUN_BASE_DIR"], flush=True)

        key = jax.random.PRNGKey(config["SEED"])
        model_config = config["model"]
        env_config = config["env"]
        env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
        context_length = int(model_config.get("CONTEXT_LENGTH", 5))
        stay_action = min(4, env.action_space(env.agents[0]).n - 1)
        num_runs = num_seeds

        def _run_iteration(
            key,
            state_buffer,
            prev_train_state=None,
            update_step_start=None,
            update_step_end=None,
        ):
            num_update_steps = update_step_end - update_step_start
            config_copy = copy.deepcopy(config)
            config_copy["env"]["ENV_KWARGS"].pop("obs_shape", None)
            config_copy["env"]["ENV_KWARGS"]["initial_state_buffer"] = state_buffer
            keys = jax.random.split(key, num_seeds)
            train_jit = jax.jit(
                make_train(
                    config_copy,
                    update_step_offset=update_step_start,
                    update_step_num_overwrite=num_update_steps,
                )
            )
            num_devices = get_num_devices()
            mapped_train = mini_batch_pmap(train_jit, num_devices)
            if prev_train_state is None:
                return mapped_train(keys)
            return mapped_train(keys, initial_train_state=prev_train_state)

        def _init_policy(key):
            network = get_actor_critic(config)
            init_x = _network_init_input(
                config,
                env,
                model_config["NUM_ENVS"],
                context_length,
                stay_action,
            )
            init_hstate = initialize_carry(config, model_config["NUM_ENVS"])
            return network.init(key, init_hstate, init_x)

        def _collect_states(previous_policies, key):
            start_time = datetime.now()
            env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
            env_kwargs.pop("obs_shape", None)
            state_env = OvercookedV2(**env_kwargs)

            def _process_combination(policies, rollout_key):
                policies = policy_checkoints_to_policy_pairing(policies, config)

                def _rollout_seed(seed_key):
                    return get_rollout(policies, state_env, seed_key)

                rollout_keys = jax.random.split(rollout_key, 10)
                rollouts = jax.vmap(_rollout_seed)(rollout_keys)
                take_idxs = jnp.arange(0, state_env.max_steps, 10)
                sampled = jax.tree_util.tree_map(
                    lambda x: x[:, take_idxs],
                    rollouts.state_seq,
                )
                return combine_first_two_tree_dim(sampled)

            run_combinations = list(itertools.permutations(range(num_runs), 2))
            run_combinations += [[i, i] for i in range(num_runs)]
            all_pairings = []
            for run_combination in run_combinations:
                def _get_policy(run_num):
                    params = jax.tree_util.tree_map(
                        lambda x: x[run_num],
                        previous_policies,
                    )
                    return E3TPPOParams(params=params)

                all_pairings.append(
                    PolicyPairing(*[_get_policy(run_num) for run_num in run_combination])
                )

            all_pairings = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *all_pairings)
            key, subkey = jax.random.split(key)
            combination_keys = jax.random.split(subkey, len(run_combinations))
            num_collect_batches = math.gcd(len(run_combinations), 10)
            state_collect_jit = jax.jit(_process_combination)
            state_buffer = scanned_mini_batch_map(
                state_collect_jit,
                num_collect_batches,
            )(all_pairings, combination_keys)
            state_buffer = combine_first_two_tree_dim(state_buffer)
            print("State buffer shape: ", jax.tree_util.tree_map(lambda x: x.shape, state_buffer), flush=True)
            print(f"Collecting states took {datetime.now() - start_time}", flush=True)
            return state_buffer

        key, subkey = jax.random.split(key)
        policy_init_keys = jax.random.split(subkey, num_runs)
        previous_policies = jax.vmap(_init_policy)(policy_init_keys)

        num_updates = (
            model_config["TOTAL_TIMESTEPS"]
            // model_config["NUM_STEPS"]
            // model_config["NUM_ENVS"]
        )
        update_step_offsets = jnp.linspace(
            0,
            num_updates,
            num_iterations + 1,
            dtype=jnp.int32,
        )
        print("Update step offsets: ", update_step_offsets, flush=True)

        prev_train_state = None
        for i in range(num_iterations):
            key, subkey = jax.random.split(key)
            state_buffer = _collect_states(previous_policies, subkey)
            update_step_start = update_step_offsets[i]
            update_step_end = update_step_offsets[i + 1]

            key, subkey = jax.random.split(key)
            out = _run_iteration(
                subkey,
                state_buffer,
                prev_train_state=prev_train_state,
                update_step_start=update_step_start,
                update_step_end=update_step_end,
            )
            prev_train_state = out["runner_state"][0]
            all_params = prev_train_state.params
            previous_policies = all_params

            final = i == num_iterations - 1
            for run_num in range(num_runs):
                params = jax.tree_util.tree_map(lambda x: x[run_num], all_params)
                store_checkpoint(
                    config,
                    params,
                    run_num,
                    i,
                    final=final,
                )

        if config["VISUALIZE"]:
            visualize_ppo_policy(
                config["RUN_BASE_DIR"],
                key=jax.random.PRNGKey(config["SEED"]),
                final_only=True,
                num_seeds=2,
            )
            visualize_ppo_policy(
                config["RUN_BASE_DIR"],
                key=jax.random.PRNGKey(config["SEED"]),
                final_only=True,
                num_seeds=500,
                cross=True,
                no_viz=True,
            )
