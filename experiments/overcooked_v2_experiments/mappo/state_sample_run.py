import copy
import itertools
import math
from datetime import datetime

import jax
import jax.numpy as jnp
import jaxmarl
import wandb
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2, ObservationType
from omegaconf import OmegaConf

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.mappo.train import make_train
from overcooked_v2_experiments.mappo.policy import (
    MAPPOParams,
    MAPPOPolicy,
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.mappo.models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.mappo.utils.store import store_checkpoint
from overcooked_v2_experiments.mappo.utils.utils import (
    combine_first_two_tree_dim,
    get_num_devices,
    get_run_base_dir,
)
from overcooked_v2_experiments.mappo.utils.visualize import visualize_mappo_policy
from overcooked_v2_experiments.utils.utils import scanned_mini_batch_map

hp_indices = {
    "cramped_room": 2,
    "asymm_advantages": 5,
    "coord_ring": 8,
    "forced_coord": 5,
    "counter_circuit": 0,
}


def state_sample_run(config):
    print("Running MAPPO state sample run", flush=True)

    config = OmegaConf.to_container(config)

    if config["model"]["TYPE"] != "RNN":
        raise NotImplementedError(
            "MAPPO state augmentation currently supports the RNN configuration only."
        )

    num_seeds = config["NUM_SEEDS"]
    num_iterations = config["NUM_ITERATIONS"]
    assert num_iterations > 0

    model_name = config["model"]["TYPE"]
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    optional_prefix = config.get("OPTIONAL_PREFIX", "")
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    run_name = f"mappo_{model_name.lower()}_ov2_{layout_name}_{avs_str}_sa-{num_iterations}"
    if optional_prefix:
        run_name = f"{optional_prefix}_{run_name}"

    num_runs = num_seeds

    hp_policy = None
    if "BC" in config:
        from overcooked_v2_experiments.human_rl.imitation.bc_policy import BCPolicy

        print("Training with BC", flush=True)
        split = "all"
        run_id = hp_indices[layout_name]
        print(f"Loading BC policy from {layout_name}-{split}-{run_id}", flush=True)
        hp_policy = BCPolicy.from_pretrained(layout_name, split, run_id)

    with wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=["MAPPO", model_name, "OvercookedV2"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    ) as run:
        run_base_dir = get_run_base_dir(run.id, config)
        config["RUN_BASE_DIR"] = run_base_dir
        print("Run base dir:", run_base_dir, flush=True)

        key = jax.random.PRNGKey(config["SEED"])

        def _run_iteration(
            key,
            state_buffer,
            prev_train_state=None,
            update_step_start=None,
            update_step_end=None,
        ):
            num_update_steps = update_step_end - update_step_start

            config_copy = copy.deepcopy(config)
            config_copy["env"]["ENV_KWARGS"]["initial_state_buffer"] = state_buffer

            keys = jax.random.split(key, num_seeds)
            train_jit = jax.jit(
                make_train(
                    config_copy,
                    update_step_offset=update_step_start,
                    update_step_num_overwrite=num_update_steps,
                )
            )

            num_devices = min(get_num_devices(), num_seeds)
            if num_seeds % num_devices != 0:
                num_devices = math.gcd(num_seeds, num_devices)

            keys = keys.reshape((num_devices, -1, *keys.shape[1:]))
            if prev_train_state is not None:
                prev_train_state = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_devices, -1, *x.shape[1:])),
                    prev_train_state,
                )

            ret = jax.pmap(jax.vmap(train_jit))(
                keys,
                initial_train_state=prev_train_state,
            )

            return jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), ret)

        def _init_actor_policy(key):
            env_config = config["env"]
            model_config = config["model"]
            env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
            network = get_actor_critic(config)

            init_x = (
                jnp.zeros((1, model_config["NUM_ENVS"], *env.observation_space().shape)),
                jnp.zeros((1, model_config["NUM_ENVS"])),
            )
            init_hstate = initialize_carry(config, model_config["NUM_ENVS"])
            return network.init(key, init_hstate, init_x)

        def _collect_states(previous_policies, key):
            start_time = datetime.now()

            def _process_combination_wrapper(
                num_rollouts=10,
                state_step_size=10,
                from_policy_params=False,
                featurized_index=None,
            ):
                env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])

                if featurized_index is not None:
                    obs_type = [ObservationType.DEFAULT] * 2
                    obs_type[featurized_index] = ObservationType.FEATURIZED
                    env_kwargs["observation_type"] = obs_type

                env = OvercookedV2(**env_kwargs)

                def _process_combination(policies, key):
                    if from_policy_params:
                        policies = policy_checkoints_to_policy_pairing(policies, config)

                    def _rollout_seed(key):
                        return get_rollout(policies, env, key)

                    keys = jax.random.split(key, num_rollouts)
                    rollouts = jax.vmap(_rollout_seed)(keys)

                    state_sequences = rollouts.state_seq
                    take_idxs = jnp.arange(0, env.max_steps, state_step_size)
                    sampled_state_sequences = jax.tree_util.tree_map(
                        lambda x: x[:, take_idxs], state_sequences
                    )
                    return combine_first_two_tree_dim(sampled_state_sequences)

                return _process_combination

            if hp_policy is None:
                run_combinations = list(itertools.permutations(range(num_runs), 2))
                run_combinations += [[i, i] for i in range(num_runs)]

                all_pairings = []
                for run_combination in run_combinations:
                    def _get_policy(run_num):
                        params = jax.tree_util.tree_map(
                            lambda x: x[run_num], previous_policies
                        )
                        return MAPPOParams(params=params)

                    policy_combination = PolicyPairing(
                        *[_get_policy(run_num) for run_num in run_combination]
                    )
                    all_pairings.append(policy_combination)

                all_pairings = jax.tree_util.tree_map(
                    lambda *v: jnp.stack(v), *all_pairings
                )
                key, subkey = jax.random.split(key)
                comb_keys = jax.random.split(subkey, len(run_combinations))
                num_collect_batches = math.gcd(len(run_combinations), 10)
                state_collect_jit = jax.jit(
                    _process_combination_wrapper(
                        num_rollouts=10,
                        state_step_size=10,
                        from_policy_params=True,
                    )
                )
                state_buffer = scanned_mini_batch_map(
                    state_collect_jit, num_collect_batches
                )(all_pairings, comb_keys)
                state_buffer = combine_first_two_tree_dim(state_buffer)
            else:
                _state_collect_func_0 = _process_combination_wrapper(
                    num_rollouts=10,
                    state_step_size=10,
                    featurized_index=0,
                )
                _state_collect_func_1 = _process_combination_wrapper(
                    num_rollouts=10,
                    state_step_size=10,
                    featurized_index=1,
                )

                @jax.jit
                def _collect_bc_states(params, key):
                    mappo_policy = MAPPOPolicy(params, config)
                    mappo_hp_pairing = PolicyPairing(mappo_policy, hp_policy)
                    hp_mappo_pairing = PolicyPairing(hp_policy, mappo_policy)

                    hp_mappo_states = _state_collect_func_0(hp_mappo_pairing, key)
                    mappo_hp_states = _state_collect_func_1(mappo_hp_pairing, key)
                    return jax.tree_util.tree_map(
                        lambda x, y: jnp.concatenate([x, y], axis=0),
                        hp_mappo_states,
                        mappo_hp_states,
                    )

                key, subkey = jax.random.split(key)
                collect_keys = jax.random.split(subkey, num_seeds)
                state_buffer = jax.vmap(_collect_bc_states)(previous_policies, collect_keys)
                state_buffer = combine_first_two_tree_dim(state_buffer)

            print(
                "State buffer shape:",
                jax.tree_util.tree_map(lambda x: x.shape, state_buffer),
                flush=True,
            )
            print(f"Collecting states took {datetime.now() - start_time}", flush=True)
            return state_buffer

        key, subkey = jax.random.split(key)
        policy_init_keys = jax.random.split(subkey, num_runs)
        previous_policies = jax.vmap(_init_actor_policy)(policy_init_keys)

        model_config = config["model"]
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
        print("Update step offsets:", update_step_offsets, flush=True)

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
            actor_train_state, critic_train_state = prev_train_state
            all_params = {
                "actor": actor_train_state.params,
                "critic": critic_train_state.params,
            }
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
            visualize_mappo_policy(
                run_base_dir,
                key=jax.random.PRNGKey(config["SEED"]),
                final_only=True,
                num_seeds=2,
            )
            visualize_mappo_policy(
                run_base_dir,
                key=jax.random.PRNGKey(config["SEED"]),
                final_only=True,
                num_seeds=500,
                cross=True,
                no_viz=True,
            )
