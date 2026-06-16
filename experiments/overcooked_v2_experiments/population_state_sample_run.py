"""State-sampling augmentation for real-population PPO variants.

This mirrors the PPO state_sample_run loop for methods that train an ego policy
with a real partner population (MEP / TrajeDi). Each iteration collects rollout
states from the current ego policies and their partner populations, injects the
states as ``initial_state_buffer``, and continues training from the previous ego
and partner TrainStates.
"""

import copy
import itertools
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl
import orbax.checkpoint as ocp
import wandb
from flax.training import orbax_utils
from omegaconf import OmegaConf
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.ppo.models.model import get_actor_critic, initialize_carry
from overcooked_v2_experiments.ppo.policy import PPOParams, policy_checkoints_to_policy_pairing
from overcooked_v2_experiments.ppo.utils.utils import combine_first_two_tree_dim, get_num_devices, get_run_base_dir
from overcooked_v2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy
from overcooked_v2_experiments.utils.utils import scanned_mini_batch_map


def _checkpoint_name(checkpoint, final=False):
    return "ckpt_final" if final else f"ckpt_{checkpoint}"


def _choose_num_devices(num_runs, label):
    num_devices = min(get_num_devices(), num_runs)
    while num_devices > 1 and num_runs % num_devices != 0:
        num_devices -= 1
    print(f"Using {num_devices} devices for {num_runs} {label} state-aug runs")
    return num_devices


def _reshape_for_devices(tree, num_devices):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_devices, -1, *x.shape[1:])), tree
    )


def _flatten_devices(tree):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])), tree
    )


def _save_orbax_checkpoint(path: Path, config, params):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    checkpoint = {"config": config, "params": params}
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(checkpoint)
    checkpointer.save(path, checkpoint, save_args=save_args)


def _save_population_checkpoint(run_base_dir, config, ego_params, partner_params, checkpoint, final=False):
    ckpt_name = _checkpoint_name(checkpoint, final=final)
    num_runs = jax.tree_util.tree_leaves(ego_params)[0].shape[0]
    population_size = jax.tree_util.tree_leaves(partner_params)[0].shape[1]

    for run_num in range(num_runs):
        ego_run_params = jax.tree_util.tree_map(lambda x: x[run_num], ego_params)
        _save_orbax_checkpoint(
            run_base_dir / f"run_{run_num}" / ckpt_name,
            config,
            ego_run_params,
        )
        _save_orbax_checkpoint(
            run_base_dir / f"run_{run_num}" / "ego" / ckpt_name,
            config,
            ego_run_params,
        )
        for partner_idx in range(population_size):
            partner_run_params = jax.tree_util.tree_map(
                lambda x: x[run_num, partner_idx], partner_params
            )
            _save_orbax_checkpoint(
                run_base_dir / f"run_{run_num}" / f"partner_{partner_idx}" / ckpt_name,
                config,
                partner_run_params,
            )


def _write_state_aug_audit(run_base_dir, config, label, num_runs, include_partner_pairings):
    audit = {
        "method": f"{label}-real-population-state-aug-overcooked-v2",
        "state_aug": {
            "num_iterations": config["NUM_ITERATIONS"],
            "mechanism": "PPO-style state-sampling augmentation with initial_state_buffer",
            "collects_from": "ego self/cross pairings plus same-run ego-partner population pairings" if include_partner_pairings else "ego self/cross pairings only",
            "continues_training": "ego TrainState and partner-population TrainState are carried across iterations",
        },
        "population_is_real": True,
        "num_independent_runs": num_runs,
        "config": config,
    }
    with open(run_base_dir / f"{label.lower()}_state_aug_audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)


def population_state_sample_run(
    config,
    *,
    label,
    make_train_fn,
    method_config_key,
    population_size_key="POPULATION_SIZE",
):
    print(f"Running {label} state sample run")
    config = OmegaConf.to_container(config, resolve=True)

    num_seeds = int(config["NUM_SEEDS"])
    num_iterations = int(config["NUM_ITERATIONS"])
    assert num_iterations > 0

    model_config = config["model"]
    env_config = config["env"]
    method_config = config.get(method_config_key, {})
    population_size = int(method_config.get(population_size_key, 5))
    include_partner_pairings = bool(method_config.get("STATE_AUG_INCLUDE_PARTNERS", True))
    num_state_rollouts = int(method_config.get("STATE_AUG_NUM_ROLLOUTS", 10))
    state_step_size = int(method_config.get("STATE_AUG_STEP_SIZE", 10))

    model_name = model_config["TYPE"]
    layout_name = env_config["ENV_KWARGS"]["layout"]
    agent_view_size = env_config["ENV_KWARGS"].get("agent_view_size", None)
    optional_prefix = config.get("OPTIONAL_PREFIX", "")
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    run_name = f"{label.lower()}_K{population_size}_{model_name}_ov2_{layout_name}_{avs_str}_sa-{num_iterations}"
    if optional_prefix:
        run_name = f"{optional_prefix}_{run_name}"

    with wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=[label, model_name, "OvercookedV2", "real-population", "state-aug"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    ) as run:
        run_id = run.id
        run_base_dir = get_run_base_dir(run_id, config)
        config["RUN_BASE_DIR"] = run_base_dir
        print(f"{label} state-aug run_dir", run_base_dir)

        key = jax.random.PRNGKey(config["SEED"])
        num_devices = _choose_num_devices(num_seeds, label)

        def _init_policy(policy_key):
            env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
            network = get_actor_critic(config)
            init_x = (
                jnp.zeros((1, model_config["NUM_ENVS"], *env.observation_space().shape)),
                jnp.zeros((1, model_config["NUM_ENVS"])),
            )
            init_hstate = initialize_carry(config, model_config["NUM_ENVS"])
            return network.init(policy_key, init_hstate, init_x)

        def _init_partner_population(pop_key):
            pop_keys = jax.random.split(pop_key, population_size)
            return jax.vmap(_init_policy)(pop_keys)

        def _run_iteration(
            iter_key,
            state_buffer,
            prev_ego_state,
            prev_partner_state,
            update_step_start,
            update_step_end,
        ):
            num_update_steps = int(update_step_end - update_step_start)
            config_copy = copy.deepcopy(config)
            config_copy["env"]["ENV_KWARGS"]["initial_state_buffer"] = state_buffer
            config_copy["NUM_CHECKPOINTS"] = 0

            keys = jax.random.split(iter_key, num_seeds)
            run_indices = jnp.arange(num_seeds, dtype=jnp.int32)
            train_jit = jax.jit(
                make_train_fn(
                    config_copy,
                    update_step_offset=int(update_step_start),
                    update_step_num_overwrite=num_update_steps,
                )
            )

            keys = _reshape_for_devices(keys, num_devices)
            run_indices = _reshape_for_devices(run_indices, num_devices)

            if prev_ego_state is None:
                ret = jax.pmap(jax.vmap(train_jit))(keys, run_indices)
            else:
                ret = jax.pmap(jax.vmap(train_jit))(
                    keys,
                    run_indices,
                    initial_ego_state=_reshape_for_devices(prev_ego_state, num_devices),
                    initial_partner_state=_reshape_for_devices(prev_partner_state, num_devices),
                )
            return _flatten_devices(ret)

        def _collect_states(previous_ego_params, previous_partner_params, collect_key):
            start_time = datetime.now()
            env_kwargs = copy.deepcopy(env_config["ENV_KWARGS"])
            env_kwargs.pop("initial_state_buffer", None)
            env = OvercookedV2(**env_kwargs)

            def _process_pairing(policies, rollout_key):
                policies = policy_checkoints_to_policy_pairing(policies, config)

                def _rollout_seed(seed_key):
                    return get_rollout(policies, env, seed_key)

                rollout_keys = jax.random.split(rollout_key, num_state_rollouts)
                rollouts = jax.vmap(_rollout_seed)(rollout_keys)
                take_idxs = jnp.arange(0, env.max_steps, state_step_size)
                sampled = jax.tree_util.tree_map(
                    lambda x: x[:, take_idxs], rollouts.state_seq
                )
                return combine_first_two_tree_dim(sampled)

            pairings = []

            ego_pairs = list(itertools.permutations(range(num_seeds), 2))
            ego_pairs += [(i, i) for i in range(num_seeds)]
            for lhs, rhs in ego_pairs:
                lhs_params = jax.tree_util.tree_map(lambda x: x[lhs], previous_ego_params)
                rhs_params = jax.tree_util.tree_map(lambda x: x[rhs], previous_ego_params)
                pairings.append(PolicyPairing(PPOParams(params=lhs_params), PPOParams(params=rhs_params)))

            if include_partner_pairings:
                for run_idx in range(num_seeds):
                    ego_params = jax.tree_util.tree_map(lambda x: x[run_idx], previous_ego_params)
                    for partner_idx in range(population_size):
                        partner_params = jax.tree_util.tree_map(
                            lambda x: x[run_idx, partner_idx], previous_partner_params
                        )
                        pairings.append(PolicyPairing(PPOParams(params=ego_params), PPOParams(params=partner_params)))
                        pairings.append(PolicyPairing(PPOParams(params=partner_params), PPOParams(params=ego_params)))

            all_pairings = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *pairings)
            collect_keys = jax.random.split(collect_key, len(pairings))
            num_collect_batches = math.gcd(len(pairings), 10)
            state_collect_jit = jax.jit(_process_pairing)
            state_buffer = scanned_mini_batch_map(state_collect_jit, num_collect_batches)(
                all_pairings, collect_keys
            )
            state_buffer = combine_first_two_tree_dim(state_buffer)
            print(
                f"{label} state buffer pairings={len(pairings)} rollouts={num_state_rollouts} step={state_step_size} shape=",
                jax.tree_util.tree_map(lambda x: x.shape, state_buffer),
            )
            print(f"Collecting {label} states took {datetime.now() - start_time}")
            return state_buffer

        key, subkey = jax.random.split(key)
        ego_init_keys = jax.random.split(subkey, num_seeds)
        previous_ego_params = jax.vmap(_init_policy)(ego_init_keys)
        key, subkey = jax.random.split(key)
        partner_init_keys = jax.random.split(subkey, num_seeds)
        previous_partner_params = jax.vmap(_init_partner_population)(partner_init_keys)

        num_updates = (
            model_config["TOTAL_TIMESTEPS"]
            // model_config["NUM_STEPS"]
            // model_config["NUM_ENVS"]
        )
        update_step_offsets = jnp.linspace(
            0, num_updates, num_iterations + 1, dtype=jnp.int32
        )
        print("Update step offsets:", update_step_offsets)

        prev_ego_state = None
        prev_partner_state = None

        for iteration in range(num_iterations):
            key, subkey = jax.random.split(key)
            state_buffer = _collect_states(previous_ego_params, previous_partner_params, subkey)

            update_step_start = int(update_step_offsets[iteration])
            update_step_end = int(update_step_offsets[iteration + 1])
            print(
                f"{label} state-aug iteration {iteration + 1}/{num_iterations}: "
                f"updates {update_step_start}->{update_step_end}"
            )
            key, subkey = jax.random.split(key)
            out = _run_iteration(
                subkey,
                state_buffer,
                prev_ego_state,
                prev_partner_state,
                update_step_start,
                update_step_end,
            )
            prev_ego_state = out["runner_state"][0]
            prev_partner_state = out["runner_state"][1]
            previous_ego_params = prev_ego_state.params
            previous_partner_params = prev_partner_state.params

            final = iteration == num_iterations - 1
            _save_population_checkpoint(
                run_base_dir,
                config,
                previous_ego_params,
                previous_partner_params,
                iteration,
                final=final,
            )

        _write_state_aug_audit(
            run_base_dir,
            config,
            label,
            num_seeds,
            include_partner_pairings,
        )

        if config.get("VISUALIZE", False):
            visualize_ppo_policy(
                run_base_dir,
                key=jax.random.PRNGKey(config["SEED"]),
                final_only=True,
                num_seeds=500,
                cross=True,
                no_viz=True,
            )

    return run_base_dir
