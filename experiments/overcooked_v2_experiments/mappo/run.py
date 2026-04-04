import copy
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from overcooked_v2_experiments.mappo.utils.utils import get_num_devices
from overcooked_v2_experiments.utils.utils import mini_batch_pmap

from .train import make_train


def load_fcp_populations(population_dir):
    from overcooked_v2_experiments.mappo.utils.store import load_all_checkpoints
    def _load_fcp_population(dir):
        all_checkpoints, fcp_config = load_all_checkpoints(
            dir, final_only=False, skip_initial=True
        )
        all_population_params, _ = jax.tree_util.tree_flatten(
            all_checkpoints, is_leaf=lambda x: hasattr(x, "params")
        )
        all_population_params = jax.tree_util.tree_map(
            lambda *v: jnp.stack(v), *all_population_params
        )
        return all_population_params, fcp_config

    all_populations = []
    first_fcp_config = None
    for dir in population_dir.iterdir():
        if not dir.is_dir() or "fcp_" not in dir.name:
            continue
        population, fcp_config = _load_fcp_population(dir)
        all_populations.append(population)
        if first_fcp_config is None:
            first_fcp_config = fcp_config

    all_populations = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *all_populations)
    return all_populations, first_fcp_config


def single_run(config):
    num_seeds = config["NUM_SEEDS"]
    num_runs = num_seeds

    all_populations = None
    if "FCP" in config:
        assert num_seeds == 1
        population_dir = Path(config["FCP"])
        all_populations, fcp_population_config = load_fcp_populations(population_dir)
        all_populations = all_populations.params
        num_runs = jax.tree_util.tree_flatten(all_populations)[0][0].shape[0]

    bc_policy = None
    if "BC" in config:
        from overcooked_v2_experiments.human_rl.imitation.bc_policy import BCPolicy

        layout_name = config["env"]["ENV_KWARGS"]["layout"]
        split = "all"
        run_id = 1
        bc_policy = BCPolicy.from_pretrained(layout_name, split, run_id)

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_runs)

        config_copy = copy.deepcopy(config)
        if bc_policy is not None:
            config_copy["env"]["ENV_KWARGS"]["force_path_planning"] = True

        population_config = None
        if all_populations is not None:
            population_config = fcp_population_config

        print("[MAPPO] make_train", flush=True)
        train_func = make_train(config_copy, population_config=population_config)
        print("[MAPPO] jitting train function", flush=True)
        train_jit = jax.jit(train_func)

        num_devices = min(get_num_devices(), num_runs)
        if num_runs % num_devices != 0:
            num_devices = math.gcd(num_runs, num_devices)
        print("Using", num_devices, "devices", flush=True)

        train_extra_args = {}
        if all_populations is not None:
            train_extra_args["population"] = all_populations
        elif bc_policy is not None:
            train_extra_args["population"] = bc_policy

        print("[MAPPO] launching mini_batch_pmap", flush=True)
        out = mini_batch_pmap(train_jit, num_devices)(rngs, **train_extra_args)
        print("[MAPPO] mini_batch_pmap returned", flush=True)
        return out
