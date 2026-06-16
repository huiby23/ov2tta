import copy
from pathlib import Path

import jax
import jax.numpy as jnp

from overcooked_v2_experiments.fcp_mep_frozen_mm.train import make_train
from overcooked_v2_experiments.utils.utils import mini_batch_pmap
from overcooked_v2_experiments.ppo.utils.store import load_checkpoint
from overcooked_v2_experiments.ppo.utils.utils import get_num_devices

jax.config.update("jax_debug_nans", True)


def _choose_num_devices(num_runs):
    num_devices = min(get_num_devices(), num_runs)
    while num_devices > 1 and num_runs % num_devices != 0:
        num_devices -= 1
    print("Using", num_devices, "devices for", num_runs, "FCP-MEP-Frozen-MM runs")
    return num_devices


def _load_frozen_population(config):
    cfg = config.get("FCP_MEP_FROZEN_MM", {})
    pop_dir = cfg.get("POPULATION_RUN_DIR")
    if not pop_dir:
        raise ValueError("FCP_MEP_FROZEN_MM.POPULATION_RUN_DIR is required")
    pop_dir = Path(pop_dir)
    if not pop_dir.exists():
        raise FileNotFoundError(f"Missing population run dir: {pop_dir}")
    population_size = int(cfg.get("POPULATION_SIZE", 10))
    checkpoint = str(cfg.get("POPULATION_CHECKPOINT", "final"))
    params = []
    for run_num in range(population_size):
        _, p = load_checkpoint(pop_dir, run_num, checkpoint)
        params.append(p)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *params)
    print("Loaded frozen FCP-MEP-Frozen-MM population", pop_dir, "K", population_size, "checkpoint", checkpoint)
    return stacked


def single_run(config):
    num_runs = config["NUM_SEEDS"]
    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_runs)
        run_indices = jnp.arange(num_runs, dtype=jnp.int32)

        config_copy = copy.deepcopy(config)
        frozen_partner_params = _load_frozen_population(config_copy)
        frozen_partner_params = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_runs,) + x.shape), frozen_partner_params
        )
        train_func = make_train(config_copy)
        train_jit = jax.jit(train_func)
        num_devices = _choose_num_devices(num_runs)
        out = mini_batch_pmap(train_jit, num_devices)(
            rngs, run_indices, frozen_partner_params
        )
        return out
