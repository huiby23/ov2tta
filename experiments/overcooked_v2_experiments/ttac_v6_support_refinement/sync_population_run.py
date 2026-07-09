import copy

import jax
import jax.numpy as jnp

from overcooked_v2_experiments.ttac_v6_support_refinement.sync_population_train import make_train
from overcooked_v2_experiments.utils.utils import mini_batch_pmap
from overcooked_v2_experiments.ppo.utils.utils import get_num_devices

jax.config.update("jax_debug_nans", True)


def _choose_num_devices(num_runs):
    num_devices = min(get_num_devices(), num_runs)
    while num_devices > 1 and num_runs % num_devices != 0:
        num_devices -= 1
    print("Using", num_devices, "devices for", num_runs, "TTACv2 sync-population runs")
    return num_devices


def single_run(config):
    num_runs = config["NUM_SEEDS"]
    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_runs)
        run_indices = jnp.arange(num_runs, dtype=jnp.int32)

        config_copy = copy.deepcopy(config)
        train_func = make_train(config_copy)
        train_jit = jax.jit(train_func)
        num_devices = _choose_num_devices(num_runs)
        out = mini_batch_pmap(train_jit, num_devices)(rngs, run_indices)
        return out
