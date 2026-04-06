import copy
import math

import jax
import jax.numpy as jnp

from overcooked_v2_experiments.e3t_cec.train import make_train
from overcooked_v2_experiments.e3t_cec.utils.utils import get_num_devices
from overcooked_v2_experiments.utils.utils import mini_batch_pmap


def _concat_seed_batches(outputs):
    if len(outputs) == 1:
        return outputs[0]
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outputs)


def _slice_num_runs(tree, num_runs):
    return jax.tree_util.tree_map(lambda x: x[:num_runs], tree)


def _run_seed_batched(config, rng):
    num_runs = config["NUM_SEEDS"]
    rngs = jax.random.split(rng, num_runs)
    train_func = make_train(copy.deepcopy(config))
    num_devices = min(get_num_devices(), max(num_runs, 1))

    padded_num_runs = int(math.ceil(num_runs / num_devices) * num_devices)
    if padded_num_runs != num_runs:
        pad_needed = padded_num_runs - num_runs
        pad_rng = jax.random.split(jax.random.fold_in(rng, num_runs), pad_needed)
        rngs = jnp.concatenate([rngs, pad_rng], axis=0)
    num_seed_batches = max(padded_num_runs // num_devices, 1)

    print(
        f"[E3T-CEC] Using {num_devices} devices across {num_seed_batches} sequential seed batches (padded runs: {padded_num_runs})",
        flush=True,
    )

    if num_seed_batches == 1:
        return _slice_num_runs(mini_batch_pmap(train_func, num_devices)(rngs), num_runs)

    parallel_train = jax.pmap(train_func)
    seed_batches = rngs.reshape((num_seed_batches, num_devices, -1))
    outputs = []
    for batch_idx in range(num_seed_batches):
        print(
            f"[E3T-CEC] Launching seed batch {batch_idx + 1}/{num_seed_batches} on {num_devices} devices",
            flush=True,
        )
        outputs.append(parallel_train(seed_batches[batch_idx]))
    return _slice_num_runs(_concat_seed_batches(outputs), num_runs)


def single_run(config):
    if "FCP" in config or "BC" in config:
        raise NotImplementedError("E3T-CEC currently supports standard self-play only.")

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        return _run_seed_batched(config, rng)
