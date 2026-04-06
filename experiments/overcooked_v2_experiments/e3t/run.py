import copy
import math

import jax
import jax.numpy as jnp

from overcooked_v2_experiments.e3t.train import make_train
from overcooked_v2_experiments.e3t.utils.utils import get_num_devices
from overcooked_v2_experiments.utils.utils import mini_batch_pmap


def _concat_seed_batches(outputs):
    if len(outputs) == 1:
        return outputs[0]
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outputs)


def _slice_num_runs(tree, num_runs):
    return jax.tree_util.tree_map(lambda x: x[:num_runs], tree)


def _run_env_sharded(config, rng, num_devices):
    config_copy = copy.deepcopy(config)
    total_envs = config_copy["model"]["NUM_ENVS"]
    num_devices = min(num_devices, total_envs)
    if total_envs % num_devices != 0:
        num_devices = math.gcd(total_envs, num_devices)
    if num_devices <= 1:
        raise ValueError(
            f"ENV_SHARD_ACROSS_DEVICES requires at least 2 evenly-divisible devices, got total_envs={total_envs}, visible_devices={num_devices}"
        )

    config_copy["ENV_SHARD_ACROSS_DEVICES"] = True
    config_copy["ENV_SHARD_AXIS_NAME"] = "env_shard"
    config_copy["ENV_SHARD_TOTAL_ENVS"] = total_envs
    config_copy["model"]["NUM_ENVS"] = total_envs // num_devices

    print(
        f"[E3T] Env-sharded single-seed run across {num_devices} devices: total_envs={total_envs}, local_envs={config_copy['model']['NUM_ENVS']}",
        flush=True,
    )

    train_func = make_train(config_copy)
    parallel_train = jax.pmap(train_func, axis_name=config_copy["ENV_SHARD_AXIS_NAME"])
    device_rngs = jax.random.split(rng, num_devices)
    out = parallel_train(device_rngs)
    return jax.tree_util.tree_map(lambda x: x[:1], out)


def _run_seed_batched(config, rng):
    num_runs = config["NUM_SEEDS"]
    rngs = jax.random.split(rng, num_runs)
    train_func = make_train(copy.deepcopy(config))
    num_devices = min(get_num_devices(), num_runs)

    padded_num_runs = int(math.ceil(num_runs / num_devices) * num_devices)
    if padded_num_runs != num_runs:
        pad_needed = padded_num_runs - num_runs
        pad_rng = jax.random.split(jax.random.fold_in(rng, num_runs), pad_needed)
        rngs = jnp.concatenate([rngs, pad_rng], axis=0)
    num_seed_batches = max(padded_num_runs // num_devices, 1)

    print(
        f"[E3T] Using {num_devices} devices across {num_seed_batches} sequential seed batches (padded runs: {padded_num_runs})",
        flush=True,
    )

    if num_seed_batches == 1:
        return _slice_num_runs(mini_batch_pmap(train_func, num_devices)(rngs), num_runs)

    parallel_train = jax.pmap(train_func)
    seed_batches = rngs.reshape((num_seed_batches, num_devices, -1))
    outputs = []
    for batch_idx in range(num_seed_batches):
        print(
            f"[E3T] Launching seed batch {batch_idx + 1}/{num_seed_batches} on {num_devices} devices",
            flush=True,
        )
        outputs.append(parallel_train(seed_batches[batch_idx]))
    return _slice_num_runs(_concat_seed_batches(outputs), num_runs)


def single_run(config):
    if "FCP" in config or "BC" in config:
        raise NotImplementedError("The current E3T implementation supports standard self-play only.")

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        auto_env_shard = config.get("AUTO_ENV_SHARD_SINGLE_SEED", True)
        visible_devices = get_num_devices()
        env_shard = config.get("ENV_SHARD_ACROSS_DEVICES", False)
        if (
            not env_shard
            and auto_env_shard
            and config["NUM_SEEDS"] == 1
            and visible_devices > 1
        ):
            env_shard = True
        if env_shard:
            if config["NUM_SEEDS"] != 1:
                raise NotImplementedError("ENV_SHARD_ACROSS_DEVICES currently supports NUM_SEEDS=1 only.")
            return _run_env_sharded(config, rng, visible_devices)
        return _run_seed_batched(config, rng)
