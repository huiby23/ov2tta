from __future__ import annotations

import copy

import jax
import jax.numpy as jnp

from overcooked_v2_experiments.talents.ippo import make_train
from overcooked_v2_experiments.talents.policy import TalentsGeneratedPartnerPolicy
from overcooked_v2_experiments.talents.utils.store import store_checkpoint
from overcooked_v2_experiments.talents.utils.utils import get_num_devices
from overcooked_v2_experiments.utils.utils import mini_batch_pmap


def single_run(config):
    num_runs = config["NUM_SEEDS"]
    tcfg = config.get("TALENTS", {})
    if not tcfg.get("VAE_CHECKPOINT") or not tcfg.get("CLUSTER_CHECKPOINT"):
        raise ValueError("TALENTS.VAE_CHECKPOINT and TALENTS.CLUSTER_CHECKPOINT are required.")
    population = TalentsGeneratedPartnerPolicy.from_checkpoints(
        tcfg["VAE_CHECKPOINT"],
        tcfg["CLUSTER_CHECKPOINT"],
        stochastic=tcfg.get("STOCHASTIC_PARTNER", True),
        z_sample_scale=tcfg.get("Z_SAMPLE_SCALE", 1.0),
    )
    config_copy = copy.deepcopy(config)
    config_copy["TALENTS"]["NUM_CLUSTERS"] = int(population.num_clusters)
    config_copy["model"]["NUM_STRATEGY_CLUSTERS"] = int(population.num_clusters)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, num_runs)
    run_indices = jnp.arange(num_runs, dtype=jnp.int32)
    train_func = make_train(config_copy)

    def train_with_talents_partner(rng, run_index):
        return train_func(rng, run_index, population=population)

    train_jit = jax.jit(train_with_talents_partner)
    num_devices = min(get_num_devices(), num_runs)
    print("Using", num_devices, "devices")
    return mini_batch_pmap(train_jit, num_devices)(rngs, run_indices)


def save_ppo_style_checkpoints(config, out):
    if config["NUM_CHECKPOINTS"] <= 0:
        return
    checkpoints = out["runner_state"][1]
    num_runs = jax.tree_util.tree_leaves(checkpoints)[0].shape[0]
    for run_num in range(num_runs):
        for checkpoint in range(config["NUM_CHECKPOINTS"]):
            params = jax.tree_util.tree_map(lambda x: x[run_num][checkpoint], checkpoints)
            store_checkpoint(config, params, run_num, checkpoint, final=checkpoint == config["NUM_CHECKPOINTS"] - 1)
