import copy

import jax

from overcooked_v2_experiments.e3t_ppo.train import make_train
from overcooked_v2_experiments.e3t_ppo.utils.utils import get_num_devices
from overcooked_v2_experiments.utils.utils import mini_batch_pmap


def single_run(config):
    if "FCP" in config or "BC" in config:
        raise NotImplementedError("E3T-PPO currently supports standard self-play only.")

    num_runs = config["NUM_SEEDS"]

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_runs)

        train_func = make_train(copy.deepcopy(config))
        num_devices = min(get_num_devices(), max(num_runs, 1))
        print(f"[E3T-PPO] Using {num_devices} devices for {num_runs} seeds", flush=True)

        return mini_batch_pmap(jax.jit(train_func), num_devices)(rngs)
