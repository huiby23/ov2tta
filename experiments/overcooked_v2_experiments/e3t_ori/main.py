import os
import sys

import hydra
import jax
import wandb
from omegaconf import OmegaConf

from overcooked_v2_experiments.e3t_ori.utils.utils import get_run_base_dir

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))


def _store_all_checkpoints(config, out, num_checkpoints):
    from overcooked_v2_experiments.e3t_ori.utils.store import store_checkpoint

    if num_checkpoints <= 0:
        return
    checkpoints = out["checkpoints"]
    num_runs = next(iter(jax.tree_util.tree_leaves(checkpoints))).shape[0]
    for run_num in range(num_runs):
        for checkpoint in range(num_checkpoints):
            params = jax.tree_util.tree_map(lambda x: x[run_num][checkpoint], checkpoints)
            store_checkpoint(config, params, run_num, checkpoint, final=checkpoint == num_checkpoints - 1)


def single_run_with_checkpoint(config):
    from overcooked_v2_experiments.e3t_ori.run import single_run

    config = OmegaConf.to_container(config)
    if config.get("TUNE", False):
        raise NotImplementedError("E3T original reproduction tuning entry is not implemented yet.")
    if "NUM_ITERATIONS" in config:
        raise NotImplementedError("E3T original reproduction state augmentation entry is not implemented yet.")
    if "FCP" in config or "BC" in config:
        raise NotImplementedError("E3T original reproduction currently supports standard self-play only.")

    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    optional_prefix = config.get("OPTIONAL_PREFIX", "")
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    run_name = f"e3t_ori_ov2_{layout_name}_{avs_str}"
    if optional_prefix:
        run_name = f"{optional_prefix}_{run_name}"

    run = wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=["E3T_ORI", "OvercookedV2"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    )
    try:
        config["RUN_BASE_DIR"] = get_run_base_dir(run.id, config)
        out = single_run(config)
        _store_all_checkpoints(config, out, config["NUM_CHECKPOINTS"])
    finally:
        if os.environ.get("E3T_SKIP_WANDB_FINISH") != "1":
            run.finish()

    if os.environ.get("E3T_FORCE_OS_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    single_run_with_checkpoint(config)


if __name__ == "__main__":
    main()
