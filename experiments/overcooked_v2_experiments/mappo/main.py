from pathlib import Path
import os
import sys

import hydra
import jax
import wandb
from omegaconf import OmegaConf

from overcooked_v2_experiments.mappo.utils.utils import get_run_base_dir

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))

jax.config.update("jax_debug_nans", True)


def _store_all_checkpoints(config, out, num_checkpoints):
    from overcooked_v2_experiments.mappo.utils.store import store_checkpoint

    if num_checkpoints <= 0:
        return

    checkpoints = out["runner_state"][1]
    num_runs = jax.tree_util.tree_flatten(checkpoints)[0][0].shape[0]
    print(f"Saving MAPPO checkpoints for {num_runs} runs to {config['RUN_BASE_DIR']}")
    for run_num in range(num_runs):
        for checkpoint in range(num_checkpoints):
            params = jax.tree_util.tree_map(
                lambda x: x[run_num][checkpoint], checkpoints
            )
            store_checkpoint(
                config,
                params,
                run_num,
                checkpoint,
                final=checkpoint == num_checkpoints - 1,
            )


def single_run_with_checkpoint(config):
    print("[MAPPO] single_run_with_checkpoint entry", flush=True)
    from overcooked_v2_experiments.mappo.run import single_run

    config = OmegaConf.to_container(config)
    if config["model"]["TYPE"] != "RNN":
        raise NotImplementedError(
            "The current MAPPO baseline is implemented for the RNN configuration only."
        )
    if config.get("TUNE", False):
        raise NotImplementedError("MAPPO tuning entry is not implemented yet.")

    num_checkpoints = config["NUM_CHECKPOINTS"]
    model_name = config["model"]["TYPE"]
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    optional_prefix = config.get("OPTIONAL_PREFIX", "")
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    run_name = f"mappo_{model_name.lower()}_ov2_{layout_name}_{avs_str}"
    if "FCP" in config:
        population_dir = Path(config["FCP"])
        run_name = f"FCP_{population_dir.name}"
    if optional_prefix:
        run_name = f"{optional_prefix}_{run_name}"

    print("[MAPPO] wandb.init", flush=True)
    run = wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=["MAPPO", model_name, "OvercookedV2"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    )
    try:
        print(f"[MAPPO] wandb run id: {run.id}", flush=True)
        run_base_dir = get_run_base_dir(run.id, config)
        config["RUN_BASE_DIR"] = run_base_dir
        print("[MAPPO] entering single_run", flush=True)
        out = single_run(config)
        print("[MAPPO] single_run returned", flush=True)
        print("[MAPPO] storing checkpoints", flush=True)
        _store_all_checkpoints(config, out, num_checkpoints)
        print("[MAPPO] checkpoints stored", flush=True)
    finally:
        if os.environ.get("MAPPO_SKIP_WANDB_FINISH") != "1":
            run.finish()

    if os.environ.get("MAPPO_FORCE_OS_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    print("[MAPPO] hydra main entry", flush=True)
    if config.get("TUNE", False):
        raise NotImplementedError("MAPPO tuning entry is not implemented yet.")
    if "NUM_ITERATIONS" in config:
        from overcooked_v2_experiments.mappo.state_sample_run import state_sample_run

        state_sample_run(config)
    else:
        single_run_with_checkpoint(config)


if __name__ == "__main__":
    main()
