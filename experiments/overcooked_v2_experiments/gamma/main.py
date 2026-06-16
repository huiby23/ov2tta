
from __future__ import annotations

import os
import sys

import hydra
import jax
import wandb
from omegaconf import OmegaConf

from overcooked_v2_experiments.gamma.run import save_ppo_style_checkpoints, single_run
from overcooked_v2_experiments.ppo.utils.utils import get_run_base_dir
from overcooked_v2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))

jax.config.update("jax_debug_nans", True)


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    model_name = config["model"]["TYPE"]
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    avs = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    avs_str = f"avs-{avs}" if avs is not None else "avs-full"
    optional_prefix = config.get("OPTIONAL_PREFIX", "")
    run_name = f"gamma_{model_name}_ov2_{layout_name}_{avs_str}"
    if optional_prefix:
        run_name = f"{optional_prefix}_{run_name}"
    with wandb.init(entity=config["wandb"]["ENTITY"], project=config["wandb"]["PROJECT"], tags=["GAMMA", model_name, "OvercookedV2", "generated-partner"], config=config, mode=config["wandb"]["WANDB_MODE"], name=run_name) as run:
        run_base_dir = get_run_base_dir(run.id, config)
        config["RUN_BASE_DIR"] = run_base_dir
        out = single_run(config)
    save_ppo_style_checkpoints(config, out)
    print("GAMMA run_dir", run_base_dir)
    if config.get("VISUALIZE", False):
        visualize_ppo_policy(run_base_dir, key=jax.random.PRNGKey(config["SEED"]), final_only=True, num_seeds=500, cross=True, no_viz=True)


if __name__ == "__main__":
    main()
