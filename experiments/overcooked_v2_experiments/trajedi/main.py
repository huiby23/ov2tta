from pathlib import Path
import json
import os
import shutil
import sys

import hydra
import jax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from omegaconf import OmegaConf
import wandb

from overcooked_v2_experiments.trajedi.run import single_run
from overcooked_v2_experiments.ppo.utils.utils import get_run_base_dir

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))

jax.config.update("jax_debug_nans", True)


def _checkpoint_name(checkpoint, final=False):
    return "ckpt_final" if final else f"ckpt_{checkpoint}"


def _save_orbax_checkpoint(path: Path, config, params):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    checkpoint = {"config": config, "params": params}
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(checkpoint)
    checkpointer.save(path, checkpoint, save_args=save_args)


def _write_audit(run_base_dir: Path, config, num_runs: int):
    audit = {
        "method": "TrajeDi-sim-MI-real-population-overcooked-v2",
        "source_reference": {
            "repo": "https://github.com/huiby23/ZSC/tree/PBL-subnet",
            "source_files": [
                "pyhanabi/r2d2.py:get_sim_mi",
                "pyhanabi/selfplay.py:div_type/div_weight/no_sharing",
                "pyhanabi/net.py:play_styles action-stack",
            ],
        },
        "faithfulness": {
            "migrated_objective": "maximize log pi_current(a|s) - log mean_k pi_k(a|s) for partner population policies",
            "not_migrated_by_design": "single-network play_styles/subnet heads; replaced by real independent PPO partner policies",
            "population_is_real": True,
            "num_independent_runs": num_runs,
        },
        "config": config,
    }
    with open(run_base_dir / "trajedi_reproduction_audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)


def single_run_with_viz(config):
    config = OmegaConf.to_container(config, resolve=True)
    num_checkpoints = config["NUM_CHECKPOINTS"]
    model_name = config["model"]["TYPE"]
    layout_name = config["env"]["ENV_KWARGS"]["layout"]
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    optional_prefix = config.get("OPTIONAL_PREFIX", "")
    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    trajedi_cfg = config.get("TRAJEDI", {})
    pop_size = trajedi_cfg.get("POPULATION_SIZE", 5)
    run_name = f"trajedi_K{pop_size}_{model_name}_ov2_{layout_name}_{avs_str}"
    if optional_prefix:
        run_name = f"{optional_prefix}_{run_name}"

    with wandb.init(
        entity=config["wandb"]["ENTITY"],
        project=config["wandb"]["PROJECT"],
        tags=["TrajeDi", "div0", model_name, "OvercookedV2", "real-population"],
        config=config,
        mode=config["wandb"]["WANDB_MODE"],
        name=run_name,
    ) as run:
        run_id = run.id
        run_base_dir = get_run_base_dir(run_id, config)
        config["RUN_BASE_DIR"] = run_base_dir
        out = single_run(config)

    if num_checkpoints > 0:
        ego_checkpoints = out["runner_state"][2]
        partner_checkpoints = out["runner_state"][3]
        num_runs = jax.tree_util.tree_leaves(ego_checkpoints)[0].shape[0]
        population_size = jax.tree_util.tree_leaves(partner_checkpoints)[0].shape[2]
        for run_num in range(num_runs):
            for checkpoint in range(num_checkpoints):
                final = checkpoint == num_checkpoints - 1
                ckpt_name = _checkpoint_name(checkpoint, final=final)
                ego_params = jax.tree_util.tree_map(
                    lambda x: x[run_num][checkpoint], ego_checkpoints
                )
                # Compatibility checkpoint: existing PPO visualizers can load ego from run_i/ckpt_final.
                _save_orbax_checkpoint(
                    run_base_dir / f"run_{run_num}" / ckpt_name,
                    config,
                    ego_params,
                )
                _save_orbax_checkpoint(
                    run_base_dir / f"run_{run_num}" / "ego" / ckpt_name,
                    config,
                    ego_params,
                )
                for partner_idx in range(population_size):
                    partner_params = jax.tree_util.tree_map(
                        lambda x: x[run_num][checkpoint][partner_idx],
                        partner_checkpoints,
                    )
                    _save_orbax_checkpoint(
                        run_base_dir
                        / f"run_{run_num}"
                        / f"partner_{partner_idx}"
                        / ckpt_name,
                        config,
                        partner_params,
                    )
        _write_audit(run_base_dir, config, num_runs)

    print("TrajeDi run_dir", run_base_dir)


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    print(config)
    if config["TUNE"]:
        raise NotImplementedError("TrajeDi tune is not implemented.")
    if "NUM_ITERATIONS" in config:
        from overcooked_v2_experiments.population_state_sample_run import (
            population_state_sample_run,
        )
        from overcooked_v2_experiments.trajedi.train import make_train

        population_state_sample_run(
            config,
            label="TRAJEDI",
            make_train_fn=make_train,
            method_config_key="TRAJEDI",
        )
        return
    single_run_with_viz(config)


if __name__ == "__main__":
    main()
