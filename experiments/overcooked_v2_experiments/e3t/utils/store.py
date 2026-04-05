import os
import pickle

import orbax.checkpoint as ocp
from flax.training import orbax_utils

from overcooked_v2_experiments.e3t.policy import E3TParams


def _stored_filenames(filename_base):
    model_filename = os.path.join(filename_base, "model.pkl")
    config_filename = os.path.join(filename_base, "config.pkl")
    return model_filename, config_filename


def store_model(network_params, config, filename_base):
    model_filename, config_filename = _stored_filenames(filename_base)
    with open(model_filename, "wb") as f:
        pickle.dump(network_params, f)
    with open(config_filename, "wb") as f:
        pickle.dump(config, f)


def load_model(filename_base):
    model_filename, config_filename = _stored_filenames(filename_base)
    with open(model_filename, "rb") as f:
        network_params = pickle.load(f)
    with open(config_filename, "rb") as f:
        config = pickle.load(f)
    return network_params, config


def _get_checkpoint_dir(run_base_dir, run_num, checkpoint, final=False):
    ckpt_name = "ckpt_final" if final else f"ckpt_{checkpoint}"
    return (run_base_dir / f"run_{run_num}" / ckpt_name).resolve()


def store_checkpoint(config, params, run_num, checkpoint, final=False):
    checkpoint_dir = _get_checkpoint_dir(config["RUN_BASE_DIR"], run_num, checkpoint, final=final)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint = {"config": config, "params": params}
    save_args = orbax_utils.save_args_from_target(checkpoint)
    orbax_checkpointer.save(checkpoint_dir, checkpoint, save_args=save_args)


def load_checkpoint(run_dir, run_num, checkpoint):
    checkpoint_dir = _get_checkpoint_dir(run_dir, run_num, checkpoint)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(checkpoint_dir, item=None)
    return ckpt["config"], ckpt["params"]


def load_all_checkpoints(run_dir, final_only=True, skip_initial=False):
    first_config = None
    all_checkpoints = {}
    for run_num_dir in run_dir.iterdir():
        if not run_num_dir.is_dir() or "run_" not in run_num_dir.name:
            continue
        checkpoints = {}
        for checkpoint_dir in run_num_dir.iterdir():
            if not checkpoint_dir.is_dir() or "ckpt_" not in checkpoint_dir.name:
                continue
            if final_only and "final" not in checkpoint_dir.name:
                continue
            if skip_initial and "ckpt_0" in checkpoint_dir.name:
                continue
            ckpt_id = checkpoint_dir.name.split("_")[1]
            config, params = load_checkpoint(run_dir, int(run_num_dir.name.split("_")[1]), ckpt_id)
            checkpoints[checkpoint_dir.name] = E3TParams(params=params)
            if first_config is None:
                first_config = config
        all_checkpoints[run_num_dir.name] = checkpoints
    return all_checkpoints, first_config
