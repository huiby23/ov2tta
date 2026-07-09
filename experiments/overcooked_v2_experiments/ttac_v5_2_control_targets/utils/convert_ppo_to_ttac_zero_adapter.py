from __future__ import annotations

import argparse
import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl
import orbax.checkpoint as ocp
from flax import core
from flax.training import orbax_utils

from overcooked_v2_experiments.ttac_v5_2_control_targets.models.model import (
    get_actor_critic,
    initialize_carry,
)


TTAC_DEFAULTS = {
    "TTAC_ADAPTER_DIM": 64,
    "TTAC_ADAPTER_SCALE": 0.5,
    "TTAC_ZERO_INIT_ADAPTER_UP": True,
    "TTAC_TRAIN_ADAPTER_READOUT_SCALE": 1.0,
    "TTAC_AGREEMENT_COEF": 0.0,
    "TTAC_TRAIN_KL_COEF": 0.0,
    "TTAC_TRAIN_SURROGATE": "none_posthoc_zero_adapter",
    "TTAC_TRAIN_PROJECT_BETA": 2.0,
    "TTAC_TRAIN_SUPPORT_MIN_PROB": 0.05,
    "TTAC_TRAIN_SUPPORT_MAX_ENTROPY": 1.5,
    "TTAC_TRAIN_ADVANTAGE_POWER": 1.0,
    "TTAC_HISTORY_LEN": 50,
    "TTAC_TEST_UPDATE_STEPS": 3,
    "TTAC_TEST_LR": 0.003,
    "TTAC_TEST_HIST_KL_COEF": 0.0,
    "TTAC_TEST_EGO_KL_COEF": 0.01,
    "TTAC_TEST_CUR_KL_COEF": 0.01,
    "TTAC_TEST_ENTROPY_COEF": 0.0,
    "TTAC_TEST_PROJECT_BETA": 2.0,
    "TTAC_TEST_SUPPORT_MIN_PROB": 0.05,
    "TTAC_TEST_SUPPORT_MAX_ENTROPY": 1.5,
    "TTAC_TEST_ADVANTAGE_POWER": 1.0,
    "TTAC_TEST_VALUE_GATE_TEMP": 10.0,
}


def _load_ckpt(path: Path):
    return ocp.PyTreeCheckpointer().restore(path, item=None)


def _save_ckpt(path: Path, ckpt):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.save(path, ckpt, save_args=save_args)


def _init_ttac_params(config: dict, seed: int):
    env_config = config["env"]
    model_config = config["model"]
    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    network = get_actor_critic(config)
    init_x = (
        jnp.zeros((1, model_config["NUM_ENVS"], *env.observation_space().shape)),
        jnp.zeros((1, model_config["NUM_ENVS"])),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    init_hstate = initialize_carry(config, model_config["NUM_ENVS"])
    return network.init(jax.random.PRNGKey(seed), init_hstate, init_x)


def _migrate_params(ppo_params, ttac_params):
    ppo = ppo_params["params"]
    ttac = core.unfreeze(ttac_params)

    ttac["params"]["CNNSimple_0"] = core.unfreeze(ppo["CNNSimple_0"])
    ttac["params"]["feature_norm"] = core.unfreeze(ppo["LayerNorm_0"])
    ttac["params"]["base_actor_hidden"] = core.unfreeze(ppo["Dense_0"])
    ttac["params"]["base_actor_logits"] = core.unfreeze(ppo["Dense_1"])
    ttac["params"]["critic_hidden"] = core.unfreeze(ppo["Dense_2"])
    ttac["params"]["critic_value"] = core.unfreeze(ppo["Dense_3"])

    # Enforce exact zero readout at conversion time even if config defaults drift.
    ttac["params"]["ttac_adapter_up"]["kernel"] = jnp.zeros_like(
        ttac["params"]["ttac_adapter_up"]["kernel"]
    )
    ttac["params"]["ttac_adapter_up"]["bias"] = jnp.zeros_like(
        ttac["params"]["ttac_adapter_up"]["bias"]
    )
    return core.freeze(ttac)


def convert_run(
    ppo_run_dir: Path,
    output_run_dir: Path,
    adapter_scale: float,
    test_lr: float,
    test_update_steps: int,
    history_len: int,
    force: bool,
):
    ppo_run_dir = ppo_run_dir.resolve()
    output_run_dir = output_run_dir.resolve()
    run_dirs = sorted(
        [p for p in ppo_run_dir.glob("run_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under {ppo_run_dir}")

    output_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        "# PPO State-Aug To TTAC Zero-Adapter Conversion",
        "",
        f"source: `{ppo_run_dir}`",
        f"output: `{output_run_dir}`",
        f"adapter_scale: `{adapter_scale}`",
        f"test_lr: `{test_lr}`",
        f"test_update_steps: `{test_update_steps}`",
        f"history_len: `{history_len}`",
        "",
        "## Runs",
    ]

    for run_dir in run_dirs:
        run_idx = int(run_dir.name.split("_")[1])
        src_ckpt = run_dir / "ckpt_final"
        dst_ckpt = output_run_dir / run_dir.name / "ckpt_final"
        if dst_ckpt.exists() and not force:
            raise FileExistsError(f"Destination exists; pass --force: {dst_ckpt}")
        if dst_ckpt.exists() and force:
            import shutil

            shutil.rmtree(dst_ckpt)

        ppo_ckpt = _load_ckpt(src_ckpt)
        config = copy.deepcopy(ppo_ckpt["config"])
        model = config["model"]
        if model["TYPE"] != "CNN":
            raise ValueError(f"Only CNN PPO checkpoints are supported, got {model['TYPE']}")
        for key, value in TTAC_DEFAULTS.items():
            model[key] = value
        model["TTAC_ADAPTER_DIM"] = int(model.get("FC_DIM_SIZE", 64))
        model["TTAC_ADAPTER_SCALE"] = float(adapter_scale)
        model["TTAC_TEST_LR"] = float(test_lr)
        model["TTAC_TEST_UPDATE_STEPS"] = int(test_update_steps)
        model["TTAC_HISTORY_LEN"] = int(history_len)
        config["RUN_BASE_DIR"] = output_run_dir
        config["OPTIONAL_PREFIX"] = "ttac_posthoc_zero_adapter_from_ppo_state_aug"
        config["TTAC_POSTHOC_BASE_RUN_DIR"] = str(ppo_run_dir)
        config["TTAC_POSTHOC_CONVERSION"] = {
            "source_run_dir": str(ppo_run_dir),
            "source_run": run_dir.name,
            "adapter_init": "zero_up",
            "base_policy": "frozen PPO CNN state-aug",
        }

        ttac_init = _init_ttac_params(config, seed=10_000 + run_idx)
        ttac_params = _migrate_params(ppo_ckpt["params"], ttac_init)
        _save_ckpt(dst_ckpt, {"config": config, "params": ttac_params})
        manifest_lines.append(f"- `{run_dir.name}`: `{src_ckpt}` -> `{dst_ckpt}`")
        print(f"converted {run_dir.name}: {src_ckpt} -> {dst_ckpt}", flush=True)

    (output_run_dir / "conversion_manifest.md").write_text("\n".join(manifest_lines) + "\n")
    print(f"wrote manifest: {output_run_dir / 'conversion_manifest.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_run_dir", type=Path, required=True)
    parser.add_argument("--output_run_dir", type=Path, required=True)
    parser.add_argument("--adapter_scale", type=float, default=0.5)
    parser.add_argument("--test_lr", type=float, default=0.003)
    parser.add_argument("--test_update_steps", type=int, default=3)
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert_run(
        args.ppo_run_dir,
        args.output_run_dir,
        args.adapter_scale,
        args.test_lr,
        args.test_update_steps,
        args.history_len,
        args.force,
    )


if __name__ == "__main__":
    main()
