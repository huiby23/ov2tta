#!/usr/bin/env python3
"""OV2 adapter for JaxMARL QMIX RNN training.

The upstream qmix_rnn.py has an Overcooked branch for the older Overcooked
layout registry. This wrapper keeps the upstream training logic but patches
environment construction for overcooked_v2 layouts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "JaxMARL"))
sys.path.insert(0, str(ROOT / "JaxMARL" / "baselines" / "QLearning"))

from jaxmarl import make  # noqa: E402
from jaxmarl.wrappers.baselines import LogWrapper, save_params  # noqa: E402

import qmix_rnn  # noqa: E402


def ov2_env_from_config(config):
    env_name = config["ENV_NAME"]
    if env_name == "overcooked_v2":
        layout = config["ENV_KWARGS"]["layout"]
        env = make(env_name, **config["ENV_KWARGS"])
        env = LogWrapper(env, replace_info=True)
        return env, f"{env_name}_{layout}"
    return qmix_rnn._original_env_from_config(config)


def build_config(args: argparse.Namespace) -> dict:
    return {
        "NUM_SEEDS": args.num_seeds,
        "SEED": args.seed,
        "HYP_TUNE": False,
        "ENTITY": "",
        "PROJECT": args.project,
        "WANDB_MODE": args.wandb_mode,
        "WANDB_LOG_ALL_SEEDS": False,
        "SAVE_PATH": str(args.save_path),
        "alg": {
            "ALG_NAME": "qmix_rnn",
            "TOTAL_TIMESTEPS": args.total_timesteps,
            "NUM_ENVS": args.num_envs,
            "NUM_STEPS": args.num_steps,
            "BUFFER_SIZE": args.buffer_size,
            "BUFFER_BATCH_SIZE": args.buffer_batch_size,
            "SAMPLE_SEQUENCE_LENGTH": args.sample_sequence_length,
            "HIDDEN_SIZE": args.hidden_size,
            "USE_CNN": bool(args.use_cnn),
            "MIXER_EMBEDDING_DIM": args.mixer_embedding_dim,
            "MIXER_HYPERNET_HIDDEN_DIM": args.mixer_hypernet_hidden_dim,
            "MIXER_INIT_SCALE": args.mixer_init_scale,
            "EPS_START": args.eps_start,
            "EPS_FINISH": args.eps_finish,
            "EPS_DECAY": args.eps_decay,
            "MAX_GRAD_NORM": args.max_grad_norm,
            "TARGET_UPDATE_INTERVAL": args.target_update_interval,
            "TAU": 1.0,
            "NUM_EPOCHS": args.num_epochs,
            "LR": args.lr,
            "LEARNING_STARTS": args.learning_starts,
            "LR_LINEAR_DECAY": bool(args.lr_linear_decay),
            "GAMMA": args.gamma,
            "REW_SHAPING_HORIZON": args.rew_shaping_horizon,
            "ENV_NAME": "overcooked_v2",
            "ENV_KWARGS": {"layout": args.layout},
            "TEST_DURING_TRAINING": bool(args.test_during_training),
            "TEST_INTERVAL": args.test_interval,
            "TEST_NUM_STEPS": args.test_num_steps,
            "TEST_NUM_ENVS": args.test_num_envs,
            "LOG_AGENTS_SEPARATELY": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--buffer_batch_size", type=int, default=128)
    parser.add_argument("--sample_sequence_length", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--use_cnn", type=int, default=0)
    parser.add_argument("--mixer_embedding_dim", type=int, default=32)
    parser.add_argument("--mixer_hypernet_hidden_dim", type=int, default=128)
    parser.add_argument("--mixer_init_scale", type=float, default=0.001)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_finish", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--target_update_interval", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--lr_linear_decay", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--rew_shaping_horizon", type=float, default=2_500_000.0)
    parser.add_argument("--test_during_training", type=int, default=1)
    parser.add_argument("--test_interval", type=float, default=0.05)
    parser.add_argument("--test_num_steps", type=int, default=400)
    parser.add_argument("--test_num_envs", type=int, default=256)
    parser.add_argument("--project", type=str, default="ov2_qlearning_1zsc")
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--split_seed_index", type=int)
    parser.add_argument("--split_seed_count", type=int, default=10)
    parser.add_argument("--output_vmap_index", type=int)
    return parser.parse_args()


def single_split_seed_run(config: dict, args: argparse.Namespace):
    merged = {**config, **config["alg"]}
    env, env_name = ov2_env_from_config(merged)
    rngs = jax.random.split(jax.random.PRNGKey(args.seed), args.split_seed_count)
    rng = rngs[args.split_seed_index]
    train_jit = jax.jit(qmix_rnn.make_train(merged, env))
    outs = jax.block_until_ready(train_jit(rng))

    model_state = outs["runner_state"][0]
    save_dir = args.save_path / env_name
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        merged,
        save_dir / f'qmix_rnn_{env_name}_seed{args.seed}_config.yaml',
    )
    output_idx = args.output_vmap_index
    if output_idx is None:
        output_idx = args.split_seed_index
    save_params(
        model_state.params,
        os.fspath(save_dir / f'qmix_rnn_{env_name}_seed{args.seed}_vmap{output_idx}.safetensors'),
    )
    print(f"Saved QMIX split seed {args.split_seed_index} as vmap{output_idx} to {save_dir}")


def main():
    args = parse_args()
    args.save_path.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    print("Config:\n", OmegaConf.to_yaml(config))
    if not hasattr(qmix_rnn, "_original_env_from_config"):
        qmix_rnn._original_env_from_config = qmix_rnn.env_from_config
    qmix_rnn.env_from_config = ov2_env_from_config
    if args.split_seed_index is not None:
        single_split_seed_run(config, args)
    else:
        qmix_rnn.single_run(config)


if __name__ == "__main__":
    main()
