#!/usr/bin/env python3
"""OV2 adapter for JaxMARL SHAQ training with all-vmap checkpoint saving."""

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

import shaq  # noqa: E402


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
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "BUFFER_SIZE": args.buffer_size,
        "BUFFER_BATCH_SIZE": args.buffer_batch_size,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "AGENT_HIDDEN_DIM": args.agent_hidden_dim,
        "AGENT_INIT_SCALE": args.agent_init_scale,
        "PARAMETERS_SHARING": True,
        "EPSILON_START": args.epsilon_start,
        "EPSILON_FINISH": args.epsilon_finish,
        "EPSILON_ANNEAL_TIME": args.epsilon_anneal_time,
        "MIXER_EMBEDDING_DIM": args.mixer_embedding_dim,
        "MIXER_HYPERNET_HIDDEN_DIM": args.mixer_hypernet_hidden_dim,
        "MIXER_INIT_SCALE": args.mixer_init_scale,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "TARGET_UPDATE_INTERVAL": args.target_update_interval,
        "LR": args.lr,
        "LR_LINEAR_DECAY": bool(args.lr_linear_decay),
        "EPS_ADAM": args.eps_adam,
        "WEIGHT_DECAY_ADAM": args.weight_decay_adam,
        "TD_LAMBDA_LOSS": bool(args.td_lambda_loss),
        "TD_LAMBDA": args.td_lambda,
        "GAMMA": args.gamma,
        "REW_SHAPING_HORIZON": args.rew_shaping_horizon,
        "VERBOSE": False,
        "WANDB_ONLINE_REPORT": False,
        "NUM_TEST_EPISODES": args.num_test_episodes,
        "TEST_NUM_STEPS": args.test_num_steps,
        "TEST_INTERVAL": args.test_interval,
        "SAMPLE_SIZE": args.sample_size,
        "MANUAL_ALPHA_ESTIMATES": None,
        "LR_ALPHA": args.lr_alpha,
        "ALG_NAME": "shaq_ps",
        "ENV_NAME": "overcooked_v2",
        "ENV_KWARGS": {"layout": args.layout},
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
    parser.add_argument("--agent_hidden_dim", type=int, default=64)
    parser.add_argument("--agent_init_scale", type=float, default=2.0)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_finish", type=float, default=0.05)
    parser.add_argument("--epsilon_anneal_time", type=int, default=100_000)
    parser.add_argument("--mixer_embedding_dim", type=int, default=32)
    parser.add_argument("--mixer_hypernet_hidden_dim", type=int, default=64)
    parser.add_argument("--mixer_init_scale", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--target_update_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--lr_linear_decay", type=int, default=1)
    parser.add_argument("--eps_adam", type=float, default=0.001)
    parser.add_argument("--weight_decay_adam", type=float, default=1e-5)
    parser.add_argument("--td_lambda_loss", type=int, default=1)
    parser.add_argument("--td_lambda", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--rew_shaping_horizon", type=float, default=2_500_000.0)
    parser.add_argument("--num_test_episodes", type=int, default=256)
    parser.add_argument("--test_num_steps", type=int, default=400)
    parser.add_argument("--test_interval", type=int, default=50_000)
    parser.add_argument("--sample_size", type=int, default=5)
    parser.add_argument("--lr_alpha", type=float, default=0.001)
    parser.add_argument("--project", type=str, default="ov2_qlearning_1zsc")
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--split_seed_index", type=int)
    parser.add_argument("--split_seed_count", type=int, default=10)
    parser.add_argument("--output_vmap_index", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    args.save_path.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    print("Config:\n", OmegaConf.to_yaml(config))

    env = make("overcooked_v2", layout=args.layout)
    env = LogWrapper(env, replace_info=True)
    config["NUM_STEPS"] = config.get("NUM_STEPS", env.max_steps)
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]

    env_name = f'{config["ENV_NAME"]}_{args.layout}'
    save_dir = args.save_path / env_name
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config,
        save_dir / f'shaq_ps_{env_name}_seed{config["SEED"]}_config.yaml',
    )

    if args.split_seed_index is not None:
        rngs = jax.random.split(jax.random.PRNGKey(config["SEED"]), args.split_seed_count)
        rng = rngs[args.split_seed_index]
        train_jit = jax.jit(shaq.make_train(config, env))
        outs = jax.block_until_ready(train_jit(rng))
        model_state = outs["runner_state"][0]
        output_idx = args.output_vmap_index
        if output_idx is None:
            output_idx = args.split_seed_index
        save_params(
            model_state.params,
            os.fspath(save_dir / f'shaq_ps_{env_name}_seed{config["SEED"]}_vmap{output_idx}.safetensors'),
        )
        print(f"Saved SHAQ split seed {args.split_seed_index} as vmap{output_idx} to {save_dir}")
    else:
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(shaq.make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

        model_state = outs["runner_state"][0]
        for i, _rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_params(
                params,
                os.fspath(save_dir / f'shaq_ps_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors'),
            )
        print(f"Saved {len(rngs)} SHAQ agent checkpoints to {save_dir}")


if __name__ == "__main__":
    main()
