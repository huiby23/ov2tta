#!/usr/bin/env python3
"""Train one OV2 partner checkpoint for classic MARL baselines.

This wrapper runs one split seed at a time and saves it with a chosen vmap
index. It keeps the downstream evaluator contract used by the existing
IQL/VDN/PQN partner pools while avoiding a heavy 10-seed vmap for larger
centralized-critic or mixer models.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import jax
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "JaxMARL"))
sys.path.insert(0, str(ROOT / "JaxMARL" / "baselines" / "QLearning"))

from jaxmarl.wrappers.baselines import save_params  # noqa: E402

import coma_cnn_overcooked  # noqa: E402
import pqn_qplex_cnn_overcooked  # noqa: E402
import pqn_soft_cnn_overcooked  # noqa: E402
import pqn_wqmix_cnn_overcooked  # noqa: E402
import ppo_coma_cnn_overcooked  # noqa: E402
import qplex_cnn_overcooked  # noqa: E402
import wqmix_cnn_overcooked  # noqa: E402


METHODS = {
    "pqn_wqmix": {
        "module": pqn_wqmix_cnn_overcooked,
        "subdir": "pqn_wqmix",
        "prefix": "pqn_wqmix_cnn",
        "kind": "pqn",
    },
    "pqn_soft": {
        "module": pqn_soft_cnn_overcooked,
        "subdir": "pqn_soft",
        "prefix": "pqn_soft_cnn",
        "kind": "pqn",
    },
    "pqn_qplex": {
        "module": pqn_qplex_cnn_overcooked,
        "subdir": "pqn_qplex",
        "prefix": "pqn_qplex_cnn",
        "kind": "pqn",
        "save_agent_params": True,
    },
    "coma": {
        "module": coma_cnn_overcooked,
        "subdir": "coma",
        "prefix": "coma_cnn",
        "kind": "actor_critic",
    },
    "ppo_coma": {
        "module": ppo_coma_cnn_overcooked,
        "subdir": "ppo_coma",
        "prefix": "ppo_coma_cnn",
        "kind": "actor_critic",
    },
    "qplex": {
        "module": qplex_cnn_overcooked,
        "subdir": "qplex",
        "prefix": "qplex_cnn",
        "kind": "q_mixer",
    },
    "wqmix": {
        "module": wqmix_cnn_overcooked,
        "subdir": "wqmix",
        "prefix": "wqmix_cnn",
        "kind": "q_mixer",
    },
}



def pqn_config(args: argparse.Namespace) -> dict:
    return {
        "NUM_SEEDS": 1,
        "SEED": args.seed,
        "HYP_TUNE": False,
        "ENTITY": "",
        "PROJECT": args.project,
        "WANDB_MODE": args.wandb_mode,
        "WANDB_LOG_ALL_SEEDS": False,
        "SAVE_PATH": str(args.save_path),
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "HIDDEN_SIZE": args.pqn_hidden_size,
        "NUM_LAYERS": args.pqn_num_layers,
        "NORM_TYPE": args.pqn_norm_type,
        "NORM_INPUT": bool(args.pqn_norm_input),
        "EPS_START": args.eps_start,
        "EPS_FINISH": args.pqn_eps_finish,
        "EPS_DECAY": args.pqn_eps_decay,
        "MAX_GRAD_NORM": args.pqn_max_grad_norm,
        "NUM_MINIBATCHES": args.num_minibatches,
        "NUM_EPOCHS": args.num_epochs,
        "LR": args.lr,
        "LR_LINEAR_DECAY": bool(args.lr_linear_decay),
        "LAMBDA": args.pqn_lambda,
        "GAMMA": args.gamma,
        "ENV_NAME": "overcooked_v2",
        "ENV_KWARGS": {"layout": args.layout},
        "REW_SHAPING_HORIZON": args.rew_shaping_horizon,
        "TEST_DURING_TRAINING": bool(args.test_during_training),
        "TEST_INTERVAL": args.test_interval,
        "TEST_NUM_STEPS": args.test_num_steps,
        "TEST_NUM_ENVS": args.test_num_envs,
        "REW_SCALE": args.rew_scale,
        "WQMIX_ALPHA": args.wqmix_alpha,
        "SOFT_Q_TEMPERATURE": args.soft_q_temperature,
        "SOFT_Q_CENTER": bool(args.soft_q_center),
        "MIXER_EMBEDDING_DIM": args.mixer_embedding_dim,
        "MIXER_HYPERNET_HIDDEN_DIM": args.mixer_hypernet_hidden_dim,
        "MIXER_INIT_SCALE": args.mixer_init_scale,
        "MIXER_RESIDUAL_SUM": bool(args.mixer_residual_sum),
        "QPLEX_VDN_AUX_COEF": args.qplex_vdn_aux_coef,
    }

def q_mixer_config(args: argparse.Namespace) -> dict:
    return {
        "NUM_SEEDS": 1,
        "SEED": args.seed,
        "HYP_TUNE": False,
        "ENTITY": "",
        "PROJECT": args.project,
        "WANDB_MODE": args.wandb_mode,
        "WANDB_LOG_ALL_SEEDS": False,
        "SAVE_PATH": str(args.save_path),
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "BUFFER_SIZE": args.buffer_size,
        "BUFFER_BATCH_SIZE": args.buffer_batch_size,
        "HIDDEN_SIZE": args.hidden_size,
        "EPS_START": args.eps_start,
        "EPS_FINISH": args.eps_finish,
        "EPS_DECAY": args.eps_decay,
        "LEARNING_STARTS": args.learning_starts,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "TARGET_UPDATE_INTERVAL": args.target_update_interval,
        "NUM_EPOCHS": args.num_epochs,
        "LR": args.lr,
        "LR_LINEAR_DECAY": bool(args.lr_linear_decay),
        "GAMMA": args.gamma,
        "TAU": 1.0,
        "LOSS_TYPE": "vdn",
        "ENV_NAME": "overcooked_v2",
        "ENV_KWARGS": {"layout": args.layout},
        "REW_SHAPING_HORIZON": args.rew_shaping_horizon,
        "TEST_DURING_TRAINING": bool(args.test_during_training),
        "TEST_INTERVAL": args.test_interval,
        "TEST_NUM_STEPS": args.test_num_steps,
        "TEST_NUM_ENVS": args.test_num_envs,
        "MIXER_EMBEDDING_DIM": args.mixer_embedding_dim,
        "MIXER_HYPERNET_HIDDEN_DIM": args.mixer_hypernet_hidden_dim,
        "MIXER_INIT_SCALE": args.mixer_init_scale,
        "MIXER_RESIDUAL_SUM": bool(args.mixer_residual_sum),
        "WQMIX_ALPHA": args.wqmix_alpha,
    }


def actor_critic_config(args: argparse.Namespace) -> dict:
    return {
        "NUM_SEEDS": 1,
        "SEED": args.seed,
        "HYP_TUNE": False,
        "ENTITY": "",
        "PROJECT": args.project,
        "WANDB_MODE": args.wandb_mode,
        "WANDB_LOG_ALL_SEEDS": False,
        "SAVE_PATH": str(args.save_path),
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.coma_num_steps,
        "LR": args.coma_lr,
        "ANNEAL_LR": bool(args.lr_linear_decay),
        "MAX_GRAD_NORM": args.coma_max_grad_norm,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "VF_COEF": 0.5,
        "ENT_COEF": args.ent_coef,
        "COMA_CRITIC_COEF": args.coma_critic_coef,
        "COMA_CRITIC_HIDDEN_SIZE": args.coma_critic_hidden_size,
        "CLIP_EPS": args.clip_eps,
        "UPDATE_EPOCHS": args.num_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "ACTIVATION": "relu",
        "REW_SHAPING_HORIZON": args.rew_shaping_horizon,
        "NORMALIZE_ADVANTAGES": True,
        "ENV_NAME": "overcooked_v2",
        "ENV_KWARGS": {"layout": args.layout},
    }


def build_config(method: str, args: argparse.Namespace) -> dict:
    kind = METHODS[method]["kind"]
    if kind == "pqn":
        return pqn_config(args)
    if kind == "q_mixer":
        return q_mixer_config(args)
    return actor_critic_config(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed_index", type=int, required=True)
    parser.add_argument("--split_seed_count", type=int, default=10)
    parser.add_argument("--output_vmap_index", type=int)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--buffer_batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_finish", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.1)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_update_interval", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=16)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--pqn_hidden_size", type=int, default=512)
    parser.add_argument("--pqn_num_layers", type=int, default=2)
    parser.add_argument("--pqn_norm_type", default="layer_norm")
    parser.add_argument("--pqn_norm_input", type=int, default=0)
    parser.add_argument("--pqn_eps_finish", type=float, default=0.2)
    parser.add_argument("--pqn_eps_decay", type=float, default=0.2)
    parser.add_argument("--pqn_max_grad_norm", type=float, default=10.0)
    parser.add_argument("--pqn_lambda", type=float, default=0.5)
    parser.add_argument("--rew_scale", type=float, default=1.0)
    parser.add_argument("--soft_q_temperature", type=float, default=0.25)
    parser.add_argument("--soft_q_center", type=int, default=1)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--lr_linear_decay", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--rew_shaping_horizon", type=float, default=2_500_000.0)
    parser.add_argument("--test_during_training", type=int, default=0)
    parser.add_argument("--test_interval", type=float, default=0.05)
    parser.add_argument("--test_num_steps", type=int, default=400)
    parser.add_argument("--test_num_envs", type=int, default=128)
    parser.add_argument("--mixer_embedding_dim", type=int, default=32)
    parser.add_argument("--mixer_hypernet_hidden_dim", type=int, default=128)
    parser.add_argument("--mixer_init_scale", type=float, default=0.001)
    parser.add_argument("--mixer_residual_sum", type=int, default=1)
    parser.add_argument("--wqmix_alpha", type=float, default=0.1)
    parser.add_argument("--qplex_vdn_aux_coef", type=float, default=1.0)
    parser.add_argument("--coma_num_steps", type=int, default=128)
    parser.add_argument("--coma_lr", type=float, default=2.5e-4)
    parser.add_argument("--coma_max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--coma_critic_coef", type=float, default=0.5)
    parser.add_argument("--coma_critic_hidden_size", type=int, default=128)
    parser.add_argument("--project", default="ov2_classic_marl_partners")
    parser.add_argument("--wandb_mode", default="disabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = METHODS[args.method]
    module = spec["module"]
    config = build_config(args.method, args)
    env, env_name = module.env_from_config(copy.deepcopy(config))
    rngs = jax.random.split(jax.random.PRNGKey(args.seed), args.split_seed_count)
    rng = rngs[args.split_seed_index]

    train_jit = jax.jit(module.make_train(config, env))
    outs = jax.block_until_ready(train_jit(rng))
    model_state = outs["runner_state"][0]

    save_dir = args.save_path / spec["subdir"] / env_name
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config,
        save_dir / f'{spec["prefix"]}_{env_name}_seed{args.seed}_config.yaml',
    )
    output_idx = args.output_vmap_index
    if output_idx is None:
        output_idx = args.split_seed_index
    params_to_save = model_state.params
    if spec.get("save_agent_params", False):
        params_to_save = model_state.params["agent"]
    save_params(
        params_to_save,
        os.fspath(
            save_dir
            / f'{spec["prefix"]}_{env_name}_seed{args.seed}_vmap{output_idx}.safetensors'
        ),
    )
    print(
        f"Saved {args.method} split_seed={args.split_seed_index} "
        f"as vmap{output_idx} under {save_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
