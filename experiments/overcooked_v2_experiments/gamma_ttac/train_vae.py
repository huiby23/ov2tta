
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from .dataset import GammaTrajectoryDataset
from .models import GammaVAE, gamma_vae_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAMMA-style VAE on OV2 trajectories.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--chunk-length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--action-dim", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = GammaTrajectoryDataset.load_npz(args.dataset, "train")
    test_data = GammaTrajectoryDataset.load_npz(args.dataset, "test")
    rng_np = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    model = GammaVAE(action_dim=args.action_dim, z_dim=args.z_dim, hidden_dim=args.hidden_dim, activation_name=args.activation)
    dummy_obs, dummy_actions = train_data.sample_batch(rng_np, min(args.batch_size, train_data.num_episodes), args.chunk_length)
    dummy_onehot = jax.nn.one_hot(jnp.asarray(dummy_actions), args.action_dim)
    key, init_key, vae_key = jax.random.split(key, 3)
    params = model.init(init_key, jnp.asarray(dummy_obs), dummy_onehot, vae_key)["params"]
    tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(args.lr, eps=1e-5))
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, obs, actions, key):
        def loss_fn(p):
            return gamma_vae_loss(model, p, obs, actions, key, args.kl_coef)
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def eval_step(params, obs, actions, key):
        _, metrics = gamma_vae_loss(model, params, obs, actions, key, args.kl_coef)
        return metrics

    history = []
    progress = trange(args.epochs, desc="gamma-vae")
    for epoch in progress:
        epoch_metrics = []
        for _ in range(args.steps_per_epoch):
            obs, actions = train_data.sample_batch(rng_np, args.batch_size, args.chunk_length)
            key, step_key = jax.random.split(key)
            params, opt_state, metrics = train_step(params, opt_state, jnp.asarray(obs), jnp.asarray(actions), step_key)
            epoch_metrics.append({k: float(v) for k, v in jax.device_get(metrics).items()})
        eval_obs, eval_actions = test_data.sample_batch(rng_np, min(args.batch_size, test_data.num_episodes), args.chunk_length)
        key, eval_key = jax.random.split(key)
        eval_metrics = {k: float(v) for k, v in jax.device_get(eval_step(params, jnp.asarray(eval_obs), jnp.asarray(eval_actions), eval_key)).items()}
        train_metrics = {k: float(np.mean([m[k] for m in epoch_metrics])) for k in epoch_metrics[0]}
        row = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}, **{f"eval/{k}": v for k, v in eval_metrics.items()}}
        history.append(row)
        progress.set_postfix({"loss": row["train/loss"], "acc": row["eval/acc"], "kl": row["eval/kl"]})

    config = {"action_dim": args.action_dim, "z_dim": args.z_dim, "hidden_dim": args.hidden_dim, "activation": args.activation, "chunk_length": args.chunk_length, "obs_shape": train_data.obs_shape}
    z_obs, z_actions = train_data.sample_batch(rng_np, min(256, args.batch_size * 4), args.chunk_length)
    onehot = jax.nn.one_hot(jnp.asarray(z_actions), args.action_dim)
    z_mean = model.apply({"params": params}, jnp.asarray(z_obs), onehot, method=GammaVAE.encode_mean)
    z_mean_np = np.asarray(jax.device_get(z_mean))
    ckpt = {"config": config, "params": jax.device_get(params), "z_mean": z_mean_np.mean(axis=0), "z_std": z_mean_np.std(axis=0) + 1e-3, "history": history}
    with open(output_dir / "gamma_vae.pkl", "wb") as f:
        pickle.dump(ckpt, f)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved GAMMA VAE checkpoint to {output_dir / 'gamma_vae.pkl'}")


if __name__ == "__main__":
    main()
