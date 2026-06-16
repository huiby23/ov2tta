from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from overcooked_v2_experiments.gamma.dataset import GammaTrajectoryDataset
from overcooked_v2_experiments.gamma.models import GammaVAE


def parse_args():
    p = argparse.ArgumentParser(description="Cluster TALENTS VAE latent strategies.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--vae-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-length", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-samples", type=int, default=4096)
    p.add_argument("--num-clusters", type=int, default=0, help="If 0, choose by silhouette in [min_k,max_k].")
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--max-k", type=int, default=8)
    return p.parse_args()


def _kmeans(x, k, rng, num_iters=100):
    n = x.shape[0]
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(num_iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centers[i] = x[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    labels = dist.argmin(axis=1)
    return centers, labels


def _silhouette(x, labels, k):
    # Deterministic O(N^2) silhouette. num_samples is capped by CLI, so this is
    # acceptable and avoids adding sklearn as a hard dependency.
    n = x.shape[0]
    if n < k + 1:
        return -1.0
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1) + 1e-12)
    scores = []
    for i in range(n):
        own = labels == labels[i]
        own_count = own.sum()
        if own_count <= 1:
            a = 0.0
        else:
            a = d[i, own].sum() / max(own_count - 1, 1)
        b = np.inf
        for c in range(k):
            if c == labels[i]:
                continue
            other = labels == c
            if other.any():
                b = min(b, d[i, other].mean())
        if not np.isfinite(b):
            continue
        denom = max(a, b, 1e-8)
        scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else -1.0


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.vae_checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    model = GammaVAE(action_dim=int(cfg["action_dim"]), z_dim=int(cfg["z_dim"]), hidden_dim=int(cfg["hidden_dim"]), activation_name=cfg.get("activation", "relu"))
    data = GammaTrajectoryDataset.load_npz(args.dataset, "train")
    rng = np.random.default_rng(args.seed)
    latents = []
    while sum(x.shape[0] for x in latents) < args.num_samples:
        bs = min(args.batch_size, args.num_samples - sum(x.shape[0] for x in latents))
        obs, actions = data.sample_batch(rng, bs, args.chunk_length)
        onehot = jax.nn.one_hot(jnp.asarray(actions), int(cfg["action_dim"]))
        z = model.apply({"params": ckpt["params"]}, jnp.asarray(obs), onehot, method=GammaVAE.encode_mean)
        latents.append(np.asarray(jax.device_get(z)))
    z = np.concatenate(latents, axis=0)[: args.num_samples]
    search = [args.num_clusters] if args.num_clusters > 0 else list(range(args.min_k, args.max_k + 1))
    best = None
    history = []
    for k in search:
        centers, labels = _kmeans(z, k, rng)
        sil = _silhouette(z, labels, k) if k > 1 else -1.0
        history.append({"k": int(k), "silhouette": sil})
        if best is None or sil > best[0]:
            best = (sil, k, centers, labels)
    _, k, centers, labels = best
    cluster_std = np.zeros_like(centers)
    for c in range(k):
        mask = labels == c
        if mask.any():
            cluster_std[c] = z[mask].std(axis=0) + 1e-3
        else:
            cluster_std[c] = z.std(axis=0) + 1e-3
    payload = {
        "cluster_mean": centers.astype(np.float32),
        "cluster_std": cluster_std.astype(np.float32),
        "assignments": labels.astype(np.int32),
        "latent_samples": z.astype(np.float32),
        "num_clusters": int(k),
        "history": history,
        "vae_checkpoint": str(args.vae_checkpoint),
        "config": cfg,
    }
    with open(out / "talents_clusters.pkl", "wb") as f:
        pickle.dump(payload, f)
    with open(out / "cluster_summary.json", "w") as f:
        json.dump({"num_clusters": int(k), "history": history}, f, indent=2)
    print(f"Saved TALENTS clusters k={k} to {out / 'talents_clusters.pkl'}")


if __name__ == "__main__":
    main()
