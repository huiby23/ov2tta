from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.latent_partner_decoder import (
    apply_latent_partner_decoder,
    categorical_kl,
    categorical_tv,
    load_latent_decoder_npz,
    save_latent_decoder_npz,
)

EPS = 1e-6


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def delayed_history(history):
    history = np.asarray(history, dtype=np.int32)
    return np.concatenate(
        [np.zeros((history.shape[0], 1), dtype=np.int32), history[:, :-1]], axis=1
    )


def strip_lora(params):
    return {
        key: value
        for key, value in params.items()
        if not key.startswith("lora_") and key != "latent_lora_residual_enabled"
    }


def add_lora_params(
    params,
    rng,
    rank,
    coeff_hidden_dim,
    action_dim,
    gate_bias,
    residual_clip,
    lora_alpha,
    no_gate,
):
    if "attn_frame_w" not in params or "vae_mu_b" not in params:
        raise ValueError("LoRA residual v1 expects a query-attn VAE decoder checkpoint.")
    frame_dim = int(params["attn_frame_w"].shape[1])
    z_dim = int(params["vae_mu_b"].shape[0])
    keys = jax.random.split(rng, 5)
    scale = 0.02
    params = dict(params)
    params.update(
        {
            "latent_lora_residual_enabled": jnp.asarray(1.0, dtype=jnp.float32),
            "lora_b": scale * jax.random.normal(keys[0], (frame_dim, rank)),
            "lora_a": jnp.zeros((rank, action_dim), dtype=jnp.float32),
            "lora_bias": jnp.zeros((action_dim,), dtype=jnp.float32),
            "lora_coeff_w1": scale * jax.random.normal(keys[1], (z_dim, coeff_hidden_dim)),
            "lora_coeff_b1": jnp.zeros((coeff_hidden_dim,), dtype=jnp.float32),
            "lora_coeff_w2": scale * jax.random.normal(keys[2], (coeff_hidden_dim, rank)),
            "lora_coeff_b2": jnp.zeros((rank,), dtype=jnp.float32),
            "lora_gate_w": jnp.zeros((frame_dim + z_dim, 1), dtype=jnp.float32),
            "lora_gate_b": jnp.full((1,), float(gate_bias), dtype=jnp.float32),
            "lora_residual_clip": jnp.asarray(float(residual_clip), dtype=jnp.float32),
            "lora_residual_scale": jnp.asarray(1.0, dtype=jnp.float32),
            "lora_alpha": jnp.asarray(float(lora_alpha), dtype=jnp.float32),
        }
    )
    if no_gate:
        params["lora_no_gate"] = jnp.asarray(1.0, dtype=jnp.float32)
    return params


def split_by_episode(episode_ids, val_fraction, seed):
    rng = np.random.default_rng(seed)
    unique_eps = np.unique(episode_ids)
    rng.shuffle(unique_eps)
    n_val = max(1, int(round(len(unique_eps) * float(val_fraction))))
    val_eps = set(unique_eps[:n_val].tolist())
    val_mask = np.isin(episode_ids, list(val_eps))
    return np.where(~val_mask)[0].astype(np.int32), np.where(val_mask)[0].astype(np.int32)


def batch_from_arrays(arrays, idx):
    return {
        "query_obs": arrays["query_obs"][idx].astype(np.float32),
        "partner_obs_history": arrays["partner_obs_history"][idx].astype(np.float32),
        "partner_action_history": arrays["partner_action_history"][idx].astype(np.int32),
        "target_partner_probs": normalize_probs(arrays["target_partner_probs"][idx]),
    }


def make_train_step(action_dim, args, base_params):
    opt = optax.adam(args.lr)

    @jax.jit
    def init_opt(params):
        return opt.init(params)

    @jax.jit
    def train_step(params, opt_state, batch, rng):
        del rng
        target = batch["target_partner_probs"]

        def loss_fn(p):
            logits, aux = apply_latent_partner_decoder(
                p,
                batch["query_obs"],
                batch["partner_obs_history"],
                batch["partner_action_history"],
                action_dim,
                deterministic=True,
                return_aux=True,
            )
            base_logits = apply_latent_partner_decoder(
                base_params,
                batch["query_obs"],
                batch["partner_obs_history"],
                batch["partner_action_history"],
                action_dim,
                deterministic=True,
                return_aux=False,
            )
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            ce = -jnp.sum(jax.lax.stop_gradient(target) * log_probs, axis=-1).mean()
            kl = categorical_kl(target, logits).mean()
            tv = categorical_tv(target, logits).mean()
            base_kl = categorical_kl(target, base_logits).mean()
            anchor_kl = jnp.sum(
                jax.nn.softmax(base_logits, axis=-1)
                * (jax.nn.log_softmax(base_logits, axis=-1) - log_probs),
                axis=-1,
            ).mean()
            residual_l2 = jnp.mean(jnp.square(aux["lora_residual_logits"]))
            gate_mean = jnp.mean(aux["lora_gate"])
            residual_tv = categorical_tv(jax.nn.softmax(base_logits, axis=-1), logits).mean()
            acc = (
                jnp.argmax(logits, axis=-1) == jnp.argmax(target, axis=-1)
            ).mean()
            top2 = jnp.argsort(logits, axis=-1)[:, -2:]
            top2_acc = jnp.any(top2 == jnp.argmax(target, axis=-1)[:, None], axis=-1).mean()
            loss = (
                ce
                + jnp.asarray(args.anchor_kl_coef, dtype=jnp.float32) * anchor_kl
                + jnp.asarray(args.residual_l2_coef, dtype=jnp.float32) * residual_l2
            )
            metrics = {
                "loss": loss,
                "ce": ce,
                "kl": kl,
                "tv": tv,
                "base_kl": base_kl,
                "anchor_kl": anchor_kl,
                "residual_l2": residual_l2,
                "residual_tv": residual_tv,
                "gate_mean": gate_mean,
                "acc": acc,
                "top2_acc": top2_acc,
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        del loss
        grads = {
            key: (value if key.startswith("lora_") else jnp.zeros_like(value))
            for key, value in grads.items()
        }
        if args.freeze_gate and "lora_gate_w" in grads:
            grads["lora_gate_w"] = jnp.zeros_like(grads["lora_gate_w"])
            grads["lora_gate_b"] = jnp.zeros_like(grads["lora_gate_b"])
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    return init_opt, train_step


def compute_base_kl_scores(base_params, arrays, action_dim, batch_size):
    scores = []
    n_rows = int(arrays["query_obs"].shape[0])
    for start in range(0, n_rows, batch_size):
        end = min(n_rows, start + batch_size)
        logits = apply_latent_partner_decoder(
            base_params,
            jnp.asarray(arrays["query_obs"][start:end], dtype=jnp.float32),
            jnp.asarray(arrays["partner_obs_history"][start:end], dtype=jnp.float32),
            jnp.asarray(arrays["partner_action_history"][start:end], dtype=jnp.int32),
            action_dim,
            deterministic=True,
            return_aux=False,
        )
        target = jnp.asarray(arrays["target_partner_probs"][start:end], dtype=jnp.float32)
        scores.append(np.asarray(categorical_kl(target, logits), dtype=np.float32))
    return np.concatenate(scores, axis=0)


def eval_history(params, base_params, arrays, idx, action_dim, args, history_mode):
    idx = np.asarray(idx, dtype=np.int32)
    if args.eval_max_rows and idx.shape[0] > args.eval_max_rows:
        rng = np.random.default_rng(args.seed + 812)
        idx = rng.choice(idx, size=int(args.eval_max_rows), replace=False).astype(np.int32)
    totals = {
        "count": 0,
        "kl": 0.0,
        "tv": 0.0,
        "acc": 0.0,
        "top2_acc": 0.0,
        "base_kl": 0.0,
        "base_tv": 0.0,
        "residual_tv": 0.0,
        "gate_mean": 0.0,
    }
    rng = np.random.default_rng(args.seed + 991)
    for start in range(0, idx.shape[0], args.eval_batch_size):
        part_idx = idx[start : start + args.eval_batch_size]
        batch = batch_from_arrays(arrays, part_idx)
        obs_hist = batch["partner_obs_history"]
        act_hist = batch["partner_action_history"]
        if history_mode == "wrong":
            if "ego_action_history" in arrays:
                act_hist = arrays["ego_action_history"][part_idx].astype(np.int32)
            else:
                act_hist = np.zeros_like(act_hist, dtype=np.int32)
        elif history_mode == "delayed":
            act_hist = delayed_history(act_hist)
        elif history_mode == "random":
            act_hist = rng.integers(0, action_dim, size=act_hist.shape, dtype=np.int32)
        elif history_mode == "no_history":
            obs_hist = np.zeros_like(obs_hist, dtype=np.float32)
            act_hist = np.zeros_like(act_hist, dtype=np.int32)
        query_obs = jnp.asarray(batch["query_obs"])
        obs_hist_j = jnp.asarray(obs_hist)
        act_hist_j = jnp.asarray(act_hist)
        target = jnp.asarray(batch["target_partner_probs"])
        logits, aux = apply_latent_partner_decoder(
            params,
            query_obs,
            obs_hist_j,
            act_hist_j,
            action_dim,
            deterministic=True,
            return_aux=True,
        )
        base_logits = apply_latent_partner_decoder(
            base_params,
            query_obs,
            obs_hist_j,
            act_hist_j,
            action_dim,
            deterministic=True,
            return_aux=False,
        )
        probs = jax.nn.softmax(logits, axis=-1)
        top2 = jnp.argsort(logits, axis=-1)[:, -2:]
        target_arg = jnp.argmax(target, axis=-1)
        count = int(part_idx.shape[0])
        metrics = {
            "kl": float(categorical_kl(target, logits).mean()),
            "tv": float(categorical_tv(target, logits).mean()),
            "acc": float((jnp.argmax(logits, axis=-1) == target_arg).mean()),
            "top2_acc": float(jnp.any(top2 == target_arg[:, None], axis=-1).mean()),
            "base_kl": float(categorical_kl(target, base_logits).mean()),
            "base_tv": float(categorical_tv(target, base_logits).mean()),
            "residual_tv": float(categorical_tv(jax.nn.softmax(base_logits, axis=-1), logits).mean()),
            "gate_mean": float(jnp.mean(aux["lora_gate"])),
        }
        for key, value in metrics.items():
            totals[key] += value * count
        totals["count"] += count
    return {
        key: (value / max(totals["count"], 1) if key != "count" else totals["count"])
        for key, value in totals.items()
    }


def write_summary(path, args, metrics_by_mode):
    lines = [
        "# LoRA Residual Latent Partner Decoder",
        "",
        f"- dataset: `{args.dataset}`",
        f"- init_decoder_path: `{args.init_decoder_path}`",
        f"- rank: `{args.rank}`",
        f"- lora_alpha: `{args.lora_alpha}`",
        f"- no_gate: `{args.no_gate}`",
        f"- coeff_hidden_dim: `{args.coeff_hidden_dim}`",
        f"- gate_bias: `{args.gate_bias}`",
        f"- freeze_gate: `{args.freeze_gate}`",
        f"- hard_sample_fraction: `{args.hard_sample_fraction}`",
        f"- hard_top_fraction: `{args.hard_top_fraction}`",
        f"- anchor_kl_coef: `{args.anchor_kl_coef}`",
        f"- residual_l2_coef: `{args.residual_l2_coef}`",
        f"- residual_clip: `{args.residual_clip}`",
        f"- steps: `{args.steps}`",
        "",
        "| history | count | KL | base KL | TV | acc | top2 | residual TV | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in metrics_by_mode.items():
        lines.append(
            "| {mode} | {count} | {kl:.4f} | {base_kl:.4f} | {tv:.4f} | {acc:.4f} | {top2_acc:.4f} | {residual_tv:.4f} | {gate_mean:.4f} |".format(
                mode=mode, **metrics
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init_decoder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--coeff_hidden_dim", type=int, default=64)
    parser.add_argument("--gate_bias", type=float, default=-2.0)
    parser.add_argument("--residual_clip", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anchor_kl_coef", type=float, default=0.05)
    parser.add_argument("--residual_l2_coef", type=float, default=1e-3)
    parser.add_argument("--freeze_gate", action="store_true")
    parser.add_argument("--hard_sample_fraction", type=float, default=0.0)
    parser.add_argument("--hard_top_fraction", type=float, default=0.25)
    parser.add_argument("--hard_score_batch_size", type=int, default=2048)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--eval_max_rows", type=int, default=60000)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files if k != "pair_label"}
    required = [
        "query_obs",
        "partner_obs_history",
        "partner_action_history",
        "target_partner_probs",
        "episode_id",
    ]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"dataset missing fields: {missing}")
    arrays["target_partner_probs"] = normalize_probs(arrays["target_partner_probs"])

    train_idx, val_idx = split_by_episode(arrays["episode_id"], args.val_fraction, args.seed)
    rng_np = np.random.default_rng(args.seed)
    base_params = load_latent_decoder_npz(Path(args.init_decoder_path))
    if args.lora_alpha is None:
        args.lora_alpha = float(args.rank)
    params = add_lora_params(
        base_params,
        jax.random.PRNGKey(args.seed + 171),
        int(args.rank),
        int(args.coeff_hidden_dim),
        int(args.action_dim),
        float(args.gate_bias),
        float(args.residual_clip),
        float(args.lora_alpha),
        bool(args.no_gate),
    )
    base_params = strip_lora(params)
    hard_train_idx = np.asarray([], dtype=np.int32)
    hard_sample_fraction = min(max(float(args.hard_sample_fraction), 0.0), 1.0)
    if hard_sample_fraction > 0.0:
        print("[lora_decoder] computing frozen old-VAE KL scores for hard sampling", flush=True)
        base_kl_scores = compute_base_kl_scores(
            base_params,
            arrays,
            args.action_dim,
            args.hard_score_batch_size,
        )
        train_scores = base_kl_scores[train_idx]
        hard_top_fraction = min(max(float(args.hard_top_fraction), EPS), 1.0)
        cutoff = np.quantile(train_scores, 1.0 - hard_top_fraction)
        hard_train_idx = train_idx[train_scores >= cutoff]
        if hard_train_idx.shape[0] == 0:
            hard_sample_fraction = 0.0
        print(
            f"[lora_decoder] hard rows={hard_train_idx.shape[0]} cutoff={float(cutoff):.4f}",
            flush=True,
        )
    init_opt, train_step = make_train_step(args.action_dim, args, base_params)
    opt_state = init_opt(params)
    curve = []
    rng = jax.random.PRNGKey(args.seed + 909)
    for step in range(1, args.steps + 1):
        if hard_sample_fraction > 0.0:
            hard_count = int(round(args.batch_size * hard_sample_fraction))
            normal_count = args.batch_size - hard_count
            hard_part = rng_np.choice(
                hard_train_idx,
                size=hard_count,
                replace=hard_train_idx.shape[0] < hard_count,
            ).astype(np.int32)
            normal_part = rng_np.choice(
                train_idx,
                size=normal_count,
                replace=train_idx.shape[0] < normal_count,
            ).astype(np.int32)
            batch_idx = np.concatenate([hard_part, normal_part], axis=0)
            rng_np.shuffle(batch_idx)
        else:
            batch_idx = rng_np.choice(
                train_idx,
                size=args.batch_size,
                replace=train_idx.shape[0] < args.batch_size,
            ).astype(np.int32)
        rng, rng_step = jax.random.split(rng)
        batch = {
            key: jnp.asarray(value)
            for key, value in batch_from_arrays(arrays, batch_idx).items()
        }
        params, opt_state, metrics = train_step(params, opt_state, batch, rng_step)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step}
            row.update({key: float(value) for key, value in metrics.items()})
            curve.append(row)
            print(f"[lora_decoder] {row}", flush=True)

    metrics_by_mode = {
        mode: eval_history(
            params,
            base_params,
            arrays,
            val_idx,
            args.action_dim,
            args,
            history_mode=mode,
        )
        for mode in ("true", "wrong", "delayed", "random", "no_history")
    }
    save_latent_decoder_npz(out / "latent_partner_decoder.npz", params)
    write_csv(out / "training_curve.csv", curve)
    write_csv(
        out / "validation_metrics.csv",
        [{"history": mode, **metrics} for mode, metrics in metrics_by_mode.items()],
    )
    write_summary(out / "lora_residual_summary.md", args, metrics_by_mode)
    print(f"Wrote {out / 'latent_partner_decoder.npz'}")
    print(metrics_by_mode)


if __name__ == "__main__":
    main()
