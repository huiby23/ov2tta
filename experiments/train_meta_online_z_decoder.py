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
    apply_latent_partner_decoder_with_z,
    categorical_kl,
    categorical_tv,
    encode_partner_online_z,
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


def make_split(n_rows, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows, dtype=np.int32)
    rng.shuffle(idx)
    n_val = max(1, int(round(n_rows * float(val_fraction))))
    return idx[n_val:], idx[:n_val]


def batch_from_arrays(arrays, idx):
    batch = {
        "query_obs": arrays["query_obs"][idx].astype(np.float32),
        "support_obs": arrays["partner_obs_history"][idx].astype(np.float32),
        "support_actions": arrays["partner_action_history"][idx].astype(np.int32),
        "target_probs": arrays["target_partner_probs"][idx].astype(np.float32),
    }
    if "ego_action_history" in arrays:
        batch["wrong_actions"] = arrays["ego_action_history"][idx].astype(np.int32)
    else:
        batch["wrong_actions"] = np.zeros_like(batch["support_actions"], dtype=np.int32)
    return batch


def support_valid_mask(support_obs):
    obs_abs = jnp.sum(jnp.abs(support_obs), axis=tuple(range(2, support_obs.ndim)))
    valid = obs_abs > jnp.asarray(EPS, dtype=jnp.float32)
    return jnp.where(jnp.any(valid, axis=-1, keepdims=True), valid, jnp.ones_like(valid))


def support_ce(params, support_obs, support_actions, z, action_dim):
    batch, hist_len = support_actions.shape[:2]
    z_rep = jnp.broadcast_to(z[:, None, :], (batch, hist_len, z.shape[-1]))
    logits = apply_latent_partner_decoder_with_z(
        params,
        jnp.reshape(support_obs, (batch * hist_len,) + support_obs.shape[2:]),
        jnp.reshape(z_rep, (batch * hist_len, z.shape[-1])),
        action_dim,
    )
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = jnp.reshape(support_actions.astype(jnp.int32), (batch * hist_len,))
    action_log_probs = jnp.take_along_axis(
        log_probs, actions[:, None], axis=-1
    )[:, 0]
    valid = jnp.reshape(support_valid_mask(support_obs).astype(jnp.float32), (-1,))
    return -jnp.sum(action_log_probs * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def adapt_z(
    params,
    support_obs,
    support_actions,
    action_dim,
    inner_steps,
    inner_lr,
    prior_coef,
    z_clip,
    first_order,
    init_z_mode,
):
    z0 = encode_partner_online_z(params, support_obs, support_actions, action_dim)
    if init_z_mode == "zero":
        z0 = jnp.zeros_like(z0)
    elif init_z_mode == "no_history":
        z0 = encode_partner_online_z(
            params,
            jnp.zeros_like(support_obs),
            jnp.zeros_like(support_actions),
            action_dim,
        )
    z0_anchor = jax.lax.stop_gradient(z0)

    def inner_loss(z):
        ce = support_ce(params, support_obs, support_actions, z, action_dim)
        prior = jnp.mean(jnp.square(z - z0_anchor))
        return ce + jnp.asarray(prior_coef, dtype=jnp.float32) * prior

    def one_step(z, _):
        grad = jax.grad(inner_loss)(z)
        next_z = z - jnp.asarray(inner_lr, dtype=jnp.float32) * grad
        next_z = jnp.clip(next_z, -jnp.asarray(z_clip, dtype=jnp.float32), jnp.asarray(z_clip, dtype=jnp.float32))
        if first_order:
            next_z = jax.lax.stop_gradient(next_z)
        return next_z, None

    if inner_steps <= 0:
        return z0, z0
    zk, _ = jax.lax.scan(one_step, z0, None, length=inner_steps)
    return z0, zk


def make_train_step(action_dim, args):
    optimizer = optax.adam(args.lr)

    @jax.jit
    def init_opt(params):
        return optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        target = batch["target_probs"]

        def loss_fn(p):
            z0, zk = adapt_z(
                p,
                batch["support_obs"],
                batch["support_actions"],
                action_dim,
                args.inner_steps,
                args.inner_lr,
                args.inner_prior_coef,
                args.inner_z_clip,
                args.first_order,
                args.init_z_mode,
            )
            logits_k = apply_latent_partner_decoder_with_z(
                p, batch["query_obs"], zk, action_dim
            )
            outer_kl = categorical_kl(target, logits_k).mean()
            outer_tv = categorical_tv(target, logits_k).mean()
            pred_acc = (
                jnp.argmax(logits_k, axis=-1) == jnp.argmax(target, axis=-1)
            ).mean()

            logits_0 = apply_latent_partner_decoder_with_z(
                p, batch["query_obs"], z0, action_dim
            )
            pre_kl = categorical_kl(target, logits_0).mean()
            support_loss = support_ce(
                p,
                batch["support_obs"],
                batch["support_actions"],
                zk,
                action_dim,
            )
            loss = (
                outer_kl
                + jnp.asarray(args.pre_update_kl_coef, dtype=jnp.float32) * pre_kl
                + jnp.asarray(args.support_ce_coef, dtype=jnp.float32) * support_loss
            )
            metrics = {
                "loss": loss,
                "outer_kl": outer_kl,
                "outer_tv": outer_tv,
                "outer_acc": pred_acc,
                "pre_kl": pre_kl,
                "support_ce": support_loss,
                "z0_norm": jnp.mean(jnp.linalg.norm(z0, axis=-1)),
                "zk_norm": jnp.mean(jnp.linalg.norm(zk, axis=-1)),
                "z_delta": jnp.mean(jnp.linalg.norm(zk - z0, axis=-1)),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        del loss
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    return init_opt, train_step


def eval_metrics(params, arrays, idx, action_dim, args, max_rows, history_mode="true"):
    idx = np.asarray(idx, dtype=np.int32)
    if max_rows and idx.shape[0] > max_rows:
        rng = np.random.default_rng(args.seed + 991)
        idx = rng.choice(idx, size=int(max_rows), replace=False).astype(np.int32)
    rows = []
    totals = {
        "count": 0,
        "pre_kl": 0.0,
        "post_kl": 0.0,
        "post_tv": 0.0,
        "post_acc": 0.0,
        "post_top2": 0.0,
        "support_ce": 0.0,
    }
    for start in range(0, idx.shape[0], args.eval_batch_size):
        part = idx[start : start + args.eval_batch_size]
        batch = batch_from_arrays(arrays, part)
        support_obs = jnp.asarray(batch["support_obs"])
        support_actions_np = batch["support_actions"]
        if history_mode == "wrong":
            support_actions_np = batch["wrong_actions"]
        elif history_mode == "random":
            support_actions_np = (batch["support_actions"] + batch["wrong_actions"] + 1) % action_dim
        elif history_mode == "no_history":
            support_obs = jnp.zeros_like(support_obs)
            support_actions_np = np.zeros_like(support_actions_np, dtype=np.int32)
        support_actions = jnp.asarray(support_actions_np)
        target = jnp.asarray(batch["target_probs"])
        z0, zk = adapt_z(
            params,
            support_obs,
            support_actions,
            action_dim,
            args.inner_steps,
            args.inner_lr,
            args.inner_prior_coef,
            args.inner_z_clip,
            True,
            args.init_z_mode,
        )
        pre_logits = apply_latent_partner_decoder_with_z(
            params, jnp.asarray(batch["query_obs"]), z0, action_dim
        )
        logits = apply_latent_partner_decoder_with_z(
            params, jnp.asarray(batch["query_obs"]), zk, action_dim
        )
        probs = jax.nn.softmax(logits, axis=-1)
        top2 = jnp.argsort(probs, axis=-1)[:, -2:]
        target_arg = jnp.argmax(target, axis=-1)
        count = int(part.shape[0])
        metrics = {
            "count": count,
            "pre_kl": float(categorical_kl(target, pre_logits).mean()),
            "post_kl": float(categorical_kl(target, logits).mean()),
            "post_tv": float(categorical_tv(target, logits).mean()),
            "post_acc": float((jnp.argmax(logits, axis=-1) == target_arg).mean()),
            "post_top2": float(jnp.any(top2 == target_arg[:, None], axis=-1).mean()),
            "support_ce": float(support_ce(params, support_obs, support_actions, zk, action_dim)),
        }
        rows.append(metrics)
        for key, value in metrics.items():
            if key == "count":
                continue
            totals[key] += value * count
        totals["count"] += count
    return {
        key: (value / max(totals["count"], 1) if key != "count" else totals["count"])
        for key, value in totals.items()
    }


def write_summary(path, args, metrics_by_mode):
    lines = [
        "# Meta Online-Z Decoder Training",
        "",
        f"- dataset: `{args.dataset}`",
        f"- init_decoder_path: `{args.init_decoder_path}`",
        f"- inner_steps: `{args.inner_steps}`",
        f"- inner_lr: `{args.inner_lr}`",
        f"- inner_prior_coef: `{args.inner_prior_coef}`",
        f"- first_order: `{args.first_order}`",
        f"- pre_update_kl_coef: `{args.pre_update_kl_coef}`",
        f"- support_ce_coef: `{args.support_ce_coef}`",
        f"- steps: `{args.steps}`",
        "",
        "| history | count | pre KL | post KL | post TV | acc | top2 | support CE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in metrics_by_mode.items():
        lines.append(
            "| {mode} | {count} | {pre_kl:.4f} | {post_kl:.4f} | {post_tv:.4f} | {post_acc:.4f} | {post_top2:.4f} | {support_ce:.4f} |".format(
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
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--inner_lr", type=float, default=0.05)
    parser.add_argument("--inner_prior_coef", type=float, default=0.05)
    parser.add_argument("--inner_z_clip", type=float, default=5.0)
    parser.add_argument(
        "--init_z_mode",
        choices=("encoder", "zero", "no_history"),
        default="encoder",
        help="Initial latent for the inner online update.",
    )
    parser.add_argument("--first_order", action="store_true")
    parser.add_argument("--pre_update_kl_coef", type=float, default=0.25)
    parser.add_argument("--support_ce_coef", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--eval_max_rows", type=int, default=30000)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {
        "query_obs": data["query_obs"],
        "partner_obs_history": data["partner_obs_history"],
        "partner_action_history": data["partner_action_history"],
        "target_partner_probs": normalize_probs(data["target_partner_probs"]),
    }
    if "ego_action_history" in data.files:
        arrays["ego_action_history"] = data["ego_action_history"]

    n_rows = int(arrays["query_obs"].shape[0])
    train_idx, val_idx = make_split(n_rows, args.val_fraction, args.seed)
    rng = np.random.default_rng(args.seed)
    params = load_latent_decoder_npz(Path(args.init_decoder_path))
    init_opt, train_step = make_train_step(args.action_dim, args)
    opt_state = init_opt(params)

    curve = []
    for step in range(1, args.steps + 1):
        batch_idx = rng.choice(train_idx, size=args.batch_size, replace=train_idx.shape[0] < args.batch_size)
        batch = {
            key: jnp.asarray(value)
            for key, value in batch_from_arrays(arrays, batch_idx).items()
            if key != "wrong_actions"
        }
        params, opt_state, metrics = train_step(params, opt_state, batch)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step}
            row.update({key: float(value) for key, value in metrics.items()})
            curve.append(row)
            print(f"[meta_online_z] {row}", flush=True)

    metrics_by_mode = {
        mode: eval_metrics(
            params,
            arrays,
            val_idx,
            args.action_dim,
            args,
            args.eval_max_rows,
            history_mode=mode,
        )
        for mode in ("true", "wrong", "random", "no_history")
    }
    save_latent_decoder_npz(out / "latent_partner_decoder.npz", params)
    write_csv(out / "training_curve.csv", curve)
    write_csv(
        out / "validation_metrics.csv",
        [{"history": mode, **metrics} for mode, metrics in metrics_by_mode.items()],
    )
    write_summary(out / "meta_online_z_summary.md", args, metrics_by_mode)
    print(f"Wrote {out / 'latent_partner_decoder.npz'}")
    print(metrics_by_mode)


if __name__ == "__main__":
    main()
