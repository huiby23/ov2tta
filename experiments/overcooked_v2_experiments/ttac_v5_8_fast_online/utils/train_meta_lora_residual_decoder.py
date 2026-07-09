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
    apply_latent_partner_decoder_with_z,
    categorical_kl,
    categorical_tv,
    encode_partner_online_z,
    load_latent_decoder_npz,
    save_latent_decoder_npz,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.train_lora_residual_decoder import (
    add_lora_params,
    delayed_history,
    split_by_episode,
    strip_lora,
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


def batch_from_arrays(arrays, idx):
    return {
        "query_obs": arrays["query_obs"][idx].astype(np.float32),
        "partner_obs_history": arrays["partner_obs_history"][idx].astype(np.float32),
        "partner_action_history": arrays["partner_action_history"][idx].astype(np.int32),
        "ego_action_history": arrays.get(
            "ego_action_history",
            np.zeros_like(arrays["partner_action_history"]),
        )[idx].astype(np.int32),
        "target_partner_probs": normalize_probs(arrays["target_partner_probs"][idx]),
    }


def lora_grad_mask(grads):
    return {
        key: (value if key.startswith("lora_") else jnp.zeros_like(value))
        for key, value in grads.items()
    }


def lora_l2_to_init(params, init_params):
    terms = []
    for key, value in params.items():
        if key.startswith("lora_") and key in init_params:
            terms.append(jnp.mean(jnp.square(value - jax.lax.stop_gradient(init_params[key]))))
    if not terms:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.mean(jnp.stack(terms))


def support_valid_mask(support_obs):
    obs_abs = jnp.sum(jnp.abs(support_obs), axis=tuple(range(1, support_obs.ndim)))
    valid = obs_abs > jnp.asarray(EPS, dtype=jnp.float32)
    valid = jnp.where(jnp.any(valid), valid, jnp.ones_like(valid))
    return valid.astype(jnp.float32)


def make_meta_step(action_dim, args, init_params):
    opt = optax.adam(args.lr)
    inner_lr = jnp.asarray(args.inner_lr, dtype=jnp.float32)
    inner_steps = int(args.inner_steps)
    first_order = bool(args.first_order)

    @jax.jit
    def init_opt(params):
        return opt.init(params)

    def inner_loss_single(params, anchor_params, support_obs, support_actions):
        z = encode_partner_online_z(
            anchor_params,
            support_obs[None, ...],
            support_actions[None, :],
            action_dim,
        )[0]
        z_batch = jnp.broadcast_to(z[None, :], (support_obs.shape[0], z.shape[0]))
        logits = apply_latent_partner_decoder_with_z(
            params,
            support_obs,
            z_batch,
            action_dim,
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        action_log_probs = jnp.take_along_axis(
            log_probs, support_actions[:, None], axis=-1
        )[:, 0]
        valid = support_valid_mask(support_obs)
        denom = jnp.maximum(jnp.sum(valid), 1.0)
        ce = -jnp.sum(action_log_probs * valid) / denom

        anchor_logits = apply_latent_partner_decoder_with_z(
            anchor_params,
            support_obs,
            z_batch,
            action_dim,
        )
        init_probs = jax.lax.stop_gradient(jax.nn.softmax(anchor_logits, axis=-1))
        init_log_probs = jax.lax.stop_gradient(jax.nn.log_softmax(anchor_logits, axis=-1))
        anchor_kl = jnp.sum(
            init_probs * (init_log_probs - log_probs),
            axis=-1,
        )
        anchor_kl = jnp.sum(anchor_kl * valid) / denom
        l2 = lora_l2_to_init(params, anchor_params)
        return (
            ce
            + jnp.asarray(args.inner_anchor_kl_coef, dtype=jnp.float32) * anchor_kl
            + jnp.asarray(args.inner_l2_coef, dtype=jnp.float32) * l2
        )

    def adapt_single(params, support_obs, support_actions):
        anchor_params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)

        def one_step(p, _):
            grads = jax.grad(inner_loss_single)(p, anchor_params, support_obs, support_actions)
            grads = lora_grad_mask(grads)
            if first_order:
                grads = jax.tree_util.tree_map(jax.lax.stop_gradient, grads)
            next_p = jax.tree_util.tree_map(lambda x, g: x - inner_lr * g, p, grads)
            return next_p, None

        if inner_steps <= 0:
            return params
        next_params, _ = jax.lax.scan(one_step, params, None, length=inner_steps)
        return next_params

    def outer_loss_single(params, support_obs, support_actions, query_obs, target):
        adapted = adapt_single(params, support_obs, support_actions)
        logits, aux = apply_latent_partner_decoder(
            adapted,
            query_obs[None, ...],
            support_obs[None, ...],
            support_actions[None, :],
            action_dim,
            deterministic=True,
            return_aux=True,
        )
        base_logits = apply_latent_partner_decoder(
            params,
            query_obs[None, ...],
            support_obs[None, ...],
            support_actions[None, :],
            action_dim,
            deterministic=True,
            return_aux=False,
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.sum(jax.lax.stop_gradient(target[None, :]) * log_probs, axis=-1)[0]
        kl = categorical_kl(target[None, :], logits)[0]
        anchor_kl = jnp.sum(
            jax.nn.softmax(base_logits, axis=-1)
            * (jax.nn.log_softmax(base_logits, axis=-1) - log_probs),
            axis=-1,
        )[0]
        residual_l2 = jnp.mean(jnp.square(aux["lora_residual_logits"]))
        loss = (
            ce
            + jnp.asarray(args.outer_anchor_kl_coef, dtype=jnp.float32) * anchor_kl
            + jnp.asarray(args.outer_residual_l2_coef, dtype=jnp.float32) * residual_l2
        )
        return loss, (ce, kl, anchor_kl, residual_l2)

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            losses, aux = jax.vmap(
                lambda so, sa, qo, tg: outer_loss_single(p, so, sa, qo, tg),
                in_axes=(0, 0, 0, 0),
            )(
                batch["partner_obs_history"],
                batch["partner_action_history"],
                batch["query_obs"],
                batch["target_partner_probs"],
            )
            ce, kl, anchor_kl, residual_l2 = aux
            loss = jnp.mean(losses)
            metrics = {
                "loss": loss,
                "outer_ce": jnp.mean(ce),
                "outer_kl": jnp.mean(kl),
                "outer_anchor_kl": jnp.mean(anchor_kl),
                "outer_residual_l2": jnp.mean(residual_l2),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        del loss
        grads = lora_grad_mask(grads)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def eval_batch(params, batch):
        def eval_single(support_obs, support_actions, query_obs, target):
            adapted = adapt_single(params, support_obs, support_actions)
            pre_logits = apply_latent_partner_decoder(
                params,
                query_obs[None, ...],
                support_obs[None, ...],
                support_actions[None, :],
                action_dim,
                deterministic=True,
                return_aux=False,
            )
            post_logits, aux = apply_latent_partner_decoder(
                adapted,
                query_obs[None, ...],
                support_obs[None, ...],
                support_actions[None, :],
                action_dim,
                deterministic=True,
                return_aux=True,
            )
            pre_kl = categorical_kl(target[None, :], pre_logits)[0]
            post_kl = categorical_kl(target[None, :], post_logits)[0]
            pre_tv = categorical_tv(target[None, :], pre_logits)[0]
            post_tv = categorical_tv(target[None, :], post_logits)[0]
            target_arg = jnp.argmax(target, axis=-1)
            pre_acc = (jnp.argmax(pre_logits[0], axis=-1) == target_arg).astype(jnp.float32)
            post_acc = (jnp.argmax(post_logits[0], axis=-1) == target_arg).astype(jnp.float32)
            post_top2 = jnp.argsort(post_logits[0], axis=-1)[-2:]
            post_top2_acc = jnp.any(post_top2 == target_arg).astype(jnp.float32)
            gate = jnp.mean(aux["lora_gate"])
            residual_tv = categorical_tv(jax.nn.softmax(pre_logits, axis=-1), post_logits)[0]
            return pre_kl, post_kl, pre_tv, post_tv, pre_acc, post_acc, post_top2_acc, residual_tv, gate

        return jax.vmap(eval_single, in_axes=(0, 0, 0, 0))(
            batch["partner_obs_history"],
            batch["partner_action_history"],
            batch["query_obs"],
            batch["target_partner_probs"],
        )

    return init_opt, train_step, eval_batch


def eval_history(params, arrays, idx, action_dim, args, eval_batch_fn, history_mode):
    idx = np.asarray(idx, dtype=np.int32)
    if args.eval_max_rows and idx.shape[0] > args.eval_max_rows:
        rng = np.random.default_rng(args.seed + 812)
        idx = rng.choice(idx, size=int(args.eval_max_rows), replace=False).astype(np.int32)
    totals = {
        "count": 0,
        "pre_kl": 0.0,
        "post_kl": 0.0,
        "pre_tv": 0.0,
        "post_tv": 0.0,
        "pre_acc": 0.0,
        "post_acc": 0.0,
        "post_top2_acc": 0.0,
        "residual_tv": 0.0,
        "gate_mean": 0.0,
    }
    rng = np.random.default_rng(args.seed + 991)
    for start in range(0, idx.shape[0], args.eval_batch_size):
        part_idx = idx[start : start + args.eval_batch_size]
        batch = batch_from_arrays(arrays, part_idx)
        if history_mode == "wrong":
            batch["partner_action_history"] = batch["ego_action_history"]
        elif history_mode == "delayed":
            batch["partner_action_history"] = delayed_history(batch["partner_action_history"])
        elif history_mode == "random":
            batch["partner_action_history"] = rng.integers(
                0,
                action_dim,
                size=batch["partner_action_history"].shape,
                dtype=np.int32,
            )
        elif history_mode == "no_history":
            batch["partner_obs_history"] = np.zeros_like(batch["partner_obs_history"], dtype=np.float32)
            batch["partner_action_history"] = np.zeros_like(batch["partner_action_history"], dtype=np.int32)
        jbatch = {key: jnp.asarray(value) for key, value in batch.items()}
        values = eval_batch_fn(params, jbatch)
        keys = (
            "pre_kl",
            "post_kl",
            "pre_tv",
            "post_tv",
            "pre_acc",
            "post_acc",
            "post_top2_acc",
            "residual_tv",
            "gate_mean",
        )
        count = int(part_idx.shape[0])
        for key, value in zip(keys, values):
            totals[key] += float(jnp.mean(value)) * count
        totals["count"] += count
    return {
        key: (value / max(totals["count"], 1) if key != "count" else totals["count"])
        for key, value in totals.items()
    }


def write_summary(path, args, metrics_by_mode):
    lines = [
        "# Meta LoRA Residual Latent Partner Decoder",
        "",
        f"- dataset: `{args.dataset}`",
        f"- init_decoder_path: `{args.init_decoder_path}`",
        f"- rank: `{args.rank}`",
        f"- lora_alpha: `{args.lora_alpha}`",
        f"- no_gate: `{args.no_gate}`",
        f"- reuse_lora_if_present: `{args.reuse_lora_if_present}`",
        f"- inner_lr: `{args.inner_lr}`",
        f"- inner_steps: `{args.inner_steps}`",
        f"- first_order: `{args.first_order}`",
        f"- inner_anchor_kl_coef: `{args.inner_anchor_kl_coef}`",
        f"- inner_l2_coef: `{args.inner_l2_coef}`",
        f"- outer_anchor_kl_coef: `{args.outer_anchor_kl_coef}`",
        f"- outer_residual_l2_coef: `{args.outer_residual_l2_coef}`",
        f"- steps: `{args.steps}`",
        "",
        "| history | count | pre KL | post KL | delta KL | pre TV | post TV | pre acc | post acc | post top2 | residual TV | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in metrics_by_mode.items():
        lines.append(
            "| {mode} | {count} | {pre_kl:.4f} | {post_kl:.4f} | {delta_kl:.4f} | {pre_tv:.4f} | {post_tv:.4f} | {pre_acc:.4f} | {post_acc:.4f} | {post_top2_acc:.4f} | {residual_tv:.4f} | {gate_mean:.4f} |".format(
                mode=mode,
                delta_kl=metrics["post_kl"] - metrics["pre_kl"],
                **metrics,
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
    parser.add_argument("--reuse_lora_if_present", action="store_true")
    parser.add_argument("--coeff_hidden_dim", type=int, default=64)
    parser.add_argument("--gate_bias", type=float, default=-2.0)
    parser.add_argument("--residual_clip", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--first_order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inner_anchor_kl_coef", type=float, default=0.05)
    parser.add_argument("--inner_l2_coef", type=float, default=1e-3)
    parser.add_argument("--outer_anchor_kl_coef", type=float, default=0.1)
    parser.add_argument("--outer_residual_l2_coef", type=float, default=0.01)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_max_rows", type=int, default=12000)
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
    if args.reuse_lora_if_present and "latent_lora_residual_enabled" in base_params:
        params = dict(base_params)
    else:
        params = add_lora_params(
            base_params,
            jax.random.PRNGKey(args.seed + 271),
            int(args.rank),
            int(args.coeff_hidden_dim),
            int(args.action_dim),
            float(args.gate_bias),
            float(args.residual_clip),
            float(args.lora_alpha),
            bool(args.no_gate),
        )
    # Keep the old VAE fixed, but train the LoRA meta-initialization.
    init_params = params
    init_opt, train_step, eval_batch_fn = make_meta_step(args.action_dim, args, init_params)
    opt_state = init_opt(params)
    curve = []

    for step in range(1, args.steps + 1):
        batch_idx = rng_np.choice(
            train_idx,
            size=args.batch_size,
            replace=train_idx.shape[0] < args.batch_size,
        ).astype(np.int32)
        batch = {
            key: jnp.asarray(value)
            for key, value in batch_from_arrays(arrays, batch_idx).items()
        }
        params, opt_state, metrics = train_step(params, opt_state, batch)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step}
            row.update({key: float(value) for key, value in metrics.items()})
            curve.append(row)
            print(f"[meta_lora_decoder] {row}", flush=True)

    metrics_by_mode = {
        mode: eval_history(
            params,
            arrays,
            val_idx,
            args.action_dim,
            args,
            eval_batch_fn,
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
    write_summary(out / "meta_lora_summary.md", args, metrics_by_mode)
    print(f"Wrote {out / 'latent_partner_decoder.npz'}")
    print(metrics_by_mode)


if __name__ == "__main__":
    main()
