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


def load_dataset_arrays(path):
    data = np.load(path, allow_pickle=False)
    arrays = {
        "query_obs": data["query_obs"],
        "partner_obs_history": data["partner_obs_history"],
        "partner_action_history": data["partner_action_history"],
        "target_partner_probs": normalize_probs(data["target_partner_probs"]),
    }
    if "mean_teacher_probs" in data.files:
        arrays["mean_teacher_probs"] = normalize_probs(data["mean_teacher_probs"])
    else:
        arrays["mean_teacher_probs"] = arrays["target_partner_probs"]
    if "query_disagreement_tv" in data.files:
        arrays["query_disagreement_tv"] = data["query_disagreement_tv"].astype(np.float32)
    else:
        arrays["query_disagreement_tv"] = np.ones(
            (arrays["query_obs"].shape[0],), dtype=np.float32
        )
    if "ego_action_history" in data.files:
        arrays["ego_action_history"] = data["ego_action_history"]
    return arrays


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
        "mean_teacher_probs": arrays["mean_teacher_probs"][idx].astype(np.float32),
        "query_disagreement_tv": arrays["query_disagreement_tv"][idx].astype(np.float32),
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


def split_support_future(obs_hist, action_hist, support_fraction):
    hist_len = int(action_hist.shape[1])
    n_support = int(round(hist_len * float(support_fraction)))
    n_support = max(1, min(hist_len, n_support))
    support_obs = obs_hist[:, :n_support]
    support_actions = action_hist[:, :n_support]
    if n_support >= hist_len:
        future_obs = obs_hist[:, -1:]
        future_actions = action_hist[:, -1:]
        has_future = False
    else:
        future_obs = obs_hist[:, n_support:]
        future_actions = action_hist[:, n_support:]
        has_future = True
    return support_obs, support_actions, future_obs, future_actions, has_future


def init_residual_params(base_params, rng, c_dim, hidden_dim, action_dim, residual_scale, residual_clip):
    q_dim = int(base_params["attn_frame_b"].shape[0])
    z_dim = int(base_params["vae_mu_b"].shape[0])
    k1, k2 = jax.random.split(rng)
    return {
        "latent_online_residual_c_enabled": np.asarray(1.0, dtype=np.float32),
        "orc_c_init": np.zeros((c_dim,), dtype=np.float32),
        "orc_feat_w": np.asarray(
            jax.random.normal(k1, (q_dim + z_dim, hidden_dim), dtype=jnp.float32)
            * np.float32(1.0 / np.sqrt(q_dim + z_dim))
        ),
        "orc_feat_b": np.zeros((hidden_dim,), dtype=np.float32),
        "orc_basis_w": np.asarray(
            jax.random.normal(k2, (hidden_dim, c_dim * action_dim), dtype=jnp.float32)
            * np.float32(residual_scale / np.sqrt(hidden_dim))
        ),
        "orc_basis_b": np.zeros((c_dim * action_dim,), dtype=np.float32),
        "online_residual_c_clip": np.asarray(residual_clip, dtype=np.float32),
    }


def merge_params(base_params, residual_params):
    merged = dict(base_params)
    merged.update(residual_params)
    return merged


def support_ce(params, support_obs, support_actions, latent_z, action_dim):
    batch, hist_len = support_actions.shape[:2]
    z_rep = jnp.broadcast_to(latent_z[:, None, :], (batch, hist_len, latent_z.shape[-1]))
    logits = apply_latent_partner_decoder_with_z(
        params,
        jnp.reshape(support_obs, (batch * hist_len,) + support_obs.shape[2:]),
        jnp.reshape(z_rep, (batch * hist_len, latent_z.shape[-1])),
        action_dim,
    )
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = jnp.reshape(support_actions.astype(jnp.int32), (batch * hist_len,))
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]
    valid = jnp.reshape(support_valid_mask(support_obs).astype(jnp.float32), (-1,))
    return -jnp.sum(action_log_probs * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def centered_action_logits(x):
    return x - jnp.mean(x, axis=-1, keepdims=True)


def residual_log_ratio_mse(target_probs, mean_probs, pre_logits, post_logits, weights=None):
    target_log_ratio = jnp.log(jnp.maximum(target_probs, EPS)) - jnp.log(
        jnp.maximum(mean_probs, EPS)
    )
    target_log_ratio = centered_action_logits(target_log_ratio)
    pred_delta = centered_action_logits(post_logits - pre_logits)
    per_row = jnp.mean(jnp.square(pred_delta - target_log_ratio), axis=-1)
    if weights is None:
        return jnp.mean(per_row)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    return jnp.sum(per_row * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def adapt_c(
    params,
    support_obs,
    support_actions,
    action_dim,
    inner_steps,
    inner_lr,
    inner_prior_coef,
    inner_c_clip,
    first_order,
):
    base_zc = encode_partner_online_z(params, support_obs, support_actions, action_dim)
    z_dim = int(params["vae_mu_b"].shape[0])
    z0 = jax.lax.stop_gradient(base_zc[:, :z_dim])
    c0 = jnp.zeros_like(base_zc[:, z_dim:])

    def pack(c):
        return jnp.concatenate([z0, c], axis=-1)

    def inner_loss(c):
        ce = support_ce(params, support_obs, support_actions, pack(c), action_dim)
        prior = jnp.mean(jnp.square(c))
        return ce + jnp.asarray(inner_prior_coef, dtype=jnp.float32) * prior

    def one_step(c, _):
        grad = jax.grad(inner_loss)(c)
        next_c = c - jnp.asarray(inner_lr, dtype=jnp.float32) * grad
        next_c = jnp.clip(
            next_c,
            -jnp.asarray(inner_c_clip, dtype=jnp.float32),
            jnp.asarray(inner_c_clip, dtype=jnp.float32),
        )
        if first_order:
            next_c = jax.lax.stop_gradient(next_c)
        return next_c, None

    if inner_steps <= 0:
        ck = c0
    else:
        ck, _ = jax.lax.scan(one_step, c0, None, length=inner_steps)
    return pack(c0), pack(ck), ck


def make_train_step(base_params, action_dim, args):
    optimizer = optax.adam(args.lr)

    @jax.jit
    def init_opt(residual_params):
        return optimizer.init(residual_params)

    @jax.jit
    def train_step(residual_params, opt_state, batch):
        target = batch["target_probs"]
        mean_target = batch["mean_teacher_probs"]

        def loss_fn(rp):
            params = merge_params(base_params, rp)
            adapt_obs, adapt_actions, future_obs, future_actions, has_future = split_support_future(
                batch["support_obs"], batch["support_actions"], args.inner_support_fraction
            )
            z_pre, z_post, c_post = adapt_c(
                params,
                adapt_obs,
                adapt_actions,
                action_dim,
                args.inner_steps,
                args.inner_lr,
                args.inner_prior_coef,
                args.inner_c_clip,
                args.first_order,
            )
            pre_logits = apply_latent_partner_decoder_with_z(
                params, batch["query_obs"], z_pre, action_dim
            )
            post_logits = apply_latent_partner_decoder_with_z(
                params, batch["query_obs"], z_post, action_dim
            )
            post_probs = jax.nn.softmax(post_logits, axis=-1)
            pre_kl_per = categorical_kl(target, pre_logits)
            post_kl_per = categorical_kl(target, post_logits)
            pre_kl = pre_kl_per.mean()
            post_kl = post_kl_per.mean()
            post_tv = categorical_tv(target, post_logits).mean()
            acc = (jnp.argmax(post_logits, axis=-1) == jnp.argmax(target, axis=-1)).mean()
            support_loss_pre = support_ce(
                params, adapt_obs, adapt_actions, z_pre, action_dim
            )
            support_loss = support_ce(
                params, adapt_obs, adapt_actions, z_post, action_dim
            )
            if has_future:
                future_loss_pre = support_ce(params, future_obs, future_actions, z_pre, action_dim)
                future_loss = support_ce(params, future_obs, future_actions, z_post, action_dim)
            else:
                future_loss_pre = jnp.asarray(0.0, dtype=jnp.float32)
                future_loss = jnp.asarray(0.0, dtype=jnp.float32)
            c_norm = jnp.mean(jnp.linalg.norm(c_post, axis=-1))
            residual_logits = post_logits - pre_logits
            pre_probs = jax.nn.softmax(pre_logits, axis=-1)
            residual_tv_per = categorical_tv(pre_probs, post_logits)
            residual_tv = residual_tv_per.mean()
            residual_l2 = jnp.mean(jnp.square(residual_logits))
            residual_weights = 1.0 + jnp.asarray(
                args.residual_disagreement_weight, dtype=jnp.float32
            ) * batch["query_disagreement_tv"]
            residual_log_ratio_loss = residual_log_ratio_mse(
                target,
                mean_target,
                pre_logits,
                post_logits,
                weights=residual_weights,
            )
            residual_prob_mse = jnp.mean(
                jnp.square((post_probs - pre_probs) - (target - mean_target))
            )

            anchor_kl = categorical_kl(pre_probs, post_logits).mean()
            low_mask = batch["query_disagreement_tv"] < jnp.asarray(
                args.low_disagreement_threshold, dtype=jnp.float32
            )
            low_den = jnp.maximum(jnp.sum(low_mask.astype(jnp.float32)), 1.0)
            low_residual_penalty = (
                jnp.sum(jnp.square(residual_tv_per) * low_mask.astype(jnp.float32)) / low_den
            )

            batch_size = batch["support_actions"].shape[0]
            rolled_obs = jnp.roll(batch["support_obs"], shift=1, axis=0)
            rolled_actions = jnp.roll(batch["support_actions"], shift=1, axis=0)
            rolled_obs = jnp.where(batch_size <= 1, batch["support_obs"], rolled_obs)
            rolled_actions = jnp.where(batch_size <= 1, batch["support_actions"], rolled_actions)
            no_obs = jnp.zeros_like(batch["support_obs"])
            no_actions = jnp.zeros_like(batch["support_actions"])
            wrong_adapt_obs, wrong_adapt_actions, _, _, _ = split_support_future(
                rolled_obs, rolled_actions, args.inner_support_fraction
            )
            delayed_actions = jnp.concatenate(
                [
                    jnp.zeros(
                        (batch["support_actions"].shape[0], 1),
                        dtype=batch["support_actions"].dtype,
                    ),
                    batch["support_actions"][:, :-1],
                ],
                axis=1,
            )
            delayed_adapt_obs, delayed_adapt_actions, _, _, _ = split_support_future(
                batch["support_obs"], delayed_actions, args.inner_support_fraction
            )
            no_adapt_obs, no_adapt_actions, _, _, _ = split_support_future(
                no_obs, no_actions, args.inner_support_fraction
            )

            wrong_pre_z, wrong_post_z, _wrong_c = adapt_c(
                params,
                wrong_adapt_obs,
                wrong_adapt_actions,
                action_dim,
                args.inner_steps,
                args.inner_lr,
                args.inner_prior_coef,
                args.inner_c_clip,
                args.first_order,
            )
            delayed_pre_z, delayed_post_z, _delayed_c = adapt_c(
                params,
                delayed_adapt_obs,
                delayed_adapt_actions,
                action_dim,
                args.inner_steps,
                args.inner_lr,
                args.inner_prior_coef,
                args.inner_c_clip,
                args.first_order,
            )
            no_pre_z, no_post_z, _no_c = adapt_c(
                params,
                no_adapt_obs,
                no_adapt_actions,
                action_dim,
                args.inner_steps,
                args.inner_lr,
                args.inner_prior_coef,
                args.inner_c_clip,
                args.first_order,
            )
            wrong_pre_logits = apply_latent_partner_decoder_with_z(
                params, batch["query_obs"], wrong_pre_z, action_dim
            )
            wrong_logits = apply_latent_partner_decoder_with_z(
                params, batch["query_obs"], wrong_post_z, action_dim
            )
            delayed_pre_logits = apply_latent_partner_decoder_with_z(
                params, batch["query_obs"], delayed_pre_z, action_dim
            )
            delayed_logits = apply_latent_partner_decoder_with_z(
                params, batch["query_obs"], delayed_post_z, action_dim
            )
            no_pre_logits = apply_latent_partner_decoder_with_z(
                params, batch["query_obs"], no_pre_z, action_dim
            )
            no_logits = apply_latent_partner_decoder_with_z(
                params, batch["query_obs"], no_post_z, action_dim
            )
            wrong_pre_kl_per = categorical_kl(target, wrong_pre_logits)
            wrong_kl_per = categorical_kl(target, wrong_logits)
            delayed_pre_kl_per = categorical_kl(target, delayed_pre_logits)
            delayed_kl_per = categorical_kl(target, delayed_logits)
            no_pre_kl_per = categorical_kl(target, no_pre_logits)
            no_kl_per = categorical_kl(target, no_logits)
            margin = jnp.asarray(args.margin, dtype=jnp.float32)
            wrong_margin = jnp.maximum(0.0, margin + post_kl_per - wrong_kl_per).mean()
            delayed_margin = jnp.maximum(0.0, margin + post_kl_per - delayed_kl_per).mean()
            no_history_margin = jnp.maximum(0.0, margin + post_kl_per - no_kl_per).mean()
            margin_loss = (wrong_margin + delayed_margin + no_history_margin) / 3.0
            anti_margin = jnp.asarray(args.anti_generic_margin, dtype=jnp.float32)
            wrong_generic_improvement = jnp.maximum(
                0.0, wrong_pre_kl_per - wrong_kl_per + anti_margin
            ).mean()
            delayed_generic_improvement = jnp.maximum(
                0.0, delayed_pre_kl_per - delayed_kl_per + anti_margin
            ).mean()
            no_history_generic_improvement = jnp.maximum(
                0.0, no_pre_kl_per - no_kl_per + anti_margin
            ).mean()
            anti_generic_loss = (
                wrong_generic_improvement
                + delayed_generic_improvement
                + no_history_generic_improvement
            ) / 3.0
            c_l2 = jnp.mean(jnp.square(c_post))
            loss = (
                post_kl
                + jnp.asarray(args.pre_kl_coef, dtype=jnp.float32) * pre_kl
                + jnp.asarray(args.support_ce_coef, dtype=jnp.float32) * support_loss
                + jnp.asarray(args.future_ce_coef, dtype=jnp.float32) * future_loss
                + jnp.asarray(args.residual_log_ratio_coef, dtype=jnp.float32)
                * residual_log_ratio_loss
                + jnp.asarray(args.anchor_kl_coef, dtype=jnp.float32) * anchor_kl
                + jnp.asarray(args.low_disagreement_residual_coef, dtype=jnp.float32)
                * low_residual_penalty
                + jnp.asarray(args.margin_loss_coef, dtype=jnp.float32) * margin_loss
                + jnp.asarray(args.anti_generic_coef, dtype=jnp.float32)
                * anti_generic_loss
                + jnp.asarray(args.c_l2_coef, dtype=jnp.float32) * c_l2
                + jnp.asarray(args.residual_l2_coef, dtype=jnp.float32) * residual_l2
            )
            metrics = {
                "loss": loss,
                "pre_kl": pre_kl,
                "post_kl": post_kl,
                "delta_kl": post_kl - pre_kl,
                "post_tv": post_tv,
                "post_acc": acc,
                "support_ce_pre": support_loss_pre,
                "support_ce": support_loss,
                "future_ce_pre": future_loss_pre,
                "future_ce": future_loss,
                "c_norm": c_norm,
                "residual_tv": residual_tv,
                "residual_l2": residual_l2,
                "residual_log_ratio_loss": residual_log_ratio_loss,
                "residual_prob_mse": residual_prob_mse,
                "anchor_kl": anchor_kl,
                "low_residual_penalty": low_residual_penalty,
                "margin_loss": margin_loss,
                "anti_generic_loss": anti_generic_loss,
                "wrong_margin": wrong_margin,
                "delayed_margin": delayed_margin,
                "no_history_margin": no_history_margin,
                "wrong_generic_improvement": wrong_generic_improvement,
                "delayed_generic_improvement": delayed_generic_improvement,
                "no_history_generic_improvement": no_history_generic_improvement,
                "wrong_pre_kl": wrong_pre_kl_per.mean(),
                "wrong_kl": wrong_kl_per.mean(),
                "delayed_pre_kl": delayed_pre_kl_per.mean(),
                "delayed_kl": delayed_kl_per.mean(),
                "no_history_pre_kl": no_pre_kl_per.mean(),
                "no_history_kl": no_kl_per.mean(),
                "c_l2": c_l2,
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(residual_params)
        del loss
        updates, opt_state = optimizer.update(grads, opt_state, residual_params)
        residual_params = optax.apply_updates(residual_params, updates)
        return residual_params, opt_state, metrics

    return init_opt, train_step


def eval_metrics(base_params, residual_params, arrays, idx, action_dim, args, history_mode):
    params = merge_params(base_params, residual_params)
    idx = np.asarray(idx, dtype=np.int32)
    if args.eval_max_rows and idx.shape[0] > args.eval_max_rows:
        rng = np.random.default_rng(args.seed + 991)
        idx = rng.choice(idx, size=args.eval_max_rows, replace=False).astype(np.int32)
    totals = {
        "count": 0,
        "pre_kl": 0.0,
        "post_kl": 0.0,
        "post_tv": 0.0,
        "post_acc": 0.0,
        "post_top2": 0.0,
        "support_ce_pre": 0.0,
        "support_ce": 0.0,
        "future_ce_pre": 0.0,
        "future_ce": 0.0,
        "c_norm": 0.0,
        "residual_tv": 0.0,
        "residual_log_ratio_mse": 0.0,
        "high_count": 0,
        "high_post_kl": 0.0,
        "low_count": 0,
        "low_post_kl": 0.0,
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
        elif history_mode == "delayed":
            support_actions_np = np.concatenate(
                [
                    np.zeros((support_actions_np.shape[0], 1), dtype=np.int32),
                    support_actions_np[:, :-1],
                ],
                axis=1,
            )
        elif history_mode == "no_history":
            support_obs = jnp.zeros_like(support_obs)
            support_actions_np = np.zeros_like(support_actions_np, dtype=np.int32)
        support_actions = jnp.asarray(support_actions_np)
        adapt_obs, adapt_actions, future_obs, future_actions, has_future = split_support_future(
            support_obs, support_actions, args.inner_support_fraction
        )
        target = jnp.asarray(batch["target_probs"])
        mean_target = jnp.asarray(batch["mean_teacher_probs"])
        z_pre, z_post, c_post = adapt_c(
            params,
            adapt_obs,
            adapt_actions,
            action_dim,
            args.inner_steps,
            args.inner_lr,
            args.inner_prior_coef,
            args.inner_c_clip,
            True,
        )
        pre_logits = apply_latent_partner_decoder_with_z(
            params, jnp.asarray(batch["query_obs"]), z_pre, action_dim
        )
        logits = apply_latent_partner_decoder_with_z(
            params, jnp.asarray(batch["query_obs"]), z_post, action_dim
        )
        probs = jax.nn.softmax(logits, axis=-1)
        top2 = jnp.argsort(probs, axis=-1)[:, -2:]
        target_arg = jnp.argmax(target, axis=-1)
        post_kl_per = categorical_kl(target, logits)
        disagreement = jnp.asarray(batch["query_disagreement_tv"], dtype=jnp.float32)
        high_mask = disagreement >= jnp.asarray(args.high_disagreement_threshold, dtype=jnp.float32)
        low_mask = disagreement < jnp.asarray(args.low_disagreement_threshold, dtype=jnp.float32)
        high_count = int(jnp.sum(high_mask.astype(jnp.int32)))
        low_count = int(jnp.sum(low_mask.astype(jnp.int32)))
        count = int(part.shape[0])
        metrics = {
            "pre_kl": float(categorical_kl(target, pre_logits).mean()),
            "post_kl": float(post_kl_per.mean()),
            "post_tv": float(categorical_tv(target, logits).mean()),
            "post_acc": float((jnp.argmax(logits, axis=-1) == target_arg).mean()),
            "post_top2": float(jnp.any(top2 == target_arg[:, None], axis=-1).mean()),
            "support_ce_pre": float(support_ce(params, adapt_obs, adapt_actions, z_pre, action_dim)),
            "support_ce": float(support_ce(params, adapt_obs, adapt_actions, z_post, action_dim)),
            "future_ce_pre": float(
                support_ce(params, future_obs, future_actions, z_pre, action_dim)
                if has_future
                else 0.0
            ),
            "future_ce": float(
                support_ce(params, future_obs, future_actions, z_post, action_dim)
                if has_future
                else 0.0
            ),
            "c_norm": float(jnp.mean(jnp.linalg.norm(c_post, axis=-1))),
            "residual_tv": float(categorical_tv(jax.nn.softmax(pre_logits, axis=-1), logits).mean()),
            "residual_log_ratio_mse": float(
                residual_log_ratio_mse(target, mean_target, pre_logits, logits)
            ),
            "high_post_kl": float(
                jnp.sum(post_kl_per * high_mask.astype(jnp.float32))
                / jnp.maximum(jnp.sum(high_mask.astype(jnp.float32)), 1.0)
            ),
            "low_post_kl": float(
                jnp.sum(post_kl_per * low_mask.astype(jnp.float32))
                / jnp.maximum(jnp.sum(low_mask.astype(jnp.float32)), 1.0)
            ),
        }
        for key, value in metrics.items():
            if key in ("high_post_kl", "low_post_kl"):
                continue
            totals[key] += value * count
        totals["high_post_kl"] += metrics["high_post_kl"] * high_count
        totals["low_post_kl"] += metrics["low_post_kl"] * low_count
        totals["high_count"] += high_count
        totals["low_count"] += low_count
        totals["count"] += count
    out = {}
    for key, value in totals.items():
        if key in ("count", "high_count", "low_count"):
            out[key] = value
        elif key == "high_post_kl":
            out[key] = value / max(totals["high_count"], 1)
        elif key == "low_post_kl":
            out[key] = value / max(totals["low_count"], 1)
        else:
            out[key] = value / max(totals["count"], 1)
    return out


def write_summary(path, args, metrics_by_mode):
    lines = [
        "# Meta Residual-C Latent Partner Decoder",
        "",
        f"- dataset: `{args.dataset}`",
        f"- eval_dataset: `{args.eval_dataset or ''}`",
        f"- init_decoder_path: `{args.init_decoder_path}`",
        f"- c_dim: `{args.c_dim}`",
        f"- hidden_dim: `{args.hidden_dim}`",
        f"- inner_steps: `{args.inner_steps}`",
        f"- inner_lr: `{args.inner_lr}`",
        f"- inner_prior_coef: `{args.inner_prior_coef}`",
        f"- inner_support_fraction: `{args.inner_support_fraction}`",
        f"- first_order: `{args.first_order}`",
        f"- support_ce_coef: `{args.support_ce_coef}`",
        f"- future_ce_coef: `{args.future_ce_coef}`",
        f"- residual_log_ratio_coef: `{args.residual_log_ratio_coef}`",
        f"- residual_disagreement_weight: `{args.residual_disagreement_weight}`",
        f"- anchor_kl_coef: `{args.anchor_kl_coef}`",
        f"- low_disagreement_residual_coef: `{args.low_disagreement_residual_coef}`",
        f"- low_disagreement_threshold: `{args.low_disagreement_threshold}`",
        f"- high_disagreement_threshold: `{args.high_disagreement_threshold}`",
        f"- margin_loss_coef: `{args.margin_loss_coef}`",
        f"- margin: `{args.margin}`",
        f"- anti_generic_coef: `{args.anti_generic_coef}`",
        f"- anti_generic_margin: `{args.anti_generic_margin}`",
        f"- c_l2_coef: `{args.c_l2_coef}`",
        f"- residual_l2_coef: `{args.residual_l2_coef}`",
        f"- steps: `{args.steps}`",
        "",
        "| history | count | pre KL | post KL | delta KL | post TV | acc | top2 | support CE pre | support CE post | future CE pre | future CE post | c norm | residual TV | resid LR MSE | high KL | low KL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in metrics_by_mode.items():
        lines.append(
            "| {mode} | {count} | {pre_kl:.4f} | {post_kl:.4f} | {delta_kl:.4f} | {post_tv:.4f} | {post_acc:.4f} | {post_top2:.4f} | {support_ce_pre:.4f} | {support_ce:.4f} | {future_ce_pre:.4f} | {future_ce:.4f} | {c_norm:.4f} | {residual_tv:.4f} | {residual_log_ratio_mse:.4f} | {high_post_kl:.4f} | {low_post_kl:.4f} |".format(
                mode=mode,
                delta_kl=metrics["post_kl"] - metrics["pre_kl"],
                **metrics,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval_dataset", default="")
    parser.add_argument("--init_decoder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--c_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--residual_init_scale", type=float, default=0.02)
    parser.add_argument("--residual_clip", type=float, default=2.0)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--inner_lr", type=float, default=1.0)
    parser.add_argument("--inner_prior_coef", type=float, default=0.01)
    parser.add_argument("--inner_c_clip", type=float, default=3.0)
    parser.add_argument("--inner_support_fraction", type=float, default=1.0)
    parser.add_argument("--first_order", action="store_true")
    parser.add_argument("--pre_kl_coef", type=float, default=0.0)
    parser.add_argument("--support_ce_coef", type=float, default=0.0)
    parser.add_argument("--future_ce_coef", type=float, default=0.0)
    parser.add_argument("--residual_log_ratio_coef", type=float, default=0.0)
    parser.add_argument("--residual_disagreement_weight", type=float, default=0.0)
    parser.add_argument("--anchor_kl_coef", type=float, default=0.0)
    parser.add_argument("--low_disagreement_residual_coef", type=float, default=0.0)
    parser.add_argument("--low_disagreement_threshold", type=float, default=0.10)
    parser.add_argument("--high_disagreement_threshold", type=float, default=0.25)
    parser.add_argument("--margin_loss_coef", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--anti_generic_coef", type=float, default=0.0)
    parser.add_argument("--anti_generic_margin", type=float, default=0.0)
    parser.add_argument("--c_l2_coef", type=float, default=0.0)
    parser.add_argument("--residual_l2_coef", type=float, default=0.001)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--eval_max_rows", type=int, default=30000)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = load_dataset_arrays(args.dataset)
    n_rows = int(arrays["query_obs"].shape[0])
    if args.eval_dataset:
        train_idx = np.arange(n_rows, dtype=np.int32)
        val_arrays = load_dataset_arrays(args.eval_dataset)
        val_idx = np.arange(int(val_arrays["query_obs"].shape[0]), dtype=np.int32)
    else:
        train_idx, val_idx = make_split(n_rows, args.val_fraction, args.seed)
        val_arrays = arrays

    base_params = load_latent_decoder_npz(Path(args.init_decoder_path))
    rng = jax.random.PRNGKey(args.seed)
    residual_params = init_residual_params(
        base_params,
        rng,
        args.c_dim,
        args.hidden_dim,
        args.action_dim,
        args.residual_init_scale,
        args.residual_clip,
    )
    residual_params = {key: jnp.asarray(value) for key, value in residual_params.items()}
    rng_np = np.random.default_rng(args.seed)
    init_opt, train_step = make_train_step(base_params, args.action_dim, args)
    opt_state = init_opt(residual_params)

    curve = []
    for step in range(1, args.steps + 1):
        batch_idx = rng_np.choice(
            train_idx, size=args.batch_size, replace=train_idx.shape[0] < args.batch_size
        )
        batch = {
            key: jnp.asarray(value)
            for key, value in batch_from_arrays(arrays, batch_idx).items()
            if key != "wrong_actions"
        }
        residual_params, opt_state, metrics = train_step(residual_params, opt_state, batch)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step}
            row.update({key: float(value) for key, value in metrics.items()})
            curve.append(row)
            print(f"[meta_residual_c] {row}", flush=True)

    metrics_by_mode = {
        mode: eval_metrics(
            base_params,
            residual_params,
            val_arrays,
            val_idx,
            args.action_dim,
            args,
            history_mode=mode,
        )
        for mode in ("true", "wrong", "random", "delayed", "no_history")
    }
    merged = merge_params(base_params, residual_params)
    save_latent_decoder_npz(out / "latent_partner_decoder.npz", merged)
    write_csv(out / "training_curve.csv", curve)
    write_csv(
        out / "validation_metrics.csv",
        [{"history": mode, **metrics} for mode, metrics in metrics_by_mode.items()],
    )
    write_summary(out / "meta_residual_c_summary.md", args, metrics_by_mode)
    print(f"Wrote {out / 'latent_partner_decoder.npz'}")
    print(metrics_by_mode)


if __name__ == "__main__":
    main()
