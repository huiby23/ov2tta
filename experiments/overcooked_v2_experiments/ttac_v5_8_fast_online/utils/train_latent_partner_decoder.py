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
    apply_latent_compatible_decoder,
    categorical_kl,
    categorical_tv,
    load_latent_decoder_npz,
    save_latent_decoder_npz,
)

EPS = 1e-6


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def make_reward_weights(values, transform="linear", min_weight=0.25, max_weight=2.0, temperature=1.0):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    if transform == "linear":
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        scaled = (values - lo) / max(hi - lo, EPS)
        weights = min_weight + (max_weight - min_weight) * scaled
    elif transform == "exp_zscore":
        centered = (values - float(np.nanmean(values))) / max(float(np.nanstd(values)), EPS)
        weights = np.exp(centered / max(float(temperature), EPS))
        weights = np.clip(weights, min_weight, max_weight)
    else:
        raise ValueError(f"unknown reward weight transform: {transform}")
    weights = weights / max(float(np.nanmean(weights)), EPS)
    return weights.astype(np.float32)


def weighted_mean(values, weights):
    return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), EPS)


def obs_stats(arrays, sample_limit=5000, history_frame_limit=5000, seed=0):
    n = arrays["query_obs"].shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= sample_limit else rng.choice(n, size=sample_limit, replace=False)
    query_flat = arrays["query_obs"][idx].astype(np.float32).reshape(len(idx), -1)
    hist_idx = rng.choice(idx, size=min(len(idx), history_frame_limit), replace=False)
    hist_pos = rng.integers(0, arrays["partner_obs_history"].shape[1], size=len(hist_idx))
    hist_flat = arrays["partner_obs_history"][hist_idx, hist_pos].astype(np.float32).reshape(len(hist_idx), -1)
    flat = np.concatenate([query_flat, hist_flat], axis=0)
    return flat.mean(axis=0).astype(np.float32), (flat.std(axis=0) + 1e-3).astype(np.float32)


def init_params(rng, obs_dim, action_dim, frame_dim, gru_dim, z_dim, hidden_dim, mean, std, use_vae=False):
    keys = jax.random.split(rng, 17)
    scale = 0.02
    params = {
        "latent_obs_mean": jnp.asarray(mean, dtype=jnp.float32),
        "latent_obs_std": jnp.asarray(std, dtype=jnp.float32),
        "latent_frame_w": scale * jax.random.normal(keys[0], (obs_dim, frame_dim)),
        "latent_frame_b": jnp.zeros((frame_dim,), dtype=jnp.float32),
        "latent_action_w": scale * jax.random.normal(keys[1], (action_dim, frame_dim)),
        "latent_action_b": jnp.zeros((frame_dim,), dtype=jnp.float32),
        "latent_input_b": jnp.zeros((frame_dim,), dtype=jnp.float32),
        "latent_gru_wz": scale * jax.random.normal(keys[2], (frame_dim, gru_dim)),
        "latent_gru_uz": scale * jax.random.normal(keys[3], (gru_dim, gru_dim)),
        "latent_gru_bz": jnp.zeros((gru_dim,), dtype=jnp.float32),
        "latent_gru_wr": scale * jax.random.normal(keys[4], (frame_dim, gru_dim)),
        "latent_gru_ur": scale * jax.random.normal(keys[5], (gru_dim, gru_dim)),
        "latent_gru_br": jnp.zeros((gru_dim,), dtype=jnp.float32),
        "latent_gru_wh": scale * jax.random.normal(keys[6], (frame_dim, gru_dim)),
        "latent_gru_uh": scale * jax.random.normal(keys[7], (gru_dim, gru_dim)),
        "latent_gru_bh": jnp.zeros((gru_dim,), dtype=jnp.float32),
        "latent_z_w": scale * jax.random.normal(keys[8], (gru_dim, z_dim)),
        "latent_z_b": jnp.zeros((z_dim,), dtype=jnp.float32),
        "latent_dec_w1": scale * jax.random.normal(keys[9], (frame_dim + z_dim, hidden_dim)),
        "latent_dec_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "latent_dec_w2": scale * jax.random.normal(keys[10], (hidden_dim, hidden_dim)),
        "latent_dec_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "latent_dec_w3": scale * jax.random.normal(keys[11], (hidden_dim, action_dim)),
        "latent_dec_b3": jnp.zeros((action_dim,), dtype=jnp.float32),
    }
    if use_vae:
        params.update({
            "vae_mu_w": scale * jax.random.normal(keys[12], (gru_dim, z_dim)),
            "vae_mu_b": jnp.zeros((z_dim,), dtype=jnp.float32),
            "vae_logvar_w": scale * jax.random.normal(keys[13], (gru_dim, z_dim)),
            "vae_logvar_b": jnp.zeros((z_dim,), dtype=jnp.float32),
        })
    return params


def init_query_attn_params(
    rng,
    obs_shape,
    action_dim,
    frame_dim,
    hidden_dim,
    mean,
    std,
    history_len,
    conv_channels,
    num_heads,
    num_layers,
    z_dim,
    use_vae=False,
):
    if frame_dim % num_heads != 0:
        raise ValueError(f"frame_dim={frame_dim} must be divisible by num_heads={num_heads}")
    h, w, c = obs_shape
    head_dim = frame_dim // num_heads
    keys = list(jax.random.split(rng, 8 + (2 if use_vae else 0) + 7 * num_layers))
    scale = 0.02
    dec_input_dim = frame_dim + z_dim if use_vae else frame_dim
    params = {
        "latent_obs_mean": jnp.asarray(mean, dtype=jnp.float32),
        "latent_obs_std": jnp.asarray(std, dtype=jnp.float32),
        "attn_conv1_w": scale * jax.random.normal(keys.pop(0), (3, 3, c, conv_channels)),
        "attn_conv1_b": jnp.zeros((conv_channels,), dtype=jnp.float32),
        "attn_conv2_w": scale * jax.random.normal(keys.pop(0), (3, 3, conv_channels, conv_channels)),
        "attn_conv2_b": jnp.zeros((conv_channels,), dtype=jnp.float32),
        "attn_frame_w": scale * jax.random.normal(keys.pop(0), (h * w * conv_channels, frame_dim)),
        "attn_frame_b": jnp.zeros((frame_dim,), dtype=jnp.float32),
        "attn_action_w": scale * jax.random.normal(keys.pop(0), (action_dim, frame_dim)),
        "attn_action_b": jnp.zeros((frame_dim,), dtype=jnp.float32),
        "attn_pos_emb": scale * jax.random.normal(keys.pop(0), (history_len, frame_dim)),
        "attn_dec_w1": scale * jax.random.normal(keys.pop(0), (dec_input_dim, hidden_dim)),
        "attn_dec_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "attn_dec_w2": scale * jax.random.normal(keys.pop(0), (hidden_dim, hidden_dim)),
        "attn_dec_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "attn_dec_w3": scale * jax.random.normal(keys.pop(0), (hidden_dim, action_dim)),
        "attn_dec_b3": jnp.zeros((action_dim,), dtype=jnp.float32),
    }
    if use_vae:
        params.update({
            "vae_mu_w": scale * jax.random.normal(keys.pop(0), (frame_dim, z_dim)),
            "vae_mu_b": jnp.zeros((z_dim,), dtype=jnp.float32),
            "vae_logvar_w": scale * jax.random.normal(keys.pop(0), (frame_dim, z_dim)),
            "vae_logvar_b": jnp.zeros((z_dim,), dtype=jnp.float32),
        })
    for layer_idx in range(num_layers):
        prefix = f"attn_l{layer_idx}"
        params[f"{prefix}_wq"] = scale * jax.random.normal(
            keys.pop(0), (frame_dim, num_heads, head_dim)
        )
        params[f"{prefix}_wk"] = scale * jax.random.normal(
            keys.pop(0), (frame_dim, num_heads, head_dim)
        )
        params[f"{prefix}_wv"] = scale * jax.random.normal(
            keys.pop(0), (frame_dim, num_heads, head_dim)
        )
        params[f"{prefix}_wo"] = scale * jax.random.normal(keys.pop(0), (frame_dim, frame_dim))
        params[f"{prefix}_ff1_w"] = scale * jax.random.normal(keys.pop(0), (frame_dim, hidden_dim))
        params[f"{prefix}_ff1_b"] = jnp.zeros((hidden_dim,), dtype=jnp.float32)
        params[f"{prefix}_ff2_w"] = scale * jax.random.normal(keys.pop(0), (hidden_dim, frame_dim))
        params[f"{prefix}_ff2_b"] = jnp.zeros((frame_dim,), dtype=jnp.float32)
    return params


def add_compatible_head_params(params, rng, input_dim, hidden_dim, action_dim):
    keys = jax.random.split(rng, 3)
    scale = 0.02
    params = dict(params)
    params.update({
        "compat_dec_w1": scale * jax.random.normal(keys[0], (input_dim, hidden_dim)),
        "compat_dec_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "compat_dec_w2": scale * jax.random.normal(keys[1], (hidden_dim, hidden_dim)),
        "compat_dec_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "compat_dec_w3": scale * jax.random.normal(keys[2], (hidden_dim, action_dim)),
        "compat_dec_b3": jnp.zeros((action_dim,), dtype=jnp.float32),
    })
    return params


def add_residual_gated_head_params(params, rng, frame_dim, z_dim, hidden_dim, action_dim, residual_clip):
    keys = jax.random.split(rng, 9)
    scale = 0.02
    residual_input_dim = frame_dim + z_dim
    params = dict(params)
    params.update({
        "latent_residual_gate_enabled": jnp.asarray(1.0, dtype=jnp.float32),
        "latent_residual_clip": jnp.asarray(residual_clip, dtype=jnp.float32),
        "prior_dec_w1": scale * jax.random.normal(keys[0], (frame_dim, hidden_dim)),
        "prior_dec_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "prior_dec_w2": scale * jax.random.normal(keys[1], (hidden_dim, hidden_dim)),
        "prior_dec_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "prior_dec_w3": scale * jax.random.normal(keys[2], (hidden_dim, action_dim)),
        "prior_dec_b3": jnp.zeros((action_dim,), dtype=jnp.float32),
        "residual_dec_w1": scale * jax.random.normal(keys[3], (residual_input_dim, hidden_dim)),
        "residual_dec_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "residual_dec_w2": scale * jax.random.normal(keys[4], (hidden_dim, hidden_dim)),
        "residual_dec_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "residual_dec_w3": scale * jax.random.normal(keys[5], (hidden_dim, action_dim)),
        "residual_dec_b3": jnp.zeros((action_dim,), dtype=jnp.float32),
        "gate_dec_w1": scale * jax.random.normal(keys[6], (residual_input_dim, hidden_dim)),
        "gate_dec_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "gate_dec_w2": scale * jax.random.normal(keys[7], (hidden_dim, hidden_dim)),
        "gate_dec_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "gate_dec_w3": scale * jax.random.normal(keys[8], (hidden_dim, 1)),
        "gate_dec_b3": jnp.zeros((1,), dtype=jnp.float32),
    })
    return params


def add_factorized_prior_residual_head_params(
    params, rng, frame_dim, z_dim, hidden_dim, action_dim, residual_clip
):
    keys = jax.random.split(rng, 6)
    scale = 0.02
    params = dict(params)
    params.update({
        "latent_factorized_prior_residual_enabled": jnp.asarray(1.0, dtype=jnp.float32),
        "latent_residual_clip": jnp.asarray(residual_clip, dtype=jnp.float32),
        "prior_dec_w1": scale * jax.random.normal(keys[0], (frame_dim, hidden_dim)),
        "prior_dec_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "prior_dec_w2": scale * jax.random.normal(keys[1], (hidden_dim, hidden_dim)),
        "prior_dec_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "prior_dec_w3": scale * jax.random.normal(keys[2], (hidden_dim, action_dim)),
        "prior_dec_b3": jnp.zeros((action_dim,), dtype=jnp.float32),
        "basis_dec_w1": scale * jax.random.normal(keys[3], (frame_dim, hidden_dim)),
        "basis_dec_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "basis_dec_w2": scale * jax.random.normal(keys[4], (hidden_dim, hidden_dim)),
        "basis_dec_b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "basis_dec_w3": scale * jax.random.normal(keys[5], (hidden_dim, z_dim * action_dim)),
        "basis_dec_b3": jnp.zeros((z_dim * action_dim,), dtype=jnp.float32),
    })
    return params


def attach_prior_anchor(params, prior_params, initialize_matching=True):
    params = dict(params)
    copied = 0
    if initialize_matching:
        for key, value in prior_params.items():
            if key in params and tuple(params[key].shape) == tuple(value.shape):
                params[key] = jnp.asarray(value, dtype=jnp.float32)
                copied += 1
    for key, value in prior_params.items():
        params[f"prior_anchor__{key}"] = jnp.asarray(value, dtype=jnp.float32)
    params["latent_prior_anchor_enabled"] = jnp.asarray(1.0, dtype=jnp.float32)
    params["latent_prior_anchor_copied_count"] = jnp.asarray(copied, dtype=jnp.float32)
    return params


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
    return np.concatenate([np.zeros((history.shape[0], 1), dtype=np.int32), history[:, :-1]], axis=1)


def history_for_mode(arrays, idx, mode):
    obs_hist = np.asarray(arrays["partner_obs_history"][idx], dtype=np.float32)
    act_hist = np.asarray(arrays["partner_action_history"][idx], dtype=np.int32)
    if mode == "action_only":
        obs_hist = np.zeros_like(obs_hist, dtype=np.float32)
    elif mode == "no_history":
        obs_hist = np.zeros_like(obs_hist, dtype=np.float32)
        act_hist = np.zeros_like(act_hist, dtype=np.int32)
    return obs_hist, act_hist


def string_array(data, key, n_rows, fallback="unknown"):
    if key in data.files:
        return np.asarray(data[key]).astype(str)
    return np.full((n_rows,), fallback, dtype=object).astype(str)


def encode_string_labels(labels):
    labels = np.asarray(labels).astype(str)
    vocab = {label: idx for idx, label in enumerate(sorted(set(labels.tolist())))}
    encoded = np.asarray([vocab[label] for label in labels], dtype=np.int32)
    inverse = {idx: label for label, idx in vocab.items()}
    return encoded, vocab, inverse


def build_query_groups(data, n_rows, scope="pair_role", allowed_idx=None):
    allowed = (
        np.arange(n_rows, dtype=np.int32)
        if allowed_idx is None
        else np.asarray(allowed_idx, dtype=np.int32)
    )
    if scope == "none":
        return [np.asarray([i], dtype=np.int32) for i in range(n_rows)], np.arange(n_rows, dtype=np.int32)

    if scope == "partner_policy" and "partner_policy_id" in data.files:
        keys = np.asarray(data["partner_policy_id"]).astype(str)
    elif "pair_label" in data.files and "role" in data.files:
        pair_labels = np.asarray(data["pair_label"]).astype(str)
        roles = np.asarray(data["role"]).astype(str)
        keys = np.char.add(np.char.add(pair_labels, "|role="), roles)
    else:
        return [np.asarray([i], dtype=np.int32) for i in range(n_rows)], np.arange(n_rows, dtype=np.int32)

    group_map = {}
    row_group = np.zeros((n_rows,), dtype=np.int32)
    groups = []
    for idx in allowed:
        key = str(keys[int(idx)])
        group_id = group_map.get(key)
        if group_id is None:
            group_id = len(groups)
            group_map[key] = group_id
            groups.append([])
        groups[group_id].append(int(idx))

    for idx in range(n_rows):
        key = str(keys[idx])
        group_id = group_map.get(key)
        if group_id is None:
            group_id = len(groups)
            groups.append([idx])
        row_group[idx] = group_id
    groups = [np.asarray(x, dtype=np.int32) for x in groups]
    return groups, row_group


def supervised_contrastive_loss(z, labels, temperature):
    z = z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), EPS)
    labels = labels.astype(jnp.int32)
    batch = labels.shape[0]
    eye = jnp.eye(batch, dtype=jnp.bool_)
    positive = (labels[:, None] == labels[None, :]) & (~eye)
    logits = (z @ z.T) / jnp.maximum(jnp.asarray(temperature, dtype=jnp.float32), EPS)
    logits = jnp.where(eye, jnp.asarray(-1e9, dtype=jnp.float32), logits)
    log_prob = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)
    pos_count = jnp.sum(positive.astype(jnp.float32), axis=1)
    anchor_loss = -jnp.sum(positive.astype(jnp.float32) * log_prob, axis=1) / jnp.maximum(pos_count, 1.0)
    valid = pos_count > 0
    loss = jnp.sum(anchor_loss * valid.astype(jnp.float32)) / jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    return loss, jnp.mean(valid.astype(jnp.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--init_decoder_path",
        type=str,
        default="",
        help="Optional latent decoder checkpoint used to initialize all requested parameters.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--frame_dim", type=int, default=128)
    parser.add_argument("--gru_dim", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--arch", choices=("gru", "query_attn"), default="gru")
    parser.add_argument("--attn_conv_channels", type=int, default=32)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_layers", type=int, default=2)
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--vae_beta", type=float, default=1e-3)
    parser.add_argument(
        "--vae_use_query_attn",
        action="store_true",
        help="For query-attn VAE estimators, also run query cross-attention before decoding.",
    )
    parser.add_argument(
        "--query_conditioned_vae",
        action="store_true",
        help=(
            "For query-attn VAE estimators, infer the stochastic latent from the "
            "query-conditioned cross-attention state instead of a mean history summary."
        ),
    )
    parser.add_argument(
        "--decoder_z_norm",
        action="store_true",
        help="Layer-normalize the history latent before concatenating it with query features.",
    )
    parser.add_argument("--decoder_z_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--history_mode", choices=("full", "action_only", "no_history"), default="full")
    parser.add_argument(
        "--query_source",
        choices=("query_obs", "partner_obs"),
        default="query_obs",
        help="Observation field used as the supervised decoder input during training.",
    )
    parser.add_argument(
        "--target_source",
        choices=("target_partner_probs", "partner_action_onehot"),
        default="target_partner_probs",
        help="Supervised action target used during training.",
    )
    parser.add_argument(
        "--target_transform",
        choices=("none", "shuffle", "marginal", "uniform"),
        default="none",
        help=(
            "Ablate the supervised target after loading the dataset. "
            "shuffle preserves target probabilities but breaks query-target pairing; "
            "marginal replaces every row by the dataset mean target; "
            "uniform replaces every row by a uniform action distribution."
        ),
    )
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--compatible_action_head", action="store_true")
    parser.add_argument("--compatible_loss_coef", type=float, default=1.0)
    parser.add_argument("--partner_loss_coef", type=float, default=1.0)
    parser.add_argument(
        "--reward_weight_field",
        choices=("none", "episode_return", "return_to_go", "step_reward"),
        default="none",
        help="Dataset reward field used to weight supervised losses.",
    )
    parser.add_argument(
        "--reward_weight_apply_to",
        choices=("partner", "compatible", "both"),
        default="compatible",
        help="Which supervised loss receives reward weights.",
    )
    parser.add_argument(
        "--reward_weight_transform",
        choices=("linear", "exp_zscore"),
        default="linear",
    )
    parser.add_argument("--reward_weight_min", type=float, default=0.25)
    parser.add_argument("--reward_weight_max", type=float, default=2.0)
    parser.add_argument("--reward_weight_temperature", type=float, default=1.0)
    parser.add_argument(
        "--query_expansion_k",
        type=int,
        default=1,
        help="Number of query observations sampled per history. The first query is the aligned timestep; extra queries are sampled from the same pair/role teacher-policy group.",
    )
    parser.add_argument(
        "--query_expansion_scope",
        choices=("pair_role", "partner_policy", "none"),
        default="pair_role",
        help="Candidate pool for extra/cross query observations.",
    )
    parser.add_argument(
        "--cross_context_fraction",
        type=float,
        default=0.0,
        help="Probability that a sampled query is drawn from another episode with the same partner_policy_id.",
    )
    parser.add_argument("--contrastive_coef", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--partner_balanced_batch", action="store_true")
    parser.add_argument("--partners_per_batch", type=int, default=16)
    parser.add_argument(
        "--query_group_balanced_batch",
        action="store_true",
        help="Sample multiple rows from the same query_group_id so pairwise same-query losses are active.",
    )
    parser.add_argument("--query_groups_per_batch", type=int, default=64)
    parser.add_argument("--rows_per_query_group", type=int, default=8)
    parser.add_argument(
        "--prototype_prior_field",
        type=str,
        default="prototype_prior_probs",
        help="Optional dataset field containing PLASTIC-style prototype prior probabilities.",
    )
    parser.add_argument(
        "--prototype_prior_coef",
        type=float,
        default=0.0,
        help="Weight for KL(predicted partner policy || prototype prior). Default keeps old behavior.",
    )
    parser.add_argument(
        "--residual_gated",
        action="store_true",
        help="Use query-only prior plus history-conditioned residual/gate decoder.",
    )
    parser.add_argument(
        "--factorized_prior_residual",
        action="store_true",
        help=(
            "Use a stricter prior+residual decoder: query_obs learns a prior and "
            "query-dependent action bases, while centered history z supplies the "
            "only nonzero partner-specific residual."
        ),
    )
    parser.add_argument("--prior_loss_coef", type=float, default=0.5)
    parser.add_argument("--pairwise_tv_coef", type=float, default=0.2)
    parser.add_argument("--low_disagreement_residual_coef", type=float, default=0.05)
    parser.add_argument("--low_disagreement_threshold", type=float, default=0.10)
    parser.add_argument("--residual_clip", type=float, default=2.0)
    parser.add_argument("--residual_logit_coef", type=float, default=0.0)
    parser.add_argument(
        "--prior_decoder_path",
        type=str,
        default="",
        help="Optional frozen latent decoder used as the residual-gated prior anchor.",
    )
    parser.add_argument(
        "--no_init_from_prior",
        action="store_true",
        help="Do not initialize matching residual model encoder/VAE parameters from --prior_decoder_path.",
    )
    parser.add_argument("--gate_target_coef", type=float, default=0.0)
    parser.add_argument("--gate_target_threshold", type=float, default=0.10)
    parser.add_argument("--gate_target_scale", type=float, default=12.0)
    parser.add_argument(
        "--residual_tv_coef",
        type=float,
        default=0.0,
        help="Global penalty on TV(final policy, prior policy), in addition to the low-disagreement penalty.",
    )
    args = parser.parse_args()
    residual_model = bool(args.residual_gated or args.factorized_prior_residual)
    if args.residual_gated and args.factorized_prior_residual:
        raise ValueError("choose only one of --residual_gated and --factorized_prior_residual")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files if k != "pair_label"}
    if args.query_source == "partner_obs":
        if "partner_obs" not in arrays:
            raise ValueError("--query_source partner_obs requires dataset field 'partner_obs'")
        arrays["query_obs"] = arrays["partner_obs"]
    if args.target_source == "partner_action_onehot":
        if "partner_action" not in arrays:
            raise ValueError("--target_source partner_action_onehot requires dataset field 'partner_action'")
        actions = np.asarray(arrays["partner_action"], dtype=np.int32)
        if np.any(actions < 0) or np.any(actions >= args.action_dim):
            raise ValueError("partner_action contains out-of-range actions")
        arrays["target_partner_probs"] = np.eye(args.action_dim, dtype=np.float32)[actions]
    if args.target_transform != "none":
        target = normalize_probs(arrays["target_partner_probs"])
        if args.target_transform == "shuffle":
            perm = np.random.default_rng(args.seed + 7919).permutation(target.shape[0])
            target = target[perm]
        elif args.target_transform == "marginal":
            mean_target = normalize_probs(target.mean(axis=0, keepdims=True))
            target = np.repeat(mean_target, target.shape[0], axis=0)
        elif args.target_transform == "uniform":
            target = np.full_like(target, 1.0 / float(target.shape[-1]), dtype=np.float32)
        arrays["target_partner_probs"] = target.astype(np.float32)
    required = [
        "query_obs",
        "partner_obs_history",
        "partner_action_history",
        "ego_action_history",
        "target_partner_probs",
        "ego_action",
        "episode_id",
    ]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ValueError(f"dataset missing fields: {missing}")
    if residual_model:
        if args.arch != "query_attn":
            raise ValueError("residual models currently require --arch query_attn")
        if not args.vae:
            raise ValueError("residual models currently require --vae")
        residual_required = [
            "mean_teacher_probs",
            "query_disagreement_tv",
            "query_group_id",
        ]
        residual_missing = [k for k in residual_required if k not in arrays]
        if residual_missing:
            raise ValueError(
                f"--residual_gated requires grouped dataset fields: {residual_missing}"
            )
    n_rows = arrays["query_obs"].shape[0]
    partner_policy_labels = string_array(data, "partner_policy_id", n_rows)
    partner_policy_codes, partner_vocab, _partner_inverse = encode_string_labels(partner_policy_labels)
    partner_policy_codes_jax = jnp.asarray(partner_policy_codes, dtype=jnp.int32)
    query_expansion_k = max(1, int(args.query_expansion_k))
    cross_context_fraction = min(max(float(args.cross_context_fraction), 0.0), 1.0)
    if args.reward_weight_field == "none":
        reward_weights = np.ones((arrays["query_obs"].shape[0],), dtype=np.float32)
    else:
        if args.reward_weight_field not in arrays:
            raise ValueError(
                f"dataset missing reward field {args.reward_weight_field!r}; "
                "recollect with a reward-aware collector first"
            )
        reward_weights = make_reward_weights(
            arrays[args.reward_weight_field],
            transform=args.reward_weight_transform,
            min_weight=args.reward_weight_min,
            max_weight=args.reward_weight_max,
            temperature=args.reward_weight_temperature,
        )
    prototype_prior_probs = None
    if float(args.prototype_prior_coef) > 0.0:
        if args.prototype_prior_field not in arrays:
            raise ValueError(
                f"dataset missing prototype prior field {args.prototype_prior_field!r}; "
                "build a prototype-prior dataset first or set --prototype_prior_coef 0"
            )
        prototype_prior_probs = normalize_probs(arrays[args.prototype_prior_field])
    mean_teacher_probs = (
        normalize_probs(arrays["mean_teacher_probs"])
        if "mean_teacher_probs" in arrays
        else normalize_probs(arrays["target_partner_probs"])
    )
    query_disagreement_tv = (
        np.asarray(arrays["query_disagreement_tv"], dtype=np.float32)
        if "query_disagreement_tv" in arrays
        else np.zeros((n_rows,), dtype=np.float32)
    )
    query_group_ids = (
        np.asarray(arrays["query_group_id"], dtype=np.int32)
        if "query_group_id" in arrays
        else np.arange(n_rows, dtype=np.int32)
    )

    episode_ids = np.asarray(arrays["episode_id"])
    unique_eps = np.unique(episode_ids)
    rng_np = np.random.default_rng(args.seed)
    rng_np.shuffle(unique_eps)
    val_eps = set(unique_eps[: max(1, int(len(unique_eps) * args.val_fraction))].tolist())
    val_mask = np.isin(episode_ids, list(val_eps))
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("empty train/val split")
    query_groups, row_group = build_query_groups(
        data,
        arrays["query_obs"].shape[0],
        scope=args.query_expansion_scope,
        allowed_idx=train_idx,
    )
    partner_to_train_idx = {
        code: train_idx[partner_policy_codes[train_idx] == code]
        for code in sorted(set(partner_policy_codes[train_idx].tolist()))
    }
    partner_to_train_idx = {
        code: idxs for code, idxs in partner_to_train_idx.items() if len(idxs) > 0
    }
    train_partner_codes = np.asarray(sorted(partner_to_train_idx.keys()), dtype=np.int32)
    group_to_train_idx = {}
    if "query_group_id" in arrays:
        for group_id in sorted(set(query_group_ids[train_idx].tolist())):
            idxs = train_idx[query_group_ids[train_idx] == group_id]
            if len(idxs) > 0:
                group_to_train_idx[int(group_id)] = idxs
    train_group_ids = np.asarray(sorted(group_to_train_idx.keys()), dtype=np.int32)

    mean, std = obs_stats(arrays, seed=args.seed)
    if args.arch == "query_attn":
        params = init_query_attn_params(
            jax.random.PRNGKey(args.seed),
            arrays["query_obs"].shape[1:],
            args.action_dim,
            args.frame_dim,
            args.hidden_dim,
            mean,
            std,
            arrays["partner_action_history"].shape[1],
            args.attn_conv_channels,
            args.attn_heads,
            args.attn_layers,
            args.z_dim,
            args.vae,
        )
    else:
        params = init_params(
            jax.random.PRNGKey(args.seed),
            mean.shape[0],
            args.action_dim,
            args.frame_dim,
            args.gru_dim,
            args.z_dim,
            args.hidden_dim,
            mean,
            std,
            args.vae,
        )
    if args.compatible_action_head:
        if args.arch == "query_attn":
            dec_input_dim = args.frame_dim + args.z_dim if args.vae else args.frame_dim
        else:
            dec_input_dim = args.frame_dim + args.z_dim
        params = add_compatible_head_params(
            params,
            jax.random.PRNGKey(args.seed + 9917),
            dec_input_dim + args.action_dim,
            args.hidden_dim,
            args.action_dim,
        )
    if args.vae and args.arch == "query_attn" and args.vae_use_query_attn:
        params["latent_vae_use_query_attn"] = jnp.asarray(1.0, dtype=jnp.float32)
    if args.vae and args.arch == "query_attn" and args.query_conditioned_vae:
        params["latent_query_conditioned_vae"] = jnp.asarray(1.0, dtype=jnp.float32)
    if args.decoder_z_norm:
        params["latent_decoder_z_norm"] = jnp.asarray(1.0, dtype=jnp.float32)
        params["latent_decoder_z_scale"] = jnp.asarray(args.decoder_z_scale, dtype=jnp.float32)
    if args.init_decoder_path:
        init_params_loaded = load_latent_decoder_npz(Path(args.init_decoder_path))
        missing = sorted(set(params.keys()) - set(init_params_loaded.keys()))
        extra = sorted(set(init_params_loaded.keys()) - set(params.keys()))
        shape_mismatch = sorted(
            key
            for key in set(params.keys()) & set(init_params_loaded.keys())
            if tuple(params[key].shape) != tuple(init_params_loaded[key].shape)
        )
        if missing or extra or shape_mismatch:
            raise ValueError(
                "init decoder is not shape-compatible with requested architecture: "
                f"missing={missing[:8]} extra={extra[:8]} shape_mismatch={shape_mismatch[:8]}"
            )
        params = {key: jnp.asarray(init_params_loaded[key]) for key in params.keys()}
    if args.residual_gated:
        params = add_residual_gated_head_params(
            params,
            jax.random.PRNGKey(args.seed + 19937),
            args.frame_dim,
            args.z_dim,
            args.hidden_dim,
            args.action_dim,
            args.residual_clip,
        )
        if args.prior_decoder_path:
            prior_params = load_latent_decoder_npz(Path(args.prior_decoder_path))
            params = attach_prior_anchor(
                params,
                prior_params,
                initialize_matching=not args.no_init_from_prior,
            )
    if args.factorized_prior_residual:
        params = add_factorized_prior_residual_head_params(
            params,
            jax.random.PRNGKey(args.seed + 29927),
            args.frame_dim,
            args.z_dim,
            args.hidden_dim,
            args.action_dim,
            args.residual_clip,
        )

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, batch, rng):
        (
            query_obs,
            obs_hist,
            act_hist,
            target_probs,
            ego_action,
            sample_weight,
            policy_codes,
            prototype_prior,
            mean_teacher,
            disagreement_tv,
            group_ids,
        ) = batch

        def loss_fn(p):
            result = apply_latent_partner_decoder(
                p,
                query_obs,
                obs_hist,
                act_hist,
                args.action_dim,
                rng=rng,
                deterministic=not args.vae,
                return_aux=args.vae,
            )
            if args.vae:
                logits, vae_aux = result
                posterior_kl = vae_aux["vae_kl"].mean()
                z_mu_abs = jnp.abs(vae_aux["vae_mu"]).mean()
                z_std = jnp.exp(0.5 * vae_aux["vae_logvar"]).mean()
                contrast_loss, contrast_valid = supervised_contrastive_loss(
                    vae_aux["vae_mu"],
                    policy_codes,
                    args.contrastive_temperature,
                )
            else:
                logits = result
                posterior_kl = jnp.asarray(0.0, dtype=jnp.float32)
                z_mu_abs = jnp.asarray(0.0, dtype=jnp.float32)
                z_std = jnp.asarray(0.0, dtype=jnp.float32)
                contrast_loss = jnp.asarray(0.0, dtype=jnp.float32)
                contrast_valid = jnp.asarray(0.0, dtype=jnp.float32)
            target = target_probs / jnp.maximum(target_probs.sum(axis=-1, keepdims=True), EPS)
            mean_teacher_batch = mean_teacher / jnp.maximum(
                mean_teacher.sum(axis=-1, keepdims=True), EPS
            )
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            ce = -jnp.sum(jax.lax.stop_gradient(target) * log_probs, axis=-1)
            partner_loss = (
                weighted_mean(ce, sample_weight)
                if args.reward_weight_apply_to in ("partner", "both")
                else ce.mean()
            )
            kl = categorical_kl(target, logits)
            tv = categorical_tv(target, logits)
            teacher_argmax = jnp.argmax(target, axis=-1)
            acc = (jnp.argmax(logits, axis=-1) == teacher_argmax).mean()
            top2 = jnp.argsort(logits, axis=-1)[:, -2:]
            top2_acc = jnp.any(top2 == teacher_argmax[:, None], axis=-1).mean()
            pred_probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.sum(pred_probs * log_probs, axis=-1).mean()
            prior_loss = jnp.asarray(0.0, dtype=jnp.float32)
            pairwise_tv_loss = jnp.asarray(0.0, dtype=jnp.float32)
            low_residual_loss = jnp.asarray(0.0, dtype=jnp.float32)
            residual_tv_loss = jnp.asarray(0.0, dtype=jnp.float32)
            gate_target_loss = jnp.asarray(0.0, dtype=jnp.float32)
            residual_logit_loss = jnp.asarray(0.0, dtype=jnp.float32)
            gate_mean = jnp.asarray(0.0, dtype=jnp.float32)
            residual_tv = jnp.asarray(0.0, dtype=jnp.float32)
            if residual_model:
                prior_logits = vae_aux["prior_logits"]
                prior_probs = jax.nn.softmax(prior_logits, axis=-1)
                prior_loss = jnp.sum(
                    mean_teacher_batch
                    * (
                        jnp.log(jnp.maximum(mean_teacher_batch, EPS))
                        - jax.nn.log_softmax(prior_logits, axis=-1)
                    ),
                    axis=-1,
                ).mean()
                same_group = group_ids[:, None] == group_ids[None, :]
                not_self = ~jnp.eye(group_ids.shape[0], dtype=jnp.bool_)
                pair_mask = same_group & not_self
                target_pair_tv = 0.5 * jnp.sum(
                    jnp.abs(target[:, None, :] - target[None, :, :]), axis=-1
                )
                pred_pair_tv = 0.5 * jnp.sum(
                    jnp.abs(pred_probs[:, None, :] - pred_probs[None, :, :]), axis=-1
                )
                pair_values = jnp.square(pred_pair_tv - jax.lax.stop_gradient(target_pair_tv))
                pairwise_tv_loss = jnp.sum(pair_values * pair_mask.astype(jnp.float32)) / jnp.maximum(
                    jnp.sum(pair_mask.astype(jnp.float32)), 1.0
                )
                low_mask = disagreement_tv < jnp.asarray(args.low_disagreement_threshold, dtype=jnp.float32)
                residual_tv_rows = 0.5 * jnp.sum(jnp.abs(pred_probs - prior_probs), axis=-1)
                residual_tv_loss = jnp.square(residual_tv_rows).mean()
                low_residual_loss = jnp.sum(
                    jnp.square(residual_tv_rows) * low_mask.astype(jnp.float32)
                ) / jnp.maximum(jnp.sum(low_mask.astype(jnp.float32)), 1.0)
                gate_mean = vae_aux["residual_gate"].mean()
                residual_tv = residual_tv_rows.mean()
                target_prior_tv = 0.5 * jnp.sum(
                    jnp.abs(jax.lax.stop_gradient(target) - jax.lax.stop_gradient(prior_probs)),
                    axis=-1,
                )
                gate_target = jax.nn.sigmoid(
                    jnp.asarray(args.gate_target_scale, dtype=jnp.float32)
                    * (
                        target_prior_tv
                        - jnp.asarray(args.gate_target_threshold, dtype=jnp.float32)
                    )
                )
                gate_pred = jnp.squeeze(vae_aux["residual_gate"], axis=-1)
                gate_target_loss = jnp.square(gate_pred - jax.lax.stop_gradient(gate_target)).mean()
                clipped_target_residual = jnp.clip(
                    jnp.log(jnp.maximum(target, EPS))
                    - jnp.log(jnp.maximum(mean_teacher_batch, EPS)),
                    -jnp.asarray(args.residual_clip, dtype=jnp.float32),
                    jnp.asarray(args.residual_clip, dtype=jnp.float32),
                )
                residual_logit_loss = jnp.square(
                    vae_aux["residual_logits"] - jax.lax.stop_gradient(clipped_target_residual)
                ).mean()
            prototype_prior_batch = prototype_prior / jnp.maximum(
                prototype_prior.sum(axis=-1, keepdims=True), EPS
            )
            prototype_prior_loss = jnp.sum(
                pred_probs
                * (
                    jnp.log(jnp.maximum(pred_probs, EPS))
                    - jnp.log(jnp.maximum(jax.lax.stop_gradient(prototype_prior_batch), EPS))
                ),
                axis=-1,
            ).mean()
            compat_ce = jnp.asarray(0.0, dtype=jnp.float32)
            compat_loss = jnp.asarray(0.0, dtype=jnp.float32)
            compat_acc = jnp.asarray(0.0, dtype=jnp.float32)
            if args.compatible_action_head:
                compat_result = apply_latent_compatible_decoder(
                    p,
                    query_obs,
                    obs_hist,
                    act_hist,
                    args.action_dim,
                    rng=rng,
                    deterministic=not args.vae,
                    return_aux=args.vae,
                )
                compat_logits = compat_result[0] if args.vae else compat_result
                compat_log_probs = jax.nn.log_softmax(compat_logits, axis=-1)
                ego_action_i = ego_action.astype(jnp.int32)
                compat_ce = -compat_log_probs[jnp.arange(ego_action_i.shape[0]), ego_action_i]
                compat_acc = (jnp.argmax(compat_logits, axis=-1) == ego_action_i).mean()
                compat_loss = (
                    weighted_mean(compat_ce, sample_weight)
                    if args.reward_weight_apply_to in ("compatible", "both")
                    else compat_ce.mean()
                )
                compat_ce = compat_ce.mean()
            total_loss = (
                args.partner_loss_coef * partner_loss
                + args.compatible_loss_coef * compat_loss
                + args.vae_beta * posterior_kl
                + args.contrastive_coef * contrast_loss
                + args.prototype_prior_coef * prototype_prior_loss
                + args.prior_loss_coef * prior_loss
                + args.pairwise_tv_coef * pairwise_tv_loss
                + args.low_disagreement_residual_coef * low_residual_loss
                + args.residual_tv_coef * residual_tv_loss
                + args.gate_target_coef * gate_target_loss
                + args.residual_logit_coef * residual_logit_loss
            )
            return total_loss, (
                ce.mean(),
                partner_loss,
                kl.mean(),
                tv.mean(),
                acc,
                top2_acc,
                entropy,
                posterior_kl,
                z_mu_abs,
                z_std,
                compat_ce,
                compat_loss,
                compat_acc,
                sample_weight.mean(),
                contrast_loss,
                contrast_valid,
                prototype_prior_loss,
                prior_loss,
                pairwise_tv_loss,
                low_residual_loss,
                residual_tv_loss,
                gate_target_loss,
                residual_logit_loss,
                gate_mean,
                residual_tv,
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads["latent_obs_mean"] = jnp.zeros_like(grads["latent_obs_mean"])
        grads["latent_obs_std"] = jnp.zeros_like(grads["latent_obs_std"])
        for key in list(grads.keys()):
            if key.startswith("prior_anchor__") or key.startswith("latent_prior_anchor"):
                grads[key] = jnp.zeros_like(grads[key])
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    def sample_expanded_queries(hist_idx):
        hist_idx = np.asarray(hist_idx, dtype=np.int32)
        if query_expansion_k <= 1:
            query_idx = hist_idx.copy()
            if cross_context_fraction > 0.0:
                for pos, row_idx in enumerate(hist_idx):
                    if rng_np.random() >= cross_context_fraction:
                        continue
                    candidates = query_groups[int(row_group[int(row_idx)])]
                    candidates = candidates[
                        np.asarray(arrays["episode_id"][candidates]) != arrays["episode_id"][int(row_idx)]
                    ]
                    if len(candidates):
                        query_idx[pos] = rng_np.choice(candidates)
            return hist_idx, query_idx
        query_blocks = [hist_idx]
        for _ in range(query_expansion_k - 1):
            sampled = np.empty_like(hist_idx)
            for pos, row_idx in enumerate(hist_idx):
                candidates = query_groups[int(row_group[int(row_idx)])]
                if cross_context_fraction > 0.0 and rng_np.random() < cross_context_fraction:
                    cross_candidates = candidates[
                        np.asarray(arrays["episode_id"][candidates]) != arrays["episode_id"][int(row_idx)]
                    ]
                    if len(cross_candidates):
                        candidates = cross_candidates
                sampled[pos] = rng_np.choice(candidates)
            query_blocks.append(sampled)
        return np.repeat(hist_idx, query_expansion_k), np.stack(query_blocks, axis=1).reshape(-1)

    def sample_train_batch_indices():
        if args.query_group_balanced_batch and len(train_group_ids) > 0:
            groups_per_batch = min(max(1, int(args.query_groups_per_batch)), len(train_group_ids))
            rows_per_group = max(2, int(args.rows_per_query_group))
            chosen_groups = rng_np.choice(
                train_group_ids,
                size=groups_per_batch,
                replace=len(train_group_ids) < groups_per_batch,
            )
            chunks = []
            for group_id in chosen_groups:
                candidates = group_to_train_idx[int(group_id)]
                chunks.append(
                    rng_np.choice(
                        candidates,
                        size=rows_per_group,
                        replace=len(candidates) < rows_per_group,
                    )
                )
            batch_idx = np.concatenate(chunks, axis=0)
            if len(batch_idx) > args.batch_size:
                batch_idx = batch_idx[: args.batch_size]
            elif len(batch_idx) < args.batch_size:
                extra = rng_np.choice(
                    train_idx,
                    size=args.batch_size - len(batch_idx),
                    replace=len(train_idx) < args.batch_size,
                )
                batch_idx = np.concatenate([batch_idx, extra], axis=0)
            rng_np.shuffle(batch_idx)
            return batch_idx.astype(np.int32)
        if not args.partner_balanced_batch or len(train_partner_codes) == 0:
            return rng_np.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        partners_per_batch = min(max(1, int(args.partners_per_batch)), len(train_partner_codes))
        chosen_partners = rng_np.choice(
            train_partner_codes,
            size=partners_per_batch,
            replace=len(train_partner_codes) < partners_per_batch,
        )
        per_partner = max(2, int(np.ceil(args.batch_size / partners_per_batch)))
        chunks = []
        for code in chosen_partners:
            candidates = partner_to_train_idx[int(code)]
            chunks.append(
                rng_np.choice(
                    candidates,
                    size=per_partner,
                    replace=len(candidates) < per_partner,
                )
            )
        batch_idx = np.concatenate(chunks, axis=0)
        if len(batch_idx) > args.batch_size:
            batch_idx = batch_idx[: args.batch_size]
        elif len(batch_idx) < args.batch_size:
            extra = rng_np.choice(train_idx, size=args.batch_size - len(batch_idx), replace=len(train_idx) < args.batch_size)
            batch_idx = np.concatenate([batch_idx, extra], axis=0)
        rng_np.shuffle(batch_idx)
        return batch_idx.astype(np.int32)

    def make_batch(idx):
        hist_idx, query_idx = sample_expanded_queries(idx)
        obs_hist, act_hist = history_for_mode(arrays, hist_idx, args.history_mode)
        return (
            jnp.asarray(arrays["query_obs"][query_idx], dtype=jnp.float32),
            jnp.asarray(obs_hist, dtype=jnp.float32),
            jnp.asarray(act_hist, dtype=jnp.int32),
            jnp.asarray(normalize_probs(arrays["target_partner_probs"][query_idx]), dtype=jnp.float32),
            jnp.asarray(arrays["ego_action"][query_idx], dtype=jnp.int32),
            jnp.asarray(reward_weights[query_idx], dtype=jnp.float32),
            partner_policy_codes_jax[hist_idx],
            jnp.asarray(
                prototype_prior_probs[query_idx]
                if prototype_prior_probs is not None
                else normalize_probs(arrays["target_partner_probs"][query_idx]),
                dtype=jnp.float32,
            ),
            jnp.asarray(mean_teacher_probs[query_idx], dtype=jnp.float32),
            jnp.asarray(query_disagreement_tv[query_idx], dtype=jnp.float32),
            jnp.asarray(query_group_ids[query_idx], dtype=jnp.int32),
        )

    log_rows = []
    rng_train = jax.random.PRNGKey(args.seed + 7001)
    for step in range(1, args.steps + 1):
        batch_idx = sample_train_batch_indices()
        rng_train, rng_step = jax.random.split(rng_train)
        params, opt_state, loss, aux = train_step(params, opt_state, make_batch(batch_idx), rng_step)
        if step == 1 or step % max(1, args.steps // 30) == 0:
            (
                ce,
                partner_loss,
                kl,
                tv,
                acc,
                top2_acc,
                entropy,
                posterior_kl,
                z_mu_abs,
                z_std,
                compat_ce,
                compat_loss,
                compat_acc,
                reward_weight_mean,
                contrast_loss,
                contrast_valid,
                prototype_prior_loss,
                prior_loss,
                pairwise_tv_loss,
                low_residual_loss,
                residual_tv_loss,
                gate_target_loss,
                residual_logit_loss,
                gate_mean,
                residual_tv,
            ) = [float(x) for x in aux]
            row = {
                "step": step,
                "loss": float(loss),
                "ce": ce,
                "partner_loss": partner_loss,
                "kl": kl,
                "tv": tv,
                "argmax_acc": acc,
                "top2_acc": top2_acc,
                "entropy": entropy,
                "posterior_kl": posterior_kl,
                "z_mu_abs": z_mu_abs,
                "z_std": z_std,
                "compat_ce": compat_ce,
                "compat_loss": compat_loss,
                "compat_acc": compat_acc,
                "reward_weight_mean": reward_weight_mean,
                "contrast_loss": contrast_loss,
                "contrast_valid_fraction": contrast_valid,
                "prototype_prior_loss": prototype_prior_loss,
                "prior_loss": prior_loss,
                "pairwise_tv_loss": pairwise_tv_loss,
                "low_residual_loss": low_residual_loss,
                "residual_tv_loss": residual_tv_loss,
                "gate_target_loss": gate_target_loss,
                "residual_logit_loss": residual_logit_loss,
                "gate_mean": gate_mean,
                "residual_tv": residual_tv,
            }
            log_rows.append(row)
            print(f"[latent_decoder] {row}", flush=True)

    eval_idx = val_idx if len(val_idx) <= 60000 else rng_np.choice(val_idx, size=60000, replace=False)
    query_obs = jnp.asarray(arrays["query_obs"][eval_idx], dtype=jnp.float32)
    target_probs = jnp.asarray(normalize_probs(arrays["target_partner_probs"][eval_idx]), dtype=jnp.float32)
    ego_actions_eval = jnp.asarray(arrays["ego_action"][eval_idx], dtype=jnp.int32)
    true_obs_hist, true_act_hist = history_for_mode(arrays, eval_idx, args.history_mode)
    wrong_obs_hist = true_obs_hist
    wrong_act_hist = np.asarray(arrays["ego_action_history"][eval_idx], dtype=np.int32)
    if args.history_mode == "no_history":
        wrong_obs_hist = np.zeros_like(wrong_obs_hist, dtype=np.float32)
        wrong_act_hist = np.zeros_like(wrong_act_hist, dtype=np.int32)
    elif args.history_mode == "action_only":
        wrong_obs_hist = np.zeros_like(wrong_obs_hist, dtype=np.float32)
    delayed_act_hist = delayed_history(true_act_hist)
    random_act_hist = rng_np.integers(0, args.action_dim, size=true_act_hist.shape, dtype=np.int32)
    histories = {
        "true": (true_obs_hist, true_act_hist),
        "wrong": (wrong_obs_hist, wrong_act_hist),
        "delayed": (true_obs_hist, delayed_act_hist),
        "random": (true_obs_hist, random_act_hist),
    }

    def eval_history(name, obs_hist, act_hist):
        parts = []
        for start in range(0, len(eval_idx), args.eval_batch_size):
            end = min(len(eval_idx), start + args.eval_batch_size)
            result = apply_latent_partner_decoder(
                params,
                query_obs[start:end],
                jnp.asarray(obs_hist[start:end], dtype=jnp.float32),
                jnp.asarray(act_hist[start:end], dtype=jnp.int32),
                args.action_dim,
                deterministic=True,
                return_aux=args.vae,
            )
            if args.vae:
                logits, vae_aux = result
                vae_part = {
                    "posterior_kl": np.asarray(vae_aux["vae_kl"]),
                    "z_mu_abs": np.asarray(jnp.abs(vae_aux["vae_mu"]).mean(axis=-1)),
                    "z_std": np.asarray(jnp.exp(0.5 * vae_aux["vae_logvar"]).mean(axis=-1)),
                }
                if residual_model:
                    prior_logits = vae_aux["prior_logits"]
                    prior_probs = jax.nn.softmax(prior_logits, axis=-1)
                    final_probs = jax.nn.softmax(logits, axis=-1)
                    mean_teacher_part = jnp.asarray(
                        mean_teacher_probs[eval_idx[start:end]], dtype=jnp.float32
                    )
                    mean_teacher_part = mean_teacher_part / jnp.maximum(
                        mean_teacher_part.sum(axis=-1, keepdims=True), EPS
                    )
                    prior_kl = jnp.sum(
                        mean_teacher_part
                        * (
                            jnp.log(jnp.maximum(mean_teacher_part, EPS))
                            - jax.nn.log_softmax(prior_logits, axis=-1)
                        ),
                        axis=-1,
                    )
                    vae_part.update({
                        "prior_kl": np.asarray(prior_kl),
                        "gate_mean": np.asarray(vae_aux["residual_gate"].mean(axis=-1)),
                        "residual_tv": np.asarray(
                            0.5 * jnp.sum(jnp.abs(final_probs - prior_probs), axis=-1)
                        ),
                    })
            else:
                logits = result
                vae_part = {}
            target = target_probs[start:end]
            part = {
                "kl": np.asarray(categorical_kl(target, logits)),
                "tv": np.asarray(categorical_tv(target, logits)),
                "acc": np.asarray(jnp.argmax(logits, axis=-1) == jnp.argmax(target, axis=-1), dtype=np.float32),
                "top2_acc": np.asarray(
                    jnp.any(
                        jnp.argsort(logits, axis=-1)[:, -2:]
                        == jnp.argmax(target, axis=-1)[:, None],
                        axis=-1,
                    ),
                    dtype=np.float32,
                ),
            }
            if args.compatible_action_head:
                compat_result = apply_latent_compatible_decoder(
                    params,
                    query_obs[start:end],
                    jnp.asarray(obs_hist[start:end], dtype=jnp.float32),
                    jnp.asarray(act_hist[start:end], dtype=jnp.int32),
                    args.action_dim,
                    deterministic=True,
                    return_aux=args.vae,
                )
                compat_logits = compat_result[0] if args.vae else compat_result
                compat_log_probs = jax.nn.log_softmax(compat_logits, axis=-1)
                ego_part = ego_actions_eval[start:end]
                part["compat_ce"] = np.asarray(-compat_log_probs[jnp.arange(ego_part.shape[0]), ego_part])
                part["compat_acc"] = np.asarray(jnp.argmax(compat_logits, axis=-1) == ego_part, dtype=np.float32)
            part.update(vae_part)
            parts.append(part)
        kl = np.concatenate([x["kl"] for x in parts])
        tv = np.concatenate([x["tv"] for x in parts])
        acc = np.concatenate([x["acc"] for x in parts])
        top2_acc = np.concatenate([x["top2_acc"] for x in parts])
        result = {
            f"{name}_kl": float(kl.mean()),
            f"{name}_tv": float(tv.mean()),
            f"{name}_argmax_acc": float(acc.mean()),
            f"{name}_top2_acc": float(top2_acc.mean()),
        }
        if args.compatible_action_head:
            result[f"{name}_compat_ce"] = float(np.concatenate([x["compat_ce"] for x in parts]).mean())
            result[f"{name}_compat_acc"] = float(np.concatenate([x["compat_acc"] for x in parts]).mean())
            if args.vae:
                result[f"{name}_posterior_kl"] = float(np.concatenate([x["posterior_kl"] for x in parts]).mean())
                result[f"{name}_z_mu_abs"] = float(np.concatenate([x["z_mu_abs"] for x in parts]).mean())
                result[f"{name}_z_std"] = float(np.concatenate([x["z_std"] for x in parts]).mean())
            if residual_model:
                result[f"{name}_prior_kl"] = float(np.concatenate([x["prior_kl"] for x in parts]).mean())
                result[f"{name}_gate_mean"] = float(np.concatenate([x["gate_mean"] for x in parts]).mean())
                result[f"{name}_residual_tv"] = float(np.concatenate([x["residual_tv"] for x in parts]).mean())
        return result

    def predict_partner_probs(obs_hist, act_hist):
        probs = []
        for start in range(0, len(eval_idx), args.eval_batch_size):
            end = min(len(eval_idx), start + args.eval_batch_size)
            result = apply_latent_partner_decoder(
                params,
                query_obs[start:end],
                jnp.asarray(obs_hist[start:end], dtype=jnp.float32),
                jnp.asarray(act_hist[start:end], dtype=jnp.int32),
                args.action_dim,
                deterministic=True,
                return_aux=args.vae,
            )
            logits = result[0] if args.vae else result
            probs.append(np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float32))
        return np.concatenate(probs, axis=0)

    def same_query_history_sensitivity(max_groups=1000, max_rows_per_group=8):
        if "query_group_id" not in arrays:
            return {}
        group_ids = np.asarray(arrays["query_group_id"])[eval_idx]
        unique_groups, counts = np.unique(group_ids, return_counts=True)
        eligible = unique_groups[counts >= 2]
        if len(eligible) == 0:
            return {
                "same_query_group_count": 0,
                "same_query_pair_count": 0,
                "same_query_output_tv": float("nan"),
                "same_query_target_tv": float("nan"),
                "same_query_output_target_tv_corr": float("nan"),
                "same_query_output_target_tv_ratio": float("nan"),
            }
        if len(eligible) > max_groups:
            eligible = rng_np.choice(eligible, size=max_groups, replace=False)
        pred_probs = predict_partner_probs(true_obs_hist, true_act_hist)
        target_np = np.asarray(target_probs, dtype=np.float32)
        output_tvs = []
        target_tvs = []
        for group_id in eligible:
            local = np.where(group_ids == group_id)[0]
            if len(local) > max_rows_per_group:
                local = rng_np.choice(local, size=max_rows_per_group, replace=False)
            for i in range(len(local)):
                for j in range(i + 1, len(local)):
                    lhs = local[i]
                    rhs = local[j]
                    output_tvs.append(0.5 * np.abs(pred_probs[lhs] - pred_probs[rhs]).sum())
                    target_tvs.append(0.5 * np.abs(target_np[lhs] - target_np[rhs]).sum())
        if not output_tvs:
            return {
                "same_query_group_count": int(len(eligible)),
                "same_query_pair_count": 0,
                "same_query_output_tv": float("nan"),
                "same_query_target_tv": float("nan"),
                "same_query_output_target_tv_corr": float("nan"),
                "same_query_output_target_tv_ratio": float("nan"),
            }
        output_tvs = np.asarray(output_tvs, dtype=np.float32)
        target_tvs = np.asarray(target_tvs, dtype=np.float32)
        if float(np.std(output_tvs)) < EPS or float(np.std(target_tvs)) < EPS:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(output_tvs, target_tvs)[0, 1])
        return {
            "same_query_group_count": int(len(eligible)),
            "same_query_pair_count": int(len(output_tvs)),
            "same_query_output_tv": float(output_tvs.mean()),
            "same_query_target_tv": float(target_tvs.mean()),
            "same_query_output_target_tv_corr": corr,
            "same_query_output_target_tv_ratio": float(output_tvs.mean() / max(float(target_tvs.mean()), EPS)),
        }

    def embedding_retrieval_acc():
        if not args.vae:
            return float("nan")
        embeddings = []
        for start in range(0, len(eval_idx), args.eval_batch_size):
            end = min(len(eval_idx), start + args.eval_batch_size)
            result = apply_latent_partner_decoder(
                params,
                query_obs[start:end],
                jnp.asarray(true_obs_hist[start:end], dtype=jnp.float32),
                jnp.asarray(true_act_hist[start:end], dtype=jnp.int32),
                args.action_dim,
                deterministic=True,
                return_aux=True,
            )
            _logits, vae_aux = result
            embeddings.append(np.asarray(vae_aux["vae_mu"], dtype=np.float32))
        z = np.concatenate(embeddings, axis=0)
        labels = partner_policy_codes[eval_idx]
        if len(np.unique(labels)) < 2 or len(labels) < 4:
            return float("nan")
        ref_mask = (np.arange(len(labels)) % 2) == 0
        query_mask = ~ref_mask
        centroids = {}
        for code in sorted(set(labels[ref_mask].tolist())):
            code_mask = ref_mask & (labels == code)
            if np.any(code_mask):
                centroids[int(code)] = z[code_mask].mean(axis=0)
        if not centroids or not np.any(query_mask):
            return float("nan")
        codes = np.asarray(sorted(centroids.keys()), dtype=np.int32)
        centroid_arr = np.stack([centroids[int(code)] for code in codes], axis=0)
        zq = z[query_mask]
        zq = zq / np.maximum(np.linalg.norm(zq, axis=-1, keepdims=True), EPS)
        centroid_arr = centroid_arr / np.maximum(np.linalg.norm(centroid_arr, axis=-1, keepdims=True), EPS)
        pred = codes[np.argmax(zq @ centroid_arr.T, axis=1)]
        return float(np.mean(pred == labels[query_mask]))

    metrics = {}
    for name, (obs_hist, act_hist) in histories.items():
        metrics.update(eval_history(name, obs_hist, act_hist))
    metrics.update(same_query_history_sensitivity())
    metrics["embedding_retrieval_acc"] = embedding_retrieval_acc()
    metrics["train_transitions"] = int(len(train_idx))
    metrics["val_transitions"] = int(len(eval_idx))
    metrics["partner_policy_count"] = int(len(partner_vocab))
    metrics["init_decoder_path"] = args.init_decoder_path
    metrics["reward_weight_field"] = args.reward_weight_field
    metrics["reward_weight_apply_to"] = args.reward_weight_apply_to
    metrics["reward_weight_mean"] = float(np.mean(reward_weights))
    metrics["reward_weight_min"] = float(np.min(reward_weights))
    metrics["reward_weight_max"] = float(np.max(reward_weights))
    metrics["query_expansion_scope"] = args.query_expansion_scope
    metrics["cross_context_fraction"] = float(cross_context_fraction)
    metrics["contrastive_coef"] = float(args.contrastive_coef)
    metrics["contrastive_temperature"] = float(args.contrastive_temperature)
    metrics["prototype_prior_coef"] = float(args.prototype_prior_coef)
    metrics["prototype_prior_field"] = args.prototype_prior_field
    metrics["vae_use_query_attn"] = bool(args.vae_use_query_attn)
    metrics["query_conditioned_vae"] = bool(args.query_conditioned_vae)
    metrics["decoder_z_norm"] = bool(args.decoder_z_norm)
    metrics["decoder_z_scale"] = float(np.asarray(params.get("latent_decoder_z_scale", args.decoder_z_scale)))
    metrics["residual_gated"] = bool(args.residual_gated)
    metrics["factorized_prior_residual"] = bool(args.factorized_prior_residual)
    metrics["prior_loss_coef"] = float(args.prior_loss_coef)
    metrics["pairwise_tv_coef"] = float(args.pairwise_tv_coef)
    metrics["low_disagreement_residual_coef"] = float(args.low_disagreement_residual_coef)
    metrics["low_disagreement_threshold"] = float(args.low_disagreement_threshold)
    metrics["residual_clip"] = float(args.residual_clip)
    metrics["residual_logit_coef"] = float(args.residual_logit_coef)
    metrics["prior_decoder_path"] = args.prior_decoder_path
    metrics["init_from_prior"] = bool(args.prior_decoder_path and not args.no_init_from_prior)
    metrics["prior_anchor_copied_count"] = float(
        np.asarray(params.get("latent_prior_anchor_copied_count", 0.0))
    )
    metrics["gate_target_coef"] = float(args.gate_target_coef)
    metrics["gate_target_threshold"] = float(args.gate_target_threshold)
    metrics["gate_target_scale"] = float(args.gate_target_scale)
    metrics["residual_tv_coef"] = float(args.residual_tv_coef)
    metrics["partner_balanced_batch"] = bool(args.partner_balanced_batch)
    metrics["partners_per_batch"] = int(args.partners_per_batch)
    metrics["query_group_balanced_batch"] = bool(args.query_group_balanced_batch)
    metrics["query_groups_per_batch"] = int(args.query_groups_per_batch)
    metrics["rows_per_query_group"] = int(args.rows_per_query_group)
    metrics["wrong_kl_minus_true"] = metrics["wrong_kl"] - metrics["true_kl"]
    metrics["delayed_kl_minus_true"] = metrics["delayed_kl"] - metrics["true_kl"]
    metrics["random_kl_minus_true"] = metrics["random_kl"] - metrics["true_kl"]

    estimator_path = out / "latent_partner_decoder.npz"
    save_latent_decoder_npz(estimator_path, params)
    write_csv(out / "training_curve.csv", log_rows)
    write_csv(out / "validation_metrics.csv", [metrics])

    lines = ["# TTAC v5.8 Latent Partner Decoder", ""]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- decoder: `{estimator_path}`")
    lines.append(f"- arch: `{args.arch}`")
    lines.append(f"- history_mode: `{args.history_mode}`")
    lines.append(f"- query_source: `{args.query_source}`")
    lines.append(f"- target_source: `{args.target_source}`")
    lines.append(f"- target_transform: `{args.target_transform}`")
    lines.append(f"- query_expansion_k: `{query_expansion_k}`")
    lines.append(f"- query_expansion_scope: `{args.query_expansion_scope}`")
    lines.append(f"- cross_context_fraction: `{cross_context_fraction}`")
    lines.append(f"- vae: `{args.vae}`")
    lines.append(f"- vae_beta: `{args.vae_beta}`")
    lines.append(f"- vae_use_query_attn: `{args.vae_use_query_attn}`")
    lines.append(f"- query_conditioned_vae: `{args.query_conditioned_vae}`")
    lines.append(f"- decoder_z_norm: `{args.decoder_z_norm}`")
    lines.append(f"- decoder_z_scale_init: `{args.decoder_z_scale}`")
    lines.append(f"- residual_gated: `{args.residual_gated}`")
    lines.append(f"- factorized_prior_residual: `{args.factorized_prior_residual}`")
    lines.append(f"- prior_loss_coef: `{args.prior_loss_coef}`")
    lines.append(f"- pairwise_tv_coef: `{args.pairwise_tv_coef}`")
    lines.append(f"- low_disagreement_residual_coef: `{args.low_disagreement_residual_coef}`")
    lines.append(f"- low_disagreement_threshold: `{args.low_disagreement_threshold}`")
    lines.append(f"- residual_clip: `{args.residual_clip}`")
    lines.append(f"- residual_logit_coef: `{args.residual_logit_coef}`")
    lines.append(f"- prior_decoder_path: `{args.prior_decoder_path}`")
    lines.append(f"- init_from_prior: `{bool(args.prior_decoder_path and not args.no_init_from_prior)}`")
    lines.append(f"- gate_target_coef: `{args.gate_target_coef}`")
    lines.append(f"- gate_target_threshold: `{args.gate_target_threshold}`")
    lines.append(f"- gate_target_scale: `{args.gate_target_scale}`")
    lines.append(f"- residual_tv_coef: `{args.residual_tv_coef}`")
    lines.append(f"- contrastive_coef: `{args.contrastive_coef}`")
    lines.append(f"- contrastive_temperature: `{args.contrastive_temperature}`")
    lines.append(f"- prototype_prior_coef: `{args.prototype_prior_coef}`")
    lines.append(f"- prototype_prior_field: `{args.prototype_prior_field}`")
    lines.append(f"- partner_balanced_batch: `{args.partner_balanced_batch}`")
    lines.append(f"- partners_per_batch: `{args.partners_per_batch}`")
    lines.append(f"- query_group_balanced_batch: `{args.query_group_balanced_batch}`")
    lines.append(f"- query_groups_per_batch: `{args.query_groups_per_batch}`")
    lines.append(f"- rows_per_query_group: `{args.rows_per_query_group}`")
    lines.append(f"- compatible_action_head: `{args.compatible_action_head}`")
    lines.append(f"- compatible_loss_coef: `{args.compatible_loss_coef}`")
    lines.append(f"- partner_loss_coef: `{args.partner_loss_coef}`")
    lines.append(f"- reward_weight_field: `{args.reward_weight_field}`")
    lines.append(f"- reward_weight_apply_to: `{args.reward_weight_apply_to}`")
    lines.append(f"- reward_weight_transform: `{args.reward_weight_transform}`")
    lines.append(f"- reward_weight_min: `{args.reward_weight_min}`")
    lines.append(f"- reward_weight_max: `{args.reward_weight_max}`")
    lines.append(f"- train transitions: `{len(train_idx)}`")
    lines.append(f"- val transitions: `{len(eval_idx)}`")
    lines.append("")
    for key, value in metrics.items():
        lines.append(f"- {key}: `{value}`")
    (out / "latent_partner_decoder_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[latent_decoder] wrote {estimator_path}")
    print(metrics)


if __name__ == "__main__":
    main()
