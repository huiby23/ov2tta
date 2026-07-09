from __future__ import annotations

from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-6


def load_agreement_npz(path):
    data = np.load(Path(path), allow_pickle=False)
    return {key: jnp.asarray(data[key]) for key in data.files}


def save_agreement_npz(path, params: Mapping[str, np.ndarray | jnp.ndarray]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def one_hot(actions, action_dim):
    return jax.nn.one_hot(actions.astype(jnp.int32), int(action_dim))


def make_agreement_features(query_obs, partner_obs, partner_action_history, action_dim):
    query_flat = jnp.reshape(query_obs.astype(jnp.float32), (query_obs.shape[0], -1))
    partner_flat = jnp.reshape(partner_obs.astype(jnp.float32), (partner_obs.shape[0], -1))
    hist = partner_action_history.astype(jnp.int32)
    if hist.ndim == 1:
        hist = hist[:, None]
    hist_oh = one_hot(hist, action_dim).astype(jnp.float32)
    hist_flat = jnp.reshape(hist_oh, (hist_oh.shape[0], -1))
    return jnp.concatenate([query_flat, partner_flat, hist_flat], axis=-1)


def make_temporal_obs_stats(*obs_arrays):
    flat_parts = []
    for obs in obs_arrays:
        obs = jnp.asarray(obs, dtype=jnp.float32)
        flat_parts.append(jnp.reshape(obs, (-1, obs.shape[-1] * obs.shape[-2] * obs.shape[-3])))
    flat = jnp.concatenate(flat_parts, axis=0)
    return flat.mean(axis=0), flat.std(axis=0) + 1e-3


def normalize(x, mean, std):
    return (x - jax.lax.stop_gradient(mean)) / jnp.maximum(jax.lax.stop_gradient(std), EPS)


def mlp_apply(params, prefix, x):
    h = jnp.tanh(x @ params[f"{prefix}_w1"] + params[f"{prefix}_b1"])
    h = jnp.tanh(h @ params[f"{prefix}_w2"] + params[f"{prefix}_b2"])
    return h @ params[f"{prefix}_w3"] + params[f"{prefix}_b3"]


def _frame_encode(params, obs):
    obs = jnp.asarray(obs, dtype=jnp.float32)
    flat = jnp.reshape(obs, (-1, obs.shape[-3] * obs.shape[-2] * obs.shape[-1]))
    flat = normalize(flat, params["temporal_obs_mean"], params["temporal_obs_std"])
    return jnp.tanh(flat @ params["temporal_frame_w"] + params["temporal_frame_b"])


def _gru_encode(params, seq):
    batch_size = seq.shape[0]
    h0 = jnp.zeros((batch_size, params["temporal_gru_bz"].shape[0]), dtype=jnp.float32)

    def step(h, x):
        z = jax.nn.sigmoid(x @ params["temporal_gru_wz"] + h @ params["temporal_gru_uz"] + params["temporal_gru_bz"])
        r = jax.nn.sigmoid(x @ params["temporal_gru_wr"] + h @ params["temporal_gru_ur"] + params["temporal_gru_br"])
        h_tilde = jnp.tanh(x @ params["temporal_gru_wh"] + (r * h) @ params["temporal_gru_uh"] + params["temporal_gru_bh"])
        h_next = (1.0 - z) * h + z * h_tilde
        return h_next, h_next

    h_final, _ = jax.lax.scan(step, h0, jnp.swapaxes(seq, 0, 1))
    return h_final


def apply_temporal_agreement_estimator(
    params,
    query_obs,
    partner_obs,
    partner_action_history,
    partner_obs_history,
    action_dim,
):
    query_obs = jnp.asarray(query_obs, dtype=jnp.float32)
    partner_obs = jnp.asarray(partner_obs, dtype=jnp.float32)
    partner_obs_history = jnp.asarray(partner_obs_history, dtype=jnp.float32)
    partner_action_history = jnp.asarray(partner_action_history, dtype=jnp.int32)
    if partner_action_history.ndim == 1:
        partner_action_history = partner_action_history[:, None]
    if partner_obs_history.ndim == query_obs.ndim:
        partner_obs_history = partner_obs_history[:, None, ...]

    batch_size, history_len = partner_action_history.shape[:2]
    query_emb = _frame_encode(params, query_obs)
    partner_query_emb = _frame_encode(params, partner_obs)
    hist_obs_flat = jnp.reshape(partner_obs_history, (batch_size * history_len,) + partner_obs_history.shape[2:])
    hist_obs_emb = jnp.reshape(_frame_encode(params, hist_obs_flat), (batch_size, history_len, -1))
    hist_action_oh = one_hot(partner_action_history, action_dim).astype(jnp.float32)
    hist_action_emb = hist_action_oh @ params["temporal_action_w"] + params["temporal_action_b"]
    hist_input = jnp.tanh(hist_obs_emb + hist_action_emb + params["temporal_input_b"])
    hist_emb = _gru_encode(params, hist_input)
    x = jnp.concatenate([query_emb, partner_query_emb, hist_emb], axis=-1)
    h = jnp.tanh(x @ params["temporal_head_w1"] + params["temporal_head_b1"])
    h = jnp.tanh(h @ params["temporal_head_w2"] + params["temporal_head_b2"])
    return h @ params["temporal_head_w3"] + params["temporal_head_b3"]


def apply_agreement_estimator(
    params,
    query_obs,
    partner_obs,
    partner_action_history,
    action_dim,
    partner_obs_history=None,
):
    if "temporal_frame_w" in params:
        if partner_obs_history is None:
            partner_obs_history = jnp.repeat(partner_obs[:, None, ...], partner_action_history.shape[1], axis=1)
        return apply_temporal_agreement_estimator(
            params,
            query_obs,
            partner_obs,
            partner_action_history,
            partner_obs_history,
            action_dim,
        )
    x = make_agreement_features(query_obs, partner_obs, partner_action_history, action_dim)
    x = normalize(x, params["agreement_mean"], params["agreement_std"])
    return mlp_apply(params, "agreement", x)


def categorical_kl(target_probs, pred_logits):
    target_probs = jnp.asarray(target_probs, dtype=jnp.float32)
    target_probs = target_probs / jnp.maximum(target_probs.sum(axis=-1, keepdims=True), EPS)
    target_log_probs = jnp.log(jnp.maximum(target_probs, EPS))
    pred_log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
    return jnp.sum(target_probs * (target_log_probs - pred_log_probs), axis=-1)


def categorical_tv(target_probs, pred_logits):
    target_probs = jnp.asarray(target_probs, dtype=jnp.float32)
    target_probs = target_probs / jnp.maximum(target_probs.sum(axis=-1, keepdims=True), EPS)
    pred_probs = jax.nn.softmax(pred_logits, axis=-1)
    return 0.5 * jnp.sum(jnp.abs(target_probs - pred_probs), axis=-1)
