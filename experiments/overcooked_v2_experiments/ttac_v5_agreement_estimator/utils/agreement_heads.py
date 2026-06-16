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


def normalize(x, mean, std):
    return (x - jax.lax.stop_gradient(mean)) / jnp.maximum(jax.lax.stop_gradient(std), EPS)


def mlp_apply(params, prefix, x):
    h = jnp.tanh(x @ params[f"{prefix}_w1"] + params[f"{prefix}_b1"])
    h = jnp.tanh(h @ params[f"{prefix}_w2"] + params[f"{prefix}_b2"])
    return h @ params[f"{prefix}_w3"] + params[f"{prefix}_b3"]


def apply_agreement_estimator(params, query_obs, partner_obs, partner_action_history, action_dim):
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
