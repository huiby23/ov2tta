from __future__ import annotations

from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-6


def load_surrogate_npz(path):
    data = np.load(Path(path), allow_pickle=False)
    return {key: jnp.asarray(data[key]) for key in data.files}


def save_surrogate_npz(path, params: Mapping[str, np.ndarray | jnp.ndarray]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def one_hot(actions, action_dim):
    return jax.nn.one_hot(actions.astype(jnp.int32), int(action_dim))


def make_q_features(partner_obs, partner_action_history, action_dim):
    obs_flat = jnp.reshape(partner_obs.astype(jnp.float32), (partner_obs.shape[0], -1))
    hist = partner_action_history.astype(jnp.int32)
    if hist.ndim == 1:
        hist = hist[:, None]
    hist_oh = one_hot(hist, action_dim).astype(jnp.float32)
    hist_flat = jnp.reshape(hist_oh, (hist_oh.shape[0], -1))
    return jnp.concatenate([obs_flat, hist_flat], axis=-1)


def make_joint_q_features(ego_obs, partner_obs, ego_action_oh, partner_action_oh):
    ego_flat = jnp.reshape(ego_obs.astype(jnp.float32), (ego_obs.shape[0], -1))
    partner_flat = jnp.reshape(partner_obs.astype(jnp.float32), (partner_obs.shape[0], -1))
    return jnp.concatenate(
        [ego_flat, partner_flat, ego_action_oh.astype(jnp.float32), partner_action_oh.astype(jnp.float32)],
        axis=-1,
    )


def normalize(x, mean, std):
    return (x - jax.lax.stop_gradient(mean)) / jnp.maximum(jax.lax.stop_gradient(std), EPS)


def mlp_apply(params, prefix, x):
    h = jnp.tanh(x @ params[f"{prefix}_w1"] + params[f"{prefix}_b1"])
    return h @ params[f"{prefix}_w2"] + params[f"{prefix}_b2"]


def apply_q_eta(params, partner_obs, partner_action_history, action_dim):
    x = make_q_features(partner_obs, partner_action_history, action_dim)
    x = normalize(x, params["q_mean"], params["q_std"])
    return mlp_apply(params, "q", x)


def apply_joint_q(params, ego_obs, partner_obs, ego_action_oh, partner_action_oh):
    x = make_joint_q_features(ego_obs, partner_obs, ego_action_oh, partner_action_oh)
    x = normalize(x, params["joint_mean"], params["joint_std"])
    return jnp.squeeze(mlp_apply(params, "joint", x), axis=-1)


def joint_q_grid(params, ego_obs, partner_obs, action_dim):
    batch_size = ego_obs.shape[0]
    action_eye = jnp.eye(action_dim, dtype=jnp.float32)
    ego_actions = jnp.repeat(action_eye, action_dim, axis=0)
    partner_actions = jnp.tile(action_eye, (action_dim, 1))
    num_joint = action_dim * action_dim
    ego_rep = jnp.repeat(ego_obs, num_joint, axis=0)
    partner_rep = jnp.repeat(partner_obs, num_joint, axis=0)
    ego_action_rep = jnp.tile(ego_actions, (batch_size, 1))
    partner_action_rep = jnp.tile(partner_actions, (batch_size, 1))
    q = apply_joint_q(params, ego_rep, partner_rep, ego_action_rep, partner_action_rep)
    return jnp.reshape(q, (batch_size, action_dim, action_dim))


def expected_joint_q(params, ego_obs, partner_obs, ego_logits, partner_action_probs, action_dim):
    ego_probs = jax.nn.softmax(ego_logits, axis=-1)
    q_grid = joint_q_grid(params, ego_obs, partner_obs, action_dim)
    return jnp.einsum("ba,bp,bap->b", ego_probs, partner_action_probs, q_grid)
