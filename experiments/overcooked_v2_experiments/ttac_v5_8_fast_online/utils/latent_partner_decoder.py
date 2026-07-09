from __future__ import annotations

from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-6


def load_latent_decoder_npz(path):
    data = np.load(Path(path), allow_pickle=False)
    return {key: jnp.asarray(data[key]) for key in data.files}


def save_latent_decoder_npz(path, params: Mapping[str, np.ndarray | jnp.ndarray]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def prefixed_params(params, prefix):
    marker = f"{prefix}__"
    return {
        key[len(marker):]: value
        for key, value in params.items()
        if key.startswith(marker)
    }


def one_hot(actions, action_dim):
    return jax.nn.one_hot(actions.astype(jnp.int32), int(action_dim))


def normalize(x, mean, std):
    return (x - jax.lax.stop_gradient(mean)) / jnp.maximum(jax.lax.stop_gradient(std), EPS)


def _frame_encode(params, obs):
    obs = jnp.asarray(obs, dtype=jnp.float32)
    flat = jnp.reshape(obs, (-1, obs.shape[-3] * obs.shape[-2] * obs.shape[-1]))
    flat = normalize(flat, params["latent_obs_mean"], params["latent_obs_std"])
    return jnp.tanh(flat @ params["latent_frame_w"] + params["latent_frame_b"])


def _attn_cnn_frame_encode(params, obs):
    obs = jnp.asarray(obs, dtype=jnp.float32)
    obs_shape = obs.shape[-3:]
    x = jnp.reshape(obs, (-1,) + obs_shape)
    flat = jnp.reshape(x, (x.shape[0], obs_shape[0] * obs_shape[1] * obs_shape[2]))
    flat = normalize(flat, params["latent_obs_mean"], params["latent_obs_std"])
    x = jnp.reshape(flat, x.shape)
    dim_nums = ("NHWC", "HWIO", "NHWC")
    x = jax.lax.conv_general_dilated(
        x,
        params["attn_conv1_w"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=dim_nums,
    )
    x = jnp.tanh(x + params["attn_conv1_b"])
    x = jax.lax.conv_general_dilated(
        x,
        params["attn_conv2_w"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=dim_nums,
    )
    x = jnp.tanh(x + params["attn_conv2_b"])
    x = jnp.reshape(x, (x.shape[0], -1))
    return jnp.tanh(x @ params["attn_frame_w"] + params["attn_frame_b"])


def _layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def _gru_encode(params, seq):
    batch_size = seq.shape[0]
    h = jnp.zeros((batch_size, params["latent_gru_bz"].shape[0]), dtype=jnp.float32)

    def step(carry, x):
        z = jax.nn.sigmoid(x @ params["latent_gru_wz"] + carry @ params["latent_gru_uz"] + params["latent_gru_bz"])
        r = jax.nn.sigmoid(x @ params["latent_gru_wr"] + carry @ params["latent_gru_ur"] + params["latent_gru_br"])
        h_tilde = jnp.tanh(x @ params["latent_gru_wh"] + (r * carry) @ params["latent_gru_uh"] + params["latent_gru_bh"])
        next_h = (1.0 - z) * carry + z * h_tilde
        return next_h, next_h

    h, _ = jax.lax.scan(step, h, jnp.swapaxes(seq, 0, 1))
    return h


def _diag_gaussian_kl(mu, logvar):
    logvar = jnp.clip(logvar, -8.0, 8.0)
    return 0.5 * jnp.sum(jnp.exp(logvar) + jnp.square(mu) - 1.0 - logvar, axis=-1)


def _vae_latent_from_hidden(params, hidden, rng=None, deterministic=True):
    mu = hidden @ params["vae_mu_w"] + params["vae_mu_b"]
    logvar = jnp.clip(hidden @ params["vae_logvar_w"] + params["vae_logvar_b"], -8.0, 8.0)
    if deterministic or rng is None:
        z = mu
    else:
        eps = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
        z = mu + jnp.exp(0.5 * logvar) * eps
    return z, {
        "vae_mu": mu,
        "vae_logvar": logvar,
        "vae_kl": _diag_gaussian_kl(mu, logvar),
    }


def _decoder_latent(params, z):
    if "latent_decoder_z_norm" in params:
        z = _layer_norm(z)
    if "latent_decoder_z_scale" in params:
        z = z * params["latent_decoder_z_scale"]
    return z


def _mlp3(x, params, prefix, normalize_input=False):
    if normalize_input:
        x = _layer_norm(x)
    h = jnp.tanh(x @ params[f"{prefix}_w1"] + params[f"{prefix}_b1"])
    h = jnp.tanh(h @ params[f"{prefix}_w2"] + params[f"{prefix}_b2"])
    return h @ params[f"{prefix}_w3"] + params[f"{prefix}_b3"]


def _lora_residual(params, q, z, action_dim):
    qn = _layer_norm(q)
    zn = _layer_norm(z)
    rank = params["lora_a"].shape[0]
    q_low = qn @ params["lora_b"]
    coeff_h = jnp.tanh(zn @ params["lora_coeff_w1"] + params["lora_coeff_b1"])
    coeff = jnp.tanh(coeff_h @ params["lora_coeff_w2"] + params["lora_coeff_b2"])
    residual = (q_low * coeff) @ params["lora_a"]
    residual = residual + params.get("lora_bias", jnp.zeros((int(action_dim),), dtype=residual.dtype))
    if "lora_alpha" in params:
        residual_scale = jnp.asarray(params["lora_alpha"], dtype=jnp.float32)
        residual = residual * residual_scale / jnp.asarray(rank, dtype=jnp.float32)
    else:
        residual_scale = jnp.asarray(params.get("lora_residual_scale", 1.0), dtype=jnp.float32)
        residual = residual * residual_scale / jnp.sqrt(jnp.asarray(rank, dtype=jnp.float32))
    residual_clip = jnp.asarray(params.get("lora_residual_clip", 2.0), dtype=jnp.float32)
    residual = jnp.clip(residual, -residual_clip, residual_clip)
    if "lora_no_gate" in params:
        gate = jnp.ones((residual.shape[0], 1), dtype=residual.dtype)
    else:
        gate_input = jnp.concatenate([qn, zn], axis=-1)
        gate = jax.nn.sigmoid(gate_input @ params["lora_gate_w"] + params["lora_gate_b"])
    return gate * residual, {
        "lora_residual_logits": residual,
        "lora_gate": gate,
        "lora_coeff": coeff,
    }


def _attn_base_logits_with_z(params, q, z, action_dim):
    dec_input = jnp.concatenate([q, z], axis=-1)
    h = jnp.tanh(_layer_norm(dec_input) @ params["attn_dec_w1"] + params["attn_dec_b1"])
    h = jnp.tanh(h @ params["attn_dec_w2"] + params["attn_dec_b2"])
    logits = h @ params["attn_dec_w3"] + params["attn_dec_b3"]
    if "latent_lora_residual_enabled" in params:
        lora_delta, _lora_aux = _lora_residual(params, q, z, action_dim)
        logits = logits + lora_delta
    return logits


def _online_residual_c_split(params, latent_z):
    base_z_dim = int(params["vae_mu_b"].shape[0])
    z0 = latent_z[:, :base_z_dim]
    c = latent_z[:, base_z_dim:]
    return z0, c


def _online_residual_c_delta(params, q, z0, c, action_dim):
    qz = _layer_norm(jnp.concatenate([q, z0], axis=-1))
    h = jnp.tanh(qz @ params["orc_feat_w"] + params["orc_feat_b"])
    basis_flat = h @ params["orc_basis_w"] + params["orc_basis_b"]
    c_dim = c.shape[-1]
    basis = jnp.reshape(basis_flat, (q.shape[0], c_dim, int(action_dim)))
    residual = jnp.einsum("bc,bca->ba", c, basis) / jnp.sqrt(
        jnp.asarray(jnp.maximum(c_dim, 1), dtype=jnp.float32)
    )
    residual_clip = jnp.asarray(
        params.get("online_residual_c_clip", 2.0), dtype=jnp.float32
    )
    return jnp.clip(residual, -residual_clip, residual_clip)


def encode_partner_history_embedding(params, partner_obs_history, partner_action_history, action_dim):
    partner_obs_history = jnp.asarray(partner_obs_history, dtype=jnp.float32)
    partner_action_history = jnp.asarray(partner_action_history, dtype=jnp.int32)
    if partner_action_history.ndim == 1:
        partner_action_history = partner_action_history[:, None]
    if partner_obs_history.ndim == 4:
        partner_obs_history = partner_obs_history[:, None, ...]

    batch_size, history_len = partner_action_history.shape[:2]
    hist_obs_flat = jnp.reshape(partner_obs_history, (batch_size * history_len,) + partner_obs_history.shape[2:])
    hist_obs_emb = jnp.reshape(_frame_encode(params, hist_obs_flat), (batch_size, history_len, -1))
    hist_action_oh = one_hot(partner_action_history, action_dim).astype(jnp.float32)
    hist_action_emb = hist_action_oh @ params["latent_action_w"] + params["latent_action_b"]
    hist_input = jnp.tanh(hist_obs_emb + hist_action_emb + params["latent_input_b"])
    return _gru_encode(params, hist_input)


def encode_partner_latent(params, partner_obs_history, partner_action_history, action_dim):
    hist_emb = encode_partner_history_embedding(
        params, partner_obs_history, partner_action_history, action_dim
    )
    return jnp.tanh(hist_emb @ params["latent_z_w"] + params["latent_z_b"])


def _query_cross_attention_layer(params, q, tokens, valid, prefix):
    num_heads = params[f"{prefix}_wq"].shape[1]
    head_dim = params[f"{prefix}_wq"].shape[2]
    qn = _layer_norm(q)
    tn = _layer_norm(tokens)
    qh = jnp.einsum("bd,dhm->bhm", qn, params[f"{prefix}_wq"])
    kh = jnp.einsum("bld,dhm->blhm", tn, params[f"{prefix}_wk"])
    vh = jnp.einsum("bld,dhm->blhm", tn, params[f"{prefix}_wv"])
    kh = jnp.swapaxes(kh, 1, 2)
    vh = jnp.swapaxes(vh, 1, 2)
    scores = jnp.einsum("bhd,bhld->bhl", qh, kh) / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
    scores = jnp.where(valid[:, None, :], scores, jnp.asarray(-1e9, dtype=scores.dtype))
    attn = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("bhl,bhld->bhd", attn, vh)
    context = jnp.reshape(context, (q.shape[0], num_heads * head_dim))
    q = q + context @ params[f"{prefix}_wo"]
    ff = jnp.tanh(_layer_norm(q) @ params[f"{prefix}_ff1_w"] + params[f"{prefix}_ff1_b"])
    q = q + ff @ params[f"{prefix}_ff2_w"] + params[f"{prefix}_ff2_b"]
    return q


def apply_query_attention_partner_decoder(
    params,
    query_obs,
    partner_obs_history,
    partner_action_history,
    action_dim,
    rng=None,
    deterministic=True,
    return_aux=False,
):
    query_obs = jnp.asarray(query_obs, dtype=jnp.float32)
    partner_obs_history = jnp.asarray(partner_obs_history, dtype=jnp.float32)
    partner_action_history = jnp.asarray(partner_action_history, dtype=jnp.int32)
    if partner_action_history.ndim == 1:
        partner_action_history = partner_action_history[:, None]
    if partner_obs_history.ndim == 4:
        partner_obs_history = partner_obs_history[:, None, ...]

    batch_size, history_len = partner_action_history.shape[:2]
    hist_obs_flat = jnp.reshape(
        partner_obs_history, (batch_size * history_len,) + partner_obs_history.shape[2:]
    )
    hist_obs_emb = jnp.reshape(
        _attn_cnn_frame_encode(params, hist_obs_flat), (batch_size, history_len, -1)
    )
    hist_action_oh = one_hot(partner_action_history, action_dim).astype(jnp.float32)
    hist_action_emb = hist_action_oh @ params["attn_action_w"] + params["attn_action_b"]
    pos_emb = params["attn_pos_emb"][-history_len:]
    tokens = jnp.tanh(hist_obs_emb + hist_action_emb + pos_emb[None, :, :])

    obs_abs = jnp.sum(jnp.abs(partner_obs_history), axis=tuple(range(2, partner_obs_history.ndim)))
    valid = obs_abs > jnp.asarray(EPS, dtype=jnp.float32)
    valid = jnp.where(jnp.any(valid, axis=-1, keepdims=True), valid, jnp.ones_like(valid))

    aux = {}
    q = _attn_cnn_frame_encode(params, query_obs)
    if "vae_mu_w" in params:
        use_query_context = (
            "latent_vae_use_query_attn" in params
            or "latent_query_conditioned_vae" in params
        )
        if use_query_context:
            layer_idx = 0
            while f"attn_l{layer_idx}_wq" in params:
                q = _query_cross_attention_layer(params, q, tokens, valid, f"attn_l{layer_idx}")
                layer_idx += 1
        if "latent_query_conditioned_vae" in params:
            posterior_hidden = q
        else:
            valid_f = valid.astype(jnp.float32)[..., None]
            denom = jnp.maximum(jnp.sum(valid_f, axis=1), 1.0)
            posterior_hidden = jnp.sum(tokens * valid_f, axis=1) / denom
        z, aux = _vae_latent_from_hidden(
            params, posterior_hidden, rng=rng, deterministic=deterministic
        )
        z = _decoder_latent(params, z)
        dec_input = jnp.concatenate([q, z], axis=-1)
    else:
        layer_idx = 0
        while f"attn_l{layer_idx}_wq" in params:
            q = _query_cross_attention_layer(params, q, tokens, valid, f"attn_l{layer_idx}")
            layer_idx += 1
        dec_input = q
    h = jnp.tanh(_layer_norm(dec_input) @ params["attn_dec_w1"] + params["attn_dec_b1"])
    h = jnp.tanh(h @ params["attn_dec_w2"] + params["attn_dec_b2"])
    logits = h @ params["attn_dec_w3"] + params["attn_dec_b3"]
    if "latent_lora_residual_enabled" in params and "vae_mu_w" in params:
        lora_delta, lora_aux = _lora_residual(params, q, z, action_dim)
        logits = logits + lora_delta
        aux = dict(aux)
        aux.update(lora_aux)
    return (logits, aux) if return_aux else logits


def apply_query_attention_residual_gated_decoder(
    params,
    query_obs,
    partner_obs_history,
    partner_action_history,
    action_dim,
    rng=None,
    deterministic=True,
    return_aux=False,
):
    query_obs = jnp.asarray(query_obs, dtype=jnp.float32)
    partner_obs_history = jnp.asarray(partner_obs_history, dtype=jnp.float32)
    partner_action_history = jnp.asarray(partner_action_history, dtype=jnp.int32)
    if partner_action_history.ndim == 1:
        partner_action_history = partner_action_history[:, None]
    if partner_obs_history.ndim == 4:
        partner_obs_history = partner_obs_history[:, None, ...]

    batch_size, history_len = partner_action_history.shape[:2]
    hist_obs_flat = jnp.reshape(
        partner_obs_history, (batch_size * history_len,) + partner_obs_history.shape[2:]
    )
    hist_obs_emb = jnp.reshape(
        _attn_cnn_frame_encode(params, hist_obs_flat), (batch_size, history_len, -1)
    )
    hist_action_oh = one_hot(partner_action_history, action_dim).astype(jnp.float32)
    hist_action_emb = hist_action_oh @ params["attn_action_w"] + params["attn_action_b"]
    pos_emb = params["attn_pos_emb"][-history_len:]
    tokens = jnp.tanh(hist_obs_emb + hist_action_emb + pos_emb[None, :, :])

    obs_abs = jnp.sum(jnp.abs(partner_obs_history), axis=tuple(range(2, partner_obs_history.ndim)))
    valid = obs_abs > jnp.asarray(EPS, dtype=jnp.float32)
    valid = jnp.where(jnp.any(valid, axis=-1, keepdims=True), valid, jnp.ones_like(valid))

    q_prior = _attn_cnn_frame_encode(params, query_obs)
    anchor_params = prefixed_params(params, "prior_anchor")
    if "latent_prior_anchor_enabled" in params and anchor_params:
        prior_logits = apply_latent_partner_decoder(
            anchor_params,
            query_obs,
            partner_obs_history,
            partner_action_history,
            action_dim,
            rng=None,
            deterministic=True,
            return_aux=False,
        )
        prior_logits = jax.lax.stop_gradient(prior_logits)
    else:
        prior_logits = _mlp3(q_prior, params, "prior_dec", normalize_input=True)

    q_context = q_prior
    layer_idx = 0
    while f"attn_l{layer_idx}_wq" in params:
        q_context = _query_cross_attention_layer(
            params, q_context, tokens, valid, f"attn_l{layer_idx}"
        )
        layer_idx += 1

    valid_f = valid.astype(jnp.float32)[..., None]
    denom = jnp.maximum(jnp.sum(valid_f, axis=1), 1.0)
    posterior_hidden = jnp.sum(tokens * valid_f, axis=1) / denom
    aux = {}
    if "vae_mu_w" in params:
        z, aux = _vae_latent_from_hidden(
            params, posterior_hidden, rng=rng, deterministic=deterministic
        )
        z = _decoder_latent(params, z)
    else:
        z = _decoder_latent(params, posterior_hidden)

    residual_input = jnp.concatenate([q_context, z], axis=-1)
    residual_logits = _mlp3(residual_input, params, "residual_dec", normalize_input=True)
    residual_clip = jnp.asarray(params.get("latent_residual_clip", 2.0), dtype=jnp.float32)
    residual_logits = jnp.clip(residual_logits, -residual_clip, residual_clip)
    gate_logits = _mlp3(residual_input, params, "gate_dec", normalize_input=True)
    gate = jax.nn.sigmoid(gate_logits)
    logits = prior_logits + gate * residual_logits
    aux = dict(aux)
    aux.update({
        "prior_logits": prior_logits,
        "residual_logits": residual_logits,
        "residual_gate": gate,
    })
    return (logits, aux) if return_aux else logits


def apply_query_attention_factorized_prior_residual_decoder(
    params,
    query_obs,
    partner_obs_history,
    partner_action_history,
    action_dim,
    rng=None,
    deterministic=True,
    return_aux=False,
):
    del rng, deterministic
    query_obs = jnp.asarray(query_obs, dtype=jnp.float32)
    partner_obs_history = jnp.asarray(partner_obs_history, dtype=jnp.float32)
    partner_action_history = jnp.asarray(partner_action_history, dtype=jnp.int32)
    if partner_action_history.ndim == 1:
        partner_action_history = partner_action_history[:, None]
    if partner_obs_history.ndim == 4:
        partner_obs_history = partner_obs_history[:, None, ...]

    batch_size, history_len = partner_action_history.shape[:2]

    def history_z(obs_hist, act_hist):
        hist_obs_flat = jnp.reshape(
            obs_hist, (batch_size * history_len,) + obs_hist.shape[2:]
        )
        hist_obs_emb = jnp.reshape(
            _attn_cnn_frame_encode(params, hist_obs_flat), (batch_size, history_len, -1)
        )
        hist_action_oh = one_hot(act_hist, action_dim).astype(jnp.float32)
        hist_action_emb = hist_action_oh @ params["attn_action_w"] + params["attn_action_b"]
        pos_emb = params["attn_pos_emb"][-history_len:]
        tokens = jnp.tanh(hist_obs_emb + hist_action_emb + pos_emb[None, :, :])
        obs_abs = jnp.sum(jnp.abs(obs_hist), axis=tuple(range(2, obs_hist.ndim)))
        valid = obs_abs > jnp.asarray(EPS, dtype=jnp.float32)
        valid = jnp.where(jnp.any(valid, axis=-1, keepdims=True), valid, jnp.ones_like(valid))
        valid_f = valid.astype(jnp.float32)[..., None]
        denom = jnp.maximum(jnp.sum(valid_f, axis=1), 1.0)
        posterior_hidden = jnp.sum(tokens * valid_f, axis=1) / denom
        z, aux = _vae_latent_from_hidden(
            params, posterior_hidden, rng=None, deterministic=True
        )
        return _decoder_latent(params, z), aux

    z, aux = history_z(partner_obs_history, partner_action_history)
    zero_obs = jnp.zeros_like(partner_obs_history)
    zero_actions = jnp.zeros_like(partner_action_history)
    z0, _z0_aux = history_z(zero_obs, zero_actions)
    z_eff = z - jax.lax.stop_gradient(z0)

    q = _attn_cnn_frame_encode(params, query_obs)
    prior_logits = _mlp3(q, params, "prior_dec", normalize_input=True)
    basis_flat = _mlp3(q, params, "basis_dec", normalize_input=True)
    z_dim = z_eff.shape[-1]
    basis = jnp.reshape(basis_flat, (batch_size, z_dim, int(action_dim)))
    residual_logits = jnp.einsum("bz,bza->ba", z_eff, basis) / jnp.sqrt(
        jnp.asarray(z_dim, dtype=jnp.float32)
    )
    residual_clip = jnp.asarray(params.get("latent_residual_clip", 2.0), dtype=jnp.float32)
    residual_logits = jnp.clip(residual_logits, -residual_clip, residual_clip)
    logits = prior_logits + residual_logits
    aux = dict(aux)
    aux.update({
        "prior_logits": prior_logits,
        "residual_logits": residual_logits,
        "residual_gate": jnp.ones((batch_size, 1), dtype=jnp.float32),
        "z_eff": z_eff,
    })
    return (logits, aux) if return_aux else logits


def encode_partner_online_z(
    params,
    partner_obs_history,
    partner_action_history,
    action_dim,
):
    """Encode history into the latent used by online-z adaptation.

    The returned vector is in the same space consumed by
    ``apply_latent_partner_decoder_with_z``. For factorized prior-residual
    decoders this is the residual latent, so zero history maps to zero.
    """
    partner_obs_history = jnp.asarray(partner_obs_history, dtype=jnp.float32)
    partner_action_history = jnp.asarray(partner_action_history, dtype=jnp.int32)
    if partner_action_history.ndim == 1:
        partner_action_history = partner_action_history[:, None]
    if partner_obs_history.ndim == 4:
        partner_obs_history = partner_obs_history[:, None, ...]

    if "attn_conv1_w" in params:
        batch_size, history_len = partner_action_history.shape[:2]
        hist_obs_flat = jnp.reshape(
            partner_obs_history,
            (batch_size * history_len,) + partner_obs_history.shape[2:],
        )
        hist_obs_emb = jnp.reshape(
            _attn_cnn_frame_encode(params, hist_obs_flat),
            (batch_size, history_len, -1),
        )
        hist_action_oh = one_hot(partner_action_history, action_dim).astype(jnp.float32)
        hist_action_emb = hist_action_oh @ params["attn_action_w"] + params["attn_action_b"]
        pos_emb = params["attn_pos_emb"][-history_len:]
        tokens = jnp.tanh(hist_obs_emb + hist_action_emb + pos_emb[None, :, :])
        obs_abs = jnp.sum(
            jnp.abs(partner_obs_history),
            axis=tuple(range(2, partner_obs_history.ndim)),
        )
        valid = obs_abs > jnp.asarray(EPS, dtype=jnp.float32)
        valid = jnp.where(jnp.any(valid, axis=-1, keepdims=True), valid, jnp.ones_like(valid))
        valid_f = valid.astype(jnp.float32)[..., None]
        denom = jnp.maximum(jnp.sum(valid_f, axis=1), 1.0)
        posterior_hidden = jnp.sum(tokens * valid_f, axis=1) / denom
        if "vae_mu_w" in params:
            z, _aux = _vae_latent_from_hidden(
                params, posterior_hidden, rng=None, deterministic=True
            )
        else:
            z = posterior_hidden
        z = _decoder_latent(params, z)
        if "latent_factorized_prior_residual_enabled" in params:
            zero_obs = jnp.zeros_like(partner_obs_history)
            zero_actions = jnp.zeros_like(partner_action_history)
            zero_z = encode_partner_online_z(
                {
                    key: value
                    for key, value in params.items()
                    if key != "latent_factorized_prior_residual_enabled"
                },
                zero_obs,
                zero_actions,
                action_dim,
            )
            z = z - jax.lax.stop_gradient(zero_z)
        if "latent_online_residual_c_enabled" in params:
            c_dim = int(params["orc_c_init"].shape[0])
            c0 = jnp.broadcast_to(params["orc_c_init"][None, :], (z.shape[0], c_dim))
            z = jnp.concatenate([z, c0], axis=-1)
        return z

    if "vae_mu_w" in params:
        hist_emb = encode_partner_history_embedding(
            params, partner_obs_history, partner_action_history, action_dim
        )
        z, _aux = _vae_latent_from_hidden(params, hist_emb, rng=None, deterministic=True)
        return _decoder_latent(params, z)
    return _decoder_latent(
        params,
        encode_partner_latent(
            params, partner_obs_history, partner_action_history, action_dim
        ),
    )


def apply_latent_partner_decoder_with_z(params, query_obs, latent_z, action_dim):
    """Decode partner policy for query observations from an explicit latent z."""
    query_obs = jnp.asarray(query_obs, dtype=jnp.float32)
    latent_z = jnp.asarray(latent_z, dtype=jnp.float32)
    if latent_z.ndim == 1:
        latent_z = latent_z[None, :]

    if "attn_conv1_w" in params:
        q = _attn_cnn_frame_encode(params, query_obs)
        if "latent_online_residual_c_enabled" in params:
            z0, c = _online_residual_c_split(params, latent_z)
            base_logits = _attn_base_logits_with_z(params, q, z0, action_dim)
            residual_logits = _online_residual_c_delta(params, q, z0, c, action_dim)
            return base_logits + residual_logits
        if "latent_factorized_prior_residual_enabled" in params:
            prior_logits = _mlp3(q, params, "prior_dec", normalize_input=True)
            basis_flat = _mlp3(q, params, "basis_dec", normalize_input=True)
            z_dim = latent_z.shape[-1]
            basis = jnp.reshape(basis_flat, (query_obs.shape[0], z_dim, int(action_dim)))
            residual_logits = jnp.einsum("bz,bza->ba", latent_z, basis) / jnp.sqrt(
                jnp.asarray(z_dim, dtype=jnp.float32)
            )
            residual_clip = jnp.asarray(
                params.get("latent_residual_clip", 2.0), dtype=jnp.float32
            )
            residual_logits = jnp.clip(residual_logits, -residual_clip, residual_clip)
            return prior_logits + residual_logits
        if "latent_residual_gate_enabled" in params:
            q_context = q
            residual_input = jnp.concatenate([q_context, latent_z], axis=-1)
            prior_logits = _mlp3(q, params, "prior_dec", normalize_input=True)
            residual_logits = _mlp3(
                residual_input, params, "residual_dec", normalize_input=True
            )
            residual_clip = jnp.asarray(
                params.get("latent_residual_clip", 2.0), dtype=jnp.float32
            )
            residual_logits = jnp.clip(residual_logits, -residual_clip, residual_clip)
            gate_logits = _mlp3(residual_input, params, "gate_dec", normalize_input=True)
            return prior_logits + jax.nn.sigmoid(gate_logits) * residual_logits
        return _attn_base_logits_with_z(params, q, latent_z, action_dim)

    query_emb = _frame_encode(params, query_obs)
    x = jnp.concatenate([query_emb, latent_z], axis=-1)
    h = jnp.tanh(x @ params["latent_dec_w1"] + params["latent_dec_b1"])
    h = jnp.tanh(h @ params["latent_dec_w2"] + params["latent_dec_b2"])
    return h @ params["latent_dec_w3"] + params["latent_dec_b3"]


def apply_latent_partner_decoder(
    params,
    query_obs,
    partner_obs_history,
    partner_action_history,
    action_dim,
    rng=None,
    deterministic=True,
    return_aux=False,
):
    if "attn_conv1_w" in params:
        if "latent_factorized_prior_residual_enabled" in params:
            return apply_query_attention_factorized_prior_residual_decoder(
                params,
                query_obs,
                partner_obs_history,
                partner_action_history,
                action_dim,
                rng=rng,
                deterministic=deterministic,
                return_aux=return_aux,
            )
        if "latent_residual_gate_enabled" in params:
            return apply_query_attention_residual_gated_decoder(
                params,
                query_obs,
                partner_obs_history,
                partner_action_history,
                action_dim,
                rng=rng,
                deterministic=deterministic,
                return_aux=return_aux,
            )
        return apply_query_attention_partner_decoder(
            params,
            query_obs,
            partner_obs_history,
            partner_action_history,
            action_dim,
            rng=rng,
            deterministic=deterministic,
            return_aux=return_aux,
        )
    query_obs = jnp.asarray(query_obs, dtype=jnp.float32)
    aux = {}
    if "vae_mu_w" in params:
        hist_emb = encode_partner_history_embedding(
            params, partner_obs_history, partner_action_history, action_dim
        )
        z, aux = _vae_latent_from_hidden(
            params, hist_emb, rng=rng, deterministic=deterministic
        )
        z = _decoder_latent(params, z)
    else:
        z = encode_partner_latent(params, partner_obs_history, partner_action_history, action_dim)
    query_emb = _frame_encode(params, query_obs)
    x = jnp.concatenate([query_emb, z], axis=-1)
    h = jnp.tanh(x @ params["latent_dec_w1"] + params["latent_dec_b1"])
    h = jnp.tanh(h @ params["latent_dec_w2"] + params["latent_dec_b2"])
    logits = h @ params["latent_dec_w3"] + params["latent_dec_b3"]
    return (logits, aux) if return_aux else logits



def apply_query_attention_compatible_decoder(
    params,
    query_obs,
    partner_obs_history,
    partner_action_history,
    action_dim,
    rng=None,
    deterministic=True,
    return_aux=False,
):
    query_obs = jnp.asarray(query_obs, dtype=jnp.float32)
    partner_obs_history = jnp.asarray(partner_obs_history, dtype=jnp.float32)
    partner_action_history = jnp.asarray(partner_action_history, dtype=jnp.int32)
    if partner_action_history.ndim == 1:
        partner_action_history = partner_action_history[:, None]
    if partner_obs_history.ndim == 4:
        partner_obs_history = partner_obs_history[:, None, ...]

    batch_size, history_len = partner_action_history.shape[:2]
    hist_obs_flat = jnp.reshape(
        partner_obs_history, (batch_size * history_len,) + partner_obs_history.shape[2:]
    )
    hist_obs_emb = jnp.reshape(
        _attn_cnn_frame_encode(params, hist_obs_flat), (batch_size, history_len, -1)
    )
    hist_action_oh = one_hot(partner_action_history, action_dim).astype(jnp.float32)
    hist_action_emb = hist_action_oh @ params["attn_action_w"] + params["attn_action_b"]
    pos_emb = params["attn_pos_emb"][-history_len:]
    tokens = jnp.tanh(hist_obs_emb + hist_action_emb + pos_emb[None, :, :])

    obs_abs = jnp.sum(jnp.abs(partner_obs_history), axis=tuple(range(2, partner_obs_history.ndim)))
    valid = obs_abs > jnp.asarray(EPS, dtype=jnp.float32)
    valid = jnp.where(jnp.any(valid, axis=-1, keepdims=True), valid, jnp.ones_like(valid))

    aux = {}
    q = _attn_cnn_frame_encode(params, query_obs)
    if "vae_mu_w" in params:
        use_query_context = (
            "latent_vae_use_query_attn" in params
            or "latent_query_conditioned_vae" in params
        )
        if use_query_context:
            layer_idx = 0
            while f"attn_l{layer_idx}_wq" in params:
                q = _query_cross_attention_layer(params, q, tokens, valid, f"attn_l{layer_idx}")
                layer_idx += 1
        if "latent_query_conditioned_vae" in params:
            posterior_hidden = q
        else:
            valid_f = valid.astype(jnp.float32)[..., None]
            denom = jnp.maximum(jnp.sum(valid_f, axis=1), 1.0)
            posterior_hidden = jnp.sum(tokens * valid_f, axis=1) / denom
        z, aux = _vae_latent_from_hidden(
            params, posterior_hidden, rng=rng, deterministic=deterministic
        )
        z = _decoder_latent(params, z)
        dec_input = jnp.concatenate([q, z], axis=-1)
    else:
        layer_idx = 0
        while f"attn_l{layer_idx}_wq" in params:
            q = _query_cross_attention_layer(params, q, tokens, valid, f"attn_l{layer_idx}")
            layer_idx += 1
        dec_input = q

    h_partner = jnp.tanh(_layer_norm(dec_input) @ params["attn_dec_w1"] + params["attn_dec_b1"])
    h_partner = jnp.tanh(h_partner @ params["attn_dec_w2"] + params["attn_dec_b2"])
    partner_logits = h_partner @ params["attn_dec_w3"] + params["attn_dec_b3"]
    partner_probs = jax.lax.stop_gradient(jax.nn.softmax(partner_logits, axis=-1))
    compat_input = jnp.concatenate([dec_input, partner_probs], axis=-1)
    h = jnp.tanh(_layer_norm(compat_input) @ params["compat_dec_w1"] + params["compat_dec_b1"])
    h = jnp.tanh(h @ params["compat_dec_w2"] + params["compat_dec_b2"])
    logits = h @ params["compat_dec_w3"] + params["compat_dec_b3"]
    return (logits, aux) if return_aux else logits


def apply_latent_compatible_decoder(
    params,
    query_obs,
    partner_obs_history,
    partner_action_history,
    action_dim,
    rng=None,
    deterministic=True,
    return_aux=False,
):
    if "compat_dec_w1" not in params:
        return apply_latent_partner_decoder(
            params,
            query_obs,
            partner_obs_history,
            partner_action_history,
            action_dim,
            rng=rng,
            deterministic=deterministic,
            return_aux=return_aux,
        )
    if "attn_conv1_w" in params:
        return apply_query_attention_compatible_decoder(
            params,
            query_obs,
            partner_obs_history,
            partner_action_history,
            action_dim,
            rng=rng,
            deterministic=deterministic,
            return_aux=return_aux,
        )
    query_obs = jnp.asarray(query_obs, dtype=jnp.float32)
    aux = {}
    if "vae_mu_w" in params:
        hist_emb = encode_partner_history_embedding(
            params, partner_obs_history, partner_action_history, action_dim
        )
        z, aux = _vae_latent_from_hidden(
            params, hist_emb, rng=rng, deterministic=deterministic
        )
        z = _decoder_latent(params, z)
    else:
        z = encode_partner_latent(params, partner_obs_history, partner_action_history, action_dim)
    query_emb = _frame_encode(params, query_obs)
    dec_input = jnp.concatenate([query_emb, z], axis=-1)
    h_partner = jnp.tanh(dec_input @ params["latent_dec_w1"] + params["latent_dec_b1"])
    h_partner = jnp.tanh(h_partner @ params["latent_dec_w2"] + params["latent_dec_b2"])
    partner_logits = h_partner @ params["latent_dec_w3"] + params["latent_dec_b3"]
    partner_probs = jax.lax.stop_gradient(jax.nn.softmax(partner_logits, axis=-1))
    compat_input = jnp.concatenate([dec_input, partner_probs], axis=-1)
    h = jnp.tanh(compat_input @ params["compat_dec_w1"] + params["compat_dec_b1"])
    h = jnp.tanh(h @ params["compat_dec_w2"] + params["compat_dec_b2"])
    logits = h @ params["compat_dec_w3"] + params["compat_dec_b3"]
    return (logits, aux) if return_aux else logits


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
