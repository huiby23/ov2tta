
from __future__ import annotations

from typing import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal

from overcooked_v2_experiments.ppo.models.common import CNNSimple


def _activation(name: str) -> Callable:
    if name == "tanh":
        return nn.tanh
    if name == "gelu":
        return nn.gelu
    return nn.relu


class AddZMLP(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: Callable

    @nn.compact
    def __call__(self, feature: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        z_feature = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(z)
        z_feature = self.activation(z_feature)
        z_feature = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(z_feature)
        z_feature = self.activation(z_feature)
        z_feature = nn.LayerNorm()(z_feature)
        x = jnp.concatenate([feature, z_feature], axis=-1)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.LayerNorm()(x)
        return x


class GammaEncoder(nn.Module):
    hidden_dim: int
    z_dim: int
    action_dim: int
    activation_name: str = "relu"

    def setup(self):
        activation = _activation(self.activation_name)
        self.obs_encoder = CNNSimple(output_size=self.hidden_dim, activation=activation)
        self.add_action_mlp = AddZMLP(hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, activation=activation)
        self.gru = nn.GRUCell(features=self.hidden_dim)
        self.mean = nn.Dense(self.z_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.log_std = nn.Dense(self.z_dim, kernel_init=orthogonal(0.01), bias_init=constant(-0.5))

    def __call__(self, obs_seq: jnp.ndarray, action_onehot_seq: jnp.ndarray):
        batch_size = obs_seq.shape[1]
        obs_features = jax.vmap(self.obs_encoder)(obs_seq)
        features = self.add_action_mlp(obs_features, action_onehot_seq)
        carry = self.gru.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_dim))

        for t in range(features.shape[0]):
            carry, _ = self.gru(carry, features[t])
        mean = self.mean(carry)
        log_std = jnp.clip(self.log_std(carry), -5.0, 2.0)
        return mean, log_std


class GammaDecoder(nn.Module):
    hidden_dim: int
    z_dim: int
    action_dim: int
    activation_name: str = "relu"

    def setup(self):
        activation = _activation(self.activation_name)
        self.obs_encoder = CNNSimple(output_size=self.hidden_dim, activation=activation)
        self.add_z_mlp = AddZMLP(hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, activation=activation)
        self.gru = nn.GRUCell(features=self.hidden_dim)
        self.action_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

    def initial_carry(self, batch_size: int) -> jnp.ndarray:
        return self.gru.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_dim))

    def decode_step(self, obs: jnp.ndarray, z: jnp.ndarray, carry: jnp.ndarray):
        obs_features = self.obs_encoder(obs)
        features = self.add_z_mlp(obs_features, z)
        new_carry, y = self.gru(carry, features)
        logits = self.action_head(y)
        return logits, new_carry

    def __call__(self, obs_seq: jnp.ndarray, z: jnp.ndarray):
        batch_size = obs_seq.shape[1]
        z_seq = jnp.broadcast_to(z[jnp.newaxis, ...], (obs_seq.shape[0], batch_size, self.z_dim))
        obs_features = jax.vmap(self.obs_encoder)(obs_seq)
        features = self.add_z_mlp(obs_features, z_seq)
        carry = self.initial_carry(batch_size)

        logits = []
        for t in range(features.shape[0]):
            carry, y = self.gru(carry, features[t])
            logits.append(self.action_head(y))
        return jnp.stack(logits, axis=0)


class GammaVAE(nn.Module):
    action_dim: int
    z_dim: int = 16
    hidden_dim: int = 64
    activation_name: str = "relu"

    def setup(self):
        self.encoder = GammaEncoder(hidden_dim=self.hidden_dim, z_dim=self.z_dim, action_dim=self.action_dim, activation_name=self.activation_name)
        self.decoder = GammaDecoder(hidden_dim=self.hidden_dim, z_dim=self.z_dim, action_dim=self.action_dim, activation_name=self.activation_name)

    def __call__(self, obs_seq: jnp.ndarray, action_onehot_seq: jnp.ndarray, rng: jax.Array):
        mean, log_std = self.encoder(obs_seq, action_onehot_seq)
        eps = jax.random.normal(rng, mean.shape)
        z = mean + jnp.exp(log_std) * eps
        logits = self.decoder(obs_seq, z)
        return logits, mean, log_std, z

    def encode_mean(self, obs_seq: jnp.ndarray, action_onehot_seq: jnp.ndarray):
        mean, _ = self.encoder(obs_seq, action_onehot_seq)
        return mean

    def decode_step(self, obs: jnp.ndarray, z: jnp.ndarray, carry: jnp.ndarray):
        return self.decoder.decode_step(obs, z, carry)


def gamma_vae_loss(model: GammaVAE, params, obs_seq: jnp.ndarray, actions: jnp.ndarray, rng: jax.Array, kl_coef: float):
    action_onehot = jax.nn.one_hot(actions, model.action_dim)
    logits, mean, log_std, _ = model.apply({"params": params}, obs_seq, action_onehot, rng)
    dist = distrax.Categorical(logits=logits)
    log_likelihood_per_step = dist.log_prob(actions)
    log_likelihood = log_likelihood_per_step.sum(axis=0).mean()
    variance = jnp.exp(2.0 * log_std)
    kl = 0.5 * jnp.sum(jnp.square(mean) + variance - 1.0 - 2.0 * log_std, axis=-1).mean()
    loss = -(log_likelihood - kl_coef * kl)
    pred = jnp.argmax(logits, axis=-1)
    acc = (pred == actions).mean()
    meaningful = actions != 4
    meaningful_acc = jnp.where(meaningful.any(), ((pred == actions) & meaningful).sum() / jnp.maximum(meaningful.sum(), 1), acc)
    entropy = dist.entropy().mean()
    return loss, {"loss": loss, "log_likelihood": log_likelihood, "kl": kl, "acc": acc, "meaningful_acc": meaningful_acc, "entropy": entropy}
