import functools

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal

from .abstract import ActorCriticBase
from .common import CNN


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        new_carry = jnp.zeros_like(rnn_state)
        rnn_state = jnp.where(resets[:, jnp.newaxis], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(ActorCriticBase):
    @nn.compact
    def __call__(self, hidden, x, train=False):
        obs, dones, history_obs, history_actions = x
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        context_hidden_dim = self.config.get("CONTEXT_HIDDEN_DIM", self.config.get("PREDICTOR_HIDDEN_DIM", 64))

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )

        obs_flat = obs.reshape((-1,) + obs.shape[-3:])
        obs_embedding = jax.vmap(embed_model)(obs_flat)
        obs_embedding = obs_embedding.reshape(obs.shape[:2] + (-1,))
        obs_embedding = nn.LayerNorm()(obs_embedding)

        history_obs = history_obs.astype(obs.dtype)
        history_flat = history_obs.reshape((-1,) + history_obs.shape[-3:])
        history_embedding = jax.vmap(embed_model)(history_flat)
        history_embedding = history_embedding.reshape(history_obs.shape[:3] + (-1,))
        history_embedding = nn.LayerNorm()(history_embedding)

        clipped_history_actions = jnp.clip(history_actions, 0, self.action_dim - 1)
        history_action_oh = jax.nn.one_hot(clipped_history_actions, self.action_dim)
        history_features = jnp.concatenate([history_embedding, history_action_oh], axis=-1)
        history_features = nn.Dense(
            context_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(history_features)
        history_features = nn.leaky_relu(history_features)
        history_features = nn.Dense(
            context_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(history_features)
        history_features = nn.leaky_relu(history_features)

        context_embedding = history_features.mean(axis=2)
        context_embedding = nn.LayerNorm()(context_embedding)

        rnn_in = (obs_embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        prediction_input = jnp.concatenate([embedding, context_embedding], axis=-1)
        prediction_other = nn.Dense(
            context_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(prediction_input)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            context_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            context_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            context_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = nn.tanh(prediction_other)
        prediction_other = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = prediction_other / jnp.sqrt(
            jnp.sum(prediction_other**2, axis=-1, keepdims=True) + 1e-10
        )
        other_pi = distrax.Categorical(logits=prediction_other)

        actor_input = jnp.concatenate([embedding, prediction_other, context_embedding], axis=-1)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(actor_input)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] * 3 // 4,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 4,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"] * 3 // 4,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"] // 2,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1), other_pi
