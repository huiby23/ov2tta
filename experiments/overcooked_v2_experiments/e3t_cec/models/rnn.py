import functools

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from .abstract import ActorCriticBase


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
        lstm_state = carry
        ins, resets = x
        lstm_state = jax.tree_util.tree_map(
            lambda h: jnp.where(resets[:, None], jnp.zeros_like(h), h),
            lstm_state,
        )
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[-1])(lstm_state, ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0),
            (batch_size, hidden_size),
        )


class ActorCriticRNN(ActorCriticBase):
    @nn.compact
    def __call__(self, hidden, x, train=False):
        del train
        obs, dones, agent_positions = x
        del agent_positions

        batch_size, num_actors, _ = obs.shape
        is_overcooked = self.config.get("ENV_NAME") in {"overcooked", "overcooked_v2"}

        if self.config.get("GRAPH_NET", False):
            reshaped_obs = obs.reshape((-1,) + tuple(self.config["OBS_SHAPE"]))
            embedding = nn.Conv(
                features=64 if is_overcooked else 2 * self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32 if is_overcooked else self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)
            embedding = embedding.reshape((batch_size, num_actors, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if is_overcooked else self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = nn.relu(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        prediction_other = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(prediction_other)
        prediction_other = nn.tanh(prediction_other)
        prediction_other = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = prediction_other / jnp.sqrt(
            jnp.sum(prediction_other**2, axis=-1, keepdims=True) + 1e-10
        )
        other_pi = distrax.Categorical(logits=prediction_other)

        actor_embedding = jnp.concatenate([embedding, prediction_other], axis=-1)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(actor_embedding)
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
        if is_overcooked:
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
        if is_overcooked:
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
