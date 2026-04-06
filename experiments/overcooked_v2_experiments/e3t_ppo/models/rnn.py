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
        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])
        rnn_state = jnp.where(resets[:, None], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(ActorCriticBase):
    @nn.compact
    def __call__(self, hidden, x, train=False):
        del train
        obs, dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        moa_hidden = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(embedding)
        moa_hidden = activation(moa_hidden)
        moa_hidden = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(moa_hidden)
        moa_hidden = activation(moa_hidden)
        moa_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(moa_hidden)
        other_pi = distrax.Categorical(logits=moa_logits)

        actor_input = jnp.concatenate([embedding, moa_hidden], axis=-1)
        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(actor_input)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1), other_pi
