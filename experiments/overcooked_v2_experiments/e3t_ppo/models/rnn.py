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

        if len(x) == 2:
            obs, dones = x
            history_obs = None
            history_actions = None
        elif len(x) == 4:
            obs, dones, history_obs, history_actions = x
        else:
            raise ValueError(f"Unexpected input tuple length for E3T-PPO RNN: {len(x)}")

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        predictor_hidden_dim = self.config.get(
            "PREDICTOR_HIDDEN_DIM",
            self.config["FC_DIM_SIZE"],
        )
        context_hidden_dim = self.config.get(
            "CONTEXT_HIDDEN_DIM",
            predictor_hidden_dim,
        )
        use_history_context = self.config.get("USE_HISTORY_CONTEXT", False)
        moa_to_actor = self.config.get("MOA_TO_ACTOR", False)
        moa_to_actor_detach = self.config.get("MOA_TO_ACTOR_DETACH", True)

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        if use_history_context and history_obs is not None and history_actions is not None:
            history_embedding = jax.vmap(
                lambda history_t: jax.vmap(
                    embed_model,
                    in_axes=1,
                    out_axes=1,
                )(history_t)
            )(history_obs)
            history_embedding = nn.LayerNorm()(history_embedding)

            clipped_history_actions = jnp.clip(history_actions, 0, self.action_dim - 1)
            history_action_oh = jax.nn.one_hot(clipped_history_actions, self.action_dim)
            history_features = jnp.concatenate(
                [history_embedding, history_action_oh],
                axis=-1,
            )
            history_features = nn.Dense(
                context_hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
            )(history_features)
            history_features = nn.leaky_relu(history_features)
            history_features = nn.Dense(
                context_hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
            )(history_features)
            history_features = nn.leaky_relu(history_features)
            context_embedding = history_features.mean(axis=2)
            context_embedding = nn.LayerNorm()(context_embedding)
        else:
            context_embedding = jnp.zeros(
                embedding.shape[:-1] + (context_hidden_dim,),
                dtype=embedding.dtype,
            )

        moa_input = (
            jnp.concatenate([embedding, context_embedding], axis=-1)
            if use_history_context
            else embedding
        )
        moa_hidden = nn.Dense(
            predictor_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(moa_input)
        moa_hidden = activation(moa_hidden)
        moa_hidden = nn.Dense(
            predictor_hidden_dim,
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

        actor_input = embedding
        if moa_to_actor:
            actor_condition = moa_hidden
            if use_history_context:
                actor_condition = jnp.concatenate(
                    [actor_condition, context_embedding],
                    axis=-1,
                )
            if moa_to_actor_detach:
                actor_condition = jax.lax.stop_gradient(actor_condition)
            actor_input = jnp.concatenate([embedding, actor_condition], axis=-1)

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
