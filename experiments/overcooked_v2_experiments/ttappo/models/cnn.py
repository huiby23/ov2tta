import functools
from typing import Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from .abstract import ActorCriticBase
from .common import CNN, MLP, CNNSimple, ResNet


class ActorCriticCNN(ActorCriticBase):

    def setup(self):
        self.ttt_adapter_down_layer = nn.Dense(
            self.config["TTT_ADAPTER_DIM"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="ttt_adapter_down",
        )
        self.ttt_adapter_up_layer = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="ttt_adapter_up",
        )
        self.partner_hidden_layer = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="partner_hidden",
        )
        self.partner_logits_layer = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="partner_logits",
        )

    def _ttt_forward_from_base(self, base_feature):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        adapter_hidden = self.ttt_adapter_down_layer(base_feature)
        adapter_hidden = activation(adapter_hidden)
        adapter_out = self.ttt_adapter_up_layer(adapter_hidden)
        feature_z = base_feature + adapter_out

        partner_hidden = self.partner_hidden_layer(feature_z)
        partner_hidden = activation(partner_hidden)
        partner_logits = self.partner_logits_layer(partner_hidden)

        return feature_z, partner_hidden, partner_logits

    @nn.compact
    def __call__(self, hidden, x):
        obs, done = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = obs

        # embedding_mlp = MLP(
        #     hidden_size=self.config["FC_DIM_SIZE"],
        #     output_size=self.config["FC_DIM_SIZE"],
        #     activation=activation,
        # )

        # embedding = jax.vmap(
        #     jax.vmap(embedding_mlp, in_axes=-2, out_axes=-2), in_axes=-2, out_axes=-2
        # )(obs)

        # embedding = nn.LayerNorm()(embedding)

        embed_model = CNNSimple(
            output_size=self.config["FC_DIM_SIZE"],  # * 2,
            activation=activation,
        )

        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"],
        #     kernel_init=orthogonal(jnp.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(embedding)
        # embedding = activation(embedding)

        feature_z, feature_s, partner_logits = self._ttt_forward_from_base(embedding)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(feature_z)
        # actor_mean = nn.BatchNorm(use_running_average=not self.train)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(feature_z)
        # critic = nn.BatchNorm(use_running_average=not self.train)(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        aux = {
            "partner_logits": partner_logits,
            "feature_base": embedding,
            "feature_z": feature_z,
            "feature_s": feature_s,
        }
        return hidden, pi, jnp.squeeze(critic, axis=-1), aux
