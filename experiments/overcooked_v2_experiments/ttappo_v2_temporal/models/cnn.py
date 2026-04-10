import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal

from .abstract import ActorCriticBase
from .common import CNNSimple
from .rnn import ScannedRNN


class ActorCriticCNN(ActorCriticBase):
    def setup(self):
        self.temporal_proj_layer = nn.Dense(
            self.config.get("TEMPORAL_HIDDEN_DIM", self.config["FC_DIM_SIZE"]),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="partner_temporal_proj",
        )
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

    def _activation(self):
        if self.config["ACTIVATION"] == "relu":
            return nn.relu
        return nn.tanh

    def _ttt_forward_from_base(self, cached_feature):
        activation = self._activation()
        base_dim = int(self.config["FC_DIM_SIZE"])
        base_feature = cached_feature[..., :base_dim]
        temporal_feature = cached_feature[..., base_dim:]

        adapter_hidden = self.ttt_adapter_down_layer(base_feature)
        adapter_hidden = activation(adapter_hidden)
        adapter_out = self.ttt_adapter_up_layer(adapter_hidden)
        feature_z = base_feature + adapter_out

        partner_input = jnp.concatenate([feature_z, temporal_feature], axis=-1)
        partner_hidden = self.partner_hidden_layer(partner_input)
        partner_hidden = activation(partner_hidden)
        partner_logits = self.partner_logits_layer(partner_hidden)

        return feature_z, partner_hidden, partner_logits

    @nn.compact
    def __call__(self, hidden, x):
        obs, done = x
        activation = self._activation()

        embed_model = CNNSimple(
            output_size=self.config["FC_DIM_SIZE"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)

        temporal_input = self.temporal_proj_layer(embedding)
        temporal_input = activation(temporal_input)
        hidden, temporal_feature = ScannedRNN()(hidden, (temporal_input, done))

        cached_feature = jnp.concatenate([embedding, temporal_feature], axis=-1)
        feature_z, feature_s, partner_logits = self._ttt_forward_from_base(
            cached_feature
        )

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(feature_z)
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
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        aux = {
            "partner_logits": partner_logits,
            "feature_base": cached_feature,
            "feature_z": feature_z,
            "feature_s": feature_s,
        }
        return hidden, pi, jnp.squeeze(critic, axis=-1), aux
