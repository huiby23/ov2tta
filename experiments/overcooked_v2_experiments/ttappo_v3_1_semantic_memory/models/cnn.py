import functools

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal

from .abstract import ActorCriticBase
from .common import CNNSimple
from .model import PartnerMemoryCarry
from .rnn import ScannedRNN


class ScannedPartnerMemoryUpdater(nn.Module):
    action_dim: int
    action_embed_dim: int
    memory_dim: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, memory, x):
        feature_z, temporal_feature, partner_action, done, update_mask = x
        action_embed = nn.Embed(
            num_embeddings=self.action_dim,
            features=self.action_embed_dim,
            embedding_init=orthogonal(jnp.sqrt(2)),
            name="partner_action_embedding",
        )(partner_action.astype(jnp.int32))
        memory_input = jnp.concatenate(
            [
                jax.lax.stop_gradient(feature_z),
                jax.lax.stop_gradient(temporal_feature),
                action_embed,
            ],
            axis=-1,
        )
        new_memory, _ = nn.GRUCell(
            features=self.memory_dim,
            name="partner_memory_gru",
        )(memory, memory_input)
        new_memory = jnp.where(update_mask[..., None], new_memory, memory)
        new_memory = jnp.where(done[..., None], jnp.zeros_like(new_memory), new_memory)
        return new_memory, memory


class ActorCriticCNN(ActorCriticBase):
    def setup(self):
        fc_dim = self.config["FC_DIM_SIZE"]
        temporal_dim = self.config.get("TEMPORAL_HIDDEN_DIM", fc_dim)
        memory_dim = self.config.get("PARTNER_MEMORY_DIM", temporal_dim)

        self.temporal_proj_layer = nn.Dense(
            temporal_dim,
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
            fc_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="ttt_adapter_up",
        )
        self.partner_memory_updater = ScannedPartnerMemoryUpdater(
            action_dim=self.action_dim,
            action_embed_dim=self.config.get("PARTNER_ACTION_EMBED_DIM", 16),
            memory_dim=memory_dim,
        )
        self.film_gamma_layer = nn.Dense(
            fc_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="film_gamma",
        )
        self.film_beta_layer = nn.Dense(
            fc_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="film_beta",
        )
        self.partner_hidden_layer = nn.Dense(
            fc_dim,
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

    def _extract_inputs(self, x):
        if len(x) == 2:
            obs, done = x
            partner_action = None
            update_mask = None
            readout_scale = 1.0
        elif len(x) == 3:
            obs, done, readout_scale = x
            partner_action = None
            update_mask = None
        elif len(x) == 4:
            obs, done, partner_action, update_mask = x
            readout_scale = 1.0
        elif len(x) == 5:
            obs, done, partner_action, update_mask, readout_scale = x
        else:
            raise ValueError(
                "ActorCriticCNN expects input as (obs, done), "
                "(obs, done, readout_scale), "
                "(obs, done, partner_action, update_mask), or "
                "(obs, done, partner_action, update_mask, readout_scale)"
            )
        return obs, done, partner_action, update_mask, readout_scale

    def _compute_feature_z(self, embedding):
        activation = self._activation()
        adapter_hidden = self.ttt_adapter_down_layer(embedding)
        adapter_hidden = activation(adapter_hidden)
        adapter_out = self.ttt_adapter_up_layer(adapter_hidden)
        return embedding + adapter_out

    def _roll_partner_memory(
        self,
        init_memory,
        feature_z,
        temporal_feature,
        partner_action,
        done,
        update_mask,
    ):
        if partner_action is None:
            time_steps = feature_z.shape[0]
            memory_seq = jnp.broadcast_to(
                init_memory[jnp.newaxis, ...],
                (time_steps,) + init_memory.shape,
            )
            update_rate = jnp.zeros_like(done, dtype=jnp.float32)
            return init_memory, memory_seq, update_rate

        final_memory, memory_seq = self.partner_memory_updater(
            init_memory,
            (feature_z, temporal_feature, partner_action, done, update_mask),
        )
        return final_memory, memory_seq, update_mask.astype(jnp.float32)

    def update_memory_state(
        self,
        hidden: PartnerMemoryCarry,
        feature_z,
        temporal_feature,
        partner_action,
        done,
        update_mask=None,
    ):
        if update_mask is None:
            update_mask = jnp.ones_like(done, dtype=jnp.bool_)
        next_memory, _ = self.partner_memory_updater(
            hidden.partner_memory,
            (
                feature_z[jnp.newaxis, ...],
                temporal_feature[jnp.newaxis, ...],
                partner_action[jnp.newaxis, ...],
                done[jnp.newaxis, ...],
                update_mask[jnp.newaxis, ...],
            ),
        )
        return PartnerMemoryCarry(
            temporal_hidden=hidden.temporal_hidden,
            partner_memory=next_memory,
        )

    @nn.compact
    def __call__(self, hidden: PartnerMemoryCarry, x):
        obs, done, partner_action, update_mask, readout_scale = self._extract_inputs(x)
        activation = self._activation()
        film_scale = float(self.config.get("FILM_SCALE", 0.2))

        embed_model = CNNSimple(
            output_size=self.config["FC_DIM_SIZE"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)

        temporal_input = self.temporal_proj_layer(embedding)
        temporal_input = activation(temporal_input)
        next_temporal_hidden, temporal_feature = ScannedRNN()(
            hidden.temporal_hidden,
            (temporal_input, done),
        )

        feature_z = self._compute_feature_z(embedding)
        _ = self.partner_memory_updater(
            hidden.partner_memory,
            (
                feature_z[:1],
                temporal_feature[:1],
                jnp.zeros(feature_z.shape[:2], dtype=jnp.int32)[:1],
                jnp.zeros(done.shape, dtype=jnp.bool_)[:1],
                jnp.zeros(done.shape, dtype=jnp.bool_)[:1],
            ),
        )
        if update_mask is None and partner_action is not None:
            update_mask = jnp.ones_like(partner_action, dtype=jnp.bool_)
        final_memory, memory_seq, memory_update_rate = self._roll_partner_memory(
            hidden.partner_memory,
            feature_z,
            temporal_feature,
            partner_action,
            done,
            update_mask,
        )

        gamma = 1.0 + film_scale * jnp.tanh(self.film_gamma_layer(memory_seq))
        beta = film_scale * jnp.tanh(self.film_beta_layer(memory_seq))
        readout_scale = jnp.asarray(readout_scale, dtype=feature_z.dtype)
        feature_mod = feature_z + readout_scale[..., None] * (
            gamma * feature_z + beta - feature_z
        )

        partner_input = jnp.concatenate(
            [feature_z, temporal_feature, memory_seq],
            axis=-1,
        )
        partner_hidden = self.partner_hidden_layer(partner_input)
        partner_hidden = activation(partner_hidden)
        partner_logits = self.partner_logits_layer(partner_hidden)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(feature_mod)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(feature_mod)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        aux = {
            "partner_logits": partner_logits,
            "feature_z": feature_z,
            "temporal_feature": temporal_feature,
            "partner_memory": memory_seq,
            "film_gamma": gamma,
            "film_beta": beta,
            "feature_mod": feature_mod,
            "memory_update_rate": memory_update_rate,
        }
        next_hidden = PartnerMemoryCarry(
            temporal_hidden=next_temporal_hidden,
            partner_memory=final_memory,
        )
        return next_hidden, pi, jnp.squeeze(critic, axis=-1), aux
