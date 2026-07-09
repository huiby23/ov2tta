
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal

from .abstract import ActorCriticBase
from .common import CNNSimple


class ActorCriticCNN(ActorCriticBase):
    """PPO-CNN with a train-time adapter and policy-level base logits.

    The base actor path and adapter path are both trained by PPO. The adapter
    produces a bounded logit delta so test-time TTAC can later update only the
    adapter parameters while keeping the base policy available as a trust-region
    reference.
    """

    def _activation(self):
        if self.config["ACTIVATION"] == "relu":
            return nn.relu
        return nn.tanh

    def _extract_inputs(self, x):
        if len(x) == 2:
            obs, done = x
            adapter_readout_scale = 1.0
        elif len(x) == 3:
            obs, done, adapter_readout_scale = x
        else:
            raise ValueError(
                "ActorCriticCNN expects (obs, done) or "
                "(obs, done, adapter_readout_scale)."
            )
        return obs, done, adapter_readout_scale

    @nn.compact
    def __call__(self, hidden, x):
        del hidden
        obs, _done, adapter_readout_scale = self._extract_inputs(x)
        activation = self._activation()
        fc_dim = self.config["FC_DIM_SIZE"]
        adapter_dim = int(self.config.get("TTAC_ADAPTER_DIM", fc_dim))
        adapter_scale = float(self.config.get("TTAC_ADAPTER_SCALE", 0.25))

        embed_model = CNNSimple(output_size=fc_dim, activation=activation)
        feature = jax.vmap(embed_model)(obs)
        feature = nn.LayerNorm(name="feature_norm")(feature)

        base_hidden = nn.Dense(
            fc_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="base_actor_hidden",
        )(feature)
        base_hidden = activation(base_hidden)
        base_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="base_actor_logits",
        )(base_hidden)

        adapter_hidden = nn.Dense(
            adapter_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="ttac_adapter_down",
        )(feature)
        adapter_hidden = activation(adapter_hidden)
        adapter_up_kernel_init = (
            constant(0.0)
            if bool(self.config.get("TTAC_ZERO_INIT_ADAPTER_UP", False))
            else orthogonal(0.01)
        )
        adapter_delta_raw = nn.Dense(
            self.action_dim,
            kernel_init=adapter_up_kernel_init,
            bias_init=constant(0.0),
            name="ttac_adapter_up",
        )(adapter_hidden)
        adapter_delta = adapter_scale * jnp.tanh(adapter_delta_raw)
        adapter_readout_scale = jnp.asarray(adapter_readout_scale, dtype=feature.dtype)
        adapted_logits = base_logits + adapter_readout_scale * adapter_delta

        pi = distrax.Categorical(logits=adapted_logits)

        critic = nn.Dense(
            fc_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_hidden",
        )(feature)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_value",
        )(critic)

        aux = {
            "feature": feature,
            "base_logits": base_logits,
            "adapted_logits": adapted_logits,
            "adapter_delta": adapter_delta,
            "adapter_delta_norm": jnp.linalg.norm(adapter_delta, axis=-1),
        }
        return None, pi, jnp.squeeze(critic, axis=-1), aux
