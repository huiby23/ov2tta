import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


def _activation(name: str):
    if name == "relu":
        return nn.relu
    if name == "tanh":
        return nn.tanh
    if name == "leaky_relu":
        return nn.leaky_relu
    return nn.relu


class OriginalConvEncoder(nn.Module):
    num_filters: int
    num_convs: int

    @nn.compact
    def __call__(self, obs):
        added_batch = obs.ndim == 3
        if added_batch:
            obs = obs[None, ...]

        x = nn.Conv(
            features=self.num_filters,
            kernel_size=(5, 5),
            padding="SAME",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_initial",
        )(obs)
        x = nn.leaky_relu(x)

        for i in range(max(self.num_convs - 1, 0)):
            padding = "SAME" if i < self.num_convs - 2 else "VALID"
            x = nn.Conv(
                features=self.num_filters,
                kernel_size=(3, 3),
                padding=padding,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"conv_{i}",
            )(x)
            if i < self.num_convs - 2:
                x = nn.leaky_relu(x)

        x = x.reshape((x.shape[0], -1))
        if added_batch:
            x = x[0]
        return x


class ContextEncoder(nn.Module):
    config: dict
    action_dim: int

    @nn.compact
    def __call__(self, hist_obs, hist_actions):
        model_config = self.config
        latent_dim = model_config["LATENT_DIM"]
        length = model_config["CONTEXT_LENGTH"]

        encoder = OriginalConvEncoder(
            num_filters=model_config["NUM_FILTERS"],
            num_convs=model_config["NUM_CONV_LAYERS"],
        )

        batch_size = hist_obs.shape[0]
        hist_flat = hist_obs.reshape((-1,) + hist_obs.shape[-3:])
        obs_embed = encoder(hist_flat)

        action_onehot = jax.nn.one_hot(
            jnp.clip(hist_actions.reshape(-1), 0, self.action_dim - 1), self.action_dim
        )
        action_embed = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="word_embd",
        )(action_onehot)
        step_features = jnp.concatenate([obs_embed, action_embed], axis=-1)
        step_features = nn.leaky_relu(step_features)

        x = nn.Dense(
            model_config["SIZE_HIDDEN_LAYERS"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="context_dense_0",
        )(step_features)
        x = nn.leaky_relu(x)
        x = nn.Dense(
            latent_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="context_dense_1",
        )(x)
        x = nn.leaky_relu(x)

        x = x.reshape((batch_size, -1))
        for i in range(model_config["NUM_HIDDEN_LAYERS"]):
            x = nn.Dense(
                latent_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"context_temporal_{i}",
            )(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(
            latent_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="human_prob_pre_0",
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            latent_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="human_prob_pre_1",
        )(x)
        x = nn.tanh(x)
        return x


class ContextDecoder(nn.Module):
    config: dict
    action_dim: int

    @nn.compact
    def __call__(self, obs, context_latent):
        model_config = self.config
        latent_dim = model_config["LATENT_DIM"]

        encoder = OriginalConvEncoder(
            num_filters=model_config["NUM_FILTERS"],
            num_convs=model_config["NUM_CONV_LAYERS"],
        )
        obs_embed = encoder(obs)

        x = jnp.concatenate([obs_embed, context_latent], axis=-1)
        for i in range(max(model_config["NUM_HIDDEN_LAYERS"] - 1, 0)):
            x = nn.Dense(
                model_config["SIZE_HIDDEN_LAYERS"],
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name=f"decoder_dense_{i}",
            )(x)
            x = nn.leaky_relu(x)
        decoded = nn.Dense(
            latent_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="decoder_out",
        )(x)
        decoded = nn.leaky_relu(decoded)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="partner_logits",
        )(decoded)
        return logits


class EgoPolicyValue(nn.Module):
    config: dict
    action_dim: int

    @nn.compact
    def __call__(self, obs, context_probs):
        model_config = self.config
        encoder = OriginalConvEncoder(
            num_filters=model_config["NUM_FILTERS"],
            num_convs=model_config["NUM_CONV_LAYERS"],
        )
        obs_embed = encoder(obs)

        x = jnp.concatenate([obs_embed, context_probs], axis=-1)
        for i in range(model_config["NUM_HIDDEN_LAYERS"]):
            x = nn.Dense(
                model_config["SIZE_HIDDEN_LAYERS"],
                kernel_init=orthogonal(2.0),
                bias_init=constant(0.0),
                name=f"policy_dense_{i}",
            )(x)
            x = nn.leaky_relu(x)

        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_logits",
        )(x)
        pi = distrax.Categorical(logits=actor_logits)

        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(x)
        return pi, jnp.squeeze(value, axis=-1)


class E3TContextModule(nn.Module):
    config: dict
    action_dim: int

    @nn.compact
    def __call__(self, obs, hist_obs, hist_actions):
        latent = ContextEncoder(self.config, self.action_dim, name="context_encoder")(hist_obs, hist_actions)
        logits = ContextDecoder(self.config, self.action_dim, name="context_decoder")(obs, latent)
        return logits


class E3TPolicyModule(nn.Module):
    config: dict
    action_dim: int

    @nn.compact
    def __call__(self, obs, context_probs):
        return EgoPolicyValue(self.config, self.action_dim, name="ego_policy")(obs, context_probs)
