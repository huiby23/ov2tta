import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal

from .abstract import ActorCriticBase
from .common import CNNSimple


class ActorCriticCNN(ActorCriticBase):
    @nn.compact
    def __call__(self, hidden, x):
        if len(x) == 2:
            obs, done = x
            history_obs = None
            history_actions = None
        elif len(x) == 4:
            obs, done, history_obs, history_actions = x
        else:
            raise ValueError(
                f"Unexpected input tuple length for E3T-PPO CNN: {len(x)}"
            )

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
        moa_to_actor = self.config.get("MOA_TO_ACTOR", True)
        moa_to_actor_detach = self.config.get("MOA_TO_ACTOR_DETACH", False)

        policy_embed_model = CNNSimple(
            output_size=self.config["FC_DIM_SIZE"],
            activation=activation,
            name="policy_encoder",
        )

        obs_flat = obs.reshape((-1,) + obs.shape[-3:])
        policy_embedding = jax.vmap(policy_embed_model)(obs_flat)
        policy_embedding = policy_embedding.reshape(obs.shape[:2] + (-1,))
        policy_embedding = nn.LayerNorm(name="policy_ln")(policy_embedding)

        context_obs_embedding = policy_embedding
        if use_history_context and history_obs is not None and history_actions is not None:
            context_embed_model = CNNSimple(
                output_size=self.config["FC_DIM_SIZE"],
                activation=activation,
                name="context_encoder",
            )
            context_obs_embedding = jax.vmap(context_embed_model)(obs_flat)
            context_obs_embedding = context_obs_embedding.reshape(obs.shape[:2] + (-1,))
            context_obs_embedding = nn.LayerNorm(name="context_obs_ln")(context_obs_embedding)

        if use_history_context and history_obs is not None and history_actions is not None:
            history_obs = history_obs.astype(obs.dtype)
            history_flat = history_obs.reshape((-1,) + history_obs.shape[-3:])
            history_embedding = jax.vmap(context_embed_model)(history_flat)
            history_embedding = history_embedding.reshape(
                history_obs.shape[:3] + (-1,)
            )
            history_embedding = nn.LayerNorm(name="context_hist_ln")(history_embedding)

            clipped_history_actions = jnp.clip(
                history_actions, 0, self.action_dim - 1
            )
            history_action_oh = jax.nn.one_hot(
                clipped_history_actions, self.action_dim
            )
            history_features = jnp.concatenate(
                [history_embedding, history_action_oh],
                axis=-1,
            )
            history_features = nn.Dense(
                context_hidden_dim,
                name="context_proj_0",
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
            )(history_features)
            history_features = nn.leaky_relu(history_features)
            history_features = nn.Dense(
                context_hidden_dim,
                name="context_proj_1",
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
            )(history_features)
            history_features = nn.leaky_relu(history_features)
            context_embedding = history_features.mean(axis=2)
            context_embedding = nn.LayerNorm(name="context_ln")(context_embedding)
        else:
            context_embedding = jnp.zeros(
                policy_embedding.shape[:-1] + (context_hidden_dim,),
                dtype=policy_embedding.dtype,
            )

        prediction_input = (
            jnp.concatenate([context_obs_embedding, context_embedding], axis=-1)
            if use_history_context
            else context_obs_embedding
        )
        prediction_other = nn.Dense(
            predictor_hidden_dim,
            name="predictor_0",
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(prediction_input)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            predictor_hidden_dim,
            name="predictor_1",
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            predictor_hidden_dim,
            name="predictor_2",
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(
            predictor_hidden_dim,
            name="predictor_3",
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = nn.tanh(prediction_other)
        prediction_other = nn.Dense(
            self.action_dim,
            name="predictor_out",
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(prediction_other)
        prediction_other = prediction_other / jnp.sqrt(
            jnp.sum(prediction_other**2, axis=-1, keepdims=True) + 1e-10
        )
        other_pi = distrax.Categorical(logits=prediction_other)

        actor_input = policy_embedding
        if moa_to_actor:
            actor_condition = prediction_other
            if use_history_context:
                actor_condition = jnp.concatenate(
                    [actor_condition, context_embedding],
                    axis=-1,
                )
            if moa_to_actor_detach:
                actor_condition = jax.lax.stop_gradient(actor_condition)
            actor_input = jnp.concatenate([policy_embedding, actor_condition], axis=-1)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            name="actor_0",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_input)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            name="actor_out",
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            name="critic_0",
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(policy_embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            name="critic_out",
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1), other_pi
