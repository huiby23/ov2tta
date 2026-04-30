import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal, xavier_uniform

from .abstract import ActorCriticBase


class ActorCriticCNN(ActorCriticBase):
    """JAX/Flax port of the official E3T encoder policy architecture.

    Official mapping:
    - context_encoder: conv_and_mlp_embd(pre_obs, action_reward)
    - human_prob_pre: 2-layer tanh MLP over context embedding
    - decoder: conv_and_mlp_new(obs, human_latent_pre) -> partner action logits
    - pi/vf: conv_and_mlp(obs, softmax(partner logits)) -> actor/value
    - step_human: separate human policy branch conditioned on softmax(context_human)
    """

    def _extract_inputs(self, x):
        if len(x) == 2:
            obs, done = x
            return obs, done, None, None, None
        if len(x) == 4:
            obs, done, history_obs, history_actions = x
            return obs, done, history_obs, history_actions, None
        if len(x) == 5:
            obs, done, history_obs, history_actions, context_human = x
            return obs, done, history_obs, history_actions, context_human
        raise ValueError(
            "E3T ActorCriticCNN expects (obs, done), "
            "(obs, done, history_obs, history_actions), or "
            "(obs, done, history_obs, history_actions, context_human)."
        )

    def _cfg_int(self, key, default):
        return int(self.config.get(key, default))

    def _conv_flat(self, flat_obs, prefix, context_encoder=False):
        num_filters = self._cfg_int("E3T_NUM_FILTERS", 25)
        num_convs = self._cfg_int("E3T_NUM_CONV_LAYERS", 3)

        x = nn.Conv(
            features=num_filters,
            kernel_size=(5, 5),
            padding="SAME",
            name=f"{prefix}_conv_initial",
            kernel_init=xavier_uniform(),
            bias_init=constant(0.0),
        )(flat_obs)
        x = nn.leaky_relu(x, negative_slope=0.2)

        for i in range(num_convs - 1):
            padding = "SAME" if i < num_convs - 2 else "VALID"
            x = nn.Conv(
                features=num_filters,
                kernel_size=(3, 3),
                padding=padding,
                name=f"{prefix}_conv_{i}",
                kernel_init=xavier_uniform(),
                bias_init=constant(0.0),
            )(x)
            if not (context_encoder and i == num_convs - 2):
                x = nn.leaky_relu(x, negative_slope=0.2)

        return x.reshape((x.shape[0], -1))

    def _context_encoder(self, history_obs, history_actions):
        latent_dim = self._cfg_int("LATENT_DIM", 64)
        num_hidden_layers = self._cfg_int("E3T_NUM_HIDDEN_LAYERS", 3)
        size_hidden_layers = self._cfg_int("E3T_SIZE_HIDDEN_LAYERS", 64)
        action_dim = int(self.action_dim)

        lead_shape = history_obs.shape[:-4]
        length = history_obs.shape[-4]
        flat_obs = history_obs.reshape((-1,) + history_obs.shape[-3:])
        out = self._conv_flat(flat_obs, "context", context_encoder=True)

        flat_actions = history_actions.reshape((-1, length))
        action_oh = jax.nn.one_hot(
            jnp.clip(flat_actions.reshape((-1,)), 0, action_dim - 1), action_dim
        )
        action_emb = nn.Dense(
            action_dim,
            name="context_word_embd",
            kernel_init=xavier_uniform(),
            bias_init=constant(0.0),
        )(action_oh)
        out = jnp.concatenate([out, action_emb.reshape((out.shape[0], -1))], axis=-1)
        out = nn.leaky_relu(out, negative_slope=0.2)

        for i in range(num_hidden_layers - 1):
            out = nn.Dense(
                size_hidden_layers,
                name=f"context_dense_{i}",
                kernel_init=xavier_uniform(),
                bias_init=constant(0.0),
            )(out)
            out = nn.leaky_relu(out, negative_slope=0.2)
        out = nn.Dense(
            latent_dim,
            name="context_dense_out",
            kernel_init=xavier_uniform(),
            bias_init=constant(0.0),
        )(out)
        out = nn.leaky_relu(out, negative_slope=0.2)

        if length > 1 and "MLP" in self.config.get("TRAIN_MODE", "MLP"):
            out = out.reshape((-1, length * latent_dim))
            for i in range(num_hidden_layers):
                out = nn.Dense(
                    latent_dim,
                    name=f"context_temporal_dense_{i}",
                    kernel_init=xavier_uniform(),
                    bias_init=constant(0.0),
                )(out)
                out = nn.leaky_relu(out, negative_slope=0.2)
        else:
            out = out.reshape((-1, length, latent_dim)).mean(axis=1)

        return out.reshape(lead_shape + (latent_dim,))

    def _human_prob_pre(self, encoded_context):
        latent_dim = self._cfg_int("LATENT_DIM", 64)
        x = encoded_context
        for i in range(2):
            x = nn.Dense(
                latent_dim,
                name=f"human_prob_pre_{i}",
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
            )(x)
            x = nn.tanh(x)
        return x

    def _decoder(self, obs, human_latent_pre):
        latent_dim = self._cfg_int("LATENT_DIM", 64)
        num_hidden_layers = self._cfg_int("E3T_NUM_HIDDEN_LAYERS", 3)
        size_hidden_layers = self._cfg_int("E3T_SIZE_HIDDEN_LAYERS", 64)

        lead_shape = obs.shape[:-3]
        flat_obs = obs.reshape((-1,) + obs.shape[-3:])
        out = self._conv_flat(flat_obs, "decoder", context_encoder=False)
        latent = human_latent_pre.reshape((out.shape[0], -1))
        out = jnp.concatenate([out, latent], axis=-1)

        for i in range(num_hidden_layers - 1):
            out = nn.Dense(
                size_hidden_layers,
                name=f"decoder_dense_{i}",
                kernel_init=xavier_uniform(),
                bias_init=constant(0.0),
            )(out)
            out = nn.leaky_relu(out, negative_slope=0.2)
        out = nn.Dense(
            latent_dim,
            name="decoder_dense_out",
            kernel_init=xavier_uniform(),
            bias_init=constant(0.0),
        )(out)
        out = nn.leaky_relu(out, negative_slope=0.2)
        out = out.reshape(lead_shape + (latent_dim,))

        if self.config.get("E3T_CONTEXT_MODE", False):
            return out, None

        logits = nn.Dense(
            int(self.action_dim),
            name="decoder_pi",
            kernel_init=xavier_uniform(),
            bias_init=constant(0.0),
        )(out)
        return out, logits

    def _policy_network(self, obs, condition, prefix):
        num_hidden_layers = self._cfg_int("E3T_NUM_HIDDEN_LAYERS", 3)
        size_hidden_layers = self._cfg_int("E3T_SIZE_HIDDEN_LAYERS", 64)

        lead_shape = obs.shape[:-3]
        flat_obs = obs.reshape((-1,) + obs.shape[-3:])
        out = self._conv_flat(flat_obs, prefix, context_encoder=False)
        cond = condition.reshape((out.shape[0], -1))
        out = jnp.concatenate([out, cond], axis=-1)

        for i in range(num_hidden_layers):
            out = nn.Dense(
                size_hidden_layers,
                name=f"{prefix}_dense_{i}",
                kernel_init=xavier_uniform(),
                bias_init=constant(0.0),
            )(out)
            out = nn.leaky_relu(out, negative_slope=0.2)
        return out.reshape(lead_shape + (size_hidden_layers,))

    @nn.compact
    def __call__(self, hidden, x):
        obs, done, history_obs, history_actions, context_human = self._extract_inputs(x)
        del done
        obs = obs.astype(jnp.float32)

        action_dim = int(self.action_dim)

        if history_obs is None or history_actions is None:
            lead_shape = obs.shape[:-3]
            context_length = self._cfg_int("CONTEXT_LENGTH", 5)
            history_obs = jnp.repeat(obs[..., jnp.newaxis, :, :, :], context_length, axis=-4)
            history_actions = jnp.full(
                lead_shape + (context_length,),
                self._cfg_int("STAY_ACTION", 4),
                dtype=jnp.int32,
            )

        encoded_context = self._context_encoder(history_obs.astype(obs.dtype), history_actions)
        human_latent_pre = self._human_prob_pre(encoded_context)
        decoder_latent, action_logits = self._decoder(obs, human_latent_pre)

        actor_condition_mode = str(
            self.config.get("E3T_ACTOR_CONDITION", "predicted_partner")
        ).lower()
        if self.config.get("E3T_CONTEXT_MODE", False):
            policy_condition = decoder_latent / (
                jnp.linalg.norm(decoder_latent, axis=-1, keepdims=True) + 1e-8
            )
            if actor_condition_mode in ("constant", "dummy", "none"):
                policy_condition = jnp.zeros_like(policy_condition)
        else:
            policy_condition = jax.nn.softmax(action_logits, axis=-1)
            if actor_condition_mode in ("constant", "dummy", "none"):
                policy_condition = jnp.ones_like(policy_condition) / action_dim

        if actor_condition_mode not in (
            "predicted_partner",
            "predicted",
            "constant",
            "dummy",
            "none",
        ):
            raise ValueError(f"Unknown E3T_ACTOR_CONDITION={actor_condition_mode}")

        policy_latent = self._policy_network(obs, policy_condition, "policy")
        actor_logits = nn.Dense(
            action_dim,
            name="actor_out",
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(policy_latent)
        pi = distrax.Categorical(logits=actor_logits)

        value = nn.Dense(
            1,
            name="critic_out",
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(policy_latent)

        if context_human is None:
            context_human = jnp.zeros(obs.shape[:-3] + (action_dim,), dtype=obs.dtype)
        if self.config.get("E3T_CONTEXT_MODE", False):
            human_condition = context_human / (
                jnp.linalg.norm(context_human, axis=-1, keepdims=True) + 1e-8
            )
        else:
            human_condition = jax.nn.softmax(context_human, axis=-1)

        human_policy_latent = self._policy_network(obs, human_condition, "human_policy")
        human_logits = nn.Dense(
            action_dim,
            name="human_actor_out",
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(human_policy_latent)
        _ = nn.Dense(
            1,
            name="human_critic_out",
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(human_policy_latent)
        human_pi = distrax.Categorical(logits=human_logits)

        partner_pi = distrax.Categorical(logits=action_logits)
        return hidden, pi, jnp.squeeze(value, axis=-1), partner_pi, human_pi
