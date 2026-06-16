import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from .abstract import ActorCriticBase
from .common import CNNSimple


class ActorCriticCNN(ActorCriticBase):
    """PPO CNN with TALENTS cluster-conditioned action bias.

    Paper mapping: Algorithm 1 lines 8-10 compute a cluster-specific action
    bias vector b_c and add it to actor logits. The critic intentionally stays
    cluster-agnostic and only reads the observation feature, matching the idea
    that c selects a best-response convention rather than changing task value
    estimation semantics.
    """

    @nn.compact
    def __call__(self, hidden, x):
        if len(x) == 2:
            obs, done = x
            cluster_id = None
        elif len(x) == 3:
            obs, done, cluster_id = x
        else:
            raise ValueError("ActorCriticCNN expects (obs, done) or (obs, done, cluster_id)")

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNNSimple(output_size=self.config["FC_DIM_SIZE"], activation=activation)
        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)

        actor_hidden = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        actor_hidden = activation(actor_hidden)
        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_hidden)

        if cluster_id is not None:
            num_clusters = int(self.config.get("NUM_STRATEGY_CLUSTERS", 1))
            cluster_dim = int(self.config.get("CLUSTER_EMBEDDING_DIM", 32))
            bias_weight = float(self.config.get("ACTION_BIAS_WEIGHT", 2.0))
            cluster_id = jnp.clip(cluster_id.astype(jnp.int32), 0, max(num_clusters - 1, 0))
            cluster_embedding = nn.Embed(
                num_embeddings=num_clusters,
                features=cluster_dim,
                embedding_init=orthogonal(0.01),
                name="strategy_cluster_embedding",
            )(cluster_id)
            bias = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
                name="strategy_action_bias",
            )(cluster_embedding)
            actor_logits = actor_logits + bias_weight * bias

        pi = distrax.Categorical(logits=actor_logits)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
