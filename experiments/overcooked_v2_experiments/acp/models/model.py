import chex
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v2.common import Actions

from .abstract import ActorCriticBase
from .rnn import ActorCriticRNN, ScannedRNN


@chex.dataclass
class PartnerMemoryCarry:
    temporal_hidden: chex.Array
    partner_memory: chex.Array


def get_actor_critic(config) -> ActorCriticBase:
    model_config = config["model"]

    match model_config["TYPE"]:
        case "RNN":
            actor_critic = ActorCriticRNN
        case "CNN":
            from .cnn import ActorCriticCNN

            actor_critic = ActorCriticCNN
        case _:
            raise NotImplementedError("Only RNN and CNN models are supported.")

    return actor_critic(
        len(Actions),
        config=model_config,
    )


def initialize_carry(config, batch_size: int):
    model_config = config["model"]

    if model_config["TYPE"] == "RNN":
        return ScannedRNN.initialize_carry(batch_size, model_config["GRU_HIDDEN_DIM"])

    if model_config["TYPE"] == "CNN" and model_config.get(
        "TEMPORAL_PARTNER_ENCODER", False
    ):
        temporal_dim = model_config.get(
            "TEMPORAL_HIDDEN_DIM", model_config["FC_DIM_SIZE"]
        )
        memory_dim = model_config.get("PARTNER_MEMORY_DIM", temporal_dim)
        return PartnerMemoryCarry(
            temporal_hidden=ScannedRNN.initialize_carry(batch_size, temporal_dim),
            partner_memory=jnp.zeros((batch_size, memory_dim), dtype=jnp.float32),
        )

    return None
