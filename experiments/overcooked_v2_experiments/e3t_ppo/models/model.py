from jaxmarl.environments.overcooked_v2.common import Actions

from .abstract import ActorCriticBase
from .rnn import ActorCriticRNN, ScannedRNN


def get_actor_critic(config) -> ActorCriticBase:
    model_config = config["model"]
    if model_config["TYPE"] != "RNN":
        raise NotImplementedError("E3T-PPO currently supports only the RNN model.")
    return ActorCriticRNN(len(Actions), config=model_config)


def initialize_carry(config, batch_size: int):
    model_config = config["model"]
    if model_config["TYPE"] != "RNN":
        return None
    return ScannedRNN.initialize_carry(batch_size, model_config["GRU_HIDDEN_DIM"])
