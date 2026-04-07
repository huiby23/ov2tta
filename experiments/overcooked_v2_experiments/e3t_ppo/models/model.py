from jaxmarl.environments.overcooked_v2.common import Actions

from .abstract import ActorCriticBase
from .cnn import ActorCriticCNN
from .rnn import ActorCriticRNN, ScannedRNN


def get_actor_critic(config) -> ActorCriticBase:
    model_config = config["model"]
    if model_config["TYPE"] == "RNN":
        return ActorCriticRNN(len(Actions), config=model_config)
    if model_config["TYPE"] == "CNN":
        return ActorCriticCNN(len(Actions), config=model_config)
    raise NotImplementedError(
        f"E3T-PPO does not support model type {model_config['TYPE']!r}."
    )


def initialize_carry(config, batch_size: int):
    model_config = config["model"]
    if model_config["TYPE"] != "RNN":
        return None
    return ScannedRNN.initialize_carry(batch_size, model_config["GRU_HIDDEN_DIM"])
