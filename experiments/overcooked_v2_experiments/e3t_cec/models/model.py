import copy

from jaxmarl.environments.overcooked_v2.common import Actions

from .abstract import ActorCriticBase
from .rnn import ActorCriticRNN, ScannedRNN


def _get_rnn_hidden_size(config) -> int:
    model_config = config["model"]
    env_name = config["env"]["ENV_NAME"]
    is_overcooked = env_name in {"overcooked", "overcooked_v2"}
    return model_config["FC_DIM_SIZE"] * 2 if is_overcooked else model_config["FC_DIM_SIZE"]


def get_actor_critic(config) -> ActorCriticBase:
    model_config = copy.deepcopy(config["model"])
    env_config = config.get("env", {})
    model_config.setdefault("ENV_NAME", env_config.get("ENV_NAME"))
    env_kwargs = env_config.get("ENV_KWARGS", {})
    model_config.setdefault("LAYOUT_NAME", env_kwargs.get("layout"))
    model_config.setdefault("OBS_SHAPE", env_kwargs.get("obs_shape"))
    if model_config["TYPE"] != "RNN":
        raise NotImplementedError("E3T-CEC currently supports the RNN configuration only.")
    return ActorCriticRNN(len(Actions), config=model_config)


def initialize_carry(config, batch_size: int):
    model_config = config["model"]
    if model_config["TYPE"] != "RNN":
        return None
    return ScannedRNN.initialize_carry(batch_size, _get_rnn_hidden_size(config))
