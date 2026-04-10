import abc
from typing import List, Tuple
import chex
from functools import partial
import jax
from jax.tree_util import register_pytree_node_class
from flax import struct


class AbstractPolicy(abc.ABC):
    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate, key) -> Tuple[int, chex.Array]:
        """
        Compute an action given an observation, done flag, hidden state, and random key.
        """
        pass

    def init_hstate(self, batch_size, key=None) -> chex.Array:
        return None

    def update_after_step(self, hstate, partner_obs, partner_action, done):
        return hstate


@struct.dataclass
class FunctionalPolicyState:
    value: chex.ArrayTree


@struct.dataclass
class FunctionalPolicyPairing:
    policies: Tuple[chex.ArrayTree, ...]
    backend: str = struct.field(pytree_node=False, default="")
    config: dict = struct.field(pytree_node=False, default_factory=dict)


@register_pytree_node_class
class PolicyPairing:
    policies: List[AbstractPolicy]

    def __init__(self, *policies):
        self.policies = policies

    @staticmethod
    def from_single_policy(policy: AbstractPolicy, num_agents: int):
        return PolicyPairing(*[policy for _ in range(num_agents)])

    def __getitem__(self, i):
        return self.policies[i]

    def __len__(self):
        return len(self.policies)

    def __iter__(self):
        return iter(self.policies)

    def __repr__(self):
        return f"PolicyPairing({self.policies})"

    def __str__(self):
        return f"PolicyPairing({self.policies})"

    def tree_flatten(self):
        children = self.policies
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
