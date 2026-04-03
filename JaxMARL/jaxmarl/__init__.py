from types import SimpleNamespace

import jax

from .registration import make, registered_envs

if not hasattr(jax, "tree"):
    jax.tree = SimpleNamespace(
        map=jax.tree_util.tree_map,
        leaves=jax.tree_util.tree_leaves,
        flatten=jax.tree_util.tree_flatten,
        unflatten=jax.tree_util.tree_unflatten,
        structure=jax.tree_util.tree_structure,
    )

__all__ = ["make", "registered_envs"]
__version__ = "0.0.7"
