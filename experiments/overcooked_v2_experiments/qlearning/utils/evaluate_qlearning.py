#!/usr/bin/env python3
"""Evaluate OV2 Q-learning checkpoints with greedy valid-action masking.

Outputs SP, XP, and all-cross CSV/Markdown summaries for IQL, VDN, and PQN-VDN
safetensors checkpoints produced by JaxMARL/baselines/QLearning.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "JaxMARL"))
sys.path.insert(0, str(ROOT / "JaxMARL" / "baselines" / "QLearning"))
sys.path.insert(0, str(ROOT / "experiments"))

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2  # noqa: E402
from jaxmarl.wrappers.baselines import load_params  # noqa: E402

import iql_cnn_overcooked  # noqa: E402
import vdn_cnn_overcooked  # noqa: E402
import pqn_vdn_cnn_overcooked  # noqa: E402


METHOD_SPECS = {
    "iql": {
        "subdir": "iql",
        "file_prefix": "iql_cnn",
        "module": iql_cnn_overcooked,
        "params_kind": "variables",
    },
    "vdn": {
        "subdir": "vdn",
        "file_prefix": "vdn_cnn",
        "module": vdn_cnn_overcooked,
        "params_kind": "variables",
    },
    "pqn_vdn": {
        "subdir": "pqn_vdn",
        "file_prefix": "pqn_vdn_cnn",
        "module": pqn_vdn_cnn_overcooked,
        "params_kind": "params_only",
    },
}


@dataclass(frozen=True)
class LoadedMethod:
    method: str
    params: List[dict]
    config: dict
    env_kwargs: dict
    apply_one: Callable
    param_labels: List[str]


def _natural_vmap_key(path: Path) -> int:
    match = re.search(r"_vmap(\d+)\.safetensors$", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _to_plain_config(config_path: Path) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    flat = dict(cfg)
    if isinstance(cfg.get("alg"), dict):
        flat.update(cfg["alg"])
    return flat


def _stack_tree(items: Sequence[dict]) -> dict:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *items)


def _masked_q(q_values: jnp.ndarray, valid_actions: jnp.ndarray) -> jnp.ndarray:
    return q_values - (1.0 - valid_actions.astype(q_values.dtype)) * 1e10


def _masked_argmax(q_values: jnp.ndarray, valid_actions: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(_masked_q(q_values, valid_actions), axis=-1).astype(jnp.int32)


def _masked_softmax_sample(
    key: jax.Array, q_values: jnp.ndarray, valid_actions: jnp.ndarray, temperature: float
) -> jnp.ndarray:
    logits = _masked_q(q_values, valid_actions) / jnp.maximum(jnp.asarray(temperature), 1e-6)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


def _masked_random_sample(key: jax.Array, valid_actions: jnp.ndarray) -> jnp.ndarray:
    logits = jnp.where(valid_actions > 0, 0.0, -1e10)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


def _select_actions(
    key: jax.Array,
    q_values: jnp.ndarray,
    valid_actions: jnp.ndarray,
    action_mode: str,
    temperature: float,
    epsilon: float,
) -> jnp.ndarray:
    greedy = _masked_argmax(q_values, valid_actions)
    if action_mode == "greedy":
        return greedy
    if action_mode == "softmax":
        return _masked_softmax_sample(key, q_values, valid_actions, temperature)
    if action_mode == "epsilon_greedy":
        key_eps, key_rand = jax.random.split(key)
        random_actions = _masked_random_sample(key_rand, valid_actions)
        explore = jax.random.uniform(key_eps, greedy.shape) < epsilon
        return jnp.where(explore, random_actions, greedy).astype(jnp.int32)
    raise ValueError(f"unknown action_mode: {action_mode}")


def load_method(run_root: Path, method: str, layout: str, max_policies: int | None) -> LoadedMethod:
    if method not in METHOD_SPECS:
        raise ValueError(f"Unknown method {method}; expected one of {sorted(METHOD_SPECS)}")
    spec = METHOD_SPECS[method]
    env_name = f"overcooked_v2_{layout}"
    method_dir = run_root / spec["subdir"] / env_name
    if not method_dir.exists():
        raise FileNotFoundError(f"Missing method dir: {method_dir}")

    config_files = sorted(method_dir.glob(f"{spec["file_prefix"]}_{env_name}_seed*_config.yaml"))
    if not config_files:
        raise FileNotFoundError(f"Missing config in {method_dir}")
    config = _to_plain_config(config_files[0])
    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    env_kwargs["layout"] = layout

    ckpts = sorted(method_dir.glob(f"{spec["file_prefix"]}_{env_name}_seed*_vmap*.safetensors"), key=_natural_vmap_key)
    if max_policies is not None:
        ckpts = ckpts[:max_policies]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {method_dir}")

    env = OvercookedV2(**env_kwargs)
    action_dim = int(env.action_space(env.agents[0]).n)

    if method in {"iql", "vdn"}:
        network = spec["module"].QNetwork(
            action_dim=action_dim,
            hidden_size=int(config.get("HIDDEN_SIZE", 64)),
        )

        def apply_one(params, obs):
            return network.apply(params, obs[jnp.newaxis, ...])[0]

    else:
        network = spec["module"].QNetwork(
            action_dim=action_dim,
            hidden_size=int(config.get("HIDDEN_SIZE", 512)),
            num_layers=int(config.get("NUM_LAYERS", 2)),
            norm_type=str(config.get("NORM_TYPE", "layer_norm")),
            norm_input=bool(config.get("NORM_INPUT", False)),
        )
        dummy_obs = jnp.zeros((1, *env.observation_space().shape), dtype=jnp.float32)
        init_vars = network.init(jax.random.PRNGKey(0), dummy_obs, train=False)
        batch_stats = init_vars.get("batch_stats", {})

        def apply_one(params, obs):
            variables = {"params": params, "batch_stats": batch_stats}
            return network.apply(variables, obs[jnp.newaxis, ...], False)[0]

    params = [load_params(p) for p in ckpts]
    labels = [f"run_{_natural_vmap_key(p)}" for p in ckpts]
    return LoadedMethod(method, params, config, env_kwargs, apply_one, labels)


def make_pairs(num_policies: int, mode: str) -> List[Tuple[int, int]]:
    if mode == "sp":
        return [(i, i) for i in range(num_policies)]
    if mode == "xp":
        return [(i, j) for i in range(num_policies) for j in range(num_policies) if i != j]
    if mode == "cross":
        return [(i, j) for i in range(num_policies) for j in range(num_policies)]
    raise ValueError(mode)


def evaluate_pairs(
    loaded: LoadedMethod,
    pairs: Sequence[Tuple[int, int]],
    num_eval_seeds: int,
    seed: int,
    pairing_batch_size: int,
    action_mode: str = "greedy",
    temperature: float = 1.0,
    epsilon: float = 0.0,
) -> List[dict]:
    """Evaluate pairings without repeating checkpoint parameters per eval seed.

    The first implementation stacked params for every (pair, seed). That is fine for
    small IQL/VDN nets but can OOM for PQN-VDN because its dense layers are much
    larger. Here each JIT call evaluates one policy pair over all eval seeds while
    keeping a single copy of each checkpoint parameter tree.
    """
    env = OvercookedV2(**loaded.env_kwargs)
    max_steps = int(env.max_steps)
    base_keys = jax.random.split(jax.random.PRNGKey(seed), num_eval_seeds)

    def q_batch(params_tree, obs_batch):
        return jax.vmap(loaded.apply_one, in_axes=(None, 0))(params_tree, obs_batch)

    def rollout_pair(params0, params1, keys):
        split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        reset_keys = split_keys[:, 0]
        scan_keys = split_keys[:, 1]
        obs, state = jax.vmap(env.reset)(reset_keys)
        total = jnp.zeros((keys.shape[0],), dtype=jnp.float32)
        step_keys = jax.vmap(lambda k: jax.random.split(k, max_steps), in_axes=0)(scan_keys)
        step_keys = jnp.swapaxes(step_keys, 0, 1)

        def step(carry, keys_t):
            obs_t, state_t, total_t = carry
            q0 = q_batch(params0, obs_t["agent_0"])
            q1 = q_batch(params1, obs_t["agent_1"])
            valid = {"agent_0": jnp.ones_like(q0), "agent_1": jnp.ones_like(q1)}
            key0, key1 = jax.random.split(keys_t[0], 2)
            a0 = _select_actions(key0, q0, valid["agent_0"], action_mode, temperature, epsilon)
            a1 = _select_actions(key1, q1, valid["agent_1"], action_mode, temperature, epsilon)
            actions = {"agent_0": a0, "agent_1": a1}
            next_obs, next_state, reward, _done, _info = jax.vmap(env.step)(keys_t, state_t, actions)
            return (next_obs, next_state, total_t + reward["agent_0"]), None

        (_, _, total), _ = jax.lax.scan(step, (obs, state, total), step_keys)
        return total

    jit_rollout = jax.jit(rollout_pair)
    rows: List[dict] = []
    params = loaded.params
    labels = loaded.param_labels
    keys = jnp.asarray(base_keys)

    del pairing_batch_size  # kept in CLI for compatibility with older invocations.
    for pair_idx, (i, j) in enumerate(pairs):
        rewards = np.asarray(jax.block_until_ready(jit_rollout(params[i], params[j], keys)))
        for seed_idx, reward in enumerate(rewards):
            rows.append(
                {
                    "method": loaded.method,
                    "pairing": f"{labels[i]}__{labels[j]}",
                    "agent0": labels[i],
                    "agent1": labels[j],
                    "eval_seed_index": seed_idx,
                    "total_reward": float(reward),
                }
            )
        if (pair_idx + 1) % 10 == 0 or (pair_idx + 1) == len(pairs):
            print(
                f"[qlearning-eval] evaluated {pair_idx + 1}/{len(pairs)} pairings",
                flush=True,
            )
    return rows

def write_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "pairing", "agent0", "agent1", "eval_seed_index", "total_reward"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> dict:
    rewards = np.array([r["total_reward"] for r in rows], dtype=np.float64)
    pair_values: Dict[str, List[float]] = {}
    for r in rows:
        pair_values.setdefault(r["pairing"], []).append(float(r["total_reward"]))
    pair_means = np.array([np.mean(v) for v in pair_values.values()], dtype=np.float64)
    return {
        "mean": float(np.mean(rewards)) if rewards.size else float("nan"),
        "std_episode": float(np.std(rewards)) if rewards.size else float("nan"),
        "std_pair": float(np.std(pair_means)) if pair_means.size else float("nan"),
        "num_pairs": int(len(pair_values)),
        "num_episodes": int(len(rows)),
    }


def write_summary(out_dir: Path, method: str, summaries: Dict[str, dict]) -> None:
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "mode", "mean", "std_episode", "std_pair", "num_pairs", "num_episodes"],
        )
        writer.writeheader()
        for mode, s in summaries.items():
            writer.writerow({"method": method, "mode": mode, **s})

    md_path = out_dir / "summary.md"
    with md_path.open("w") as f:
        f.write(f"# QLearning OV2 evaluation: {method}\n\n")
        f.write("| mode | mean | std_episode | std_pair | num_pairs | num_episodes |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for mode, s in summaries.items():
            f.write(
                "| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |\n".format(
                    mode,
                    s["mean"],
                    s["std_episode"],
                    s["std_pair"],
                    s["num_pairs"],
                    s["num_episodes"],
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=sorted(METHOD_SPECS))
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_seeds", type=int, default=500)
    parser.add_argument("--pairing_batch_size", type=int, default=10)
    parser.add_argument("--max_policies", type=int, default=None)
    parser.add_argument("--modes", default="sp,xp,cross")
    parser.add_argument("--action_mode", choices=["greedy", "softmax", "epsilon_greedy"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = args.output_dir or (args.run_root / "eval" / args.method)
    loaded = load_method(args.run_root, args.method, args.layout, args.max_policies)
    print(
        f"[qlearning-eval] method={args.method} policies={len(loaded.params)} seeds={args.num_eval_seeds} action_mode={args.action_mode} temperature={args.temperature} epsilon={args.epsilon} out={out_dir}",
        flush=True,
    )

    summaries: Dict[str, dict] = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        pairs = make_pairs(len(loaded.params), mode)
        print(f"[qlearning-eval] mode={mode} pairs={len(pairs)}", flush=True)
        rows = evaluate_pairs(
            loaded,
            pairs,
            num_eval_seeds=args.num_eval_seeds,
            seed=args.seed,
            pairing_batch_size=args.pairing_batch_size,
            action_mode=args.action_mode,
            temperature=args.temperature,
            epsilon=args.epsilon,
        )
        write_rows(out_dir / f"reward_summary_{mode}.csv", rows)
        summaries[mode] = summarize(rows)
        print(f"[qlearning-eval] mode={mode} summary={summaries[mode]}", flush=True)

    write_summary(out_dir, args.method, summaries)
    print(f"[qlearning-eval] wrote {out_dir / "summary.md"}", flush=True)


if __name__ == "__main__":
    main()
