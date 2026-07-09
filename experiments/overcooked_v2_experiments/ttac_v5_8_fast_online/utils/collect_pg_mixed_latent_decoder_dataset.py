from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import init_rollout
from overcooked_v2_experiments.mappo.policy import MAPPOPolicy
from overcooked_v2_experiments.mappo.utils.store import (
    load_all_checkpoints as load_mappo_checkpoints,
)
from overcooked_v2_experiments.ppo.policy import PPOPolicy as BasePPOPolicy
from overcooked_v2_experiments.ppo.utils.store import (
    load_all_checkpoints as load_base_ppo_checkpoints,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.policy import PPOPolicy
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.store import (
    load_all_checkpoints as load_ppo_checkpoints,
)
from overcooked_v2_experiments.qlearning.utils.evaluate_mixed_1zsc import (
    QLearningPolicy,
    load_q_method,
)


@dataclass(frozen=True)
class PolicyEntry:
    pool: str
    label: str
    policy: object


def parse_policy_indices(spec: str | None):
    if spec is None or not str(spec).strip():
        return None
    return {int(x.strip()) for x in str(spec).split(",") if x.strip()}


def load_ppo_pool(
    run_dir: Path,
    max_policies: int | None,
    stochastic: bool,
    pool_name: str = "ppo",
    backend: str = "ttac",
    policy_indices: set[int] | None = None,
):
    if backend == "ttac":
        all_ckpts, config = load_ppo_checkpoints(run_dir, final_only=True)
        policy_ctor = lambda params, cfg: PPOPolicy(  # noqa: E731
            params, cfg, stochastic=stochastic, eval_mode="base_no_test_adapt"
        )
    elif backend == "base":
        all_ckpts, config = load_base_ppo_checkpoints(run_dir, final_only=True)
        policy_ctor = lambda params, cfg: BasePPOPolicy(params, cfg, stochastic=stochastic)  # noqa: E731
    else:
        raise ValueError(f"unknown PPO backend: {backend}")
    run_keys = sorted(all_ckpts.keys(), key=lambda x: int(x.split("_")[1]))
    if policy_indices is not None:
        run_keys = [k for k in run_keys if int(k.split("_")[1]) in policy_indices]
    elif max_policies is not None:
        run_keys = run_keys[: max(0, int(max_policies))]
    entries = []
    for run_key in run_keys:
        ckpt = all_ckpts[run_key].get("ckpt_final")
        if ckpt is None:
            continue
        entries.append(
            PolicyEntry(
                pool_name,
                run_key,
                policy_ctor(ckpt.params, config),
            )
        )
    if not entries:
        raise FileNotFoundError(f"no PPO-compatible policies loaded from {run_dir}")
    return entries, config


def parse_extra_ppo_pool_spec(spec: str) -> tuple[str, Path, int | None]:
    parts = [p for p in spec.split(":") if p]
    if len(parts) < 2:
        raise ValueError(
            "extra PPO-compatible pools must be formatted as name:path or name:path:max_policies"
        )
    pool_name = parts[0]
    max_policies = None
    if len(parts) > 2 and parts[-1].isdigit():
        max_policies = int(parts[-1])
        path = ":".join(parts[1:-1])
    else:
        path = ":".join(parts[1:])
    if not pool_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid pool name: {pool_name}")
    return pool_name, Path(path), max_policies


def load_mappo_pool(
    run_dir: Path,
    max_policies: int | None,
    stochastic: bool,
    policy_indices: set[int] | None = None,
):
    all_ckpts, config = load_mappo_checkpoints(run_dir, final_only=True)
    run_keys = sorted(all_ckpts.keys(), key=lambda x: int(x.split("_")[1]))
    if policy_indices is not None:
        run_keys = [k for k in run_keys if int(k.split("_")[1]) in policy_indices]
    elif max_policies is not None:
        run_keys = run_keys[: max(0, int(max_policies))]
    entries = []
    for run_key in run_keys:
        ckpt = all_ckpts[run_key].get("ckpt_final")
        if ckpt is None:
            continue
        entries.append(PolicyEntry("mappo", run_key, MAPPOPolicy(ckpt.params, config, stochastic=stochastic)))
    if not entries:
        raise FileNotFoundError(f"no MAPPO policies loaded from {run_dir}")
    return entries, config


def get_rollout_seq(policies: PolicyPairing, env, key):
    init_hstate, get_actions = init_rollout(policies, env)

    @jax.jit
    def perform_step(carry, key):
        obs, state, done, total_reward, hstate = carry
        key_sample, key_step = jax.random.split(key, 2)
        actions, next_hstate = get_actions(obs, done, hstate, key_sample)
        next_obs, next_state, reward, next_done, _info = env.step(key_step, state, actions)
        if env.num_agents == 2:
            updated_hstate = {}
            for i, policy in enumerate(policies):
                agent_id = f"agent_{i}"
                partner_id = f"agent_{1-i}"
                updated_hstate[agent_id] = policy.update_after_step(
                    next_hstate[agent_id], obs[partner_id], actions[partner_id], next_done[agent_id]
                )
        else:
            updated_hstate = next_hstate
        carry = (next_obs, next_state, next_done, total_reward + reward["agent_0"], updated_hstate)
        return carry, (obs, done, actions, reward["agent_0"])

    key, key_reset = jax.random.split(key, 2)
    obs, state = env.reset(key_reset)
    done = {f"agent_{i}": False for i in range(env.num_agents)}
    done["__all__"] = False
    keys = jax.random.split(key, env.max_steps)
    carry = (obs, state, done, 0.0, init_hstate)
    carry, seq = jax.lax.scan(perform_step, carry, keys)
    return seq, carry[-2]


def _action_history(actions, idx, history_len):
    hist = np.zeros((len(idx), history_len), dtype=np.int16)
    for row, t in enumerate(idx):
        t = int(t)
        start = max(0, t - history_len)
        seq = actions[start:t]
        if len(seq):
            hist[row, -len(seq):] = seq
    return hist


def _obs_history(obs, idx, history_len):
    obs = np.asarray(obs)
    hist = np.zeros((len(idx), history_len) + obs.shape[1:], dtype=np.float16)
    for row, t in enumerate(idx):
        t = int(t)
        start = max(0, t - history_len)
        seq = obs[start:t]
        if len(seq):
            hist[row, -len(seq):] = seq.astype(np.float16)
    return hist


def policy_probs_on_query_obs(policy, query_obs: np.ndarray, batch_size: int = 1024):
    outs = []
    for start in range(0, len(query_obs), batch_size):
        chunk = jnp.asarray(query_obs[start:start + batch_size], dtype=jnp.float32)
        if hasattr(policy, "_apply_batch"):
            logits, _value, _aux = policy._apply_batch(policy.params, chunk, adapter_readout_scale=0.0)
            probs = jax.nn.softmax(logits, axis=-1)
        elif isinstance(policy, QLearningPolicy):
            if policy.is_rnn:
                raise ValueError("RNN Q policies are not supported as latent-decoder teachers.")

            def apply_obs(obs):
                obs = jnp.asarray(obs, dtype=jnp.float32)
                if policy.preprocess_flat_obs:
                    agent_id = jax.nn.one_hot(policy.agent_index, policy.num_agents, dtype=jnp.float32)
                    obs = jnp.concatenate([jnp.ravel(obs), agent_id], axis=-1)
                return policy.apply_one(policy.params, obs)

            q_values = jax.vmap(apply_obs)(chunk)
            valid_actions = jnp.ones_like(q_values)
            masked_q = QLearningPolicy._masked_q(q_values, valid_actions)
            greedy = jax.nn.one_hot(jnp.argmax(masked_q, axis=-1), masked_q.shape[-1])
            if policy.action_mode == "softmax":
                probs = jax.nn.softmax(masked_q / max(float(policy.temperature), 1e-6), axis=-1)
            elif policy.action_mode == "epsilon_greedy":
                uniform = valid_actions / jnp.maximum(valid_actions.sum(axis=-1, keepdims=True), 1.0)
                probs = (1.0 - float(policy.epsilon)) * greedy + float(policy.epsilon) * uniform
            else:
                probs = greedy
        else:
            done = jnp.zeros((chunk.shape[0],), dtype=jnp.bool_)
            probs, _value, _hstate = policy.forward_diagnostics_batch(chunk, done)
        outs.append(np.asarray(probs, dtype=np.float32))
    return np.concatenate(outs, axis=0)


def load_q_pool(run_root: Path, methods: str, layout: str, max_policies: int | None, action_mode: str, temperature: float, epsilon: float):
    entries = []
    for method in [x.strip() for x in methods.split(",") if x.strip()]:
        loaded = load_q_method(run_root, method, layout, max_policies)
        if loaded.is_rnn:
            print(f"[collect_pg_mixed] skip RNN Q method for teacher dataset: {method}", flush=True)
            continue
        for label, params in zip(loaded.labels, loaded.params):
            entries.append(
                PolicyEntry(
                    method,
                    label,
                    QLearningPolicy(
                        params,
                        loaded.apply_one,
                        action_mode=action_mode,
                        temperature=temperature,
                        epsilon=epsilon,
                        is_rnn=False,
                        hidden_size=None,
                    ),
                )
            )
    return entries


def append_role(
    rows,
    query_obs,
    partner_obs,
    ego_action,
    partner_action,
    step_reward,
    episode_return,
    partner_target_probs,
    pair_label,
    episode_id,
    role,
    stride,
    history_len,
    ego_pool,
    partner_pool,
    ego_policy_id,
    partner_policy_id,
):
    ego_action = np.asarray(ego_action, dtype=np.int16)
    partner_action = np.asarray(partner_action, dtype=np.int16)
    step_reward = np.asarray(step_reward, dtype=np.float32)
    return_to_go = np.cumsum(step_reward[::-1], dtype=np.float32)[::-1]
    idx = np.arange(0, len(ego_action), stride, dtype=np.int32)
    prev_partner = np.concatenate([[0], partner_action[:-1]]).astype(np.int16)
    prev_ego = np.concatenate([[0], ego_action[:-1]]).astype(np.int16)
    rows["query_obs"].append(np.asarray(query_obs, dtype=np.float16)[idx])
    rows["partner_obs"].append(np.asarray(partner_obs, dtype=np.float16)[idx])
    rows["ego_action"].append(ego_action[idx])
    rows["partner_action"].append(partner_action[idx])
    rows["prev_partner_action"].append(prev_partner[idx])
    rows["prev_ego_action"].append(prev_ego[idx])
    rows["partner_action_history"].append(_action_history(partner_action, idx, history_len))
    rows["ego_action_history"].append(_action_history(ego_action, idx, history_len))
    rows["partner_obs_history"].append(_obs_history(partner_obs, idx, history_len))
    rows["target_partner_probs"].append(np.asarray(partner_target_probs, dtype=np.float32)[idx])
    rows["step_reward"].append(step_reward[idx].astype(np.float32))
    rows["return_to_go"].append(return_to_go[idx].astype(np.float32))
    rows["episode_return"].append(
        np.full(len(idx), float(episode_return), dtype=np.float32)
    )
    rows["pair_label"].extend([pair_label] * len(idx))
    rows["ego_pool"].extend([ego_pool] * len(idx))
    rows["partner_pool"].extend([partner_pool] * len(idx))
    rows["ego_family"].extend([ego_pool] * len(idx))
    rows["partner_family"].extend([partner_pool] * len(idx))
    rows["ego_policy_id"].extend([ego_policy_id] * len(idx))
    rows["partner_policy_id"].extend([partner_policy_id] * len(idx))
    rows["episode_id"].append(np.full(len(idx), episode_id, dtype=np.int32))
    rows["timestep"].append(idx.astype(np.int16))
    rows["role"].append(np.full(len(idx), role, dtype=np.int8))


def cap_rows_in_memory(rows, max_transitions, rng):
    if not max_transitions or max_transitions <= 0:
        return
    total = len(rows["pair_label"])
    if total <= max_transitions:
        return
    keep = np.sort(rng.choice(total, size=max_transitions, replace=False))
    string_keys = {
        "pair_label",
        "ego_pool",
        "partner_pool",
        "ego_family",
        "partner_family",
        "ego_policy_id",
        "partner_policy_id",
    }
    for key, parts in list(rows.items()):
        if key in string_keys:
            rows[key] = np.asarray(parts)[keep].tolist()
        else:
            rows[key] = [np.concatenate(parts, axis=0)[keep]]


def maybe_shuffle_and_cap(candidates, max_pairings, rng):
    candidates = list(candidates)
    if rng is not None and len(candidates) > 1:
        rng.shuffle(candidates)
    if max_pairings is not None:
        candidates = candidates[: max(0, int(max_pairings))]
    return candidates


def build_pair_blocks(pg_pools, q_entries, max_pairings_per_block, max_q_pairings_per_block, rng=None):
    pairs = []
    for lhs_name, lhs in pg_pools:
        for rhs_name, rhs in pg_pools:
            if not lhs or not rhs:
                continue
            block_name = f"{lhs_name}_{rhs_name}"
            same_pool = lhs_name == rhs_name
            if same_pool:
                candidates = [(lhs[i], rhs[j]) for i, j in itertools.permutations(range(len(lhs)), 2)]
            else:
                candidates = list(itertools.product(lhs, rhs))
            candidates = maybe_shuffle_and_cap(candidates, max_pairings_per_block, rng)
            pairs.extend((block_name, left, right) for left, right in candidates)
    if q_entries:
        pg_entries = [entry for _name, entries in pg_pools for entry in entries]
        q_blocks = [
            ("pg_q", pg_entries, q_entries),
            ("q_pg", q_entries, pg_entries),
            ("q_q", q_entries, q_entries),
        ]
        for block_name, lhs, rhs in q_blocks:
            candidates = list(itertools.product(lhs, rhs))
            if block_name == "q_q":
                candidates = [(l, r) for l, r in candidates if (l.pool, l.label) != (r.pool, r.label)]
            candidates = maybe_shuffle_and_cap(candidates, max_q_pairings_per_block, rng)
            pairs.extend((block_name, left, right) for left, right in candidates)
    return pairs


def build_pair_blocks_legacy(ppo_entries, mappo_entries, q_entries, max_pairings_per_block, max_q_pairings_per_block, rng=None):
    blocks = [
        ("ppo_ppo", ppo_entries, ppo_entries, True),
        ("mappo_mappo", mappo_entries, mappo_entries, True),
        ("ppo_mappo", ppo_entries, mappo_entries, False),
        ("mappo_ppo", mappo_entries, ppo_entries, False),
    ]
    pairs = []
    for block_name, lhs, rhs, same_pool in blocks:
        if same_pool:
            candidates = [(lhs[i], rhs[j]) for i, j in itertools.permutations(range(len(lhs)), 2)]
        else:
            candidates = list(itertools.product(lhs, rhs))
        candidates = maybe_shuffle_and_cap(candidates, max_pairings_per_block, rng)
        pairs.extend((block_name, left, right) for left, right in candidates)
    if q_entries:
        pg_entries = ppo_entries + mappo_entries
        q_blocks = [
            ("pg_q", pg_entries, q_entries),
            ("q_pg", q_entries, pg_entries),
            ("q_q", q_entries, q_entries),
        ]
        for block_name, lhs, rhs in q_blocks:
            candidates = list(itertools.product(lhs, rhs))
            if block_name == "q_q":
                candidates = [(l, r) for l, r in candidates if (l.pool, l.label) != (r.pool, r.label)]
            candidates = maybe_shuffle_and_cap(candidates, max_q_pairings_per_block, rng)
            pairs.extend((block_name, left, right) for left, right in candidates)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppo_run_dir",
        type=Path,
        default=Path("runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606"),
    )
    parser.add_argument(
        "--mappo_run_dir",
        type=Path,
        default=Path("runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_standard_20260503-201254/20260503-201315_wvbb7dde_counter_circuit_avs-full"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_seeds", type=int, default=50)
    parser.add_argument("--max_ppo_policies", type=int, default=10)
    parser.add_argument("--max_mappo_policies", type=int, default=10)
    parser.add_argument("--ppo_policy_indices", type=str)
    parser.add_argument("--mappo_policy_indices", type=str)
    parser.add_argument("--max_pairings_per_block", type=int, default=20)
    parser.add_argument(
        "--extra_ppo_pool",
        action="append",
        default=[],
        help="Additional PPO-compatible pool formatted as name:path or name:path:max_policies. Repeatable.",
    )
    parser.add_argument(
        "--all_pool_pairing",
        action="store_true",
        help="Pair every non-Q pool with every non-Q pool, capped per ordered family block.",
    )
    parser.add_argument(
        "--shuffle_pairings",
        action="store_true",
        help="Deterministically shuffle candidates within each ordered family block before applying caps.",
    )
    parser.add_argument("--q_run_root", type=Path, default=Path("runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M"))
    parser.add_argument("--q_methods", type=str, default="")
    parser.add_argument("--max_q_policies", type=int, default=3)
    parser.add_argument("--max_q_pairings_per_block", type=int, default=20)
    parser.add_argument("--q_action_mode", type=str, default="greedy", choices=("greedy", "softmax", "epsilon_greedy"))
    parser.add_argument("--q_temperature", type=float, default=1.0)
    parser.add_argument("--q_epsilon", type=float, default=0.0)
    parser.add_argument("--transition_stride", type=int, default=4)
    parser.add_argument("--max_transitions", type=int, default=300000)
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--policy_prob_batch_size", type=int, default=1024)
    parser.add_argument("--deterministic_rollout", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ppo_entries, ppo_config = load_ppo_pool(
        args.ppo_run_dir,
        args.max_ppo_policies,
        stochastic=not args.deterministic_rollout,
        policy_indices=parse_policy_indices(args.ppo_policy_indices),
    )
    mappo_entries, mappo_config = load_mappo_pool(
        args.mappo_run_dir,
        args.max_mappo_policies,
        stochastic=not args.deterministic_rollout,
        policy_indices=parse_policy_indices(args.mappo_policy_indices),
    )
    env_kwargs = dict(ppo_config["env"]["ENV_KWARGS"])
    mappo_layout = str(mappo_config["env"]["ENV_KWARGS"].get("layout", env_kwargs.get("layout")))
    if mappo_layout != str(env_kwargs.get("layout")):
        raise ValueError(f"PPO layout {env_kwargs.get('layout')} does not match MAPPO layout {mappo_layout}")
    pg_pools = [("ppo", ppo_entries), ("mappo", mappo_entries)]
    extra_pool_specs = [parse_extra_ppo_pool_spec(spec) for spec in args.extra_ppo_pool]
    for pool_name, run_dir, max_policies in extra_pool_specs:
        extra_entries, extra_config = load_ppo_pool(
            run_dir,
            max_policies,
            stochastic=not args.deterministic_rollout,
            pool_name=pool_name,
            backend="base",
        )
        extra_layout = str(extra_config["env"]["ENV_KWARGS"].get("layout", env_kwargs.get("layout")))
        if extra_layout != str(env_kwargs.get("layout")):
            raise ValueError(
                f"pool {pool_name} layout {extra_layout} does not match PPO layout {env_kwargs.get('layout')}"
            )
        pg_pools.append((pool_name, extra_entries))
        print(
            f"[collect_pg_mixed] loaded extra pool {pool_name} policies={len(extra_entries)} from {run_dir}",
            flush=True,
        )
    env = OvercookedV2(**env_kwargs)
    q_entries = load_q_pool(
        args.q_run_root,
        args.q_methods,
        str(env_kwargs.get("layout", "counter_circuit")),
        args.max_q_policies,
        args.q_action_mode,
        args.q_temperature,
        args.q_epsilon,
    )
    pair_rng = np.random.default_rng(args.seed + 1701) if args.shuffle_pairings else None
    if args.all_pool_pairing:
        pairs = build_pair_blocks(
            pg_pools,
            q_entries,
            args.max_pairings_per_block,
            args.max_q_pairings_per_block,
            rng=pair_rng,
        )
    else:
        pairs = build_pair_blocks_legacy(
            ppo_entries,
            mappo_entries,
            q_entries,
            args.max_pairings_per_block,
            args.max_q_pairings_per_block,
            rng=pair_rng,
        )

    rows = {k: [] for k in [
        "query_obs",
        "partner_obs",
        "ego_action",
        "partner_action",
        "prev_partner_action",
        "prev_ego_action",
        "partner_action_history",
        "ego_action_history",
        "partner_obs_history",
        "target_partner_probs",
        "step_reward",
        "return_to_go",
        "episode_return",
        "episode_id",
        "timestep",
        "role",
    ]}
    rows["pair_label"] = []
    rows["ego_pool"] = []
    rows["partner_pool"] = []
    rows["ego_family"] = []
    rows["partner_family"] = []
    rows["ego_policy_id"] = []
    rows["partner_policy_id"] = []
    summary_rows = []
    episode_id = 0
    cap_rng = np.random.default_rng(args.seed + 9917)

    for pair_idx, (block_name, left, right) in enumerate(pairs):
        pair_label = f"{block_name}:{left.pool}_{left.label}x{right.pool}_{right.label}"
        left_policy_id = f"{left.pool}:{left.label}"
        right_policy_id = f"{right.pool}:{right.label}"
        pairing = PolicyPairing(left.policy, right.policy)
        keys = jax.random.split(jax.random.PRNGKey(args.seed + 1000 * pair_idx), args.num_eval_seeds)
        print(f"[collect_pg_mixed] pair {pair_idx + 1}/{len(pairs)} {pair_label}", flush=True)
        rollout_fn = jax.jit(jax.vmap(lambda rollout_key: get_rollout_seq(pairing, env, rollout_key)))
        (obs_batch, _done_batch, actions_batch, reward_batch), total_reward_batch = rollout_fn(keys)
        total_reward_batch = np.asarray(total_reward_batch, dtype=np.float32)
        reward_batch = np.asarray(reward_batch, dtype=np.float32)
        rewards = []
        for seed_idx in range(args.num_eval_seeds):
            obs0 = np.asarray(obs_batch["agent_0"][seed_idx])
            obs1 = np.asarray(obs_batch["agent_1"][seed_idx])
            a0 = np.asarray(actions_batch["agent_0"][seed_idx])
            a1 = np.asarray(actions_batch["agent_1"][seed_idx])
            step_reward = np.asarray(reward_batch[seed_idx], dtype=np.float32)
            episode_return = float(total_reward_batch[seed_idx])
            target_right_on_obs0 = policy_probs_on_query_obs(right.policy, obs0, args.policy_prob_batch_size)
            target_left_on_obs1 = policy_probs_on_query_obs(left.policy, obs1, args.policy_prob_batch_size)
            append_role(
                rows,
                obs0,
                obs1,
                a0,
                a1,
                step_reward,
                episode_return,
                target_right_on_obs0,
                pair_label,
                episode_id,
                0,
                args.transition_stride,
                args.history_len,
                left.pool,
                right.pool,
                left_policy_id,
                right_policy_id,
            )
            append_role(
                rows,
                obs1,
                obs0,
                a1,
                a0,
                step_reward,
                episode_return,
                target_left_on_obs1,
                pair_label,
                episode_id,
                1,
                args.transition_stride,
                args.history_len,
                right.pool,
                left.pool,
                right_policy_id,
                left_policy_id,
            )
            rewards.append(episode_return)
            episode_id += 1
        summary_rows.append({
            "pair_label": pair_label,
            "block": block_name,
            "episodes": args.num_eval_seeds,
            "mean_reward": float(np.mean(rewards)),
        })
        cap_rows_in_memory(rows, args.max_transitions, cap_rng)

    arrays = {}
    for key, parts in rows.items():
        if key in {
            "pair_label",
            "ego_pool",
            "partner_pool",
            "ego_family",
            "partner_family",
            "ego_policy_id",
            "partner_policy_id",
        }:
            arrays[key] = np.asarray(parts)
        else:
            arrays[key] = np.concatenate(parts, axis=0)
    total = arrays["ego_action"].shape[0]
    if args.max_transitions and total > args.max_transitions:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(total, size=args.max_transitions, replace=False))
        for key in arrays:
            arrays[key] = arrays[key][keep]
        total = args.max_transitions

    dataset_path = output / "pg_mixed_latent_decoder_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    with (output / "dataset_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_label", "block", "episodes", "mean_reward"])
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "dataset_summary.md").open("w") as f:
        f.write("# PPO/MAPPO mixed latent-decoder dataset\n\n")
        f.write(f"- ppo_run_dir: `{args.ppo_run_dir}`\n")
        f.write(f"- mappo_run_dir: `{args.mappo_run_dir}`\n")
        f.write(f"- extra_ppo_pool: `{args.extra_ppo_pool}`\n")
        f.write(f"- ppo_policy_indices: `{args.ppo_policy_indices}`\n")
        f.write(f"- mappo_policy_indices: `{args.mappo_policy_indices}`\n")
        f.write(f"- all_pool_pairing: `{args.all_pool_pairing}`\n")
        f.write(f"- shuffle_pairings: `{args.shuffle_pairings}`\n")
        for pool_name, entries in pg_pools:
            f.write(f"- pool {pool_name}: `{len(entries)}` policies\n")
        f.write(f"- q_run_root: `{args.q_run_root}`\n")
        f.write(f"- q_methods: `{args.q_methods}`\n")
        f.write(f"- q_action_mode: `{args.q_action_mode}`\n")
        f.write(f"- max_q_policies: `{args.max_q_policies}`\n")
        f.write(f"- pairings: `{len(pairs)}`\n")
        f.write(f"- eval_seeds_per_pair: `{args.num_eval_seeds}`\n")
        f.write(f"- transition_stride: `{args.transition_stride}`\n")
        f.write(f"- history_len: `{args.history_len}`\n")
        f.write(f"- stored_transitions: `{total}`\n")
        f.write(f"- dataset: `{dataset_path}`\n")
        f.write("- target: partner policy distribution on ego/query observation.\n")
        f.write("- history: partner obs/action history before the query timestep.\n")
        f.write("- reward fields: `step_reward`, `return_to_go`, `episode_return`.\n")
    print(f"[collect_pg_mixed] wrote {dataset_path} transitions={total}", flush=True)


if __name__ == "__main__":
    main()
