import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2


DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout_with_observations
from overcooked_v2_experiments.mappo.policy import MAPPOPolicy
from overcooked_v2_experiments.mappo.utils.store import (
    load_all_checkpoints as load_mappo_all_checkpoints,
)
from overcooked_v2_experiments.mappo.utils.visualize import (
    visualize_mappo_policy as visualize_mappo_run,
)
from overcooked_v2_experiments.ttac.policy import PPOPolicy as PPODiagnosticsPolicy
from overcooked_v2_experiments.ttac.utils.store import (
    load_all_checkpoints as load_ppo_all_checkpoints,
)
from overcooked_v2_experiments.ttac.utils.visualize_ppo import (
    visualize_ppo_policy as visualize_ppo_run,
)
from overcooked_v2_experiments.ttappo_v3_memory.policy import (
    AdaptivePPOPolicy as TTAPPOV3AdaptivePolicy,
)
from overcooked_v2_experiments.ttappo_v3_memory.policy import (
    PPOPolicy as TTAPPOV3Policy,
)
from overcooked_v2_experiments.ttappo_v3_memory.utils.store import (
    load_all_checkpoints as load_ttappo_v3_all_checkpoints,
)
from overcooked_v2_experiments.ttappo_v3_memory.utils.visualize_ppo import (
    visualize_ppo_policy as visualize_ttappo_v3_run,
)


EPS = 1e-8
FORWARD_CHUNK_SIZE = int(os.environ.get("ZSC_FORWARD_CHUNK_SIZE", "8192"))
COMPATIBILITY_SAMPLE_LIMIT = 512


@dataclass
class SignatureInfo:
    digest: str
    summary: str


@dataclass
class SupportStats:
    unique_views: np.ndarray
    unique_state_count: int
    total_state_visits: int


@dataclass
class FeatureSpec:
    num_agents: int
    dynamic_size: int


@dataclass
class ComplementarityStats:
    sampled_shared_states: int
    joint_optimal_rate: float
    joint_regret: float
    unilateral_regret: float
    agent0_unilateral_regret: float
    agent1_unilateral_regret: float


def jaccard_index(a, b):
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def symmetric_kl(p, q):
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return 0.5 * (
        np.sum(p * (np.log(p) - np.log(q))) + np.sum(q * (np.log(q) - np.log(p)))
    )


def total_variation(p, q):
    return 0.5 * np.abs(p - q).sum()


def row_view(matrix):
    if matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1])))
    contiguous = np.ascontiguousarray(matrix)
    row_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return contiguous.view(row_dtype).reshape(-1)


def flatten_tree_first_two_dims(tree):
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x).reshape((-1,) + tuple(np.asarray(x).shape[2:])),
        tree,
    )


def key_bytes(view_item):
    return np.asarray(view_item).tobytes()


def digest_key(key):
    return hashlib.blake2b(key, digest_size=16).hexdigest()


def summarize_feature_row(row, spec: FeatureSpec):
    idx = 0
    num_agents = spec.num_agents

    agent_x = row[idx : idx + num_agents].astype(int).tolist()
    idx += num_agents
    agent_y = row[idx : idx + num_agents].astype(int).tolist()
    idx += num_agents
    dirs = row[idx : idx + num_agents].astype(int).tolist()
    idx += num_agents
    inventory = row[idx : idx + num_agents].astype(int).tolist()
    idx += num_agents
    recipe = int(row[idx])
    idx += 1
    dynamic_flat = np.asarray(row[idx : idx + spec.dynamic_size], dtype=np.int16)
    dynamic_hash = hashlib.blake2b(dynamic_flat.tobytes(), digest_size=8).hexdigest()

    return json.dumps(
        {
            "recipe": recipe,
            "agent_x": agent_x,
            "agent_y": agent_y,
            "dir": dirs,
            "inventory": inventory,
            "dynamic_grid": dynamic_hash,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def register_signature_rows(unique_views, first_rows, signature_info, spec):
    for view_item, row in zip(unique_views, first_rows):
        key = key_bytes(view_item)
        if key not in signature_info:
            signature_info[key] = SignatureInfo(
                digest=digest_key(key),
                summary=summarize_feature_row(row, spec),
            )


def sample_indices(total, limit, seed):
    if total <= limit:
        return np.arange(total, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=limit, replace=False).astype(np.int32))


def _summary_filename(backend, eval_mode):
    if backend in {"ppo", "mappo"}:
        return "reward_summary_cross.csv"
    mode = eval_mode or "memory_off"
    return f"reward_summary_cross_{mode}.csv"


def ensure_cross_reward_summary(
    run_dir,
    eval_seed,
    num_eval_seeds,
    backend="ppo",
    eval_mode="memory_off",
    force_eval=False,
):
    summary_path = run_dir / _summary_filename(backend, eval_mode)
    if summary_path.exists() and not force_eval:
        return summary_path

    print(
        f"[zsc_diagnostics] recomputing cross summary for {run_dir} "
        f"(backend={backend}, eval_mode={eval_mode}) with {num_eval_seeds} seeds"
    )
    if backend == "ppo":
        visualize_ppo_run(
            run_dir,
            key=jax.random.PRNGKey(eval_seed),
            final_only=True,
            num_seeds=num_eval_seeds,
            cross=True,
            no_viz=True,
        )
    elif backend == "mappo":
        visualize_mappo_run(
            run_dir,
            key=jax.random.PRNGKey(eval_seed),
            final_only=True,
            num_seeds=num_eval_seeds,
            cross=True,
            no_viz=True,
        )
    elif backend == "ttappo_v3_memory":
        visualize_ttappo_v3_run(
            run_dir,
            key=jax.random.PRNGKey(eval_seed),
            final_only=True,
            num_seeds=num_eval_seeds,
            cross=True,
            no_viz=True,
            ttt_mode=eval_mode,
            output_tag=eval_mode,
        )
    elif backend == "ttac":
        visualize_ppo_run(
            run_dir,
            key=jax.random.PRNGKey(eval_seed),
            final_only=True,
            num_seeds=num_eval_seeds,
            cross=True,
            no_viz=True,
            ttac_mode=eval_mode,
            output_tag=eval_mode,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return summary_path


def parse_cross_reward_summary(summary_path):
    per_pair = defaultdict(list)
    with open(summary_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            per_pair[row["policy_labels"]].append(float(row["total_reward"]))

    sp_values = []
    xp_values = []
    for pair_label, rewards in per_pair.items():
        lhs, rhs = pair_label.replace("cross-", "").split("_")
        mean_reward = float(np.mean(rewards))
        if lhs == rhs:
            sp_values.append(mean_reward)
        else:
            xp_values.append(mean_reward)

    return {
        "sp_mean": float(np.mean(sp_values)) if sp_values else float("nan"),
        "xp_mean": float(np.mean(xp_values)) if xp_values else float("nan"),
        "num_sp_pairs": len(sp_values),
        "num_xp_pairs": len(xp_values),
    }


def load_policies(run_dir, backend="ppo", eval_mode="memory_off"):
    if backend in {"ppo", "ttac"}:
        all_checkpoints, config = load_ppo_all_checkpoints(run_dir, final_only=True)
    elif backend == "mappo":
        all_checkpoints, config = load_mappo_all_checkpoints(run_dir, final_only=True)
    elif backend == "ttappo_v3_memory":
        all_checkpoints, config = load_ttappo_v3_all_checkpoints(run_dir, final_only=True)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    run_keys = sorted(all_checkpoints.keys(), key=lambda name: int(name.split("_")[1]))
    policies = []
    for run_key in run_keys:
        checkpoint = all_checkpoints[run_key]["ckpt_final"]
        if backend == "ppo":
            policies.append(PPODiagnosticsPolicy(checkpoint.params, config))
            continue
        if backend == "ttac":
            policies.append(PPODiagnosticsPolicy(checkpoint.params, config, eval_mode=eval_mode))
            continue
        if backend == "mappo":
            policies.append(MAPPOPolicy(checkpoint.params, config))
            continue

        if eval_mode == "memory_off":
            policy = TTAPPOV3Policy(checkpoint.params, config)
            policy.eval_mode = eval_mode
        else:
            policy = TTAPPOV3AdaptivePolicy(
                checkpoint,
                config,
                eval_mode=eval_mode,
            )
        policies.append(policy)
    return run_keys, policies, config


def build_env(config, layout_override=None):
    env_kwargs = dict(config["env"]["ENV_KWARGS"])
    if layout_override is not None:
        env_kwargs["layout"] = layout_override
    return OvercookedV2(**env_kwargs)


def rollout_batch(pairing, env, episode_keys):
    rollout_fn = lambda key: get_rollout_with_observations(pairing, env, key)
    return jax.vmap(rollout_fn)(episode_keys)


def state_feature_matrix(state_seq):
    num_agents = int(np.asarray(state_seq.agents.pos.x).shape[-1])
    dynamic_flat = np.asarray(state_seq.grid[..., 1:], dtype=np.int16).reshape(
        -1, int(np.asarray(state_seq.grid[..., 1:]).shape[-3] * np.asarray(state_seq.grid[..., 1:]).shape[-2] * np.asarray(state_seq.grid[..., 1:]).shape[-1])
    )

    features = [
        np.asarray(state_seq.agents.pos.x, dtype=np.int16).reshape(-1, num_agents),
        np.asarray(state_seq.agents.pos.y, dtype=np.int16).reshape(-1, num_agents),
        np.asarray(state_seq.agents.dir, dtype=np.int16).reshape(-1, num_agents),
        np.asarray(state_seq.agents.inventory, dtype=np.int16).reshape(-1, num_agents),
        np.asarray(state_seq.recipe, dtype=np.int16).reshape(-1, 1),
        dynamic_flat,
    ]
    matrix = np.concatenate(features, axis=1)
    spec = FeatureSpec(num_agents=num_agents, dynamic_size=dynamic_flat.shape[1])
    return matrix, spec


def extract_agent_batches(obs_seq, done_seq, mask, num_agents):
    obs_batches = []
    done_batches = []
    for agent_idx in range(num_agents):
        agent_id = f"agent_{agent_idx}"
        obs_flat = np.asarray(obs_seq[agent_id]).reshape(
            -1, *np.asarray(obs_seq[agent_id]).shape[2:]
        )
        done_flat = np.asarray(done_seq[agent_id]).reshape(-1)
        obs_batches.append(obs_flat[mask])
        done_batches.append(done_flat[mask])

    if not obs_batches:
        return np.empty((0,)), np.empty((0,))

    return np.concatenate(obs_batches, axis=0), np.concatenate(done_batches, axis=0)


def extract_agent_role_batches(obs_seq, done_seq, mask):
    obs0 = np.asarray(obs_seq["agent_0"]).reshape(
        -1, *np.asarray(obs_seq["agent_0"]).shape[2:]
    )[mask]
    obs1 = np.asarray(obs_seq["agent_1"]).reshape(
        -1, *np.asarray(obs_seq["agent_1"]).shape[2:]
    )[mask]
    done0 = np.asarray(done_seq["agent_0"]).reshape(-1)[mask]
    done1 = np.asarray(done_seq["agent_1"]).reshape(-1)[mask]
    return obs0, obs1, done0, done1


def batched_policy_outputs(policy, obs_batch, done_batch, chunk_size=FORWARD_CHUNK_SIZE):
    probs_chunks = []
    value_chunks = []
    for start in range(0, obs_batch.shape[0], chunk_size):
        end = min(start + chunk_size, obs_batch.shape[0])
        probs, values, _ = policy.forward_diagnostics_batch(
            obs_batch[start:end], done_batch[start:end], None
        )
        probs_chunks.append(np.asarray(probs, dtype=np.float64))
        value_chunks.append(np.asarray(values, dtype=np.float64))

    if not probs_chunks:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return np.concatenate(probs_chunks, axis=0), np.concatenate(value_chunks, axis=0)


def evaluate_complementarity(
    env,
    state_batch,
    obs0_batch,
    obs1_batch,
    done0_batch,
    done1_batch,
    lhs_policy,
    rhs_policy,
    gamma,
    seed,
    value_mode="policy_value",
):
    num_samples = obs0_batch.shape[0]
    if num_samples == 0:
        return ComplementarityStats(
            sampled_shared_states=0,
            joint_optimal_rate=0.0,
            joint_regret=0.0,
            unilateral_regret=0.0,
            agent0_unilateral_regret=0.0,
            agent1_unilateral_regret=0.0,
        )

    lhs_probs, _ = batched_policy_outputs(lhs_policy, obs0_batch, done0_batch)
    rhs_probs, _ = batched_policy_outputs(rhs_policy, obs1_batch, done1_batch)
    chosen_a0 = np.argmax(lhs_probs, axis=-1).astype(np.int32)
    chosen_a1 = np.argmax(rhs_probs, axis=-1).astype(np.int32)

    action_grid = np.array(
        list(itertools.product(range(len(env.action_set)), repeat=env.num_agents)),
        dtype=np.int32,
    )
    num_joint_actions = action_grid.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples * num_joint_actions)

    repeated_states = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.asarray(x), num_joint_actions, axis=0), state_batch
    )
    a0 = jnp.asarray(np.tile(action_grid[:, 0], num_samples), dtype=jnp.int32)
    a1 = jnp.asarray(np.tile(action_grid[:, 1], num_samples), dtype=jnp.int32)

    step_fn = lambda key, state, act0, act1: env.step(
        key, state, {"agent_0": act0, "agent_1": act1}
    )
    next_obs, _, reward, next_done, _ = jax.vmap(step_fn)(
        keys, repeated_states, a0, a1
    )

    team_reward = np.asarray(reward["agent_0"], dtype=np.float64)
    if value_mode == "reward_only":
        team_score = team_reward
    elif value_mode == "policy_value":
        next_obs0 = np.asarray(next_obs["agent_0"])
        next_obs1 = np.asarray(next_obs["agent_1"])
        next_done0 = np.asarray(next_done["agent_0"])
        next_done1 = np.asarray(next_done["agent_1"])
        _, lhs_next_values = batched_policy_outputs(lhs_policy, next_obs0, next_done0)
        _, rhs_next_values = batched_policy_outputs(rhs_policy, next_obs1, next_done1)
        team_score = team_reward + gamma * 0.5 * (lhs_next_values + rhs_next_values)
    else:
        raise ValueError(f"Unsupported compatibility value mode: {value_mode}")
    score_cube = team_score.reshape(num_samples, len(env.action_set), len(env.action_set))

    sample_idx = np.arange(num_samples)
    chosen_score = score_cube[sample_idx, chosen_a0, chosen_a1]
    best_joint_score = score_cube.reshape(num_samples, -1).max(axis=1)
    best_a0_score = score_cube[sample_idx, :, chosen_a1].max(axis=1)
    best_a1_score = score_cube[sample_idx, chosen_a0, :].max(axis=1)

    joint_best_flat = score_cube.reshape(num_samples, -1).argmax(axis=1)
    chosen_flat = chosen_a0 * len(env.action_set) + chosen_a1

    return ComplementarityStats(
        sampled_shared_states=int(num_samples),
        joint_optimal_rate=float(np.mean(joint_best_flat == chosen_flat)),
        joint_regret=float(np.mean(best_joint_score - chosen_score)),
        unilateral_regret=float(
            np.mean(0.5 * ((best_a0_score - chosen_score) + (best_a1_score - chosen_score)))
        ),
        agent0_unilateral_regret=float(np.mean(best_a0_score - chosen_score)),
        agent1_unilateral_regret=float(np.mean(best_a1_score - chosen_score)),
    )


def summarize_unique_subset(view_subset, feature_subset, signature_info, spec):
    if view_subset.shape[0] == 0:
        return np.empty((0,), dtype=view_subset.dtype), np.empty((0,), dtype=np.int64)

    unique_views, unique_idx, counts = np.unique(
        view_subset, return_index=True, return_counts=True
    )
    register_signature_rows(unique_views, feature_subset[unique_idx], signature_info, spec)
    return unique_views, counts


def collect_selfplay_supports(run_keys, policies, env, base_seed, num_episodes):
    signature_info = {}
    support_stats = {}
    per_run_rows = []

    for run_idx, run_key in enumerate(run_keys):
        pairing = PolicyPairing(policies[run_idx], policies[run_idx])
        seed_key = jax.random.PRNGKey(base_seed + run_idx * 1000)
        episode_keys = jax.random.split(seed_key, num_episodes)

        print(f"[zsc_diagnostics] self-play support collection {run_key}")
        rollout = rollout_batch(pairing, env, episode_keys)
        rewards = np.asarray(rollout.total_reward, dtype=np.float64)
        feature_matrix, spec = state_feature_matrix(rollout.state_seq)
        views = row_view(feature_matrix)
        unique_views, unique_idx, counts = np.unique(
            views, return_index=True, return_counts=True
        )
        register_signature_rows(unique_views, feature_matrix[unique_idx], signature_info, spec)

        support_stats[run_key] = SupportStats(
            unique_views=unique_views,
            unique_state_count=int(unique_views.shape[0]),
            total_state_visits=int(feature_matrix.shape[0]),
        )
        per_run_rows.append(
            {
                "run_key": run_key,
                "unique_state_count": int(unique_views.shape[0]),
                "total_state_visits": int(feature_matrix.shape[0]),
                "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
            }
        )

    return support_stats, signature_info, per_run_rows


def iter_pair_indices(num_runs, max_pairs=None):
    pairs = list(itertools.permutations(range(num_runs), 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def support_overlap_stats(lhs_support, rhs_support):
    lhs_views = lhs_support.unique_views
    rhs_views = rhs_support.unique_views
    intersection = np.intersect1d(lhs_views, rhs_views)
    union_size = len(lhs_views) + len(rhs_views) - len(intersection)
    overlap = float(len(intersection) / union_size) if union_size else 0.0
    return overlap, len(intersection)


def collect_cross_diagnostics(
    run_keys,
    policies,
    env,
    base_seed,
    num_episodes,
    support_stats,
    signature_info,
    top_k,
    compatibility_sample_limit,
    compatibility_value_mode="policy_value",
    max_pairs=None,
):
    coverage_rows = []
    mismatch_rows = []
    complementarity_rows = []
    bottleneck_rows = []

    for pair_idx, (lhs_idx, rhs_idx) in enumerate(
        iter_pair_indices(len(run_keys), max_pairs=max_pairs)
    ):
        lhs_key = run_keys[lhs_idx]
        rhs_key = run_keys[rhs_idx]
        lhs_support = support_stats[lhs_key]
        rhs_support = support_stats[rhs_key]
        support_overlap, shared_support_state_count = support_overlap_stats(
            lhs_support, rhs_support
        )

        pairing = PolicyPairing(policies[lhs_idx], policies[rhs_idx])
        seed_key = jax.random.PRNGKey(base_seed + 10_000 + pair_idx * 1000)
        episode_keys = jax.random.split(seed_key, num_episodes)

        print(f"[zsc_diagnostics] cross diagnostics {lhs_key} x {rhs_key}")
        rollout = rollout_batch(pairing, env, episode_keys)
        rewards = np.asarray(rollout.total_reward, dtype=np.float64)
        feature_matrix, spec = state_feature_matrix(rollout.state_seq)
        views = row_view(feature_matrix)
        flat_states = flatten_tree_first_two_dims(rollout.state_seq)

        total_steps = int(views.shape[0])
        cross_unique_views = np.unique(views)
        in_lhs_support = np.isin(views, lhs_support.unique_views)
        in_rhs_support = np.isin(views, rhs_support.unique_views)
        in_either_support = in_lhs_support | in_rhs_support
        in_both_support = in_lhs_support & in_rhs_support

        either_support_steps = int(in_either_support.sum())
        both_support_steps = int(in_both_support.sum())
        shared_steps = both_support_steps
        out_of_support_mask = ~in_either_support

        unique_out_views, out_counts = summarize_unique_subset(
            views[out_of_support_mask],
            feature_matrix[out_of_support_mask],
            signature_info,
            spec,
        )
        unique_shared_views, shared_counts = summarize_unique_subset(
            views[in_both_support],
            feature_matrix[in_both_support],
            signature_info,
            spec,
        )

        out_mass_denom = int(out_counts.sum()) if out_counts.size else 0
        shared_mass_denom = int(shared_counts.sum()) if shared_counts.size else 0
        top_out_idx = np.argsort(out_counts)[::-1][:top_k] if out_counts.size else np.array([], dtype=int)
        top_shared_idx = (
            np.argsort(shared_counts)[::-1][:top_k] if shared_counts.size else np.array([], dtype=int)
        )
        top_out_mass = (
            float(out_counts[top_out_idx].sum() / out_mass_denom) if out_mass_denom else 0.0
        )

        action_agreement = 0.0
        policy_tv = 0.0
        symmetric_kl_value = 0.0
        value_shift = 0.0
        complementarity = ComplementarityStats(
            sampled_shared_states=0,
            joint_optimal_rate=0.0,
            joint_regret=0.0,
            unilateral_regret=0.0,
            agent0_unilateral_regret=0.0,
            agent1_unilateral_regret=0.0,
        )

        if shared_steps > 0:
            shared_obs, shared_done = extract_agent_batches(
                rollout.obs_seq, rollout.done_seq, in_both_support, env.num_agents
            )
            lhs_probs, lhs_values = batched_policy_outputs(
                policies[lhs_idx], shared_obs, shared_done
            )
            rhs_probs, rhs_values = batched_policy_outputs(
                policies[rhs_idx], shared_obs, shared_done
            )

            action_agreement = float(
                np.mean(np.argmax(lhs_probs, axis=-1) == np.argmax(rhs_probs, axis=-1))
            )
            policy_tv = float(np.mean(0.5 * np.abs(lhs_probs - rhs_probs).sum(axis=-1)))
            symmetric_kl_value = float(
                np.mean(
                    0.5
                    * (
                        np.sum(
                            np.clip(lhs_probs, EPS, 1.0)
                            * (
                                np.log(np.clip(lhs_probs, EPS, 1.0))
                                - np.log(np.clip(rhs_probs, EPS, 1.0))
                            ),
                            axis=-1,
                        )
                        + np.sum(
                            np.clip(rhs_probs, EPS, 1.0)
                            * (
                                np.log(np.clip(rhs_probs, EPS, 1.0))
                                - np.log(np.clip(lhs_probs, EPS, 1.0))
                            ),
                            axis=-1,
                        )
                    )
                )
            )
            value_shift = float(np.mean(np.abs(lhs_values - rhs_values)))

            shared_obs0, shared_obs1, shared_done0, shared_done1 = extract_agent_role_batches(
                rollout.obs_seq, rollout.done_seq, in_both_support
            )
            shared_states = jax.tree_util.tree_map(
                lambda x: x[in_both_support], flat_states
            )
            sample_idx = sample_indices(
                shared_steps,
                compatibility_sample_limit,
                seed=base_seed + 100_000 + pair_idx,
            )
            sampled_states = jax.tree_util.tree_map(lambda x: x[sample_idx], shared_states)
            complementarity = evaluate_complementarity(
                env=env,
                state_batch=sampled_states,
                obs0_batch=shared_obs0[sample_idx],
                obs1_batch=shared_obs1[sample_idx],
                done0_batch=shared_done0[sample_idx],
                done1_batch=shared_done1[sample_idx],
                lhs_policy=policies[lhs_idx],
                rhs_policy=policies[rhs_idx],
                gamma=float(policies[lhs_idx].config["model"]["GAMMA"]),
                seed=base_seed + 200_000 + pair_idx,
                value_mode=compatibility_value_mode,
            )

        coverage_rows.append(
            {
                "run_i": lhs_key,
                "run_j": rhs_key,
                "support_overlap": support_overlap,
                "cross_unique_state_count": int(cross_unique_views.shape[0]),
                "shared_support_state_count": shared_support_state_count,
                "total_cross_steps": total_steps,
                "in_either_support_rate": either_support_steps / total_steps
                if total_steps
                else 0.0,
                "in_both_support_rate": both_support_steps / total_steps
                if total_steps
                else 0.0,
                "xp_out_of_sp_support_rate": 1.0 - (either_support_steps / total_steps)
                if total_steps
                else 0.0,
                "topk_bottleneck_state_mass": top_out_mass,
                "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
            }
        )

        mismatch_rows.append(
            {
                "run_i": lhs_key,
                "run_j": rhs_key,
                "shared_state_steps": shared_steps,
                "shared_state_rate": shared_steps / total_steps if total_steps else 0.0,
                "action_agreement": action_agreement,
                "policy_tv": policy_tv,
                "symmetric_kl": symmetric_kl_value,
                "value_shift": value_shift,
            }
        )
        complementarity_rows.append(
            {
                "run_i": lhs_key,
                "run_j": rhs_key,
                "sampled_shared_states": complementarity.sampled_shared_states,
                "joint_optimal_rate": complementarity.joint_optimal_rate,
                "joint_regret": complementarity.joint_regret,
                "unilateral_regret": complementarity.unilateral_regret,
                "agent0_unilateral_regret": complementarity.agent0_unilateral_regret,
                "agent1_unilateral_regret": complementarity.agent1_unilateral_regret,
            }
        )

        for rank, idx in enumerate(top_out_idx, start=1):
            key = key_bytes(unique_out_views[idx])
            bottleneck_rows.append(
                {
                    "category": "coverage_out_of_support",
                    "run_i": lhs_key,
                    "run_j": rhs_key,
                    "rank": rank,
                    "state_digest": signature_info[key].digest,
                    "state_count": int(out_counts[idx]),
                    "state_mass": float(out_counts[idx] / out_mass_denom)
                    if out_mass_denom
                    else 0.0,
                    "state_summary": signature_info[key].summary,
                }
            )

        for rank, idx in enumerate(top_shared_idx, start=1):
            key = key_bytes(unique_shared_views[idx])
            bottleneck_rows.append(
                {
                    "category": "shared_state_hotspot",
                    "run_i": lhs_key,
                    "run_j": rhs_key,
                    "rank": rank,
                    "state_digest": signature_info[key].digest,
                    "state_count": int(shared_counts[idx]),
                    "state_mass": float(shared_counts[idx] / shared_mass_denom)
                    if shared_mass_denom
                    else 0.0,
                    "state_summary": signature_info[key].summary,
                }
            )

    return coverage_rows, mismatch_rows, complementarity_rows, bottleneck_rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_method(
    method_name,
    run_dir,
    backend,
    eval_mode,
    layout,
    eval_seed,
    num_eval_seeds,
    num_diag_episodes,
    top_k,
    compatibility_sample_limit,
    compatibility_value_mode="policy_value",
    force_eval=False,
    max_pairs=None,
):
    run_dir = Path(run_dir)
    summary_path = ensure_cross_reward_summary(
        run_dir,
        eval_seed=eval_seed,
        num_eval_seeds=num_eval_seeds,
        backend=backend,
        eval_mode=eval_mode,
        force_eval=force_eval,
    )
    reward_summary = parse_cross_reward_summary(summary_path)

    run_keys, policies, config = load_policies(
        run_dir,
        backend=backend,
        eval_mode=eval_mode,
    )
    if config["model"]["TYPE"] != "CNN":
        print(
            f"[zsc_diagnostics] warning: {method_name} is configured as "
            f"{config['model']['TYPE']}; policy-distribution diagnostics use "
            "reset hidden states and should be interpreted as auxiliary readouts."
        )

    env = build_env(config, layout_override=layout)
    support_stats, signature_info, selfplay_rows = collect_selfplay_supports(
        run_keys,
        policies,
        env,
        base_seed=eval_seed,
        num_episodes=num_diag_episodes,
    )
    coverage_rows, mismatch_rows, complementarity_rows, bottleneck_rows = collect_cross_diagnostics(
        run_keys,
        policies,
        env,
        base_seed=eval_seed,
        num_episodes=num_diag_episodes,
        support_stats=support_stats,
        signature_info=signature_info,
        top_k=top_k,
        compatibility_sample_limit=compatibility_sample_limit,
        compatibility_value_mode=compatibility_value_mode,
        max_pairs=max_pairs,
    )

    for row in selfplay_rows:
        row["method"] = method_name
        row["row_type"] = "selfplay_support"
    for row in coverage_rows:
        row["method"] = method_name
        row["row_type"] = "cross_coverage"
    for row in mismatch_rows:
        row["method"] = method_name
    for row in complementarity_rows:
        row["method"] = method_name
    for row in bottleneck_rows:
        row["method"] = method_name

    return {
        "method": method_name,
        "backend": backend,
        "eval_mode": eval_mode,
        "run_dir": str(run_dir),
        "reward_summary": reward_summary,
        "selfplay_rows": selfplay_rows,
        "coverage_rows": coverage_rows,
        "mismatch_rows": mismatch_rows,
        "complementarity_rows": complementarity_rows,
        "bottleneck_rows": bottleneck_rows,
    }


def aggregate_rows(rows, key):
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def render_report(output_dir, summaries, layout, num_diag_episodes):
    report_lines = [
        "# ZSC Diagnostics",
        "",
        f"- layout: `{layout}`",
        f"- diagnostic episodes per pairing: `{num_diag_episodes}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Reward Summary",
        "",
        "| Method | SP | XP |",
        "|---|---:|---:|",
    ]

    for summary in summaries:
        reward = summary["reward_summary"]
        report_lines.append(
            f"| {summary['method']} | {reward['sp_mean']:.4f} | {reward['xp_mean']:.4f} |"
        )

    report_lines.extend(["", "## Methods", ""])
    for summary in summaries:
        report_lines.append(
            f"- `{summary['method']}`: backend=`{summary['backend']}`, eval_mode=`{summary['eval_mode']}`"
        )

    report_lines.extend(["", "## Diagnostic Summary", ""])

    for summary in summaries:
        coverage_rows = summary["coverage_rows"]
        mismatch_rows = summary["mismatch_rows"]
        complementarity_rows = summary["complementarity_rows"]
        selfplay_rows = summary["selfplay_rows"]
        report_lines.append(f"### {summary['method']}")
        report_lines.append(
            f"- self-play unique states: {aggregate_rows(selfplay_rows, 'unique_state_count'):.2f}"
        )
        report_lines.append(
            f"- cross unique states: {aggregate_rows(coverage_rows, 'cross_unique_state_count'):.2f}"
        )
        report_lines.append(
            f"- support overlap: {aggregate_rows(coverage_rows, 'support_overlap'):.4f}"
        )
        report_lines.append(
            f"- out-of-support rate: {aggregate_rows(coverage_rows, 'xp_out_of_sp_support_rate'):.4f}"
        )
        report_lines.append(
            f"- in-both-support rate: {aggregate_rows(coverage_rows, 'in_both_support_rate'):.4f}"
        )
        report_lines.append(
            f"- action agreement on shared states: {aggregate_rows(mismatch_rows, 'action_agreement'):.4f}"
        )
        report_lines.append(
            f"- policy TV on shared states: {aggregate_rows(mismatch_rows, 'policy_tv'):.4f}"
        )
        report_lines.append(
            f"- symmetric KL on shared states: {aggregate_rows(mismatch_rows, 'symmetric_kl'):.4f}"
        )
        report_lines.append(
            f"- value shift on shared states: {aggregate_rows(mismatch_rows, 'value_shift'):.4f}"
        )
        report_lines.append(
            f"- joint optimal rate on shared states: {aggregate_rows(complementarity_rows, 'joint_optimal_rate'):.4f}"
        )
        report_lines.append(
            f"- unilateral regret on shared states: {aggregate_rows(complementarity_rows, 'unilateral_regret'):.4f}"
        )
        report_lines.append(
            f"- joint regret on shared states: {aggregate_rows(complementarity_rows, 'joint_regret'):.4f}"
        )
        report_lines.append("")

    if len(summaries) >= 2:
        base = summaries[0]
        other = summaries[1]
        base_cov = aggregate_rows(base["coverage_rows"], "xp_out_of_sp_support_rate")
        other_cov = aggregate_rows(other["coverage_rows"], "xp_out_of_sp_support_rate")
        base_tv = aggregate_rows(base["mismatch_rows"], "policy_tv")
        other_tv = aggregate_rows(other["mismatch_rows"], "policy_tv")
        report_lines.append("## Readout")
        report_lines.append("")
        report_lines.append(
            f"- `{other['method']}` vs `{base['method']}` XP delta: "
            f"{other['reward_summary']['xp_mean'] - base['reward_summary']['xp_mean']:+.4f}"
        )
        report_lines.append(
            f"- out-of-support rate delta: {other_cov - base_cov:+.4f}"
        )
        report_lines.append(f"- policy TV delta on shared states: {other_tv - base_tv:+.4f}")
        report_lines.append("")
        report_lines.append(
            "- If XP improves mainly with a lower out-of-support rate, coverage is doing most of the work."
        )
        report_lines.append(
            "- If shared-state TV/KL stays high after coverage improves, there is still real partner-mismatch space for later TTA."
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_run_dir", type=str)
    parser.add_argument("--comparison_run_dir", type=str)
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--comparison_name", type=str, default="comparison")
    parser.add_argument("--baseline_backend", type=str, default="ppo")
    parser.add_argument("--comparison_backend", type=str, default="ppo")
    parser.add_argument("--baseline_eval_mode", type=str, default="memory_off")
    parser.add_argument("--comparison_eval_mode", type=str, default="memory_off")
    parser.add_argument("--single_run_dir", type=str)
    parser.add_argument("--single_name", type=str, default="method")
    parser.add_argument("--single_backend", type=str, default="ppo")
    parser.add_argument("--single_eval_mode", type=str, default="memory_off")
    parser.add_argument("--standard_run_dir", type=str)
    parser.add_argument("--state_aug_run_dir", type=str)
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--num_eval_seeds", type=int, default=500)
    parser.add_argument("--num_diag_episodes", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--compatibility_sample_limit",
        type=int,
        default=COMPATIBILITY_SAMPLE_LIMIT,
    )
    parser.add_argument(
        "--compatibility_value_mode",
        choices=("policy_value", "reward_only"),
        default="policy_value",
    )
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--max_pairs", type=int)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"ppo_zsc_diagnostics_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_run_dir:
        method_specs = [
            {
                "method_name": args.single_name,
                "run_dir": args.single_run_dir,
                "backend": args.single_backend,
                "eval_mode": args.single_eval_mode,
            },
        ]
    elif args.baseline_run_dir and args.comparison_run_dir:
        method_specs = [
            {
                "method_name": args.baseline_name,
                "run_dir": args.baseline_run_dir,
                "backend": args.baseline_backend,
                "eval_mode": args.baseline_eval_mode,
            },
            {
                "method_name": args.comparison_name,
                "run_dir": args.comparison_run_dir,
                "backend": args.comparison_backend,
                "eval_mode": args.comparison_eval_mode,
            },
        ]
    elif args.standard_run_dir and args.state_aug_run_dir:
        method_specs = [
            {
                "method_name": "ppo_cnn_standard",
                "run_dir": args.standard_run_dir,
                "backend": "ppo",
                "eval_mode": "memory_off",
            },
            {
                "method_name": "ppo_cnn_state_aug",
                "run_dir": args.state_aug_run_dir,
                "backend": "ppo",
                "eval_mode": "memory_off",
            },
        ]
    else:
        raise ValueError(
            "Provide single_run_dir, baseline/comparison run dirs, or legacy standard/state_aug run dirs."
        )

    summaries = [
        summarize_method(
            method_name=spec["method_name"],
            run_dir=spec["run_dir"],
            backend=spec["backend"],
            eval_mode=spec["eval_mode"],
            layout=args.layout,
            eval_seed=args.eval_seed,
            num_eval_seeds=args.num_eval_seeds,
            num_diag_episodes=args.num_diag_episodes,
            top_k=args.top_k,
            compatibility_sample_limit=args.compatibility_sample_limit,
            compatibility_value_mode=args.compatibility_value_mode,
            force_eval=args.force_eval,
            max_pairs=args.max_pairs,
        )
        for spec in method_specs
    ]

    coverage_rows = []
    mismatch_rows = []
    complementarity_rows = []
    bottleneck_rows = []
    reward_rows = []
    for summary in summaries:
        reward_rows.append({"method": summary["method"], **summary["reward_summary"]})
        coverage_rows.extend(summary["selfplay_rows"])
        coverage_rows.extend(summary["coverage_rows"])
        mismatch_rows.extend(summary["mismatch_rows"])
        complementarity_rows.extend(summary["complementarity_rows"])
        bottleneck_rows.extend(summary["bottleneck_rows"])

    write_csv(
        output_dir / "reward_summary.csv",
        ["method", "sp_mean", "xp_mean", "num_sp_pairs", "num_xp_pairs"],
        reward_rows,
    )
    write_csv(
        output_dir / "coverage_summary.csv",
        [
            "method",
            "row_type",
            "run_key",
            "unique_state_count",
            "total_state_visits",
            "mean_reward",
            "run_i",
            "run_j",
            "support_overlap",
            "cross_unique_state_count",
            "shared_support_state_count",
            "total_cross_steps",
            "in_either_support_rate",
            "in_both_support_rate",
            "xp_out_of_sp_support_rate",
            "topk_bottleneck_state_mass",
        ],
        coverage_rows,
    )
    write_csv(
        output_dir / "shared_state_mismatch.csv",
        [
            "method",
            "run_i",
            "run_j",
            "shared_state_steps",
            "shared_state_rate",
            "action_agreement",
            "policy_tv",
            "symmetric_kl",
            "value_shift",
        ],
        mismatch_rows,
    )
    write_csv(
        output_dir / "complementarity_summary.csv",
        [
            "method",
            "run_i",
            "run_j",
            "sampled_shared_states",
            "joint_optimal_rate",
            "joint_regret",
            "unilateral_regret",
            "agent0_unilateral_regret",
            "agent1_unilateral_regret",
        ],
        complementarity_rows,
    )
    write_csv(
        output_dir / "bottleneck_states.csv",
        [
            "method",
            "category",
            "run_i",
            "run_j",
            "rank",
            "state_digest",
            "state_count",
            "state_mass",
            "state_summary",
        ],
        bottleneck_rows,
    )

    report_path = render_report(
        output_dir,
        summaries=summaries,
        layout=args.layout,
        num_diag_episodes=args.num_diag_episodes,
    )
    print(f"[zsc_diagnostics] wrote report to {report_path}")


if __name__ == "__main__":
    main()
