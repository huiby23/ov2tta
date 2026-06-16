import argparse
import csv
import hashlib
import itertools
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.mappo.policy import MAPPOPolicy
from overcooked_v2_experiments.mappo.utils.store import load_all_checkpoints as load_mappo_all_checkpoints
from overcooked_v2_experiments.ttac_v3.policy import PPOPolicy as PPOPolicy
from overcooked_v2_experiments.ttac_v3.utils.store import load_all_checkpoints as load_ppo_all_checkpoints
from overcooked_v2_experiments.ttac_v3.utils.zsc_diagnostics import (
    EPS,
    FeatureSpec,
    parse_cross_reward_summary,
    row_view,
    state_feature_matrix,
)


@chex.dataclass
class HiddenDiagnosticRollout:
    state_seq: chex.ArrayTree
    done_seq: chex.ArrayTree
    actions_seq: chex.ArrayTree
    probs_seq: chex.ArrayTree
    total_reward: chex.Array


@dataclass
class RunProfile:
    unique_views: np.ndarray
    role_probs: dict
    role_counts: dict
    unique_state_count: int
    total_state_visits: int
    mean_reward: float


def entropy(probs):
    probs = np.clip(probs, EPS, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def symmetric_kl_np(p, q):
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return 0.5 * (
        np.sum(p * (np.log(p) - np.log(q)), axis=-1)
        + np.sum(q * (np.log(q) - np.log(p)), axis=-1)
    )


def digest_view(view_item):
    return hashlib.blake2b(np.asarray(view_item).tobytes(), digest_size=16).hexdigest()


def load_policies(run_dir, backend):
    run_dir = Path(run_dir)
    if backend == "ppo":
        all_checkpoints, config = load_ppo_all_checkpoints(run_dir, final_only=True)
        policy_cls = PPOPolicy
    elif backend == "mappo":
        all_checkpoints, config = load_mappo_all_checkpoints(run_dir, final_only=True)
        policy_cls = MAPPOPolicy
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    run_keys = sorted(all_checkpoints.keys(), key=lambda name: int(name.split("_")[1]))
    policies = []
    for run_key in run_keys:
        checkpoint = all_checkpoints[run_key]["ckpt_final"]
        policies.append(policy_cls(checkpoint.params, config))
    return run_keys, policies, config


def build_env(config, layout):
    env_kwargs = dict(config["env"]["ENV_KWARGS"])
    env_kwargs["layout"] = layout
    return OvercookedV2(**env_kwargs)


def get_hidden_rollout(pairing: PolicyPairing, env, key):
    num_agents = env.num_agents
    init_hstate = {f"agent_{i}": pairing[i].init_hstate(1) for i in range(num_agents)}

    @jax.jit
    def _perform_step(carry, step_key):
        obs, state, done, total_reward, hstate = carry
        sample_key, env_key = jax.random.split(step_key, 2)
        sample_keys = jax.random.split(sample_key, num_agents)

        actions = {}
        next_hstate = {}
        probs_out = {}
        for i, policy in enumerate(pairing):
            agent_id = f"agent_{i}"
            probs, _, new_h = policy.forward_diagnostics(
                obs[agent_id], done[agent_id], hstate[agent_id]
            )
            if policy.stochastic:
                action = jax.random.categorical(sample_keys[i], jnp.log(jnp.clip(probs, EPS, 1.0)))
            else:
                action = jnp.argmax(probs, axis=-1)
            actions[agent_id] = action.astype(jnp.int32)
            next_hstate[agent_id] = new_h
            probs_out[agent_id] = probs

        next_obs, next_state, reward, next_done, _ = env.step(env_key, state, actions)

        if env.num_agents == 2:
            updated_hstate = {}
            for i, policy in enumerate(pairing):
                agent_id = f"agent_{i}"
                partner_id = f"agent_{1 - i}"
                updated_hstate[agent_id] = policy.update_after_step(
                    next_hstate[agent_id],
                    obs[partner_id],
                    actions[partner_id],
                    next_done[agent_id],
                )
        else:
            updated_hstate = next_hstate

        carry = (next_obs, next_state, next_done, total_reward + reward["agent_0"], updated_hstate)
        return carry, (state, done, actions, probs_out)

    key, reset_key = jax.random.split(key, 2)
    obs, state = env.reset(reset_key)
    done = {f"agent_{i}": False for i in range(num_agents)}
    done["__all__"] = False
    keys = jax.random.split(key, env.max_steps)
    carry = (obs, state, done, 0.0, init_hstate)
    carry, (state_seq, done_seq, actions_seq, probs_seq) = jax.lax.scan(_perform_step, carry, keys)
    return HiddenDiagnosticRollout(
        state_seq=state_seq,
        done_seq=done_seq,
        actions_seq=actions_seq,
        probs_seq=probs_seq,
        total_reward=carry[-2],
    )


def rollout_batch(pairing, env, episode_keys):
    return jax.vmap(lambda key: get_hidden_rollout(pairing, env, key))(episode_keys)


def aggregate_role_probs(views, probs_flat):
    unique_views, inverse, counts = np.unique(views, return_inverse=True, return_counts=True)
    sums = np.zeros((unique_views.shape[0], probs_flat.shape[-1]), dtype=np.float64)
    np.add.at(sums, inverse, probs_flat.astype(np.float64))
    means = sums / counts[:, None]
    return unique_views, {np.asarray(k).tobytes(): means[i] for i, k in enumerate(unique_views)}, {
        np.asarray(k).tobytes(): int(counts[i]) for i, k in enumerate(unique_views)
    }


def collect_selfplay_profiles(run_keys, policies, env, base_seed, num_episodes):
    profiles = {}
    rows = []
    for run_idx, run_key in enumerate(run_keys):
        print(f"[hidden-diag] selfplay profile {run_key}", flush=True)
        pairing = PolicyPairing(policies[run_idx], policies[run_idx])
        keys = jax.random.split(jax.random.PRNGKey(base_seed + run_idx * 1000), num_episodes)
        rollout = rollout_batch(pairing, env, keys)
        rewards = np.asarray(rollout.total_reward, dtype=np.float64)
        feature_matrix, _ = state_feature_matrix(rollout.state_seq)
        views = row_view(feature_matrix)
        role_probs = {}
        role_counts = {}
        for role in range(env.num_agents):
            agent_id = f"agent_{role}"
            probs = np.asarray(rollout.probs_seq[agent_id]).reshape(-1, np.asarray(rollout.probs_seq[agent_id]).shape[-1])
            unique_views, probs_by_key, counts_by_key = aggregate_role_probs(views, probs)
            role_probs[role] = probs_by_key
            role_counts[role] = counts_by_key
        unique_views = np.unique(views)
        profiles[run_key] = RunProfile(
            unique_views=unique_views,
            role_probs=role_probs,
            role_counts=role_counts,
            unique_state_count=int(unique_views.shape[0]),
            total_state_visits=int(views.shape[0]),
            mean_reward=float(np.mean(rewards)) if rewards.size else 0.0,
        )
        rows.append({
            "run_key": run_key,
            "unique_state_count": int(unique_views.shape[0]),
            "total_state_visits": int(views.shape[0]),
            "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
        })
    return profiles, rows


def support_overlap(lhs_views, rhs_views):
    intersection = np.intersect1d(lhs_views, rhs_views)
    union_size = len(lhs_views) + len(rhs_views) - len(intersection)
    return (float(len(intersection) / union_size) if union_size else 0.0), int(len(intersection))


def reward_pair_maps(summary_path):
    per_pair = defaultdict(list)
    with open(summary_path, newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["policy_labels"].replace("cross-", "")
            per_pair[label].append(float(row["total_reward"]))
    return {k: float(np.mean(v)) for k, v in per_pair.items()}


def collect_pair_diagnostics(run_keys, policies, profiles, env, base_seed, num_episodes, reward_pairs, max_pairs=None):
    rows = []
    hidden_rows = []
    pairs = list(itertools.permutations(range(len(run_keys)), 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    for pair_idx, (lhs_idx, rhs_idx) in enumerate(pairs):
        lhs_key = run_keys[lhs_idx]
        rhs_key = run_keys[rhs_idx]
        print(f"[hidden-diag] cross pair {lhs_key} x {rhs_key}", flush=True)
        lhs_profile = profiles[lhs_key]
        rhs_profile = profiles[rhs_key]
        overlap, shared_support_count = support_overlap(lhs_profile.unique_views, rhs_profile.unique_views)

        pairing = PolicyPairing(policies[lhs_idx], policies[rhs_idx])
        keys = jax.random.split(jax.random.PRNGKey(base_seed + 10_000 + pair_idx * 1000), num_episodes)
        rollout = rollout_batch(pairing, env, keys)
        rewards = np.asarray(rollout.total_reward, dtype=np.float64)
        feature_matrix, _ = state_feature_matrix(rollout.state_seq)
        views = row_view(feature_matrix)
        total_steps = int(views.shape[0])
        in_lhs = np.isin(views, lhs_profile.unique_views)
        in_rhs = np.isin(views, rhs_profile.unique_views)
        in_either = in_lhs | in_rhs
        in_both = in_lhs & in_rhs
        shared_indices = np.nonzero(in_both)[0]

        tv_values = []
        kl_values = []
        agree_values = []
        support_counts = []
        for idx in shared_indices:
            key = np.asarray(views[idx]).tobytes()
            for role in range(env.num_agents):
                lhs_probs = lhs_profile.role_probs[role].get(key)
                rhs_probs = rhs_profile.role_probs[role].get(key)
                if lhs_probs is None or rhs_probs is None:
                    continue
                tv_values.append(0.5 * np.abs(lhs_probs - rhs_probs).sum())
                kl_values.append(float(symmetric_kl_np(lhs_probs[None, :], rhs_probs[None, :])[0]))
                agree_values.append(float(np.argmax(lhs_probs) == np.argmax(rhs_probs)))
                support_counts.append(min(lhs_profile.role_counts[role].get(key, 0), rhs_profile.role_counts[role].get(key, 0)))

        actual_entropies = []
        actual_confidences = []
        chosen_confidences = []
        for role in range(env.num_agents):
            agent_id = f"agent_{role}"
            probs = np.asarray(rollout.probs_seq[agent_id]).reshape(-1, np.asarray(rollout.probs_seq[agent_id]).shape[-1])
            actions = np.asarray(rollout.actions_seq[agent_id]).reshape(-1).astype(np.int64)
            mask = in_both
            if mask.any():
                shared_probs = probs[mask]
                shared_actions = actions[mask]
                actual_entropies.extend(entropy(shared_probs).tolist())
                actual_confidences.extend(np.max(shared_probs, axis=-1).tolist())
                chosen_confidences.extend(shared_probs[np.arange(shared_probs.shape[0]), shared_actions].tolist())

        pair_label = f"{lhs_idx}_{rhs_idx}"
        sp_lhs = reward_pairs.get(f"{lhs_idx}_{lhs_idx}", float("nan"))
        sp_rhs = reward_pairs.get(f"{rhs_idx}_{rhs_idx}", float("nan"))
        xp_eval = reward_pairs.get(pair_label, float("nan"))
        row = {
            "run_i": lhs_key,
            "run_j": rhs_key,
            "run_i_idx": lhs_idx,
            "run_j_idx": rhs_idx,
            "diag_reward": float(np.mean(rewards)) if rewards.size else 0.0,
            "eval_xp_reward": xp_eval,
            "sp_i": sp_lhs,
            "sp_j": sp_rhs,
            "mean_sp": float(np.nanmean([sp_lhs, sp_rhs])),
            "support_overlap": overlap,
            "shared_support_state_count": shared_support_count,
            "cross_unique_state_count": int(np.unique(views).shape[0]),
            "total_cross_steps": total_steps,
            "in_either_support_rate": float(in_either.mean()) if total_steps else 0.0,
            "in_both_support_rate": float(in_both.mean()) if total_steps else 0.0,
            "xp_out_of_sp_support_rate": float(1.0 - in_either.mean()) if total_steps else 0.0,
            "shared_steps": int(in_both.sum()),
            "hidden_profile_samples": int(len(tv_values)),
            "hidden_action_agreement": float(np.mean(agree_values)) if agree_values else 0.0,
            "hidden_policy_tv": float(np.mean(tv_values)) if tv_values else 0.0,
            "hidden_symmetric_kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "hidden_min_support_count": float(np.mean(support_counts)) if support_counts else 0.0,
            "actual_cross_entropy": float(np.mean(actual_entropies)) if actual_entropies else 0.0,
            "actual_cross_max_prob": float(np.mean(actual_confidences)) if actual_confidences else 0.0,
            "actual_chosen_action_prob": float(np.mean(chosen_confidences)) if chosen_confidences else 0.0,
        }
        rows.append(row)
        hidden_rows.append({k: row[k] for k in [
            "run_i", "run_j", "shared_steps", "hidden_profile_samples",
            "hidden_action_agreement", "hidden_policy_tv", "hidden_symmetric_kl",
            "actual_cross_entropy", "actual_cross_max_prob", "actual_chosen_action_prob",
        ]})
    return rows, hidden_rows


def aggregate(rows, key):
    vals = [float(r[key]) for r in rows if not np.isnan(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def regression_summary(rows):
    y = np.asarray([r["eval_xp_reward"] for r in rows], dtype=np.float64)
    cols = ["mean_sp", "xp_out_of_sp_support_rate", "hidden_policy_tv", "hidden_action_agreement"]
    X_raw = np.asarray([[r[c] for c in cols] for r in rows], dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(X_raw).all(axis=1)
    out = []
    for c_idx, col in enumerate(cols):
        out.append({"metric": f"corr_xp_{col}", "value": corr(X_raw[:, c_idx], y)})
    if mask.sum() >= len(cols) + 1:
        X = X_raw[mask]
        yy = y[mask]
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        yy = (yy - yy.mean()) / (yy.std() + 1e-8)
        design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        coef, *_ = np.linalg.lstsq(design, yy, rcond=None)
        out.append({"metric": "ols_intercept", "value": float(coef[0])})
        for col, val in zip(cols, coef[1:]):
            out.append({"metric": f"ols_beta_{col}", "value": float(val)})
    return out


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(path, method_name, reward_summary, selfplay_rows, pair_rows, regression_rows, notes):
    lines = [
        f"# Hidden-State Mismatch Diagnostics: {method_name}",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- SP: `{reward_summary['sp_mean']:.4f}`",
        f"- XP: `{reward_summary['xp_mean']:.4f}`",
        "",
        "## Aggregate Diagnostics",
        "",
        f"- self-play unique states: `{aggregate(selfplay_rows, 'unique_state_count'):.4f}`",
        f"- out-of-support rate: `{aggregate(pair_rows, 'xp_out_of_sp_support_rate'):.4f}`",
        f"- in-both-support rate: `{aggregate(pair_rows, 'in_both_support_rate'):.4f}`",
        f"- hidden action agreement: `{aggregate(pair_rows, 'hidden_action_agreement'):.4f}`",
        f"- hidden policy TV: `{aggregate(pair_rows, 'hidden_policy_tv'):.4f}`",
        f"- hidden symmetric KL: `{aggregate(pair_rows, 'hidden_symmetric_kl'):.4f}`",
        f"- actual cross entropy: `{aggregate(pair_rows, 'actual_cross_entropy'):.4f}`",
        f"- actual chosen action prob: `{aggregate(pair_rows, 'actual_chosen_action_prob'):.4f}`",
        "",
        "## Pair-Level Decomposition",
        "",
    ]
    for row in regression_rows:
        lines.append(f"- {row['metric']}: `{row['value']:.4f}`")
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {note}" for note in notes])
    path.write_text("\n".join(lines) + "\n")


def run_method(args, name, run_dir, backend):
    run_keys, policies, config = load_policies(run_dir, backend)
    env = build_env(config, args.layout)
    reward_path = Path(run_dir) / "reward_summary_cross.csv"
    reward_summary = parse_cross_reward_summary(reward_path)
    reward_pairs = reward_pair_maps(reward_path)
    profiles, selfplay_rows = collect_selfplay_profiles(
        run_keys, policies, env, args.eval_seed, args.num_diag_episodes
    )
    pair_rows, hidden_rows = collect_pair_diagnostics(
        run_keys, policies, profiles, env, args.eval_seed, args.num_diag_episodes, reward_pairs, args.max_pairs
    )
    reg_rows = regression_summary(pair_rows)
    out_dir = Path(args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "selfplay_support.csv", selfplay_rows)
    write_csv(out_dir / "pair_decomposition.csv", pair_rows)
    write_csv(out_dir / "hidden_mismatch_summary.csv", hidden_rows)
    write_csv(out_dir / "regression_summary.csv", reg_rows)
    render_report(
        out_dir / "report.md",
        name,
        reward_summary,
        selfplay_rows,
        pair_rows,
        reg_rows,
        [
            "Hidden mismatch compares self-play hidden-state convention profiles for the same state and role.",
            "Pair-level XP uses the existing full Figure4 reward_summary_cross.csv when available.",
        ],
    )
    return {"name": name, "reward": reward_summary, "pairs": pair_rows, "selfplay": selfplay_rows, "reg": reg_rows}


def render_comparison(output_dir, summaries):
    lines = ["# Hidden-State Mismatch Comparison", ""]
    lines.append("| Method | SP | XP | Out-of-support | In-both-support | Hidden agreement | Hidden TV | Hidden KL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        pairs = s["pairs"]
        lines.append(
            f"| {s['name']} | {s['reward']['sp_mean']:.4f} | {s['reward']['xp_mean']:.4f} | "
            f"{aggregate(pairs, 'xp_out_of_sp_support_rate'):.4f} | {aggregate(pairs, 'in_both_support_rate'):.4f} | "
            f"{aggregate(pairs, 'hidden_action_agreement'):.4f} | {aggregate(pairs, 'hidden_policy_tv'):.4f} | "
            f"{aggregate(pairs, 'hidden_symmetric_kl'):.4f} |"
        )
    if len(summaries) == 2:
        a, b = summaries
        lines.extend([
            "",
            "## Readout",
            "",
            f"- XP delta `{b['name']} - {a['name']}`: `{b['reward']['xp_mean'] - a['reward']['xp_mean']:+.4f}`",
            f"- out-of-support delta: `{aggregate(b['pairs'], 'xp_out_of_sp_support_rate') - aggregate(a['pairs'], 'xp_out_of_sp_support_rate'):+.4f}`",
            f"- hidden TV delta: `{aggregate(b['pairs'], 'hidden_policy_tv') - aggregate(a['pairs'], 'hidden_policy_tv'):+.4f}`",
            f"- hidden agreement delta: `{aggregate(b['pairs'], 'hidden_action_agreement') - aggregate(a['pairs'], 'hidden_action_agreement'):+.4f}`",
        ])
    (Path(output_dir) / "comparison_report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_rnn_run_dir", required=True)
    parser.add_argument("--mappo_rnn_run_dir", required=True)
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--num_diag_episodes", type=int, default=100)
    parser.add_argument("--max_pairs", type=int)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_method(args, "ppo_rnn_state_aug", args.ppo_rnn_run_dir, "ppo"),
        run_method(args, "mappo_rnn_state_aug", args.mappo_rnn_run_dir, "mappo"),
    ]
    render_comparison(output_dir, summaries)
    print(f"[hidden-diag] wrote comparison report to {output_dir / 'comparison_report.md'}", flush=True)


if __name__ == "__main__":
    main()
