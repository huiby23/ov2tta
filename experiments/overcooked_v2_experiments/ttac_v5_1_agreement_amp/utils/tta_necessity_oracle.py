import argparse
import csv
import itertools
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.ttac_v5_1_agreement_amp.utils.zsc_diagnostics import (
    batched_policy_outputs,
    build_env,
    flatten_tree_first_two_dims,
    load_policies,
    rollout_batch,
)


def one_hot(values, size):
    values = np.asarray(values, dtype=np.int32)
    out = np.zeros((values.shape[0], size), dtype=np.float32)
    valid = (values >= 0) & (values < size)
    out[np.arange(values.shape[0])[valid], values[valid]] = 1.0
    return out


def build_partner_history_features(partner_actions, history_window, num_actions):
    actions = np.asarray(partner_actions, dtype=np.int32)
    if actions.ndim != 2:
        raise ValueError(f"partner_actions must be [episodes, time], got {actions.shape}")
    num_eps, horizon = actions.shape
    flat_features = []
    for ep in range(num_eps):
        ep_actions = actions[ep]
        for t in range(horizon):
            hist = ep_actions[max(0, t - history_window):t]
            padded = np.full((history_window,), -1, dtype=np.int32)
            if hist.size:
                padded[-hist.size:] = hist
            seq_oh = one_hot(padded, num_actions).reshape(-1)
            counts = np.zeros((num_actions,), dtype=np.float32)
            if hist.size:
                counts += np.bincount(hist, minlength=num_actions).astype(np.float32)
                counts /= float(hist.size)
            last = np.full((1,), -1, dtype=np.int32) if hist.size == 0 else hist[-1:]
            last_oh = one_hot(last, num_actions).reshape(-1)
            flat_features.append(np.concatenate([seq_oh, counts, last_oh], axis=0))
    return np.asarray(flat_features, dtype=np.float32)


def concat_state_batches(chunks):
    return jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate([jnp.asarray(x) for x in xs], axis=0), *chunks
    )


def collect_xp_samples(run_keys, policies, env, seed, num_episodes_per_pair, max_pairs, history_window):
    pairs = list(itertools.permutations(range(len(run_keys)), 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    num_actions = len(env.action_set)
    num_modes = len(run_keys)

    state_chunks = []
    obs_chunks = []
    done_chunks = []
    partner_action_chunks = []
    role_chunks = []
    t_chunks = []
    current_mode_chunks = []
    partner_id_chunks = []
    history_chunks = []

    for pair_idx, (ego0_idx, ego1_idx) in enumerate(pairs):
        print(f"[tta-oracle] collect pair {pair_idx + 1}/{len(pairs)}: {run_keys[ego0_idx]} x {run_keys[ego1_idx]}", flush=True)
        pairing = PolicyPairing(policies[ego0_idx], policies[ego1_idx])
        pair_key = jax.random.PRNGKey(seed + 10_000 + pair_idx * 997)
        episode_keys = jax.random.split(pair_key, num_episodes_per_pair)
        rollout = rollout_batch(pairing, env, episode_keys)

        flat_states = flatten_tree_first_two_dims(rollout.state_seq)
        horizon = np.asarray(rollout.actions_seq["agent_0"]).shape[1]
        num_steps = num_episodes_per_pair * horizon
        t_norm = np.tile(
            np.arange(horizon, dtype=np.float32) / max(horizon - 1, 1),
            num_episodes_per_pair,
        )

        for role in (0, 1):
            ego_id = f"agent_{role}"
            partner_id = f"agent_{1 - role}"
            current_mode = ego0_idx if role == 0 else ego1_idx
            partner_mode = ego1_idx if role == 0 else ego0_idx
            obs_arr = np.asarray(rollout.obs_seq[ego_id])
            obs = obs_arr.reshape(num_steps, *obs_arr.shape[2:])
            done = np.asarray(rollout.done_seq[ego_id]).reshape(num_steps)
            partner_actions = np.asarray(rollout.actions_seq[partner_id], dtype=np.int32)
            history = build_partner_history_features(partner_actions, history_window, num_actions)

            state_chunks.append(flat_states)
            obs_chunks.append(obs)
            done_chunks.append(done)
            partner_action_chunks.append(partner_actions.reshape(-1))
            role_chunks.append(np.full((num_steps,), role, dtype=np.int32))
            t_chunks.append(t_norm.copy())
            current_mode_chunks.append(np.full((num_steps,), current_mode, dtype=np.int32))
            partner_id_chunks.append(np.full((num_steps,), partner_mode, dtype=np.int32))
            history_chunks.append(history)

    return {
        "states": concat_state_batches(state_chunks),
        "obs": np.concatenate(obs_chunks, axis=0),
        "done": np.concatenate(done_chunks, axis=0),
        "partner_action": np.concatenate(partner_action_chunks, axis=0),
        "role": np.concatenate(role_chunks, axis=0),
        "t_norm": np.concatenate(t_chunks, axis=0),
        "current_mode": np.concatenate(current_mode_chunks, axis=0),
        "partner_id": np.concatenate(partner_id_chunks, axis=0),
        "history": np.concatenate(history_chunks, axis=0),
        "num_modes": num_modes,
        "num_actions": num_actions,
    }


def subsample_samples(samples, sample_limit, seed):
    total = samples["obs"].shape[0]
    if sample_limit is None or total <= sample_limit:
        idx = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=sample_limit, replace=False))
    out = {}
    for key, value in samples.items():
        if key in {"num_modes", "num_actions"}:
            out[key] = value
        elif key == "states":
            out[key] = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[idx], value)
        else:
            out[key] = value[idx]
    out["sample_indices"] = idx
    out["total_before_sample"] = total
    return out


def select_next_obs_by_role(next_obs, role):
    obs0 = np.asarray(next_obs["agent_0"])
    obs1 = np.asarray(next_obs["agent_1"])
    selector_shape = (role.shape[0],) + (1,) * (obs0.ndim - 1)
    return np.where(role.reshape(selector_shape) == 0, obs0, obs1)


def score_candidate_modes(env, policies, samples, gamma, chunk_size):
    obs = samples["obs"]
    done = samples["done"]
    partner_action = samples["partner_action"].astype(np.int32)
    role = samples["role"].astype(np.int32)
    states = samples["states"]
    num_samples = obs.shape[0]
    num_modes = len(policies)
    score_table = np.zeros((num_samples, num_modes), dtype=np.float32)

    for mode_idx, policy in enumerate(policies):
        print(f"[tta-oracle] score candidate mode {mode_idx + 1}/{num_modes}", flush=True)
        probs, _ = batched_policy_outputs(policy, obs, done, chunk_size=chunk_size)
        ego_actions = np.argmax(probs, axis=-1).astype(np.int32)
        mode_scores = []
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk_states = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[start:end], states)
            chunk_role = role[start:end]
            chunk_partner = partner_action[start:end]
            chunk_ego = ego_actions[start:end]
            act0 = np.where(chunk_role == 0, chunk_ego, chunk_partner).astype(np.int32)
            act1 = np.where(chunk_role == 0, chunk_partner, chunk_ego).astype(np.int32)
            keys = jax.random.split(jax.random.PRNGKey(100_000 + mode_idx * 10_000 + start), end - start)
            step_fn = lambda key, state, a0, a1: env.step(key, state, {"agent_0": a0, "agent_1": a1})
            next_obs, _, reward, next_done, _ = jax.vmap(step_fn)(
                keys,
                chunk_states,
                jnp.asarray(act0),
                jnp.asarray(act1),
            )
            next_obs_ego = select_next_obs_by_role(next_obs, chunk_role)
            next_done_ego = np.where(
                chunk_role == 0,
                np.asarray(next_done["agent_0"]),
                np.asarray(next_done["agent_1"]),
            )
            _, next_values = batched_policy_outputs(
                policy,
                next_obs_ego,
                next_done_ego,
                chunk_size=chunk_size,
            )
            immediate_reward = np.asarray(reward["agent_0"], dtype=np.float32)
            mode_scores.append(immediate_reward + gamma * np.asarray(next_values, dtype=np.float32))
        score_table[:, mode_idx] = np.concatenate(mode_scores, axis=0)
    return score_table


def fit_probe(name, x, labels, current_modes, score_table, current_scores, oracle_scores, seed):
    indices = np.arange(labels.shape[0])
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if unique.shape[0] > 1 and counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(x[train_idx], labels[train_idx])
    pred = clf.predict(x[test_idx])
    majority = np.bincount(labels[train_idx], minlength=score_table.shape[1]).argmax()
    majority_pred = np.full_like(labels[test_idx], majority)
    current_pred = current_modes[test_idx]

    pred_scores = score_table[test_idx, pred]
    current = current_scores[test_idx]
    oracle = oracle_scores[test_idx]
    oracle_gain = oracle - current
    pred_gain = pred_scores - current
    positive = oracle_gain > 1e-6
    recovered = (
        float(np.sum(np.maximum(pred_gain[positive], 0.0)) / (np.sum(oracle_gain[positive]) + 1e-8))
        if np.any(positive)
        else 0.0
    )

    return {
        "probe": name,
        "num_train": int(train_idx.shape[0]),
        "num_test": int(test_idx.shape[0]),
        "accuracy": float(accuracy_score(labels[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], pred)),
        "majority_accuracy": float(accuracy_score(labels[test_idx], majority_pred)),
        "current_mode_accuracy": float(accuracy_score(labels[test_idx], current_pred)),
        "mean_oracle_gain": float(np.mean(oracle_gain)),
        "mean_predicted_gain": float(np.mean(pred_gain)),
        "positive_gain_recovered": recovered,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Partner-history oracle diagnostic for TTA necessity.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--method_name", default="method")
    parser.add_argument("--backend", default="ppo", choices=["ppo", "mappo", "ttappo_v3_memory"])
    parser.add_argument("--eval_mode", default="memory_off")
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes_per_pair", type=int, default=8)
    parser.add_argument("--max_pairs", type=int, default=30)
    parser.add_argument("--sample_limit", type=int, default=20000)
    parser.add_argument("--history_window", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--margin_threshold", type=float, default=0.1)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_keys, policies, config = load_policies(args.run_dir, backend=args.backend, eval_mode=args.eval_mode)
    env = build_env(config, layout_override=args.layout)
    gamma = float(config["model"]["GAMMA"])

    samples = collect_xp_samples(
        run_keys=run_keys,
        policies=policies,
        env=env,
        seed=args.seed,
        num_episodes_per_pair=args.num_episodes_per_pair,
        max_pairs=args.max_pairs,
        history_window=args.history_window,
    )
    samples = subsample_samples(samples, args.sample_limit, args.seed + 777)
    print(
        f"[tta-oracle] sampled {samples['obs'].shape[0]} / {samples['total_before_sample']} role-state samples",
        flush=True,
    )

    score_table = score_candidate_modes(
        env=env,
        policies=policies,
        samples=samples,
        gamma=gamma,
        chunk_size=args.chunk_size,
    )
    labels = np.argmax(score_table, axis=1).astype(np.int32)
    oracle_scores = score_table[np.arange(score_table.shape[0]), labels]
    current_modes = samples["current_mode"].astype(np.int32)
    current_scores = score_table[np.arange(score_table.shape[0]), current_modes]
    margins = oracle_scores - current_scores

    num_modes = samples["num_modes"]
    role_oh = one_hot(samples["role"], 2)
    current_oh = one_hot(current_modes, num_modes)
    t_col = samples["t_norm"][:, None].astype(np.float32)
    features_base = np.concatenate([role_oh, current_oh, t_col], axis=1).astype(np.float32)
    features_history = np.concatenate([features_base, samples["history"].astype(np.float32)], axis=1)

    rows = []
    rows.append(fit_probe("base_no_partner_history", features_base, labels, current_modes, score_table, current_scores, oracle_scores, args.seed))
    rows.append(fit_probe("partner_history", features_history, labels, current_modes, score_table, current_scores, oracle_scores, args.seed))

    switch_mask = margins > args.margin_threshold
    if int(switch_mask.sum()) >= max(100, num_modes * 5) and np.unique(labels[switch_mask]).shape[0] > 1:
        rows.append(fit_probe("base_no_partner_history_switch_only", features_base[switch_mask], labels[switch_mask], current_modes[switch_mask], score_table[switch_mask], current_scores[switch_mask], oracle_scores[switch_mask], args.seed))
        rows.append(fit_probe("partner_history_switch_only", features_history[switch_mask], labels[switch_mask], current_modes[switch_mask], score_table[switch_mask], current_scores[switch_mask], oracle_scores[switch_mask], args.seed))

    partner_rows = []
    try:
        partner_rows.append(fit_probe("partner_id_from_history", features_history, samples["partner_id"].astype(np.int32), current_modes, score_table, current_scores, oracle_scores, args.seed))
    except Exception as exc:
        partner_rows.append({"probe": "partner_id_from_history", "error": repr(exc)})
    rows.extend(partner_rows)

    summary = {
        "method": args.method_name,
        "run_dir": str(args.run_dir),
        "backend": args.backend,
        "layout": args.layout,
        "num_modes": int(num_modes),
        "num_samples": int(score_table.shape[0]),
        "num_samples_before_subsample": int(samples["total_before_sample"]),
        "num_episodes_per_pair": int(args.num_episodes_per_pair),
        "max_pairs": int(args.max_pairs) if args.max_pairs is not None else None,
        "history_window": int(args.history_window),
        "switch_needed_rate": float(np.mean(labels != current_modes)),
        "positive_margin_rate": float(np.mean(margins > args.margin_threshold)),
        "mean_oracle_minus_current_score": float(np.mean(margins)),
        "median_oracle_minus_current_score": float(np.median(margins)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    write_csv(args.output_dir / "probe_summary.csv", rows)
    np.savez_compressed(
        args.output_dir / "oracle_arrays.npz",
        labels=labels,
        current_modes=current_modes,
        partner_ids=samples["partner_id"].astype(np.int32),
        margins=margins,
        score_table=score_table,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# TTA Necessity Oracle Diagnostic",
        "",
        f"- method: `{args.method_name}`",
        f"- run_dir: `{args.run_dir}`",
        f"- samples: `{summary['num_samples']}` / `{summary['num_samples_before_subsample']}`",
        f"- history_window: `{args.history_window}`",
        f"- switch_needed_rate: `{summary['switch_needed_rate']:.4f}`",
        f"- positive_margin_rate(margin>{args.margin_threshold}): `{summary['positive_margin_rate']:.4f}`",
        f"- mean_oracle_minus_current_score: `{summary['mean_oracle_minus_current_score']:.4f}`",
        "",
        "## Probe Summary",
        "",
        "| Probe | Acc | Bal Acc | Majority Acc | Current-mode Acc | Pred Gain | Oracle Gain Recovered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['probe']} | error | | | | | |")
            continue
        lines.append(
            f"| {row['probe']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['majority_accuracy']:.4f} | {row['current_mode_accuracy']:.4f} | "
            f"{row['mean_predicted_gain']:.4f} | {row['positive_gain_recovered']:.4f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[tta-oracle] wrote {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
