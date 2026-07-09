from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

EPS = 1e-8


def string_array(data, key, n):
    if key in data.files:
        return np.asarray(data[key]).astype(str)
    return np.asarray(["unknown"] * n, dtype=str)


def obs_key(obs):
    return np.ascontiguousarray(obs).tobytes()


def valid_mask(obs_hist):
    axes = tuple(range(2, obs_hist.ndim))
    valid = np.sum(np.abs(obs_hist), axis=axes) > EPS
    has_any = np.any(valid, axis=1, keepdims=True)
    return np.where(has_any, valid, np.ones_like(valid, dtype=bool))


def sample_indices(n, max_rows, seed):
    idx = np.arange(n, dtype=np.int32)
    if max_rows and n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_rows, replace=False).astype(np.int32)
    return np.sort(idx)


def build_count_bank(data, idx, action_dim, max_history_len):
    obs_hist = np.asarray(data["partner_obs_history"][idx], dtype=np.float32)
    act_hist = np.asarray(data["partner_action_history"][idx], dtype=np.int32)
    n = obs_hist.shape[0]
    policy_ids = string_array(data, "partner_policy_id", data["partner_action_history"].shape[0])[idx]
    families = string_array(data, "partner_family", data["partner_action_history"].shape[0])[idx]
    h = min(int(max_history_len), act_hist.shape[1])
    valid = valid_mask(obs_hist[:, :h])

    policy_to_family = {}
    obs_counts = defaultdict(lambda: defaultdict(lambda: np.ones(action_dim, dtype=np.float64)))
    unconditional = defaultdict(lambda: np.ones(action_dim, dtype=np.float64))
    family_obs_counts = defaultdict(lambda: defaultdict(lambda: np.ones(action_dim, dtype=np.float64)))
    family_unconditional = defaultdict(lambda: np.ones(action_dim, dtype=np.float64))

    for row in range(n):
        policy = str(policy_ids[row])
        family = str(families[row])
        policy_to_family.setdefault(policy, family)
        for t in range(h):
            if not valid[row, t]:
                continue
            action = int(act_hist[row, t])
            if action < 0 or action >= action_dim:
                continue
            key = obs_key(obs_hist[row, t])
            obs_counts[policy][key][action] += 1.0
            unconditional[policy][action] += 1.0
            family_obs_counts[family][key][action] += 1.0
            family_unconditional[family][action] += 1.0

    return {
        "policy_to_family": policy_to_family,
        "policy_ids": sorted(policy_to_family.keys()),
        "family_ids": sorted(set(policy_to_family.values())),
        "obs_counts": obs_counts,
        "unconditional": unconditional,
        "family_obs_counts": family_obs_counts,
        "family_unconditional": family_unconditional,
    }


def log_prob_for_candidate(obs_hist, act_hist, valid, candidate, obs_counts, unconditional, action_dim, h):
    total = 0.0
    count = 0
    default_counts = np.ones(action_dim, dtype=np.float64)
    backoff = unconditional.get(candidate, default_counts)
    for t in range(h):
        if not valid[t]:
            continue
        action = int(act_hist[t])
        if action < 0 or action >= action_dim:
            continue
        counts = obs_counts.get(candidate, {}).get(obs_key(obs_hist[t]), backoff)
        probs = counts / np.maximum(np.sum(counts), EPS)
        total += float(np.log(np.maximum(probs[action], EPS)))
        count += 1
    if count == 0:
        return float("-inf")
    return total / float(count)


def evaluate_bank(data, idx, bank, action_dim, history_lengths, seed):
    obs_hist = np.asarray(data["partner_obs_history"][idx], dtype=np.float32)
    act_hist = np.asarray(data["partner_action_history"][idx], dtype=np.int32)
    n_total = data["partner_action_history"].shape[0]
    true_policy = string_array(data, "partner_policy_id", n_total)[idx]
    true_family = string_array(data, "partner_family", n_total)[idx]
    valid_all = valid_mask(obs_hist)
    rng = np.random.default_rng(seed + 123)
    rows = []
    policies = bank["policy_ids"]
    families = bank["family_ids"]
    policy_to_family = bank["policy_to_family"]

    for requested_h in history_lengths:
        h = min(int(requested_h), act_hist.shape[1])
        policy_correct = 0
        policy_covered = 0
        family_correct = 0
        family_covered = 0
        policy_margins = []
        family_margins = []
        true_scores = []
        random_margins = []

        for row in range(len(idx)):
            policy_scores = np.asarray(
                [
                    log_prob_for_candidate(
                        obs_hist[row],
                        act_hist[row],
                        valid_all[row],
                        policy,
                        bank["obs_counts"],
                        bank["unconditional"],
                        action_dim,
                        h,
                    )
                    for policy in policies
                ],
                dtype=np.float64,
            )
            pred_policy = policies[int(np.argmax(policy_scores))]
            pred_family_from_policy = policy_to_family.get(pred_policy, "unknown")
            if str(true_policy[row]) in policy_to_family:
                policy_covered += 1
                if pred_policy == str(true_policy[row]):
                    policy_correct += 1
                true_pos = policies.index(str(true_policy[row]))
                true_score = float(policy_scores[true_pos])
                true_scores.append(true_score)
                wrong_scores = np.delete(policy_scores, true_pos)
                if len(wrong_scores):
                    policy_margins.append(true_score - float(np.max(wrong_scores)))
                if len(policies) > 1:
                    random_policy = rng.choice([p for p in policies if p != str(true_policy[row])])
                    random_pos = policies.index(str(random_policy))
                    random_margins.append(true_score - float(policy_scores[random_pos]))

            if str(true_family[row]) in families:
                family_covered += 1
                if pred_family_from_policy == str(true_family[row]):
                    family_correct += 1

                family_scores = np.asarray(
                    [
                        log_prob_for_candidate(
                            obs_hist[row],
                            act_hist[row],
                            valid_all[row],
                            family,
                            bank["family_obs_counts"],
                            bank["family_unconditional"],
                            action_dim,
                            h,
                        )
                        for family in families
                    ],
                    dtype=np.float64,
                )
                true_family_pos = families.index(str(true_family[row]))
                true_family_score = float(family_scores[true_family_pos])
                wrong_family_scores = np.delete(family_scores, true_family_pos)
                if len(wrong_family_scores):
                    family_margins.append(true_family_score - float(np.max(wrong_family_scores)))

        rows.append(
            {
                "history_len_requested": int(requested_h),
                "history_len_effective": int(h),
                "n": int(len(idx)),
                "policy_candidates": int(len(policies)),
                "family_candidates": int(len(families)),
                "policy_coverage": float(policy_covered / max(len(idx), 1)),
                "family_coverage": float(family_covered / max(len(idx), 1)),
                "policy_acc": float(policy_correct / max(policy_covered, 1)),
                "family_acc_from_policy": float(family_correct / max(family_covered, 1)),
                "true_loglik": float(np.mean(true_scores)) if true_scores else float("nan"),
                "true_minus_best_wrong_policy": (
                    float(np.mean(policy_margins)) if policy_margins else float("nan")
                ),
                "true_minus_random_policy": (
                    float(np.mean(random_margins)) if random_margins else float("nan")
                ),
                "true_minus_best_wrong_family": (
                    float(np.mean(family_margins)) if family_margins else float("nan")
                ),
            }
        )
    return rows


def write_outputs(output_dir, args, rows):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "trajectory_identifiability.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Trajectory Identifiability Diagnostic",
        "",
        f"- candidate_dataset: `{args.candidate_dataset}`",
        f"- eval_dataset: `{args.eval_dataset}`",
        f"- candidate_max_rows: `{args.candidate_max_rows}`",
        f"- eval_max_rows: `{args.eval_max_rows}`",
        f"- action_dim: `{args.action_dim}`",
        "",
        "This diagnostic uses empirical behavior likelihood with exact-observation counts and action-frequency backoff.",
        "",
        "| history len | effective len | n | policy candidates | family candidates | policy coverage | family coverage | policy acc | family acc | true ll | true-best-wrong policy | true-random policy | true-best-wrong family |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {history_len_requested} | {history_len_effective} | {n} | {policy_candidates} | {family_candidates} | {policy_coverage:.3f} | {family_coverage:.3f} | {policy_acc:.3f} | {family_acc_from_policy:.3f} | {true_loglik:.4f} | {true_minus_best_wrong_policy:.4f} | {true_minus_random_policy:.4f} | {true_minus_best_wrong_family:.4f} |".format(
                **row
            )
        )
    (out / "trajectory_identifiability.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_dataset", required=True)
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--history_lengths", default="10,25,50,100")
    parser.add_argument("--candidate_max_rows", type=int, default=12000)
    parser.add_argument("--eval_max_rows", type=int, default=3000)
    parser.add_argument("--max_history_len", type=int, default=100)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidate = np.load(args.candidate_dataset, allow_pickle=False)
    eval_data = np.load(args.eval_dataset, allow_pickle=False)
    c_n = int(candidate["partner_action_history"].shape[0])
    e_n = int(eval_data["partner_action_history"].shape[0])
    c_idx = sample_indices(c_n, args.candidate_max_rows, args.seed)
    e_idx = sample_indices(e_n, args.eval_max_rows, args.seed + 1)
    history_lengths = [
        int(x.strip()) for x in args.history_lengths.split(",") if x.strip()
    ]
    bank = build_count_bank(
        candidate,
        c_idx,
        args.action_dim,
        max(max(history_lengths), args.max_history_len),
    )
    rows = evaluate_bank(
        eval_data,
        e_idx,
        bank,
        args.action_dim,
        history_lengths,
        args.seed,
    )
    write_outputs(args.output_dir, args, rows)


if __name__ == "__main__":
    main()
