from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.collect_pg_mixed_latent_decoder_dataset import (  # noqa: E501
    load_ppo_pool,
    load_q_pool,
    parse_policy_indices,
    policy_probs_on_query_obs,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.latent_partner_decoder import (
    apply_latent_partner_decoder,
    load_latent_decoder_npz,
)

EPS = 1e-6


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def categorical_kl_np(target, probs):
    target = normalize_probs(target)
    probs = normalize_probs(probs)
    return np.sum(target * (np.log(np.maximum(target, EPS)) - np.log(np.maximum(probs, EPS))), axis=-1)


def categorical_tv_np(lhs, rhs):
    lhs = normalize_probs(lhs)
    rhs = normalize_probs(rhs)
    return 0.5 * np.abs(lhs - rhs).sum(axis=-1)


def entropy_np(probs):
    probs = normalize_probs(probs)
    return -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=-1)


def mean_pairwise_tv(probs):
    out = np.zeros((probs.shape[1],), dtype=np.float32)
    count = 0
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            out += categorical_tv_np(probs[i], probs[j])
            count += 1
    if count == 0:
        return out
    return out / float(count)


def string_array(data, key, n_rows, fallback="unknown"):
    if key in data.files:
        return np.asarray(data[key]).astype(str)
    return np.full((n_rows,), fallback, dtype=object).astype(str)


def method_from_policy_id(policy_id):
    return str(policy_id).split(":", 1)[0]


def quantile_mask(values, q, high=True):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros_like(values, dtype=bool), float("nan")
    threshold = float(np.quantile(values, q))
    if high:
        return values >= threshold, threshold
    return values <= threshold, threshold


def predict_decoder_probs(path, query_obs, obs_hist, act_hist, action_dim, batch_size):
    params = load_latent_decoder_npz(path)
    outs = []
    for start in range(0, len(query_obs), batch_size):
        end = min(len(query_obs), start + batch_size)
        logits = apply_latent_partner_decoder(
            params,
            jnp.asarray(query_obs[start:end], dtype=jnp.float32),
            jnp.asarray(obs_hist[start:end], dtype=jnp.float32),
            jnp.asarray(act_hist[start:end], dtype=jnp.int32),
            action_dim,
            deterministic=True,
            return_aux=False,
        )
        outs.append(np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float32))
    return np.concatenate(outs, axis=0)


def load_base_probs(args, query_obs, ego_policy_ids):
    ppo_entries, _config = load_ppo_pool(
        args.ego_run_dir,
        args.max_ego_policies,
        stochastic=False,
        policy_indices=parse_policy_indices(args.ppo_policy_indices),
    )
    policy_by_id = {f"{entry.pool}:{entry.label}": entry.policy for entry in ppo_entries}
    probs = np.full((len(query_obs), args.action_dim), np.nan, dtype=np.float32)
    for policy_id in sorted(set(ego_policy_ids.tolist())):
        if policy_id not in policy_by_id:
            continue
        rows = np.where(ego_policy_ids == policy_id)[0]
        probs[rows] = policy_probs_on_query_obs(
            policy_by_id[policy_id],
            query_obs[rows],
            batch_size=args.policy_prob_batch_size,
        )
    valid = np.all(np.isfinite(probs), axis=-1)
    return probs, valid


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(bucket_name, method_group, partner_group, mask, target, probs_by_method, aux):
    rows = []
    if not np.any(mask):
        return rows
    for method_name, probs in probs_by_method.items():
        idx = np.where(mask)[0]
        method_probs = probs[idx]
        target_probs = target[idx]
        rows.append({
            "bucket": bucket_name,
            "partner_group": partner_group,
            "method": method_name,
            "n": int(len(idx)),
            "kl": float(np.mean(categorical_kl_np(target_probs, method_probs))),
            "tv": float(np.mean(categorical_tv_np(target_probs, method_probs))),
            "argmax_acc": float(np.mean(np.argmax(target_probs, axis=-1) == np.argmax(method_probs, axis=-1))),
            "target_entropy": float(np.mean(entropy_np(target_probs))),
            "pred_entropy": float(np.mean(entropy_np(method_probs))),
            "oracle_base_tv": float(np.mean(aux["oracle_base_tv"][idx])),
            "partner_disagreement": float(np.mean(aux["partner_disagreement"][idx])),
            "return_to_go": float(np.mean(aux["return_to_go"][idx])),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--old_decoder_path",
        default="reports/query_attn_qexp4_vae_bottleneck_b001_train_20260702_174039/latent_partner_decoder.npz",
    )
    parser.add_argument(
        "--new_decoder_path",
        default="reports/cf_disagreement_qattn_vae_znorm_pgonly_train_20260706/latent_partner_decoder.npz",
    )
    parser.add_argument(
        "--ego_run_dir",
        type=Path,
        default=Path("runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606"),
    )
    parser.add_argument("--ppo_policy_indices", type=str, default="0,1")
    parser.add_argument("--max_ego_policies", type=int, default=2)
    parser.add_argument("--q_run_root", type=Path, default=Path("runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M"))
    parser.add_argument("--q_methods", type=str, default="iql,vdn,pqn_vdn")
    parser.add_argument("--max_q_policies", type=int, default=5)
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--q_action_mode", type=str, default="greedy", choices=("greedy", "softmax", "epsilon_greedy"))
    parser.add_argument("--q_temperature", type=float, default=1.0)
    parser.add_argument("--q_epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=12000)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--policy_prob_batch_size", type=int, default=1024)
    parser.add_argument("--high_quantile", type=float, default=0.8)
    parser.add_argument("--low_quantile", type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = np.load(args.dataset, allow_pickle=False)
    n_rows = int(data["query_obs"].shape[0])
    ego_policy_ids = string_array(data, "ego_policy_id", n_rows)
    partner_policy_ids = string_array(data, "partner_policy_id", n_rows)
    partner_methods = np.asarray([method_from_policy_id(x) for x in partner_policy_ids], dtype=str)
    allowed_methods = {x.strip() for x in args.q_methods.split(",") if x.strip()}
    mask = np.asarray([m in allowed_methods for m in partner_methods], dtype=bool)
    mask &= np.char.startswith(ego_policy_ids.astype(str), "ppo:")
    candidate_idx = np.where(mask)[0]
    if len(candidate_idx) == 0:
        raise ValueError("no rows with PPO ego and requested Q partner methods")
    if args.max_rows and len(candidate_idx) > args.max_rows:
        candidate_idx = np.sort(rng.choice(candidate_idx, size=args.max_rows, replace=False))

    query_obs = np.asarray(data["query_obs"][candidate_idx], dtype=np.float32)
    obs_hist = np.asarray(data["partner_obs_history"][candidate_idx], dtype=np.float32)
    act_hist = np.asarray(data["partner_action_history"][candidate_idx], dtype=np.int32)
    target = normalize_probs(np.asarray(data["target_partner_probs"][candidate_idx], dtype=np.float32))
    ego_policy_ids = ego_policy_ids[candidate_idx]
    partner_policy_ids = partner_policy_ids[candidate_idx]
    partner_methods = np.asarray([method_from_policy_id(x) for x in partner_policy_ids], dtype=str)
    return_to_go = (
        np.asarray(data["return_to_go"][candidate_idx], dtype=np.float32)
        if "return_to_go" in data.files
        else np.zeros((len(candidate_idx),), dtype=np.float32)
    )

    print(f"[local_decision] selected rows={len(candidate_idx)}", flush=True)
    base_probs, valid_base = load_base_probs(args, query_obs, ego_policy_ids)
    if not np.all(valid_base):
        keep = np.where(valid_base)[0]
        print(f"[local_decision] drop rows without base PPO probs: {len(valid_base) - len(keep)}", flush=True)
        query_obs = query_obs[keep]
        obs_hist = obs_hist[keep]
        act_hist = act_hist[keep]
        target = target[keep]
        base_probs = base_probs[keep]
        ego_policy_ids = ego_policy_ids[keep]
        partner_policy_ids = partner_policy_ids[keep]
        partner_methods = partner_methods[keep]
        return_to_go = return_to_go[keep]
    old_probs = predict_decoder_probs(
        args.old_decoder_path,
        query_obs,
        obs_hist,
        act_hist,
        args.action_dim,
        args.eval_batch_size,
    )
    new_probs = predict_decoder_probs(
        args.new_decoder_path,
        query_obs,
        obs_hist,
        act_hist,
        args.action_dim,
        args.eval_batch_size,
    )

    q_entries = load_q_pool(
        args.q_run_root,
        args.q_methods,
        args.layout,
        args.max_q_policies,
        args.q_action_mode,
        args.q_temperature,
        args.q_epsilon,
    )
    q_stack = []
    for pos, entry in enumerate(q_entries):
        print(f"[local_decision] disagreement teacher {pos + 1}/{len(q_entries)} {entry.pool}:{entry.label}", flush=True)
        q_stack.append(
            policy_probs_on_query_obs(
                entry.policy,
                query_obs,
                batch_size=args.policy_prob_batch_size,
            ).astype(np.float32)
        )
    q_stack = np.stack(q_stack, axis=0)
    partner_disagreement = mean_pairwise_tv(q_stack)
    oracle_base_tv = categorical_tv_np(target, base_probs)
    joint_score = partner_disagreement * oracle_base_tv
    reward_joint_score = joint_score * np.maximum(return_to_go, 0.0)

    buckets = {"all": np.ones((len(query_obs),), dtype=bool)}
    bucket_thresholds = {}
    for name, values, q, high in [
        ("high_partner_disagreement", partner_disagreement, args.high_quantile, True),
        ("low_partner_disagreement", partner_disagreement, args.low_quantile, False),
        ("high_oracle_base_tv", oracle_base_tv, args.high_quantile, True),
        ("low_oracle_base_tv", oracle_base_tv, args.low_quantile, False),
        ("high_return_to_go", return_to_go, args.high_quantile, True),
        ("high_joint_disagreement_base_tv", joint_score, args.high_quantile, True),
        ("high_reward_joint", reward_joint_score, args.high_quantile, True),
    ]:
        bucket_mask, threshold = quantile_mask(values, q, high=high)
        buckets[name] = bucket_mask
        bucket_thresholds[name] = threshold

    probs_by_method = {
        "base": base_probs,
        "old_qexp4_vae": old_probs,
        "new_cf_znorm_vae": new_probs,
        "oracle": target,
    }
    aux = {
        "oracle_base_tv": oracle_base_tv,
        "partner_disagreement": partner_disagreement,
        "return_to_go": return_to_go,
    }
    rows = []
    groups = ["all"] + sorted(set(partner_methods.tolist()))
    for bucket_name, bucket_mask in buckets.items():
        for group in groups:
            if group == "all":
                group_mask = bucket_mask
            else:
                group_mask = bucket_mask & (partner_methods == group)
            rows.extend(
                summarize_metrics(
                    bucket_name,
                    "all",
                    group,
                    group_mask,
                    target,
                    probs_by_method,
                    aux,
                )
            )
    write_csv(out / "local_decision_metrics.csv", rows)

    with (out / "local_decision_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Local decision-point estimator benchmark\n\n")
        f.write(f"- dataset: `{args.dataset}`\n")
        f.write(f"- rows: `{len(query_obs)}`\n")
        f.write(f"- old_decoder_path: `{args.old_decoder_path}`\n")
        f.write(f"- new_decoder_path: `{args.new_decoder_path}`\n")
        f.write(f"- q_methods: `{args.q_methods}`\n")
        f.write(f"- max_q_policies: `{args.max_q_policies}`\n")
        f.write(f"- high_quantile: `{args.high_quantile}`\n")
        f.write(f"- low_quantile: `{args.low_quantile}`\n\n")
        f.write("## Bucket thresholds\n\n")
        for name, threshold in bucket_thresholds.items():
            f.write(f"- {name}: `{threshold}`\n")
        f.write("\n## All-partner bucket summary\n\n")
        f.write("| bucket | method | n | KL | TV | argmax | oracle-base TV | partner disagreement | RTG |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            if row["partner_group"] != "all":
                continue
            f.write(
                f"| {row['bucket']} | {row['method']} | {row['n']} | "
                f"{row['kl']:.4f} | {row['tv']:.4f} | {row['argmax_acc']:.4f} | "
                f"{row['oracle_base_tv']:.4f} | {row['partner_disagreement']:.4f} | {row['return_to_go']:.4f} |\n"
            )
    print(f"[local_decision] wrote {out / 'local_decision_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
