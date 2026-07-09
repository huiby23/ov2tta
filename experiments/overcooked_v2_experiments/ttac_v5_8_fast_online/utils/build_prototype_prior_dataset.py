#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.collect_pg_mixed_latent_decoder_dataset import (
    load_mappo_pool,
    load_ppo_pool,
    parse_policy_indices,
    policy_probs_on_query_obs,
)

EPS = 1e-6


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def load_prototypes(args):
    entries = []
    if args.max_ppo_policies != 0:
        ppo, _config = load_ppo_pool(
            args.ppo_run_dir,
            args.max_ppo_policies,
            stochastic=False,
            pool_name="ppo",
            backend="ttac",
            policy_indices=parse_policy_indices(args.ppo_policy_indices),
        )
        entries.extend(ppo)
    if args.max_mappo_policies != 0:
        mappo, _config = load_mappo_pool(
            args.mappo_run_dir,
            args.max_mappo_policies,
            stochastic=False,
            policy_indices=parse_policy_indices(args.mappo_policy_indices),
        )
        entries.extend(mappo)
    if not entries:
        raise ValueError("no prototype policies loaded")
    return entries


def history_scores(policy, obs_hist, act_hist, batch_size, score_mode):
    n, history_len = act_hist.shape[:2]
    flat_obs = obs_hist.reshape((n * history_len,) + obs_hist.shape[2:])
    probs = policy_probs_on_query_obs(policy, flat_obs, batch_size=batch_size)
    probs = probs.reshape((n, history_len, -1))
    actions = np.asarray(act_hist, dtype=np.int32)
    action_probs = np.take_along_axis(probs, actions[..., None], axis=-1)[..., 0]
    valid = np.sum(np.abs(obs_hist), axis=tuple(range(2, obs_hist.ndim))) > EPS
    logp = np.log(np.maximum(action_probs, EPS)) * valid.astype(np.float32)
    scores = logp.sum(axis=1)
    if score_mode == "mean_logp":
        scores = scores / np.maximum(valid.sum(axis=1), 1.0)
    return scores.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--ppo_policy_indices", type=str)
    parser.add_argument("--mappo_policy_indices", type=str)
    parser.add_argument("--max_ppo_policies", type=int, default=8)
    parser.add_argument("--max_mappo_policies", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--score_mode", choices=("sum_logp", "mean_logp"), default="mean_logp")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.source_dataset, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    required = ["query_obs", "partner_obs_history", "partner_action_history"]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ValueError(f"dataset missing fields: {missing}")

    n = arrays["query_obs"].shape[0]
    if args.max_rows and args.max_rows < n:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(np.arange(n), size=args.max_rows, replace=False).astype(np.int32))
    else:
        idx = np.arange(n, dtype=np.int32)
    query_obs = np.asarray(arrays["query_obs"][idx], dtype=np.float32)
    obs_hist = np.asarray(arrays["partner_obs_history"][idx], dtype=np.float32)
    act_hist = np.asarray(arrays["partner_action_history"][idx], dtype=np.int32)

    prototypes = load_prototypes(args)
    scores = []
    query_probs = []
    labels = []
    for entry in prototypes:
        label = f"{entry.pool}:{entry.label}"
        print(f"[prototype_prior] scoring {label}", flush=True)
        scores.append(history_scores(entry.policy, obs_hist, act_hist, args.batch_size, args.score_mode))
        query_probs.append(policy_probs_on_query_obs(entry.policy, query_obs, batch_size=args.batch_size))
        labels.append(label)
    scores = np.stack(scores, axis=0)
    query_probs = np.stack(query_probs, axis=0).astype(np.float32)
    temp = max(float(args.temperature), EPS)
    posterior_logits = scores / temp
    posterior_logits = posterior_logits - posterior_logits.max(axis=0, keepdims=True)
    posterior = np.exp(posterior_logits)
    posterior = posterior / np.maximum(posterior.sum(axis=0, keepdims=True), EPS)
    prior = np.sum(posterior[..., None] * query_probs, axis=0).astype(np.float32)
    prior = normalize_probs(prior)

    saved = {k: v[idx] for k, v in arrays.items()}
    saved["source_row"] = idx.astype(np.int32)
    saved["prototype_prior_probs"] = prior
    saved["prototype_prior_entropy"] = (
        -np.sum(prior * np.log(np.maximum(prior, EPS)), axis=-1).astype(np.float32)
    )
    saved["prototype_prior_max_posterior"] = posterior.max(axis=0).astype(np.float32)
    output_path = out / "prototype_prior_latent_decoder_dataset.npz"
    np.savez(output_path, **saved)

    lines = ["# Prototype Prior Dataset", ""]
    lines.append(f"- source_dataset: `{args.source_dataset}`")
    lines.append(f"- output: `{output_path}`")
    lines.append(f"- rows: `{len(idx)}`")
    lines.append(f"- prototype_count: `{len(labels)}`")
    lines.append(f"- temperature: `{args.temperature}`")
    lines.append(f"- score_mode: `{args.score_mode}`")
    lines.append(f"- mean max posterior: `{float(np.mean(saved['prototype_prior_max_posterior'])):.6f}`")
    lines.append("")
    lines.append("## Prototypes")
    lines.extend(f"- `{label}`" for label in labels)
    (out / "prototype_prior_dataset_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
