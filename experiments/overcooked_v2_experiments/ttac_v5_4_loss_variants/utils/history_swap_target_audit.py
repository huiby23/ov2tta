from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import jax
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.ttac_v5_4_loss_variants.utils.agreement_heads import load_agreement_npz
from overcooked_v2_experiments.ttac_v5_4_loss_variants.utils.store import load_all_checkpoints
from overcooked_v2_experiments.ttac_v5_4_loss_variants.utils.target_effect_audit import (
    adapt_one_policy,
    build_env,
    flatten_role_arrays,
    make_policy,
    policy_change_metrics,
    policy_pair_metrics,
    rollout_pair,
    write_csv,
)


def load_run_with_overrides(run_dir, args):
    all_checkpoints, config = load_all_checkpoints(Path(run_dir), final_only=True)
    config["model"]["TTAC_V5_ESTIMATOR"] = load_agreement_npz(args.ttac_v5_estimator_path)
    config["model"]["TTAC_V5_AGREEMENT_COEF"] = args.ttac_v5_agreement_coef
    config["model"]["TTAC_V5_SUPPORT_COEF"] = args.ttac_v5_support_coef
    config["model"]["TTAC_TEST_LR"] = args.ttac_test_lr
    config["model"]["TTAC_TEST_UPDATE_STEPS"] = args.ttac_test_update_steps
    config["model"]["TTAC_HISTORY_LEN"] = args.ttac_history_len
    config["model"]["TTAC_TEST_HIST_KL_COEF"] = args.ttac_test_hist_kl_coef
    config["model"]["TTAC_TEST_EGO_KL_COEF"] = args.ttac_test_ego_kl_coef
    config["model"]["TTAC_TEST_CUR_KL_COEF"] = args.ttac_test_cur_kl_coef
    config["model"]["TTAC_V5_CONF_MAX_ENTROPY"] = args.ttac_v5_conf_max_entropy
    config["model"]["TTAC_V5_CONF_MIN_TARGET_BASE_TV"] = args.ttac_v5_conf_min_target_base_tv
    run_keys = sorted(all_checkpoints.keys(), key=lambda name: int(name.split("_")[1]))
    params = [all_checkpoints[key]["ckpt_final"].params for key in run_keys]
    return run_keys, params, config


def first_swap_idx(num_runs, *exclude):
    excluded = set(exclude)
    for idx in range(num_runs):
        if idx not in excluded:
            return idx
    return 0


def current_prefix_arrays(rollout, role):
    ego_id = f"agent_{role}"
    partner_id = f"agent_{1-role}"
    return (
        np.asarray(rollout.obs_seq[ego_id]),
        np.asarray(rollout.done_seq[ego_id]),
        np.asarray(rollout.obs_seq[partner_id]),
        np.asarray(rollout.actions_seq[partner_id]),
    )


def swapped_prefix_arrays(current_rollout, swap_rollout, role):
    ego_id = f"agent_{role}"
    # swap_rollout is constructed with the same ego policy in agent_0 and a different partner in agent_1.
    return (
        np.asarray(current_rollout.obs_seq[ego_id]),
        np.asarray(current_rollout.done_seq[ego_id]),
        np.asarray(swap_rollout.obs_seq["agent_1"]),
        np.asarray(swap_rollout.actions_seq["agent_1"]),
    )


def mean_numeric(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ttac_v5_estimator_path", required=True)
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pairs", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--prefix_steps", type=int, default=80)
    parser.add_argument("--eval_suffix_start", type=int, default=80)
    parser.add_argument("--compatibility_sample_limit", type=int, default=64)
    parser.add_argument("--eval_mode", default="ttac_v5_1_multiquery")
    parser.add_argument("--ttac_v5_agreement_coef", type=float, default=5.0)
    parser.add_argument("--ttac_v5_support_coef", type=float, default=1.0)
    parser.add_argument("--ttac_test_lr", type=float, default=0.006)
    parser.add_argument("--ttac_test_update_steps", type=int, default=5)
    parser.add_argument("--ttac_history_len", type=int, default=50)
    parser.add_argument("--ttac_test_hist_kl_coef", type=float, default=0.0)
    parser.add_argument("--ttac_test_ego_kl_coef", type=float, default=0.005)
    parser.add_argument("--ttac_test_cur_kl_coef", type=float, default=0.005)
    parser.add_argument("--ttac_v5_conf_max_entropy", type=float, default=1.25)
    parser.add_argument("--ttac_v5_conf_min_target_base_tv", type=float, default=0.03)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_keys, params, config = load_run_with_overrides(args.run_dir, args)
    env = build_env(config, layout_override=args.layout)
    pair_indices = list(itertools.permutations(range(len(run_keys)), 2))[: args.max_pairs]
    rows = []

    for pair_pos, (lhs_idx, rhs_idx) in enumerate(pair_indices):
        lhs_key, rhs_key = run_keys[lhs_idx], run_keys[rhs_idx]
        lhs_swap_idx = first_swap_idx(len(run_keys), lhs_idx, rhs_idx)
        rhs_swap_idx = first_swap_idx(len(run_keys), rhs_idx, lhs_idx, lhs_swap_idx)
        print(f"[swap_audit] pair {pair_pos+1}/{len(pair_indices)} {lhs_key}x{rhs_key} swap_lhs={run_keys[lhs_swap_idx]} swap_rhs={run_keys[rhs_swap_idx]}", flush=True)
        current = rollout_pair(params[lhs_idx], params[rhs_idx], config, env, args.seed + 10000 + pair_pos * 1000, args.num_episodes)
        swap_lhs = rollout_pair(params[lhs_idx], params[lhs_swap_idx], config, env, args.seed + 20000 + pair_pos * 1000, args.num_episodes)
        swap_rhs = rollout_pair(params[rhs_idx], params[rhs_swap_idx], config, env, args.seed + 30000 + pair_pos * 1000, args.num_episodes)

        total_steps = np.asarray(current.done_seq["agent_0"]).shape[1]
        suffix_start = min(args.eval_suffix_start, total_steps - 1)
        eval_mask = np.zeros(np.asarray(current.done_seq["agent_0"]).shape[:2], dtype=bool)
        eval_mask[:, suffix_start:] = True
        states, obs0, obs1, done0, done1 = flatten_role_arrays(current, eval_mask)
        base_lhs = make_policy(params[lhs_idx], config, "base_no_test_adapt", stochastic=False)
        base_rhs = make_policy(params[rhs_idx], config, "base_no_test_adapt", stochastic=False)
        metric_seed = args.seed + 40000 + pair_pos
        base_metrics = policy_pair_metrics(env, states, obs0, obs1, done0, done1, base_lhs, base_rhs, args.compatibility_sample_limit, metric_seed)

        for history_source in ("true", "swap"):
            spec = {"eval_mode": args.eval_mode, "do_update": True, "shuffle_actions": False, "use_loss": True}
            if history_source == "true":
                lhs_inputs = current_prefix_arrays(current, 0)
                rhs_inputs = current_prefix_arrays(current, 1)
            else:
                lhs_inputs = swapped_prefix_arrays(current, swap_lhs, 0)
                rhs_inputs = swapped_prefix_arrays(current, swap_rhs, 1)
            adapted_lhs_params, lhs_before, lhs_after = adapt_one_policy(params[lhs_idx], config, spec, *lhs_inputs, seed=args.seed + 50000 + pair_pos, prefix_steps=args.prefix_steps)
            adapted_rhs_params, rhs_before, rhs_after = adapt_one_policy(params[rhs_idx], config, spec, *rhs_inputs, seed=args.seed + 60000 + pair_pos, prefix_steps=args.prefix_steps)
            adapted_lhs = make_policy(adapted_lhs_params, config, "base_no_test_adapt", stochastic=False)
            adapted_rhs = make_policy(adapted_rhs_params, config, "base_no_test_adapt", stochastic=False)
            after = policy_pair_metrics(env, states, obs0, obs1, done0, done1, adapted_lhs, adapted_rhs, args.compatibility_sample_limit, metric_seed)
            lhs_change = policy_change_metrics(base_lhs, adapted_lhs, obs0, obs1, done0, done1)
            rhs_change = policy_change_metrics(base_rhs, adapted_rhs, obs0, obs1, done0, done1)
            row = {
                "pair": f"{lhs_key}x{rhs_key}",
                "history_source": history_source,
                "swap_lhs_partner": run_keys[lhs_swap_idx],
                "swap_rhs_partner": run_keys[rhs_swap_idx],
                "objective_loss_before": float(np.nanmean([lhs_before, rhs_before])),
                "objective_loss_after": float(np.nanmean([lhs_after, rhs_after])),
                "delta_agreement": after["action_agreement"] - base_metrics["action_agreement"],
                "delta_policy_tv": after["policy_tv"] - base_metrics["policy_tv"],
                "delta_joint_optimal": after["joint_optimal_rate"] - base_metrics["joint_optimal_rate"],
                "delta_joint_regret": after["joint_regret"] - base_metrics["joint_regret"],
                "adapter_action_change_rate": 0.5 * (lhs_change["adapter_action_change_rate"] + rhs_change["adapter_action_change_rate"]),
                "adapter_policy_tv_to_base": 0.5 * (lhs_change["adapter_policy_tv_to_base"] + rhs_change["adapter_policy_tv_to_base"]),
            }
            rows.append(row)
            print(f"[swap_audit] {history_source} {row}", flush=True)

    write_csv(output_dir / "history_swap_pairs.csv", rows)
    summary = []
    for source in ("true", "swap"):
        source_rows = [r for r in rows if r["history_source"] == source]
        summary.append({
            "history_source": source,
            "pairs": len(source_rows),
            "delta_agreement": mean_numeric(source_rows, "delta_agreement"),
            "delta_policy_tv": mean_numeric(source_rows, "delta_policy_tv"),
            "delta_joint_optimal": mean_numeric(source_rows, "delta_joint_optimal"),
            "delta_joint_regret": mean_numeric(source_rows, "delta_joint_regret"),
            "adapter_action_change_rate": mean_numeric(source_rows, "adapter_action_change_rate"),
            "adapter_policy_tv_to_base": mean_numeric(source_rows, "adapter_policy_tv_to_base"),
        })
    write_csv(output_dir / "history_swap_summary.csv", summary)
    md = ["# TTAC v5.1 history-swap target audit", "", "| history_source | pairs | d_agreement | d_policy_tv | d_joint_optimal | d_joint_regret | action_change | policy_tv_to_base |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        md.append(f"| {r['history_source']} | {r['pairs']} | {r['delta_agreement']:.6f} | {r['delta_policy_tv']:.6f} | {r['delta_joint_optimal']:.6f} | {r['delta_joint_regret']:.6f} | {r['adapter_action_change_rate']:.6f} | {r['adapter_policy_tv_to_base']:.6f} |")
    (output_dir / "history_swap_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
