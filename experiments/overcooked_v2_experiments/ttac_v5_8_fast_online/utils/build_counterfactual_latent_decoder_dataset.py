from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.collect_pg_mixed_latent_decoder_dataset import (  # noqa: E501
    load_mappo_pool,
    load_ppo_pool,
    parse_policy_indices,
    policy_probs_on_query_obs,
)


STRING_KEYS = {
    "pair_label",
    "ego_pool",
    "partner_pool",
    "ego_family",
    "partner_family",
    "ego_policy_id",
    "partner_policy_id",
}


def string_array(data, key, n_rows, fallback="unknown"):
    if key in data.files:
        return np.asarray(data[key]).astype(str)
    return np.full((n_rows,), fallback, dtype=object).astype(str)


def family_from_policy_id(policy_id: str) -> str:
    return str(policy_id).split(":", 1)[0]


def mean_pairwise_tv(probs: np.ndarray) -> np.ndarray:
    scores = np.zeros((probs.shape[1],), dtype=np.float32)
    count = 0
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            scores += 0.5 * np.abs(probs[i] - probs[j]).sum(axis=-1)
            count += 1
    if count == 0:
        return scores
    return scores / float(count)


def mean_argmax_disagreement(probs: np.ndarray) -> np.ndarray:
    argmax = np.argmax(probs, axis=-1)
    scores = np.zeros((probs.shape[1],), dtype=np.float32)
    count = 0
    for i in range(argmax.shape[0]):
        for j in range(i + 1, argmax.shape[0]):
            scores += (argmax[i] != argmax[j]).astype(np.float32)
            count += 1
    if count == 0:
        return scores
    return scores / float(count)


def choose_support_indices(
    rng: np.random.Generator,
    support_by_policy: dict[str, np.ndarray],
    policy_ids: np.ndarray,
    avoid_episodes: np.ndarray,
    episode_ids: np.ndarray,
) -> np.ndarray:
    support_idx = np.empty((len(policy_ids),), dtype=np.int32)
    for row, policy_id in enumerate(policy_ids.astype(str)):
        candidates = support_by_policy[policy_id]
        if len(candidates) == 0:
            raise ValueError(f"no support rows for policy {policy_id}")
        different_episode = candidates[episode_ids[candidates] != avoid_episodes[row]]
        if len(different_episode):
            candidates = different_episode
        support_idx[row] = int(rng.choice(candidates))
    return support_idx


def build_policy_entries(args):
    ppo_entries, _ppo_config = load_ppo_pool(
        args.ppo_run_dir,
        args.max_ppo_policies,
        stochastic=False,
        policy_indices=parse_policy_indices(args.ppo_policy_indices),
    )
    mappo_entries, _mappo_config = load_mappo_pool(
        args.mappo_run_dir,
        args.max_mappo_policies,
        stochastic=False,
        policy_indices=parse_policy_indices(args.mappo_policy_indices),
    )
    return ppo_entries + mappo_entries


def add_optional_source_fields(arrays, data, row_idx, prefix=""):
    del prefix
    out = {}
    for key in [
        "partner_obs",
        "partner_action",
        "prev_partner_action",
        "prev_ego_action",
        "step_reward",
        "return_to_go",
        "episode_return",
        "timestep",
        "role",
    ]:
        if key in data.files:
            out[key] = np.asarray(data[key])[row_idx]
    return out


def concatenate_parts(parts):
    keys = sorted({key for part in parts for key in part})
    arrays = {}
    for key in keys:
        values = [part[key] for part in parts if key in part]
        if key in STRING_KEYS:
            arrays[key] = np.concatenate([np.asarray(v).astype(str) for v in values], axis=0)
        else:
            arrays[key] = np.concatenate(values, axis=0)
    return arrays


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
    parser.add_argument("--ppo_policy_indices", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--mappo_policy_indices", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_ppo_policies", type=int, default=None)
    parser.add_argument("--max_mappo_policies", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate_queries", type=int, default=20000)
    parser.add_argument("--top_query_fraction", type=float, default=0.5)
    parser.add_argument("--max_queries", type=int, default=5000)
    parser.add_argument("--policies_per_query", type=int, default=8)
    parser.add_argument("--original_fraction", type=float, default=0.5)
    parser.add_argument("--max_rows", type=int, default=160000)
    parser.add_argument("--policy_prob_batch_size", type=int, default=2048)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = np.load(args.source_dataset, allow_pickle=False)
    n_rows = int(data["query_obs"].shape[0])
    episode_ids = np.asarray(data["episode_id"], dtype=np.int32)
    source_partner_ids = string_array(data, "partner_policy_id", n_rows)
    source_ego_ids = string_array(data, "ego_policy_id", n_rows)
    source_ego_families = string_array(data, "ego_family", n_rows)
    source_pair_labels = string_array(data, "pair_label", n_rows)
    source_partner_families = string_array(data, "partner_family", n_rows)

    entries = build_policy_entries(args)
    policy_ids = np.asarray([f"{entry.pool}:{entry.label}" for entry in entries])
    support_by_policy = {
        policy_id: np.where(source_partner_ids == policy_id)[0].astype(np.int32)
        for policy_id in policy_ids
    }
    available = np.asarray([len(support_by_policy[policy_id]) > 0 for policy_id in policy_ids])
    if not np.any(available):
        raise ValueError("none of the loaded policies has support rows in source_dataset")
    if not np.all(available):
        skipped = policy_ids[~available].tolist()
        print(f"[counterfactual_dataset] skip policies without support: {skipped}", flush=True)
    entries = [entry for entry, keep in zip(entries, available) if keep]
    policy_ids = policy_ids[available]
    support_by_policy = {policy_id: support_by_policy[policy_id] for policy_id in policy_ids}

    candidate_count = min(n_rows, max(1, int(args.candidate_queries)))
    candidate_source_idx = rng.choice(n_rows, size=candidate_count, replace=n_rows < candidate_count)
    candidate_query_obs = np.asarray(data["query_obs"][candidate_source_idx], dtype=np.float32)
    teacher_probs = []
    for pos, entry in enumerate(entries):
        policy_id = f"{entry.pool}:{entry.label}"
        print(
            f"[counterfactual_dataset] teacher {pos + 1}/{len(entries)} {policy_id}",
            flush=True,
        )
        teacher_probs.append(
            policy_probs_on_query_obs(
                entry.policy,
                candidate_query_obs,
                batch_size=args.policy_prob_batch_size,
            ).astype(np.float32)
        )
    teacher_probs = np.stack(teacher_probs, axis=0)
    tv_scores = mean_pairwise_tv(teacher_probs)
    argmax_disagreement = mean_argmax_disagreement(teacher_probs)

    query_order = np.argsort(-tv_scores)
    top_count = max(1, int(np.ceil(candidate_count * min(max(args.top_query_fraction, 0.0), 1.0))))
    if args.max_queries and args.max_queries > 0:
        top_count = min(top_count, int(args.max_queries))
    selected_query_pos = query_order[:top_count].astype(np.int32)

    policies_per_query = min(max(1, int(args.policies_per_query)), len(policy_ids))
    cf_query_pos = []
    cf_policy_pos = []
    cf_group_id = []
    for group_id, query_pos in enumerate(selected_query_pos):
        chosen = rng.choice(
            len(policy_ids),
            size=policies_per_query,
            replace=len(policy_ids) < policies_per_query,
        )
        cf_query_pos.append(np.full((len(chosen),), int(query_pos), dtype=np.int32))
        cf_policy_pos.append(chosen.astype(np.int32))
        cf_group_id.append(np.full((len(chosen),), int(group_id), dtype=np.int32))
    cf_query_pos = np.concatenate(cf_query_pos, axis=0)
    cf_policy_pos = np.concatenate(cf_policy_pos, axis=0)
    cf_group_id = np.concatenate(cf_group_id, axis=0)

    original_fraction = min(max(float(args.original_fraction), 0.0), 0.95)
    if args.max_rows and args.max_rows > 0:
        max_cf_rows = int(args.max_rows * (1.0 - original_fraction))
        if max_cf_rows > 0 and len(cf_query_pos) > max_cf_rows:
            keep = rng.choice(len(cf_query_pos), size=max_cf_rows, replace=False)
            keep = np.sort(keep)
            cf_query_pos = cf_query_pos[keep]
            cf_policy_pos = cf_policy_pos[keep]
            cf_group_id = cf_group_id[keep]

    cf_source_idx = candidate_source_idx[cf_query_pos]
    cf_policy_ids = policy_ids[cf_policy_pos].astype(str)
    cf_support_idx = choose_support_indices(
        rng,
        support_by_policy,
        cf_policy_ids,
        episode_ids[cf_source_idx],
        episode_ids,
    )
    cf_targets = teacher_probs[cf_policy_pos, cf_query_pos]
    cf_part = {
        "query_obs": np.asarray(data["query_obs"][cf_source_idx], dtype=np.float16),
        "partner_obs_history": np.asarray(data["partner_obs_history"][cf_support_idx], dtype=np.float16),
        "partner_action_history": np.asarray(data["partner_action_history"][cf_support_idx], dtype=np.int16),
        "ego_action_history": np.asarray(data["ego_action_history"][cf_support_idx], dtype=np.int16),
        "target_partner_probs": cf_targets.astype(np.float32),
        "ego_action": np.asarray(data["ego_action"][cf_source_idx], dtype=np.int16),
        "episode_id": (1_000_000 + cf_group_id).astype(np.int32),
        "query_group_id": cf_group_id.astype(np.int32),
        "query_disagreement_tv": tv_scores[cf_query_pos].astype(np.float32),
        "query_argmax_disagreement": argmax_disagreement[cf_query_pos].astype(np.float32),
        "source_row": cf_source_idx.astype(np.int32),
        "support_row": cf_support_idx.astype(np.int32),
        "pair_label": np.asarray(
            [
                f"counterfactual:{policy_id}@{source_pair_labels[src]}"
                for policy_id, src in zip(cf_policy_ids, cf_source_idx)
            ],
            dtype=str,
        ),
        "ego_pool": source_ego_families[cf_source_idx].astype(str),
        "partner_pool": np.asarray([family_from_policy_id(x) for x in cf_policy_ids], dtype=str),
        "ego_family": source_ego_families[cf_source_idx].astype(str),
        "partner_family": np.asarray([family_from_policy_id(x) for x in cf_policy_ids], dtype=str),
        "ego_policy_id": source_ego_ids[cf_source_idx].astype(str),
        "partner_policy_id": cf_policy_ids.astype(str),
    }
    cf_part.update(add_optional_source_fields(cf_part, data, cf_source_idx))

    parts = [cf_part]
    if original_fraction > 0.0 and len(cf_source_idx) > 0:
        original_count = int(round(len(cf_source_idx) * original_fraction / max(1.0 - original_fraction, 1e-6)))
        if args.max_rows and args.max_rows > 0:
            original_count = min(original_count, max(0, int(args.max_rows) - len(cf_source_idx)))
        original_count = min(original_count, n_rows)
        if original_count > 0:
            orig_idx = rng.choice(n_rows, size=original_count, replace=n_rows < original_count)
            orig_group = (2_000_000 + np.arange(original_count, dtype=np.int32))
            orig_part = {
                "query_obs": np.asarray(data["query_obs"][orig_idx], dtype=np.float16),
                "partner_obs_history": np.asarray(data["partner_obs_history"][orig_idx], dtype=np.float16),
                "partner_action_history": np.asarray(data["partner_action_history"][orig_idx], dtype=np.int16),
                "ego_action_history": np.asarray(data["ego_action_history"][orig_idx], dtype=np.int16),
                "target_partner_probs": np.asarray(data["target_partner_probs"][orig_idx], dtype=np.float32),
                "ego_action": np.asarray(data["ego_action"][orig_idx], dtype=np.int16),
                "episode_id": np.asarray(data["episode_id"][orig_idx], dtype=np.int32),
                "query_group_id": orig_group,
                "query_disagreement_tv": np.zeros((original_count,), dtype=np.float32),
                "query_argmax_disagreement": np.zeros((original_count,), dtype=np.float32),
                "source_row": orig_idx.astype(np.int32),
                "support_row": orig_idx.astype(np.int32),
                "pair_label": source_pair_labels[orig_idx].astype(str),
                "ego_pool": source_ego_families[orig_idx].astype(str),
                "partner_pool": source_partner_families[orig_idx].astype(str),
                "ego_family": source_ego_families[orig_idx].astype(str),
                "partner_family": source_partner_families[orig_idx].astype(str),
                "ego_policy_id": source_ego_ids[orig_idx].astype(str),
                "partner_policy_id": source_partner_ids[orig_idx].astype(str),
            }
            orig_part.update(add_optional_source_fields(orig_part, data, orig_idx))
            parts.append(orig_part)

    arrays = concatenate_parts(parts)
    if args.max_rows and args.max_rows > 0 and len(arrays["ego_action"]) > args.max_rows:
        keep = rng.choice(len(arrays["ego_action"]), size=int(args.max_rows), replace=False)
        keep = np.sort(keep)
        for key in list(arrays.keys()):
            arrays[key] = arrays[key][keep]

    dataset_path = output / "counterfactual_latent_decoder_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)

    summary_rows = []
    for policy_id in policy_ids.astype(str):
        summary_rows.append({
            "policy_id": policy_id,
            "support_rows": int(len(support_by_policy[policy_id])),
            "counterfactual_rows": int(np.sum(arrays["partner_policy_id"].astype(str) == policy_id)),
        })
    with (output / "policy_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["policy_id", "support_rows", "counterfactual_rows"])
        writer.writeheader()
        writer.writerows(summary_rows)

    cf_mask = arrays["source_row"] != arrays["support_row"]
    selected_scores = tv_scores[selected_query_pos]
    with (output / "dataset_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Counterfactual latent-decoder dataset\n\n")
        f.write(f"- source_dataset: `{args.source_dataset}`\n")
        f.write(f"- output_dataset: `{dataset_path}`\n")
        f.write(f"- source_rows: `{n_rows}`\n")
        f.write(f"- loaded_policies: `{len(policy_ids)}`\n")
        f.write(f"- candidate_queries: `{candidate_count}`\n")
        f.write(f"- selected_queries: `{len(selected_query_pos)}`\n")
        f.write(f"- policies_per_query: `{policies_per_query}`\n")
        f.write(f"- original_fraction: `{original_fraction}`\n")
        f.write(f"- stored_rows: `{len(arrays['ego_action'])}`\n")
        f.write(f"- counterfactual_like_rows: `{int(np.sum(cf_mask))}`\n")
        f.write(f"- all_candidate_mean_tv: `{float(tv_scores.mean())}`\n")
        f.write(f"- all_candidate_p90_tv: `{float(np.quantile(tv_scores, 0.9))}`\n")
        f.write(f"- selected_mean_tv: `{float(selected_scores.mean())}`\n")
        f.write(f"- selected_min_tv: `{float(selected_scores.min())}`\n")
        f.write(f"- selected_max_tv: `{float(selected_scores.max())}`\n")
        f.write(f"- selected_argmax_disagreement_mean: `{float(argmax_disagreement[selected_query_pos].mean())}`\n")
        f.write("- target: selected policy distribution on the shared query observation.\n")
        f.write("- support: partner obs/action history sampled from the selected policy.\n")
    print(
        f"[counterfactual_dataset] wrote {dataset_path} rows={len(arrays['ego_action'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
