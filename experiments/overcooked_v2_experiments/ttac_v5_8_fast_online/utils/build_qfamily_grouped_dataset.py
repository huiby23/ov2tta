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
    load_q_pool,
    policy_probs_on_query_obs,
)


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
    return scores / max(float(count), 1.0)


def mean_argmax_disagreement(probs: np.ndarray) -> np.ndarray:
    argmax = np.argmax(probs, axis=-1)
    scores = np.zeros((argmax.shape[1],), dtype=np.float32)
    count = 0
    for i in range(argmax.shape[0]):
        for j in range(i + 1, argmax.shape[0]):
            scores += (argmax[i] != argmax[j]).astype(np.float32)
            count += 1
    return scores / max(float(count), 1.0)


def choose_support_indices(
    rng: np.random.Generator,
    support_by_policy: dict[str, np.ndarray],
    policy_ids: np.ndarray,
    avoid_episodes: np.ndarray,
    episode_ids: np.ndarray,
) -> np.ndarray:
    out = np.empty((len(policy_ids),), dtype=np.int32)
    for row, policy_id in enumerate(policy_ids.astype(str)):
        candidates = support_by_policy[policy_id]
        if len(candidates) == 0:
            raise ValueError(f"no support rows for policy {policy_id}")
        different_episode = candidates[episode_ids[candidates] != avoid_episodes[row]]
        if len(different_episode):
            candidates = different_episode
        out[row] = int(rng.choice(candidates))
    return out


def balanced_query_positions(
    rng: np.random.Generator,
    scores: np.ndarray,
    query_budget: int,
    high_fraction: float,
    medium_fraction: float,
    low_fraction: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    q33 = float(np.quantile(scores, 1.0 / 3.0))
    q66 = float(np.quantile(scores, 2.0 / 3.0))
    bucket_indices = {
        "low": np.where(scores <= q33)[0].astype(np.int32),
        "medium": np.where((scores > q33) & (scores < q66))[0].astype(np.int32),
        "high": np.where(scores >= q66)[0].astype(np.int32),
    }
    desired = {
        "high": int(round(query_budget * high_fraction)),
        "medium": int(round(query_budget * medium_fraction)),
        "low": int(round(query_budget * low_fraction)),
    }
    desired["high"] += max(0, query_budget - sum(desired.values()))
    selected = []
    labels = []
    used = set()
    for bucket in ("high", "medium", "low"):
        candidates = bucket_indices[bucket]
        take = min(len(candidates), max(0, desired[bucket]))
        if take:
            chosen = rng.choice(candidates, size=take, replace=False)
            selected.append(chosen)
            labels.extend([bucket] * len(chosen))
            used.update(int(x) for x in chosen.tolist())
    deficit = query_budget - sum(len(x) for x in selected)
    if deficit > 0:
        remaining = np.asarray([i for i in range(len(scores)) if i not in used], dtype=np.int32)
        if len(remaining):
            fill_order = remaining[np.argsort(-scores[remaining])]
            chosen = fill_order[: min(deficit, len(fill_order))]
            selected.append(chosen)
            labels.extend(["fill"] * len(chosen))
    if not selected:
        raise ValueError("no query positions selected")
    selected = np.concatenate(selected, axis=0).astype(np.int32)
    labels = np.asarray(labels, dtype=str)
    order = rng.permutation(len(selected))
    return selected[order], labels[order], (q33, q66)


def add_optional_query_fields(data, source_idx):
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
            out[key] = np.asarray(data[key])[source_idx]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--q_run_root",
        type=Path,
        default=Path("runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M"),
    )
    parser.add_argument("--q_methods", type=str, default="iql,vdn,pqn_vdn")
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--max_q_policies", type=int, default=5)
    parser.add_argument("--q_action_mode", type=str, default="greedy", choices=("greedy", "softmax", "epsilon_greedy"))
    parser.add_argument("--q_temperature", type=float, default=1.0)
    parser.add_argument("--q_epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate_queries", type=int, default=40000)
    parser.add_argument("--max_queries", type=int, default=12000)
    parser.add_argument("--policies_per_query", type=int, default=14)
    parser.add_argument("--max_rows", type=int, default=160000)
    parser.add_argument("--high_fraction", type=float, default=0.4)
    parser.add_argument("--medium_fraction", type=float, default=0.4)
    parser.add_argument("--low_fraction", type=float, default=0.2)
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

    entries = load_q_pool(
        args.q_run_root,
        args.q_methods,
        args.layout,
        args.max_q_policies,
        args.q_action_mode,
        args.q_temperature,
        args.q_epsilon,
    )
    policy_ids = np.asarray([f"{entry.pool}:{entry.label}" for entry in entries], dtype=str)
    support_by_policy = {
        policy_id: np.where(source_partner_ids == policy_id)[0].astype(np.int32)
        for policy_id in policy_ids
    }
    available = np.asarray([len(support_by_policy[policy_id]) > 0 for policy_id in policy_ids])
    if not np.any(available):
        raise ValueError("none of the requested Q policies has support rows in source_dataset")
    if not np.all(available):
        print(
            f"[qfamily_grouped_dataset] skip policies without support: {policy_ids[~available].tolist()}",
            flush=True,
        )
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
            f"[qfamily_grouped_dataset] teacher {pos + 1}/{len(entries)} {policy_id}",
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
    mean_teacher = teacher_probs.mean(axis=0).astype(np.float32)
    tv_scores = mean_pairwise_tv(teacher_probs)
    argmax_disagreement = mean_argmax_disagreement(teacher_probs)

    policies_per_query = min(max(1, int(args.policies_per_query)), len(policy_ids))
    query_budget = min(int(args.max_queries), int(args.max_rows) // policies_per_query)
    query_budget = max(1, min(query_budget, candidate_count))
    selected_query_pos, selected_buckets, (q33, q66) = balanced_query_positions(
        rng,
        tv_scores,
        query_budget,
        args.high_fraction,
        args.medium_fraction,
        args.low_fraction,
    )

    query_pos_parts = []
    policy_pos_parts = []
    group_parts = []
    bucket_parts = []
    for group_id, (query_pos, bucket) in enumerate(zip(selected_query_pos, selected_buckets)):
        chosen = rng.choice(
            len(policy_ids),
            size=policies_per_query,
            replace=len(policy_ids) < policies_per_query,
        )
        query_pos_parts.append(np.full((len(chosen),), int(query_pos), dtype=np.int32))
        policy_pos_parts.append(chosen.astype(np.int32))
        group_parts.append(np.full((len(chosen),), int(group_id), dtype=np.int32))
        bucket_parts.extend([str(bucket)] * len(chosen))

    row_query_pos = np.concatenate(query_pos_parts, axis=0)
    row_policy_pos = np.concatenate(policy_pos_parts, axis=0)
    query_group_id = np.concatenate(group_parts, axis=0)
    bucket_labels = np.asarray(bucket_parts, dtype=str)
    if args.max_rows and len(row_query_pos) > args.max_rows:
        keep = np.sort(rng.choice(len(row_query_pos), size=int(args.max_rows), replace=False))
        row_query_pos = row_query_pos[keep]
        row_policy_pos = row_policy_pos[keep]
        query_group_id = query_group_id[keep]
        bucket_labels = bucket_labels[keep]

    source_idx = candidate_source_idx[row_query_pos]
    row_policy_ids = policy_ids[row_policy_pos].astype(str)
    support_idx = choose_support_indices(
        rng,
        support_by_policy,
        row_policy_ids,
        episode_ids[source_idx],
        episode_ids,
    )
    target = teacher_probs[row_policy_pos, row_query_pos].astype(np.float32)
    mean_target = mean_teacher[row_query_pos].astype(np.float32)
    residual_target_tv = 0.5 * np.abs(target - mean_target).sum(axis=-1).astype(np.float32)
    arrays = {
        "query_obs": np.asarray(data["query_obs"][source_idx], dtype=np.float16),
        "partner_obs_history": np.asarray(data["partner_obs_history"][support_idx], dtype=np.float16),
        "partner_action_history": np.asarray(data["partner_action_history"][support_idx], dtype=np.int16),
        "ego_action_history": np.asarray(data["ego_action_history"][support_idx], dtype=np.int16),
        "target_partner_probs": target,
        "mean_teacher_probs": mean_target,
        "residual_target_tv": residual_target_tv,
        "query_group_id": query_group_id.astype(np.int32),
        "query_disagreement_tv": tv_scores[row_query_pos].astype(np.float32),
        "query_argmax_disagreement": argmax_disagreement[row_query_pos].astype(np.float32),
        "source_row": source_idx.astype(np.int32),
        "support_row": support_idx.astype(np.int32),
        "ego_action": np.asarray(data["ego_action"][source_idx], dtype=np.int16),
        "episode_id": (1_000_000 + query_group_id).astype(np.int32),
        "pair_label": np.asarray(
            [
                f"q_identifiable:{policy_id}@{source_pair_labels[src]}"
                for policy_id, src in zip(row_policy_ids, source_idx)
            ],
            dtype=str,
        ),
        "ego_pool": source_ego_families[source_idx].astype(str),
        "partner_pool": np.asarray([family_from_policy_id(x) for x in row_policy_ids], dtype=str),
        "ego_family": source_ego_families[source_idx].astype(str),
        "partner_family": np.asarray([family_from_policy_id(x) for x in row_policy_ids], dtype=str),
        "ego_policy_id": source_ego_ids[source_idx].astype(str),
        "partner_policy_id": row_policy_ids.astype(str),
        "disagreement_bucket": bucket_labels,
    }
    arrays.update(add_optional_query_fields(data, source_idx))

    dataset_path = output / "qfamily_grouped_latent_decoder_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)

    bucket_rows = []
    for bucket in ["high", "medium", "low", "fill"]:
        mask = arrays["disagreement_bucket"].astype(str) == bucket
        if np.any(mask):
            bucket_rows.append({
                "bucket": bucket,
                "rows": int(np.sum(mask)),
                "groups": int(len(np.unique(arrays["query_group_id"][mask]))),
                "mean_query_disagreement_tv": float(np.mean(arrays["query_disagreement_tv"][mask])),
                "mean_residual_target_tv": float(np.mean(arrays["residual_target_tv"][mask])),
            })
    with (output / "bucket_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "rows",
                "groups",
                "mean_query_disagreement_tv",
                "mean_residual_target_tv",
            ],
        )
        writer.writeheader()
        writer.writerows(bucket_rows)

    same_episode_support = episode_ids[source_idx] == episode_ids[support_idx]
    with (output / "dataset_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Q-family grouped latent-decoder dataset\n\n")
        f.write(f"- source_dataset: `{args.source_dataset}`\n")
        f.write(f"- output_dataset: `{dataset_path}`\n")
        f.write(f"- q_methods: `{args.q_methods}`\n")
        f.write(f"- q_action_mode: `{args.q_action_mode}`\n")
        f.write(f"- loaded_policies: `{len(policy_ids)}`\n")
        f.write(f"- loaded_policy_ids: `{policy_ids.tolist()}`\n")
        f.write(f"- source_rows: `{n_rows}`\n")
        f.write(f"- candidate_queries: `{candidate_count}`\n")
        f.write(f"- selected_query_groups: `{len(np.unique(query_group_id))}`\n")
        f.write(f"- policies_per_query: `{policies_per_query}`\n")
        f.write(f"- stored_rows: `{len(arrays['ego_action'])}`\n")
        f.write(f"- tv_q33: `{q33}`\n")
        f.write(f"- tv_q66: `{q66}`\n")
        f.write(f"- mean_query_disagreement_tv: `{float(np.mean(arrays['query_disagreement_tv']))}`\n")
        f.write(f"- p50_query_disagreement_tv: `{float(np.quantile(arrays['query_disagreement_tv'], 0.5))}`\n")
        f.write(f"- p90_query_disagreement_tv: `{float(np.quantile(arrays['query_disagreement_tv'], 0.9))}`\n")
        f.write(f"- mean_residual_target_tv: `{float(np.mean(arrays['residual_target_tv']))}`\n")
        f.write(f"- same_episode_support_fraction: `{float(np.mean(same_episode_support))}`\n")
        f.write("- target: selected Q policy distribution on the shared query observation.\n")
        f.write("- mean_teacher_probs: mean Q policy distribution across all loaded Q policies for the query.\n")
        f.write("- support: selected Q policy partner obs/action history, different episode when available.\n")
    print(
        f"[qfamily_grouped_dataset] wrote {dataset_path} rows={len(arrays['ego_action'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
