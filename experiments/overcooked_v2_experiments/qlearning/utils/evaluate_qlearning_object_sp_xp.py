#!/usr/bin/env python3
"""Object-policy SP/XP evaluator for OV2 Q-learning checkpoints.

This complements evaluate_qlearning.py for RNN Q-learning methods such as
QMIX/SHAQ, whose policy state needs to be carried through the rollout.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "JaxMARL"))
sys.path.insert(0, str(ROOT / "JaxMARL" / "baselines" / "QLearning"))
sys.path.insert(0, str(ROOT / "experiments"))

from overcooked_v2_experiments.eval.policy import PolicyPairing  # noqa: E402
from overcooked_v2_experiments.qlearning.utils.evaluate_mixed_1zsc import (  # noqa: E402
    QLearningPolicy,
    evaluate_one_pairing,
    load_q_method,
)


def make_pairs(num_policies: int, mode: str) -> list[tuple[int, int]]:
    if mode == "sp":
        return [(i, i) for i in range(num_policies)]
    if mode == "xp":
        return [(i, j) for i in range(num_policies) for j in range(num_policies) if i != j]
    if mode == "cross":
        return [(i, j) for i in range(num_policies) for j in range(num_policies)]
    raise ValueError(f"unknown mode: {mode}")


def summarize(rows: Sequence[dict]) -> dict:
    rewards = np.asarray([float(row["total_reward"]) for row in rows], dtype=np.float64)
    pair_map: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        pair_map[str(row["pairing"])].append(float(row["total_reward"]))
    pair_means = np.asarray([np.mean(values) for values in pair_map.values()], dtype=np.float64)
    return {
        "mean": float(np.mean(rewards)) if rewards.size else float("nan"),
        "std_episode": float(np.std(rewards)) if rewards.size else float("nan"),
        "std_pair": float(np.std(pair_means)) if pair_means.size else float("nan"),
        "num_pairs": int(len(pair_map)),
        "num_episodes": int(len(rows)),
    }


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out_dir: Path, method: str, summaries: dict[str, dict], args: argparse.Namespace) -> None:
    rows = [{"method": method, "mode": mode, **summary} for mode, summary in summaries.items()]
    write_csv(
        out_dir / "summary.csv",
        rows,
        ["method", "mode", "mean", "std_episode", "std_pair", "num_pairs", "num_episodes"],
    )
    lines = [
        f"# Q-learning object-policy SP/XP: {method}",
        "",
        f"- Run root: `{args.run_root}`",
        f"- Layout: `{args.layout}`",
        f"- Eval seeds per pairing: `{args.num_eval_seeds}`",
        f"- Q action mode: `{args.action_mode}`",
        "",
        "| mode | mean | std_episode | std_pair | num_pairs | num_episodes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in summaries.items():
        lines.append(
            "| {mode} | {mean:.3f} | {std_episode:.3f} | {std_pair:.3f} | {num_pairs} | {num_episodes} |".format(
                mode=mode,
                **summary,
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "iql",
            "iql_double",
            "iql_dueling",
            "vdn",
            "vdn_dueling",
            "pqn_vdn",
            "pqn_wqmix",
            "pqn_soft",
            "pqn_qplex",
            "a2c",
            "coma",
            "ppo_coma",
            "qmix_cnn",
            "qplex",
            "wqmix",
            "qmix_rnn",
            "shaq_ps",
        ),
    )
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_seeds", type=int, default=100)
    parser.add_argument("--max_policies", type=int, default=None)
    parser.add_argument("--modes", default="sp,xp")
    parser.add_argument("--action_mode", choices=("greedy", "softmax", "epsilon_greedy"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()

    loaded = load_q_method(args.run_root, args.method, args.layout, args.max_policies)
    env_kwargs = dict(loaded.env_kwargs)
    env_kwargs.pop("layout", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        pairs = make_pairs(len(loaded.params), mode)
        mode_rows: list[dict] = []
        print(
            f"[qlearning-object-eval] method={args.method} mode={mode} policies={len(loaded.params)} pairs={len(pairs)} seeds={args.num_eval_seeds}",
            flush=True,
        )
        for pair_idx, (i, j) in enumerate(pairs):
            policy0 = QLearningPolicy(
                loaded.params[i],
                loaded.apply_one,
                action_mode=args.action_mode,
                temperature=args.temperature,
                epsilon=args.epsilon,
                is_rnn=loaded.is_rnn,
                hidden_size=loaded.hidden_size,
                agent_index=0,
                preprocess_flat_obs=loaded.preprocess_flat_obs,
            )
            policy1 = QLearningPolicy(
                loaded.params[j],
                loaded.apply_one,
                action_mode=args.action_mode,
                temperature=args.temperature,
                epsilon=args.epsilon,
                is_rnn=loaded.is_rnn,
                hidden_size=loaded.hidden_size,
                agent_index=1,
                preprocess_flat_obs=loaded.preprocess_flat_obs,
            )
            pair_id = f"{loaded.labels[i]}__{loaded.labels[j]}"
            pair_seed = args.seed + 1000 * i + j
            rows = evaluate_one_pairing(
                PolicyPairing(policy0, policy1),
                args.layout,
                env_kwargs,
                pair_seed,
                args.num_eval_seeds,
            )
            for annotation, reward in rows:
                mode_rows.append(
                    {
                        "method": args.method,
                        "mode": mode,
                        "pairing": pair_id,
                        "agent0": loaded.labels[i],
                        "agent1": loaded.labels[j],
                        "annotation": annotation,
                        "total_reward": float(reward),
                    }
                )
            if (pair_idx + 1) % 10 == 0 or pair_idx + 1 == len(pairs):
                print(f"[qlearning-object-eval] mode={mode} evaluated {pair_idx + 1}/{len(pairs)}", flush=True)

        write_csv(
            args.output_dir / f"reward_summary_{mode}.csv",
            mode_rows,
            ["method", "mode", "pairing", "agent0", "agent1", "annotation", "total_reward"],
        )
        summaries[mode] = summarize(mode_rows)
        raw_rows.extend(mode_rows)
        print(f"[qlearning-object-eval] mode={mode} summary={summaries[mode]}", flush=True)

    write_csv(
        args.output_dir / "reward_summary_all.csv",
        raw_rows,
        ["method", "mode", "pairing", "agent0", "agent1", "annotation", "total_reward"],
    )
    write_summary(args.output_dir, args.method, summaries, args)
    print(f"[qlearning-object-eval] wrote {args.output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
