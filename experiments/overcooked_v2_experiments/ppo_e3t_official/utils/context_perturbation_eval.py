
import argparse
import copy
import csv
import itertools
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.eval.evaluate import eval_pairing
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix
from overcooked_v2_experiments.ppo_e3t_official.policy import (
    PPOParams,
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.ppo_e3t_official.utils.store import load_all_checkpoints
from overcooked_v2_experiments.utils.utils import mini_batch_pmap


def configure_mode(config, mode):
    cfg = copy.deepcopy(config)
    model = cfg["model"] if "model" in cfg else cfg
    mode = mode.lower()
    model["E3T_HISTORY_PERTURBATION"] = "none"

    if mode == "normal":
        pass
    elif mode in ("actor_constant", "constant_actor"):
        model["E3T_ACTOR_CONDITION"] = "constant"
    elif mode in ("history_stay", "stay_history"):
        model["E3T_HISTORY_PERTURBATION"] = "stay"
    elif mode in ("history_shift", "shift_history", "wrong_history"):
        model["E3T_HISTORY_PERTURBATION"] = "shift"
    elif mode in ("history_freeze", "freeze_history"):
        model["E3T_HISTORY_PERTURBATION"] = "freeze"
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    return cfg


def parse_mode_summary(rows):
    per_pair = defaultdict(list)
    for row in rows:
        per_pair[row["policy_labels"]].append(float(row["total_reward"]))

    sp_values = []
    xp_values = []
    for label, rewards in per_pair.items():
        lhs, rhs = label.replace("cross-", "").split("_")
        value = float(np.mean(rewards))
        if lhs == rhs:
            sp_values.append(value)
        else:
            xp_values.append(value)

    return {
        "sp_mean": float(np.mean(sp_values)) if sp_values else float("nan"),
        "xp_mean": float(np.mean(xp_values)) if xp_values else float("nan"),
        "sp_pair_std": float(np.std(sp_values)) if sp_values else float("nan"),
        "xp_pair_std": float(np.std(xp_values)) if xp_values else float("nan"),
        "num_sp_pairs": len(sp_values),
        "num_xp_pairs": len(xp_values),
    }


def evaluate_cross_mode(run_base_dir, key, mode, num_seeds, layout_override=None):
    all_params, config = load_all_checkpoints(run_base_dir, final_only=True)
    mode_config = configure_mode(config, mode)

    initial_env_kwargs = copy.deepcopy(mode_config["env"]["ENV_KWARGS"])
    if layout_override is not None:
        initial_env_kwargs["layout"] = layout_override
    env = OvercookedV2(**initial_env_kwargs)
    num_actors = env.num_agents
    if num_actors != 2:
        raise ValueError("context perturbation eval currently expects two agents")

    run_keys = sorted(all_params.keys(), key=lambda name: int(name.split("_")[1]))
    num_runs = len(run_keys)
    run_combinations = list(itertools.permutations(range(num_runs), num_actors))
    run_combinations += [[i] * num_actors for i in range(num_runs)]

    policy_params = [all_params[run_key]["ckpt_final"] for run_key in run_keys]
    cross_combinations = {}
    for combo in run_combinations:
        run_ids = [run_keys[i].replace("run_", "") for i in combo]
        label = "cross-" + "_".join(run_ids)
        cross_combinations[label] = PolicyPairing(*[policy_params[i] for i in combo])

    pairings, treedef = jax.tree_util.tree_flatten(
        cross_combinations, is_leaf=lambda x: type(x) is PolicyPairing
    )
    pairings = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *pairings)

    env_kwargs_no_layout = copy.deepcopy(initial_env_kwargs)
    layout_name = env_kwargs_no_layout.pop("layout")

    def _eval_pairing(pairing):
        policy_pairing = policy_checkoints_to_policy_pairing(pairing, mode_config)
        return eval_pairing(
            policy_pairing,
            layout_name,
            key,
            env_kwargs=env_kwargs_no_layout,
            num_seeds=num_seeds,
            all_recipes=False,
            no_viz=True,
        )

    num_devices = max(1, len(jax.devices("gpu")))
    if len(run_combinations) % num_devices != 0:
        num_devices = 1
    evaluated = mini_batch_pmap(_eval_pairing, num_devices)(pairings)

    num_annotations = jax.tree_util.tree_leaves(evaluated)[0].shape[0]
    evaluated = [
        jax.tree_util.tree_map(lambda x: x[i], evaluated)
        for i in range(num_annotations)
    ]
    all_results = jax.tree_util.tree_unflatten(treedef, evaluated)

    rows = []
    for pair_label, seed_results in all_results.items():
        for annotation, viz in seed_results.items():
            rows.append(
                {
                    "mode": mode,
                    "policy_labels": pair_label,
                    "annotation": annotation,
                    "total_reward": float(viz.total_reward),
                }
            )
    return rows, parse_mode_summary(rows)


def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--num_eval_seeds", type=int, default=500)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["normal", "actor_constant", "history_stay", "history_shift", "history_freeze"],
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_rows = []
    key = jax.random.PRNGKey(args.eval_seed)
    for index, mode in enumerate(args.modes):
        mode_key = jax.random.fold_in(key, index)
        print(f"[e3t-perturb] method={args.method_name} mode={mode} seeds={args.num_eval_seeds}", flush=True)
        rows, summary = evaluate_cross_mode(
            run_base_dir=run_dir,
            key=mode_key,
            mode=mode,
            num_seeds=args.num_eval_seeds,
            layout_override=args.layout,
        )
        all_rows.extend(rows)
        summary_rows.append({"method": args.method_name, "mode": mode, **summary})

        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_csv = mode_dir / "reward_summary_cross.csv"
        write_rows(
            mode_csv,
            rows,
            ["mode", "policy_labels", "annotation", "total_reward"],
        )
        try:
            visualize_cross_play_matrix(mode_csv)
        except Exception as exc:
            print(f"[e3t-perturb] plot failed for {mode}: {exc}", flush=True)

    write_rows(
        output_dir / "perturbation_rewards.csv",
        all_rows,
        ["mode", "policy_labels", "annotation", "total_reward"],
    )
    write_rows(
        output_dir / "perturbation_summary.csv",
        summary_rows,
        [
            "method",
            "mode",
            "sp_mean",
            "xp_mean",
            "sp_pair_std",
            "xp_pair_std",
            "num_sp_pairs",
            "num_xp_pairs",
        ],
    )

    normal = next((r for r in summary_rows if r["mode"] == "normal"), None)
    report = [
        "# E3T Context Perturbation Diagnostics",
        "",
        f"- method: `{args.method_name}`",
        f"- run_dir: `{run_dir}`",
        f"- eval_seed: `{args.eval_seed}`",
        f"- num_eval_seeds: `{args.num_eval_seeds}`",
        f"- generated_at: `{datetime.now().replace(microsecond=0).isoformat()}`",
        "",
        "| Mode | SP | XP | Delta XP vs normal |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        delta = row["xp_mean"] - normal["xp_mean"] if normal else float("nan")
        report.append(
            "| {mode} | {sp:.4f} | {xp:.4f} | {delta:+.4f} |".format(mode=row["mode"], sp=row["sp_mean"], xp=row["xp_mean"], delta=delta)
        )
    report.append("")
    report.append(
        "`actor_constant` removes the predicted partner-action condition from the actor; "
        "history perturbation modes keep the actor path intact but corrupt/freeze the partner-action history used by the E3T context encoder."
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    print("[e3t-perturb] wrote {}".format(output_dir / "report.md"), flush=True)


if __name__ == "__main__":
    main()
