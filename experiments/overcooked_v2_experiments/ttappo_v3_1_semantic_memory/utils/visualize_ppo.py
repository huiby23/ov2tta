import argparse
import copy
import csv
import itertools
import os
import sys
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2


DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.eval.evaluate import eval_pairing
from overcooked_v2_experiments.eval.policy import FunctionalPolicyPairing
from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix
from overcooked_v2_experiments.ttappo_v3_1_semantic_memory.policy import (
    EVAL_MODES,
    PPOParams,
    policy_checkoints_to_functional_policy_pairing,
)
from overcooked_v2_experiments.ttappo_v3_1_semantic_memory.utils.store import load_all_checkpoints
from overcooked_v2_experiments.utils.utils import mini_batch_pmap


def visualize_ppo_policy(
    run_base_dir,
    key,
    final_only=True,
    extra_env_kwargs={},
    num_seeds=None,
    cross=False,
    no_viz=False,
    pairing_policy=None,
    ttt_mode="state_adapt",
    output_tag=None,
    ce_threshold_override=None,
):
    if cross and not final_only:
        raise ValueError("Cannot run cross play with all checkpoints")

    all_params, config = load_all_checkpoints(run_base_dir, final_only=final_only)
    config = copy.deepcopy(config)
    if ce_threshold_override is not None:
        config["model"]["STATE_ADAPT_CE_THRESHOLD"] = float(ce_threshold_override)
    eval_mode = ttt_mode
    adaptive_index = None if ttt_mode == "memory_off" else 0

    initial_env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
    env_kwargs = initial_env_kwargs | extra_env_kwargs
    env = OvercookedV2(**env_kwargs)

    num_actors = env.num_agents
    run_keys = list(all_params.keys())

    if cross:
        num_runs = len(run_keys)
        run_combinations = list(itertools.permutations(range(num_runs), num_actors))
        run_combinations += [[i] * num_actors for i in range(num_runs)]

        if pairing_policy is not None:
            run_combinations = [
                [pairing_policy, i] for i in range(num_runs) if i != pairing_policy
            ]
            run_combinations += [
                [i, pairing_policy] for i in range(num_runs) if i != pairing_policy
            ]

        policy_checkpoints = [
            all_params[run_keys[i]]["ckpt_final"] for i in range(num_runs)
        ]
        cross_combinations = {}
        for run_combination in run_combinations:
            run_combination = list(run_combination)
            run_ids = [run_keys[i].replace("run_", "") for i in run_combination]
            run_combination_key = "cross-" + "_".join(run_ids)
            cross_combinations[run_combination_key] = (
                policy_checkoints_to_functional_policy_pairing(
                    [policy_checkpoints[i] for i in run_combination],
                    config,
                    adaptive_index=adaptive_index,
                    eval_mode=eval_mode,
                )
            )
        all_params = {"cross": cross_combinations}
    else:
        all_params = jax.tree_util.tree_map(
            lambda x: policy_checkoints_to_functional_policy_pairing(
                [x for _ in range(num_actors)],
                config,
                adaptive_index=adaptive_index,
                eval_mode=eval_mode,
            ),
            all_params,
            is_leaf=lambda x: type(x) is PPOParams,
        )

    policy_pairings, treedef = jax.tree_util.tree_flatten(
        all_params,
        is_leaf=lambda x: type(x) is FunctionalPolicyPairing,
    )
    policy_pairings = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *policy_pairings)

    def _policy_viz(pairing):
        env_kwargs_no_layout = copy.deepcopy(env_kwargs)
        layout_name = env_kwargs_no_layout.pop("layout")
        return eval_pairing(
            pairing,
            layout_name,
            key,
            env_kwargs=env_kwargs_no_layout,
            num_seeds=num_seeds,
            all_recipes=num_seeds is None,
            no_viz=no_viz,
        )

    num_devices = len(jax.devices("gpu")) or 1
    policy_pairings = mini_batch_pmap(_policy_viz, num_devices)(policy_pairings)

    num_annotations = jax.tree_util.tree_leaves(policy_pairings)[0].shape[0]
    policy_pairings = [
        jax.tree_util.tree_map(lambda x: x[i], policy_pairings)
        for i in range(num_annotations)
    ]
    all_params = jax.tree_util.tree_unflatten(treedef, policy_pairings)

    labels = ["run", "checkpoint"]
    if cross:
        labels[1] = "policy_labels"

    rows = []
    for first_level, first_level_runs in all_params.items():
        for second_level, second_level_runs in first_level_runs.items():
            checkpoint_sum = 0.0
            for annotation, viz in second_level_runs.items():
                frame_seq = viz.frame_seq
                total_reward = viz.total_reward

                if not no_viz:
                    viz_dir = run_base_dir / first_level / second_level
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_filename = viz_dir / f"{annotation}.gif"
                    imageio.mimsave(viz_filename, frame_seq, "GIF", duration=0.5)

                checkpoint_sum += total_reward
                rows.append([first_level, second_level, annotation, total_reward])
            reward_mean = checkpoint_sum / len(second_level_runs)
            print(f"{labels[0]}={first_level} {labels[1]}={second_level} mean_reward={reward_mean}")

    summary_name = "reward_summary_cross.csv" if cross else "reward_summary_sp.csv"
    if output_tag:
        stem, suffix = os.path.splitext(summary_name)
        summary_name = f"{stem}_{output_tag}{suffix}"
    summary_file = run_base_dir / summary_name
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([labels[0], labels[1], "annotation", "total_reward"])
        writer.writerows(rows)

    print(f"Summary written to {summary_file}")
    if cross:
        visualize_cross_play_matrix(summary_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int)
    parser.add_argument("--all_ckpt", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--pairing_policy", type=int)
    parser.add_argument(
        "--ttt_mode",
        type=str,
        default="state_adapt",
        choices=list(EVAL_MODES),
    )
    parser.add_argument("--output_tag", type=str)
    parser.add_argument("--ce_threshold_override", type=float)
    args = parser.parse_args()

    directory = args.d
    num_seeds = args.num_seeds
    final_only = not args.all_ckpt
    cross = args.cross

    key = jax.random.PRNGKey(args.seed)
    key_sp, key_cross = jax.random.split(key, 2)

    viz_mode = {
        "sp": (not cross) or args.all,
        "cross": cross or args.all,
    }

    if viz_mode["sp"]:
        visualize_ppo_policy(
            Path(directory),
            key_sp,
            final_only=final_only,
            num_seeds=num_seeds,
            cross=False,
            no_viz=args.no_viz,
            ttt_mode=args.ttt_mode,
            output_tag=args.output_tag,
            ce_threshold_override=args.ce_threshold_override,
        )
    if viz_mode["cross"]:
        visualize_ppo_policy(
            Path(directory),
            key_cross,
            final_only=final_only,
            num_seeds=num_seeds,
            cross=True,
            no_viz=args.no_viz,
            pairing_policy=args.pairing_policy,
            ttt_mode=args.ttt_mode,
            output_tag=args.output_tag,
            ce_threshold_override=args.ce_threshold_override,
        )
