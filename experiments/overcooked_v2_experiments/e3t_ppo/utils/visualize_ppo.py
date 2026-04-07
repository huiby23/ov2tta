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

from overcooked_v2_experiments.e3t_ppo.policy import (
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.e3t_ppo.utils.store import load_all_checkpoints
from overcooked_v2_experiments.eval.evaluate import eval_pairing
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix
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
):
    if cross and not final_only:
        raise ValueError("Cannot run cross play with all checkpoints")

    all_params, config = load_all_checkpoints(run_base_dir, final_only=final_only)

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

        policy_pairings = [
            all_params[run_keys[i]]["ckpt_final"] for i in range(num_runs)
        ]
        cross_combinations = {}
        for run_combination in run_combinations:
            run_ids = [run_keys[i].replace("run_", "") for i in run_combination]
            run_combination_key = "cross-" + "_".join(run_ids)
            cross_combinations[run_combination_key] = PolicyPairing(
                *[policy_pairings[i] for i in run_combination]
            )
        all_params = {"cross": cross_combinations}
    else:
        all_params = jax.tree_util.tree_map(
            lambda x: PolicyPairing.from_single_policy(x, num_actors),
            all_params,
            is_leaf=lambda x: type(x) is not dict and hasattr(x, "params"),
        )

    policy_pairings, treedef = jax.tree_util.tree_flatten(
        all_params, is_leaf=lambda x: type(x) is PolicyPairing
    )
    policy_pairings = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *policy_pairings)

    def _policy_viz(pairing):
        env_kwargs_no_layout = copy.deepcopy(env_kwargs)
        layout_name = env_kwargs_no_layout.pop("layout")
        pairing = policy_checkoints_to_policy_pairing(pairing, config)
        return eval_pairing(
            pairing,
            layout_name,
            key,
            env_kwargs=env_kwargs_no_layout,
            num_seeds=num_seeds,
            all_recipes=num_seeds is None,
            no_viz=no_viz,
        )

    num_devices = max(len(jax.devices("gpu")), 1)
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
            print(f"{labels[0]}: {first_level}, {labels[1]}: {second_level}")
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
                print(f"\t{annotation}:\t{total_reward}")
            reward_mean = checkpoint_sum / len(second_level_runs)
            print(f"\tMean reward:\t{reward_mean}")

    summery_name = "reward_summary_cross.csv" if cross else "reward_summary_sp.csv"
    summery_file = run_base_dir / summery_name
    with open(summery_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([labels[0], labels[1], "annotation", "total_reward"])
        writer.writerows(rows)

    print(f"Summary written to {summery_file}")
    if cross:
        visualize_cross_play_matrix(summery_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int)
    parser.add_argument("--all_ckpt", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--pairing_policy", type=int)
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
    modes = [m for m, v in viz_mode.items() if v]

    extra_env_kwargs = {}
    if args.no_reset:
        extra_env_kwargs["random_reset"] = False
        extra_env_kwargs["op_ingredient_permutations"] = False

    for mode in modes:
        fo = final_only or (mode == "cross")
        visualize_ppo_policy(
            Path(directory),
            key_sp if mode == "sp" else key_cross,
            num_seeds=num_seeds,
            final_only=fo,
            cross=mode == "cross",
            no_viz=args.no_viz,
            extra_env_kwargs=extra_env_kwargs,
            pairing_policy=args.pairing_policy,
        )
