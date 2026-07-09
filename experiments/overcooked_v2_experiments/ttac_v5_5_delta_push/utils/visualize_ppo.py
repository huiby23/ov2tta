import argparse
import sys
import os
import itertools
import jax.numpy as jnp
import jax
import copy
from datetime import datetime
from pathlib import Path
import chex
import imageio
import csv
import numpy as np
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2


DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.ttac_v5_5_delta_push.policy import (
    PPOParams,
    policy_checkoints_to_functional_policy_pairing,
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.ttac_v5_5_delta_push.utils.store import (
    load_all_checkpoints,
)
from overcooked_v2_experiments.ttac_v5_5_delta_push.utils.surrogate_heads import load_surrogate_npz
from overcooked_v2_experiments.ttac_v5_5_delta_push.utils.agreement_heads import load_agreement_npz
from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix
from overcooked_v2_experiments.utils.utils import (
    mini_batch_pmap,
    scanned_mini_batch_map,
)
from overcooked_v2_experiments.eval.evaluate import eval_pairing
from overcooked_v2_experiments.eval.policy import PolicyPairing


def load_reward_scorer_npz(path):
    data = np.load(Path(path), allow_pickle=False)
    return {key: jnp.asarray(data[key]) for key in data.files}


def visualize_ppo_policy(
    run_base_dir,
    key,
    final_only=True,
    extra_env_kwargs={},
    num_seeds=None,
    cross=False,
    no_viz=False,
    pairing_policy=None,
    greedy=False,
    ttac_mode="base_no_test_adapt",
    output_tag=None,
    eval_batches=1,
    model_overrides=None,
    online_eval_pairings_per_chunk=9,
    pairing_start=0,
    max_pairings=None,
    functional_eval=False,
):
    if cross and not final_only:
        raise ValueError("Cannot run cross play with all checkpoints")

    all_params, config = load_all_checkpoints(run_base_dir, final_only=final_only)
    if model_overrides:
        for override_key, value in model_overrides.items():
            if value is not None:
                config["model"][override_key] = value

    initial_env_kwargs = copy.deepcopy(config["env"]["ENV_KWARGS"])
    env_kwargs = initial_env_kwargs | extra_env_kwargs
    env = OvercookedV2(**env_kwargs)

    num_actors = env.num_agents

    run_keys = list(all_params.keys())

    # restructure if cross play, layout 1. "cross", 2. "run_combinations"
    if cross:
        num_runs = len(run_keys)

        run_combinations = itertools.permutations(range(num_runs), num_actors)
        run_combinations = list(run_combinations)
        # add self play
        run_combinations += [[i] * num_actors for i in range(num_runs)]

        if pairing_policy is not None:
            run_combinations = [
                [pairing_policy, i] for i in range(num_runs) if i != pairing_policy
            ]
            run_combinations += [
                [i, pairing_policy] for i in range(num_runs) if i != pairing_policy
            ]

        pairing_start = max(0, int(pairing_start))
        if pairing_start > 0:
            run_combinations = run_combinations[pairing_start:]

        if max_pairings is not None:
            max_pairings = int(max_pairings)
            if max_pairings > 0:
                run_combinations = run_combinations[:max_pairings]
        print("Run combinations: ", run_combinations)

        policy_pairings = [
            all_params[run_keys[i]]["ckpt_final"] for i in range(num_runs)
        ]

        cross_combinations = {}
        for run_combination in run_combinations:
            run_combination = list(run_combination)

            run_ids = [run_keys[i].replace("run_", "") for i in run_combination]
            run_combination_key = "cross-" + "_".join(run_ids)
            policy_combination = PolicyPairing(
                *[policy_pairings[i] for i in run_combination]
            )

            # policy_combination = jax.tree_util.tree_map(
            #     lambda *v: jnp.stack(v), *policies
            # )

            cross_combinations[run_combination_key] = policy_combination

        all_params = {"cross": cross_combinations}
    else:
        # all_params = jax.tree_util.tree_map(
        #     lambda x: jnp.repeat(x[jnp.newaxis, :], num_actors, axis=0),
        #     all_params,
        # )
        all_params = jax.tree_util.tree_map(
            lambda x: PolicyPairing.from_single_policy(x, num_actors),
            all_params,
            is_leaf=lambda x: type(x) is PPOParams,
        )

    # print("structure", jax.tree_util.tree_structure(all_params))

    policy_pairings, treedef = jax.tree_util.tree_flatten(
        all_params, is_leaf=lambda x: type(x) is PolicyPairing
    )

    policy_pairings = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *policy_pairings)

    # print("Vals: ", vals)

    def _policy_viz(pairing):
        env_kwargs_no_layout = copy.deepcopy(env_kwargs)
        layout_name = env_kwargs_no_layout.pop("layout")

        if functional_eval:
            pairing = policy_checkoints_to_functional_policy_pairing(
                pairing, config, stochastic=not greedy, eval_mode=ttac_mode
            )
        else:
            pairing = policy_checkoints_to_policy_pairing(
                pairing, config, stochastic=not greedy, eval_mode=ttac_mode
            )

        return eval_pairing(
            pairing,
            layout_name,
            key,
            env_kwargs=env_kwargs_no_layout,
            num_seeds=num_seeds,
            all_recipes=num_seeds is None,
            no_viz=no_viz,
        )

    # policy_params = jax.vmap(_policy_viz)(policy_params)
    try:
        num_devices = len(jax.devices("gpu"))
    except RuntimeError:
        num_devices = len(jax.devices())
    num_pairings = jax.tree_util.tree_leaves(policy_pairings)[0].shape[0]

    def _resolve_eval_batches(num_items, requested_batches):
        requested_batches = max(1, min(int(requested_batches), int(num_items)))
        while requested_batches > 1 and num_items % requested_batches != 0:
            requested_batches -= 1
        return requested_batches

    eval_batches = _resolve_eval_batches(num_pairings, eval_batches)
    print(f"Evaluation pairings: {num_pairings}, eval_batches: {eval_batches}")
    if eval_batches > 1:
        # Match the original TTAC evaluation path. This is fast when SP and cross
        # are launched as separate Python processes; the old OOM came from using
        # --all, which kept both modes in one process.
        policy_pairings = scanned_mini_batch_map(
            _policy_viz, eval_batches, use_pmap=False
        )(policy_pairings)
    else:
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
    if output_tag:
        stem, suffix = os.path.splitext(summery_name)
        summery_name = f"{stem}_{output_tag}{suffix}"
    summery_file = run_base_dir / summery_name
    with open(summery_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = [labels[0], labels[1], "annotation", "total_reward"]

        writer.writerow(fieldnames)

        for row in rows:
            writer.writerow(row)

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
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax actions during evaluation instead of sampling from the policy.",
    )
    parser.add_argument("--ttac_mode", type=str, default="base_no_test_adapt")
    parser.add_argument("--output_tag", type=str)
    parser.add_argument("--eval_batches", type=int, default=1)
    parser.add_argument("--ttac_adapter_scale", type=float)
    parser.add_argument("--ttac_test_hist_kl_coef", type=float)
    parser.add_argument("--ttac_test_ego_kl_coef", type=float)
    parser.add_argument("--ttac_test_cur_kl_coef", type=float)
    parser.add_argument("--ttac_test_lr", type=float)
    parser.add_argument("--ttac_test_update_steps", type=int)
    parser.add_argument("--ttac_history_len", type=int)
    parser.add_argument("--ttac_test_project_beta", type=float)
    parser.add_argument("--ttac_test_support_min_prob", type=float)
    parser.add_argument("--ttac_test_support_max_entropy", type=float)
    parser.add_argument("--ttac_test_advantage_power", type=float)
    parser.add_argument("--ttac_test_value_gate_temp", type=float)
    parser.add_argument("--ttac_test_contrast_beta", type=float)
    parser.add_argument("--ttac_test_contrast_floor", type=float)
    parser.add_argument("--ttac_test_semantic_lambda", type=float)
    parser.add_argument("--ttac_v4_direct_semantic_coef", type=float)
    parser.add_argument("--ttac_v4_direct_margin_coef", type=float)
    parser.add_argument("--ttac_v4_direct_margin", type=float)
    parser.add_argument("--ttac_v4_direct_recency_tau", type=float)
    parser.add_argument("--ttac_v4_direct_use_change_gate", type=int)
    parser.add_argument("--ttac_v4_direct_change_gate_floor", type=float)
    parser.add_argument("--ttac_v4_surrogate_path", type=str)
    parser.add_argument("--ttac_v4_direct_q_coef", type=float)
    parser.add_argument("--ttac_v4_support_coef", type=float)
    parser.add_argument("--ttac_v5_estimator_path", type=str)
    parser.add_argument("--ttac_v5_agreement_coef", type=float)
    parser.add_argument("--ttac_v5_support_coef", type=float)
    parser.add_argument("--ttac_v5_conf_max_entropy", type=float)
    parser.add_argument("--ttac_v5_conf_min_target_base_tv", type=float)
    parser.add_argument("--ttac_v5_2_tv_threshold", type=float)
    parser.add_argument("--ttac_v5_4_delta_margin", type=float)
    parser.add_argument("--ttac_v5_4_delta_scale", type=float)
    parser.add_argument("--ttac_v5_4_delta_mass_threshold", type=float)
    parser.add_argument("--ttac_v5_4_tv_weight_floor", type=float)
    parser.add_argument("--ttac_v5_4_tv_weight_scale", type=float)
    parser.add_argument("--ttac_v5_4_value_weight_floor", type=float)
    parser.add_argument("--ttac_v5_4_value_weight_scale", type=float)
    parser.add_argument("--ttac_v5_4_value_weight_temp", type=float)
    parser.add_argument("--ttac_v5_4_change_amp", type=float)
    parser.add_argument("--ttac_v5_4_sticky_action", type=int)
    parser.add_argument("--ttac_v5_4_sticky_suppress", type=float)
    parser.add_argument("--ttac_v5_4_stay_action", type=int)
    parser.add_argument("--ttac_v5_4_anti_stay_coef", type=float)
    parser.add_argument("--ttac_v5_4_reward_scorer_path", type=str)
    parser.add_argument("--ttac_v5_4_reward_weight_floor", type=float)
    parser.add_argument("--ttac_v5_4_reward_weight_scale", type=float)
    parser.add_argument("--ttac_v5_4_reward_center_amp", type=float)
    parser.add_argument("--ttac_v5_4_reward_center_min", type=float)
    parser.add_argument("--ttac_v5_4_reward_center_max", type=float)
    parser.add_argument("--online_eval_pairings_per_chunk", type=int, default=9)
    parser.add_argument("--pairing_start", type=int, default=0)
    parser.add_argument("--max_pairings", type=int)
    parser.add_argument(
        "--functional_eval",
        action="store_true",
        help="Use the functional TTAC v5.2 evaluator backend instead of Policy objects.",
    )

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

    surrogate_params = (
        load_surrogate_npz(args.ttac_v4_surrogate_path)
        if args.ttac_v4_surrogate_path
        else None
    )
    agreement_params = (
        load_agreement_npz(args.ttac_v5_estimator_path)
        if args.ttac_v5_estimator_path
        else None
    )
    reward_scorer_params = (
        load_reward_scorer_npz(args.ttac_v5_4_reward_scorer_path)
        if args.ttac_v5_4_reward_scorer_path
        else None
    )

    model_overrides = {
        "TTAC_ADAPTER_SCALE": args.ttac_adapter_scale,
        "TTAC_TEST_HIST_KL_COEF": args.ttac_test_hist_kl_coef,
        "TTAC_TEST_EGO_KL_COEF": args.ttac_test_ego_kl_coef,
        "TTAC_TEST_CUR_KL_COEF": args.ttac_test_cur_kl_coef,
        "TTAC_TEST_LR": args.ttac_test_lr,
        "TTAC_TEST_UPDATE_STEPS": args.ttac_test_update_steps,
        "TTAC_HISTORY_LEN": args.ttac_history_len,
        "TTAC_TEST_PROJECT_BETA": args.ttac_test_project_beta,
        "TTAC_TEST_SUPPORT_MIN_PROB": args.ttac_test_support_min_prob,
        "TTAC_TEST_SUPPORT_MAX_ENTROPY": args.ttac_test_support_max_entropy,
        "TTAC_TEST_ADVANTAGE_POWER": args.ttac_test_advantage_power,
        "TTAC_TEST_VALUE_GATE_TEMP": args.ttac_test_value_gate_temp,
        "TTAC_TEST_CONTRAST_BETA": args.ttac_test_contrast_beta,
        "TTAC_TEST_CONTRAST_FLOOR": args.ttac_test_contrast_floor,
        "TTAC_TEST_SEMANTIC_LAMBDA": args.ttac_test_semantic_lambda,
        "TTAC_V3_SEMANTIC_COEF": args.ttac_v4_direct_semantic_coef,
        "TTAC_V3_MARGIN_COEF": args.ttac_v4_direct_margin_coef,
        "TTAC_V3_MARGIN": args.ttac_v4_direct_margin,
        "TTAC_V3_RECENCY_TAU": args.ttac_v4_direct_recency_tau,
        "TTAC_V3_USE_CHANGE_GATE": None if args.ttac_v4_direct_use_change_gate is None else bool(args.ttac_v4_direct_use_change_gate),
        "TTAC_V3_CHANGE_GATE_FLOOR": args.ttac_v4_direct_change_gate_floor,
        "TTAC_V4_SURROGATE": surrogate_params,
        "TTAC_V4_DIRECT_Q_COEF": args.ttac_v4_direct_q_coef,
        "TTAC_V4_SUPPORT_COEF": args.ttac_v4_support_coef,
        "TTAC_V5_ESTIMATOR": agreement_params,
        "TTAC_V5_AGREEMENT_COEF": args.ttac_v5_agreement_coef,
        "TTAC_V5_SUPPORT_COEF": args.ttac_v5_support_coef,
        "TTAC_V5_CONF_MAX_ENTROPY": args.ttac_v5_conf_max_entropy,
        "TTAC_V5_CONF_MIN_TARGET_BASE_TV": args.ttac_v5_conf_min_target_base_tv,
        "TTAC_V5_2_TV_THRESHOLD": args.ttac_v5_2_tv_threshold,
        "TTAC_V5_4_DELTA_MARGIN": args.ttac_v5_4_delta_margin,
        "TTAC_V5_4_DELTA_SCALE": args.ttac_v5_4_delta_scale,
        "TTAC_V5_4_DELTA_MASS_THRESHOLD": args.ttac_v5_4_delta_mass_threshold,
        "TTAC_V5_4_TV_WEIGHT_FLOOR": args.ttac_v5_4_tv_weight_floor,
        "TTAC_V5_4_TV_WEIGHT_SCALE": args.ttac_v5_4_tv_weight_scale,
        "TTAC_V5_4_VALUE_WEIGHT_FLOOR": args.ttac_v5_4_value_weight_floor,
        "TTAC_V5_4_VALUE_WEIGHT_SCALE": args.ttac_v5_4_value_weight_scale,
        "TTAC_V5_4_VALUE_WEIGHT_TEMP": args.ttac_v5_4_value_weight_temp,
        "TTAC_V5_4_CHANGE_AMP": args.ttac_v5_4_change_amp,
        "TTAC_V5_4_STICKY_ACTION": args.ttac_v5_4_sticky_action,
        "TTAC_V5_4_STICKY_SUPPRESS": args.ttac_v5_4_sticky_suppress,
        "TTAC_V5_4_STAY_ACTION": args.ttac_v5_4_stay_action,
        "TTAC_V5_4_ANTI_STAY_COEF": args.ttac_v5_4_anti_stay_coef,
        "TTAC_V5_4_REWARD_SCORER": reward_scorer_params,
        "TTAC_V5_4_REWARD_WEIGHT_FLOOR": args.ttac_v5_4_reward_weight_floor,
        "TTAC_V5_4_REWARD_WEIGHT_SCALE": args.ttac_v5_4_reward_weight_scale,
        "TTAC_V5_4_REWARD_CENTER_AMP": args.ttac_v5_4_reward_center_amp,
        "TTAC_V5_4_REWARD_CENTER_MIN": args.ttac_v5_4_reward_center_min,
        "TTAC_V5_4_REWARD_CENTER_MAX": args.ttac_v5_4_reward_center_max,
    }

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
            greedy=args.greedy,
            ttac_mode=args.ttac_mode,
            output_tag=args.output_tag,
            eval_batches=args.eval_batches,
            model_overrides=model_overrides,
            online_eval_pairings_per_chunk=args.online_eval_pairings_per_chunk,
            pairing_start=args.pairing_start,
            max_pairings=args.max_pairings,
            functional_eval=args.functional_eval,
        )
