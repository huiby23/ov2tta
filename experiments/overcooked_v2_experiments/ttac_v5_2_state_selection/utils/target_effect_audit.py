import argparse
import csv
import itertools
import os
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout_with_observations
from overcooked_v2_experiments.ttac_v5_2_state_selection.policy import PPOPolicy
from overcooked_v2_experiments.ttac_v5_2_state_selection.utils.store import load_all_checkpoints
from overcooked_v2_experiments.ttac_v5_2_state_selection.utils.surrogate_heads import load_surrogate_npz
from overcooked_v2_experiments.ttac_v5_2_state_selection.utils.agreement_heads import load_agreement_npz
from overcooked_v2_experiments.ttac_v5_2_state_selection.utils.zsc_diagnostics import (
    batched_policy_outputs,
    build_env,
    evaluate_complementarity,
    flatten_tree_first_two_dims,
    sample_indices,
)


AUDIT_MODES = {
    "base_no_update": {
        "eval_mode": "base_no_test_adapt",
        "do_update": False,
        "shuffle_actions": False,
        "use_loss": False,
    },
    "agreement": {
        "eval_mode": "ttac_true_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "advantage_weighted": {
        "eval_mode": "ttac_advantage_weighted",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "ego_advantage_weighted": {
        "eval_mode": "ttac_ego_advantage_weighted",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "ego_aw_wrong_history": {
        "eval_mode": "ttac_ego_aw_wrong_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "ego_aw_random_history": {
        "eval_mode": "ttac_ego_aw_random_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "ego_aw_delayed_history": {
        "eval_mode": "ttac_ego_aw_delayed_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "contrastive_ego_aw": {
        "eval_mode": "ttac_contrastive_ego_aw",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "contrastive_ego_aw_wrong_history": {
        "eval_mode": "ttac_contrastive_ego_aw_wrong_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "contrastive_ego_aw_random_history": {
        "eval_mode": "ttac_contrastive_ego_aw_random_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "contrastive_ego_aw_delayed_history": {
        "eval_mode": "ttac_contrastive_ego_aw_delayed_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "semantic_ego_aw": {
        "eval_mode": "ttac_semantic_ego_aw",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "semantic_ego_aw_wrong_history": {
        "eval_mode": "ttac_semantic_ego_aw_wrong_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "semantic_ego_aw_random_history": {
        "eval_mode": "ttac_semantic_ego_aw_random_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "semantic_ego_aw_delayed_history": {
        "eval_mode": "ttac_semantic_ego_aw_delayed_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "projected_confident": {
        "eval_mode": "ttac_projected_confident",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v4_support_aw": {
        "eval_mode": "ttac_v4_support_aw",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v4_gated_semantic": {
        "eval_mode": "ttac_v4_gated_semantic",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v4_gated_margin": {
        "eval_mode": "ttac_v4_gated_margin",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v4_margin_only": {
        "eval_mode": "ttac_v4_margin_only",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v4_direct_q": {
        "eval_mode": "ttac_v5_2_state_selection_q",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v4_direct_q_support": {
        "eval_mode": "ttac_v5_2_state_selection_q_support",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_support_aw": {
        "eval_mode": "ttac_v5_support_aw",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_agreement": {
        "eval_mode": "ttac_v5_agreement",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_agreement_wrong_history": {
        "eval_mode": "ttac_v5_agreement_wrong_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_agreement_random_history": {
        "eval_mode": "ttac_v5_agreement_random_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_agreement_delayed_history": {
        "eval_mode": "ttac_v5_agreement_delayed_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_agreement_support": {
        "eval_mode": "ttac_v5_agreement_support",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_agreement_support_wrong_history": {
        "eval_mode": "ttac_v5_agreement_support_wrong_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_agreement_support_random_history": {
        "eval_mode": "ttac_v5_agreement_support_random_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_agreement_support_delayed_history": {
        "eval_mode": "ttac_v5_agreement_support_delayed_history",
        "do_update": True,
        "shuffle_actions": False,
        "use_loss": True,
    },
    "v5_1_multiquery": {"eval_mode": "ttac_v5_1_multiquery", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_multiquery_wrong_history": {"eval_mode": "ttac_v5_1_multiquery_wrong_history", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_multiquery_random_history": {"eval_mode": "ttac_v5_1_multiquery_random_history", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_multiquery_delayed_history": {"eval_mode": "ttac_v5_1_multiquery_delayed_history", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_multiquery_support": {"eval_mode": "ttac_v5_1_multiquery_support", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_confident": {"eval_mode": "ttac_v5_1_confident", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_confident_wrong_history": {"eval_mode": "ttac_v5_1_confident_wrong_history", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_confident_random_history": {"eval_mode": "ttac_v5_1_confident_random_history", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_confident_delayed_history": {"eval_mode": "ttac_v5_1_confident_delayed_history", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_1_confident_support": {"eval_mode": "ttac_v5_1_confident_support", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_2_latest": {"eval_mode": "ttac_v5_2_latest", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_2_prev_query": {"eval_mode": "ttac_v5_2_prev_query", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_2_tv_gate": {"eval_mode": "ttac_v5_2_tv_gate", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_2_value_tv_gate": {"eval_mode": "ttac_v5_2_value_tv_gate", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "v5_2_change_tv_gate": {"eval_mode": "ttac_v5_2_change_tv_gate", "do_update": True, "shuffle_actions": False, "use_loss": True},
    "kl_only": {
        "eval_mode": "base_no_test_adapt",
        "do_update": False,
        "shuffle_actions": False,
        "use_loss": False,
    },
    "aw_shuffled_history": {
        "eval_mode": "ttac_advantage_weighted",
        "do_update": True,
        "shuffle_actions": True,
        "use_loss": True,
    },
    "ego_aw_shuffled_history": {
        "eval_mode": "ttac_ego_advantage_weighted",
        "do_update": True,
        "shuffle_actions": True,
        "use_loss": True,
    },
}


def apply_model_overrides(config, args):
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
    overrides = {
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
        "TTAC_V3_SEMANTIC_COEF": args.ttac_v5_2_state_selection_semantic_coef,
        "TTAC_V3_MARGIN_COEF": args.ttac_v5_2_state_selection_margin_coef,
        "TTAC_V3_MARGIN": args.ttac_v5_2_state_selection_margin,
        "TTAC_V3_RECENCY_TAU": args.ttac_v5_2_state_selection_recency_tau,
        "TTAC_V3_USE_CHANGE_GATE": None if args.ttac_v5_2_state_selection_use_change_gate is None else bool(args.ttac_v5_2_state_selection_use_change_gate),
        "TTAC_V3_CHANGE_GATE_FLOOR": args.ttac_v5_2_state_selection_change_gate_floor,
        "TTAC_V4_SURROGATE": surrogate_params,
        "TTAC_V4_DIRECT_Q_COEF": args.ttac_v5_2_state_selection_q_coef,
        "TTAC_V4_SUPPORT_COEF": args.ttac_v4_support_coef,
        "TTAC_V5_ESTIMATOR": agreement_params,
        "TTAC_V5_AGREEMENT_COEF": args.ttac_v5_agreement_coef,
        "TTAC_V5_SUPPORT_COEF": args.ttac_v5_support_coef,
        "TTAC_V5_CONF_MAX_ENTROPY": args.ttac_v5_conf_max_entropy,
        "TTAC_V5_CONF_MIN_TARGET_BASE_TV": args.ttac_v5_conf_min_target_base_tv,
        "TTAC_V5_2_TV_THRESHOLD": args.ttac_v5_2_tv_threshold,
    }
    for key, value in overrides.items():
        if value is not None:
            config["model"][key] = value


def load_run(run_dir, args):
    all_checkpoints, config = load_all_checkpoints(Path(run_dir), final_only=True)
    apply_model_overrides(config, args)
    run_keys = sorted(all_checkpoints.keys(), key=lambda name: int(name.split("_")[1]))
    params = [all_checkpoints[key]["ckpt_final"].params for key in run_keys]
    return run_keys, params, config


def make_policy(params, config, eval_mode="base_no_test_adapt", stochastic=False):
    return PPOPolicy(params, config, stochastic=stochastic, eval_mode=eval_mode)


def rollout_pair(lhs_params, rhs_params, config, env, seed, num_episodes):
    lhs = make_policy(lhs_params, config, "base_no_test_adapt", stochastic=True)
    rhs = make_policy(rhs_params, config, "base_no_test_adapt", stochastic=True)
    pairing = PolicyPairing(lhs, rhs)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    return jax.vmap(lambda key: get_rollout_with_observations(pairing, env, key))(keys)


def flatten_role_arrays(rollout, mask):
    obs0 = np.asarray(rollout.obs_seq["agent_0"])
    obs1 = np.asarray(rollout.obs_seq["agent_1"])
    done0 = np.asarray(rollout.done_seq["agent_0"])
    done1 = np.asarray(rollout.done_seq["agent_1"])
    mask_flat = mask.reshape(-1)
    obs0 = obs0.reshape((-1,) + obs0.shape[2:])[mask_flat]
    obs1 = obs1.reshape((-1,) + obs1.shape[2:])[mask_flat]
    done0 = done0.reshape(-1)[mask_flat]
    done1 = done1.reshape(-1)[mask_flat]
    flat_states = flatten_tree_first_two_dims(rollout.state_seq)
    states = jax.tree_util.tree_map(lambda x: x[mask_flat], flat_states)
    return states, obs0, obs1, done0, done1


def policy_pair_metrics(
    env,
    states,
    obs0,
    obs1,
    done0,
    done1,
    lhs_policy,
    rhs_policy,
    compatibility_sample_limit,
    seed,
):
    all_obs = np.concatenate([obs0, obs1], axis=0)
    all_done = np.concatenate([done0, done1], axis=0)
    lhs_probs, lhs_values = batched_policy_outputs(lhs_policy, all_obs, all_done)
    rhs_probs, rhs_values = batched_policy_outputs(rhs_policy, all_obs, all_done)

    agreement = float(np.mean(np.argmax(lhs_probs, axis=-1) == np.argmax(rhs_probs, axis=-1)))
    policy_tv = float(np.mean(0.5 * np.abs(lhs_probs - rhs_probs).sum(axis=-1)))
    value_shift = float(np.mean(np.abs(lhs_values - rhs_values)))

    num_states = obs0.shape[0]
    sample_idx = sample_indices(num_states, compatibility_sample_limit, seed)
    sampled_states = jax.tree_util.tree_map(lambda x: x[sample_idx], states)
    comp = evaluate_complementarity(
        env=env,
        state_batch=sampled_states,
        obs0_batch=obs0[sample_idx],
        obs1_batch=obs1[sample_idx],
        done0_batch=done0[sample_idx],
        done1_batch=done1[sample_idx],
        lhs_policy=lhs_policy,
        rhs_policy=rhs_policy,
        gamma=float(lhs_policy.config["model"]["GAMMA"]),
        seed=seed + 100_000,
        value_mode="policy_value",
    )
    return {
        "num_fixed_states": int(num_states),
        "action_agreement": agreement,
        "policy_tv": policy_tv,
        "value_shift": value_shift,
        "joint_optimal_rate": comp.joint_optimal_rate,
        "joint_regret": comp.joint_regret,
        "unilateral_regret": comp.unilateral_regret,
    }


def policy_change_metrics(base_policy, adapted_policy, obs0, obs1, done0, done1):
    all_obs = np.concatenate([obs0, obs1], axis=0)
    all_done = np.concatenate([done0, done1], axis=0)
    base_probs, _ = batched_policy_outputs(base_policy, all_obs, all_done)
    adapted_probs, _ = batched_policy_outputs(adapted_policy, all_obs, all_done)
    return {
        "adapter_action_change_rate": float(
            np.mean(np.argmax(base_probs, axis=-1) != np.argmax(adapted_probs, axis=-1))
        ),
        "adapter_policy_tv_to_base": float(
            np.mean(0.5 * np.abs(base_probs - adapted_probs).sum(axis=-1))
        ),
    }


def shuffled_actions(actions, rng):
    flat = np.asarray(actions).reshape(-1)
    if flat.shape[0] <= 1:
        return np.asarray(actions)
    perm = rng.permutation(flat.shape[0])
    return flat[perm].reshape(np.asarray(actions).shape)


def build_history_state(policy, ego_obs, ego_done, partner_obs, partner_actions, seed, prefix_steps):
    hstate = policy.init_hstate(1)
    key = jax.random.PRNGKey(seed)
    episodes, total_steps = ego_done.shape[:2]
    prefix = min(prefix_steps, total_steps)
    for ep in range(episodes):
        for t in range(prefix):
            key, subkey = jax.random.split(key)
            _, hstate = policy.compute_action(
                jnp.asarray(ego_obs[ep, t], dtype=jnp.float32),
                jnp.asarray(ego_done[ep, t]),
                hstate,
                subkey,
            )
            hstate = policy._append_history(
                hstate,
                jnp.asarray(partner_obs[ep, t], dtype=jnp.float32),
                jnp.asarray(partner_actions[ep, t], dtype=jnp.int32),
            )
    return hstate


def adapt_one_policy(
    params,
    config,
    audit_spec,
    ego_obs,
    ego_done,
    partner_obs,
    partner_actions,
    seed,
    prefix_steps,
):
    eval_mode = audit_spec["eval_mode"]
    policy = make_policy(params, config, eval_mode=eval_mode, stochastic=False)
    history_state = build_history_state(
        policy,
        ego_obs,
        ego_done,
        partner_obs,
        partner_actions,
        seed=seed,
        prefix_steps=prefix_steps,
    )

    loss_before = np.nan
    loss_after = np.nan
    if audit_spec["use_loss"]:
        use_kl = eval_mode != "ttac_no_kl"
        loss_before, _ = policy._ttac_loss(policy.params, history_state, use_kl)
        loss_before = float(loss_before)

    if not audit_spec["do_update"]:
        adapted_params = policy.params
    else:
        hstate = policy.init_hstate(1)
        key = jax.random.PRNGKey(seed + 17)
        episodes, total_steps = ego_done.shape[:2]
        prefix = min(prefix_steps, total_steps)
        for ep in range(episodes):
            for t in range(prefix):
                key, subkey = jax.random.split(key)
                _, hstate = policy.compute_action(
                    jnp.asarray(ego_obs[ep, t], dtype=jnp.float32),
                    jnp.asarray(ego_done[ep, t]),
                    hstate,
                    subkey,
                )
                hstate = policy.update_after_step(
                    hstate,
                    jnp.asarray(partner_obs[ep, t], dtype=jnp.float32),
                    jnp.asarray(partner_actions[ep, t], dtype=jnp.int32),
                    jnp.asarray(ego_done[ep, t]),
                )
        adapted_params = hstate.params

    if audit_spec["use_loss"]:
        use_kl = eval_mode != "ttac_no_kl"
        loss_after, _ = policy._ttac_loss(adapted_params, history_state, use_kl)
        loss_after = float(loss_after)

    return adapted_params, loss_before, loss_after


def prefix_arrays(rollout, role, audit_spec, rng):
    ego_id = f"agent_{role}"
    partner_id = f"agent_{1 - role}"
    ego_obs = np.asarray(rollout.obs_seq[ego_id])
    ego_done = np.asarray(rollout.done_seq[ego_id])
    partner_obs = np.asarray(rollout.obs_seq[partner_id])
    partner_actions = np.asarray(rollout.actions_seq[partner_id])
    if audit_spec["shuffle_actions"]:
        partner_actions = shuffled_actions(partner_actions, rng)
    return ego_obs, ego_done, partner_obs, partner_actions


def mean_numeric(rows, key):
    vals = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir, pair_rows, args):
    modes = []
    for row in pair_rows:
        if row["audit_mode"] not in modes:
            modes.append(row["audit_mode"])

    summary_rows = []
    for mode in modes:
        rows = [row for row in pair_rows if row["audit_mode"] == mode]
        summary_rows.append(
            {
                "audit_mode": mode,
                "pairs": len(rows),
                "loss_before": mean_numeric(rows, "objective_loss_before"),
                "loss_after": mean_numeric(rows, "objective_loss_after"),
                "delta_loss": mean_numeric(rows, "delta_objective_loss"),
                "agreement_before": mean_numeric(rows, "action_agreement_before"),
                "agreement_after": mean_numeric(rows, "action_agreement_after"),
                "delta_agreement": mean_numeric(rows, "delta_action_agreement"),
                "policy_tv_before": mean_numeric(rows, "policy_tv_before"),
                "policy_tv_after": mean_numeric(rows, "policy_tv_after"),
                "delta_policy_tv": mean_numeric(rows, "delta_policy_tv"),
                "joint_optimal_before": mean_numeric(rows, "joint_optimal_rate_before"),
                "joint_optimal_after": mean_numeric(rows, "joint_optimal_rate_after"),
                "delta_joint_optimal": mean_numeric(rows, "delta_joint_optimal_rate"),
                "joint_regret_before": mean_numeric(rows, "joint_regret_before"),
                "joint_regret_after": mean_numeric(rows, "joint_regret_after"),
                "delta_joint_regret": mean_numeric(rows, "delta_joint_regret"),
                "adapter_action_change_rate": mean_numeric(rows, "adapter_action_change_rate"),
                "adapter_policy_tv_to_base": mean_numeric(rows, "adapter_policy_tv_to_base"),
            }
        )

    write_csv(output_dir / "target_effect_summary.csv", summary_rows)

    headers = [
        "audit_mode",
        "pairs",
        "delta_loss",
        "delta_agreement",
        "delta_policy_tv",
        "delta_joint_optimal",
        "delta_joint_regret",
        "adapter_action_change_rate",
        "adapter_policy_tv_to_base",
    ]
    lines = [
        "# TTAC target-effect audit",
        "",
        f"- source_run: `{args.run_dir}`",
        f"- max_pairs: `{args.max_pairs}`",
        f"- num_episodes: `{args.num_episodes}`",
        f"- prefix_steps: `{args.prefix_steps}`",
        f"- eval_suffix_start: `{args.eval_suffix_start}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "## Reading guide",
            "- `delta_agreement > 0` means the update improves the diagnostics agreement metric on fixed states.",
            "- `delta_policy_tv < 0` means the two policies become closer on fixed states.",
            "- `delta_joint_optimal > 0` and `delta_joint_regret < 0` mean the update improves the complementarity proxy.",
            "- `adapter_action_change_rate` and `adapter_policy_tv_to_base` measure how much the adapter actually changes the policy.",
        ]
    )
    (output_dir / "target_effect_summary.md").write_text("\n".join(lines) + "\n")
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--prefix_steps", type=int, default=80)
    parser.add_argument("--eval_suffix_start", type=int, default=80)
    parser.add_argument("--max_pairs", type=int, default=12)
    parser.add_argument("--compatibility_sample_limit", type=int, default=128)
    parser.add_argument("--modes", default="base_no_update,agreement,advantage_weighted,projected_confident,kl_only,aw_shuffled_history")
    parser.add_argument("--ttac_adapter_scale", type=float, default=0.5)
    parser.add_argument("--ttac_test_hist_kl_coef", type=float, default=0.0)
    parser.add_argument("--ttac_test_ego_kl_coef", type=float, default=0.01)
    parser.add_argument("--ttac_test_cur_kl_coef", type=float, default=0.01)
    parser.add_argument("--ttac_test_lr", type=float, default=0.003)
    parser.add_argument("--ttac_test_update_steps", type=int, default=3)
    parser.add_argument("--ttac_history_len", type=int, default=50)
    parser.add_argument("--ttac_test_project_beta", type=float, default=2.0)
    parser.add_argument("--ttac_test_support_min_prob", type=float, default=0.05)
    parser.add_argument("--ttac_test_support_max_entropy", type=float, default=1.5)
    parser.add_argument("--ttac_test_advantage_power", type=float, default=1.0)
    parser.add_argument("--ttac_test_value_gate_temp", type=float)
    parser.add_argument("--ttac_test_contrast_beta", type=float, default=1.0)
    parser.add_argument("--ttac_test_contrast_floor", type=float, default=0.0)
    parser.add_argument("--ttac_test_semantic_lambda", type=float, default=0.25)
    parser.add_argument("--ttac_v5_2_state_selection_semantic_coef", type=float, default=0.25)
    parser.add_argument("--ttac_v5_2_state_selection_margin_coef", type=float, default=0.1)
    parser.add_argument("--ttac_v5_2_state_selection_margin", type=float, default=0.1)
    parser.add_argument("--ttac_v5_2_state_selection_recency_tau", type=float, default=10.0)
    parser.add_argument("--ttac_v5_2_state_selection_use_change_gate", type=int, default=1)
    parser.add_argument("--ttac_v5_2_state_selection_change_gate_floor", type=float, default=0.1)
    parser.add_argument("--ttac_v4_surrogate_path", type=str)
    parser.add_argument("--ttac_v5_2_state_selection_q_coef", type=float, default=1.0)
    parser.add_argument("--ttac_v4_support_coef", type=float, default=1.0)
    parser.add_argument("--ttac_v5_estimator_path", type=str)
    parser.add_argument("--ttac_v5_agreement_coef", type=float, default=1.0)
    parser.add_argument("--ttac_v5_support_coef", type=float, default=1.0)
    parser.add_argument("--ttac_v5_conf_max_entropy", type=float, default=1.25)
    parser.add_argument("--ttac_v5_conf_min_target_base_tv", type=float, default=0.03)
    parser.add_argument("--ttac_v5_2_tv_threshold", type=float, default=0.05)
    args = parser.parse_args()

    requested_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in requested_modes:
        if mode not in AUDIT_MODES:
            raise ValueError(f"Unknown audit mode: {mode}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_keys, params, config = load_run(args.run_dir, args)
    env = build_env(config, layout_override=args.layout)
    pair_indices = list(itertools.permutations(range(len(run_keys)), 2))[: args.max_pairs]
    pair_rows = []

    for pair_pos, (lhs_idx, rhs_idx) in enumerate(pair_indices):
        lhs_key = run_keys[lhs_idx]
        rhs_key = run_keys[rhs_idx]
        print(f"[target_audit] pair {pair_pos + 1}/{len(pair_indices)} {lhs_key} x {rhs_key}")
        rollout = rollout_pair(
            params[lhs_idx],
            params[rhs_idx],
            config,
            env,
            seed=args.seed + 10_000 + pair_pos * 1000,
            num_episodes=args.num_episodes,
        )
        total_steps = np.asarray(rollout.done_seq["agent_0"]).shape[1]
        suffix_start = min(args.eval_suffix_start, total_steps - 1)
        eval_mask = np.zeros(np.asarray(rollout.done_seq["agent_0"]).shape[:2], dtype=bool)
        eval_mask[:, suffix_start:] = True
        states, obs0, obs1, done0, done1 = flatten_role_arrays(rollout, eval_mask)

        base_lhs_policy = make_policy(params[lhs_idx], config, "base_no_test_adapt", stochastic=False)
        base_rhs_policy = make_policy(params[rhs_idx], config, "base_no_test_adapt", stochastic=False)
        metric_seed = args.seed + 20_000 + pair_pos
        base_metrics = policy_pair_metrics(
            env,
            states,
            obs0,
            obs1,
            done0,
            done1,
            base_lhs_policy,
            base_rhs_policy,
            compatibility_sample_limit=args.compatibility_sample_limit,
            seed=metric_seed,
        )

        for mode in requested_modes:
            spec = AUDIT_MODES[mode]
            rng = np.random.default_rng(args.seed + 30_000 + pair_pos)
            lhs_inputs = prefix_arrays(rollout, 0, spec, rng)
            rhs_inputs = prefix_arrays(rollout, 1, spec, rng)
            adapted_lhs_params, lhs_loss_before, lhs_loss_after = adapt_one_policy(
                params[lhs_idx],
                config,
                spec,
                *lhs_inputs,
                seed=args.seed + 40_000 + pair_pos,
                prefix_steps=args.prefix_steps,
            )
            adapted_rhs_params, rhs_loss_before, rhs_loss_after = adapt_one_policy(
                params[rhs_idx],
                config,
                spec,
                *rhs_inputs,
                seed=args.seed + 50_000 + pair_pos,
                prefix_steps=args.prefix_steps,
            )
            adapted_lhs_policy = make_policy(adapted_lhs_params, config, "base_no_test_adapt", stochastic=False)
            adapted_rhs_policy = make_policy(adapted_rhs_params, config, "base_no_test_adapt", stochastic=False)
            after_metrics = policy_pair_metrics(
                env,
                states,
                obs0,
                obs1,
                done0,
                done1,
                adapted_lhs_policy,
                adapted_rhs_policy,
                compatibility_sample_limit=args.compatibility_sample_limit,
                seed=metric_seed,
            )
            lhs_change = policy_change_metrics(base_lhs_policy, adapted_lhs_policy, obs0, obs1, done0, done1)
            rhs_change = policy_change_metrics(base_rhs_policy, adapted_rhs_policy, obs0, obs1, done0, done1)
            objective_loss_before = float(np.nanmean([lhs_loss_before, rhs_loss_before]))
            objective_loss_after = float(np.nanmean([lhs_loss_after, rhs_loss_after]))
            row = {
                "audit_mode": mode,
                "run_i": lhs_key,
                "run_j": rhs_key,
                "mean_reward_rollout": float(np.mean(np.asarray(rollout.total_reward, dtype=np.float64))),
                "num_fixed_states": after_metrics["num_fixed_states"],
                "objective_loss_before": objective_loss_before,
                "objective_loss_after": objective_loss_after,
                "delta_objective_loss": objective_loss_after - objective_loss_before,
                "action_agreement_before": base_metrics["action_agreement"],
                "action_agreement_after": after_metrics["action_agreement"],
                "delta_action_agreement": after_metrics["action_agreement"] - base_metrics["action_agreement"],
                "policy_tv_before": base_metrics["policy_tv"],
                "policy_tv_after": after_metrics["policy_tv"],
                "delta_policy_tv": after_metrics["policy_tv"] - base_metrics["policy_tv"],
                "joint_optimal_rate_before": base_metrics["joint_optimal_rate"],
                "joint_optimal_rate_after": after_metrics["joint_optimal_rate"],
                "delta_joint_optimal_rate": after_metrics["joint_optimal_rate"] - base_metrics["joint_optimal_rate"],
                "joint_regret_before": base_metrics["joint_regret"],
                "joint_regret_after": after_metrics["joint_regret"],
                "delta_joint_regret": after_metrics["joint_regret"] - base_metrics["joint_regret"],
                "unilateral_regret_before": base_metrics["unilateral_regret"],
                "unilateral_regret_after": after_metrics["unilateral_regret"],
                "delta_unilateral_regret": after_metrics["unilateral_regret"] - base_metrics["unilateral_regret"],
                "adapter_action_change_rate": 0.5
                * (lhs_change["adapter_action_change_rate"] + rhs_change["adapter_action_change_rate"]),
                "adapter_policy_tv_to_base": 0.5
                * (lhs_change["adapter_policy_tv_to_base"] + rhs_change["adapter_policy_tv_to_base"]),
            }
            pair_rows.append(row)
            print(
                "[target_audit] "
                f"{mode} {lhs_key}x{rhs_key} "
                f"d_agree={row['delta_action_agreement']:.6f} "
                f"d_tv={row['delta_policy_tv']:.6f} "
                f"d_joint_opt={row['delta_joint_optimal_rate']:.6f} "
                f"d_regret={row['delta_joint_regret']:.6f} "
                f"change={row['adapter_action_change_rate']:.6f}"
            )

    write_csv(output_dir / "target_effect_pairs.csv", pair_rows)
    summary_rows = write_summary(output_dir, pair_rows, args)
    print("[target_audit] wrote", output_dir)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
