from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(DIR).resolve().parents[4]
sys.path.insert(0, str(ROOT / "JaxMARL"))
sys.path.insert(0, str(ROOT / "JaxMARL" / "baselines" / "QLearning"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DIR))))

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2  # noqa: E402
from overcooked_v2_experiments.eval.policy import PolicyPairing  # noqa: E402
from overcooked_v2_experiments.eval.rollout import get_rollout_with_observations  # noqa: E402
from overcooked_v2_experiments.qlearning.utils.evaluate_mixed_1zsc import (  # noqa: E402
    QLearningPolicy,
    load_ego_policies,
    load_q_method,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.collect_pg_mixed_latent_decoder_dataset import (  # noqa: E402
    policy_probs_on_query_obs,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.latent_partner_decoder import (  # noqa: E402
    apply_latent_partner_decoder,
    load_latent_decoder_npz,
)

EPS = 1e-6


def normalize_probs(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), EPS)


def tv(a, b):
    return 0.5 * np.abs(normalize_probs(a) - normalize_probs(b)).sum(axis=-1)


def kl_target_pred(target, pred):
    target = normalize_probs(target)
    pred = normalize_probs(pred)
    return np.sum(
        target * (np.log(np.maximum(target, EPS)) - np.log(np.maximum(pred, EPS))),
        axis=-1,
    )


def entropy(probs):
    probs = normalize_probs(probs)
    return -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=-1)


def direct_blend_probs(base_probs, estimator_probs, alpha, clip):
    base_probs = normalize_probs(base_probs)
    estimator_probs = normalize_probs(estimator_probs)
    delta = np.log(np.maximum(estimator_probs, EPS)) - np.log(np.maximum(base_probs, EPS))
    logits = np.log(np.maximum(base_probs, EPS)) + float(alpha) * np.clip(
        delta, -float(clip), float(clip)
    )
    logits = logits - logits.max(axis=-1, keepdims=True)
    out = np.exp(logits)
    return out / np.maximum(out.sum(axis=-1, keepdims=True), EPS)


def make_history(obs, actions, history_len):
    obs = np.asarray(obs)
    actions = np.asarray(actions, dtype=np.int32)
    n = int(actions.shape[0])
    obs_hist = np.zeros((n, history_len) + obs.shape[1:], dtype=np.float16)
    act_hist = np.zeros((n, history_len), dtype=np.int32)
    for t in range(n):
        start = max(0, t - history_len)
        length = t - start
        if length > 0:
            obs_hist[t, -length:] = obs[start:t].astype(np.float16)
            act_hist[t, -length:] = actions[start:t]
    return obs_hist, act_hist


def predict_decoder_probs(params, query_obs, obs_hist, act_hist, action_dim, batch_size):
    probs = []
    for start in range(0, len(query_obs), batch_size):
        end = min(len(query_obs), start + batch_size)
        logits = apply_latent_partner_decoder(
            params,
            jnp.asarray(query_obs[start:end], dtype=jnp.float32),
            jnp.asarray(obs_hist[start:end], dtype=jnp.float32),
            jnp.asarray(act_hist[start:end], dtype=jnp.int32),
            action_dim,
            deterministic=True,
            return_aux=False,
        )
        probs.append(np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float32))
    return np.concatenate(probs, axis=0)


def parse_decoder_specs(specs):
    parsed = []
    for raw in specs:
        if "=" in raw:
            name, path = raw.split("=", 1)
        else:
            path = raw
            name = Path(path).parent.name or Path(path).stem
        parsed.append((name, Path(path)))
    return parsed


@dataclass
class GeometryAccumulator:
    n: int = 0
    n_episodes: int = 0
    sum_reward: float = 0.0
    sum_base_kl: float = 0.0
    sum_est_kl: float = 0.0
    sum_blend_kl: float = 0.0
    sum_base_tv: float = 0.0
    sum_est_tv: float = 0.0
    sum_blend_tv: float = 0.0
    sum_base_acc: float = 0.0
    sum_est_acc: float = 0.0
    sum_blend_acc: float = 0.0
    sum_base_top2: float = 0.0
    sum_est_top2: float = 0.0
    sum_blend_top2: float = 0.0
    sum_est_entropy: float = 0.0
    sum_blend_entropy: float = 0.0
    sum_est_maxprob: float = 0.0
    sum_blend_maxprob: float = 0.0
    sum_base_est_tv: float = 0.0
    sum_base_blend_tv: float = 0.0
    sum_est_shrink_tv: float = 0.0
    sum_blend_shrink_tv: float = 0.0
    sum_est_shrink_kl: float = 0.0
    sum_blend_shrink_kl: float = 0.0
    sum_flip: float = 0.0
    sum_helpful_flip: float = 0.0
    sum_harmful_flip: float = 0.0
    sum_wrong_to_wrong_flip: float = 0.0

    def add(self, target, base, estimator, blend, reward):
        if len(target) == 0:
            return
        target = normalize_probs(target)
        base = normalize_probs(base)
        estimator = normalize_probs(estimator)
        blend = normalize_probs(blend)

        n = int(len(target))
        target_arg = np.argmax(target, axis=-1)
        base_arg = np.argmax(base, axis=-1)
        est_arg = np.argmax(estimator, axis=-1)
        blend_arg = np.argmax(blend, axis=-1)
        est_order = np.argsort(estimator, axis=-1)
        blend_order = np.argsort(blend, axis=-1)
        base_order = np.argsort(base, axis=-1)

        base_kl = kl_target_pred(target, base)
        est_kl = kl_target_pred(target, estimator)
        blend_kl = kl_target_pred(target, blend)
        base_tv = tv(target, base)
        est_tv = tv(target, estimator)
        blend_tv = tv(target, blend)

        base_correct = base_arg == target_arg
        blend_correct = blend_arg == target_arg
        flipped = blend_arg != base_arg

        self.n += n
        self.n_episodes += 1
        self.sum_reward += float(reward)
        self.sum_base_kl += float(base_kl.sum())
        self.sum_est_kl += float(est_kl.sum())
        self.sum_blend_kl += float(blend_kl.sum())
        self.sum_base_tv += float(base_tv.sum())
        self.sum_est_tv += float(est_tv.sum())
        self.sum_blend_tv += float(blend_tv.sum())
        self.sum_base_acc += float(base_correct.sum())
        self.sum_est_acc += float((est_arg == target_arg).sum())
        self.sum_blend_acc += float(blend_correct.sum())
        self.sum_base_top2 += float(np.any(base_order[:, -2:] == target_arg[:, None], axis=-1).sum())
        self.sum_est_top2 += float(np.any(est_order[:, -2:] == target_arg[:, None], axis=-1).sum())
        self.sum_blend_top2 += float(np.any(blend_order[:, -2:] == target_arg[:, None], axis=-1).sum())
        self.sum_est_entropy += float(entropy(estimator).sum())
        self.sum_blend_entropy += float(entropy(blend).sum())
        self.sum_est_maxprob += float(np.max(estimator, axis=-1).sum())
        self.sum_blend_maxprob += float(np.max(blend, axis=-1).sum())
        self.sum_base_est_tv += float(tv(base, estimator).sum())
        self.sum_base_blend_tv += float(tv(base, blend).sum())
        self.sum_est_shrink_tv += float((est_tv < base_tv).sum())
        self.sum_blend_shrink_tv += float((blend_tv < base_tv).sum())
        self.sum_est_shrink_kl += float((est_kl < base_kl).sum())
        self.sum_blend_shrink_kl += float((blend_kl < base_kl).sum())
        self.sum_flip += float(flipped.sum())
        self.sum_helpful_flip += float((flipped & (~base_correct) & blend_correct).sum())
        self.sum_harmful_flip += float((flipped & base_correct & (~blend_correct)).sum())
        self.sum_wrong_to_wrong_flip += float((flipped & (~base_correct) & (~blend_correct)).sum())

    def row(self, estimator, partner_method, role, subset):
        denom = max(self.n, 1)
        ep_denom = max(self.n_episodes, 1)
        flip_denom = max(self.sum_flip, 1.0)
        return {
            "estimator": estimator,
            "partner_method": partner_method,
            "role": role,
            "subset": subset,
            "n_rows": self.n,
            "n_episodes": self.n_episodes,
            "mean_reward": self.sum_reward / ep_denom,
            "base_kl": self.sum_base_kl / denom,
            "estimator_kl": self.sum_est_kl / denom,
            "blend_kl": self.sum_blend_kl / denom,
            "blend_minus_base_kl": (self.sum_blend_kl - self.sum_base_kl) / denom,
            "base_tv": self.sum_base_tv / denom,
            "estimator_tv": self.sum_est_tv / denom,
            "blend_tv": self.sum_blend_tv / denom,
            "blend_minus_base_tv": (self.sum_blend_tv - self.sum_base_tv) / denom,
            "base_acc": self.sum_base_acc / denom,
            "estimator_acc": self.sum_est_acc / denom,
            "blend_acc": self.sum_blend_acc / denom,
            "base_top2": self.sum_base_top2 / denom,
            "estimator_top2": self.sum_est_top2 / denom,
            "blend_top2": self.sum_blend_top2 / denom,
            "estimator_entropy": self.sum_est_entropy / denom,
            "blend_entropy": self.sum_blend_entropy / denom,
            "estimator_maxprob": self.sum_est_maxprob / denom,
            "blend_maxprob": self.sum_blend_maxprob / denom,
            "base_estimator_tv": self.sum_base_est_tv / denom,
            "base_blend_tv": self.sum_base_blend_tv / denom,
            "estimator_tv_shrink_fraction": self.sum_est_shrink_tv / denom,
            "blend_tv_shrink_fraction": self.sum_blend_shrink_tv / denom,
            "estimator_kl_shrink_fraction": self.sum_est_shrink_kl / denom,
            "blend_kl_shrink_fraction": self.sum_blend_shrink_kl / denom,
            "flip_fraction": self.sum_flip / denom,
            "helpful_flip_fraction": self.sum_helpful_flip / denom,
            "harmful_flip_fraction": self.sum_harmful_flip / denom,
            "wrong_to_wrong_flip_fraction": self.sum_wrong_to_wrong_flip / denom,
            "helpful_flip_precision": self.sum_helpful_flip / flip_denom,
            "harmful_flip_precision": self.sum_harmful_flip / flip_denom,
        }


def add_group_metrics(accs, estimator, method, role, subset, target, base, est, blend, reward):
    keys = [
        (estimator, method, role, subset),
        (estimator, method, "both_roles", subset),
        (estimator, "all_q_partners", role, subset),
        (estimator, "all_q_partners", "both_roles", subset),
    ]
    for key in keys:
        accs[key].add(target, base, est, blend, reward)


def make_q_policy(loaded_q, params, args, agent_index):
    return QLearningPolicy(
        params,
        loaded_q.apply_one,
        action_mode=args.q_action_mode,
        temperature=args.q_temperature,
        epsilon=args.q_epsilon,
        is_rnn=loaded_q.is_rnn,
        hidden_size=loaded_q.hidden_size,
        agent_index=agent_index,
        preprocess_flat_obs=loaded_q.preprocess_flat_obs,
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ego_run_dir",
        type=Path,
        default=Path("runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606"),
    )
    parser.add_argument(
        "--q_run_root",
        type=Path,
        default=Path("runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M"),
    )
    parser.add_argument("--methods", type=str, default="iql,vdn,pqn_vdn")
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_seeds", type=int, default=10)
    parser.add_argument("--max_ego_policies", type=int, default=2)
    parser.add_argument("--max_partner_policies", type=int, default=5)
    parser.add_argument("--rollout_ego_mode", type=str, default="base_no_test_adapt")
    parser.add_argument("--rollout_latent_decoder_path", type=Path, default=None)
    parser.add_argument("--ego_greedy", action="store_true")
    parser.add_argument("--q_action_mode", type=str, default="greedy", choices=("greedy", "softmax", "epsilon_greedy"))
    parser.add_argument("--q_temperature", type=float, default=1.0)
    parser.add_argument("--q_epsilon", type=float, default=0.0)
    parser.add_argument("--decoder", action="append", required=True)
    parser.add_argument("--ttac_history_len", type=int, default=50)
    parser.add_argument("--ttac_latent_decoder_history_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoder_specs = parse_decoder_specs(args.decoder)
    decoder_params = [(name, load_latent_decoder_npz(path)) for name, path in decoder_specs]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    env = OvercookedV2(layout=args.layout)

    model_overrides = {
        "TTAC_V5_ESTIMATOR": None,
        "TTAC_HISTORY_LEN": args.ttac_history_len,
        "TTAC_LATENT_DECODER_HISTORY_LEN": args.ttac_latent_decoder_history_len,
        "TTAC_LATENT_DECODER_HISTORY_MODE": "full",
        "TTAC_LATENT_DECODER_BLEND_ALPHA": args.alpha,
        "TTAC_LATENT_DECODER_BLEND_CLIP": args.clip,
        "TTAC_LATENT_DECODER_QUERY_MODE": "real",
        "TTAC_LATENT_DECODER_GATE_MODE": "none",
    }
    rollout_latent_path = (
        args.rollout_latent_decoder_path
        if args.rollout_ego_mode.startswith("ttac_v5_8_latent_decoder_")
        else None
    )
    ego_policies, ego_config = load_ego_policies(
        args.ego_run_dir,
        args.rollout_ego_mode,
        args.max_ego_policies,
        stochastic=not args.ego_greedy,
        estimator_path=None,
        latent_decoder_path=rollout_latent_path,
        model_overrides=model_overrides,
    )
    ego_env_kwargs = dict(ego_config["env"]["ENV_KWARGS"])
    ego_layout = str(ego_env_kwargs.get("layout", args.layout))
    if ego_layout != args.layout:
        raise ValueError(f"ego layout {ego_layout} != requested layout {args.layout}")

    accs: dict[tuple[str, str, str, str], GeometryAccumulator] = defaultdict(GeometryAccumulator)
    episode_rows = []

    for method in methods:
        loaded_q = load_q_method(args.q_run_root, method, args.layout, args.max_partner_policies)
        if loaded_q.is_rnn:
            raise ValueError(f"{method} is recurrent; this diagnostic expects feed-forward Q partners.")
        for ego_idx, (ego_label, ego_policy) in enumerate(ego_policies):
            for partner_idx, (partner_label, partner_params) in enumerate(zip(loaded_q.labels, loaded_q.params)):
                pairings = {
                    "ego_agent0": (
                        PolicyPairing(
                            ego_policy,
                            make_q_policy(loaded_q, partner_params, args, agent_index=1),
                        ),
                        "agent_0",
                        "agent_1",
                        make_q_policy(loaded_q, partner_params, args, agent_index=0),
                    ),
                    "ego_agent1": (
                        PolicyPairing(
                            make_q_policy(loaded_q, partner_params, args, agent_index=0),
                            ego_policy,
                        ),
                        "agent_1",
                        "agent_0",
                        make_q_policy(loaded_q, partner_params, args, agent_index=1),
                    ),
                }
                for role, (pairing, ego_agent, partner_agent, same_view_teacher) in pairings.items():
                    pair_seed = args.seed + 100000 * ego_idx + 1000 * partner_idx + (0 if role == "ego_agent0" else 500)
                    keys = jax.random.split(jax.random.PRNGKey(pair_seed), args.num_eval_seeds)
                    pair_id = f"{args.rollout_ego_mode}__{method}__{ego_label}__{partner_label}__{role}"
                    print(f"[onpolicy-geometry] {pair_id} seeds={args.num_eval_seeds}", flush=True)
                    rollout_fn = jax.jit(jax.vmap(lambda k: get_rollout_with_observations(pairing, env, k)))
                    rollouts = rollout_fn(keys)
                    total_rewards = np.asarray(rollouts.total_reward, dtype=np.float32)
                    obs_ego = np.asarray(rollouts.obs_seq[ego_agent], dtype=np.float32)
                    obs_partner = np.asarray(rollouts.obs_seq[partner_agent], dtype=np.float32)
                    act_partner = np.asarray(rollouts.actions_seq[partner_agent], dtype=np.int32)

                    for seed_idx in range(args.num_eval_seeds):
                        query_obs = obs_ego[seed_idx]
                        partner_obs = obs_partner[seed_idx]
                        partner_action = act_partner[seed_idx]
                        reward = float(total_rewards[seed_idx])
                        obs_hist, act_hist = make_history(
                            partner_obs,
                            partner_action,
                            args.ttac_latent_decoder_history_len,
                        )
                        target = normalize_probs(
                            policy_probs_on_query_obs(
                                same_view_teacher,
                                query_obs,
                                batch_size=args.batch_size,
                            )
                        )
                        base = normalize_probs(
                            policy_probs_on_query_obs(
                                ego_policy,
                                query_obs,
                                batch_size=args.batch_size,
                            )
                        )
                        timesteps = np.arange(len(query_obs))
                        subset_masks = {
                            "all_steps": np.ones(len(query_obs), dtype=bool),
                            "t_ge_1": timesteps >= 1,
                            f"t_ge_{args.ttac_latent_decoder_history_len}": timesteps >= args.ttac_latent_decoder_history_len,
                        }
                        for estimator_name, params in decoder_params:
                            est = normalize_probs(
                                predict_decoder_probs(
                                    params,
                                    query_obs,
                                    obs_hist,
                                    act_hist,
                                    args.action_dim,
                                    args.batch_size,
                                )
                            )
                            blend = direct_blend_probs(base, est, args.alpha, args.clip)
                            for subset, mask in subset_masks.items():
                                add_group_metrics(
                                    accs,
                                    estimator_name,
                                    method,
                                    role,
                                    subset,
                                    target[mask],
                                    base[mask],
                                    est[mask],
                                    blend[mask],
                                    reward,
                                )
                            all_mask = subset_masks["all_steps"]
                            target_all = target[all_mask]
                            base_all = base[all_mask]
                            est_all = est[all_mask]
                            blend_all = blend[all_mask]
                            base_kl = kl_target_pred(target_all, base_all)
                            blend_kl = kl_target_pred(target_all, blend_all)
                            base_arg = np.argmax(base_all, axis=-1)
                            target_arg = np.argmax(target_all, axis=-1)
                            blend_arg = np.argmax(blend_all, axis=-1)
                            flipped = blend_arg != base_arg
                            episode_rows.append(
                                {
                                    "estimator": estimator_name,
                                    "pair_id": pair_id,
                                    "method": method,
                                    "role": role,
                                    "ego_label": ego_label,
                                    "partner_label": partner_label,
                                    "seed_idx": seed_idx,
                                    "reward": reward,
                                    "mean_base_kl": float(base_kl.mean()),
                                    "mean_blend_kl": float(blend_kl.mean()),
                                    "mean_blend_minus_base_kl": float((blend_kl - base_kl).mean()),
                                    "base_acc": float(np.mean(base_arg == target_arg)),
                                    "blend_acc": float(np.mean(blend_arg == target_arg)),
                                    "flip_fraction": float(np.mean(flipped)),
                                    "helpful_flip_fraction": float(np.mean(flipped & (base_arg != target_arg) & (blend_arg == target_arg))),
                                    "harmful_flip_fraction": float(np.mean(flipped & (base_arg == target_arg) & (blend_arg != target_arg))),
                                }
                            )

    rows = []
    for (estimator, method, role, subset), acc in sorted(accs.items()):
        rows.append(acc.row(estimator, method, role, subset))

    write_csv(args.output_dir / "onpolicy_blend_geometry_summary.csv", rows)
    write_csv(args.output_dir / "onpolicy_blend_geometry_episodes.csv", episode_rows)

    lines = [
        "# On-Policy Blend Geometry",
        "",
        f"- rollout_ego_mode: `{args.rollout_ego_mode}`",
        f"- rollout_latent_decoder_path: `{args.rollout_latent_decoder_path or ''}`",
        f"- decoders: `{', '.join(name for name, _path in decoder_specs)}`",
        f"- methods: `{','.join(methods)}`",
        f"- max ego policies: `{args.max_ego_policies}`",
        f"- max partner policies: `{args.max_partner_policies}`",
        f"- eval seeds per pair/role: `{args.num_eval_seeds}`",
        f"- history len: `{args.ttac_latent_decoder_history_len}`",
        f"- alpha: `{args.alpha}`",
        f"- clip: `{args.clip}`",
        "",
        "Metrics compare base, estimator target, and direct blend against the true Q partner policy on the same ego observation. Rollout states are shared across all listed decoders.",
        "",
        "| estimator | partner | role | subset | n | reward | base KL | est KL | blend KL | dKL | base TV | est TV | blend TV | dTV | base acc | est acc | blend acc | TV shrink | KL shrink | flip | helpful flip | harmful flip | est entropy | est maxprob |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["role"] != "both_roles" or row["subset"] != "all_steps":
            continue
        lines.append(
            f"| {row['estimator']} | {row['partner_method']} | {row['role']} | {row['subset']} | "
            f"{row['n_rows']} | {row['mean_reward']:.3f} | "
            f"{row['base_kl']:.4f} | {row['estimator_kl']:.4f} | {row['blend_kl']:.4f} | {row['blend_minus_base_kl']:.4f} | "
            f"{row['base_tv']:.4f} | {row['estimator_tv']:.4f} | {row['blend_tv']:.4f} | {row['blend_minus_base_tv']:.4f} | "
            f"{row['base_acc']:.4f} | {row['estimator_acc']:.4f} | {row['blend_acc']:.4f} | "
            f"{100.0 * row['blend_tv_shrink_fraction']:.1f}% | {100.0 * row['blend_kl_shrink_fraction']:.1f}% | "
            f"{100.0 * row['flip_fraction']:.1f}% | {100.0 * row['helpful_flip_fraction']:.1f}% | {100.0 * row['harmful_flip_fraction']:.1f}% | "
            f"{row['estimator_entropy']:.4f} | {row['estimator_maxprob']:.4f} |"
        )
    args.output_dir.joinpath("onpolicy_blend_geometry_summary.md").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
