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


def tv(target, pred):
    return 0.5 * np.abs(normalize_probs(target) - normalize_probs(pred)).sum(axis=-1)


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


@dataclass
class Accumulator:
    n: int = 0
    sum_kl: float = 0.0
    sum_tv: float = 0.0
    sum_acc: float = 0.0
    sum_top2: float = 0.0
    sum_pred_entropy: float = 0.0
    sum_target_entropy: float = 0.0
    sum_max_prob: float = 0.0
    sum_reward: float = 0.0
    n_episodes: int = 0

    def add_metrics(self, target, pred):
        if len(target) == 0:
            return
        target = normalize_probs(target)
        pred = normalize_probs(pred)
        target_arg = np.argmax(target, axis=-1)
        pred_arg = np.argmax(pred, axis=-1)
        pred_rank = np.argsort(pred, axis=-1)
        n = int(len(target))
        self.n += n
        self.sum_kl += float(kl_target_pred(target, pred).sum())
        self.sum_tv += float(tv(target, pred).sum())
        self.sum_acc += float(np.sum(pred_arg == target_arg))
        self.sum_top2 += float(np.sum(np.any(pred_rank[:, -2:] == target_arg[:, None], axis=-1)))
        self.sum_pred_entropy += float(entropy(pred).sum())
        self.sum_target_entropy += float(entropy(target).sum())
        self.sum_max_prob += float(np.max(pred, axis=-1).sum())

    def add_reward(self, reward):
        self.sum_reward += float(reward)
        self.n_episodes += 1

    def row(self, partner_method, role, subset):
        denom = max(self.n, 1)
        ep_denom = max(self.n_episodes, 1)
        return {
            "partner_method": partner_method,
            "role": role,
            "subset": subset,
            "n_rows": self.n,
            "n_episodes": self.n_episodes,
            "mean_reward": self.sum_reward / ep_denom,
            "kl": self.sum_kl / denom,
            "tv": self.sum_tv / denom,
            "argmax_acc": self.sum_acc / denom,
            "top2_acc": self.sum_top2 / denom,
            "pred_entropy": self.sum_pred_entropy / denom,
            "target_entropy": self.sum_target_entropy / denom,
            "pred_max_prob": self.sum_max_prob / denom,
        }


def add_group_metrics(accs, method, role, subset, target, pred, reward):
    keys = [
        (method, role, subset),
        (method, "both_roles", subset),
        ("all_q_partners", role, subset),
        ("all_q_partners", "both_roles", subset),
    ]
    for key in keys:
        accs[key].add_metrics(target, pred)
        accs[key].add_reward(reward)


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
    parser.add_argument("--num_eval_seeds", type=int, default=20)
    parser.add_argument("--max_ego_policies", type=int, default=2)
    parser.add_argument("--max_partner_policies", type=int, default=5)
    parser.add_argument("--ego_mode", type=str, default="ttac_v5_8_latent_decoder_policy")
    parser.add_argument("--ego_greedy", action="store_true")
    parser.add_argument("--q_action_mode", type=str, default="greedy", choices=("greedy", "softmax", "epsilon_greedy"))
    parser.add_argument("--q_temperature", type=float, default=1.0)
    parser.add_argument("--q_epsilon", type=float, default=0.0)
    parser.add_argument("--ttac_latent_decoder_path", type=Path, required=True)
    parser.add_argument("--ttac_history_len", type=int, default=50)
    parser.add_argument("--ttac_latent_decoder_history_len", type=int, default=50)
    parser.add_argument(
        "--ttac_latent_decoder_history_mode",
        type=str,
        default="full",
        choices=("full", "action_only", "no_history"),
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    env = OvercookedV2(layout=args.layout)

    model_overrides = {
        "TTAC_V5_ESTIMATOR": None,
        "TTAC_HISTORY_LEN": args.ttac_history_len,
        "TTAC_LATENT_DECODER_HISTORY_LEN": args.ttac_latent_decoder_history_len,
        "TTAC_LATENT_DECODER_HISTORY_MODE": args.ttac_latent_decoder_history_mode,
        "TTAC_LATENT_DECODER_BLEND_ALPHA": 0.5,
        "TTAC_LATENT_DECODER_BLEND_CLIP": 2.0,
        "TTAC_LATENT_DECODER_QUERY_MODE": "real",
        "TTAC_LATENT_DECODER_GATE_MODE": "none",
    }
    ego_policies, ego_config = load_ego_policies(
        args.ego_run_dir,
        args.ego_mode,
        args.max_ego_policies,
        stochastic=not args.ego_greedy,
        estimator_path=None,
        latent_decoder_path=args.ttac_latent_decoder_path,
        model_overrides=model_overrides,
    )
    ego_env_kwargs = dict(ego_config["env"]["ENV_KWARGS"])
    ego_layout = str(ego_env_kwargs.get("layout", args.layout))
    if ego_layout != args.layout:
        raise ValueError(f"ego layout {ego_layout} != requested layout {args.layout}")

    decoder_params = load_latent_decoder_npz(args.ttac_latent_decoder_path)
    accs: dict[tuple[str, str, str], Accumulator] = defaultdict(Accumulator)
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
                    pair_id = f"{args.ego_mode}__{method}__{ego_label}__{partner_label}__{role}"
                    print(f"[direct-onpolicy] {pair_id} seeds={args.num_eval_seeds}", flush=True)
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
                        pred = predict_decoder_probs(
                            decoder_params,
                            query_obs,
                            obs_hist,
                            act_hist,
                            args.action_dim,
                            args.batch_size,
                        )
                        target = policy_probs_on_query_obs(
                            same_view_teacher,
                            query_obs,
                            batch_size=args.batch_size,
                        )
                        target = normalize_probs(target)
                        pred = normalize_probs(pred)
                        timesteps = np.arange(len(query_obs))
                        subset_masks = {
                            "all_steps": np.ones(len(query_obs), dtype=bool),
                            "t_ge_1": timesteps >= 1,
                            f"t_ge_{args.ttac_latent_decoder_history_len}": timesteps >= args.ttac_latent_decoder_history_len,
                        }
                        for subset, mask in subset_masks.items():
                            add_group_metrics(
                                accs,
                                method,
                                role,
                                subset,
                                target[mask],
                                pred[mask],
                                reward,
                            )
                        episode_rows.append(
                            {
                                "pair_id": pair_id,
                                "method": method,
                                "role": role,
                                "ego_label": ego_label,
                                "partner_label": partner_label,
                                "seed_idx": seed_idx,
                                "reward": reward,
                            }
                        )

    rows = []
    for (method, role, subset), acc in sorted(accs.items()):
        rows.append(acc.row(method, role, subset))

    summary_csv = args.output_dir / "direct_onpolicy_accuracy_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    episode_csv = args.output_dir / "direct_onpolicy_episodes.csv"
    with episode_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    lines = [
        "# Direct Estimator On-Policy Accuracy",
        "",
        f"- ego_mode: `{args.ego_mode}`",
        f"- latent decoder: `{args.ttac_latent_decoder_path}`",
        f"- methods: `{','.join(methods)}`",
        f"- max ego policies: `{args.max_ego_policies}`",
        f"- max partner policies: `{args.max_partner_policies}`",
        f"- eval seeds per pair/role: `{args.num_eval_seeds}`",
        f"- history len: `{args.ttac_latent_decoder_history_len}`",
        f"- q action mode: `{args.q_action_mode}`",
        "",
        "Metrics compare estimator output `pi_hat(.|ego_obs, partner_history)` against the true Q partner policy on the same `ego_obs`; no blend/base is used.",
        "",
        "| partner_method | role | subset | n_rows | n_episodes | mean_reward | KL | TV | argmax acc | top2 acc | pred entropy | pred maxprob |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['partner_method']} | {row['role']} | {row['subset']} | "
            f"{row['n_rows']} | {row['n_episodes']} | {row['mean_reward']:.3f} | "
            f"{row['kl']:.4f} | {row['tv']:.4f} | {row['argmax_acc']:.4f} | "
            f"{row['top2_acc']:.4f} | {row['pred_entropy']:.4f} | {row['pred_max_prob']:.4f} |"
        )
    (args.output_dir / "direct_onpolicy_accuracy_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
