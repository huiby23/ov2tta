from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import init_rollout
from overcooked_v2_experiments.ttac_v6_support_refinement.policy import PPOPolicy
from overcooked_v2_experiments.ttac_v6_support_refinement.utils.store import load_all_checkpoints


def make_policies(run_dir: Path):
    all_ckpts, config = load_all_checkpoints(run_dir, final_only=True)
    run_keys = sorted(all_ckpts.keys(), key=lambda x: int(x.split("_")[1]))
    policies = [
        PPOPolicy(all_ckpts[k]["ckpt_final"].params, config, eval_mode="base_no_test_adapt")
        for k in run_keys
    ]
    return run_keys, policies, config


def get_rollout_seq(policies: PolicyPairing, env, key):
    init_hstate, get_actions = init_rollout(policies, env)

    @jax.jit
    def perform_step(carry, key):
        obs, state, done, total_reward, hstate = carry
        key_sample, key_step = jax.random.split(key, 2)
        actions, next_hstate = get_actions(obs, done, hstate, key_sample)
        next_obs, next_state, reward, next_done, _info = env.step(key_step, state, actions)
        if env.num_agents == 2:
            updated_hstate = {}
            for i, policy in enumerate(policies):
                agent_id = f"agent_{i}"
                partner_id = f"agent_{1-i}"
                updated_hstate[agent_id] = policy.update_after_step(
                    next_hstate[agent_id], obs[partner_id], actions[partner_id], next_done[agent_id]
                )
        else:
            updated_hstate = next_hstate
        carry = (next_obs, next_state, next_done, total_reward + reward["agent_0"], updated_hstate)
        return carry, (obs, done, actions, reward["agent_0"])

    key, key_reset = jax.random.split(key, 2)
    obs, state = env.reset(key_reset)
    done = {f"agent_{i}": False for i in range(env.num_agents)}
    done["__all__"] = False
    keys = jax.random.split(key, env.max_steps)
    carry = (obs, state, done, 0.0, init_hstate)
    carry, seq = jax.lax.scan(perform_step, carry, keys)
    return seq, carry[-2]


def _action_history(actions, idx, history_len):
    hist = np.zeros((len(idx), history_len), dtype=np.int16)
    for row, t in enumerate(idx):
        t = int(t)
        start = max(0, t - history_len)
        seq = actions[start:t]
        if len(seq):
            hist[row, -len(seq):] = seq
    return hist


def policy_probs_on_query_obs(policy: PPOPolicy, query_obs: np.ndarray, batch_size: int = 1024):
    outs = []
    for start in range(0, len(query_obs), batch_size):
        chunk = jnp.asarray(query_obs[start:start + batch_size], dtype=jnp.float32)
        logits, _value, _aux = policy._apply_batch(policy.params, chunk, adapter_readout_scale=0.0)
        outs.append(np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float32))
    return np.concatenate(outs, axis=0)


def append_role(
    rows,
    query_obs,
    partner_obs,
    ego_action,
    partner_action,
    partner_target_probs,
    pair_label,
    episode_id,
    role,
    stride,
    history_len,
):
    ego_action = np.asarray(ego_action, dtype=np.int16)
    partner_action = np.asarray(partner_action, dtype=np.int16)
    idx = np.arange(0, len(ego_action), stride, dtype=np.int32)
    prev_partner = np.concatenate([[0], partner_action[:-1]]).astype(np.int16)
    prev_ego = np.concatenate([[0], ego_action[:-1]]).astype(np.int16)
    rows["query_obs"].append(np.asarray(query_obs, dtype=np.float16)[idx])
    rows["partner_obs"].append(np.asarray(partner_obs, dtype=np.float16)[idx])
    rows["ego_action"].append(ego_action[idx])
    rows["partner_action"].append(partner_action[idx])
    rows["prev_partner_action"].append(prev_partner[idx])
    rows["prev_ego_action"].append(prev_ego[idx])
    rows["partner_action_history"].append(_action_history(partner_action, idx, history_len))
    rows["ego_action_history"].append(_action_history(ego_action, idx, history_len))
    rows["target_partner_probs"].append(np.asarray(partner_target_probs, dtype=np.float32)[idx])
    rows["pair_label"].extend([pair_label] * len(idx))
    rows["episode_id"].append(np.full(len(idx), episode_id, dtype=np.int32))
    rows["timestep"].append(idx.astype(np.int16))
    rows["role"].append(np.full(len(idx), role, dtype=np.int8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_seeds", type=int, default=100)
    parser.add_argument("--max_pairings", type=int, default=20)
    parser.add_argument("--transition_stride", type=int, default=4)
    parser.add_argument("--max_transitions", type=int, default=250000)
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--policy_prob_batch_size", type=int, default=1024)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    run_keys, policies, config = make_policies(run_dir)
    env = OvercookedV2(**config["env"]["ENV_KWARGS"])
    pairs = list(itertools.permutations(range(len(run_keys)), 2))[: args.max_pairings]

    rows = {k: [] for k in [
        "query_obs", "partner_obs", "ego_action", "partner_action", "prev_partner_action",
        "prev_ego_action", "partner_action_history", "ego_action_history",
        "target_partner_probs", "episode_id", "timestep", "role",
    ]}
    rows["pair_label"] = []
    summary_rows = []
    episode_id = 0

    for pair_idx, (i, j) in enumerate(pairs):
        pair_label = f"{run_keys[i]}x{run_keys[j]}"
        pairing = PolicyPairing(policies[i], policies[j])
        keys = jax.random.split(jax.random.PRNGKey(args.seed + 1000 * pair_idx), args.num_eval_seeds)
        print(f"[collect_v5] pair {pair_idx+1}/{len(pairs)} {pair_label}", flush=True)
        rollout_fn = jax.jit(jax.vmap(lambda rollout_key: get_rollout_seq(pairing, env, rollout_key)))
        (obs_batch, _done_batch, actions_batch, _reward_batch), total_reward_batch = rollout_fn(keys)
        total_reward_batch = np.asarray(total_reward_batch, dtype=np.float32)
        rewards = []
        for seed_idx in range(args.num_eval_seeds):
            obs0 = np.asarray(obs_batch["agent_0"][seed_idx])
            obs1 = np.asarray(obs_batch["agent_1"][seed_idx])
            a0 = np.asarray(actions_batch["agent_0"][seed_idx])
            a1 = np.asarray(actions_batch["agent_1"][seed_idx])
            target_j_on_obs0 = policy_probs_on_query_obs(policies[j], obs0, args.policy_prob_batch_size)
            target_i_on_obs1 = policy_probs_on_query_obs(policies[i], obs1, args.policy_prob_batch_size)
            append_role(rows, obs0, obs1, a0, a1, target_j_on_obs0, pair_label, episode_id, 0, args.transition_stride, args.history_len)
            append_role(rows, obs1, obs0, a1, a0, target_i_on_obs1, pair_label, episode_id, 1, args.transition_stride, args.history_len)
            rewards.append(float(total_reward_batch[seed_idx]))
            episode_id += 1
        summary_rows.append({"pair_label": pair_label, "episodes": args.num_eval_seeds, "mean_reward": float(np.mean(rewards))})

    arrays = {}
    for key, parts in rows.items():
        if key == "pair_label":
            arrays[key] = np.asarray(parts)
        else:
            arrays[key] = np.concatenate(parts, axis=0)
    total = arrays["ego_action"].shape[0]
    if args.max_transitions and total > args.max_transitions:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(total, size=args.max_transitions, replace=False))
        for key in arrays:
            arrays[key] = arrays[key][keep]
        total = args.max_transitions

    dataset_path = output / "agreement_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    with (output / "dataset_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_label", "episodes", "mean_reward"])
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "dataset_summary.md").open("w") as f:
        f.write("# TTAC v5 agreement dataset\n\n")
        f.write(f"- run_dir: `{run_dir}`\n- pairs: `{len(pairs)}`\n- eval_seeds_per_pair: `{args.num_eval_seeds}`\n")
        f.write(f"- transition_stride: {args.transition_stride}\n- history_len: {args.history_len}\n- stored_transitions: {total}\n- dataset: {dataset_path}\n")
        f.write("- target: partner policy distribution on ego/query observation, not sampled partner action.\n")
    print(f"[collect_v5] wrote {dataset_path} transitions={total}", flush=True)


if __name__ == "__main__":
    main()
