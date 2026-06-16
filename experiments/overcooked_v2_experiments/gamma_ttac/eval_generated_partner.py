from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout_with_observations
from overcooked_v2_experiments.gamma_ttac.policy import GammaGeneratedPartnerPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a GAMMA generated partner against source policies.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--backend", choices=["ppo", "mappo"], default="ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--greedy-source", action="store_true")
    parser.add_argument("--greedy-gamma", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load_policies(run_dir: Path, backend: str, greedy: bool):
    if backend == "ppo":
        from overcooked_v2_experiments.ppo.policy import PPOPolicy as PolicyCls
        from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints
    elif backend == "mappo":
        from overcooked_v2_experiments.mappo.policy import MAPPOPolicy as PolicyCls
        from overcooked_v2_experiments.mappo.utils.store import load_all_checkpoints
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    all_params, config = load_all_checkpoints(run_dir, final_only=True)
    run_keys = sorted(all_params.keys(), key=lambda x: int(x.split("_")[1]))[:]
    policies = [PolicyCls(all_params[k]["ckpt_final"].params, config, stochastic=not greedy) for k in run_keys]
    return policies, run_keys, config


def main():
    args = parse_args()
    policies, run_keys, config = _load_policies(Path(args.run_dir), args.backend, args.greedy_source)
    policies = policies[: args.max_runs]
    run_keys = run_keys[: args.max_runs]
    gamma = GammaGeneratedPartnerPolicy.from_checkpoint(args.vae_checkpoint, stochastic=not args.greedy_gamma)
    env_kwargs = dict(config["env"]["ENV_KWARGS"])
    layout = env_kwargs.pop("layout")
    env = OvercookedV2(layout=layout, **env_kwargs)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_seeds)
    results = {"source_agent0": {}, "source_agent1": {}}
    action_freq = {"gamma_agent0": np.zeros(6, dtype=np.int64), "gamma_agent1": np.zeros(6, dtype=np.int64)}

    for policy, run_key in zip(policies, run_keys):
        for role, pairing in [
            ("source_agent0", PolicyPairing(policy, gamma)),
            ("source_agent1", PolicyPairing(gamma, policy)),
        ]:
            rewards = []
            for key in keys:
                rollout = get_rollout_with_observations(pairing, env, key)
                rewards.append(float(jax.device_get(rollout.total_reward)))
                gamma_agent = "agent_1" if role == "source_agent0" else "agent_0"
                freq_key = "gamma_agent1" if role == "source_agent0" else "gamma_agent0"
                acts = np.asarray(jax.device_get(rollout.actions_seq[gamma_agent])).reshape(-1)
                action_freq[freq_key] += np.bincount(acts, minlength=6)
            results[role][run_key] = {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "values": rewards,
            }
            print(role, run_key, results[role][run_key], flush=True)

    summary = {}
    for role, rows in results.items():
        means = [v["mean"] for v in rows.values()]
        summary[role] = {"mean": float(np.mean(means)), "std_across_runs": float(np.std(means))}
    for k, counts in action_freq.items():
        total = int(counts.sum())
        summary[k + "_action_counts"] = counts.tolist()
        summary[k + "_action_freq"] = (counts / max(total, 1)).tolist()
    payload = {"summary": summary, "per_run": results}
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
