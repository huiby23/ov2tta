from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

from overcooked_v2_experiments.eval.policy import AbstractPolicy, PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout_with_observations
from overcooked_v2_experiments.talents.policy import TalentsGeneratedPartnerPolicy


class FixedClusterTalentsPartner(AbstractPolicy):
    """Generated TALENTS partner with a fixed latent cluster for a rollout."""

    uses_default_observation: bool = True

    def __init__(self, base: TalentsGeneratedPartnerPolicy, cluster_id: int):
        self.base = base
        self.cluster_id = int(cluster_id)

    def init_hstate(self, batch_size, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_init, key_cluster = jax.random.split(key)
        hstate = self.base.init_hstate(batch_size, key_init)
        cluster_id = jnp.full((batch_size,), self.cluster_id, dtype=jnp.int32)
        return self.base.set_cluster_ids(hstate, cluster_id, key_cluster)

    def compute_action(self, obs, done, hstate, key):
        return self.base.compute_action(obs, done, hstate, key)

    def update_after_step(self, hstate, partner_obs, partner_action, done):
        return hstate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TALENTS generated partner quality cluster by cluster."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--cluster-checkpoint", required=True)
    parser.add_argument("--backend", choices=["ppo", "mappo"], default="ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--clusters", default="all")
    parser.add_argument("--greedy-source", action="store_true")
    parser.add_argument("--greedy-talents", action="store_true")
    parser.add_argument("--z-sample-scale", type=float, default=1.0)
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
    run_keys = sorted(all_params.keys(), key=lambda x: int(x.split("_")[1]))
    policies = [
        PolicyCls(all_params[k]["ckpt_final"].params, config, stochastic=not greedy)
        for k in run_keys
    ]
    return policies, run_keys, config


def _parse_clusters(value: str, num_clusters: int) -> list[int]:
    if value == "all":
        return list(range(num_clusters))
    clusters = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        cluster_id = int(part)
        if cluster_id < 0 or cluster_id >= num_clusters:
            raise ValueError(f"Cluster id {cluster_id} outside [0, {num_clusters})")
        clusters.append(cluster_id)
    return clusters


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    policies, run_keys, config = _load_policies(run_dir, args.backend, args.greedy_source)
    policies = policies[: args.max_runs]
    run_keys = run_keys[: args.max_runs]
    talents_base = TalentsGeneratedPartnerPolicy.from_checkpoints(
        args.vae_checkpoint,
        args.cluster_checkpoint,
        stochastic=not args.greedy_talents,
        z_sample_scale=args.z_sample_scale,
    )
    cluster_ids = _parse_clusters(args.clusters, talents_base.num_clusters)

    env_kwargs = dict(config["env"]["ENV_KWARGS"])
    layout = env_kwargs.pop("layout")
    env = OvercookedV2(layout=layout, **env_kwargs)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_seeds)

    payload: dict[str, Any] = {
        "source_run_dir": str(run_dir),
        "vae_checkpoint": str(args.vae_checkpoint),
        "cluster_checkpoint": str(args.cluster_checkpoint),
        "backend": args.backend,
        "seed": args.seed,
        "num_eval_seeds": args.num_seeds,
        "max_runs": args.max_runs,
        "greedy_source": args.greedy_source,
        "greedy_talents": args.greedy_talents,
        "z_sample_scale": args.z_sample_scale,
        "clusters": {},
    }

    for cluster_id in cluster_ids:
        generated = FixedClusterTalentsPartner(talents_base, cluster_id)
        cluster_rows: dict[str, Any] = {"source_agent0": {}, "source_agent1": {}}
        action_freq = {
            "talents_agent0": np.zeros(talents_base.action_dim, dtype=np.int64),
            "talents_agent1": np.zeros(talents_base.action_dim, dtype=np.int64),
        }
        all_rewards = []
        for policy, run_key in zip(policies, run_keys):
            for role, pairing in [
                ("source_agent0", PolicyPairing(policy, generated)),
                ("source_agent1", PolicyPairing(generated, policy)),
            ]:
                rewards = []
                for key in keys:
                    rollout = get_rollout_with_observations(pairing, env, key)
                    reward = float(jax.device_get(rollout.total_reward))
                    rewards.append(reward)
                    all_rewards.append(reward)
                    talents_agent = "agent_1" if role == "source_agent0" else "agent_0"
                    freq_key = "talents_agent1" if role == "source_agent0" else "talents_agent0"
                    acts = np.asarray(jax.device_get(rollout.actions_seq[talents_agent])).reshape(-1)
                    action_freq[freq_key] += np.bincount(
                        acts, minlength=talents_base.action_dim
                    )
                cluster_rows[role][run_key] = {
                    **_stats(rewards),
                    "values": [float(v) for v in rewards],
                }
                print(
                    f"cluster={cluster_id} role={role} run={run_key} "
                    f"mean={cluster_rows[role][run_key]['mean']:.3f}",
                    flush=True,
                )

        summary = {"overall": _stats(all_rewards)}
        for role, rows in cluster_rows.items():
            role_means = [float(v["mean"]) for v in rows.values() if v["mean"] is not None]
            summary[role] = _stats(role_means)
        for freq_key, counts in action_freq.items():
            total = int(counts.sum())
            summary[f"{freq_key}_action_counts"] = counts.tolist()
            summary[f"{freq_key}_action_freq"] = (counts / max(total, 1)).tolist()

        payload["clusters"][str(cluster_id)] = {
            "summary": summary,
            "per_run": cluster_rows,
        }
        print(f"cluster={cluster_id} summary={json.dumps(summary['overall'])}", flush=True)

    cluster_means = {
        c: v["summary"]["overall"]["mean"] for c, v in payload["clusters"].items()
    }
    payload["summary"] = {
        "cluster_overall_mean": cluster_means,
        "best_cluster": max(cluster_means, key=cluster_means.get) if cluster_means else None,
        "worst_cluster": min(cluster_means, key=cluster_means.get) if cluster_means else None,
    }
    print(json.dumps(payload["summary"], indent=2), flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
