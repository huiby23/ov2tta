from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import jax
import numpy as np
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout_with_observations
from .dataset import split_and_save_npz


def parse_args():
    parser = argparse.ArgumentParser(description="Collect OV2 trajectories for GAMMA VAE training.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing run_i/ckpt_final checkpoints.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--cross", action="store_true", help="Collect cross-play pairs instead of only self-play.")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--backend", choices=["ppo", "mappo"], default="ppo")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--min-return",
        type=float,
        default=None,
        help="Keep only episodes with team return >= this value.",
    )
    parser.add_argument(
        "--top-episode-fraction",
        type=float,
        default=1.0,
        help="After min-return filtering, keep the top fraction by episode return.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum rollout attempts before filtering. Defaults to num-episodes, or 4x when filtering is enabled.",
    )
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
    return policies, config


def main():
    args = parse_args()
    if not (0.0 < args.top_episode_fraction <= 1.0):
        raise ValueError("--top-episode-fraction must be in (0, 1].")
    policies, config = _load_policies(Path(args.run_dir), args.backend, args.greedy)
    env_kwargs = dict(config["env"]["ENV_KWARGS"])
    layout = env_kwargs.pop("layout")
    env = OvercookedV2(layout=layout, **env_kwargs)
    self_pairs = [(i, i) for i in range(len(policies))]
    if args.cross:
        cross_pairs = list(itertools.permutations(range(len(policies)), 2))
        if args.max_pairs <= len(self_pairs):
            pairs = self_pairs[: args.max_pairs]
        else:
            # Keep self-play demonstrations even when max_pairs truncates the
            # pair list. TALENTS/GAMMA generated partners collapse if the VAE is
            # trained mostly on failed cross-play trajectories.
            pairs = self_pairs + cross_pairs[: args.max_pairs - len(self_pairs)]
    else:
        pairs = self_pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError("No policy pairs found for dataset collection.")
    obs_rows = []
    action_rows = []
    policy_rows = []
    return_rows = []
    uses_filtering = args.min_return is not None or args.top_episode_fraction < 1.0
    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = args.num_episodes * 4 if uses_filtering else args.num_episodes
    max_attempts = max(max_attempts, args.num_episodes)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), max_attempts)
    for attempt in range(max_attempts):
        i, j = pairs[attempt % len(pairs)]
        rollout = get_rollout_with_observations(PolicyPairing(policies[i], policies[j]), env, keys[attempt])
        episode_return = float(rollout.total_reward)
        obs = np.stack(
            [
                np.asarray(jax.device_get(rollout.obs_seq["agent_0"])),
                np.asarray(jax.device_get(rollout.obs_seq["agent_1"])),
            ],
            axis=1,
        )
        actions = np.stack(
            [
                np.asarray(jax.device_get(rollout.actions_seq["agent_0"])),
                np.asarray(jax.device_get(rollout.actions_seq["agent_1"])),
            ],
            axis=1,
        )[..., None]
        obs_rows.append(obs.astype(np.float32))
        action_rows.append(actions.astype(np.int32))
        policy_rows.append(np.asarray([i, j], dtype=np.int32))
        return_rows.append(episode_return)
        print(
            f"collected attempt {attempt + 1}/{max_attempts}: pair=({i},{j}), reward={episode_return:.3f}",
            flush=True,
        )
        if not uses_filtering and len(obs_rows) >= args.num_episodes:
            break

    returns = np.asarray(return_rows, dtype=np.float32)
    keep = np.arange(len(return_rows))
    if args.min_return is not None:
        keep = keep[returns[keep] >= args.min_return]
    if args.top_episode_fraction < 1.0 and keep.size > 0:
        top_n = max(1, int(np.ceil(keep.size * args.top_episode_fraction)))
        keep = keep[np.argsort(returns[keep])[::-1][:top_n]]
    if keep.size > args.num_episodes:
        keep = keep[np.argsort(returns[keep])[::-1][: args.num_episodes]]
    if keep.size == 0:
        raise RuntimeError(
            f"No trajectories survived filtering: min_return={args.min_return}, "
            f"top_episode_fraction={args.top_episode_fraction}."
        )
    keep = np.sort(keep)
    print(
        "dataset filter summary: "
        f"attempts={len(return_rows)}, kept={keep.size}, "
        f"mean_return={float(returns[keep].mean()):.3f}, "
        f"min_return={float(returns[keep].min()):.3f}, "
        f"max_return={float(returns[keep].max()):.3f}",
        flush=True,
    )

    out_path = split_and_save_npz(
        args.output,
        np.stack([obs_rows[i] for i in keep], axis=0),
        np.stack([action_rows[i] for i in keep], axis=0),
        np.stack([policy_rows[i] for i in keep], axis=0),
        episode_returns=returns[keep],
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(f"Saved GAMMA dataset to {out_path}")


if __name__ == "__main__":
    main()
