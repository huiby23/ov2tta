
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dataset import split_and_save_npz


def main():
    parser = argparse.ArgumentParser(description="Create a small synthetic dataset for GAMMA smoke tests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--channels", type=int, default=26)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    obs = rng.normal(size=(args.episodes, args.steps, 2, args.height, args.width, args.channels)).astype(np.float32)
    actions = rng.integers(0, args.action_dim, size=(args.episodes, args.steps, 2, 1), dtype=np.int32)
    policy_id = rng.integers(0, 4, size=(args.episodes, 2), dtype=np.int32)
    path = split_and_save_npz(Path(args.output), obs, actions, policy_id, validation_ratio=0.2, seed=args.seed)
    print(path)


if __name__ == "__main__":
    main()
