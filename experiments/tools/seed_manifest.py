#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import jax


def key_list(key):
    return [int(x) for x in key.tolist()]


def make_manifest(seed, num_seeds, num_iterations):
    root = jax.random.PRNGKey(seed)
    manifest = {
        'seed': seed,
        'num_seeds': num_seeds,
        'num_iterations': num_iterations,
        'single_run_keys': [key_list(k) for k in jax.random.split(root, num_seeds)],
        'state_aug': [],
    }
    key = root
    key, init_subkey = jax.random.split(key)
    manifest['state_aug_init_policy_keys'] = [key_list(k) for k in jax.random.split(init_subkey, num_seeds)]
    for i in range(num_iterations):
        key, collect_subkey = jax.random.split(key)
        key, train_subkey = jax.random.split(key)
        manifest['state_aug'].append({
            'iteration': i,
            'collect_subkey': key_list(collect_subkey),
            'train_subkey': key_list(train_subkey),
            'train_keys': [key_list(k) for k in jax.random.split(train_subkey, num_seeds)],
        })
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-seeds', type=int, default=10)
    parser.add_argument('--num-iterations', type=int, default=0)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    text = json.dumps(make_manifest(args.seed, args.num_seeds, args.num_iterations), indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
        print(args.out)
    else:
        print(text)


if __name__ == '__main__':
    main()
