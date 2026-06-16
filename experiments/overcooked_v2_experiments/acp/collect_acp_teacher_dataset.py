from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout_with_observations


def parse_args():
    parser = argparse.ArgumentParser(description='Collect ACP teacher trajectories from oracle teacher-partner pairs.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-episodes', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--max-partners', type=int, default=None)
    return parser.parse_args()


def _load_policies(run_dir: Path, backend: str, greedy: bool):
    if backend == 'ppo':
        from overcooked_v2_experiments.ppo.policy import PPOPolicy as PolicyCls
        from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints
    elif backend == 'mappo':
        from overcooked_v2_experiments.mappo.policy import MAPPOPolicy as PolicyCls
        from overcooked_v2_experiments.mappo.utils.store import load_all_checkpoints
    else:
        raise ValueError('Unsupported backend: %s' % backend)
    all_params, config = load_all_checkpoints(run_dir, final_only=True)
    run_keys = sorted(all_params.keys(), key=lambda x: int(x.split('_')[1]))
    policies = [PolicyCls(all_params[k]['ckpt_final'].params, config, stochastic=not greedy) for k in run_keys]
    return policies, config


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    policies, config = _load_policies(Path(manifest['source_run_dir']), manifest['backend'], args.greedy)
    env_kwargs = dict(config['env']['ENV_KWARGS'])
    env = OvercookedV2(**env_kwargs)
    partners = list(manifest['partners'])
    if args.max_partners is not None:
        partners = partners[: args.max_partners]
    if not partners:
        raise RuntimeError('Manifest contains no partners.')

    obs_rows, done_rows, partner_action_rows, teacher_action_rows = [], [], [], []
    teacher_id_rows, partner_id_rows, reward_rows = [], [], []
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_episodes)
    for ep in range(args.num_episodes):
        item = partners[ep % len(partners)]
        teacher_idx = int(item['teacher_run'])
        partner_idx = int(item['partner_run'])
        if teacher_idx >= len(policies) or partner_idx >= len(policies):
            raise IndexError('Teacher/partner index exceeds loaded policy count.')
        rollout = get_rollout_with_observations(PolicyPairing(policies[teacher_idx], policies[partner_idx]), env, keys[ep])
        obs_rows.append(np.asarray(jax.device_get(rollout.obs_seq['agent_0']), dtype=np.float32))
        done_rows.append(np.asarray(jax.device_get(rollout.done_seq['agent_0']), dtype=np.bool_))
        teacher_action_rows.append(np.asarray(jax.device_get(rollout.actions_seq['agent_0']), dtype=np.int32))
        partner_action_rows.append(np.asarray(jax.device_get(rollout.actions_seq['agent_1']), dtype=np.int32))
        teacher_id_rows.append(teacher_idx)
        partner_id_rows.append(partner_idx)
        reward_rows.append(float(jax.device_get(rollout.total_reward)))
        print('episode %d/%d teacher=%d partner=%d reward=%.3f' % (ep + 1, args.num_episodes, teacher_idx, partner_idx, reward_rows[-1]), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=np.stack(obs_rows, axis=0),
        done=np.stack(done_rows, axis=0),
        partner_action=np.stack(partner_action_rows, axis=0),
        teacher_action=np.stack(teacher_action_rows, axis=0),
        teacher_id=np.asarray(teacher_id_rows, dtype=np.int32),
        partner_id=np.asarray(partner_id_rows, dtype=np.int32),
        reward=np.asarray(reward_rows, dtype=np.float32),
        manifest_json=json.dumps(manifest),
        config_json=json.dumps(config, default=str),
    )
    print(json.dumps({'output': str(out), 'episodes': args.num_episodes, 'mean_reward': float(np.mean(reward_rows)), 'obs_shape': list(np.stack(obs_rows).shape)}, indent=2))


if __name__ == '__main__':
    main()
