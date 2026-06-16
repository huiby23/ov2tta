from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils

from overcooked_v2_experiments.acp.models.model import get_actor_critic, initialize_carry


def parse_args():
    parser = argparse.ArgumentParser(description='Train ACP by distilling oracle convention teacher trajectories.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-seeds', type=int, default=10)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--chunk-length', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--convention-modes', type=int, default=10)
    parser.add_argument('--action-coef', type=float, default=1.0)
    parser.add_argument('--mode-coef', type=float, default=0.2)
    parser.add_argument('--partner-coef', type=float, default=0.05)
    parser.add_argument('--use-partner-history', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-every', type=int, default=0)
    return parser.parse_args()


def _base_config(config_json: str, args) -> dict:
    config = json.loads(config_json)
    config = copy.deepcopy(config)
    model = config.setdefault('model', {})
    model['TYPE'] = 'CNN'
    model['FC_DIM_SIZE'] = int(model.get('FC_DIM_SIZE', args.hidden_dim))
    model['ACTIVATION'] = model.get('ACTIVATION', 'relu')
    model['CNN_FEATURES'] = int(model.get('CNN_FEATURES', 32))
    model['TTT_ADAPTER_DIM'] = int(model.get('TTT_ADAPTER_DIM', 32))
    model['TEMPORAL_PARTNER_ENCODER'] = True
    model['TEMPORAL_HIDDEN_DIM'] = int(model.get('TEMPORAL_HIDDEN_DIM', model['FC_DIM_SIZE']))
    model['PARTNER_MEMORY_DIM'] = int(model.get('PARTNER_MEMORY_DIM', model['TEMPORAL_HIDDEN_DIM']))
    model['PARTNER_ACTION_EMBED_DIM'] = int(model.get('PARTNER_ACTION_EMBED_DIM', 16))
    model['FILM_SCALE'] = float(model.get('FILM_SCALE', 0.2))
    model['ACTOR_ADAPTER_DIM'] = int(model.get('ACTOR_ADAPTER_DIM', model['FC_DIM_SIZE']))
    model['ACTOR_ADAPTER_SCALE'] = float(model.get('ACTOR_ADAPTER_SCALE', 0.5))
    model['CONVENTION_NUM_MODES'] = int(args.convention_modes)
    model['USE_PARTNER_HISTORY'] = bool(args.use_partner_history)
    model['ACP_ACTION_DISTILL_COEF'] = float(args.action_coef)
    model['ACP_MODE_LOSS_COEF'] = float(args.mode_coef)
    model['ACP_PARTNER_LOSS_COEF'] = float(args.partner_coef)
    # Values below keep checkpoints compatible with PPO finetune/eval utilities.
    model.setdefault('NUM_ENVS', 64)
    model.setdefault('NUM_STEPS', 256)
    model.setdefault('NUM_MINIBATCHES', 16)
    model.setdefault('UPDATE_EPOCHS', 4)
    model.setdefault('TOTAL_TIMESTEPS', 10000000)
    model.setdefault('REW_SHAPING_HORIZON', 5000000)
    model.setdefault('LR', 4e-4)
    model.setdefault('ANNEAL_LR', True)
    model.setdefault('LR_WARMUP', 0.05)
    model.setdefault('GAMMA', 0.99)
    model.setdefault('GAE_LAMBDA', 0.95)
    model.setdefault('VF_COEF', 0.5)
    model.setdefault('MAX_GRAD_NORM', 0.5)
    model.setdefault('CLIP_EPS', 0.2)
    model.setdefault('ENT_COEF', 0.04)
    config['ACP_DISTILL'] = {
        'dataset': None,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'chunk_length': args.chunk_length,
        'lr': args.lr,
        'num_seeds': args.num_seeds,
        'use_partner_history': args.use_partner_history,
    }
    return config


def _sample_batch(data, rng_np, batch_size: int, chunk_length: int):
    obs = data['obs']
    n, t = obs.shape[:2]
    length = min(chunk_length, t)
    ep_idx = rng_np.integers(0, n, size=batch_size)
    starts = rng_np.integers(0, max(t - length + 1, 1), size=batch_size)
    batch = {}
    for key in ['obs', 'done', 'partner_action', 'teacher_action']:
        arr = data[key]
        chunks = [arr[e, s:s + length] for e, s in zip(ep_idx, starts)]
        batch[key] = np.stack(chunks, axis=1)  # [T, B, ...]
    teacher_id = data['teacher_id'][ep_idx]
    batch['teacher_id'] = np.broadcast_to(teacher_id[None, :], (length, batch_size)).astype(np.int32)
    return batch


def _save_checkpoint(path: Path, config: dict, params):
    path = path.resolve()
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {'config': config, 'params': params, 'ttt_source_stats': {'acp_distill': True}}
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(item)
    checkpointer.save(path, item, save_args=save_args)


def _train_one(seed: int, config: dict, data, args, run_dir: Path):
    rng_np = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    network = get_actor_critic(config)
    dummy = _sample_batch(data, rng_np, min(args.batch_size, data['obs'].shape[0]), args.chunk_length)
    batch_size = dummy['obs'].shape[1]
    init_h = initialize_carry(config, batch_size)
    key, init_key = jax.random.split(key)
    params = network.init(
        init_key,
        init_h,
        (
            jnp.asarray(dummy['obs']),
            jnp.asarray(dummy['done']),
            jnp.asarray(dummy['partner_action']),
            jnp.ones_like(jnp.asarray(dummy['partner_action']), dtype=jnp.bool_),
        ),
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr, eps=1e-5))
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        h = initialize_carry(config, batch['obs'].shape[1])
        def loss_fn(p):
            _, pi, _, aux = network.apply(
                p,
                h,
                (
                    batch['obs'],
                    batch['done'],
                    batch['partner_action'].astype(jnp.int32),
                    jnp.ones_like(batch['partner_action'], dtype=jnp.bool_),
                ),
            )
            action_loss = optax.softmax_cross_entropy_with_integer_labels(
                pi.logits, batch['teacher_action'].astype(jnp.int32)
            ).mean()
            partner_loss = optax.softmax_cross_entropy_with_integer_labels(
                aux['partner_logits'], batch['partner_action'].astype(jnp.int32)
            ).mean()
            mode_loss = optax.softmax_cross_entropy_with_integer_labels(
                aux['convention_logits'], batch['teacher_id'].astype(jnp.int32)
            ).mean()
            action_acc = (jnp.argmax(pi.logits, axis=-1) == batch['teacher_action']).mean()
            partner_acc = (jnp.argmax(aux['partner_logits'], axis=-1) == batch['partner_action']).mean()
            mode_acc = (jnp.argmax(aux['convention_logits'], axis=-1) == batch['teacher_id']).mean()
            loss = args.action_coef * action_loss + args.partner_coef * partner_loss + args.mode_coef * mode_loss
            metrics = {
                'loss': loss,
                'action_loss': action_loss,
                'partner_loss': partner_loss,
                'mode_loss': mode_loss,
                'action_acc': action_acc,
                'partner_acc': partner_acc,
                'mode_acc': mode_acc,
            }
            return loss, metrics
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    history = []
    for step in range(1, args.steps + 1):
        batch_np = _sample_batch(data, rng_np, args.batch_size, args.chunk_length)
        batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
        params, opt_state, metrics = train_step(params, opt_state, batch)
        if step == 1 or step == args.steps or step % max(args.steps // 20, 1) == 0:
            row = {'step': step, **{k: float(v) for k, v in jax.device_get(metrics).items()}}
            history.append(row)
            print('run=%s step=%d loss=%.4f action_acc=%.3f mode_acc=%.3f partner_acc=%.3f' % (run_dir.name, step, row['loss'], row['action_acc'], row['mode_acc'], row['partner_acc']), flush=True)
        if args.save_every and step % args.save_every == 0:
            _save_checkpoint(run_dir / ('ckpt_%d' % step), config, params)
    _save_checkpoint(run_dir / 'ckpt_final', config, params)
    with (run_dir / 'distill_history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return history[-1]


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.dataset, allow_pickle=True)
    data = {k: raw[k] for k in ['obs', 'done', 'partner_action', 'teacher_action', 'teacher_id', 'partner_id', 'reward']}
    config = _base_config(str(raw['config_json'].item()), args)
    config['ACP_DISTILL']['dataset'] = str(args.dataset)
    config['RUN_BASE_DIR'] = str(out)
    (out / 'acp_distill_config.json').write_text(json.dumps(config, indent=2, default=str))
    summaries = []
    for run_idx in range(args.num_seeds):
        run_dir = out / ('run_%d' % run_idx)
        run_dir.mkdir(parents=True, exist_ok=True)
        final_metrics = _train_one(args.seed + run_idx, config, data, args, run_dir)
        summaries.append({'run': run_idx, **final_metrics})
    with (out / 'acp_distill_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({'output_dir': str(out), 'num_runs': args.num_seeds, 'final_mean_loss': float(np.mean([x['loss'] for x in summaries]))}, indent=2))


if __name__ == '__main__':
    main()
