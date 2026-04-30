#!/usr/bin/env python3
"""Fixed-batch CE overfit check for the E3T partner-action branch."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
E3T_DIR = ROOT / "experiments" / "overcooked_v2_experiments" / "e3t_state_aug_official"
for path in (ROOT / "experiments", E3T_DIR, ROOT / "JaxMARL"):
    sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax

from overcooked_v2_experiments.e3t_state_aug_official.models.model import get_actor_critic
from overcooked_v2_experiments.e3t_state_aug_official.ippo import (
    CONTEXT_PARAM_MARKERS,
    _bootstrap_history,
    _mask_tree_to_markers,
    _merge_train_and_partner_behavior,
    _network_input,
    _partner_action_from_flat,
    _restore_tree_to_markers,
    _update_history_buffers,
)


def build_config(args):
    model_config = {
        "TYPE": "CNN",
        "FC_DIM_SIZE": 64,
        "ACTIVATION": "relu",
        "ARCH": "cnn",
        "CNN_FEATURES": 32,
        "USE_HISTORY_CONTEXT": True,
        "CONTEXT_LENGTH": args.context_length,
        "LATENT_DIM": 64,
        "TRAIN_MODE": "MLP",
        "E3T_NUM_HIDDEN_LAYERS": 3,
        "E3T_SIZE_HIDDEN_LAYERS": 64,
        "E3T_NUM_FILTERS": 25,
        "E3T_NUM_CONV_LAYERS": 3,
        "E3T_CONTEXT_MODE": False,
        "E3T_ACTOR_CONDITION": args.actor_condition,
        "STAY_ACTION": 4,
        "ACTION_DIM": 6,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
    }
    return {
        "model": model_config,
        "env": {"ENV_NAME": "overcooked_v2", "ENV_KWARGS": {"layout": args.layout}},
    }


def collect_fixed_batch(args, env, network, params, config):
    model_config = config["model"]
    num_envs = args.num_envs
    num_agents = env.num_agents
    num_actors = num_envs * num_agents
    action_dim = model_config["ACTION_DIM"]
    context_length = model_config["CONTEXT_LENGTH"]
    stay_action = model_config["STAY_ACTION"]

    key = jax.random.PRNGKey(args.seed)
    key, k_reset = jax.random.split(key)
    obs, env_state = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
    obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
        (-1, *env.observation_space().shape)
    )
    history_obs, history_actions = _bootstrap_history(
        obs_batch, context_length, stay_action
    )
    last_done = jnp.zeros((num_actors,), dtype=jnp.bool_)

    train_idxs = jnp.linspace(0, num_agents, num_envs, dtype=jnp.int32, endpoint=False)
    train_mask_flat = jnp.stack(
        [train_idxs == i for i, _agent in enumerate(env.agents)]
    ).reshape((num_actors,))
    partner_mask_flat = jnp.logical_not(train_mask_flat)
    train_actor_indices = jnp.nonzero(train_mask_flat, size=num_envs, fill_value=0)[0]

    records = []
    for _ in range(args.num_steps):
        key, k_action, k_context, k_step = jax.random.split(key, 4)
        context_human = None
        if args.partner_source == "human_branch":
            context_human = jax.random.normal(k_context, (1, num_actors, action_dim))

        ac_in = _network_input(
            obs_batch[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
            history_obs[jnp.newaxis, :],
            history_actions[jnp.newaxis, :],
            True,
            context_human,
        )
        _, pi, _value, _partner_pi, human_pi = network.apply(params, None, ac_in)
        behavior_pi = pi
        if args.partner_source == "human_branch":
            behavior_pi = _merge_train_and_partner_behavior(
                pi, human_pi, partner_mask_flat, args.rand
            )

        action = behavior_pi.sample(seed=k_action).squeeze()
        partner_action = _partner_action_from_flat(action, num_envs, num_agents)
        records.append(
            (
                obs_batch[train_actor_indices],
                history_obs[train_actor_indices],
                history_actions[train_actor_indices],
                partner_action[train_actor_indices],
                last_done[train_actor_indices],
            )
        )

        action_by_agent = action.reshape((num_agents, num_envs))
        env_act = {a: action_by_agent[i] for i, a in enumerate(env.agents)}
        obs, env_state, _reward, done, _info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            jax.random.split(k_step, num_envs), env_state, env_act
        )
        done_batch = jnp.stack([done[a] for a in env.agents]).reshape((num_actors,))
        new_obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            (-1, *env.observation_space().shape)
        )
        history_obs, history_actions = _update_history_buffers(
            history_obs,
            history_actions,
            obs_batch,
            partner_action,
            jnp.tile(done["__all__"], num_agents),
            context_length,
            stay_action,
        )
        obs_batch = new_obs_batch
        last_done = done_batch

    def flatten(xs, dtype=None):
        x = jnp.stack(xs)
        x = x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        return x.astype(dtype) if dtype is not None else x

    return {
        "obs": flatten([r[0] for r in records], jnp.float32),
        "history_obs": flatten([r[1] for r in records], jnp.float32),
        "history_actions": flatten([r[2] for r in records], jnp.int32),
        "labels": flatten([r[3] for r in records], jnp.int32),
        "done": flatten([r[4] for r in records], jnp.bool_),
    }


def exact_input_bayes_acc(batch):
    groups = defaultdict(Counter)
    obs = np.asarray(batch["obs"])
    hobs = np.asarray(batch["history_obs"])
    hact = np.asarray(batch["history_actions"])
    labels = np.asarray(batch["labels"])
    for i, label in enumerate(labels):
        groups[(obs[i].tobytes(), hobs[i].tobytes(), hact[i].tobytes())][int(label)] += 1
    total = len(labels)
    conflicts = sum(1 for counts in groups.values() if len(counts) > 1)
    bayes = sum(max(counts.values()) for counts in groups.values()) / max(total, 1)
    return len(groups), conflicts, bayes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rand", type=float, default=0.7)
    parser.add_argument("--partner-source", choices=("human_branch", "main_policy"), default="human_branch")
    parser.add_argument("--actor-condition", choices=("predicted_partner", "constant"), default="predicted_partner")
    args = parser.parse_args()

    config = build_config(args)
    env = jaxmarl.make("overcooked_v2", layout=args.layout)
    network = get_actor_critic(config)
    model_config = config["model"]

    key = jax.random.PRNGKey(args.seed)
    init_obs = jnp.zeros((1, args.num_envs, *env.observation_space().shape), dtype=jnp.float32)
    init_done = jnp.zeros((1, args.num_envs), dtype=jnp.bool_)
    init_x = (
        init_obs,
        init_done,
        jnp.zeros(
            (1, args.num_envs, args.context_length, *env.observation_space().shape),
            dtype=jnp.float32,
        ),
        jnp.full((1, args.num_envs, args.context_length), model_config["STAY_ACTION"], dtype=jnp.int32),
    )
    params = network.init(key, None, init_x)
    batch = collect_fixed_batch(args, env, network, params, config)

    print("dataset", {k: tuple(v.shape) for k, v in batch.items()})
    print("label_counts", dict(Counter(int(x) for x in np.asarray(batch["labels"]))))
    unique, conflicts, bayes = exact_input_bayes_acc(batch)
    print("exact_input", {"unique": unique, "conflicts": conflicts, "bayes_acc": bayes})

    def eval_loss_acc(p):
        _, _, _, partner_pi, _ = network.apply(
            p,
            None,
            (batch["obs"], batch["done"], batch["history_obs"], batch["history_actions"]),
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            partner_pi.logits, batch["labels"]
        ).mean()
        acc = (jnp.argmax(partner_pi.logits, axis=-1) == batch["labels"]).mean()
        return loss, acc

    opt = optax.adam(args.lr, eps=1e-5)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(p, state):
        def loss_fn(pp):
            loss, acc = eval_loss_acc(pp)
            return loss, acc

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        grads = _mask_tree_to_markers(grads, CONTEXT_PARAM_MARKERS)
        updates, state = opt.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        new_p = _restore_tree_to_markers(new_p, p, CONTEXT_PARAM_MARKERS)
        return new_p, state, loss, acc

    loss, acc = eval_loss_acc(params)
    print("before", {"loss": float(loss), "acc": float(acc)})
    checkpoints = {0, 10, 50, 100, 200, args.train_steps}
    for step in range(args.train_steps + 1):
        params, opt_state, loss, acc = train_step(params, opt_state)
        if step in checkpoints:
            print("step", step, {"loss": float(loss), "acc": float(acc)})
    loss, acc = eval_loss_acc(params)
    print("after", {"loss": float(loss), "acc": float(acc)})


if __name__ == "__main__":
    main()
