import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl
import optax

from overcooked_v2_experiments.ttappo_v2_1.models.model import (
    get_actor_critic,
    initialize_carry,
)
from overcooked_v2_experiments.ttappo_v2_1.utils.store import load_all_checkpoints


def _add_dims(obs, done):
    obs = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], obs)
    done = jnp.array([[done]])
    return obs, done


def _stack_agent_batch(obs_dict, done_dict, agents):
    obs_batch = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0)[jnp.newaxis, ...],
        *(obs_dict[agent] for agent in agents),
    )
    done_batch = jnp.array([[done_dict[agent] for agent in agents]])
    return obs_batch, done_batch


def _select_run_items(all_params, runs_csv):
    items = sorted(all_params.items())
    if not runs_csv:
        return items

    selected = {token.strip() for token in runs_csv.split(",") if token.strip()}
    return [(run_key, checkpoints) for run_key, checkpoints in items if run_key in selected]


def diagnose_run(run_dir: Path, num_episodes: int, seed: int, runs_csv: str = "", output_suffix: str = ""):
    all_params, config = load_all_checkpoints(run_dir, final_only=True)
    env = jaxmarl.make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
    network = get_actor_critic(config)
    action_dim = env.action_space().n
    run_items = _select_run_items(all_params, runs_csv)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    @jax.jit
    def _policy_step(params, hstate, obs_batch, done_batch, key):
        next_hstate, pi, _, aux = network.apply(params, hstate, (obs_batch, done_batch))
        action = pi.sample(seed=key)
        pred = jnp.argmax(aux["partner_logits"], axis=-1)
        gate_values = aux.get("partner_gate", jnp.zeros_like(aux["feature_z"]))
        gate_mean = gate_values.mean(axis=-1)
        return next_hstate, action, pred, aux["partner_logits"], gate_mean

    rows = []
    aggregate_confusion = jnp.zeros((action_dim, action_dim), dtype=jnp.int32)
    aggregate_correct = jnp.zeros((action_dim,), dtype=jnp.int32)
    aggregate_total = jnp.zeros((action_dim,), dtype=jnp.int32)
    agent_partners = {
        agent: env.agents[1 - idx] for idx, agent in enumerate(env.agents)
    }

    for run_key, checkpoints in run_items:
        params = checkpoints["ckpt_final"].params
        key = jax.random.PRNGKey(seed + int(run_key.split("_")[1]))
        correct = jnp.array(0, dtype=jnp.int32)
        total = jnp.array(0, dtype=jnp.int32)
        total_gate = jnp.array(0.0, dtype=jnp.float32)
        total_loss = jnp.array(0.0, dtype=jnp.float32)
        confusion = jnp.zeros((action_dim, action_dim), dtype=jnp.int32)
        per_action_correct = jnp.zeros((action_dim,), dtype=jnp.int32)
        per_action_total = jnp.zeros((action_dim,), dtype=jnp.int32)

        for _ in range(num_episodes):
            key, key_reset = jax.random.split(key)
            obs, env_state = reset_fn(key_reset)
            done = {agent: False for agent in env.agents}
            done["__all__"] = False
            hstate = {agent: initialize_carry(config, 1) for agent in env.agents}

            while not bool(jax.device_get(done["__all__"])):
                key, key_action = jax.random.split(key)
                obs_batch, done_batch = _stack_agent_batch(obs, done, env.agents)
                hidden_batch = jnp.concatenate(
                    [hstate[agent] for agent in env.agents],
                    axis=0,
                )
                next_hidden_batch, action_batch, pred_batch, logits_batch, gate_mean_batch = _policy_step(
                    params,
                    hidden_batch,
                    obs_batch,
                    done_batch,
                    key_action,
                )

                step_actions = {}
                step_preds = {}
                step_gate_means = {}
                for idx, agent in enumerate(env.agents):
                    hstate[agent] = next_hidden_batch[idx : idx + 1]
                    step_actions[agent] = action_batch[0, idx]
                    step_preds[agent] = pred_batch[0, idx]
                    step_gate_means[agent] = gate_mean_batch[0, idx].astype(jnp.float32)

                for agent in env.agents:
                    partner = agent_partners[agent]
                    target = step_actions[partner]
                    pred = step_preds[agent]
                    match = (pred == target).astype(jnp.int32)
                    total = total + jnp.array(1, dtype=jnp.int32)
                    correct = correct + match
                    confusion = confusion.at[target, pred].add(1)
                    per_action_total = per_action_total.at[target].add(1)
                    per_action_correct = per_action_correct.at[target].add(match)
                    total_gate = total_gate + step_gate_means[agent]

                target_batch = jnp.stack(
                    [step_actions[agent_partners[agent]] for agent in env.agents],
                    axis=0,
                ).astype(jnp.int32)
                total_loss = total_loss + jnp.sum(
                    optax.softmax_cross_entropy_with_integer_labels(
                        logits_batch[0], target_batch
                    )
                )

                key, key_step = jax.random.split(key)
                obs, env_state, _, done, _ = step_fn(key_step, env_state, step_actions)

        correct_host = int(jax.device_get(correct))
        total_host = int(jax.device_get(total))
        total_gate_host = float(jax.device_get(total_gate))
        total_loss_host = float(jax.device_get(total_loss))
        confusion_host = jax.device_get(confusion)
        per_action_correct_host = jax.device_get(per_action_correct)
        per_action_total_host = jax.device_get(per_action_total)

        acc = correct_host / max(total_host, 1)
        rows.append(
            {
                "run": run_key,
                "partner_pred_acc": acc,
                "partner_pred_loss": total_loss_host / max(total_host, 1),
                "partner_gate_mean": total_gate_host / max(total_host, 1),
                "correct": correct_host,
                "total": total_host,
                "num_episodes": num_episodes,
            }
        )
        aggregate_confusion = aggregate_confusion + confusion
        aggregate_correct = aggregate_correct + per_action_correct
        aggregate_total = aggregate_total + per_action_total

        run_csv = run_dir / run_key / "partner_confusion_ckpt_final.csv"
        with open(run_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target_action", "pred_action", "count"])
            for t in range(action_dim):
                for p in range(action_dim):
                    writer.writerow([t, p, int(confusion_host[t, p])])

    suffix = f"_{output_suffix}" if output_suffix else ""
    summary_csv = run_dir / f"partner_diagnostics_summary{suffix}.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "partner_pred_acc",
                "partner_pred_loss",
                "partner_gate_mean",
                "correct",
                "total",
                "num_episodes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    agg_csv = run_dir / f"partner_confusion_aggregate{suffix}.csv"
    aggregate_confusion_host = jax.device_get(aggregate_confusion)
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_action", "pred_action", "count"])
        for t in range(action_dim):
            for p in range(action_dim):
                writer.writerow([t, p, int(aggregate_confusion_host[t, p])])

    per_action_csv = run_dir / f"partner_per_action_accuracy{suffix}.csv"
    aggregate_correct_host = jax.device_get(aggregate_correct)
    aggregate_total_host = jax.device_get(aggregate_total)
    with open(per_action_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_action", "correct", "total", "accuracy"])
        for t in range(action_dim):
            total_t = int(aggregate_total_host[t])
            correct_t = int(aggregate_correct_host[t])
            acc_t = correct_t / max(total_t, 1)
            writer.writerow([t, correct_t, total_t, acc_t])

    mean_acc = sum(r["partner_pred_acc"] for r in rows) / max(len(rows), 1)
    mean_loss = sum(r["partner_pred_loss"] for r in rows) / max(len(rows), 1)
    mean_gate = sum(r["partner_gate_mean"] for r in rows) / max(len(rows), 1)
    print(f"partner_diagnostics_summary={summary_csv}")
    print(f"partner_confusion_aggregate={agg_csv}")
    print(f"partner_per_action_accuracy={per_action_csv}")
    print(f"mean_partner_pred_acc={mean_acc:.6f}")
    print(f"mean_partner_pred_loss={mean_loss:.6f}")
    print(f"mean_partner_gate={mean_gate:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs_csv", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    diagnose_run(
        Path(args.d),
        args.num_episodes,
        args.seed,
        runs_csv=args.runs_csv,
        output_suffix=args.output_suffix,
    )
