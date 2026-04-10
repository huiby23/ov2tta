import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl

from overcooked_v2_experiments.ttappo_v2_temporal.models.model import (
    get_actor_critic,
    initialize_carry,
)
from overcooked_v2_experiments.ttappo_v2_temporal.utils.store import load_all_checkpoints


def _add_dims(obs, done):
    obs = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], obs)
    done = jnp.array([[done]])
    return obs, done


def diagnose_run(run_dir: Path, num_episodes: int, seed: int):
    all_params, config = load_all_checkpoints(run_dir, final_only=True)
    env = jaxmarl.make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
    network = get_actor_critic(config)
    action_dim = env.action_space().n

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    @jax.jit
    def _policy_step(params, hstate, obs, done, key):
        next_hstate, pi, _, aux = network.apply(params, hstate, (obs, done))
        action = pi.sample(seed=key)
        pred = jnp.argmax(aux["partner_logits"], axis=-1)
        return next_hstate, action, pred

    rows = []
    aggregate_confusion = jnp.zeros((action_dim, action_dim), dtype=jnp.int32)

    for run_key, checkpoints in sorted(all_params.items()):
        params = checkpoints["ckpt_final"].params
        key = jax.random.PRNGKey(seed + int(run_key.split("_")[1]))
        correct = 0
        total = 0
        confusion = jnp.zeros((action_dim, action_dim), dtype=jnp.int32)

        for _ in range(num_episodes):
            key, key_reset = jax.random.split(key)
            obs, env_state = reset_fn(key_reset)
            done = {agent: False for agent in env.agents}
            done["__all__"] = False
            hstate = {agent: initialize_carry(config, 1) for agent in env.agents}

            while not bool(jax.device_get(done["__all__"])):
                step_actions = {}
                step_preds = {}

                for agent in env.agents:
                    key, key_action = jax.random.split(key)
                    agent_obs, agent_done = _add_dims(obs[agent], done[agent])
                    next_hstate, action, pred = _policy_step(
                        params,
                        hstate[agent],
                        agent_obs,
                        agent_done,
                        key_action,
                    )
                    hstate[agent] = next_hstate
                    step_actions[agent] = int(jax.device_get(action[0, 0]))
                    step_preds[agent] = int(jax.device_get(pred[0, 0]))

                for agent_idx, agent in enumerate(env.agents):
                    partner = env.agents[1 - agent_idx]
                    target = step_actions[partner]
                    pred = step_preds[agent]
                    total += 1
                    correct += int(pred == target)
                    confusion = confusion.at[target, pred].add(1)

                key, key_step = jax.random.split(key)
                obs, env_state, _, done, _ = step_fn(key_step, env_state, step_actions)

        acc = correct / max(total, 1)
        rows.append(
            {
                "run": run_key,
                "partner_pred_acc": acc,
                "correct": correct,
                "total": total,
                "num_episodes": num_episodes,
            }
        )
        aggregate_confusion = aggregate_confusion + confusion

        run_csv = run_dir / run_key / "partner_confusion_ckpt_final.csv"
        with open(run_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target_action", "pred_action", "count"])
            for t in range(action_dim):
                for p in range(action_dim):
                    writer.writerow([t, p, int(confusion[t, p])])

    summary_csv = run_dir / "partner_diagnostics_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "partner_pred_acc", "correct", "total", "num_episodes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    agg_csv = run_dir / "partner_confusion_aggregate.csv"
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_action", "pred_action", "count"])
        for t in range(action_dim):
            for p in range(action_dim):
                writer.writerow([t, p, int(aggregate_confusion[t, p])])

    mean_acc = sum(r["partner_pred_acc"] for r in rows) / max(len(rows), 1)
    print(f"partner_diagnostics_summary={summary_csv}")
    print(f"partner_confusion_aggregate={agg_csv}")
    print(f"mean_partner_pred_acc={mean_acc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    diagnose_run(Path(args.d), args.num_episodes, args.seed)
