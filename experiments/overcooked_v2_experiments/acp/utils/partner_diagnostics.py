import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl

from overcooked_v2_experiments.acp.models.model import (
    get_actor_critic,
    initialize_carry,
)
from overcooked_v2_experiments.acp.policy import (
    EVAL_MODES,
    _readout_scale,
    _resolve_memory_action,
)
from overcooked_v2_experiments.acp.utils.store import load_all_checkpoints


def _parse_runs_csv(run_dir: Path, runs_csv: str | None):
    if not runs_csv:
        return sorted(
            p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
        )
    requested = {token.strip() for token in runs_csv.split(",") if token.strip()}
    return sorted(name for name in requested if (run_dir / name).is_dir())


def diagnose_run(
    run_dir: Path,
    num_episodes: int,
    seed: int,
    mode: str,
    runs_csv: str | None = None,
    output_suffix: str | None = None,
    ce_threshold_override: float | None = None,
):
    all_params, config = load_all_checkpoints(run_dir, final_only=True)
    if ce_threshold_override is not None:
        config["model"]["STATE_ADAPT_CE_THRESHOLD"] = float(ce_threshold_override)
    env = jaxmarl.make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
    network = get_actor_critic(config)
    action_dim = env.action_space().n
    gate_threshold = float(config["model"].get("STATE_ADAPT_CE_THRESHOLD", 0.8))

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    @jax.jit
    def _policy_step(params, hstate, obs_batch, done_batch, key):
        next_hstate, pi, _, aux = network.apply(
            params,
            hstate,
            (
                obs_batch[jnp.newaxis, ...],
                done_batch[jnp.newaxis, ...],
                _readout_scale(mode)[jnp.newaxis, ...],
            ),
        )
        action = pi.sample(seed=key)[0]
        partner_pred = jnp.argmax(aux["partner_logits"][0], axis=-1)
        return next_hstate, action, aux, partner_pred

    @jax.jit
    def _update_memory(
        params,
        hstate,
        aux,
        action,
        partner_action,
        prev_partner_action,
        done_batch,
        key,
    ):
        if mode == "memory_off":
            update_mask = jnp.zeros_like(done_batch, dtype=jnp.bool_)
        elif mode == "state_adapt_gated":
            log_probs = jax.nn.log_softmax(aux["partner_logits"][0], axis=-1)
            ce = -jnp.take_along_axis(
                log_probs,
                partner_action[:, None],
                axis=-1,
            ).squeeze(-1)
            update_mask = ce > gate_threshold
        else:
            update_mask = jnp.ones_like(done_batch, dtype=jnp.bool_)
        memory_action, next_prev_partner_action = _resolve_memory_action(
            mode,
            partner_action.astype(jnp.int32),
            action.astype(jnp.int32),
            prev_partner_action.astype(jnp.int32),
            key,
            action_dim,
        )

        next_hstate = network.apply(
            params,
            hstate,
            aux["feature_z"][0],
            aux["temporal_feature"][0],
            memory_action,
            done_batch.astype(jnp.bool_),
            update_mask.astype(jnp.bool_),
            method=network.update_memory_state,
        )
        return (
            next_hstate,
            update_mask.astype(jnp.float32),
            memory_action,
            next_prev_partner_action,
        )

    rows = []
    aggregate_confusion = jnp.zeros((action_dim, action_dim), dtype=jnp.int32)
    per_action_correct = jnp.zeros((action_dim,), dtype=jnp.int32)
    per_action_total = jnp.zeros((action_dim,), dtype=jnp.int32)

    run_keys = _parse_runs_csv(run_dir, runs_csv)
    for run_key in run_keys:
        checkpoints = all_params[run_key]
        params = checkpoints["ckpt_final"].params
        key = jax.random.PRNGKey(seed + int(run_key.split("_")[1]))
        correct = 0
        total = 0
        pred_loss_sum = 0.0
        gate_sum = 0.0
        memory_norm_sum = 0.0
        gamma_sum = 0.0
        beta_sum = 0.0
        gamma_abs_sum = 0.0
        beta_abs_sum = 0.0
        feature_mod_delta_norm_sum = 0.0
        partner_logit_ce_sum = 0.0
        early_reward_sum = 0.0
        late_reward_sum = 0.0
        early_steps = 0
        late_steps = 0
        confusion = jnp.zeros((action_dim, action_dim), dtype=jnp.int32)

        for _ in range(num_episodes):
            key, key_reset = jax.random.split(key)
            obs, env_state = reset_fn(key_reset)
            done = {agent: False for agent in env.agents}
            done["__all__"] = False
            hstate = initialize_carry(config, env.num_agents)
            prev_partner_action = jnp.zeros((env.num_agents,), dtype=jnp.int32)
            step_idx = 0

            while not bool(jax.device_get(done["__all__"])):
                obs_batch = jnp.stack([obs[a] for a in env.agents], axis=0)
                done_batch = jnp.array([done[a] for a in env.agents], dtype=jnp.bool_)
                key, key_action = jax.random.split(key)
                next_hstate, action, aux, pred = _policy_step(
                    params,
                    hstate,
                    obs_batch,
                    done_batch,
                    key_action,
                )

                env_actions = {
                    agent: action[i]
                    for i, agent in enumerate(env.agents)
                }
                key, key_step = jax.random.split(key)
                obs, env_state, reward, done, _ = step_fn(key_step, env_state, env_actions)

                partner_action = action[::-1]
                log_probs = jax.nn.log_softmax(aux["partner_logits"][0], axis=-1)
                step_loss = -jnp.take_along_axis(
                    log_probs,
                    partner_action[:, None],
                    axis=-1,
                ).squeeze(-1)

                next_done_batch = jnp.array(
                    [done[a] for a in env.agents],
                    dtype=jnp.bool_,
                )
                key, key_update = jax.random.split(key)
                hstate, update_rate, _, prev_partner_action = _update_memory(
                    params,
                    next_hstate,
                    aux,
                    action,
                    partner_action,
                    prev_partner_action,
                    next_done_batch,
                    key_update,
                )

                for agent_idx in range(env.num_agents):
                    target = int(partner_action[agent_idx])
                    pred_action = int(pred[agent_idx])
                    total += 1
                    correct += int(pred_action == target)
                    confusion = confusion.at[target, pred_action].add(1)
                    per_action_total = per_action_total.at[target].add(1)
                    per_action_correct = per_action_correct.at[target].add(
                        int(pred_action == target)
                    )

                pred_loss_sum += float(step_loss.mean())
                gate_sum += float(update_rate.mean())
                memory_norm_sum += float(
                    jnp.linalg.norm(aux["partner_memory"][0], axis=-1).mean()
                )
                gamma_sum += float(aux["film_gamma"][0].mean())
                beta_sum += float(aux["film_beta"][0].mean())
                gamma_abs_sum += float(jnp.abs(aux["film_gamma"][0]).mean())
                beta_abs_sum += float(jnp.abs(aux["film_beta"][0]).mean())
                feature_mod_delta_norm_sum += float(
                    jnp.linalg.norm(
                        aux["feature_mod"][0] - aux["feature_z"][0], axis=-1
                    ).mean()
                )
                partner_logit_ce_sum += float(step_loss.mean())

                reward_scalar = float(reward["agent_0"])
                if step_idx < env.max_steps // 2:
                    early_reward_sum += reward_scalar
                    early_steps += 1
                else:
                    late_reward_sum += reward_scalar
                    late_steps += 1
                step_idx += 1

        mean_acc = correct / max(total, 1)
        mean_loss = pred_loss_sum / max(total // env.num_agents, 1)
        mean_gate = gate_sum / max(total // env.num_agents, 1)
        memory_norm_mean = memory_norm_sum / max(total // env.num_agents, 1)
        gamma_mean = gamma_sum / max(total // env.num_agents, 1)
        beta_mean = beta_sum / max(total // env.num_agents, 1)
        gamma_abs_mean = gamma_abs_sum / max(total // env.num_agents, 1)
        beta_abs_mean = beta_abs_sum / max(total // env.num_agents, 1)
        feature_mod_delta_norm = feature_mod_delta_norm_sum / max(total // env.num_agents, 1)
        partner_logit_ce_mean = partner_logit_ce_sum / max(total // env.num_agents, 1)
        early_reward = early_reward_sum / max(early_steps, 1)
        late_reward = late_reward_sum / max(late_steps, 1)

        rows.append(
            {
                "run": run_key,
                "partner_pred_acc": mean_acc,
                "partner_pred_loss": mean_loss,
                "memory_update_rate": mean_gate,
                "memory_norm_mean": memory_norm_mean,
                "gamma_mean": gamma_mean,
                "beta_mean": beta_mean,
                "gamma_abs_mean": gamma_abs_mean,
                "beta_abs_mean": beta_abs_mean,
                "feature_mod_delta_norm": feature_mod_delta_norm,
                "partner_logit_ce_mean": partner_logit_ce_mean,
                "early_reward": early_reward,
                "late_reward": late_reward,
                "late_minus_early": late_reward - early_reward,
                "late_minus_early_xp_proxy": late_reward - early_reward,
                "correct": correct,
                "total": total,
                "num_episodes": num_episodes,
            }
        )
        aggregate_confusion = aggregate_confusion + confusion

    suffix = f"_{output_suffix}" if output_suffix else ""
    summary_csv = run_dir / f"partner_diagnostics_summary{suffix}.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "partner_pred_acc",
                "partner_pred_loss",
                "memory_update_rate",
                "memory_norm_mean",
                "gamma_mean",
                "beta_mean",
                "gamma_abs_mean",
                "beta_abs_mean",
                "feature_mod_delta_norm",
                "partner_logit_ce_mean",
                "early_reward",
                "late_reward",
                "late_minus_early",
                "late_minus_early_xp_proxy",
                "correct",
                "total",
                "num_episodes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    agg_csv = run_dir / f"partner_confusion_aggregate{suffix}.csv"
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_action", "pred_action", "count"])
        for t in range(action_dim):
            for p in range(action_dim):
                writer.writerow([t, p, int(aggregate_confusion[t, p])])

    per_action_csv = run_dir / f"partner_per_action_accuracy{suffix}.csv"
    with open(per_action_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_action", "correct", "total", "accuracy"])
        for target in range(action_dim):
            correct_count = int(per_action_correct[target])
            total_count = int(per_action_total[target])
            accuracy = correct_count / max(total_count, 1)
            writer.writerow([target, correct_count, total_count, accuracy])

    mean_acc = sum(r["partner_pred_acc"] for r in rows) / max(len(rows), 1)
    print(f"partner_diagnostics_summary={summary_csv}")
    print(f"partner_confusion_aggregate={agg_csv}")
    print(f"partner_per_action_accuracy={per_action_csv}")
    print(f"mean_partner_pred_acc={mean_acc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="state_adapt",
        choices=list(EVAL_MODES),
    )
    parser.add_argument("--runs_csv", type=str)
    parser.add_argument("--output_suffix", type=str)
    parser.add_argument("--ce_threshold_override", type=float)
    args = parser.parse_args()

    diagnose_run(
        Path(args.d),
        args.num_episodes,
        args.seed,
        args.mode,
        runs_csv=args.runs_csv,
        output_suffix=args.output_suffix,
        ce_threshold_override=args.ce_threshold_override,
    )
