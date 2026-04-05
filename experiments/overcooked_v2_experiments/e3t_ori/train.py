from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import jaxmarl
import optax
import wandb
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from overcooked_v2_experiments.e3t_ori.models.original import E3TContextModule, E3TPolicyModule


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    hist_obs: jnp.ndarray
    hist_action: jnp.ndarray
    partner_action: jnp.ndarray
    original_reward: jnp.ndarray
    shaped_reward: jnp.ndarray
    info: jnp.ndarray


def make_train(config, update_step_offset=None, update_step_num_overwrite=None, population_config=None):
    if population_config is not None:
        raise NotImplementedError("E3T original reproduction does not use external partner populations.")

    env_config = config["env"]
    model_config = config["model"]
    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    if env.num_agents != 2:
        raise NotImplementedError("Current E3T original reproduction assumes a 2-player environment.")

    ego_agent, partner_agent = env.agents[0], env.agents[1]
    obs_shape = env.observation_space().shape
    action_dim = env.action_space(ego_agent).n
    stay_action = min(4, action_dim - 1)
    env_config.setdefault("ENV_KWARGS", {})["obs_shape"] = obs_shape

    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"] // model_config["NUM_STEPS"] // model_config["NUM_ENVS"]
    )
    total_samples = model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
    model_config["MINIBATCH_SIZE"] = total_samples // model_config["NUM_MINIBATCHES"]

    num_checkpoints = config["NUM_CHECKPOINTS"]
    checkpoint_steps = jnp.linspace(
        0,
        model_config["NUM_UPDATES"],
        num_checkpoints,
        endpoint=True,
        dtype=jnp.int32,
    )
    if num_checkpoints > 0:
        checkpoint_steps = checkpoint_steps.at[-1].set(model_config["NUM_UPDATES"])

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=model_config["REW_SHAPING_HORIZON"],
    )

    def _create_lr_fn():
        if not model_config.get("ANNEAL_LR", True):
            return optax.constant_schedule(model_config["LR"])
        return optax.linear_schedule(
            init_value=model_config["LR"],
            end_value=0.0,
            transition_steps=model_config["NUM_UPDATES"] * model_config["UPDATE_EPOCHS"],
        )

    def _context_lr_fn():
        context_lr = model_config.get("CONTEXT_LR", model_config["LR"])
        if not model_config.get("ANNEAL_LR", True):
            return optax.constant_schedule(context_lr)
        return optax.linear_schedule(
            init_value=context_lr,
            end_value=0.0,
            transition_steps=model_config["NUM_UPDATES"] * model_config["CONTEXT_UPDATE_EPOCHS"],
        )

    def _init_checkpoint_tree(policy_params, context_params):
        combined = {"policy_params": policy_params, "context_params": context_params}
        return jax.tree_util.tree_map(
            lambda p: jnp.zeros((num_checkpoints,) + p.shape, dtype=p.dtype),
            combined,
        )

    def _write_checkpoint(checkpoints, policy_params, context_params, idx):
        combined = {"policy_params": policy_params, "context_params": context_params}
        return jax.tree_util.tree_map(lambda x, y: x.at[idx].set(y), checkpoints, combined)

    def _sample_partner_context(rng_key):
        raw = jax.random.normal(rng_key, (model_config["NUM_ENVS"], action_dim))
        return jax.nn.softmax(raw, axis=-1)

    def _init_history(obs_batch):
        history_obs = jnp.repeat(obs_batch[:, None, ...], model_config["CONTEXT_LENGTH"], axis=1)
        history_actions = jnp.full(
            (model_config["NUM_ENVS"], model_config["CONTEXT_LENGTH"]),
            stay_action,
            dtype=jnp.int32,
        )
        return history_obs, history_actions

    def _partner_action_dist(policy_module, policy_params, partner_obs, context_probs):
        pi, _ = policy_module.apply(policy_params, partner_obs, context_probs)
        probs = pi.probs
        uniform_mix = model_config.get("PARTNER_UNIFORM_MIX", 0.0)
        if uniform_mix > 0:
            probs = (1.0 - uniform_mix) * probs + uniform_mix * jnp.full_like(probs, 1.0 / action_dim)
        return distrax.Categorical(probs=probs)

    def train(rng, population=None, initial_train_state=None):
        if population is not None:
            raise NotImplementedError("E3T original reproduction does not use external partner populations.")

        original_seed = rng[0]
        context_module = E3TContextModule(model_config, action_dim)
        policy_module = E3TPolicyModule(model_config, action_dim)

        rng, ctx_rng, pol_rng = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((model_config["NUM_ENVS"],) + obs_shape, dtype=jnp.float32)
        dummy_hist_obs = jnp.zeros(
            (model_config["NUM_ENVS"], model_config["CONTEXT_LENGTH"]) + obs_shape,
            dtype=jnp.float32,
        )
        dummy_hist_actions = jnp.full(
            (model_config["NUM_ENVS"], model_config["CONTEXT_LENGTH"]),
            stay_action,
            dtype=jnp.int32,
        )
        dummy_context = jnp.full((model_config["NUM_ENVS"], action_dim), 1.0 / action_dim, dtype=jnp.float32)

        context_params = context_module.init(ctx_rng, dummy_obs, dummy_hist_obs, dummy_hist_actions)
        policy_params = policy_module.init(pol_rng, dummy_obs, dummy_context)

        policy_tx = optax.chain(
            optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
            optax.adam(_create_lr_fn(), eps=1e-5),
        )
        context_tx = optax.chain(
            optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
            optax.adam(_context_lr_fn(), eps=1e-5),
        )

        policy_state = TrainState.create(apply_fn=policy_module.apply, params=policy_params, tx=policy_tx)
        context_state = TrainState.create(apply_fn=context_module.apply, params=context_params, tx=context_tx)
        if initial_train_state is not None:
            policy_state, context_state = initial_train_state
        partner_policy_params = policy_state.params

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, model_config["NUM_ENVS"])
        last_obs, env_state = jax.vmap(env.reset)(reset_rng)
        history_obs, history_actions = _init_history(last_obs[ego_agent].astype(jnp.float32))
        rng, partner_ctx_rng = jax.random.split(rng)
        partner_context = _sample_partner_context(partner_ctx_rng)

        initial_checkpoints = _init_checkpoint_tree(policy_state.params, context_state.params)
        initial_update_step = update_step_offset or 0
        if num_checkpoints > 0:
            initial_checkpoints = jax.lax.cond(
                (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                _write_checkpoint,
                lambda c, _pp, _cp, _idx: c,
                initial_checkpoints,
                policy_state.params,
                context_state.params,
                0,
            )

        def _env_step(carry, _):
            (
                policy_state,
                context_state,
                partner_policy_params,
                env_state,
                last_obs,
                history_obs,
                history_actions,
                partner_context,
                update_step,
                rng,
            ) = carry
            ego_obs = last_obs[ego_agent]
            partner_obs = last_obs[partner_agent]

            context_logits = context_module.apply(
                context_state.params,
                ego_obs,
                history_obs.astype(ego_obs.dtype),
                history_actions,
            )
            context_probs = jax.lax.stop_gradient(jax.nn.softmax(context_logits, axis=-1))
            ego_pi, value = policy_module.apply(policy_state.params, ego_obs, context_probs)

            rng, ego_rng, partner_act_rng, step_rng, refresh_ctx_rng = jax.random.split(rng, 5)
            ego_action = ego_pi.sample(seed=ego_rng)
            ego_log_prob = ego_pi.log_prob(ego_action)

            partner_dist = _partner_action_dist(
                policy_module,
                partner_policy_params,
                partner_obs,
                partner_context,
            )
            partner_action = partner_dist.sample(seed=partner_act_rng)

            env_action = {ego_agent: ego_action, partner_agent: partner_action}
            step_keys = jax.random.split(step_rng, model_config["NUM_ENVS"])
            next_obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_keys,
                env_state,
                env_action,
            )

            original_reward = reward[ego_agent]
            shaped_reward = info["shaped_reward"][ego_agent]
            current_timestep = update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
            anneal_factor = rew_shaping_anneal(current_timestep)
            mixed_reward = original_reward + shaped_reward * anneal_factor

            transition = Transition(
                done=done["__all__"],
                action=ego_action,
                value=value,
                reward=mixed_reward,
                log_prob=ego_log_prob,
                obs=ego_obs,
                hist_obs=history_obs,
                hist_action=history_actions,
                partner_action=partner_action,
                original_reward=original_reward,
                shaped_reward=shaped_reward,
                info=info,
            )

            reset_history_obs, reset_history_actions = _init_history(next_obs[ego_agent].astype(history_obs.dtype))
            done_mask_obs = done["__all__"][:, None, None, None, None]
            done_mask_act = done["__all__"][:, None]
            next_history_obs = jnp.concatenate(
                [history_obs[:, 1:], ego_obs[:, None, ...].astype(history_obs.dtype)],
                axis=1,
            )
            next_history_actions = jnp.concatenate(
                [history_actions[:, 1:], partner_action[:, None]],
                axis=1,
            )
            next_history_obs = jnp.where(done_mask_obs, reset_history_obs, next_history_obs)
            next_history_actions = jnp.where(
                done_mask_act,
                reset_history_actions,
                next_history_actions,
            )
            done_mask_context = done["__all__"][:, None]
            refreshed_context = _sample_partner_context(refresh_ctx_rng)
            next_partner_context = jnp.where(done_mask_context, refreshed_context, partner_context)

            carry = (
                policy_state,
                context_state,
                partner_policy_params,
                env_state,
                next_obs,
                next_history_obs,
                next_history_actions,
                next_partner_context,
                update_step,
                rng,
            )
            return carry, transition

        def _compute_gae(traj_batch, last_value):
            def _gae_step(carry, transition):
                gae, next_value = carry
                delta = transition.reward + model_config["GAMMA"] * next_value * (1.0 - transition.done) - transition.value
                gae = delta + model_config["GAMMA"] * model_config["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                _gae_step,
                (jnp.zeros_like(last_value), last_value),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        def _flatten_batch(traj_batch, advantages, targets):
            return {
                "obs": traj_batch.obs.reshape((-1,) + traj_batch.obs.shape[2:]),
                "action": traj_batch.action.reshape((-1,)),
                "value": traj_batch.value.reshape((-1,)),
                "log_prob": traj_batch.log_prob.reshape((-1,)),
                "hist_obs": traj_batch.hist_obs.reshape((-1,) + traj_batch.hist_obs.shape[2:]),
                "hist_action": traj_batch.hist_action.reshape((-1,) + traj_batch.hist_action.shape[2:]),
                "partner_action": traj_batch.partner_action.reshape((-1,)),
                "advantages": advantages.reshape((-1,)),
                "targets": targets.reshape((-1,)),
            }

        def _policy_loss(policy_params, context_params, batch):
            context_logits = context_module.apply(
                context_params,
                batch["obs"],
                batch["hist_obs"].astype(batch["obs"].dtype),
                batch["hist_action"],
            )
            context_probs = jax.lax.stop_gradient(jax.nn.softmax(context_logits, axis=-1))
            pi, value = policy_module.apply(policy_params, batch["obs"], context_probs)
            log_prob = pi.log_prob(batch["action"])

            value_pred_clipped = batch["value"] + (value - batch["value"]).clip(
                -model_config["CLIP_EPS"], model_config["CLIP_EPS"]
            )
            value_losses = jnp.square(value - batch["targets"])
            value_losses_clipped = jnp.square(value_pred_clipped - batch["targets"])
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            adv = batch["advantages"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = jnp.exp(log_prob - batch["log_prob"])
            loss_actor1 = ratio * adv
            loss_actor2 = jnp.clip(ratio, 1.0 - model_config["CLIP_EPS"], 1.0 + model_config["CLIP_EPS"]) * adv
            actor_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
            entropy = pi.entropy().mean()
            total_loss = actor_loss + model_config["VF_COEF"] * value_loss - model_config["ENT_COEF"] * entropy
            return total_loss, (value_loss, actor_loss, entropy, ratio.mean())

        def _context_loss(context_params, batch):
            context_logits = context_module.apply(
                context_params,
                batch["obs"],
                batch["hist_obs"].astype(batch["obs"].dtype),
                batch["hist_action"],
            )
            ce = optax.softmax_cross_entropy_with_integer_labels(context_logits, batch["partner_action"])
            return ce.mean()

        def _update_epoch(carry, _):
            policy_state, context_state, flat_batch, rng = carry
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, flat_batch["action"].shape[0])
            shuffled = jax.tree_util.tree_map(lambda x: x[permutation], flat_batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((model_config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                shuffled,
            )

            def _policy_minibatch(policy_state, batch):
                grad_fn = jax.value_and_grad(_policy_loss, has_aux=True)
                (loss, aux), grads = grad_fn(policy_state.params, context_state.params, batch)
                policy_state = policy_state.apply_gradients(grads=grads)
                return policy_state, (loss, aux)

            policy_state, policy_info = jax.lax.scan(_policy_minibatch, policy_state, minibatches)
            return (policy_state, context_state, flat_batch, rng), policy_info

        def _update_context_epoch(carry, _):
            policy_state, context_state, flat_batch, rng = carry
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, flat_batch["partner_action"].shape[0])
            shuffled = jax.tree_util.tree_map(lambda x: x[permutation], flat_batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((model_config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                shuffled,
            )

            def _context_minibatch(context_state, batch):
                grad_fn = jax.value_and_grad(_context_loss)
                loss, grads = grad_fn(context_state.params, batch)
                context_state = context_state.apply_gradients(grads=grads)
                return context_state, loss

            context_state, context_losses = jax.lax.scan(_context_minibatch, context_state, minibatches)
            return (policy_state, context_state, flat_batch, rng), context_losses

        def _update_step(runner_state, _):
            (
                policy_state,
                context_state,
                partner_policy_params,
                checkpoints,
                env_state,
                last_obs,
                history_obs,
                history_actions,
                partner_context,
                rng,
                update_step,
            ) = runner_state
            rollout_carry = (
                policy_state,
                context_state,
                partner_policy_params,
                env_state,
                last_obs,
                history_obs,
                history_actions,
                partner_context,
                update_step,
                rng,
            )
            rollout_carry, traj_batch = jax.lax.scan(_env_step, rollout_carry, None, model_config["NUM_STEPS"])
            (
                policy_state,
                context_state,
                partner_policy_params,
                env_state,
                last_obs,
                history_obs,
                history_actions,
                partner_context,
                update_step,
                rng,
            ) = rollout_carry

            last_context_logits = context_module.apply(
                context_state.params,
                last_obs[ego_agent],
                history_obs.astype(last_obs[ego_agent].dtype),
                history_actions,
            )
            last_context_probs = jax.lax.stop_gradient(jax.nn.softmax(last_context_logits, axis=-1))
            _, last_value = policy_module.apply(policy_state.params, last_obs[ego_agent], last_context_probs)
            advantages, targets = _compute_gae(traj_batch, last_value)
            flat_batch = _flatten_batch(traj_batch, advantages, targets)

            rng, ctx_rng = jax.random.split(rng)
            context_carry = (policy_state, context_state, flat_batch, ctx_rng)
            context_carry, context_losses = jax.lax.scan(
                _update_context_epoch,
                context_carry,
                None,
                model_config["CONTEXT_UPDATE_EPOCHS"],
            )
            policy_state, context_state, _, rng = context_carry

            rng, ppo_rng = jax.random.split(rng)
            update_carry = (policy_state, context_state, flat_batch, ppo_rng)
            update_carry, policy_info = jax.lax.scan(_update_epoch, update_carry, None, model_config["UPDATE_EPOCHS"])
            policy_state, context_state, flat_batch, rng = update_carry

            copy_coeff = model_config.get("COPY", 1.0)
            partner_policy_params = jax.tree_util.tree_map(
                lambda target, source: target * (1.0 - copy_coeff) + source * copy_coeff,
                partner_policy_params,
                policy_state.params,
            )

            policy_losses, policy_aux = policy_info
            value_losses, actor_losses, entropies, ratios = policy_aux
            metric = traj_batch.info
            metric["reward_mean"] = traj_batch.reward.mean()
            metric["original_reward_mean"] = traj_batch.original_reward.mean()
            metric["shaped_reward_mean"] = traj_batch.shaped_reward.mean()
            metric["total_loss"] = policy_losses.mean()
            metric["value_loss"] = value_losses.mean()
            metric["loss_actor"] = actor_losses.mean()
            metric["entropy"] = entropies.mean()
            metric["ratio"] = ratios.mean()
            metric["context_loss"] = context_losses.mean()

            # Done-only episode return diagnostics to avoid sparse-step averaging confusion.
            returned_episode = traj_batch.info["returned_episode"]
            returned_episode_returns = traj_batch.info["returned_episode_returns"]
            returned_episode_count = returned_episode.sum()
            done_return_denom = jnp.maximum(returned_episode_count, 1.0)
            metric["returned_episode_count"] = returned_episode_count
            metric["returned_episode_returns_done_mean"] = (
                (returned_episode_returns * returned_episode).sum() / done_return_denom
            )
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

            update_step += 1
            metric["update_step"] = update_step
            metric["env_step"] = update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]

            def _callback(metric, seed):
                wandb.log({f"rng{int(seed)}/{k}": v for k, v in metric.items()})

            jax.debug.callback(_callback, metric, original_seed)

            if num_checkpoints > 0:
                selector = checkpoint_steps == update_step
                checkpoints = jax.lax.cond(
                    jnp.any(selector),
                    _write_checkpoint,
                    lambda c, _pp, _cp, _idx: c,
                    checkpoints,
                    policy_state.params,
                    context_state.params,
                    jnp.argmax(selector),
                )

            runner_state = (
                policy_state,
                context_state,
                partner_policy_params,
                checkpoints,
                env_state,
                last_obs,
                history_obs,
                history_actions,
                partner_context,
                rng,
                update_step,
            )
            return runner_state, metric

        runner_state = (
            policy_state,
            context_state,
            partner_policy_params,
            initial_checkpoints,
            env_state,
            last_obs,
            history_obs,
            history_actions,
            partner_context,
            rng,
            initial_update_step,
        )
        num_update_steps = update_step_num_overwrite or model_config["NUM_UPDATES"]
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, num_update_steps)
        return {"runner_state": runner_state, "metrics": metric, "checkpoints": runner_state[3]}

    return train
