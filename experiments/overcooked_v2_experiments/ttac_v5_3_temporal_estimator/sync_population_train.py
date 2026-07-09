"""Maximum-Entropy Population training for OvercookedV2.

This module ports the MEP entropy objective from the PBL-subnet Hanabi code to
OvercookedV2 with a real synchronously trained population. The ego policy is
TTACv2, while the synchronously trained partner population uses ordinary PPO-CNN
policies so TTAC remains the only adaptive control path.
"""

from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import jaxmarl
import optax
import wandb
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper

from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.models.model import (
    get_actor_critic as get_ttac_actor_critic,
    initialize_carry,
)
from overcooked_v2_experiments.ppo.models.model import (
    get_actor_critic as get_ppo_actor_critic,
)
from overcooked_v2_experiments.ttac_v5_3_temporal_estimator.ippo import partner_batchify, apply_actor_critic, categorical_symmetric_kl, adapter_only_logits
from overcooked_v2_experiments.utils.utils import mini_batch_pmap


class MEPTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    partner_obs: jnp.ndarray
    partner_action: jnp.ndarray
    info: dict
    ego_mask: jnp.ndarray
    partner_mask: jnp.ndarray
    partner_id: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _stack_tree(*trees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)


def _actor_ids_from_env_ids(env_ids, num_agents):
    return jnp.tile(env_ids, num_agents)


def _safe_masked_mean(x, mask):
    x = jnp.where(mask, x, 0.0)
    count = jnp.maximum(mask.sum(), 1)
    return x.sum() / count


def _safe_masked_std(x, mask):
    mean = _safe_masked_mean(x, mask)
    var = _safe_masked_mean(jnp.square(x - mean), mask)
    return jnp.sqrt(var + 1e-8)


def _masked_normalize(x, mask):
    return (x - _safe_masked_mean(x, mask)) / (_safe_masked_std(x, mask) + 1e-8)


def make_train(config, update_step_offset=None, update_step_num_overwrite=None):
    env_config = config["env"]
    model_config = config["model"]
    mep_config = config.get("MEP", {})

    if model_config["TYPE"] != "CNN":
        raise NotImplementedError("MEP v1 intentionally supports PPO CNN only.")

    population_size = int(mep_config.get("POPULATION_SIZE", 5))
    mep_ent_coef = float(mep_config.get("ENT_COEF", 0.1))
    group_mm = int(mep_config.get("GROUP_MM", 1))
    group_mp = int(mep_config.get("GROUP_MP", 1))
    if group_mm < 0 or group_mp < 0 or group_mm + group_mp <= 0:
        raise ValueError("MEP requires GROUP_MM/GROUP_MP to be non-negative with positive sum.")

    env = jaxmarl.make(env_config["ENV_NAME"], **env_config["ENV_KWARGS"])
    env = OvercookedV2LogWrapper(env, replace_info=False)

    model_config["NUM_ACTORS"] = env.num_agents * model_config["NUM_ENVS"]
    model_config["NUM_UPDATES"] = (
        model_config["TOTAL_TIMESTEPS"]
        // model_config["NUM_STEPS"]
        // model_config["NUM_ENVS"]
    )
    model_config["MINIBATCH_SIZE"] = (
        model_config["NUM_ACTORS"]
        * model_config["NUM_STEPS"]
        // model_config["NUM_MINIBATCHES"]
    )

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
    print("Checkpoint steps: ", checkpoint_steps)

    train_idxs = jnp.linspace(
        0,
        env.num_agents,
        model_config["NUM_ENVS"],
        dtype=jnp.int32,
        endpoint=False,
    )
    ego_mask_dict = {a: train_idxs == i for i, a in enumerate(env.agents)}
    ego_side_mask_flat = batchify(ego_mask_dict, env.agents, model_config["NUM_ACTORS"]).squeeze()
    pair_cycle = group_mm + group_mp
    mp_mask_env = (jnp.arange(model_config["NUM_ENVS"]) % pair_cycle) >= group_mm
    mp_mask_flat = _actor_ids_from_env_ids(mp_mask_env, env.num_agents).astype(bool)
    mm_mask_flat = jnp.logical_not(mp_mask_flat)
    ego_actor_mask_flat = jnp.logical_or(mm_mask_flat, ego_side_mask_flat)
    partner_actor_mask_flat = jnp.logical_and(mp_mask_flat, jnp.logical_not(ego_side_mask_flat))
    print("ego_side_mask_flat", ego_side_mask_flat.shape, "sum", ego_side_mask_flat.sum())
    print("mep mm actors", mm_mask_flat.sum(), "mp actors", mp_mask_flat.sum())
    print("mep ego update actors", ego_actor_mask_flat.sum(), "partner update actors", partner_actor_mask_flat.sum())

    def _update_checkpoint(checkpoint_states, params, i):
        jax.debug.print("Saving MEP checkpoint {i}", i=i)
        return jax.tree_util.tree_map(lambda x, y: x.at[i].set(y), checkpoint_states, params)

    def create_learning_rate_fn():
        base_learning_rate = model_config["LR"]
        warmup_steps = int(model_config["LR_WARMUP"] * model_config["NUM_UPDATES"])
        steps_per_epoch = model_config["NUM_MINIBATCHES"] * model_config["UPDATE_EPOCHS"]
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(model_config["NUM_UPDATES"] - warmup_steps, 1)
        print("Update steps: ", model_config["NUM_UPDATES"])
        print("Warmup epochs: ", warmup_steps)
        print("Cosine epochs: ", cosine_epochs)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch,
        )
        return optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=model_config["REW_SHAPING_HORIZON"],
    )

    def train(
        rng,
        run_index=None,
        initial_ego_state=None,
        initial_partner_state=None,
    ):
        original_seed = rng[0]
        jax.debug.print("original_seed {s}", s=rng)

        ego_network = get_ttac_actor_critic(config)
        partner_network = get_ppo_actor_critic(config)
        rng, ego_init_rng, partner_init_rng = jax.random.split(rng, 3)
        init_x = (
            jnp.zeros((1, model_config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, model_config["NUM_ENVS"])),
        )
        init_hstate = initialize_carry(config, model_config["NUM_ENVS"])
        ego_params = ego_network.init(ego_init_rng, init_hstate, init_x)

        partner_keys = jax.random.split(partner_init_rng, population_size)
        partner_params = jax.vmap(lambda k: partner_network.init(k, init_hstate, init_x))(
            partner_keys
        )

        if model_config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(model_config["MAX_GRAD_NORM"]),
                optax.adam(model_config["LR"], eps=1e-5),
            )

        ego_state = TrainState.create(apply_fn=ego_network.apply, params=ego_params, tx=tx)
        partner_state = TrainState.create(
            apply_fn=partner_network.apply, params=partner_params, tx=tx
        )

        if initial_ego_state is not None:
            ego_state = initial_ego_state
        if initial_partner_state is not None:
            partner_state = initial_partner_state

        rng, reset_rng, partner_rng = jax.random.split(rng, 3)
        reset_rng = jax.random.split(reset_rng, model_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_partner_ids_env = jax.random.randint(
            partner_rng, (model_config["NUM_ENVS"],), 0, population_size
        )
        init_partner_ids = _actor_ids_from_env_ids(init_partner_ids_env, env.num_agents)

        def _forward_actor(actor_network, params, obs_batch, done_batch):
            if obs_batch.ndim == len(env.observation_space().shape) + 1:
                ac_obs = obs_batch[jnp.newaxis, :]
                ac_done = done_batch[jnp.newaxis, :]
                squeeze_time = True
            else:
                ac_obs = obs_batch
                ac_done = done_batch
                squeeze_time = False
            _, pi, value, _ = apply_actor_critic(actor_network, params, None, (ac_obs, ac_done))
            if squeeze_time:
                return pi.logits[0], value[0]
            return pi.logits, value

        def _all_partner_forward(params, obs_batch, done_batch):
            return jax.vmap(lambda p: _forward_actor(partner_network, p, obs_batch, done_batch))(params)

        def _select_by_partner_id(stacked, partner_ids):
            # stacked: [K, ...actors..., optional action_dim]
            if stacked.ndim == partner_ids.ndim + 1:
                return jnp.take_along_axis(stacked, partner_ids[None, ...], axis=0).squeeze(0)
            return jnp.take_along_axis(stacked, partner_ids[None, ..., None], axis=0).squeeze(0)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                delta = (
                    transition.reward
                    + model_config["GAMMA"] * next_value * (1 - transition.done)
                    - transition.value
                )
                gae = (
                    delta
                    + model_config["GAMMA"]
                    * model_config["GAE_LAMBDA"]
                    * (1 - transition.done)
                    * gae
                )
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        def _update_step(runner_state, unused):
            (
                ego_state,
                partner_state,
                ego_checkpoints,
                partner_checkpoints,
                env_state,
                last_obs,
                last_done,
                update_step,
                partner_ids,
                rng,
            ) = runner_state

            def _env_step(env_step_state, unused):
                (
                    ego_state,
                    partner_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    partner_ids,
                    rng,
                ) = env_step_state

                rng, ego_action_rng, partner_action_rng, step_rng, partner_reset_rng = jax.random.split(rng, 5)
                action_partner_ids = partner_ids
                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )

                ego_logits, ego_value = _forward_actor(
                    ego_network, ego_state.params, obs_batch, last_done
                )
                partner_logits_all, partner_values_all = _all_partner_forward(
                    partner_state.params, obs_batch, last_done
                )
                partner_logits = _select_by_partner_id(partner_logits_all, partner_ids)
                partner_value = _select_by_partner_id(partner_values_all, partner_ids)

                ego_pi = distrax.Categorical(logits=ego_logits)
                partner_pi = distrax.Categorical(logits=partner_logits)
                ego_action = ego_pi.sample(seed=ego_action_rng)
                partner_action = partner_pi.sample(seed=partner_action_rng)

                action = jnp.where(ego_actor_mask_flat, ego_action, partner_action)
                log_prob = jnp.where(
                    ego_actor_mask_flat,
                    ego_pi.log_prob(ego_action),
                    partner_pi.log_prob(partner_action),
                )
                value = jnp.where(ego_actor_mask_flat, ego_value, partner_value)

                env_act = unbatchify(
                    action, env.agents, model_config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}
                rng_step = jax.random.split(step_rng, model_config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                original_reward = jnp.array([reward[a] for a in env.agents])
                current_timestep = (
                    update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )
                shaped_reward = jnp.array([info["shaped_reward"][a] for a in env.agents])
                combined_reward = jnp.array([reward[a] for a in env.agents])
                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward
                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((model_config["NUM_ACTORS"])), info
                )
                partner_id_mean = _safe_masked_mean(
                    action_partner_ids.astype(jnp.float32), partner_actor_mask_flat
                )
                info["mep_pair_partner_id_mean"] = jnp.full(
                    (model_config["NUM_ACTORS"],), partner_id_mean
                )
                info["mep_pair_is_mp"] = jnp.broadcast_to(
                    mp_mask_flat.astype(jnp.float32), (model_config["NUM_ACTORS"],)
                )
                info["mep_ego_update_mask"] = jnp.broadcast_to(
                    ego_actor_mask_flat.astype(jnp.float32), (model_config["NUM_ACTORS"],)
                )
                info["mep_partner_update_mask"] = jnp.broadcast_to(
                    partner_actor_mask_flat.astype(jnp.float32), (model_config["NUM_ACTORS"],)
                )

                done_batch = batchify(done, env.agents, model_config["NUM_ACTORS"]).squeeze()
                done_actor = jnp.tile(done["__all__"], env.num_agents)
                new_partner_ids_env = jax.random.randint(
                    partner_reset_rng, (model_config["NUM_ENVS"],), 0, population_size
                )
                new_partner_ids = _actor_ids_from_env_ids(new_partner_ids_env, env.num_agents)
                partner_ids = jnp.where(done_actor, new_partner_ids, partner_ids)

                action_flat = action.squeeze()
                partner_obs_batch = partner_batchify(obs_batch, model_config["NUM_ENVS"], env.num_agents)
                partner_action_batch = partner_batchify(action_flat[:, None], model_config["NUM_ENVS"], env.num_agents).squeeze(axis=-1)

                transition = MEPTransition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action_flat,
                    value.squeeze(),
                    batchify(reward, env.agents, model_config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    partner_obs_batch,
                    partner_action_batch,
                    info,
                    ego_actor_mask_flat,
                    partner_actor_mask_flat,
                    action_partner_ids,
                )

                env_step_state = (
                    ego_state,
                    partner_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    partner_ids,
                    rng,
                )
                return env_step_state, transition

            env_step_state = (
                ego_state,
                partner_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                partner_ids,
                rng,
            )
            env_step_state, traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, model_config["NUM_STEPS"]
            )
            (
                ego_state,
                partner_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                partner_ids,
                rng,
            ) = env_step_state

            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            ego_last_logits, ego_last_val = _forward_actor(
                ego_network, ego_state.params, last_obs_batch, last_done
            )
            del ego_last_logits
            partner_last_logits_all, partner_last_values_all = _all_partner_forward(
                partner_state.params, last_obs_batch, last_done
            )
            del partner_last_logits_all
            partner_last_val = _select_by_partner_id(partner_last_values_all, partner_ids)
            last_val = jnp.where(ego_actor_mask_flat, ego_last_val, partner_last_val)
            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _loss_fn(ego_params, partner_params, traj_batch, gae, targets):
                ego_logits, ego_value = _forward_actor(
                    ego_network, ego_params, traj_batch.obs, traj_batch.done
                )
                ego_pi = distrax.Categorical(logits=ego_logits)
                ego_log_prob = ego_pi.log_prob(traj_batch.action)

                partner_logits_all, partner_values_all = jax.vmap(
                    lambda p: _forward_actor(partner_network, p, traj_batch.obs, traj_batch.done)
                )(partner_params)
                partner_logits = _select_by_partner_id(
                    partner_logits_all, traj_batch.partner_id
                )
                partner_value = _select_by_partner_id(
                    partner_values_all, traj_batch.partner_id
                )
                partner_pi = distrax.Categorical(logits=partner_logits)
                partner_log_prob = partner_pi.log_prob(traj_batch.action)

                def _ppo_loss(pi, value, log_prob, mask, norm_gae):
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-model_config["CLIP_EPS"], model_config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * _safe_masked_mean(
                        jnp.maximum(value_losses, value_losses_clipped), mask
                    )
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    loss_actor1 = ratio * norm_gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - model_config["CLIP_EPS"],
                            1.0 + model_config["CLIP_EPS"],
                        )
                        * norm_gae
                    )
                    loss_actor = -_safe_masked_mean(
                        jnp.minimum(loss_actor1, loss_actor2), mask
                    )
                    entropy = _safe_masked_mean(pi.entropy(), mask)
                    ratio_mean = _safe_masked_mean(ratio, mask)
                    total = (
                        loss_actor
                        + model_config["VF_COEF"] * value_loss
                        - model_config["ENT_COEF"] * entropy
                    )
                    return total, value_loss, loss_actor, entropy, ratio_mean

                ego_mask = traj_batch.ego_mask
                partner_mask = traj_batch.partner_mask
                ego_gae = _masked_normalize(gae, ego_mask)
                partner_gae = _masked_normalize(gae, partner_mask)

                ego_total, ego_v, ego_actor, ego_entropy, ego_ratio = _ppo_loss(
                    ego_pi, ego_value, ego_log_prob, ego_mask, ego_gae
                )
                _, _, _, ego_aux = apply_actor_critic(
                    ego_network, ego_params, None, (traj_batch.obs, traj_batch.done)
                )
                _, ego_partner_pi, _, ego_partner_aux = apply_actor_critic(
                    ego_network, ego_params, None, (traj_batch.partner_obs, traj_batch.done)
                )
                agreement_loss = _safe_masked_mean(
                    -ego_partner_pi.log_prob(traj_batch.partner_action.astype(jnp.int32)),
                    ego_mask,
                )
                policy_kl = _safe_masked_mean(
                    categorical_symmetric_kl(ego_aux["base_logits"], adapter_only_logits(ego_aux)),
                    ego_mask,
                )
                partner_policy_kl = _safe_masked_mean(
                    categorical_symmetric_kl(
                        ego_partner_aux["base_logits"], adapter_only_logits(ego_partner_aux)
                    ),
                    ego_mask,
                )
                adapter_delta_norm = _safe_masked_mean(ego_aux["adapter_delta_norm"], ego_mask)
                ego_total = (
                    ego_total
                    + model_config.get("TTAC_AGREEMENT_COEF", 0.05) * agreement_loss
                    + model_config.get("TTAC_TRAIN_KL_COEF", 0.005)
                    * (policy_kl + partner_policy_kl)
                )
                partner_ppo, partner_v, partner_actor, partner_entropy, partner_ratio = _ppo_loss(
                    partner_pi,
                    partner_value,
                    partner_log_prob,
                    partner_mask,
                    partner_gae,
                )

                partner_index_shape = (population_size,) + (1,) * traj_batch.partner_id.ndim
                partner_indices = jnp.arange(population_size).reshape(partner_index_shape)
                current_partner_mask = partner_indices == traj_batch.partner_id[jnp.newaxis, ...]
                entropy_logits_all = jnp.where(
                    current_partner_mask[..., None],
                    partner_logits_all,
                    jax.lax.stop_gradient(partner_logits_all),
                )
                all_partner_probs = jax.nn.softmax(entropy_logits_all, axis=-1)
                mean_partner_probs = jnp.mean(all_partner_probs, axis=0)
                mep_entropy = -jnp.sum(
                    mean_partner_probs * jnp.log(mean_partner_probs + 1e-8), axis=-1
                )
                mep_entropy = _safe_masked_mean(mep_entropy, partner_mask)
                partner_total = partner_ppo - mep_ent_coef * mep_entropy
                total = ego_total + partner_total

                aux = {
                    "ego_total_loss": ego_total,
                    "ego_value_loss": ego_v,
                    "ego_actor_loss": ego_actor,
                    "ego_entropy": ego_entropy,
                    "ego_ratio": ego_ratio,
                    "ttac_agreement_loss": agreement_loss,
                    "ttac_policy_kl_to_base": policy_kl,
                    "ttac_partner_policy_kl_to_base": partner_policy_kl,
                    "ttac_adapter_delta_norm": adapter_delta_norm,
                    "partner_total_loss": partner_total,
                    "partner_ppo_loss": partner_ppo,
                    "partner_value_loss": partner_v,
                    "partner_actor_loss": partner_actor,
                    "partner_entropy": partner_entropy,
                    "partner_ratio": partner_ratio,
                    "mep_entropy": mep_entropy,
                }
                return total, aux

            def _update_epoch(update_state, unused):
                ego_state, partner_state, traj_batch, advantages, targets, rng = update_state

                def _update_minibatch(states, batch_info):
                    ego_state, partner_state = states
                    traj_mb, adv_mb, target_mb = batch_info
                    grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                    (total_loss, aux), (ego_grads, partner_grads) = grad_fn(
                        ego_state.params,
                        partner_state.params,
                        traj_mb,
                        adv_mb,
                        target_mb,
                    )
                    ego_state = ego_state.apply_gradients(grads=ego_grads)
                    partner_state = partner_state.apply_gradients(grads=partner_grads)
                    aux["total_loss"] = total_loss
                    return (ego_state, partner_state), aux

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, model_config["NUM_ACTORS"])
                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], model_config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                (ego_state, partner_state), loss_info = jax.lax.scan(
                    _update_minibatch, (ego_state, partner_state), minibatches
                )
                return (ego_state, partner_state, traj_batch, advantages, targets, rng), loss_info

            rng, _rng = jax.random.split(rng)
            update_state = (ego_state, partner_state, traj_batch, advantages, targets, _rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, model_config["UPDATE_EPOCHS"]
            )
            ego_state, partner_state = update_state[0], update_state[1]

            metric = jax.tree_util.tree_map(lambda x: x.mean(), traj_batch.info)
            loss_metric = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric.update(loss_metric)
            metric["mep_population_size"] = jnp.array(population_size, dtype=jnp.float32)
            metric["mep_ent_coef"] = jnp.array(mep_ent_coef, dtype=jnp.float32)
            metric["mep_group_mm"] = jnp.array(group_mm, dtype=jnp.float32)
            metric["mep_group_mp"] = jnp.array(group_mp, dtype=jnp.float32)
            update_step += 1
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * model_config["NUM_STEPS"] * model_config["NUM_ENVS"]
            )

            def callback(metric, original_seed, run_index):
                if run_index is None:
                    prefix = f"rng{int(original_seed)}"
                else:
                    prefix = f"run_{int(run_index)}"
                wandb.log({f"{prefix}/{k}": v for k, v in metric.items()})

            jax.debug.callback(callback, metric, original_seed, run_index)

            if num_checkpoints > 0:
                checkpoint_idx_selector = checkpoint_steps == update_step
                checkpoint_idx = jnp.argmax(checkpoint_idx_selector)
                ego_checkpoints = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    ego_checkpoints,
                    ego_state.params,
                    checkpoint_idx,
                )
                partner_checkpoints = jax.lax.cond(
                    jnp.any(checkpoint_idx_selector),
                    _update_checkpoint,
                    lambda c, _p, _i: c,
                    partner_checkpoints,
                    partner_state.params,
                    checkpoint_idx,
                )

            runner_state = (
                ego_state,
                partner_state,
                ego_checkpoints,
                partner_checkpoints,
                env_state,
                last_obs,
                last_done,
                update_step,
                partner_ids,
                rng,
            )
            return runner_state, metric

        initial_update_step = 0 if update_step_offset is None else update_step_offset
        ego_checkpoints = jax.tree_util.tree_map(
            lambda p: jnp.zeros((num_checkpoints,) + p.shape, p.dtype), ego_state.params
        )
        partner_checkpoints = jax.tree_util.tree_map(
            lambda p: jnp.zeros((num_checkpoints,) + p.shape, p.dtype), partner_state.params
        )
        if num_checkpoints > 0:
            ego_checkpoints = jax.lax.cond(
                (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                _update_checkpoint,
                lambda c, _p, _i: c,
                ego_checkpoints,
                ego_state.params,
                0,
            )
            partner_checkpoints = jax.lax.cond(
                (checkpoint_steps[0] == 0) & (initial_update_step == 0),
                _update_checkpoint,
                lambda c, _p, _i: c,
                partner_checkpoints,
                partner_state.params,
                0,
            )

        runner_state = (
            ego_state,
            partner_state,
            ego_checkpoints,
            partner_checkpoints,
            env_state,
            obsv,
            jnp.zeros((model_config["NUM_ACTORS"],), dtype=bool),
            initial_update_step,
            init_partner_ids,
            rng,
        )
        num_update_steps = model_config["NUM_UPDATES"]
        if update_step_num_overwrite is not None:
            num_update_steps = update_step_num_overwrite
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_update_steps
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
