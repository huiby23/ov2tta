from typing import List
import jax
import jax.numpy as jnp
import chex
from .policy import AbstractPolicy, PolicyPairing, FunctionalPolicyPairing


@chex.dataclass
class PolicyRollout:
    state_seq: chex.Array
    actions_seq: chex.Array
    total_reward: chex.Scalar


def init_rollout(policies: List[AbstractPolicy], env):
    num_agents = env.num_agents

    assert len(policies) == num_agents

    init_hstate = {f"agent_{i}": policies[i].init_hstate(1) for i in range(num_agents)}

    @jax.jit
    def _get_actions(obs, done, hstate, key):
        sample_keys = jax.random.split(key, num_agents)

        actions = {}
        next_hstates = {}
        for i, policy in enumerate(policies):
            agent_id = f"agent_{i}"
            obs_agent, done_agent, hstate_agent = (
                obs[agent_id],
                done[agent_id],
                hstate[agent_id],
            )
            action, next_hstate = policy.compute_action(
                obs_agent, done_agent, hstate_agent, sample_keys[i]
            )
            actions[agent_id] = action
            next_hstates[agent_id] = next_hstate

        return actions, next_hstates

    return init_hstate, _get_actions


def get_rollout(policies: PolicyPairing, env, key) -> PolicyRollout:
    init_hstate, _get_actions = init_rollout(policies, env)

    @jax.jit
    def _perform_step(carry, key):
        obs, state, done, total_reward, hstate = carry

        key_sample, key_step = jax.random.split(key, 2)
        actions, next_hstate = _get_actions(obs, done, hstate, key_sample)

        next_obs, next_state, reward, next_done, info = env.step(
            key_step, state, actions
        )

        if env.num_agents == 2:
            updated_hstate = {}
            for i, policy in enumerate(policies):
                agent_id = f"agent_{i}"
                partner_id = f"agent_{1 - i}"
                updated_hstate[agent_id] = policy.update_after_step(
                    next_hstate[agent_id],
                    obs[partner_id],
                    actions[partner_id],
                    next_done[agent_id],
                )
        else:
            updated_hstate = next_hstate

        new_total_reward = total_reward + reward["agent_0"]

        carry = (next_obs, next_state, next_done, new_total_reward, updated_hstate)
        return carry, (next_state, actions)

    key, key_r = jax.random.split(key, 2)
    obs, state = env.reset(key_r)

    init_done = {f"agent_{i}": False for i in range(env.num_agents)}
    init_done["__all__"] = False

    keys = jax.random.split(key, env.max_steps)
    carry = (
        obs,
        state,
        init_done,
        0.0,
        init_hstate,
    )
    carry, (state_seq, actions_seq) = jax.lax.scan(_perform_step, carry, keys)

    total_reward = carry[-2]

    return PolicyRollout(
        state_seq=state_seq, actions_seq=actions_seq, total_reward=total_reward
    )


def get_rollout_functional(policies: FunctionalPolicyPairing, env, key) -> PolicyRollout:
    if policies.backend == "ttappo":
        from overcooked_v2_experiments.ttappo.policy import get_functional_rollout

        return get_functional_rollout(policies, env, key)
    if policies.backend == "ttappo_v2_temporal":
        from overcooked_v2_experiments.ttappo_v2_temporal.policy import (
            get_functional_rollout,
        )

        return get_functional_rollout(policies, env, key)

    raise NotImplementedError(f"Unknown functional backend: {policies.backend}")
