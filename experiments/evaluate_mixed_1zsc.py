#!/usr/bin/env python3
"""Evaluate PPO/TTAC ego policies against non-ZSC Q-learning partner pools.

This script reports a strict mixed 1-ZSC score:
  ego policy from PPO/TTAC checkpoints vs partners from IQL/VDN/PQN-VDN.

It is intentionally read-only: no training, no checkpoint mutation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "JaxMARL"))
sys.path.insert(0, str(ROOT / "JaxMARL" / "baselines" / "QLearning"))
sys.path.insert(0, str(ROOT / "JaxMARL" / "baselines" / "A2C"))
sys.path.insert(0, str(ROOT / "experiments"))

from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2  # noqa: E402
from jaxmarl.wrappers.baselines import load_params  # noqa: E402
from overcooked_v2_experiments.eval.evaluate import eval_pairing  # noqa: E402
from overcooked_v2_experiments.eval.policy import AbstractPolicy, PolicyPairing  # noqa: E402
from overcooked_v2_experiments.ttac_v5_8_fast_online.policy import PPOPolicy  # noqa: E402
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.agreement_heads import (  # noqa: E402
    load_agreement_npz,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.latent_partner_decoder import (  # noqa: E402
    load_latent_decoder_npz,
)
from overcooked_v2_experiments.ttac_v5_8_fast_online.utils.store import (  # noqa: E402
    load_all_checkpoints,
)
from overcooked_v2_experiments.mappo.policy import MAPPOPolicy  # noqa: E402
from overcooked_v2_experiments.mappo.utils.store import (  # noqa: E402
    load_all_checkpoints as load_mappo_checkpoints,
)

import a2c_cnn_overcooked  # noqa: E402
import coma_cnn_overcooked  # noqa: E402
import iql_double_cnn_overcooked  # noqa: E402
import iql_cnn_overcooked  # noqa: E402
import iql_dueling_cnn_overcooked  # noqa: E402
import pqn_vdn_cnn_overcooked  # noqa: E402
import qplex_cnn_overcooked  # noqa: E402
import qmix_cnn_overcooked  # noqa: E402
import qmix_rnn  # noqa: E402
import shaq  # noqa: E402
import vdn_cnn_overcooked  # noqa: E402
import vdn_dueling_cnn_overcooked  # noqa: E402
import wqmix_cnn_overcooked  # noqa: E402


def load_latent_decoder_bundle(path: Path):
    raw = str(path)
    parts = [p for p in raw.replace(",", "+").split("+") if p]
    if len(parts) <= 1:
        return load_latent_decoder_npz(path)
    return [load_latent_decoder_npz(Path(p)) for p in parts]


def infer_latent_decoder_z_dim(decoder_bundle) -> int | None:
    decoder = decoder_bundle[0] if isinstance(decoder_bundle, (list, tuple)) else decoder_bundle
    if decoder is None:
        return None
    for key in ("vae_mu_b", "latent_z_b", "latent_decoder_z_scale"):
        if key in decoder:
            value = decoder[key]
            if np.asarray(value).ndim == 0 and key == "latent_decoder_z_scale":
                continue
            return int(np.asarray(value).shape[0])
    if "basis_dec_w3" in decoder:
        action_dim = 6
        return int(np.asarray(decoder["basis_dec_w3"]).shape[-1] // action_dim)
    return None


METHOD_SPECS = {
    "iql": {
        "subdir": "iql",
        "file_prefix": "iql_cnn",
        "module": iql_cnn_overcooked,
    },
    "iql_double": {
        "subdir": "iql_double",
        "file_prefix": "iql_double_cnn",
        "module": iql_double_cnn_overcooked,
    },
    "iql_dueling": {
        "subdir": "iql_dueling",
        "file_prefix": "iql_dueling_cnn",
        "module": iql_dueling_cnn_overcooked,
    },
    "vdn": {
        "subdir": "vdn",
        "file_prefix": "vdn_cnn",
        "module": vdn_cnn_overcooked,
    },
    "vdn_dueling": {
        "subdir": "vdn_dueling",
        "file_prefix": "vdn_dueling_cnn",
        "module": vdn_dueling_cnn_overcooked,
    },
    "pqn_vdn": {
        "subdir": "pqn_vdn",
        "file_prefix": "pqn_vdn_cnn",
        "module": pqn_vdn_cnn_overcooked,
    },
    "a2c": {
        "subdir": "a2c",
        "file_prefix": "a2c_cnn",
        "module": a2c_cnn_overcooked,
    },
    "coma": {
        "subdir": "coma",
        "file_prefix": "coma_cnn",
        "module": coma_cnn_overcooked,
    },
    "qmix_cnn": {
        "subdir": "qmix_cnn",
        "file_prefix": "qmix_cnn",
        "module": qmix_cnn_overcooked,
    },
    "qplex": {
        "subdir": "qplex",
        "file_prefix": "qplex_cnn",
        "module": qplex_cnn_overcooked,
    },
    "wqmix": {
        "subdir": "wqmix",
        "file_prefix": "wqmix_cnn",
        "module": wqmix_cnn_overcooked,
    },
    "qmix_rnn": {
        "subdir": "qmix_rnn",
        "file_prefix": "qmix_rnn",
        "module": qmix_rnn,
    },
    "shaq_ps": {
        "subdir": "shaq_ps",
        "file_prefix": "shaq_ps",
        "module": shaq,
    },
}


@dataclass(frozen=True)
class LoadedQMethod:
    method: str
    params: list[dict]
    config: dict
    env_kwargs: dict
    apply_one: Callable
    labels: list[str]
    is_rnn: bool = False
    hidden_size: int | None = None
    preprocess_flat_obs: bool = False


class QLearningPolicy(AbstractPolicy):
    def __init__(
        self,
        params,
        apply_one,
        action_mode: str = "greedy",
        temperature: float = 1.0,
        epsilon: float = 0.0,
        is_rnn: bool = False,
        hidden_size: int | None = None,
        agent_index: int = 0,
        num_agents: int = 2,
        preprocess_flat_obs: bool = False,
    ):
        self.params = params
        self.apply_one = apply_one
        self.action_mode = action_mode
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.is_rnn = bool(is_rnn)
        self.hidden_size = hidden_size
        self.agent_index = int(agent_index)
        self.num_agents = int(num_agents)
        self.preprocess_flat_obs = bool(preprocess_flat_obs)

    @staticmethod
    def _masked_q(q_values, valid_actions):
        return q_values - (1.0 - valid_actions.astype(q_values.dtype)) * 1e10

    @staticmethod
    def _masked_argmax(q_values, valid_actions):
        return jnp.argmax(QLearningPolicy._masked_q(q_values, valid_actions), axis=-1).astype(jnp.int32)

    @staticmethod
    def _masked_softmax_sample(key, q_values, valid_actions, temperature):
        logits = QLearningPolicy._masked_q(q_values, valid_actions) / jnp.maximum(
            jnp.asarray(temperature), 1e-6
        )
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)

    @staticmethod
    def _masked_random_sample(key, valid_actions):
        logits = jnp.where(valid_actions > 0, 0.0, -1e10)
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)

    @staticmethod
    def _select_actions(key, q_values, valid_actions, action_mode, temperature, epsilon):
        greedy = QLearningPolicy._masked_argmax(q_values, valid_actions)
        if action_mode == "greedy":
            return greedy
        if action_mode == "softmax":
            return QLearningPolicy._masked_softmax_sample(key, q_values, valid_actions, temperature)
        if action_mode == "epsilon_greedy":
            key_eps, key_rand = jax.random.split(key)
            random_actions = QLearningPolicy._masked_random_sample(key_rand, valid_actions)
            explore = jax.random.uniform(key_eps, greedy.shape) < epsilon
            return jnp.where(explore, random_actions, greedy).astype(jnp.int32)
        raise ValueError(f"unknown q action mode: {action_mode}")

    @partial(jax.jit, static_argnums=(0,))
    def compute_action(self, obs, done, hstate, key):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        if self.preprocess_flat_obs:
            agent_id = jax.nn.one_hot(self.agent_index, self.num_agents, dtype=jnp.float32)
            obs = jnp.concatenate([jnp.ravel(obs), agent_id], axis=-1)
        if self.is_rnn:
            if hstate is None:
                hstate = self.init_hstate(1)
            hstate, q_values = self.apply_one(self.params, hstate, obs, done)
        else:
            del done, hstate
            q_values = self.apply_one(self.params, obs)
            hstate = None
        valid_actions = jnp.ones_like(q_values)
        action = self._select_actions(
            key,
            q_values,
            valid_actions,
            self.action_mode,
            self.temperature,
            self.epsilon,
        )
        return action, hstate

    def init_hstate(self, batch_size, key=None):
        del key
        if self.is_rnn:
            return jnp.zeros((batch_size, int(self.hidden_size)), dtype=jnp.float32)
        return None

    def update_after_step(self, hstate, partner_obs, partner_action, done):
        del partner_obs, partner_action, done
        return hstate


def _natural_vmap_key(path: Path) -> int:
    match = re.search(r"_vmap(\d+)\.safetensors$", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _to_plain_config(config_path: Path) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    flat = dict(cfg)
    if isinstance(cfg.get("alg"), dict):
        flat.update(cfg["alg"])
    return flat


def load_q_method(run_root: Path, method: str, layout: str, max_policies: int | None) -> LoadedQMethod:
    if method not in METHOD_SPECS:
        raise ValueError(f"unknown Q-learning method {method}; expected {sorted(METHOD_SPECS)}")
    spec = METHOD_SPECS[method]
    env_name = f"overcooked_v2_{layout}"
    method_dir = run_root / spec["subdir"] / env_name
    if not method_dir.exists():
        raise FileNotFoundError(f"missing method directory: {method_dir}")

    config_files = sorted(method_dir.glob(f"{spec['file_prefix']}_{env_name}_seed*_config.yaml"))
    if not config_files:
        raise FileNotFoundError(f"missing config under {method_dir}")
    config = _to_plain_config(config_files[0])
    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    env_kwargs["layout"] = layout

    ckpts = sorted(
        method_dir.glob(f"{spec['file_prefix']}_{env_name}_seed*_vmap*.safetensors"),
        key=_natural_vmap_key,
    )
    if max_policies is not None:
        ckpts = ckpts[: max(0, int(max_policies))]
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints found under {method_dir}")

    env = OvercookedV2(**env_kwargs)
    action_dim = int(env.action_space(env.agents[0]).n)

    if method in {"iql", "iql_double", "iql_dueling", "vdn", "vdn_dueling"}:
        network = spec["module"].QNetwork(
            action_dim=action_dim,
            hidden_size=int(config.get("HIDDEN_SIZE", 64)),
        )

        def apply_one(params, obs):
            return network.apply(params, obs[jnp.newaxis, ...])[0]

        is_rnn = False
        hidden_size = None

    elif method in {"a2c", "coma"}:
        network = spec["module"].ActorCritic(
            action_dim=action_dim,
            activation=str(config.get("ACTIVATION", "relu")),
        )

        def apply_one(params, obs):
            actor_params = params["actor"] if isinstance(params, dict) and "actor" in params else params
            pi, _ = network.apply(actor_params, obs[jnp.newaxis, ...])
            return pi.logits[0]

        is_rnn = False
        hidden_size = None

    elif method == "pqn_vdn":
        network = spec["module"].QNetwork(
            action_dim=action_dim,
            hidden_size=int(config.get("HIDDEN_SIZE", 512)),
            num_layers=int(config.get("NUM_LAYERS", 2)),
            norm_type=str(config.get("NORM_TYPE", "layer_norm")),
            norm_input=bool(config.get("NORM_INPUT", False)),
        )
        dummy_obs = jnp.zeros((1, *env.observation_space().shape), dtype=jnp.float32)
        init_vars = network.init(jax.random.PRNGKey(0), dummy_obs, train=False)
        batch_stats = init_vars.get("batch_stats", {})

        def apply_one(params, obs):
            variables = {"params": params, "batch_stats": batch_stats}
            return network.apply(variables, obs[jnp.newaxis, ...], False)[0]

        is_rnn = False
        hidden_size = None

    elif method in {"qmix_cnn", "qplex", "wqmix"}:
        network = spec["module"].QNetwork(
            action_dim=action_dim,
            hidden_size=int(config.get("HIDDEN_SIZE", 64)),
        )

        def apply_one(params, obs):
            agent_params = params["agent"] if isinstance(params, dict) and "agent" in params else params
            return network.apply(agent_params, obs[jnp.newaxis, ...])[0]

        is_rnn = False
        hidden_size = None

    elif method == "qmix_rnn":
        hidden_size = int(config.get("HIDDEN_SIZE", 64))
        network_cls = (
            spec["module"].CNNRNNQNetwork
            if bool(config.get("USE_CNN", False))
            else spec["module"].RNNQNetwork
        )
        network = network_cls(
            action_dim=action_dim,
            hidden_dim=hidden_size,
        )

        def apply_one(params, hstate, obs, done):
            agent_params = params["agent"] if isinstance(params, dict) and "agent" in params else params
            obs_seq = obs[jnp.newaxis, jnp.newaxis, :]
            done_seq = jnp.asarray(done, dtype=jnp.bool_).reshape(1, 1)
            next_hstate, q_values = network.apply(agent_params, hstate, obs_seq, done_seq)
            return next_hstate, q_values[0, 0]

        is_rnn = True

    elif method == "shaq_ps":
        hidden_size = int(config.get("AGENT_HIDDEN_DIM", 64))
        network = spec["module"].AgentRNN(
            action_dim=action_dim,
            hidden_dim=hidden_size,
            init_scale=float(config.get("AGENT_INIT_SCALE", 2.0)),
        )

        def apply_one(params, hstate, obs, done):
            obs_seq = obs[jnp.newaxis, jnp.newaxis, :]
            done_seq = jnp.asarray(done, dtype=jnp.bool_).reshape(1, 1)
            next_hstate, q_values = network.apply(params, hstate, (obs_seq, done_seq))
            return next_hstate, q_values[0, 0]

        is_rnn = True

    else:
        raise ValueError(f"unsupported method {method}")

    params = [load_params(path) for path in ckpts]
    labels = [f"run_{_natural_vmap_key(path)}" for path in ckpts]
    preprocess_flat_obs = bool(is_rnn and not config.get("USE_CNN", False))
    return LoadedQMethod(
        method,
        params,
        config,
        env_kwargs,
        apply_one,
        labels,
        is_rnn,
        hidden_size,
        preprocess_flat_obs,
    )


def load_ego_policies(
    run_dir: Path,
    eval_mode: str,
    max_policies: int | None,
    stochastic: bool,
    estimator_path: Path | None,
    latent_decoder_path: Path | None,
    model_overrides: dict,
) -> tuple[list[tuple[str, PPOPolicy]], dict]:
    all_params, config = load_all_checkpoints(run_dir, final_only=True)
    config = copy.deepcopy(config)
    if "model" not in config:
        raise KeyError("ego checkpoint config is missing model section")

    for key, value in model_overrides.items():
        if value is not None:
            config["model"][key] = value

    if eval_mode.startswith("ttac_v5_8_latent_decoder_"):
        if latent_decoder_path is None:
            raise ValueError("ttac_v5_8_latent_decoder modes require --ttac_latent_decoder_path")
        latent_decoder_bundle = load_latent_decoder_bundle(latent_decoder_path)
        config["model"]["TTAC_LATENT_DECODER"] = latent_decoder_bundle
        z_dim = infer_latent_decoder_z_dim(latent_decoder_bundle)
        if z_dim is not None:
            config["model"]["TTAC_V5_8_LATENT_DIM"] = int(z_dim)
    elif estimator_path is not None:
        config["model"]["TTAC_V5_ESTIMATOR"] = load_agreement_npz(estimator_path)

    policies = []
    run_keys = sorted(all_params.keys(), key=lambda x: int(x.split("_")[1]))
    if eval_mode.startswith("ttac_policy_bank"):
        bank_indices = config["model"].get("TTAC_POLICY_BANK_INDICES", None)
        bank_keys = run_keys
        if bank_indices is not None:
            if isinstance(bank_indices, str):
                bank_indices = {
                    int(idx.strip()) for idx in bank_indices.split(",") if idx.strip()
                }
            else:
                bank_indices = {int(idx) for idx in bank_indices}
            bank_keys = [
                k for k in bank_keys if int(str(k).split("_")[1]) in bank_indices
            ]
            if not bank_keys:
                raise ValueError("TTAC_POLICY_BANK_INDICES selected an empty bank.")
        config["model"]["TTAC_POLICY_BANK_PARAMS"] = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0),
            *[all_params[k]["ckpt_final"].params for k in bank_keys],
        )
        print(f"TTAC policy bank keys for {eval_mode}: {bank_keys}", flush=True)
    if max_policies is not None:
        run_keys = run_keys[: max(0, int(max_policies))]
    for run_key in run_keys:
        ckpt = all_params[run_key].get("ckpt_final")
        if ckpt is None:
            continue
        policies.append(
            (
                run_key,
                PPOPolicy(
                    ckpt.params,
                    config,
                    stochastic=stochastic,
                    eval_mode=eval_mode,
                ),
            )
        )
    if not policies:
        raise FileNotFoundError(f"no ego policies loaded from {run_dir}")
    return policies, config


def _is_oracle_partner_mode(eval_mode: str) -> bool:
    return eval_mode.startswith("ttac_oracle_partner_")


def make_oracle_partner_ego_policy(
    ego_policy: PPOPolicy,
    eval_mode: str,
    partner_params,
    apply_one: Callable,
    action_mode: str,
    temperature: float,
    epsilon: float,
    preprocess_flat_obs: bool,
    agent_index: int,
) -> PPOPolicy:
    config = copy.deepcopy(ego_policy.config)
    config["model"]["TTAC_ORACLE_Q_PARAMS"] = partner_params
    config["model"]["TTAC_ORACLE_Q_APPLY_ONE"] = apply_one
    config["model"]["TTAC_ORACLE_ACTION_MODE"] = action_mode
    config["model"]["TTAC_ORACLE_TEMPERATURE"] = float(temperature)
    config["model"]["TTAC_ORACLE_EPSILON"] = float(epsilon)
    config["model"]["TTAC_ORACLE_PREPROCESS_FLAT_OBS"] = bool(preprocess_flat_obs)
    config["model"]["TTAC_ORACLE_AGENT_INDEX"] = int(agent_index)
    config["model"]["TTAC_ORACLE_NUM_AGENTS"] = 2
    return PPOPolicy(
        ego_policy.params,
        config,
        stochastic=ego_policy.stochastic,
        eval_mode=eval_mode,
    )


def load_mappo_partner_policies(
    run_dir: Path,
    max_policies: int | None,
    stochastic: bool,
) -> tuple[list[tuple[str, MAPPOPolicy]], dict]:
    all_params, config = load_mappo_checkpoints(run_dir, final_only=True)
    config = copy.deepcopy(config)
    policies = []
    run_keys = sorted(all_params.keys(), key=lambda x: int(x.split("_")[1]))
    if max_policies is not None:
        run_keys = run_keys[: max(0, int(max_policies))]
    for run_key in run_keys:
        ckpt = all_params[run_key].get("ckpt_final")
        if ckpt is None:
            continue
        policies.append((run_key, MAPPOPolicy(ckpt.params, config, stochastic=stochastic)))
    if not policies:
        raise FileNotFoundError(f"no MAPPO policies loaded from {run_dir}")
    return policies, config


def evaluate_one_pairing(
    pairing: PolicyPairing,
    layout: str,
    env_kwargs: dict,
    seed: int,
    num_eval_seeds: int,
) -> list[tuple[str, float]]:
    env_kwargs = dict(env_kwargs)
    env_kwargs.pop("layout", None)
    result = eval_pairing(
        pairing,
        layout,
        jax.random.PRNGKey(seed),
        env_kwargs=env_kwargs,
        num_seeds=num_eval_seeds,
        no_viz=True,
    )
    rows = []
    for annotation, viz in result.items():
        rows.append((annotation, float(np.asarray(viz.total_reward))))
    return rows


def summarize(rows: Sequence[dict]) -> list[dict]:
    q_methods = {
        "iql",
        "iql_double",
        "iql_dueling",
        "vdn",
        "vdn_dueling",
        "pqn_vdn",
        "qmix_cnn",
        "qplex",
        "wqmix",
        "qmix_rnn",
        "shaq_ps",
    }
    pg_methods = {"a2c", "coma", "ppo_cnn_standard", "mappo_cnn_standard"}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["ego_mode"], row["partner_method"], row["role"])].append(row)
        grouped[(row["ego_mode"], row["partner_method"], "both_roles")].append(row)
        grouped[(row["ego_mode"], "all_partners", row["role"])].append(row)
        grouped[(row["ego_mode"], "all_partners", "both_roles")].append(row)
        if row["partner_method"] in q_methods:
            grouped[(row["ego_mode"], "all_q_partners", row["role"])].append(row)
            grouped[(row["ego_mode"], "all_q_partners", "both_roles")].append(row)
        if row["partner_method"] in pg_methods:
            grouped[(row["ego_mode"], "all_pg_partners", row["role"])].append(row)
            grouped[(row["ego_mode"], "all_pg_partners", "both_roles")].append(row)

    summary = []
    for (ego_mode, partner_method, role), items in sorted(grouped.items()):
        rewards = np.asarray([float(x["total_reward"]) for x in items], dtype=np.float64)
        pair_means_map = defaultdict(list)
        for item in items:
            pair_means_map[item["pair_id"]].append(float(item["total_reward"]))
        pair_means = np.asarray([np.mean(v) for v in pair_means_map.values()], dtype=np.float64)
        summary.append(
            {
                "ego_mode": ego_mode,
                "partner_method": partner_method,
                "role": role,
                "mean_reward": float(np.mean(rewards)) if rewards.size else float("nan"),
                "std_episode": float(np.std(rewards)) if rewards.size else float("nan"),
                "std_pair": float(np.std(pair_means)) if pair_means.size else float("nan"),
                "num_episodes": int(rewards.size),
                "num_pairs": int(pair_means.size),
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, summary_rows: Sequence[dict], args: argparse.Namespace):
    lines = [
        "# Mixed 1-ZSC Evaluation",
        "",
        "Definition: a PPO/TTAC ego policy is paired with non-ZSC Q-learning partners.",
        "Both ego roles are evaluated; `both_roles` averages agent-0 and agent-1 ego placements.",
        "",
        f"- Ego run dir: `{args.ego_run_dir}`",
        f"- Q partner root: `{args.q_run_root}`",
        f"- Methods: `{args.methods}`",
        f"- Extra partner methods: `{args.extra_partner_methods}`",
        f"- PPO partner run dir: `{args.ppo_partner_run_dir}`",
        f"- MAPPO partner run dir: `{args.mappo_partner_run_dir}`",
        f"- Ego modes: `{args.ego_modes}`",
        f"- Latent decoder path: `{args.ttac_latent_decoder_path}`",
        f"- Latent decoder history mode: `{args.ttac_latent_decoder_history_mode}`",
        f"- Latent decoder query mode: `{args.ttac_latent_decoder_query_mode}`",
        f"- Latent decoder gate mode: `{args.ttac_latent_decoder_gate_mode}`",
        f"- Latent decoder gate threshold: `{args.ttac_latent_decoder_gate_threshold}`",
        f"- Latent decoder gate scale: `{args.ttac_latent_decoder_gate_scale}`",
        f"- Latent decoder gate min: `{args.ttac_latent_decoder_gate_min}`",
        f"- Latent decoder online warmup steps: `{args.ttac_latent_decoder_online_warmup_steps}`",
        f"- Latent decoder online ramp steps: `{args.ttac_latent_decoder_online_ramp_steps}`",
        f"- Latent decoder online update interval: `{args.ttac_latent_decoder_online_update_interval}`",
        f"- Latent decoder online update steps: `{args.ttac_latent_decoder_online_update_steps}`",
        f"- Latent decoder online lr: `{args.ttac_latent_decoder_online_lr}`",
        f"- Latent decoder online prior coef: `{args.ttac_latent_decoder_online_prior_coef}`",
        f"- Latent decoder online min history: `{args.ttac_latent_decoder_online_min_history}`",
        f"- Latent decoder online init z mode: `{args.ttac_latent_decoder_online_init_z_mode}`",
        f"- Policy bank indices: `{args.ttac_policy_bank_indices}`",
        f"- Policy bank temp: `{args.ttac_policy_bank_temp}`",
        f"- Policy bank score mode: `{args.ttac_policy_bank_score_mode}`",
        f"- Layout: `{args.layout}`",
        f"- Eval seeds per pairing/role: `{args.num_eval_seeds}`",
        f"- Q action mode: `{args.q_action_mode}`",
        "",
        "| ego_mode | partner_method | role | mean_reward | std_episode | std_pair | num_pairs | num_episodes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {ego_mode} | {partner_method} | {role} | {mean_reward:.3f} | {std_episode:.3f} | {std_pair:.3f} | {num_pairs} | {num_episodes} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ego_run_dir",
        type=Path,
        default=Path("runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606"),
    )
    parser.add_argument(
        "--q_run_root",
        type=Path,
        default=Path("runs/qlearning_ov2_1zsc_20260612_qlearning_1zsc_10M"),
    )
    parser.add_argument("--methods", type=str, default="iql,vdn,pqn_vdn")
    parser.add_argument(
        "--extra_partner_methods",
        type=str,
        default="",
        help="Comma-separated extra partner pools. Supported: ppo_cnn_standard,mappo_cnn_standard.",
    )
    parser.add_argument(
        "--ppo_partner_run_dir",
        type=Path,
        default=Path("runs/ttac_posthoc_zero_adapter_from_ppo_standard_no_state_aug_64_16_seed42_10seeds_no_state_20260622_133910"),
    )
    parser.add_argument(
        "--mappo_partner_run_dir",
        type=Path,
        default=Path("runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_standard_20260503-201254/20260503-201315_wvbb7dde_counter_circuit_avs-full"),
    )
    parser.add_argument("--ego_modes", type=str, default="base_no_test_adapt,ttac_v5_8_logit_bias")
    parser.add_argument(
        "--ttac_v5_estimator_path",
        type=Path,
        default=Path("reports/strategy_estimator_weighted_training_20260623_002530/strategy_estimator_weighted.npz"),
    )
    parser.add_argument("--ttac_latent_decoder_path", type=Path)
    parser.add_argument("--layout", type=str, default="counter_circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_seeds", type=int, default=100)
    parser.add_argument("--max_ego_policies", type=int, default=10)
    parser.add_argument("--max_partner_policies", type=int, default=10)
    parser.add_argument("--q_action_mode", type=str, default="greedy", choices=("greedy", "softmax", "epsilon_greedy"))
    parser.add_argument("--q_temperature", type=float, default=1.0)
    parser.add_argument("--q_epsilon", type=float, default=0.0)
    parser.add_argument("--ego_greedy", action="store_true")
    parser.add_argument("--ttac_v5_agreement_coef", type=float, default=20.0)
    parser.add_argument("--ttac_v5_support_coef", type=float, default=1.0)
    parser.add_argument("--ttac_history_len", type=int, default=50)
    parser.add_argument("--ttac_update_interval", type=int, default=1)
    parser.add_argument("--ttac_update_gate", type=str, default="none")
    parser.add_argument("--ttac_cache_estimator_target", type=int, default=1)
    parser.add_argument("--ttac_v5_8_logit_step_size", type=float, default=0.003)
    parser.add_argument("--ttac_v5_8_bias_clip", type=float, default=2.0)
    parser.add_argument("--ttac_v5_8_bias_decay", type=float, default=0.0)
    parser.add_argument("--ttac_policy_bank_indices", type=str)
    parser.add_argument("--ttac_policy_bank_history_len", type=int, default=32)
    parser.add_argument("--ttac_policy_bank_temp", type=float, default=1.0)
    parser.add_argument("--ttac_policy_bank_score_mode", type=str, default="sum_logp")
    parser.add_argument("--ttac_policy_bank_blend_alpha", type=float, default=0.5)
    parser.add_argument("--ttac_policy_bank_blend_clip", type=float, default=2.0)
    parser.add_argument("--ttac_latent_decoder_history_len", type=int, default=50)
    parser.add_argument(
        "--ttac_latent_decoder_history_mode",
        type=str,
        default="full",
        choices=("full", "action_only", "no_history"),
    )
    parser.add_argument("--ttac_latent_decoder_blend_alpha", type=float, default=0.5)
    parser.add_argument("--ttac_latent_decoder_blend_clip", type=float, default=2.0)
    parser.add_argument(
        "--ttac_latent_decoder_query_mode",
        type=str,
        default="real",
        choices=("real", "zero", "gaussian"),
    )
    parser.add_argument(
        "--ttac_latent_decoder_gate_mode",
        type=str,
        default="none",
        choices=("none", "sensitivity_no_history", "ensemble_uncertainty"),
    )
    parser.add_argument("--ttac_latent_decoder_gate_threshold", type=float, default=0.1)
    parser.add_argument("--ttac_latent_decoder_gate_scale", type=float, default=12.0)
    parser.add_argument("--ttac_latent_decoder_gate_min", type=float, default=0.0)
    parser.add_argument("--ttac_latent_decoder_online_warmup_steps", type=int, default=50)
    parser.add_argument("--ttac_latent_decoder_online_ramp_steps", type=int, default=30)
    parser.add_argument("--ttac_latent_decoder_online_update_interval", type=int, default=10)
    parser.add_argument("--ttac_latent_decoder_online_update_steps", type=int, default=10)
    parser.add_argument("--ttac_latent_decoder_online_lr", type=float, default=0.05)
    parser.add_argument("--ttac_latent_decoder_online_prior_coef", type=float, default=0.05)
    parser.add_argument("--ttac_latent_decoder_online_min_history", type=int, default=10)
    parser.add_argument("--ttac_latent_decoder_online_z_clip", type=float, default=5.0)
    parser.add_argument(
        "--ttac_latent_decoder_online_init_z_mode",
        type=str,
        default="encoder",
        choices=("encoder", "zero", "no_history"),
    )
    parser.add_argument("--ttac_oracle_blend_alpha", type=float, default=0.5)
    parser.add_argument("--ttac_oracle_blend_clip", type=float, default=2.0)
    parser.add_argument("--ttac_oracle_action_mode", type=str, default="greedy", choices=("greedy", "softmax", "epsilon_greedy"))
    parser.add_argument("--ttac_oracle_temperature", type=float, default=1.0)
    parser.add_argument("--ttac_oracle_epsilon", type=float, default=0.0)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    extra_partner_methods = [x.strip() for x in args.extra_partner_methods.split(",") if x.strip()]
    ego_modes = [x.strip() for x in args.ego_modes.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_overrides = {
        "TTAC_V5_ESTIMATOR": None,
        "TTAC_V5_AGREEMENT_COEF": args.ttac_v5_agreement_coef,
        "TTAC_V5_SUPPORT_COEF": args.ttac_v5_support_coef,
        "TTAC_HISTORY_LEN": args.ttac_history_len,
        "TTAC_UPDATE_INTERVAL": args.ttac_update_interval,
        "TTAC_UPDATE_GATE": args.ttac_update_gate,
        "TTAC_CACHE_ESTIMATOR_TARGET": bool(args.ttac_cache_estimator_target),
        "TTAC_V5_8_LOGIT_STEP_SIZE": args.ttac_v5_8_logit_step_size,
        "TTAC_V5_8_BIAS_CLIP": args.ttac_v5_8_bias_clip,
        "TTAC_V5_8_BIAS_DECAY": args.ttac_v5_8_bias_decay,
        "TTAC_POLICY_BANK_INDICES": args.ttac_policy_bank_indices,
        "TTAC_POLICY_BANK_HISTORY_LEN": args.ttac_policy_bank_history_len,
        "TTAC_POLICY_BANK_TEMP": args.ttac_policy_bank_temp,
        "TTAC_POLICY_BANK_SCORE_MODE": args.ttac_policy_bank_score_mode,
        "TTAC_POLICY_BANK_BLEND_ALPHA": args.ttac_policy_bank_blend_alpha,
        "TTAC_POLICY_BANK_BLEND_CLIP": args.ttac_policy_bank_blend_clip,
        "TTAC_LATENT_DECODER_HISTORY_LEN": args.ttac_latent_decoder_history_len,
        "TTAC_LATENT_DECODER_HISTORY_MODE": args.ttac_latent_decoder_history_mode,
        "TTAC_LATENT_DECODER_BLEND_ALPHA": args.ttac_latent_decoder_blend_alpha,
        "TTAC_LATENT_DECODER_BLEND_CLIP": args.ttac_latent_decoder_blend_clip,
        "TTAC_LATENT_DECODER_QUERY_MODE": args.ttac_latent_decoder_query_mode,
        "TTAC_LATENT_DECODER_GATE_MODE": args.ttac_latent_decoder_gate_mode,
        "TTAC_LATENT_DECODER_GATE_THRESHOLD": args.ttac_latent_decoder_gate_threshold,
        "TTAC_LATENT_DECODER_GATE_SCALE": args.ttac_latent_decoder_gate_scale,
        "TTAC_LATENT_DECODER_GATE_MIN": args.ttac_latent_decoder_gate_min,
        "TTAC_LATENT_DECODER_ONLINE_WARMUP_STEPS": args.ttac_latent_decoder_online_warmup_steps,
        "TTAC_LATENT_DECODER_ONLINE_RAMP_STEPS": args.ttac_latent_decoder_online_ramp_steps,
        "TTAC_LATENT_DECODER_ONLINE_UPDATE_INTERVAL": args.ttac_latent_decoder_online_update_interval,
        "TTAC_LATENT_DECODER_ONLINE_UPDATE_STEPS": args.ttac_latent_decoder_online_update_steps,
        "TTAC_LATENT_DECODER_ONLINE_LR": args.ttac_latent_decoder_online_lr,
        "TTAC_LATENT_DECODER_ONLINE_PRIOR_COEF": args.ttac_latent_decoder_online_prior_coef,
        "TTAC_LATENT_DECODER_ONLINE_MIN_HISTORY": args.ttac_latent_decoder_online_min_history,
        "TTAC_LATENT_DECODER_ONLINE_Z_CLIP": args.ttac_latent_decoder_online_z_clip,
        "TTAC_LATENT_DECODER_ONLINE_INIT_Z_MODE": args.ttac_latent_decoder_online_init_z_mode,
        "TTAC_ORACLE_BLEND_ALPHA": args.ttac_oracle_blend_alpha,
        "TTAC_ORACLE_BLEND_CLIP": args.ttac_oracle_blend_clip,
    }

    all_rows: list[dict] = []
    for ego_mode in ego_modes:
        uses_latent_decoder = ego_mode.startswith("ttac_v5_8_latent_decoder_")
        uses_oracle_partner = _is_oracle_partner_mode(ego_mode)
        uses_policy_bank = ego_mode.startswith("ttac_policy_bank")
        estimator_path = (
            args.ttac_v5_estimator_path
            if ego_mode != "base_no_test_adapt"
            and not uses_latent_decoder
            and not uses_oracle_partner
            and not uses_policy_bank
            else None
        )
        latent_decoder_path = args.ttac_latent_decoder_path if uses_latent_decoder else None
        ego_policies, ego_config = load_ego_policies(
            args.ego_run_dir,
            ego_mode,
            args.max_ego_policies,
            stochastic=not args.ego_greedy,
            estimator_path=estimator_path,
            latent_decoder_path=latent_decoder_path,
            model_overrides=model_overrides,
        )
        ego_env_kwargs = dict(ego_config["env"]["ENV_KWARGS"])
        ego_layout = str(ego_env_kwargs.get("layout", args.layout))
        if ego_layout != args.layout:
            raise ValueError(f"ego layout {ego_layout} does not match requested layout {args.layout}")

        for method in methods:
            loaded_q = load_q_method(
                args.q_run_root,
                method,
                args.layout,
                args.max_partner_policies,
            )
            if uses_oracle_partner and loaded_q.is_rnn:
                raise ValueError("oracle partner direct blend currently supports feed-forward Q partners only")
            for ego_idx, (ego_label, ego_policy) in enumerate(ego_policies):
                for partner_idx, (partner_label, partner_params) in enumerate(zip(loaded_q.labels, loaded_q.params)):
                    ego_policy_agent0 = ego_policy
                    ego_policy_agent1 = ego_policy
                    if uses_oracle_partner:
                        ego_policy_agent0 = make_oracle_partner_ego_policy(
                            ego_policy,
                            ego_mode,
                            partner_params,
                            loaded_q.apply_one,
                            args.ttac_oracle_action_mode,
                            args.ttac_oracle_temperature,
                            args.ttac_oracle_epsilon,
                            loaded_q.preprocess_flat_obs,
                            agent_index=0,
                        )
                        ego_policy_agent1 = make_oracle_partner_ego_policy(
                            ego_policy,
                            ego_mode,
                            partner_params,
                            loaded_q.apply_one,
                            args.ttac_oracle_action_mode,
                            args.ttac_oracle_temperature,
                            args.ttac_oracle_epsilon,
                            loaded_q.preprocess_flat_obs,
                            agent_index=1,
                        )
                    pairings = {
                        "ego_agent0": PolicyPairing(
                            ego_policy_agent0,
                            QLearningPolicy(
                                partner_params,
                                loaded_q.apply_one,
                                action_mode=args.q_action_mode,
                                temperature=args.q_temperature,
                                epsilon=args.q_epsilon,
                                is_rnn=loaded_q.is_rnn,
                                hidden_size=loaded_q.hidden_size,
                                agent_index=1,
                                preprocess_flat_obs=loaded_q.preprocess_flat_obs,
                            ),
                        ),
                        "ego_agent1": PolicyPairing(
                            QLearningPolicy(
                                partner_params,
                                loaded_q.apply_one,
                                action_mode=args.q_action_mode,
                                temperature=args.q_temperature,
                                epsilon=args.q_epsilon,
                                is_rnn=loaded_q.is_rnn,
                                hidden_size=loaded_q.hidden_size,
                                agent_index=0,
                                preprocess_flat_obs=loaded_q.preprocess_flat_obs,
                            ),
                            ego_policy_agent1,
                        ),
                    }
                    for role, pairing in pairings.items():
                        pair_seed = (
                            args.seed
                            + 100000 * ego_idx
                            + 1000 * partner_idx
                            + (0 if role == "ego_agent0" else 500)
                        )
                        pair_id = f"{ego_mode}__{method}__{ego_label}__{partner_label}__{role}"
                        print(f"Evaluating {pair_id} with {args.num_eval_seeds} seeds", flush=True)
                        eval_rows = evaluate_one_pairing(
                            pairing,
                            args.layout,
                            ego_env_kwargs,
                            pair_seed,
                            args.num_eval_seeds,
                        )
                        for annotation, reward in eval_rows:
                            all_rows.append(
                                {
                                    "ego_mode": ego_mode,
                                    "partner_method": method,
                                    "ego_label": ego_label,
                                    "partner_label": partner_label,
                                    "role": role,
                                    "pair_id": pair_id,
                                    "annotation": annotation,
                                    "total_reward": reward,
                                }
                            )

        for partner_method in extra_partner_methods:
            if uses_oracle_partner:
                raise ValueError("oracle partner direct blend is only implemented for Q partner methods")
            if partner_method == "ppo_cnn_standard":
                partner_policies, partner_config = load_ego_policies(
                    args.ppo_partner_run_dir,
                    "base_no_test_adapt",
                    args.max_partner_policies,
                    stochastic=not args.ego_greedy,
                    estimator_path=None,
                    latent_decoder_path=None,
                    model_overrides=model_overrides,
                )
            elif partner_method == "mappo_cnn_standard":
                partner_policies, partner_config = load_mappo_partner_policies(
                    args.mappo_partner_run_dir,
                    args.max_partner_policies,
                    stochastic=not args.ego_greedy,
                )
            else:
                raise ValueError(
                    f"unknown extra partner method {partner_method}; "
                    "supported: ppo_cnn_standard,mappo_cnn_standard"
                )
            partner_layout = str(partner_config["env"]["ENV_KWARGS"].get("layout", args.layout))
            if partner_layout != args.layout:
                raise ValueError(
                    f"partner layout {partner_layout} for {partner_method} does not match {args.layout}"
                )

            for ego_idx, (ego_label, ego_policy) in enumerate(ego_policies):
                for partner_idx, (partner_label, partner_policy) in enumerate(partner_policies):
                    pairings = {
                        "ego_agent0": PolicyPairing(ego_policy, partner_policy),
                        "ego_agent1": PolicyPairing(partner_policy, ego_policy),
                    }
                    for role, pairing in pairings.items():
                        pair_seed = (
                            args.seed
                            + 100000 * ego_idx
                            + 1000 * partner_idx
                            + (0 if role == "ego_agent0" else 500)
                        )
                        pair_id = f"{ego_mode}__{partner_method}__{ego_label}__{partner_label}__{role}"
                        print(f"Evaluating {pair_id} with {args.num_eval_seeds} seeds", flush=True)
                        eval_rows = evaluate_one_pairing(
                            pairing,
                            args.layout,
                            ego_env_kwargs,
                            pair_seed,
                            args.num_eval_seeds,
                        )
                        for annotation, reward in eval_rows:
                            all_rows.append(
                                {
                                    "ego_mode": ego_mode,
                                    "partner_method": partner_method,
                                    "ego_label": ego_label,
                                    "partner_label": partner_label,
                                    "role": role,
                                    "pair_id": pair_id,
                                    "annotation": annotation,
                                    "total_reward": reward,
                                }
                            )

    raw_fields = [
        "ego_mode",
        "partner_method",
        "ego_label",
        "partner_label",
        "role",
        "pair_id",
        "annotation",
        "total_reward",
    ]
    summary_rows = summarize(all_rows)
    summary_fields = [
        "ego_mode",
        "partner_method",
        "role",
        "mean_reward",
        "std_episode",
        "std_pair",
        "num_episodes",
        "num_pairs",
    ]
    write_csv(args.output_dir / "mixed_1zsc_rows.csv", all_rows, raw_fields)
    write_csv(args.output_dir / "mixed_1zsc_summary.csv", summary_rows, summary_fields)
    write_markdown(args.output_dir / "mixed_1zsc_summary.md", summary_rows, args)
    print(f"Wrote {args.output_dir / 'mixed_1zsc_summary.md'}")


if __name__ == "__main__":
    main()
