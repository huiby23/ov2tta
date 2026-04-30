# E3T State-Aug Official Mapping

This directory is the faithful E3T reproduction line for the Overcooked-v2/state-aug training framework. It intentionally does not reuse the previous failed E3T/TTAPPO-inspired modules.

Official source checked against: `/tmp/e3t_official_check_19033` from `yanxue7/E3T-Overcooked`.

## Module Mapping

| Official E3T module | Official file/function | Remote implementation |
| --- | --- | --- |
| `conv_and_mlp_embd(pre_X, action_reward)` | `baselines/common/models.py` | `models/cnn.py::_context_encoder` |
| `human_prob_pre(..., num_layers=2, tanh)` | `baselines/common/models.py` | `models/cnn.py::_human_prob_pre` |
| `conv_and_mlp_new(X, human_latent_pre)` decoder | `baselines/common/models.py` | `models/cnn.py::_decoder` |
| decoder partner-action logits `tf.layers.dense(..., 6, name="pi")` | `encoder_policies.py` | `models/cnn.py` module `decoder_pi`; returned as `partner_pi` |
| policy branch `conv_and_mlp(X, softmax(action_logits))` | `encoder_policies.py` | `models/cnn.py::_policy_network(..., "policy")` |
| value branch uses shared `policy_latent` | `encoder_policies.py` / `PolicyWithValue` | `models/cnn.py` `critic_out(policy_latent)` |
| human branch `step_human(context_human)` | `encoder_policies.py` / `ppo2_model_human` | `models/cnn.py::_policy_network(..., "human_policy")` plus `human_actor_out` |
| human target copy `var_human=(1-COPY)*var_human+COPY*var` | `baselines/ppo2/encoder_ppo2.py` | `ippo.py::_soft_update_human_policy`, `COPY=0.1` |
| separate CE context update | `encoder_model.py::traincontext` + `encoder_ppo2.py` CE loop | `ippo.py::_context_loss_fn`; full 8-epoch CE phase before PPO; independent Adam; no grad clipping |
| separate PPO update | `encoder_model.py::train` + `encoder_ppo2.py` PPO loop | `ippo.py::_ppo_loss_fn`; full 8-epoch PPO phase after CE; independent clipped Adam; CE not in PPO loss |
| runner resets history at each rollout | `encoder_runner.py::run` | `ippo.py` resets `rollout_history_obs/actions` at each update and does not reset history on mid-rollout done |
| history observation is self `obs0_pre`; history action is partner `action1` | `encoder_runner.py` | `ippo.py::_update_history_buffers(obs_batch, partner_action, ...)`; `policy.py` stores `last_self_obs` |
| partner self-play action from `step_human`, `more`, `probs`, `addrand` | `encoder_runner.py` | `ippo.py::_merge_train_and_partner_behavior`; random `context_human`, categorical sample, `RAND=0.7` uniform mixture |
| GAE bootstraps with `last_values = 0` | `encoder_runner.py` | `ippo.py` `last_val = jnp.zeros(...)` |

## Official Hyperparameters Ported

| Parameter | Official experiment | Remote config |
| --- | --- | --- |
| `LATENT_DIM` | 64 | 64 |
| `LENGTH` / `CONTEXT_LENGTH` | 5 | 5 |
| `TRAIN_MODE` | `MLP` | `MLP` |
| `NUM_HIDDEN_LAYERS` | 3 | 3 |
| `SIZE_HIDDEN_LAYERS` | 64 | 64 |
| `NUM_FILTERS` | 25 | 25 |
| `NUM_CONV_LAYERS` | 3 | 3 |
| `LR` | `1e-3` | `1e-3` |
| `PPO_RUN_TOT_TIMESTEPS` | `4.8e6` | `4.8e6` |
| `REW_SHAPING_HORIZON` | `2.5e6` | `2.5e6` |
| `VF_COEF` | 1 | 1 |
| `COPY` | 0.1 | 0.1 |
| `RAND` | 0.7 | 0.7 |
| conv/context/decoder/policy hidden init | TF `tf.layers` default Glorot uniform | Flax `xavier_uniform` |
| actor/value heads | Baselines `fc` / pd head init | orthogonal `0.01` for policy logits, orthogonal `1.0` for value |
| leaky ReLU slope | TensorFlow default `alpha=0.2` | Flax `negative_slope=0.2` |

## Intentional Framework Adaptation Boundaries

- The environment and outer experiment protocol remain Overcooked-v2/state-aug from this repository, not the original TensorFlow Overcooked-v0 environment.
- The implementation is JAX/Flax/Optax, not TensorFlow Baselines PPO2, but the module structure, parameter-update boundaries, minibatch semantics, and training/dataflow are matched to the official E3T code.
- The previous TTAPPO modules are not included: no memory GRU, FiLM modulation, actor adapter, online adaptation, or invented partner predictor.
- `ppo2_model_human` is implemented as the functionally used `step_human(context_human)` branch. The official TensorFlow graph also instantiates unused human context/decoder variables under this scope, but the non-`lasp` official experiment does not consume them for partner action sampling.
- Legacy `USE_PARTNER_MIX` and `MOA_COEF` switches are disabled in this path.

## Verification

- Syntax compile passed for `e3t_state_aug_official` Python files.
- Static check passed: no old `partner_predictor`, `MOA_TO_ACTOR`, CE-in-PPO loss, or single masked optimizer path remains.
- Smoke run passed with `WANDB_MODE=disabled ./experiments/run_e3t_state_aug_official_suite.sh smoke`.
- Latest successful smoke output saved checkpoints under `runs/smoke_e3t_state_aug_official/20260427-003231_u2ff8lpg_counter_circuit_avs-full/run_*/ckpt_final`.
