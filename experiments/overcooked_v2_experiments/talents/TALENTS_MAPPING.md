# TALENTS to OV2 Migration Mapping

This directory is a paper-faithful TALENTS-style migration into the OV2 training/eval stack. It is not an official-code reproduction because no official TALENTS implementation is bundled in this repository.

## Paper Modules

| TALENTS component | OV2 implementation | Notes |
| --- | --- | --- |
| Strategy trajectory collection | `collect_dataset.py` | Wraps the existing OV2/GAMMA trajectory collector over a trained source population. |
| Sequential strategy VAE | `train_vae.py`, `models/strategy_vae.py` | Uses the existing sequential GammaVAE architecture with TALENTS defaults: latent dim 8, window/chunk length 50, hidden size 256, and KL beta linearly annealed from 0.0 to 0.05. |
| Latent strategy clusters | `cluster_latents.py` | Encodes trajectory chunks and runs deterministic K-means. `num_clusters=0` searches K with silhouette. |
| Generated partner policy | `policy.py::TalentsGeneratedPartnerPolicy` | Implements Algorithm 1 generated partner `p(a_partner | z_c, o_t)` using the VAE decoder and sampled cluster latents. |
| Strategy-conditioned cooperator | `models/cnn.py`, `ippo.py`, `run.py` | PPO CNN backbone plus cluster embedding -> action-bias head added to actor logits with `ACTION_BIAS_WEIGHT=2.0`. Critic remains strategy-agnostic. |
| Test-time fixed-share adaptation | `policy.py::TalentsPPOPolicy.update_after_step` | Implements Algorithm 2 exponential-weights plus fixed-share belief update over clusters from observed partner action likelihood. |

## Structural Choices

- The actor receives a sampled strategy cluster during TALENTS cooperator training, matching Algorithm 1.
- The actor conditioning is a cluster embedding mapped to action-logit bias, then applied as `l_t + w_b b_c`, matching the paper description and appendix action-bias weight.
- The critic does not receive the strategy cluster by default, so value learning stays close to PPO CNN.
- The generated partner decoder carry is preserved while the assigned cluster is unchanged and reset only when the cluster changes or an episode terminates.
- At evaluation, a single TALENTS cooperator checkpoint is loaded. The policy updates its cluster belief online from partner actions; it does not deploy a policy pool selector.

## Boundaries

- This is a TALENTS-style module migration into OV2, not a byte-for-byte official reproduction.
- OV2 uses the six low-level action space, so decoder targets are OV2 low-level actions rather than Overcooked-AI high-level actions.
- `state-aug` is not enabled by default here. In this project, `state-aug` strictly means rollout state-sampling with `initial_state_buffer`, not `agent_view_size`.
- Source policies are used only to collect strategy trajectories and train the generated partner model. TALENTS evaluation uses one strategy-conditioned cooperator checkpoint.

## Entry Points

- Smoke artifacts only: `MODE=smoke experiments/run_talents_pipeline_64_16.sh`
- Full pipeline: `MODE=full SOURCE_RUN=/path/to/source_population experiments/run_talents_pipeline_64_16.sh`
