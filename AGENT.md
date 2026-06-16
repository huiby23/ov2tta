# AGENT.md

## Remote Workspace
- Authoritative workspace: `/teams/ius_1663576043/hby/rl/ov2`
- Current temporary SSH: `ssh -p 26595 root@hz-t3.matpool.com`
- Previous SSH: `ssh -p 29937 root@hz-4.matpool.com`
- Previous SSH: `ssh -p 29207 root@hz-4.matpool.com`
- Historical SSH: `ssh -p 29887 root@hz-t3.matpool.com`
- Current hardware checked 2026-05-30: `2 x Tesla V100-SXM2 16GB`; do not assume 32GB/48GB single-GPU memory.
- Use this temporary server for future OV2 experiments unless the user explicitly changes it.
- Do not use `/teams/ius_1663576043/hby/rl/ov2_o2_clean` for this project.

## Experiment Discipline
- Prefer standard/no-state-aug as the main method-performance comparison for population methods.
- Treat state-aug as a coverage/control supplement, not the primary ranking criterion.
- Keep run names identifiable with method, setting, seed count, env/minibatch config, and timestamp.
- Use GitHub for versioning, not extra clean-copy folders.


## Terminology Guardrail
- `state-aug` means state-sampling augmentation: periodically rollout current policies, collect `rollout.state_seq`, sample states, pass them as `env.ENV_KWARGS.initial_state_buffer`, then continue training from the previous train state.
- `agent_view_size` / `avs-full` / `obs_view` means observation scope only. It is not state augmentation and must not be used as a proxy for `state_aug`.
- When reporting experiments, keep `state_aug` and `obs_view` as separate fields.
- Do not infer `state_aug` from `avs-full`; many no-state-aug runs also use `obs_view=full`.

## Current Environment Notes
- Default CUDA device visibility should be single GPU: `CUDA_VISIBLE_DEVICES=0`.
- On the temporary `2 x 16GB` V100 server, prefer conservative configs such as `64 env / 16 minibatch`, reduce VAE batch sizes if needed, and avoid heavy diagnostics parallelism.
- Python env: `/root/miniconda3/envs/myconda/bin/python`
- Repro env script: `experiments/repro_env.sh`
- Typical working command prefix:
  `cd /teams/ius_1663576043/hby/rl/ov2 && source experiments/repro_env.sh && export PYTHONPATH=/teams/ius_1663576043/hby/rl/ov2/experiments:/teams/ius_1663576043/hby/rl/ov2/JaxMARL:${PYTHONPATH:-}`

Last updated: 2026-05-30 Asia/Shanghai.
