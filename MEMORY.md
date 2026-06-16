# MEMORY.md

## 2026-05-23 09:45:39 Asia/Shanghai
- Server migrated to `ssh -p 29207 root@hz-4.matpool.com`.
- This server has one NVIDIA A40 and an existing OV2 workspace at `/teams/ius_1663576043/hby/rl/ov2`.
- Previous `hz-t3.matpool.com:29887` should not be assumed available.
- Existing remote workspace contains `runs`, `logs`, `reports`, `wandb`, and code through at least the 2026-05-17 population state-aug experiments.
- Updated `experiments/repro_env.sh` default `CUDA_VISIBLE_DEVICES` from `0,1` to `0` for the single-A40 server.

## 2026-05-23 13:22:05 Asia/Shanghai
- Corrected a recurring terminology error: `state-aug` in this repository is rollout state-sampling augmentation through `initial_state_buffer`, not `agent_view_size` or `avs-full`.
- `agent_view_size` / `obs_view` only describes observation scope. It must be tracked independently from `state_aug` in experiment tables and analysis.
- Regenerated `reports/current_complete_results_with_diagnostics_20260523.md` and `.csv` with explicit `state_aug` and `obs_view` columns.

## 2026-05-25 02:43:18 Asia/Shanghai
- Server migrated to `ssh -p 29937 root@hz-4.matpool.com`.
- Verified `/teams/ius_1663576043/hby/rl/ov2` exists on the new server.
- GPU check at connection time: one NVIDIA RTX A6000, 49140 MiB total, idle.
- Previous active server was `ssh -p 29207 root@hz-4.matpool.com`; older historical server was `ssh -p 29887 root@hz-t3.matpool.com`.
- Future OV2 experiment work should use `hz-4.matpool.com:29937` unless the user explicitly changes it.
