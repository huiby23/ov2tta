# SESSION_LOG.md

## 2026-05-23 09:45:39 Asia/Shanghai
- Confirmed new SSH host is reachable after refreshing local known_hosts.
- GPU check: one NVIDIA A40, idle at connection time.
- Confirmed project path exists: `/teams/ius_1663576043/hby/rl/ov2`.
- Added/updated project `AGENT.md` and `MEMORY.md` with new server/workspace requirements.

## 2026-05-23 13:22:05 Asia/Shanghai
- Audited `state-aug` implementation after user flagged repeated confusion.
- Confirmed `state-aug` is implemented by `state_sample_run.py`: collect rollout `state_seq`, sample states, inject via `initial_state_buffer`, and continue PPO/MAPPO/E3T training.
- Confirmed `agent_view_size` / `avs-full` only denotes observation view, not state augmentation.
- Updated current results reports to split `state_aug` and `obs_view` into separate columns.

## 2026-05-25 02:43:18 Asia/Shanghai
- User switched the active remote server to `ssh -p 29937 root@hz-4.matpool.com`.
- Refreshed local SSH known_hosts for this port and verified connection.
- Confirmed authoritative workspace exists: `/teams/ius_1663576043/hby/rl/ov2`.
- GPU check: NVIDIA RTX A6000, 49140 MiB total, idle.
- Updated remote `AGENT.md` and `MEMORY.md` to point future work to `hz-4.matpool.com:29937`.

## 2026-07-06 09:25:42 Asia/Shanghai
- Active A6000 server for this session: `ssh -p 26991 root@hz-4.matpool.com`; workspace: `/teams/ius_1663576043/hby/rl/ov2`.
- Previous quick/lightweight JAX ports of QPLEX/WQMIX/COMA produced zero SP/XP and should not be used as paper evidence.
- Cloned official EPyMARL into `third_party/epymarl` at commit `cbc38c0` and installed missing runtime deps (`sacred`, `tensorboard_logger`) in `/root/miniconda3/envs/myconda`.
- Added an OV2 EPyMARL adapter:
  - `third_party/epymarl/src/envs/ov2_wrapper.py`
  - `third_party/epymarl/src/config/envs/ov2.yaml`
  - registered `ov2` in `third_party/epymarl/src/envs/__init__.py`
- Important implementation detail: wrapper JITs complete `reset` and `step_env`; un-jitted Python/JAX stepping was about 19s per 20 steps, while jitted stepping was about 1.17s for first compile and 0.073s for the next 50 steps.
- Official EPyMARL smoke tests passed on OV2 `counter_circuit` with `max_steps=200`, `t_max=800`, `seed=0`, model saving enabled:
  - VDN log: `logs/epymarl_vdn_ov2_smoke_jit_20260706_0924.log`; model root: `runs/epymarl_ov2_smoke_20260706/models/vdn_seed0_counter_circuit_2026-07-06 09:23:03.790446`
  - QMIX log: `logs/epymarl_qmix_ov2_smoke_jit_20260706_0925.log`; model root: `runs/epymarl_ov2_smoke_20260706/models/qmix_seed0_counter_circuit_2026-07-06 09:23:46.825304`
  - COMA log: `logs/epymarl_coma_ov2_smoke_jit_20260706_0926.log`; model root: `runs/epymarl_ov2_smoke_20260706/models/coma_seed0_counter_circuit_2026-07-06 09:24:55.218332`
- Short VDN/QMIX smoke returns remained zero because the run is only a link test. COMA smoke saw nonzero shaped test return but sparse return was still zero. Next step is longer sanity training from official EPyMARL, then bridge official QPLEX/WQMIX codebases.

## 2026-07-06 09:35:35 Asia/Shanghai
- Vendored official QPLEX and WQMIX source archives:
  - `third_party/qplex_official` from `wjh720/QPLEX` archive; relevant code is under `pymarl-master/src`.
  - `third_party/wqmix_official` from `oxwhirl/wqmix` archive.
- Integrated official WQMIX/QPLEX PyMARL modules into EPyMARL without rewriting algorithm logic:
  - learners: `max_q_learner.py`, `q_learner_w.py`, `dmaq_qatten_learner.py`
  - central controller/agent: `central_basic_controller.py`, `central_rnn_agent.py`
  - mixers: `qmix_central_no_hyper.py`, `qmix_central_attention.py`, `dmaq_general.py`, `dmaq_si_weight.py`, `dmaq_qatten.py`, `dmaq_qatten_weight.py`, `qatten.py`
  - configs: `ow_qmix.yaml`, `cw_qmix.yaml`, `qplex.yaml`
- EPyMARL registry now exposes `max_q_learner`, `w_q_learner`, `dmaq_qatten_learner`, `basic_central_mac`, and `central_rnn`.
- Official-code smoke tests passed on OV2 `counter_circuit` with `max_steps=200`, `t_max=800`, `seed=0`, model saving enabled:
  - QPLEX log: `logs/epymarl_qplex_ov2_smoke_jit_20260706_0932.log`; model root: `runs/epymarl_ov2_smoke_20260706/models/qplex_seed0_counter_circuit_2026-07-06 09:33:52.225224`
  - OW-QMIX log: `logs/epymarl_ow_qmix_ov2_smoke_jit_20260706_0934.log`; model root: `runs/epymarl_ov2_smoke_20260706/models/ow_qmix_seed0_counter_circuit_2026-07-06 09:34:49.394895`
  - CW-QMIX log: `logs/epymarl_cw_qmix_ov2_smoke_jit_20260706_0934.log`; model root: `runs/epymarl_ov2_smoke_20260706/models/cw_qmix_seed0_counter_circuit_2026-07-06 09:34:49.397990`
- Interpretation: COMA, QPLEX, OW-QMIX, and CW-QMIX are now official-code OV2 training candidates. Smoke scores are still not paper evidence; they only validate code path, gradient updates, and checkpoint saving.

## 2026-07-06 09:48:11 Asia/Shanghai
- Started official-code OV2 sanity training with queue script `experiments/run_epymarl_official_ov2_sanity.sh`.
- First attempt `runs/epymarl_official_ov2_sanity_20260706_094000` launched all 6 methods concurrently and was stopped/invalidated because host memory pressure killed `qmix`, `ow_qmix`, and `cw_qmix` with status 137; COMA showed CUDA context errors under that over-parallel launch.
- Active valid run: `runs/epymarl_official_ov2_sanity_20260706_094300`; logs under `logs/epymarl_official_ov2_sanity_20260706_094300`.
- Active settings: methods `vdn qmix coma qplex ow_qmix cw_qmix`, seed `0`, `t_max=500000`, `max_steps=400`, `test_interval=25000`, `test_nepisode=20`, `reward_shaping_coef=1.0`, `MAX_PARALLEL=2`.
- At first health check, VDN and QMIX were running and had reached about `t_env=25600/500000` with zero sparse/test return so far. GPU was active and no new OOM/CUDA failure appeared. Queue will launch later methods after the current pair completes.
