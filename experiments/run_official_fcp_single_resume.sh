#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
POP_DIR="${POP_DIR:-fcp_populations/grounded_coord_simple_single_20260506-223912}"
POP_STAGE_DIR="${POP_STAGE_DIR:-runs/official_fcp_single_population_20260506-223912}"
PREFIX="${PREFIX:-official_fcp_single_rnn_fcp_resume_$(date +%Y%m%d-%H%M%S)}"
SEED="${SEED:-42}"
EVAL_SEEDS="${EVAL_SEEDS:-100}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-huiby_tsinghua23}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2-paper-repro}"
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

echo "[official-fcp-resume] start prefix=${PREFIX} pop=${POP_DIR} time=$(date -Is)"
CUDA_VISIBLE_DEVICES="${FCP_CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON}" -m overcooked_v2_experiments.ppo.main \
    +experiment=rnn-fcp \
    +env=grounded_coord_simple \
    NUM_SEEDS=1 \
    SEED="${SEED}" \
    +FCP="${POP_DIR}" \
    VISUALIZE=False \
    wandb.ENTITY="${WANDB_ENTITY}" \
    wandb.PROJECT="${WANDB_PROJECT}" \
    wandb.WANDB_MODE="${WANDB_MODE}" \
    +OPTIONAL_PREFIX="${PREFIX}"

run_dir="$(find "runs/${PREFIX}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)"
if [[ -z "${run_dir}" ]]; then
  echo "[official-fcp-resume] missing run dir for prefix=${PREFIX}" >&2
  exit 1
fi
echo "[official-fcp-resume] train_done run_dir=${run_dir} time=$(date -Is)"
CUDA_VISIBLE_DEVICES="${FCP_CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
    --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${SEED}" --no_viz
"${PYTHON}" - "${run_dir}/reward_summary_cross.csv" <<PY
import csv, re, sys
from statistics import mean
sp=[]; xp=[]
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        m=re.match(r"cross-(\d+)_(\d+)$", r.get("policy_labels", ""))
        if not m: continue
        v=float(r["total_reward"])
        (sp if m.group(1)==m.group(2) else xp).append(v)
print(f"[summary] SP={mean(sp) if sp else float(nan):.3f} XP={mean(xp) if xp else float(nan):.3f} n_sp={len(sp)} n_xp={len(xp)}")
PY
CUDA_VISIBLE_DEVICES="${FCP_CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" - "${run_dir}" "${POP_STAGE_DIR}" "${EVAL_SEEDS}" <<PY
from pathlib import Path
from statistics import mean
import sys, jax
from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints
from overcooked_v2_experiments.ppo.policy import PPOPolicy
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.evaluate import eval_pairing
fcp_dir=Path(sys.argv[1]); pop_stage=Path(sys.argv[2]); n=int(sys.argv[3])
fcp_ckpts,fcp_cfg=load_all_checkpoints(fcp_dir, final_only=True)
pop_ckpts,pop_cfg=load_all_checkpoints(pop_stage, final_only=True)
def pol(ckpts,cfg,i): return PPOPolicy(ckpts[f"run_{i}"]["ckpt_final"].params, cfg)
def score(name,pair):
    out=eval_pairing(pair,"grounded_coord_simple",jax.random.PRNGKey(42),env_kwargs={"agent_view_size":2,"negative_rewards":True,"random_agent_positions":True,"sample_recipe_on_delivery":True},num_seeds=n,no_viz=True)
    vals=[float(v.total_reward) for v in out.values()]
    print(f"[diag] {name} mean={mean(vals):.3f} min={min(vals):.3f} max={max(vals):.3f} nonzero={sum(v>0 for v in vals)}/{len(vals)}")
score("FCP0+FCP0", PolicyPairing(pol(fcp_ckpts,fcp_cfg,0), pol(fcp_ckpts,fcp_cfg,0)))
score("FCP0+POP0", PolicyPairing(pol(fcp_ckpts,fcp_cfg,0), pol(pop_ckpts,pop_cfg,0)))
score("POP0+FCP0", PolicyPairing(pol(pop_ckpts,pop_cfg,0), pol(fcp_ckpts,fcp_cfg,0)))
score("POP0+POP0", PolicyPairing(pol(pop_ckpts,pop_cfg,0), pol(pop_ckpts,pop_cfg,0)))
PY
echo "[official-fcp-resume] done time=$(date -Is)"
