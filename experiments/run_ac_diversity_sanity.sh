#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-500000}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-128}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-16}"
LAYOUT="${LAYOUT:-counter_circuit}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-ov2_ac_diversity}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EVAL_SEEDS="${EVAL_SEEDS:-50}"
EVAL_SEED="${EVAL_SEED:-42}"
METHODS="${METHODS:-ippo_cnn_standard ippo_cnn_ent001 ippo_cnn_clip010 ippo_cnn_gae090 ippo_rnn_standard mappo_cnn_world_state mappo_cnn_local_obs mappo_cnn_zero_world_state}"
PREFIX_BASE="${PREFIX_BASE:-ac_diversity_sanity}"
RUN_GROUP="${PREFIX_BASE}_${TS}"
LOG_DIR="${LOG_DIR:-logs/${RUN_GROUP}}"
REPORT_DIR="${REPORT_DIR:-reports/${RUN_GROUP}}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES

QUEUE="${LOG_DIR}/queue.log"
SUMMARY_CSV="${REPORT_DIR}/summary.csv"
SUMMARY_MD="${REPORT_DIR}/summary.md"
: > "${QUEUE}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

cat > "${REPORT_DIR}/manifest.txt" <<EOF
run_group=${RUN_GROUP}
methods=${METHODS}
seed=${SEED}
num_seeds=${NUM_SEEDS}
total_timesteps=${TOTAL_TIMESTEPS}
rew_shaping_horizon=${REW_SHAPING_HORIZON}
num_envs=${NUM_ENVS}
num_steps=${NUM_STEPS}
update_epochs=${UPDATE_EPOCHS}
num_minibatches=${NUM_MINIBATCHES}
eval_seeds=${EVAL_SEEDS}
layout=${LAYOUT}
wandb_mode=${WANDB_MODE}
EOF

echo "method,framework,run_dir,sp,xp,sp_pair_std,xp_pair_std,num_sp_pairs,num_xp_pairs,csv" > "${SUMMARY_CSV}"

latest_run_dir() {
  local prefix="$1"
  find "runs/${prefix}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

summarize_csv() {
  local method="$1"
  local framework="$2"
  local run_dir="$3"
  local csv_path="$4"
  "${PYTHON}" - "$method" "$framework" "$run_dir" "$csv_path" "$SUMMARY_CSV" <<'PY_SUM'
import csv, math, re, statistics, sys
from pathlib import Path
method, framework, run_dir, csv_path, summary_csv = sys.argv[1:]
path = Path(csv_path)
by = {}
if path.exists():
    with path.open() as f:
        for r in csv.DictReader(f):
            label = r.get('policy_labels', '')
            m = re.match(r'cross-(\d+)_(\d+)$', label)
            if not m:
                continue
            pair = (int(m.group(1)), int(m.group(2)))
            by.setdefault(pair, []).append(float(r['total_reward']))
pm = {k: sum(v) / len(v) for k, v in by.items()}
sp = [v for (i, j), v in pm.items() if i == j]
xp = [v for (i, j), v in pm.items() if i != j]
def mean(xs): return sum(xs) / len(xs) if xs else float('nan')
def pstdev(xs): return statistics.pstdev(xs) if len(xs) > 1 else 0.0
row = {
    'method': method,
    'framework': framework,
    'run_dir': run_dir,
    'sp': mean(sp),
    'xp': mean(xp),
    'sp_pair_std': pstdev(sp),
    'xp_pair_std': pstdev(xp),
    'num_sp_pairs': len(sp),
    'num_xp_pairs': len(xp),
    'csv': str(path),
}
with open(summary_csv, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['method','framework','run_dir','sp','xp','sp_pair_std','xp_pair_std','num_sp_pairs','num_xp_pairs','csv'])
    w.writerow(row)
print('[ac-diversity] result method={method} framework={framework} SP={sp:.3f} XP={xp:.3f} sp_pairs={nsp} xp_pairs={nxp} csv={csv}'.format(
    method=method, framework=framework, sp=row['sp'], xp=row['xp'], nsp=row['num_sp_pairs'], nxp=row['num_xp_pairs'], csv=path
))
PY_SUM
}

write_markdown() {
  "${PYTHON}" - "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY_MD'
import csv, math, sys
from pathlib import Path
csv_path, md_path = map(Path, sys.argv[1:])
rows = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
lines = [
    '# AC Diversity Sanity',
    '',
    f'- Summary CSV: `{csv_path}`',
    '',
    '| method | framework | SP | XP | SP pairs | XP pairs | run dir |',
    '|---|---|---:|---:|---:|---:|---|',
]
for r in rows:
    def fmt(x):
        try:
            v = float(x)
        except Exception:
            return str(x)
        return 'nan' if math.isnan(v) else f'{v:.3f}'
    lines.append(f"| {r['method']} | {r['framework']} | {fmt(r['sp'])} | {fmt(r['xp'])} | {r['num_sp_pairs']} | {r['num_xp_pairs']} | `{r['run_dir']}` |")
Path(md_path).write_text('\n'.join(lines) + '\n')
PY_MD
}

common_overrides() {
  local model_name="$1"
  COMMON=(
    +env=default
    model="${model_name}"
    +env.ENV_KWARGS.layout="${LAYOUT}"
    SEED="${SEED}"
    NUM_SEEDS="${NUM_SEEDS}"
    NUM_CHECKPOINTS="${NUM_CHECKPOINTS}"
    VISUALIZE=False
    wandb.WANDB_MODE="${WANDB_MODE}"
    wandb.PROJECT="${WANDB_PROJECT}"
    wandb.ENTITY="${WANDB_ENTITY}"
    model.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}"
    model.REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON}"
    model.NUM_ENVS="${NUM_ENVS}"
    model.NUM_STEPS="${NUM_STEPS}"
    model.UPDATE_EPOCHS="${UPDATE_EPOCHS}"
    model.NUM_MINIBATCHES="${NUM_MINIBATCHES}"
  )
}

eval_run() {
  local framework="$1"
  local method="$2"
  local run_dir="$3"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    log "ERROR missing run_dir method=${method} framework=${framework} run_dir=${run_dir}"
    return 1
  fi
  log "START eval ${method} framework=${framework} run_dir=${run_dir}"
  if [[ "${framework}" == "ppo" ]]; then
    "${PYTHON}" experiments/overcooked_v2_experiments/ppo/utils/visualize_ppo.py \
      --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${EVAL_SEED}" --no_viz \
      > "${LOG_DIR}/${method}_eval.log" 2>&1
  else
    "${PYTHON}" experiments/overcooked_v2_experiments/mappo/utils/visualize.py \
      --d "${run_dir}" --cross --num_seeds "${EVAL_SEEDS}" --seed "${EVAL_SEED}" --no_viz \
      > "${LOG_DIR}/${method}_eval.log" 2>&1
  fi
  summarize_csv "${method}" "${framework}" "${run_dir}" "${run_dir}/reward_summary_cross.csv" | tee -a "${QUEUE}"
  write_markdown
  log "DONE eval ${method}"
}

train_ppo() {
  local method="$1"
  local model_name="$2"
  shift 2
  local prefix="${RUN_GROUP}_${method}"
  common_overrides "${model_name}"
  log "START train ${method} framework=ppo prefix=${prefix}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ppo/main.py \
    "${COMMON[@]}" \
    +OPTIONAL_PREFIX="${prefix}" \
    "$@" \
    > "${LOG_DIR}/${method}_train.log" 2>&1
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  log "DONE train ${method} run_dir=${run_dir}"
  eval_run ppo "${method}" "${run_dir}"
}

train_mappo() {
  local method="$1"
  local model_name="$2"
  shift 2
  local prefix="${RUN_GROUP}_${method}"
  common_overrides "${model_name}"
  log "START train ${method} framework=mappo prefix=${prefix}"
  "${PYTHON}" experiments/overcooked_v2_experiments/mappo/main.py \
    "${COMMON[@]}" \
    +OPTIONAL_PREFIX="${prefix}" \
    "$@" \
    > "${LOG_DIR}/${method}_train.log" 2>&1
  local run_dir
  run_dir="$(latest_run_dir "${prefix}")"
  log "DONE train ${method} run_dir=${run_dir}"
  eval_run mappo "${method}" "${run_dir}"
}

for method in ${METHODS}; do
  case "${method}" in
    ippo_cnn_standard)
      train_ppo "${method}" cnn model.ENT_COEF=0.04 model.CLIP_EPS=0.2 model.GAE_LAMBDA=0.95 ;;
    ippo_cnn_ent001)
      train_ppo "${method}" cnn model.ENT_COEF=0.01 model.CLIP_EPS=0.2 model.GAE_LAMBDA=0.95 ;;
    ippo_cnn_clip010)
      train_ppo "${method}" cnn model.ENT_COEF=0.04 model.CLIP_EPS=0.1 model.GAE_LAMBDA=0.95 ;;
    ippo_cnn_gae090)
      train_ppo "${method}" cnn model.ENT_COEF=0.04 model.CLIP_EPS=0.2 model.GAE_LAMBDA=0.90 ;;
    ippo_rnn_standard)
      train_ppo "${method}" rnn model.ENT_COEF=0.01 model.CLIP_EPS=0.2 model.GAE_LAMBDA=0.95 model.MAX_GRAD_NORM=0.25 ;;
    mappo_cnn_world_state)
      train_mappo "${method}" cnn +model.CRITIC_INPUT_MODE=world_state model.ENT_COEF=0.04 model.CLIP_EPS=0.2 model.GAE_LAMBDA=0.95 ;;
    mappo_cnn_local_obs)
      train_mappo "${method}" cnn +model.CRITIC_INPUT_MODE=local_obs model.ENT_COEF=0.04 model.CLIP_EPS=0.2 model.GAE_LAMBDA=0.95 ;;
    mappo_cnn_zero_world_state)
      train_mappo "${method}" cnn +model.CRITIC_INPUT_MODE=zero_world_state model.ENT_COEF=0.04 model.CLIP_EPS=0.2 model.GAE_LAMBDA=0.95 ;;
    *) log "ERROR unknown method=${method}"; exit 2 ;;
  esac
done

write_markdown
log "ALL_DONE report=${SUMMARY_MD}"
