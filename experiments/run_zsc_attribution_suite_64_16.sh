#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/myconda/bin/python}"
JOB_ROOT="${JOB_ROOT:-logs/zsc_attribution_suite_64_16}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
mkdir -p "${JOB_ROOT}"

PPO_STANDARD="${PPO_STANDARD:-runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full}"
PPO_STATE_AUG="${PPO_STATE_AUG:-runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full}"
MAPPO_RNN_STATE_AUG="${MAPPO_RNN_STATE_AUG:-runs/figure4_mappo_rnn_64_16_rerun_mappo_rnn_state_aug_20260503-224455/20260503-233640_cvhspxsq_counter_circuit_avs-full}"
TRAJEDI="${TRAJEDI:-runs/trajedi_realpop_mm1_mp1_K5_div0.1_64_16_10000000_seed10_20260511-103234/20260511-103246_ggoqyh5r_counter_circuit_avs-full}"
TRAJEDI_STATE_AUG="${TRAJEDI_STATE_AUG:-runs/population_state_aug_64_16_trajedi_state_aug_mm1_mp1_K5_div0.1_64_16_10000000_sa10_stateaug-20260512-012924/20260512-012937_4r9b0ck9_counter_circuit_avs-full}"
MEP="${MEP:-runs/mep_realpop_mm1_mp1_K5_ent0.1_64_16_10000000_seed10_20260511-002112/20260511-002124_vnapmwlc_counter_circuit_avs-full}"
MEP_STATE_AUG="${MEP_STATE_AUG:-runs/population_state_aug_64_16_mep_state_aug_mm1_mp1_K5_ent0.1_64_16_10000000_sa10_stateaug-20260512-012924/20260512-062758_en6is7qd_counter_circuit_avs-full}"

LAYOUT="${LAYOUT:-counter_circuit}"
EVAL_SEED="${EVAL_SEED:-42}"
NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-20}"
MAX_PAIRS="${MAX_PAIRS:-30}"
COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT:-256}"
TTA_EPISODES_PER_PAIR="${TTA_EPISODES_PER_PAIR:-6}"
TTA_MAX_PAIRS="${TTA_MAX_PAIRS:-30}"
TTA_SAMPLE_LIMIT="${TTA_SAMPLE_LIMIT:-20000}"
TTA_HISTORY_WINDOW="${TTA_HISTORY_WINDOW:-20}"
TTA_CHUNK_SIZE="${TTA_CHUNK_SIZE:-4096}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/zsc_attribution_suite_64_16_${TAG}}"

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

run_pair_diag() {
  local name="$1"
  local base_name="$2"
  local base_dir="$3"
  local base_backend="$4"
  local comp_name="$5"
  local comp_dir="$6"
  local comp_backend="$7"
  local out_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "[zsc-suite] pair_diag_start $(date -Is) name=${name}"
  BASELINE_NAME="${base_name}" \
  BASELINE_RUN_DIR="${base_dir}" \
  BASELINE_BACKEND="${base_backend}" \
  COMPARISON_NAME="${comp_name}" \
  COMPARISON_RUN_DIR="${comp_dir}" \
  COMPARISON_BACKEND="${comp_backend}" \
  LAYOUT="${LAYOUT}" \
  EVAL_SEED="${EVAL_SEED}" \
  NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS}" \
  NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES}" \
  MAX_PAIRS="${MAX_PAIRS}" \
  COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT}" \
  OUTPUT_DIR="${out_dir}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  experiments/run_zsc_method_pair_diagnostics.sh
  echo "[zsc-suite] pair_diag_done $(date -Is) name=${name} out=${out_dir}"
}

run_tta_oracle() {
  local name="$1"
  local method_name="$2"
  local run_dir="$3"
  local backend="$4"
  local out_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "[zsc-suite] tta_oracle_start $(date -Is) name=${name}"
  PYTHONPATH="${ROOT}/experiments:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" experiments/overcooked_v2_experiments/ppo/utils/tta_necessity_oracle.py \
    --run_dir "${run_dir}" \
    --method_name "${method_name}" \
    --backend "${backend}" \
    --layout "${LAYOUT}" \
    --seed "${EVAL_SEED}" \
    --num_episodes_per_pair "${TTA_EPISODES_PER_PAIR}" \
    --max_pairs "${TTA_MAX_PAIRS}" \
    --sample_limit "${TTA_SAMPLE_LIMIT}" \
    --history_window "${TTA_HISTORY_WINDOW}" \
    --chunk_size "${TTA_CHUNK_SIZE}" \
    --output_dir "${out_dir}"
  echo "[zsc-suite] tta_oracle_done $(date -Is) name=${name} out=${out_dir}"
}

aggregate() {
  echo "[zsc-suite] aggregate_start $(date -Is)"
  PYTHONPATH="${ROOT}/experiments:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<PY
import csv, json
from pathlib import Path
root = Path("${OUTPUT_ROOT}")
lines = [
    "# ZSC Attribution Suite 64/16",
    "",
    f"- generated_at: `${TAG}`",
    f"- num_diag_episodes: `${NUM_DIAG_EPISODES}`",
    f"- max_pairs: `${MAX_PAIRS}`",
    f"- compatibility_sample_limit: `${COMPATIBILITY_SAMPLE_LIMIT}`",
    "",
    "## Pair Diagnostics",
    "",
    "| Block | Method | SP | XP | Out-of-support | In-both | Policy TV | Action agreement | Joint regret | Unilateral regret |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for block in sorted(root.iterdir() if root.exists() else []):
    if not block.is_dir() or not (block / "reward_summary.csv").exists():
        continue
    rewards = {r["method"]: r for r in csv.DictReader((block / "reward_summary.csv").open())}
    cov_rows = list(csv.DictReader((block / "coverage_summary.csv").open())) if (block / "coverage_summary.csv").exists() else []
    mis_rows = list(csv.DictReader((block / "shared_state_mismatch.csv").open())) if (block / "shared_state_mismatch.csv").exists() else []
    comp_rows = list(csv.DictReader((block / "complementarity_summary.csv").open())) if (block / "complementarity_summary.csv").exists() else []
    methods = sorted(rewards)
    for method in methods:
        cov = [r for r in cov_rows if r.get("method") == method and r.get("row_type") == "cross_coverage"]
        mis = [r for r in mis_rows if r.get("method") == method]
        comp = [r for r in comp_rows if r.get("method") == method]
        def avg(rows, key):
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            return sum(vals) / len(vals) if vals else float("nan")
        rw = rewards[method]
        lines.append(
            f"| {block.name} | {method} | {float(rw['sp_mean']):.3f} | {float(rw['xp_mean']):.3f} | "
            f"{avg(cov, 'xp_out_of_sp_support_rate'):.4f} | {avg(cov, 'in_both_support_rate'):.4f} | "
            f"{avg(mis, 'policy_tv'):.4f} | {avg(mis, 'action_agreement'):.4f} | "
            f"{avg(comp, 'joint_regret'):.4f} | {avg(comp, 'unilateral_regret'):.4f} |"
        )
lines += ["", "## TTA Necessity", "", "| Method | Switch needed | Positive margin | Oracle-current score | Probe | Acc | Bal Acc | Pred gain | Gain recovered |", "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
for block in sorted(root.iterdir() if root.exists() else []):
    if not block.is_dir() or not (block / "summary.json").exists() or not (block / "probe_summary.csv").exists():
        continue
    summary = json.loads((block / "summary.json").read_text())
    for row in csv.DictReader((block / "probe_summary.csv").open()):
        if row.get("error"):
            continue
        lines.append(
            f"| {summary['method']} | {summary['switch_needed_rate']:.4f} | {summary['positive_margin_rate']:.4f} | "
            f"{summary['mean_oracle_minus_current_score']:.4f} | {row['probe']} | {float(row['accuracy']):.4f} | "
            f"{float(row['balanced_accuracy']):.4f} | {float(row['mean_predicted_gain']):.4f} | {float(row['positive_gain_recovered']):.4f} |"
        )
(root / "summary.md").write_text("\n".join(lines) + "\n")
print(root / "summary.md")
PY
  echo "[zsc-suite] aggregate_done $(date -Is) report=${OUTPUT_ROOT}/summary.md"
}

pipeline() {
  mkdir -p "${OUTPUT_ROOT}"
  echo "[zsc-suite] launch $(date -Is) output_root=${OUTPUT_ROOT}"
  echo "[zsc-suite] diag episodes=${NUM_DIAG_EPISODES} max_pairs=${MAX_PAIRS} compat_limit=${COMPATIBILITY_SAMPLE_LIMIT}"
  echo "[zsc-suite] tta episodes=${TTA_EPISODES_PER_PAIR} max_pairs=${TTA_MAX_PAIRS} sample_limit=${TTA_SAMPLE_LIMIT}"
  run_pair_diag coverage_ppo ppo_cnn_standard "${PPO_STANDARD}" ppo ppo_cnn_state_aug "${PPO_STATE_AUG}" ppo
  run_pair_diag coverage_trajedi trajedi "${TRAJEDI}" ppo trajedi_state_aug "${TRAJEDI_STATE_AUG}" ppo
  run_pair_diag coverage_mep mep "${MEP}" ppo mep_state_aug "${MEP_STATE_AUG}" ppo
  run_pair_diag mismatch_ppo_vs_mappo ppo_cnn_state_aug "${PPO_STATE_AUG}" ppo mappo_rnn_state_aug_current "${MAPPO_RNN_STATE_AUG}" mappo
  run_pair_diag mismatch_ppo_vs_mep ppo_cnn_state_aug "${PPO_STATE_AUG}" ppo mep_state_aug "${MEP_STATE_AUG}" ppo
  run_tta_oracle tta_oracle_ppo_state_aug ppo_cnn_state_aug "${PPO_STATE_AUG}" ppo
  run_tta_oracle tta_oracle_mep_state_aug mep_state_aug "${MEP_STATE_AUG}" ppo
  aggregate
  rm -f "${PID_FILE}"
  echo "[zsc-suite] pipeline_done $(date -Is)"
}

status() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[zsc-suite] running pid=${pid}"
    ps -fp "${pid}" || true
    echo "[zsc-suite] children:"
    pgrep -P "${pid}" -af || true
  else
    echo "[zsc-suite] not running"
    [[ -f "${PID_FILE}" ]] && echo "[zsc-suite] stale pid=$(cat "${PID_FILE}")"
  fi
  if [[ -f "${LATEST_LOG}" ]]; then
    local log
    log="$(cat "${LATEST_LOG}")"
    echo "[zsc-suite] latest_log=${log}"
    [[ -f "${log}" ]] && tail -n 120 "${log}"
  fi
}

start() {
  if is_running; then
    status
    exit 0
  fi
  local log="${JOB_ROOT}/zsc_attribution_suite_${TAG}.log"
  printf "%s\n" "${log}" > "${LATEST_LOG}"
  nohup env TAG="${TAG}" OUTPUT_ROOT="${OUTPUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES}" MAX_PAIRS="${MAX_PAIRS}" COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT}" \
    TTA_EPISODES_PER_PAIR="${TTA_EPISODES_PER_PAIR}" TTA_MAX_PAIRS="${TTA_MAX_PAIRS}" TTA_SAMPLE_LIMIT="${TTA_SAMPLE_LIMIT}" \
    TTA_HISTORY_WINDOW="${TTA_HISTORY_WINDOW}" TTA_CHUNK_SIZE="${TTA_CHUNK_SIZE}" \
    bash "${BASH_SOURCE[0]}" pipeline > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "[zsc-suite] started pid=${pid} log=${log} output_root=${OUTPUT_ROOT}"
}

case "${1:-start}" in
  start) start ;;
  pipeline) pipeline ;;
  status) status ;;
  tail) tail -f "$(cat "${LATEST_LOG}")" ;;
  stop) if is_running; then kill "$(cat "${PID_FILE}")"; else echo "[zsc-suite] not running"; fi ;;
  *) echo "Usage: $0 [start|pipeline|status|tail|stop]" >&2; exit 2 ;;
esac
