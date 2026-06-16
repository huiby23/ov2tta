#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/current_table_independent_diagnostics_index_20260523_20260523-1053}"
DIAG_TAG="${DIAG_TAG:-current_table_20260523}"
ANALYSIS_OUT="${ANALYSIS_OUT:-reports/paper_analysis_20260523}"
LOG_DIR="${LOG_DIR:-logs/paper_analysis_20260523}"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false}"
export ZSC_FORWARD_CHUNK_SIZE="${ZSC_FORWARD_CHUNK_SIZE:-1024}"
export PYTHONUNBUFFERED=1

run_diag() {
  local slug="$1" display="$2" setting="$3" run_dir="$4" diag_py="$5" backend="$6"
  local out_dir="${run_dir}/diagnostics/${DIAG_TAG}"
  if [[ -f "${out_dir}/report.md" && "${FORCE_MISSING:-0}" != "1" ]]; then
    echo "[paper-analysis] skip existing ${setting}/${slug}"
    return 0
  fi
  mkdir -p "${out_dir}"
  echo "[paper-analysis] diag_start ${setting}/${slug} out=${out_dir}"
  "${PYTHON}" "${diag_py}" \
    --single_run_dir "${run_dir}" \
    --single_name "${slug}" \
    --single_backend "${backend}" \
    --single_eval_mode memory_off \
    --layout counter_circuit \
    --eval_seed 42 \
    --num_eval_seeds 500 \
    --num_diag_episodes "${NUM_DIAG_EPISODES:-20}" \
    --top_k 10 \
    --compatibility_sample_limit "${COMPATIBILITY_SAMPLE_LIMIT:-256}" \
    --max_pairs "${MAX_PAIRS:-30}" \
    --output_dir "${out_dir}" \
    > "${out_dir}/paper_rerun.log" 2>&1 || {
      echo "[paper-analysis] diag_failed ${setting}/${slug}; tail follows"
      tail -n 80 "${out_dir}/paper_rerun.log" || true
      return 1
    }
  echo "[paper-analysis] diag_done ${setting}/${slug}"
}

# Missing diagnostics from the current table. They are retried with smaller diagnostic batches and cudnn fallback.
FAILURES=0
run_diag "ppo_e3t_rnn_standard_fixed" "PPO-E3T RNN standard fixed" "standard" \
  "runs/figure4_e3t_ppo_rnn_64_16_standard_full_rnnfix_20260504-130101/20260504-130114_qrjr5dlz_counter_circuit_avs-full" \
  "experiments/overcooked_v2_experiments/ppo_e3t_official/utils/zsc_diagnostics.py" "ppo" || FAILURES=$((FAILURES+1))
run_diag "ppo_e3t_rnn_state_aug_fixed" "PPO-E3T RNN state-aug fixed" "state_aug" \
  "runs/figure4_e3t_ppo_rnn_64_16_state_aug_full_rnnfix_20260504-130101/20260504-175745_vqj6owwb_counter_circuit_avs-full" \
  "experiments/overcooked_v2_experiments/ppo_e3t_official/utils/zsc_diagnostics.py" "ppo" || FAILURES=$((FAILURES+1))
run_diag "ppo_e3t_cnn_state_aug_constant_ce" "PPO-E3T CNN state-aug constant CE" "state_aug" \
  "runs/figure4_ppo_e3t_state_aug_constant_ce_full_20260428-ppo-e3t-state-aug/20260428-182936_aza8mbkr_counter_circuit_avs-full" \
  "experiments/overcooked_v2_experiments/ppo_e3t_official/utils/zsc_diagnostics.py" "ppo" || FAILURES=$((FAILURES+1))
run_diag "mep_ent010_state_aug" "MEP ent=0.1 state-aug" "state_aug" \
  "runs/population_state_aug_64_16_mep_state_aug_mm1_mp1_K5_ent0.1_64_16_10000000_sa10_stateaug-20260512-012924/20260512-062758_en6is7qd_counter_circuit_avs-full" \
  "experiments/overcooked_v2_experiments/ppo/utils/zsc_diagnostics.py" "ppo" || FAILURES=$((FAILURES+1))

# Refresh the central diagnostics aggregate after any successful reruns.
OUTPUT_ROOT="${OUTPUT_ROOT}" DIAG_TAG="${DIAG_TAG}" STORE_IN_RUN_DIR=1 "${PYTHON}" - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location('diag_index', 'experiments/run_current_table_independent_diagnostics_20260523.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.aggregate()
PY

# Generate all paper analysis tables and figures from the refreshed central index.
"${PYTHON}" experiments/paper_analysis_existing_models_20260523.py \
  --summary_csv "${OUTPUT_ROOT}/summary_flat.csv" \
  --out_dir "${ANALYSIS_OUT}" \
  --bootstrap "${BOOTSTRAP:-1000}"

echo "[paper-analysis] done out=${ANALYSIS_OUT} failures=${FAILURES}"
exit 0
