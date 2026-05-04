#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate myconda
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
  conda activate myconda
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate myconda
fi

MODE="${1:-full}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
LAYOUT="${LAYOUT:-counter_circuit}"
EVAL_SEED="${EVAL_SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PPO_NO_STATE_AUG_RUN_DIR="${PPO_NO_STATE_AUG_RUN_DIR:-runs/figure4_128_32_no_state_aug_full_obs_ppo_cnn_no_state_aug_full_obs_20260502-12832-no-sa-full-after-current/20260502-033934_x6oni3d1_counter_circuit_avs-full}"
PPO_STATE_AUG_RUN_DIR="${PPO_STATE_AUG_RUN_DIR:-runs/figure4_128_32_ce8_e3tfirst_ppo_cnn_state_aug_20260501-12832-ce8-e3tfirst/20260501-215253_4xjcr9al_counter_circuit_avs-full}"
E3T_CE_NO_STATE_AUG_RUN_DIR="${E3T_CE_NO_STATE_AUG_RUN_DIR:-runs/figure4_128_32_no_state_aug_full_obs_ppo_e3t_predicted_ce_no_state_aug_full_obs_20260502-12832-no-sa-full-after-current/20260502-090053_s5532yeq_counter_circuit_avs-full}"
E3T_CE_STATE_AUG_RUN_DIR="${E3T_CE_STATE_AUG_RUN_DIR:-runs/figure4_128_32_ce8_e3tfirst_ppo_e3t_predicted_ce_state_aug_20260501-12832-ce8-e3tfirst/20260501-115329_ti1v8kbr_counter_circuit_avs-full}"
E3T_NO_CE_NO_STATE_AUG_RUN_DIR="${E3T_NO_CE_NO_STATE_AUG_RUN_DIR:-runs/figure4_128_32_no_state_aug_full_obs_ppo_e3t_no_ce_no_state_aug_full_obs_20260502-12832-no-sa-full-after-current/20260502-182735_3okxpq47_counter_circuit_avs-full}"

case "$MODE" in
  smoke)
    NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-5}"
    NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-2}"
    PERTURB_EVAL_SEEDS="${PERTURB_EVAL_SEEDS:-5}"
    MAX_PAIRS="${MAX_PAIRS:-2}"
    ;;
  quick)
    NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-100}"
    NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-20}"
    PERTURB_EVAL_SEEDS="${PERTURB_EVAL_SEEDS:-100}"
    MAX_PAIRS="${MAX_PAIRS:-20}"
    ;;
  full)
    NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
    NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-100}"
    PERTURB_EVAL_SEEDS="${PERTURB_EVAL_SEEDS:-500}"
    MAX_PAIRS="${MAX_PAIRS:-}"
    ;;
  *)
    echo "Usage: $0 [smoke|quick|full]" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/e3t_mechanism_analysis_${MODE}_${STAMP}}"
mkdir -p "$OUTPUT_ROOT"

echo "[e3t-mech] mode=$MODE output=$OUTPUT_ROOT"
echo "[e3t-mech] eval_seed=$EVAL_SEED eval_seeds=$NUM_EVAL_SEEDS diag_episodes=$NUM_DIAG_EPISODES perturb_eval_seeds=$PERTURB_EVAL_SEEDS max_pairs=${MAX_PAIRS:-all}"
echo "[e3t-mech] ppo_no_state_aug=$PPO_NO_STATE_AUG_RUN_DIR"
echo "[e3t-mech] ppo_state_aug=$PPO_STATE_AUG_RUN_DIR"
echo "[e3t-mech] e3t_ce_no_state_aug=$E3T_CE_NO_STATE_AUG_RUN_DIR"
echo "[e3t-mech] e3t_ce_state_aug=$E3T_CE_STATE_AUG_RUN_DIR"
echo "[e3t-mech] e3t_no_ce_no_state_aug=$E3T_NO_CE_NO_STATE_AUG_RUN_DIR"

run_zsc_diag() {
  local backend_script="$1"
  local name_a="$2"
  local dir_a="$3"
  local name_b="$4"
  local dir_b="$5"
  local out="$6"

  local args=(
    --baseline_run_dir "$dir_a"
    --comparison_run_dir "$dir_b"
    --baseline_name "$name_a"
    --comparison_name "$name_b"
    --baseline_backend ppo
    --comparison_backend ppo
    --layout "$LAYOUT"
    --eval_seed "$EVAL_SEED"
    --num_eval_seeds "$NUM_EVAL_SEEDS"
    --num_diag_episodes "$NUM_DIAG_EPISODES"
    --top_k 10
    --compatibility_sample_limit 512
    --output_dir "$out"
  )
  if [[ -n "$MAX_PAIRS" ]]; then
    args+=(--max_pairs "$MAX_PAIRS")
  fi

  echo "[e3t-mech] zsc_diag $name_a vs $name_b -> $out"
  PYTHONUNBUFFERED=1 PYTHONPATH=experiments "$PYTHON_BIN" "$backend_script" "${args[@]}"
}

run_perturb() {
  local method_name="$1"
  local run_dir="$2"
  local out="$3"
  echo "[e3t-mech] perturb $method_name -> $out"
  PYTHONUNBUFFERED=1 PYTHONPATH=experiments "$PYTHON_BIN" \
    experiments/overcooked_v2_experiments/ppo_e3t_official/utils/context_perturbation_eval.py \
    --run_dir "$run_dir" \
    --method_name "$method_name" \
    --output_dir "$out" \
    --layout "$LAYOUT" \
    --eval_seed "$EVAL_SEED" \
    --num_eval_seeds "$PERTURB_EVAL_SEEDS"
}

run_zsc_diag \
  experiments/overcooked_v2_experiments/ppo/utils/zsc_diagnostics.py \
  ppo_cnn_no_state_aug_full_obs "$PPO_NO_STATE_AUG_RUN_DIR" \
  ppo_cnn_state_aug "$PPO_STATE_AUG_RUN_DIR" \
  "$OUTPUT_ROOT/ppo_no_state_vs_state_aug"

run_zsc_diag \
  experiments/overcooked_v2_experiments/ppo_e3t_official/utils/zsc_diagnostics.py \
  ppo_e3t_predicted_ce_no_state_aug_full_obs "$E3T_CE_NO_STATE_AUG_RUN_DIR" \
  ppo_e3t_predicted_ce_state_aug "$E3T_CE_STATE_AUG_RUN_DIR" \
  "$OUTPUT_ROOT/e3t_ce_no_state_vs_state_aug"

run_zsc_diag \
  experiments/overcooked_v2_experiments/ppo_e3t_official/utils/zsc_diagnostics.py \
  ppo_e3t_predicted_ce_no_state_aug_full_obs "$E3T_CE_NO_STATE_AUG_RUN_DIR" \
  ppo_e3t_no_ce_no_state_aug_full_obs "$E3T_NO_CE_NO_STATE_AUG_RUN_DIR" \
  "$OUTPUT_ROOT/e3t_ce_vs_no_ce_no_state_aug"

run_perturb \
  ppo_e3t_predicted_ce_no_state_aug_full_obs \
  "$E3T_CE_NO_STATE_AUG_RUN_DIR" \
  "$OUTPUT_ROOT/perturb_e3t_ce_no_state_aug"

run_perturb \
  ppo_e3t_predicted_ce_state_aug \
  "$E3T_CE_STATE_AUG_RUN_DIR" \
  "$OUTPUT_ROOT/perturb_e3t_ce_state_aug"

cat > "$OUTPUT_ROOT/README.md" <<EOF
# E3T Mechanism Analysis

- mode: $MODE
- generated_at: $(date -Is)
- layout: $LAYOUT
- eval_seed: $EVAL_SEED
- num_eval_seeds: $NUM_EVAL_SEEDS
- num_diag_episodes: $NUM_DIAG_EPISODES
- perturb_eval_seeds: $PERTURB_EVAL_SEEDS

## Outputs

- PPO coverage/mismatch: \`ppo_no_state_vs_state_aug/report.md\`
- E3T CE coverage/mismatch: \`e3t_ce_no_state_vs_state_aug/report.md\`
- E3T CE vs no-CE no-state-aug: \`e3t_ce_vs_no_ce_no_state_aug/report.md\`
- E3T CE no-state perturbation: \`perturb_e3t_ce_no_state_aug/report.md\`
- E3T CE state-aug perturbation: \`perturb_e3t_ce_state_aug/report.md\`
EOF

echo "[e3t-mech] done output=$OUTPUT_ROOT"
