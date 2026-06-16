#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
cd "${ROOT}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

MODE=full \
WANDB_MODE="${WANDB_MODE:-online}" \
PREFIX="ttac_joint_adapter_state_aug_64_16_10M_seed42_10seeds_${TS}" \
TTAC_AGREEMENT_COEF=0.05 \
RUN_DIAGNOSTICS=0 \
./experiments/run_ttac_pipeline_64_16.sh

MODE=full \
WANDB_MODE="${WANDB_MODE:-online}" \
PREFIX="ttac_adapter_no_agreement_train_state_aug_64_16_10M_seed42_10seeds_${TS}" \
TTAC_AGREEMENT_COEF=0.0 \
RUN_DIAGNOSTICS=0 \
./experiments/run_ttac_pipeline_64_16.sh
