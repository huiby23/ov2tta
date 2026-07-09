#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/teams/ius_1663576043/hby/rl/ov2}
EPYMARL_DIR="$ROOT/third_party/epymarl"
CONDA_PY=${CONDA_PY:-/root/miniconda3/envs/myconda/bin/python}
RUN_ID=${RUN_ID:-epymarl_official_ov2_sanity_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT="$ROOT/runs/$RUN_ID"
LOG_ROOT="$ROOT/logs/$RUN_ID"
STATUS_FILE="$RUN_ROOT/status.tsv"

METHODS=${METHODS:-vdn qmix coma qplex ow_qmix cw_qmix}
SEEDS=${SEEDS:-0}
T_MAX=${T_MAX:-500000}
MAX_STEPS=${MAX_STEPS:-400}
TEST_INTERVAL=${TEST_INTERVAL:-25000}
TEST_NEPISODE=${TEST_NEPISODE:-20}
LOG_INTERVAL=${LOG_INTERVAL:-25000}
SAVE_MODEL_INTERVAL=${SAVE_MODEL_INTERVAL:-100000}
LAYOUT=${LAYOUT:-counter_circuit}
REWARD_SHAPING_COEF=${REWARD_SHAPING_COEF:-1.0}
USE_CUDA=${USE_CUDA:-True}
MAX_PARALLEL=${MAX_PARALLEL:-2}

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

{
  echo "run_id=$RUN_ID"
  echo "run_root=$RUN_ROOT"
  echo "log_root=$LOG_ROOT"
  echo "methods=$METHODS"
  echo "seeds=$SEEDS"
  echo "t_max=$T_MAX"
  echo "max_steps=$MAX_STEPS"
  echo "layout=$LAYOUT"
  echo "reward_shaping_coef=$REWARD_SHAPING_COEF"
  echo "max_parallel=$MAX_PARALLEL"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} | tee "$RUN_ROOT/manifest.txt"

echo -e "method\tseed\tpid\tlog\tstatus" > "$STATUS_FILE"

cd "$EPYMARL_DIR"
export PYTHONPATH="$EPYMARL_DIR/src:$ROOT/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]; do
    sleep 10
  done
}

for method in $METHODS; do
  for seed in $SEEDS; do
    wait_for_slot
    log="$LOG_ROOT/${method}_seed${seed}.log"
    (
      set +e
      "$CONDA_PY" src/main.py --config="$method" --env-config=ov2 with \
        seed="$seed" \
        t_max="$T_MAX" \
        test_interval="$TEST_INTERVAL" \
        test_nepisode="$TEST_NEPISODE" \
        log_interval="$LOG_INTERVAL" \
        runner_log_interval="$LOG_INTERVAL" \
        learner_log_interval="$LOG_INTERVAL" \
        save_model=True \
        save_model_interval="$SAVE_MODEL_INTERVAL" \
        local_results_path="$RUN_ROOT" \
        use_cuda="$USE_CUDA" \
        env_args.layout="$LAYOUT" \
        env_args.key="$LAYOUT" \
        env_args.max_steps="$MAX_STEPS" \
        env_args.reward_shaping_coef="$REWARD_SHAPING_COEF" \
        > "$log" 2>&1
      status=$?
      echo "$(date '+%Y-%m-%d %H:%M:%S %Z') method=$method seed=$seed status=$status" >> "$RUN_ROOT/completions.log"
      exit "$status"
    ) &
    pid=$!
    echo -e "${method}\t${seed}\t${pid}\t${log}\trunning" | tee -a "$STATUS_FILE"
  done
done

wait

echo "All jobs completed."
echo "Run root: $RUN_ROOT"
echo "Log root: $LOG_ROOT"
echo "Status: $STATUS_FILE"
