#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
cd "$ROOT"
source "$ROOT/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="$ROOT/experiments:$ROOT/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
export ZSC_FORWARD_CHUNK_SIZE="${ZSC_FORWARD_CHUNK_SIZE:-32768}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
RUN_DIR="${RUN_DIR:-$(cat reports/ttac_v2_train_surrogate_overnight_20260603_overnight_surrogate/run_dir_projconf_train_full.txt)}"
OUT_ROOT="${OUT_ROOT:-reports/ttac_v2_projconf_train_diagnostics_20260603}"
LOG_DIR="${LOG_DIR:-logs/ttac_v2_projconf_train_diagnostics_20260603}"
NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-50}"
NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-20}"
MAX_PAIRS="${MAX_PAIRS:-30}"
TOP_K="${TOP_K:-10}"
COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT:-256}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"
: > "$OUT_ROOT/queue.log"
run_one() {
  local name="$1" mode="$2"
  local out="$OUT_ROOT/$name"
  local log="$LOG_DIR/$name.log"
  mkdir -p "$out"
  echo "[$(date '+%F %T')] START $name mode=$mode run_dir=$RUN_DIR" | tee -a "$OUT_ROOT/queue.log"
  "$PYTHON" experiments/overcooked_v2_experiments/ttac_v2/utils/zsc_diagnostics.py \
    --single_run_dir "$RUN_DIR" \
    --single_name "$name" \
    --single_backend ttac \
    --single_eval_mode "$mode" \
    --layout counter_circuit \
    --eval_seed 42 \
    --num_eval_seeds "$NUM_EVAL_SEEDS" \
    --num_diag_episodes "$NUM_DIAG_EPISODES" \
    --top_k "$TOP_K" \
    --compatibility_sample_limit "$COMPATIBILITY_SAMPLE_LIMIT" \
    --max_pairs "$MAX_PAIRS" \
    --output_dir "$out" > "$log" 2>&1
  echo "[$(date '+%F %T')] DONE $name" | tee -a "$OUT_ROOT/queue.log"
}
run_one projconf_no_adapt base_no_test_adapt
run_one projconf_online_projected_confident ttac_projected_confident
"$PYTHON" - <<'PY'
from pathlib import Path
import csv, math
root=Path('reports/ttac_v2_projconf_train_diagnostics_20260603')
methods=[('projconf_no_adapt','base_no_test_adapt'),('projconf_online_projected_confident','ttac_projected_confident')]

def read_reward(path, method, field):
    try:
        with path.open() as f:
            for row in csv.DictReader(f):
                if row.get('method') == method and row.get(field,''):
                    return float(row[field])
    except FileNotFoundError:
        pass
    return math.nan

def mean_field(path, method, field, row_type=None):
    vals=[]
    try:
        with path.open() as f:
            for row in csv.DictReader(f):
                if row.get('method') != method:
                    continue
                if row_type is not None and row.get('row_type') != row_type:
                    continue
                if row.get(field,'') == '':
                    continue
                vals.append(float(row[field]))
    except FileNotFoundError:
        return math.nan
    return sum(vals)/len(vals) if vals else math.nan
rows=[]
for name,mode in methods:
    d=root/name
    rows.append({
        'method': name,
        'mode': mode,
        'SP': read_reward(d/'reward_summary.csv', name, 'sp_mean'),
        'XP': read_reward(d/'reward_summary.csv', name, 'xp_mean'),
        'OOS': mean_field(d/'coverage_summary.csv', name, 'xp_out_of_sp_support_rate', 'cross_coverage'),
        'InBoth': mean_field(d/'coverage_summary.csv', name, 'in_both_support_rate', 'cross_coverage'),
        'Agreement': mean_field(d/'shared_state_mismatch.csv', name, 'action_agreement'),
        'Policy_TV': mean_field(d/'shared_state_mismatch.csv', name, 'policy_tv'),
        'Symmetric_KL': mean_field(d/'shared_state_mismatch.csv', name, 'symmetric_kl'),
        'Value_shift': mean_field(d/'shared_state_mismatch.csv', name, 'value_shift'),
        'Joint_optimal_rate': mean_field(d/'complementarity_summary.csv', name, 'joint_optimal_rate'),
        'Joint_regret': mean_field(d/'complementarity_summary.csv', name, 'joint_regret'),
        'Unilateral_regret': mean_field(d/'complementarity_summary.csv', name, 'unilateral_regret'),
    })
fields=list(rows[0].keys())
with (root/'summary.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def fmt(v):
    return 'NA' if isinstance(v,float) and math.isnan(v) else (f'{v:.4f}' if isinstance(v,float) else str(v))
lines=[
    '# TTACv2 projconf diagnostics summary',
    '',
    '| Method | SP | XP | OOS | In-both | Agreement | Policy TV | Sym KL | Joint optimal | Joint regret |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append('| ' + ' | '.join([
        r['method'], fmt(r['SP']), fmt(r['XP']), fmt(r['OOS']), fmt(r['InBoth']),
        fmt(r['Agreement']), fmt(r['Policy_TV']), fmt(r['Symmetric_KL']),
        fmt(r['Joint_optimal_rate']), fmt(r['Joint_regret'])
    ]) + ' |')
if len(rows)==2:
    a,b=rows
    lines += ['', '## Delta online - no_adapt', '']
    for k in ['SP','XP','OOS','InBoth','Agreement','Policy_TV','Symmetric_KL','Value_shift','Joint_optimal_rate','Joint_regret','Unilateral_regret']:
        lines.append(f'- {k}: {fmt(b[k]-a[k])}')
(root/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
