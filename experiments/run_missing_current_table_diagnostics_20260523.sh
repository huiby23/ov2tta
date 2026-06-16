#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/myconda/bin/python}"
JOB_ROOT="${JOB_ROOT:-logs/missing_current_table_diagnostics_20260523}"
PID_FILE="${PID_FILE:-${JOB_ROOT}/current.pid}"
LATEST_LOG="${LATEST_LOG:-${JOB_ROOT}/latest.log}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/missing_current_table_diagnostics_20260523_${TAG}}"
mkdir -p "$JOB_ROOT"

if [[ -f experiments/repro_env.sh ]]; then
  # shellcheck disable=SC1091
  source experiments/repro_env.sh
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
export ZSC_FORWARD_CHUNK_SIZE="${ZSC_FORWARD_CHUNK_SIZE:-32768}"

LAYOUT="${LAYOUT:-counter_circuit}"
EVAL_SEED="${EVAL_SEED:-42}"
NUM_EVAL_SEEDS="${NUM_EVAL_SEEDS:-500}"
NUM_DIAG_EPISODES="${NUM_DIAG_EPISODES:-20}"
TOP_K="${TOP_K:-10}"
MAX_PAIRS="${MAX_PAIRS:-30}"
COMPATIBILITY_SAMPLE_LIMIT="${COMPATIBILITY_SAMPLE_LIMIT:-256}"
FORCE="${FORCE:-0}"

PPO_DIAG="experiments/overcooked_v2_experiments/ppo/utils/zsc_diagnostics.py"
E3T_DIAG="experiments/overcooked_v2_experiments/ppo_e3t_official/utils/zsc_diagnostics.py"

PPO_STANDARD="runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full"
PPO_STATE_AUG="runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full"
MAPPO_CNN_STANDARD="runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_standard_20260503-201254/20260503-201315_wvbb7dde_counter_circuit_avs-full"
MAPPO_CNN_STATE_AUG="runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_state_aug_20260503-201254/20260503-202330_umwsoxcy_counter_circuit_avs-full"
MAPPO_RNN_STANDARD="runs/figure4_mappo_rnn_64_16_unified_rerun_mappo_rnn_standard_20260515-unified-rerun/20260515-143455_p1b6tsdz_counter_circuit_avs-full"
MAPPO_RNN_STATE_AUG="runs/figure4_mappo_rnn_64_16_unified_rerun_mappo_rnn_state_aug_20260515-unified-rerun/20260515-152618_9mkg137e_counter_circuit_avs-full"
E3T_CNN_STANDARD_PRED="runs/figure4_ppo_e3t_official_predicted_ce_full_20260428-ppo-e3t-ablation/20260428-124613_mz4qrn7j_counter_circuit_avs-full"
E3T_CNN_STANDARD_NOCE="runs/figure4_ppo_e3t_official_no_ce_full_20260428-ppo-e3t-ablation/20260428-155737_0foocnfr_counter_circuit_avs-full"
E3T_CNN_STANDARD_CONST="runs/figure4_ppo_e3t_official_constant_ce_full_20260428-ppo-e3t-ablation/20260428-142809_ju78l6cx_counter_circuit_avs-full"
E3T_CNN_STATE_PRED="runs/figure4_ppo_e3t_state_aug_predicted_ce_full_20260428-ppo-e3t-state-aug/20260428-163847_50tx0n6v_counter_circuit_avs-full"
E3T_CNN_STATE_NOCE="runs/figure4_ppo_e3t_state_aug_no_ce_full_20260428-ppo-e3t-state-aug/20260428-200739_xsd6oyim_counter_circuit_avs-full"
E3T_CNN_STATE_CONST="runs/figure4_ppo_e3t_state_aug_constant_ce_full_20260428-ppo-e3t-state-aug/20260428-182936_aza8mbkr_counter_circuit_avs-full"
E3T_RNN_STANDARD="runs/figure4_e3t_ppo_rnn_64_16_standard_full_rnnfix_20260504-130101/20260504-130114_qrjr5dlz_counter_circuit_avs-full"
E3T_RNN_STATE_AUG="runs/figure4_e3t_ppo_rnn_64_16_state_aug_full_rnnfix_20260504-130101/20260504-175745_vqj6owwb_counter_circuit_avs-full"
FCP_OLD="runs/figure4_fcp_cnn_64_16/FCP_ppo_cnn_standard_64_16_counter_circuit_8policies_x10_ow2rp4gr_20260506-172902"
FCP_MM="runs/fcp_mep_frozen_mm_K10_64_16_10000000_20260516-090954/20260516-091009_ys6qo8lj_counter_circuit_avs-full"
MEP_005="runs/mep_realpop_mm1_mp1_K5_ent0.05_64_16_10000000_seed10_20260516-095429/20260516-095441_lvbnw6rf_counter_circuit_avs-full"
MEP_005_MP2="runs/mep_realpop_mm1_mp2_K5_ent0.05_64_16_10000000_seed10_20260516-121256/20260516-121311_ihiy6517_counter_circuit_avs-full"
MEP_005_STATE="runs/population_state_aug_64_16_div005_mep_state_aug_mm1_mp1_K5_ent0.05_64_16_10000000_sa10_stateaug-div005-20260517-124718/20260517-124732_c77pgga1_counter_circuit_avs-full"
TRAJEDI_010_CURRENT="runs/trajedi_unified_rerun_mm1_mp1_K5_div0.1_64_16_10000000_seed10_20260515-174148/20260515-174201_cglkefqe_counter_circuit_avs-full"
TRAJEDI_005="runs/trajedi_realpop_mm1_mp1_K5_div0.05_64_16_10000000_seed10_20260516-151722/20260516-151735_1gd6vw78_counter_circuit_avs-full"
TRAJEDI_005_STATE="runs/population_state_aug_64_16_div005_trajedi_state_aug_mm1_mp1_K5_div0.05_64_16_10000000_sa10_stateaug-div005-20260517-124718/20260517-154243_9i6qhkle_counter_circuit_avs-full"
GAMMA_PPO_CNN="runs/gamma_mix025_ppo_standard_source_64_16_10000000_20260514-174603/20260514-174617_nc84elts_counter_circuit_avs-full"
GAMMA_MAPPO_RNN="runs/gamma_mix025_mappo_rnn_standard_source_64_16_10000000_20260515-005018/20260515-015313_yljod4y5_counter_circuit_avs-full"

is_running() { [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; }

run_diag_py() {
  local block="$1" diag_py="$2" base_name="$3" base_dir="$4" base_backend="$5" comp_name="$6" comp_dir="$7" comp_backend="$8"
  local out="$OUTPUT_ROOT/$block"
  mkdir -p "$out"
  if [[ ! -d "$base_dir" ]]; then echo "$(date -Is),$block,missing_base,$base_name,$base_dir" | tee -a "$OUTPUT_ROOT/status.csv"; return 0; fi
  if [[ ! -d "$comp_dir" ]]; then echo "$(date -Is),$block,missing_comparison,$comp_name,$comp_dir" | tee -a "$OUTPUT_ROOT/status.csv"; return 0; fi
  if [[ "$FORCE" != "1" && -f "$out/report.md" && -f "$out/coverage_summary.csv" ]]; then echo "$(date -Is),$block,skip_existing,$base_name,$comp_name" | tee -a "$OUTPUT_ROOT/status.csv"; return 0; fi
  echo "[missing-diag] block=$block"
  echo "[missing-diag]   base=$base_name backend=$base_backend dir=$base_dir"
  echo "[missing-diag]   comp=$comp_name backend=$comp_backend dir=$comp_dir"
  echo "[missing-diag]   out=$out"
  set +e
  PYTHONPATH=experiments "$PYTHON_BIN" "$diag_py" \
    --baseline_run_dir "$base_dir" --comparison_run_dir "$comp_dir" \
    --baseline_name "$base_name" --comparison_name "$comp_name" \
    --baseline_backend "$base_backend" --comparison_backend "$comp_backend" \
    --layout "$LAYOUT" --eval_seed "$EVAL_SEED" --num_eval_seeds "$NUM_EVAL_SEEDS" \
    --num_diag_episodes "$NUM_DIAG_EPISODES" --top_k "$TOP_K" \
    --compatibility_sample_limit "$COMPATIBILITY_SAMPLE_LIMIT" --max_pairs "$MAX_PAIRS" \
    --output_dir "$out" > "$out/run.log" 2>&1
  local code=$?
  set -e
  if [[ $code -eq 0 ]]; then
    echo "$(date -Is),$block,ok,$base_name,$comp_name" | tee -a "$OUTPUT_ROOT/status.csv"
  else
    echo "$(date -Is),$block,failed_$code,$base_name,$comp_name" | tee -a "$OUTPUT_ROOT/status.csv"
    echo "[missing-diag] failed block=$block code=$code; see $out/run.log"
  fi
}

aggregate_output() {
  PYTHONPATH=experiments "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import csv, math, sys
root = Path(sys.argv[1])
rows = []
for cov in sorted(root.glob("*/coverage_summary.csv")):
    block = cov.parent.name
    mm = cov.parent / "shared_state_mismatch.csv"
    comp = cov.parent / "complementarity_summary.csv"
    methods = []
    with cov.open() as f:
        seen = set()
        for r in csv.DictReader(f):
            if r.get("row_type") == "cross_coverage" and r.get("method") not in seen:
                seen.add(r["method"]); methods.append(r["method"])
    def mean_for(path, method, field, row_type=None):
        vals = []
        if not path.exists(): return math.nan
        with path.open() as f:
            for r in csv.DictReader(f):
                if r.get("method") != method: continue
                if row_type is not None and r.get("row_type") != row_type: continue
                value = r.get(field, "")
                if value == "": continue
                try: vals.append(float(value))
                except ValueError: pass
        return sum(vals) / len(vals) if vals else math.nan
    for method in methods:
        rows.append((block, method,
            mean_for(cov, method, "mean_reward", "selfplay_support"),
            mean_for(cov, method, "mean_reward", "cross_coverage"),
            mean_for(cov, method, "xp_out_of_sp_support_rate", "cross_coverage"),
            mean_for(cov, method, "in_both_support_rate", "cross_coverage"),
            mean_for(mm, method, "action_agreement"),
            mean_for(mm, method, "policy_tv"),
            mean_for(comp, method, "joint_regret")))
def fmt(x): return "NA" if math.isnan(x) else f"{x:.4f}"
with (root / "summary_current_missing.md").open("w") as f:
    f.write("# Missing Current-Table Diagnostics Summary\n\n")
    f.write(f"Output root: `{root}`\n\n")
    f.write("| Block | Method | SP | XP | OOS | In-both | Agreement | Policy TV | Joint regret |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        f.write("| " + " | ".join([row[0], row[1], *[fmt(x) for x in row[2:]]]) + " |\n")
with (root / "summary_current_missing.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["block","method","SP","XP","OOS","InBoth","Agreement","Policy_TV","Joint_regret"]); w.writerows(rows)
print(root / "summary_current_missing.md")
PY
}

scan_existing() {
  PYTHONPATH=experiments "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import csv
current = {
"PPO CNN standard": (167.8600,61.4067,["ppo_cnn_standard"]),
"PPO CNN state-aug": (201.8360,156.8307,["ppo_cnn_state_aug"]),
"MAPPO CNN standard": (155.8480,101.3000,["mappo_cnn_standard"]),
"MAPPO CNN state-aug": (152.0120,104.8711,["mappo_cnn_state_aug"]),
"MAPPO RNN standard": (205.8200,111.4178,["mappo_rnn_standard"]),
"MAPPO RNN state-aug": (214.8720,156.8667,["mappo_rnn_state_aug","mappo_rnn_state_aug_current"]),
"PPO-E3T CNN no-state predicted CE": (163.3360,74.5582,["ppo_e3t_predicted_ce_no_state_aug","ppo_e3t_predicted_ce_no_state_aug_full_obs"]),
"PPO-E3T CNN no-state no CE": (164.8480,57.4662,["ppo_e3t_no_ce_no_state_aug","ppo_e3t_no_ce_no_state_aug_full_obs"]),
"PPO-E3T CNN no-state constant CE": (167.2640,64.7502,["ppo_e3t_constant_ce_no_state_aug"]),
"PPO-E3T CNN state-aug predicted CE": (180.3080,146.9471,["ppo_e3t_predicted_ce_state_aug"]),
"PPO-E3T CNN state-aug no CE": (196.5600,159.3311,["ppo_e3t_no_ce_state_aug"]),
"PPO-E3T CNN state-aug constant CE": (174.1240,143.5440,["ppo_e3t_constant_ce_state_aug"]),
"PPO-E3T RNN standard fixed": (122.3920,30.8596,["ppo_e3t_rnn_standard_fixed"]),
"PPO-E3T RNN state-aug fixed": (150.4680,93.7609,["ppo_e3t_rnn_state_aug_fixed"]),
"FCP old": (0.0,0.0,["fcp_old"]),
"FCP + MM frozen PPO partner": (145.7480,91.4218,["fcp_mm_frozen_ppo_partner"]),
"MEP ent=0.1 standard": (163.3560,90.4582,["mep"]),
"MEP ent=0.05 standard": (170.0160,123.0564,["mep_ent005"]),
"MEP ent=0.05 MP=2 standard": (168.7600,97.8396,["mep_ent005_mp2"]),
"MEP ent=0.1 state-aug": (171.0240,151.3302,["mep_state_aug"]),
"MEP ent=0.05 state-aug": (169.4080,148.1084,["mep_ent005_state_aug"]),
"TrajeDi div=0.1 standard": (162.1120,97.3053,["trajedi"]),
"TrajeDi div=0.05 standard": (166.4320,118.7120,["trajedi_div005"]),
"TrajeDi div=0.1 state-aug": (166.4480,144.4791,["trajedi_state_aug"]),
"TrajeDi div=0.05 state-aug": (173.6560,153.0924,["trajedi_div005_state_aug"]),
"GAMMA mix25 PPO-CNN standard source": (152.4280,118.6653,["gamma_mix25_ppo_cnn"]),
"GAMMA mix25 MAPPO-RNN standard source": (181.8160,160.2213,["gamma_mix25_mappo_rnn"]),
}
roots=[Path("runs/zsc_attribution_suite_64_16_20260512-172337"),Path("runs/e3t_mechanism_analysis_full_20260502-mech-full"),Path("runs/mep_vs_trajedi_zsc_diagnostics_20260512-002115")]
roots += sorted(Path("runs").glob("missing_current_table_diagnostics_20260523_*"))
diag=[]
for root in roots:
    if not root.exists(): continue
    for cov in root.rglob("coverage_summary.csv"):
        methods=set()
        with cov.open() as f:
            for r in csv.DictReader(f):
                if r.get("row_type")=="cross_coverage": methods.add(r.get("method",""))
        for m in methods:
            sp=[]; xp=[]
            with cov.open() as f:
                for r in csv.DictReader(f):
                    if r.get("method") != m: continue
                    if r.get("row_type") == "selfplay_support": sp.append(float(r["mean_reward"]))
                    if r.get("row_type") == "cross_coverage": xp.append(float(r["mean_reward"]))
            if sp and xp: diag.append((m,sum(sp)/len(sp),sum(xp)/len(xp),str(cov.parent)))
print("# Existing diagnostics scan")
print("\n## Exact/reward-compatible matches")
print("| Current row | Diag label | SP | XP | Source |")
print("|---|---|---:|---:|---|")
matched=set()
for row,(sp,xp,aliases) in current.items():
    for label,dsp,dxp,src in diag:
        if label in aliases and abs(dsp-sp)<0.15 and abs(dxp-xp)<0.15:
            print(f"| {row} | {label} | {dsp:.3f} | {dxp:.3f} | `{src}` |")
            matched.add(row)
print("\n## Stale/name-compatible but reward-mismatched diagnostics")
print("| Current row | Expected SP/XP | Diag label | Diag SP/XP | Source |")
print("|---|---:|---|---:|---|")
for row,(sp,xp,aliases) in current.items():
    for label,dsp,dxp,src in diag:
        if label in aliases and not (abs(dsp-sp)<0.15 and abs(dxp-xp)<0.15):
            print(f"| {row} | {sp:.3f}/{xp:.3f} | {label} | {dsp:.3f}/{dxp:.3f} | `{src}` |")
print("\n## Still missing exact diagnostics")
for row in current:
    if row not in matched: print(f"- {row}")
PY
}

run_all() {
  mkdir -p "$OUTPUT_ROOT"
  echo "timestamp,block,status,baseline,comparison" > "$OUTPUT_ROOT/status.csv"
  cat > "$OUTPUT_ROOT/README.md" <<EOF
# Missing Current-Table Diagnostics

- generated_at: $(date -Is)
- cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}
- eval_seed: ${EVAL_SEED}
- num_eval_seeds: ${NUM_EVAL_SEEDS}
- num_diag_episodes: ${NUM_DIAG_EPISODES}
- max_pairs: ${MAX_PAIRS}
- compatibility_sample_limit: ${COMPATIBILITY_SAMPLE_LIMIT}
EOF
  run_diag_py mappo_cnn_standard_vs_mappo_rnn_standard "$PPO_DIAG" mappo_cnn_standard "$MAPPO_CNN_STANDARD" mappo mappo_rnn_standard "$MAPPO_RNN_STANDARD" mappo
  run_diag_py mappo_cnn_state_aug_vs_ppo_state_aug "$PPO_DIAG" mappo_cnn_state_aug "$MAPPO_CNN_STATE_AUG" mappo ppo_cnn_state_aug "$PPO_STATE_AUG" ppo
  run_diag_py mappo_rnn_state_aug_current_vs_ppo_state_aug "$PPO_DIAG" mappo_rnn_state_aug "$MAPPO_RNN_STATE_AUG" mappo ppo_cnn_state_aug "$PPO_STATE_AUG" ppo
  run_diag_py e3t_cnn_standard_pred_vs_noce "$E3T_DIAG" ppo_e3t_predicted_ce_no_state_aug "$E3T_CNN_STANDARD_PRED" ppo ppo_e3t_no_ce_no_state_aug "$E3T_CNN_STANDARD_NOCE" ppo
  run_diag_py e3t_cnn_standard_const_vs_pred "$E3T_DIAG" ppo_e3t_constant_ce_no_state_aug "$E3T_CNN_STANDARD_CONST" ppo ppo_e3t_predicted_ce_no_state_aug "$E3T_CNN_STANDARD_PRED" ppo
  run_diag_py e3t_cnn_state_pred_vs_noce "$E3T_DIAG" ppo_e3t_predicted_ce_state_aug "$E3T_CNN_STATE_PRED" ppo ppo_e3t_no_ce_state_aug "$E3T_CNN_STATE_NOCE" ppo
  run_diag_py e3t_cnn_state_const_vs_pred "$E3T_DIAG" ppo_e3t_constant_ce_state_aug "$E3T_CNN_STATE_CONST" ppo ppo_e3t_predicted_ce_state_aug "$E3T_CNN_STATE_PRED" ppo
  run_diag_py e3t_rnn_standard_vs_state_aug "$E3T_DIAG" ppo_e3t_rnn_standard_fixed "$E3T_RNN_STANDARD" ppo ppo_e3t_rnn_state_aug_fixed "$E3T_RNN_STATE_AUG" ppo
  run_diag_py fcp_old_vs_fcp_mm "$PPO_DIAG" fcp_old "$FCP_OLD" ppo fcp_mm_frozen_ppo_partner "$FCP_MM" ppo
  run_diag_py mep_ent005_vs_mep_ent005_mp2 "$PPO_DIAG" mep_ent005 "$MEP_005" ppo mep_ent005_mp2 "$MEP_005_MP2" ppo
  run_diag_py trajedi_div01_current_vs_div005 "$PPO_DIAG" trajedi "$TRAJEDI_010_CURRENT" ppo trajedi_div005 "$TRAJEDI_005" ppo
  run_diag_py mep_ent005_state_vs_trajedi_div005_state "$PPO_DIAG" mep_ent005_state_aug "$MEP_005_STATE" ppo trajedi_div005_state_aug "$TRAJEDI_005_STATE" ppo
  run_diag_py gamma_mix25_ppo_vs_mappo_source "$PPO_DIAG" gamma_mix25_ppo_cnn "$GAMMA_PPO_CNN" ppo gamma_mix25_mappo_rnn "$GAMMA_MAPPO_RNN" ppo
  aggregate_output
  echo "[missing-diag] done output=$OUTPUT_ROOT"
}

case "${1:-status}" in
  scan) scan_existing ;;
  run) run_all ;;
  start)
    if is_running; then echo "already running pid=$(cat "$PID_FILE") log=$LATEST_LOG"; exit 0; fi
    nohup "$0" run > "$LATEST_LOG" 2>&1 & echo $! > "$PID_FILE"
    echo "started pid=$(cat "$PID_FILE") log=$LATEST_LOG output=$OUTPUT_ROOT" ;;
  status)
    if is_running; then echo "running pid=$(cat "$PID_FILE") log=$LATEST_LOG"; else echo "not running"; [[ -f "$PID_FILE" ]] && echo "last pid=$(cat "$PID_FILE")"; fi
    [[ -f "$LATEST_LOG" ]] && tail -n 30 "$LATEST_LOG" ;;
  tail) tail -n "${N:-80}" "$LATEST_LOG" ;;
  stop)
    if is_running; then kill "$(cat "$PID_FILE")"; echo "stopped pid=$(cat "$PID_FILE")"; else echo "not running"; fi ;;
  *) echo "Usage: $0 {scan|start|run|status|tail|stop}" >&2; exit 2 ;;
esac
