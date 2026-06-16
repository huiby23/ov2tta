#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/teams/ius_1663576043/hby/rl/ov2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
cd "${ROOT}"
source "${ROOT}/experiments/repro_env.sh" 2>/dev/null || true
export PYTHONPATH="${ROOT}/experiments:${ROOT}/JaxMARL:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606}"
REPORT_DIR="${REPORT_DIR:-reports/ttac_v3_alignment_sweep_${TS}}"
LOG_DIR="${LOG_DIR:-logs/ttac_v3_alignment_pair20_${TS}}"
SEED="${SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-100}"
MAX_PAIRINGS="${MAX_PAIRINGS:-20}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE="${LOG_DIR}/queue.log"
: > "${QUEUE}"
echo "${RUN_DIR}" > "${REPORT_DIR}/run_dir.txt"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }
tag_float() { echo "$1" | sed 's/-/m/g; s/\./p/g'; }

TTAC_ADAPTER_SCALE="${TTAC_ADAPTER_SCALE:-0.5}"
TTAC_TEST_HIST_KL_COEF="${TTAC_TEST_HIST_KL_COEF:-0.0}"
TTAC_TEST_EGO_KL_COEF="${TTAC_TEST_EGO_KL_COEF:-0.01}"
TTAC_TEST_CUR_KL_COEF="${TTAC_TEST_CUR_KL_COEF:-0.01}"
TTAC_TEST_LR="${TTAC_TEST_LR:-0.003}"
TTAC_TEST_UPDATE_STEPS="${TTAC_TEST_UPDATE_STEPS:-3}"
TTAC_HISTORY_LEN="${TTAC_HISTORY_LEN:-50}"
TTAC_TEST_PROJECT_BETA="${TTAC_TEST_PROJECT_BETA:-2.0}"
TTAC_TEST_SUPPORT_MIN_PROB="${TTAC_TEST_SUPPORT_MIN_PROB:-0.05}"
TTAC_TEST_SUPPORT_MAX_ENTROPY="${TTAC_TEST_SUPPORT_MAX_ENTROPY:-1.5}"
TTAC_TEST_ADVANTAGE_POWER="${TTAC_TEST_ADVANTAGE_POWER:-1.0}"
TTAC_TEST_CONTRAST_BETA="${TTAC_TEST_CONTRAST_BETA:-1.0}"
TTAC_TEST_CONTRAST_FLOOR="${TTAC_TEST_CONTRAST_FLOOR:-0.0}"
TTAC_TEST_SEMANTIC_LAMBDA="${TTAC_TEST_SEMANTIC_LAMBDA:-0.25}"
TTAC_V3_SEMANTIC_COEF="${TTAC_V3_SEMANTIC_COEF:-0.25}"
TTAC_V3_MARGIN="${TTAC_V3_MARGIN:-0.1}"
TTAC_V3_USE_CHANGE_GATE="${TTAC_V3_USE_CHANGE_GATE:-1}"
TTAC_V3_CHANGE_GATE_FLOOR="${TTAC_V3_CHANGE_GATE_FLOOR:-0.1}"
MARGIN_COEFS="${MARGIN_COEFS:-0.1 0.3}"
RECENCY_TAUS="${RECENCY_TAUS:-10.0 20.0}"

run_eval() {
  local group="$1" mode="$2" margin_coef="$3" tau="$4"
  local tag="${group}_mc$(tag_float "${margin_coef}")_tau$(tag_float "${tau}")_${mode}"
  local expected="${RUN_DIR}/reward_summary_cross_v3_align_${tag}.csv"
  if [ -f "${expected}" ]; then
    log "SKIP existing ${tag}"
    return 0
  fi
  log "EVAL_START group=${group} mode=${mode} margin_coef=${margin_coef} tau=${tau}"
  "${PYTHON}" experiments/overcooked_v2_experiments/ttac_v3/utils/visualize_ppo.py \
    --d "${RUN_DIR}" \
    --seed "${SEED}" \
    --num_seeds "${NUM_SEEDS}" \
    --cross \
    --no_viz \
    --ttac_mode "${mode}" \
    --output_tag "v3_align_${tag}" \
    --eval_batches "${EVAL_BATCHES}" \
    --max_pairings "${MAX_PAIRINGS}" \
    --ttac_adapter_scale "${TTAC_ADAPTER_SCALE}" \
    --ttac_test_hist_kl_coef "${TTAC_TEST_HIST_KL_COEF}" \
    --ttac_test_ego_kl_coef "${TTAC_TEST_EGO_KL_COEF}" \
    --ttac_test_cur_kl_coef "${TTAC_TEST_CUR_KL_COEF}" \
    --ttac_test_lr "${TTAC_TEST_LR}" \
    --ttac_test_update_steps "${TTAC_TEST_UPDATE_STEPS}" \
    --ttac_history_len "${TTAC_HISTORY_LEN}" \
    --ttac_test_project_beta "${TTAC_TEST_PROJECT_BETA}" \
    --ttac_test_support_min_prob "${TTAC_TEST_SUPPORT_MIN_PROB}" \
    --ttac_test_support_max_entropy "${TTAC_TEST_SUPPORT_MAX_ENTROPY}" \
    --ttac_test_advantage_power "${TTAC_TEST_ADVANTAGE_POWER}" \
    --ttac_test_contrast_beta "${TTAC_TEST_CONTRAST_BETA}" \
    --ttac_test_contrast_floor "${TTAC_TEST_CONTRAST_FLOOR}" \
    --ttac_test_semantic_lambda "${TTAC_TEST_SEMANTIC_LAMBDA}" \
    --ttac_v3_semantic_coef "${TTAC_V3_SEMANTIC_COEF}" \
    --ttac_v3_margin_coef "${margin_coef}" \
    --ttac_v3_margin "${TTAC_V3_MARGIN}" \
    --ttac_v3_recency_tau "${tau}" \
    --ttac_v3_use_change_gate "${TTAC_V3_USE_CHANGE_GATE}" \
    --ttac_v3_change_gate_floor "${TTAC_V3_CHANGE_GATE_FLOOR}" \
    > "${LOG_DIR}/eval_${tag}.log" 2>&1
  log "EVAL_DONE group=${group} mode=${mode} margin_coef=${margin_coef} tau=${tau}"
}

summarize() {
  REPORT_DIR="${REPORT_DIR}" RUN_DIR="${RUN_DIR}" "${PYTHON}" - <<'PY'
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
run=Path(os.environ["RUN_DIR"])
report=Path(os.environ["REPORT_DIR"])
rows=[]
for path in sorted(run.glob("reward_summary_cross_v3_align_*.csv")):
    stem=path.name.removeprefix("reward_summary_cross_v3_align_").removesuffix(".csv")
    parts=stem.split("_")
    # group may contain underscores; find mc/tau markers.
    mc_idx=next((i for i,p in enumerate(parts) if p.startswith("mc")), None)
    tau_idx=next((i for i,p in enumerate(parts) if p.startswith("tau")), None)
    if mc_idx is None or tau_idx is None:
        continue
    group="_".join(parts[:mc_idx])
    margin_coef=parts[mc_idx].removeprefix("mc").replace("p", ".")
    tau=parts[tau_idx].removeprefix("tau").replace("p", ".")
    mode="_".join(parts[tau_idx+1:])
    data=defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            data[row["policy_labels"]].append(float(row["total_reward"]))
    per_pair={k:float(np.mean(v)) for k,v in data.items()}
    rows.append({"group":group,"margin_coef":float(margin_coef),"tau":float(tau),"mode":mode,"mean":float(np.mean(list(per_pair.values()))),"std_pair":float(np.std(list(per_pair.values()))),"num_pairs":len(per_pair),"csv":str(path)})
df=pd.DataFrame(rows)
if df.empty:
    print("No v3 align CSVs yet.")
    raise SystemExit(0)
summary=df.sort_values(["group","margin_coef","tau","mode"])
summary.to_csv(report/"pair20_reward_raw.csv", index=False)
# Base/support are global sanity baselines emitted once under group=support.
# Use them for every candidate group so the pass gate does not become impossible.
global_base=float("nan")
global_support=float("nan")
base_rows=summary[summary["mode"] == "base_no_test_adapt"]
support_rows=summary[summary["mode"] == "ttac_v3_support_aw"]
if len(base_rows):
    global_base=float(base_rows.iloc[0]["mean"])
if len(support_rows):
    global_support=float(support_rows.iloc[0]["mean"])
out=[]
for (group,mc,tau), sub in summary.groupby(["group","margin_coef","tau"]):
    means={r.mode:r.mean for r in sub.itertuples()}
    true_mode={"support":"ttac_v3_support_aw","gated_semantic":"ttac_v3_gated_semantic","gated_margin":"ttac_v3_gated_margin","margin_only":"ttac_v3_margin_only"}.get(group)
    if true_mode is None or true_mode not in means:
        continue
    true=means[true_mode]
    wrong=means.get(true_mode+"_wrong_history", float("nan"))
    delayed=means.get(true_mode+"_delayed_history", float("nan"))
    random=means.get(true_mode+"_random_history", float("nan"))
    base=means.get("base_no_test_adapt", global_base)
    support=means.get("ttac_v3_support_aw", global_support)
    complete=all(not np.isnan(x) for x in [true, wrong, delayed, random, base, support])
    # Pair win-rate vs max(wrong, delayed) when files exist.
    winrate=float("nan")
    try:
        per={}
        for mode in [true_mode, true_mode+"_wrong_history", true_mode+"_delayed_history"]:
            p=Path(sub[sub.mode==mode].iloc[0].csv)
            d=defaultdict(list)
            with p.open(newline="") as f:
                for row in csv.DictReader(f):
                    d[row["policy_labels"]].append(float(row["total_reward"]))
            per[mode]={k:float(np.mean(v)) for k,v in d.items()}
        keys=sorted(set(per[true_mode]) & set(per[true_mode+"_wrong_history"]) & set(per[true_mode+"_delayed_history"]))
        winrate=float(np.mean([per[true_mode][k] > max(per[true_mode+"_wrong_history"][k], per[true_mode+"_delayed_history"][k]) for k in keys]))
    except Exception:
        pass
    gap=true - max(wrong, delayed) if not np.isnan(wrong) and not np.isnan(delayed) else float("nan")
    passed=bool(complete and true>base and true>support and true>wrong and true>delayed and true>random and (gap>=2.0 or winrate>=0.75))
    out.append({"group":group,"margin_coef":mc,"tau":tau,"complete":complete,"true_mean":true,"base_mean":base,"support_aw_mean":support,"wrong_mean":wrong,"delayed_mean":delayed,"random_mean":random,"true_minus_base":true-base if not np.isnan(base) else float("nan"),"true_minus_support_aw":true-support if not np.isnan(support) else float("nan"),"true_minus_wrong":true-wrong,"true_minus_delayed":true-delayed,"true_minus_random":true-random,"true_minus_max_wrong_delayed":gap,"winrate_true_vs_max_wrong_delayed":winrate,"pass_success_criterion":passed})
outdf=pd.DataFrame(out).sort_values(["pass_success_criterion","true_minus_max_wrong_delayed","true_mean"], ascending=[False,False,False])
outdf.to_csv(report/"pair20_reward_summary.csv", index=False)
try:
    table=outdf.to_markdown(index=False)
except Exception:
    table=outdf.to_csv(index=False)
decision="PASS: run best full eval." if bool(outdf.pass_success_criterion.any()) else "FAIL/PENDING: no v3 target passes pair20 gate yet."
(report/"pair20_reward_summary.md").write_text("# TTAC v3 pair20 reward sweep\n\n"+f"decision: **{decision}**\n\n"+table+"\n")
if len(outdf):
    fig, ax = plt.subplots(figsize=(10,5))
    labels=[f"{r.group}\nmc={r.margin_coef},tau={r.tau}" for r in outdf.itertuples()]
    ax.bar(range(len(outdf)), outdf["true_minus_max_wrong_delayed"])
    ax.axhline(0,color="black",linewidth=0.8)
    ax.axhline(2,color="gray",linestyle="--",linewidth=0.8)
    ax.set_xticks(range(len(outdf)))
    ax.set_xticklabels(labels,rotation=60,ha="right",fontsize=8)
    ax.set_ylabel("true - max(wrong, delayed)")
    ax.set_title("TTAC v3 true-history semantic gap")
    fig.tight_layout()
    fig.savefig(report/"true_vs_controls_gap.png", dpi=180)
    plt.close(fig)
(report/"v3_decision.md").write_text(f"# TTAC v3 decision\n\n{decision}\n\nSee `pair20_reward_summary.csv`.\n")
print(report/"pair20_reward_summary.md")
print(outdf.to_string(index=False))
print(decision)
PY
}

log "PAIR20_SWEEP_START report=${REPORT_DIR}"
# Baselines/sanity.
run_eval support base_no_test_adapt 0.0 10.0
run_eval support ttac_v3_support_aw 0.0 10.0
summarize | tee -a "${QUEUE}"

for tau in ${RECENCY_TAUS}; do
  run_eval gated_semantic ttac_v3_gated_semantic 0.0 "${tau}"
  run_eval gated_semantic ttac_v3_gated_semantic_wrong_history 0.0 "${tau}"
  run_eval gated_semantic ttac_v3_gated_semantic_random_history 0.0 "${tau}"
  run_eval gated_semantic ttac_v3_gated_semantic_delayed_history 0.0 "${tau}"
  summarize | tee -a "${QUEUE}"
done

for margin_coef in ${MARGIN_COEFS}; do
  for tau in ${RECENCY_TAUS}; do
    for mode in ttac_v3_gated_margin ttac_v3_gated_margin_wrong_history ttac_v3_gated_margin_random_history ttac_v3_gated_margin_delayed_history; do
      run_eval gated_margin "${mode}" "${margin_coef}" "${tau}"
    done
    summarize | tee -a "${QUEUE}"
  done
done

for margin_coef in ${MARGIN_COEFS}; do
  for tau in ${RECENCY_TAUS}; do
    for mode in ttac_v3_margin_only ttac_v3_margin_only_wrong_history ttac_v3_margin_only_random_history ttac_v3_margin_only_delayed_history; do
      run_eval margin_only "${mode}" "${margin_coef}" "${tau}"
    done
    summarize | tee -a "${QUEUE}"
  done
done

summarize | tee -a "${QUEUE}"
log "PAIR20_SWEEP_DONE"
