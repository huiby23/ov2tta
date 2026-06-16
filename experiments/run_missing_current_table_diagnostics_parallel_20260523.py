#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PYTHON_BIN = os.environ.get("PYTHON_BIN", "/root/miniconda3/envs/myconda/bin/python")
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "runs/missing_current_table_diagnostics_20260523_20260523-101808"))
MAX_WORKERS = int(os.environ.get("PARALLEL_BLOCKS", "4"))
LAYOUT = os.environ.get("LAYOUT", "counter_circuit")
EVAL_SEED = os.environ.get("EVAL_SEED", "42")
NUM_EVAL_SEEDS = os.environ.get("NUM_EVAL_SEEDS", "500")
NUM_DIAG_EPISODES = os.environ.get("NUM_DIAG_EPISODES", "20")
TOP_K = os.environ.get("TOP_K", "10")
MAX_PAIRS = os.environ.get("MAX_PAIRS", "30")
COMPATIBILITY_SAMPLE_LIMIT = os.environ.get("COMPATIBILITY_SAMPLE_LIMIT", "256")
FORCE = os.environ.get("FORCE", "0") == "1"

PPO_DIAG = "experiments/overcooked_v2_experiments/ppo/utils/zsc_diagnostics.py"
E3T_DIAG = "experiments/overcooked_v2_experiments/ppo_e3t_official/utils/zsc_diagnostics.py"

PPO_STATE_AUG = "runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full"
MAPPO_CNN_STANDARD = "runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_standard_20260503-201254/20260503-201315_wvbb7dde_counter_circuit_avs-full"
MAPPO_CNN_STATE_AUG = "runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_state_aug_20260503-201254/20260503-202330_umwsoxcy_counter_circuit_avs-full"
MAPPO_RNN_STANDARD = "runs/figure4_mappo_rnn_64_16_unified_rerun_mappo_rnn_standard_20260515-unified-rerun/20260515-143455_p1b6tsdz_counter_circuit_avs-full"
MAPPO_RNN_STATE_AUG = "runs/figure4_mappo_rnn_64_16_unified_rerun_mappo_rnn_state_aug_20260515-unified-rerun/20260515-152618_9mkg137e_counter_circuit_avs-full"
E3T_CNN_STANDARD_PRED = "runs/figure4_ppo_e3t_official_predicted_ce_full_20260428-ppo-e3t-ablation/20260428-124613_mz4qrn7j_counter_circuit_avs-full"
E3T_CNN_STANDARD_NOCE = "runs/figure4_ppo_e3t_official_no_ce_full_20260428-ppo-e3t-ablation/20260428-155737_0foocnfr_counter_circuit_avs-full"
E3T_CNN_STANDARD_CONST = "runs/figure4_ppo_e3t_official_constant_ce_full_20260428-ppo-e3t-ablation/20260428-142809_ju78l6cx_counter_circuit_avs-full"
E3T_CNN_STATE_PRED = "runs/figure4_ppo_e3t_state_aug_predicted_ce_full_20260428-ppo-e3t-state-aug/20260428-163847_50tx0n6v_counter_circuit_avs-full"
E3T_CNN_STATE_NOCE = "runs/figure4_ppo_e3t_state_aug_no_ce_full_20260428-ppo-e3t-state-aug/20260428-200739_xsd6oyim_counter_circuit_avs-full"
E3T_CNN_STATE_CONST = "runs/figure4_ppo_e3t_state_aug_constant_ce_full_20260428-ppo-e3t-state-aug/20260428-182936_aza8mbkr_counter_circuit_avs-full"
E3T_RNN_STANDARD = "runs/figure4_e3t_ppo_rnn_64_16_standard_full_rnnfix_20260504-130101/20260504-130114_qrjr5dlz_counter_circuit_avs-full"
E3T_RNN_STATE_AUG = "runs/figure4_e3t_ppo_rnn_64_16_state_aug_full_rnnfix_20260504-130101/20260504-175745_vqj6owwb_counter_circuit_avs-full"
FCP_OLD = "runs/figure4_fcp_cnn_64_16/FCP_ppo_cnn_standard_64_16_counter_circuit_8policies_x10_ow2rp4gr_20260506-172902"
FCP_MM = "runs/fcp_mep_frozen_mm_K10_64_16_10000000_20260516-090954/20260516-091009_ys6qo8lj_counter_circuit_avs-full"
MEP_005 = "runs/mep_realpop_mm1_mp1_K5_ent0.05_64_16_10000000_seed10_20260516-095429/20260516-095441_lvbnw6rf_counter_circuit_avs-full"
MEP_005_MP2 = "runs/mep_realpop_mm1_mp2_K5_ent0.05_64_16_10000000_seed10_20260516-121256/20260516-121311_ihiy6517_counter_circuit_avs-full"
MEP_005_STATE = "runs/population_state_aug_64_16_div005_mep_state_aug_mm1_mp1_K5_ent0.05_64_16_10000000_sa10_stateaug-div005-20260517-124718/20260517-124732_c77pgga1_counter_circuit_avs-full"
TRAJEDI_010_CURRENT = "runs/trajedi_unified_rerun_mm1_mp1_K5_div0.1_64_16_10000000_seed10_20260515-174148/20260515-174201_cglkefqe_counter_circuit_avs-full"
TRAJEDI_005 = "runs/trajedi_realpop_mm1_mp1_K5_div0.05_64_16_10000000_seed10_20260516-151722/20260516-151735_1gd6vw78_counter_circuit_avs-full"
TRAJEDI_005_STATE = "runs/population_state_aug_64_16_div005_trajedi_state_aug_mm1_mp1_K5_div0.05_64_16_10000000_sa10_stateaug-div005-20260517-124718/20260517-154243_9i6qhkle_counter_circuit_avs-full"
GAMMA_PPO_CNN = "runs/gamma_mix025_ppo_standard_source_64_16_10000000_20260514-174603/20260514-174617_nc84elts_counter_circuit_avs-full"
GAMMA_MAPPO_RNN = "runs/gamma_mix025_mappo_rnn_standard_source_64_16_10000000_20260515-005018/20260515-015313_yljod4y5_counter_circuit_avs-full"

# block, diag_py, base_name, base_dir, base_backend, comp_name, comp_dir, comp_backend
BLOCKS = [
    ("mappo_cnn_standard_vs_mappo_rnn_standard", PPO_DIAG, "mappo_cnn_standard", MAPPO_CNN_STANDARD, "mappo", "mappo_rnn_standard", MAPPO_RNN_STANDARD, "mappo"),
    ("mappo_state_aug_cnn_vs_rnn", PPO_DIAG, "mappo_cnn_state_aug", MAPPO_CNN_STATE_AUG, "mappo", "mappo_rnn_state_aug", MAPPO_RNN_STATE_AUG, "mappo"),
    ("e3t_cnn_standard_pred_vs_noce", E3T_DIAG, "ppo_e3t_predicted_ce_no_state_aug", E3T_CNN_STANDARD_PRED, "ppo", "ppo_e3t_no_ce_no_state_aug", E3T_CNN_STANDARD_NOCE, "ppo"),
    ("e3t_cnn_const_vs_state_pred", E3T_DIAG, "ppo_e3t_constant_ce_no_state_aug", E3T_CNN_STANDARD_CONST, "ppo", "ppo_e3t_predicted_ce_state_aug", E3T_CNN_STATE_PRED, "ppo"),
    ("e3t_cnn_state_noce_vs_state_const", E3T_DIAG, "ppo_e3t_no_ce_state_aug", E3T_CNN_STATE_NOCE, "ppo", "ppo_e3t_constant_ce_state_aug", E3T_CNN_STATE_CONST, "ppo"),
    ("e3t_rnn_standard_vs_state_aug", E3T_DIAG, "ppo_e3t_rnn_standard_fixed", E3T_RNN_STANDARD, "ppo", "ppo_e3t_rnn_state_aug_fixed", E3T_RNN_STATE_AUG, "ppo"),
    ("fcp_old_vs_fcp_mm", PPO_DIAG, "fcp_old", FCP_OLD, "ppo", "fcp_mm_frozen_ppo_partner", FCP_MM, "ppo"),
    ("mep_ent005_vs_mep_ent005_mp2", PPO_DIAG, "mep_ent005", MEP_005, "ppo", "mep_ent005_mp2", MEP_005_MP2, "ppo"),
    ("trajedi_div01_current_vs_div005", PPO_DIAG, "trajedi", TRAJEDI_010_CURRENT, "ppo", "trajedi_div005", TRAJEDI_005, "ppo"),
    ("population_state_aug_mep005_vs_trajedi005", PPO_DIAG, "mep_ent005_state_aug", MEP_005_STATE, "ppo", "trajedi_div005_state_aug", TRAJEDI_005_STATE, "ppo"),
    ("gamma_mix25_ppo_vs_mappo_source", PPO_DIAG, "gamma_mix25_ppo_cnn", GAMMA_PPO_CNN, "ppo", "gamma_mix25_mappo_rnn", GAMMA_MAPPO_RNN, "ppo"),
]


def command_for(block):
    name, diag_py, base_name, base_dir, base_backend, comp_name, comp_dir, comp_backend = block
    out = OUTPUT_ROOT / name
    return [
        PYTHON_BIN, diag_py,
        "--baseline_run_dir", base_dir,
        "--comparison_run_dir", comp_dir,
        "--baseline_name", base_name,
        "--comparison_name", comp_name,
        "--baseline_backend", base_backend,
        "--comparison_backend", comp_backend,
        "--layout", LAYOUT,
        "--eval_seed", EVAL_SEED,
        "--num_eval_seeds", NUM_EVAL_SEEDS,
        "--num_diag_episodes", NUM_DIAG_EPISODES,
        "--top_k", TOP_K,
        "--compatibility_sample_limit", COMPATIBILITY_SAMPLE_LIMIT,
        "--max_pairs", MAX_PAIRS,
        "--output_dir", str(out),
    ]


def aggregate():
    rows = []
    valid_blocks = {b[0] for b in BLOCKS}
    for cov in sorted(OUTPUT_ROOT.glob("*/coverage_summary.csv")):
        block = cov.parent.name
        if block not in valid_blocks:
            continue
        mm = cov.parent / "shared_state_mismatch.csv"
        comp = cov.parent / "complementarity_summary.csv"
        methods = []
        with cov.open() as f:
            seen = set()
            for r in csv.DictReader(f):
                if r.get("row_type") == "cross_coverage" and r.get("method") not in seen:
                    seen.add(r["method"])
                    methods.append(r["method"])
        def mean_for(path, method, field, row_type=None):
            vals = []
            if not path.exists():
                return math.nan
            with path.open() as f:
                for r in csv.DictReader(f):
                    if r.get("method") != method:
                        continue
                    if row_type is not None and r.get("row_type") != row_type:
                        continue
                    value = r.get(field, "")
                    if not value:
                        continue
                    try:
                        vals.append(float(value))
                    except ValueError:
                        pass
            return sum(vals) / len(vals) if vals else math.nan
        for method in methods:
            rows.append((
                block, method,
                mean_for(cov, method, "mean_reward", "selfplay_support"),
                mean_for(cov, method, "mean_reward", "cross_coverage"),
                mean_for(cov, method, "xp_out_of_sp_support_rate", "cross_coverage"),
                mean_for(cov, method, "in_both_support_rate", "cross_coverage"),
                mean_for(mm, method, "action_agreement"),
                mean_for(mm, method, "policy_tv"),
                mean_for(comp, method, "joint_regret"),
            ))
    def fmt(x):
        return "NA" if math.isnan(x) else f"{x:.4f}"
    with (OUTPUT_ROOT / "summary_current_missing.md").open("w") as f:
        f.write("# Missing Current-Table Diagnostics Summary\n\n")
        f.write(f"Output root: `{OUTPUT_ROOT}`\n\n")
        f.write("| Block | Method | SP | XP | OOS | In-both | Agreement | Policy TV | Joint regret |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write("| " + " | ".join([row[0], row[1], *[fmt(x) for x in row[2:]]]) + " |\n")
    with (OUTPUT_ROOT / "summary_current_missing.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "method", "SP", "XP", "OOS", "InBoth", "Agreement", "Policy_TV", "Joint_regret"])
        writer.writerows(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "experiments")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("ZSC_FORWARD_CHUNK_SIZE", "32768")
    env.setdefault("PYTHONUNBUFFERED", "1")

    with (OUTPUT_ROOT / "parallel_readme.txt").open("w") as f:
        f.write(f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        f.write(f"parallel_blocks={MAX_WORKERS}\n")
        f.write(f"num_eval_seeds={NUM_EVAL_SEEDS}\nnum_diag_episodes={NUM_DIAG_EPISODES}\nmax_pairs={MAX_PAIRS}\ncompatibility_sample_limit={COMPATIBILITY_SAMPLE_LIMIT}\n")

    status_path = OUTPUT_ROOT / "parallel_status.csv"
    if not status_path.exists() or FORCE:
        status_path.write_text("timestamp,block,status,returncode\n")

    pending = []
    for block in BLOCKS:
        name = block[0]
        out = OUTPUT_ROOT / name
        report = out / "report.md"
        coverage = out / "coverage_summary.csv"
        if report.exists() and coverage.exists() and not FORCE:
            with status_path.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},{name},skip_existing,0\n")
            continue
        pending.append(block)

    running = []
    completed = 0
    while pending or running:
        while pending and len(running) < MAX_WORKERS:
            block = pending.pop(0)
            name = block[0]
            out = OUTPUT_ROOT / name
            out.mkdir(parents=True, exist_ok=True)
            log = out / "run.log"
            with log.open("w") as handle:
                proc = subprocess.Popen(command_for(block), stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            running.append((block, proc, log))
            with status_path.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},{name},started,{proc.pid}\n")
            print(f"[parallel-diag] started {name} pid={proc.pid} ({len(running)}/{MAX_WORKERS})", flush=True)

        time.sleep(5)
        still = []
        for block, proc, log in running:
            rc = proc.poll()
            name = block[0]
            if rc is None:
                still.append((block, proc, log))
                continue
            completed += 1
            status = "ok" if rc == 0 else f"failed_{rc}"
            with status_path.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},{name},{status},{rc}\n")
            print(f"[parallel-diag] finished {name} status={status} ({completed} done)", flush=True)
        running = still

    aggregate()
    print(f"[parallel-diag] done output={OUTPUT_ROOT}", flush=True)

if __name__ == "__main__":
    main()
