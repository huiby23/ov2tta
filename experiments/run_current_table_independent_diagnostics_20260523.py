#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PYTHON_BIN = os.environ.get("PYTHON_BIN", "/root/miniconda3/envs/myconda/bin/python")
TAG = os.environ.get("TAG", time.strftime("%Y%m%d-%H%M%S"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", f"runs/current_table_independent_diagnostics_index_20260523_{TAG}"))
DIAG_TAG = os.environ.get("DIAG_TAG", "current_table_20260523")
STORE_IN_RUN_DIR = os.environ.get("STORE_IN_RUN_DIR", "1") != "0"
PARALLEL_METHODS = int(os.environ.get("PARALLEL_METHODS", "4"))
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


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    display_name: str
    setting: str
    run_dir: str
    diag_py: str
    backend: str
    eval_mode: str = "memory_off"


METHODS = [
    MethodSpec("ppo_cnn_standard", "PPO CNN standard", "standard", "runs/figure4_standard/20260403-114253_lhan2u0o_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("mappo_cnn_standard", "MAPPO CNN standard", "standard", "runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_standard_20260503-201254/20260503-201315_wvbb7dde_counter_circuit_avs-full", PPO_DIAG, "mappo"),
    MethodSpec("mappo_rnn_standard", "MAPPO RNN standard", "standard", "runs/figure4_mappo_rnn_64_16_unified_rerun_mappo_rnn_standard_20260515-unified-rerun/20260515-143455_p1b6tsdz_counter_circuit_avs-full", PPO_DIAG, "mappo"),
    MethodSpec("ppo_e3t_cnn_no_state_predicted_ce", "PPO-E3T CNN no-state predicted CE", "standard", "runs/figure4_ppo_e3t_official_predicted_ce_full_20260428-ppo-e3t-ablation/20260428-124613_mz4qrn7j_counter_circuit_avs-full", E3T_DIAG, "ppo"),
    MethodSpec("ppo_e3t_cnn_no_state_no_ce", "PPO-E3T CNN no-state no CE", "standard", "runs/figure4_ppo_e3t_official_no_ce_full_20260428-ppo-e3t-ablation/20260428-155737_0foocnfr_counter_circuit_avs-full", E3T_DIAG, "ppo"),
    MethodSpec("ppo_e3t_cnn_no_state_constant_ce", "PPO-E3T CNN no-state constant CE", "standard", "runs/figure4_ppo_e3t_official_constant_ce_full_20260428-ppo-e3t-ablation/20260428-142809_ju78l6cx_counter_circuit_avs-full", E3T_DIAG, "ppo"),
    MethodSpec("ppo_e3t_rnn_standard_fixed", "PPO-E3T RNN standard fixed", "standard", "runs/figure4_e3t_ppo_rnn_64_16_standard_full_rnnfix_20260504-130101/20260504-130114_qrjr5dlz_counter_circuit_avs-full", E3T_DIAG, "ppo"),
    MethodSpec("fcp_old", "FCP old", "standard", "runs/figure4_fcp_cnn_64_16/FCP_ppo_cnn_standard_64_16_counter_circuit_8policies_x10_ow2rp4gr_20260506-172902", PPO_DIAG, "ppo"),
    MethodSpec("fcp_mm_frozen_ppo_partner", "FCP + MM frozen PPO partner", "standard", "runs/fcp_mep_frozen_mm_K10_64_16_10000000_20260516-090954/20260516-091009_ys6qo8lj_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("mep_ent010_standard", "MEP ent=0.1 standard", "standard", "runs/mep_realpop_mm1_mp1_K5_ent0.1_64_16_10000000_seed10_20260511-002112/20260511-002124_vnapmwlc_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("mep_ent005_standard", "MEP ent=0.05 standard", "standard", "runs/mep_realpop_mm1_mp1_K5_ent0.05_64_16_10000000_seed10_20260516-095429/20260516-095441_lvbnw6rf_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("mep_ent005_mp2_standard", "MEP ent=0.05 MP=2 standard", "standard", "runs/mep_realpop_mm1_mp2_K5_ent0.05_64_16_10000000_seed10_20260516-121256/20260516-121311_ihiy6517_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("trajedi_div010_standard", "TrajeDi div=0.1 standard", "standard", "runs/trajedi_unified_rerun_mm1_mp1_K5_div0.1_64_16_10000000_seed10_20260515-174148/20260515-174201_cglkefqe_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("trajedi_div005_standard", "TrajeDi div=0.05 standard", "standard", "runs/trajedi_realpop_mm1_mp1_K5_div0.05_64_16_10000000_seed10_20260516-151722/20260516-151735_1gd6vw78_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("gamma_mix25_ppo_cnn_standard_source", "GAMMA mix25 PPO-CNN standard source", "standard", "runs/gamma_mix025_ppo_standard_source_64_16_10000000_20260514-174603/20260514-174617_nc84elts_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("gamma_mix25_mappo_rnn_standard_source", "GAMMA mix25 MAPPO-RNN standard source", "standard", "runs/gamma_mix025_mappo_rnn_standard_source_64_16_10000000_20260515-005018/20260515-015313_yljod4y5_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("ppo_cnn_state_aug", "PPO CNN state-aug", "state_aug", "runs/figure4_state_aug/20260403-120112_f9p8h3cq_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("mappo_cnn_state_aug", "MAPPO CNN state-aug", "state_aug", "runs/figure4_mappo_cnn_64_16_rerun_mappo_cnn_state_aug_20260503-201254/20260503-202330_umwsoxcy_counter_circuit_avs-full", PPO_DIAG, "mappo"),
    MethodSpec("mappo_rnn_state_aug", "MAPPO RNN state-aug", "state_aug", "runs/figure4_mappo_rnn_64_16_unified_rerun_mappo_rnn_state_aug_20260515-unified-rerun/20260515-152618_9mkg137e_counter_circuit_avs-full", PPO_DIAG, "mappo"),
    MethodSpec("ppo_e3t_cnn_state_aug_predicted_ce", "PPO-E3T CNN state-aug predicted CE", "state_aug", "runs/figure4_ppo_e3t_state_aug_predicted_ce_full_20260428-ppo-e3t-state-aug/20260428-163847_50tx0n6v_counter_circuit_avs-full", E3T_DIAG, "ppo"),
    MethodSpec("ppo_e3t_cnn_state_aug_no_ce", "PPO-E3T CNN state-aug no CE", "state_aug", "runs/figure4_ppo_e3t_state_aug_no_ce_full_20260428-ppo-e3t-state-aug/20260428-200739_xsd6oyim_counter_circuit_avs-full", E3T_DIAG, "ppo"),
    MethodSpec("ppo_e3t_cnn_state_aug_constant_ce", "PPO-E3T CNN state-aug constant CE", "state_aug", "runs/figure4_ppo_e3t_state_aug_constant_ce_full_20260428-ppo-e3t-state-aug/20260428-182936_aza8mbkr_counter_circuit_avs-full", E3T_DIAG, "ppo"),
    MethodSpec("ppo_e3t_rnn_state_aug_fixed", "PPO-E3T RNN state-aug fixed", "state_aug", "runs/figure4_e3t_ppo_rnn_64_16_state_aug_full_rnnfix_20260504-130101/20260504-175745_vqj6owwb_counter_circuit_avs-full", E3T_DIAG, "ppo"),
    MethodSpec("mep_ent010_state_aug", "MEP ent=0.1 state-aug", "state_aug", "runs/population_state_aug_64_16_mep_state_aug_mm1_mp1_K5_ent0.1_64_16_10000000_sa10_stateaug-20260512-012924/20260512-062758_en6is7qd_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("mep_ent005_state_aug", "MEP ent=0.05 state-aug", "state_aug", "runs/population_state_aug_64_16_div005_mep_state_aug_mm1_mp1_K5_ent0.05_64_16_10000000_sa10_stateaug-div005-20260517-124718/20260517-124732_c77pgga1_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("trajedi_div010_state_aug", "TrajeDi div=0.1 state-aug", "state_aug", "runs/population_state_aug_64_16_trajedi_state_aug_mm1_mp1_K5_div0.1_64_16_10000000_sa10_stateaug-20260512-012924/20260512-012937_4r9b0ck9_counter_circuit_avs-full", PPO_DIAG, "ppo"),
    MethodSpec("trajedi_div005_state_aug", "TrajeDi div=0.05 state-aug", "state_aug", "runs/population_state_aug_64_16_div005_trajedi_state_aug_mm1_mp1_K5_div0.05_64_16_10000000_sa10_stateaug-div005-20260517-124718/20260517-154243_9i6qhkle_counter_circuit_avs-full", PPO_DIAG, "ppo"),
]


def method_dir(spec: MethodSpec) -> Path:
    if STORE_IN_RUN_DIR:
        return Path(spec.run_dir) / "diagnostics" / DIAG_TAG
    return OUTPUT_ROOT / spec.setting / spec.slug


def command_for(spec: MethodSpec) -> list[str]:
    return [
        PYTHON_BIN,
        spec.diag_py,
        "--single_run_dir", spec.run_dir,
        "--single_name", spec.slug,
        "--single_backend", spec.backend,
        "--single_eval_mode", spec.eval_mode,
        "--layout", LAYOUT,
        "--eval_seed", EVAL_SEED,
        "--num_eval_seeds", NUM_EVAL_SEEDS,
        "--num_diag_episodes", NUM_DIAG_EPISODES,
        "--top_k", TOP_K,
        "--compatibility_sample_limit", COMPATIBILITY_SAMPLE_LIMIT,
        "--max_pairs", MAX_PAIRS,
        "--output_dir", str(method_dir(spec)),
    ]


def mean_csv(path: Path, method: str, field: str, row_type: str | None = None) -> float:
    vals = []
    if not path.exists():
        return math.nan
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("method") != method:
                continue
            if row_type is not None and row.get("row_type") != row_type:
                continue
            value = row.get(field, "")
            if value == "":
                continue
            try:
                vals.append(float(value))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else math.nan


def read_reward(path: Path, method: str, field: str) -> float:
    if not path.exists():
        return math.nan
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("method") == method and row.get(field, ""):
                return float(row[field])
    return math.nan


def aggregate() -> None:
    rows = []
    for spec in METHODS:
        out = method_dir(spec)
        rows.append({
            "setting": spec.setting,
            "method": spec.display_name,
            "slug": spec.slug,
            "backend": spec.backend,
            "run_dir": spec.run_dir,
            "diag_dir": str(out),
            "SP": read_reward(out / "reward_summary.csv", spec.slug, "sp_mean"),
            "XP": read_reward(out / "reward_summary.csv", spec.slug, "xp_mean"),
            "OOS": mean_csv(out / "coverage_summary.csv", spec.slug, "xp_out_of_sp_support_rate", "cross_coverage"),
            "InBoth": mean_csv(out / "coverage_summary.csv", spec.slug, "in_both_support_rate", "cross_coverage"),
            "Agreement": mean_csv(out / "shared_state_mismatch.csv", spec.slug, "action_agreement"),
            "Policy_TV": mean_csv(out / "shared_state_mismatch.csv", spec.slug, "policy_tv"),
            "Joint_regret": mean_csv(out / "complementarity_summary.csv", spec.slug, "joint_regret"),
            "status": "available" if (out / "report.md").exists() else "missing",
        })
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = ["setting", "method", "slug", "SP", "XP", "OOS", "InBoth", "Agreement", "Policy_TV", "Joint_regret", "backend", "status", "diag_dir", "run_dir"]
    with (OUTPUT_ROOT / "summary_flat.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    def fmt(x):
        return "NA" if isinstance(x, float) and math.isnan(x) else (f"{x:.4f}" if isinstance(x, float) else str(x))
    with (OUTPUT_ROOT / "summary_flat.md").open("w") as f:
        f.write("# Current Table Independent Diagnostics Summary\n\n")
        f.write(f"- output_root: `{OUTPUT_ROOT}`\n")
        f.write(f"- store_in_run_dir: `{STORE_IN_RUN_DIR}`\n")
        f.write(f"- diag_tag: `{DIAG_TAG}`\n")
        f.write(f"- generated_at: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`\n")
        f.write(f"- num_eval_seeds: `{NUM_EVAL_SEEDS}`\n")
        f.write(f"- num_diag_episodes: `{NUM_DIAG_EPISODES}`\n")
        f.write(f"- max_pairs: `{MAX_PAIRS}`\n")
        f.write(f"- compatibility_sample_limit: `{COMPATIBILITY_SAMPLE_LIMIT}`\n\n")
        for setting in ["standard", "state_aug"]:
            f.write(f"## {setting}\n\n")
            f.write("| Method | SP | XP | OOS | In-both | Agreement | Policy TV | Joint regret | Status |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
            for row in rows:
                if row["setting"] != setting:
                    continue
                f.write("| " + " | ".join([
                    row["method"], fmt(row["SP"]), fmt(row["XP"]), fmt(row["OOS"]), fmt(row["InBoth"]),
                    fmt(row["Agreement"]), fmt(row["Policy_TV"]), fmt(row["Joint_regret"]), row["status"],
                ]) + " |\n")
            f.write("\n")


def write_manifest() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "README.md").open("w") as f:
        f.write("# Current Table Independent Diagnostics Index\n\n")
        f.write("Each method is diagnosed independently. No cross-algorithm pairing is used; XP is always within-method cross-play across seeds.\n\n")
        f.write("Diagnostics artifacts are stored inside each original training run directory under `diagnostics/<diag_tag>/`; this directory is only the central index and summary.\n\n")
        f.write(f"- started_at: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`\n")
        f.write(f"- parallel_methods: `{PARALLEL_METHODS}`\n")
        f.write(f"- layout: `{LAYOUT}`\n")
        f.write(f"- eval_seed: `{EVAL_SEED}`\n")
        f.write(f"- num_eval_seeds: `{NUM_EVAL_SEEDS}`\n")
        f.write(f"- num_diag_episodes: `{NUM_DIAG_EPISODES}`\n")
        f.write(f"- max_pairs: `{MAX_PAIRS}`\n")
        f.write(f"- compatibility_sample_limit: `{COMPATIBILITY_SAMPLE_LIMIT}`\n")
        f.write(f"- forward_chunk_size: `{os.environ.get('ZSC_FORWARD_CHUNK_SIZE', '32768')}`\n")
        f.write(f"- store_in_run_dir: `{STORE_IN_RUN_DIR}`\n")
        f.write(f"- diag_tag: `{DIAG_TAG}`\n\n")
        f.write("## Directory Layout\n\n")
        f.write("- original run: `<run_dir>/diagnostics/<diag_tag>/` contains `reward_summary.csv`, `coverage_summary.csv`, `shared_state_mismatch.csv`, `complementarity_summary.csv`, `report.md`, and `run.log`\n")
        f.write("- index root: this directory contains `methods.csv`, `status.csv`, and `summary_flat.csv/md` linking back to the original run diagnostics\n")
    with (OUTPUT_ROOT / "methods.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "slug", "display_name", "backend", "diag_py", "run_dir", "diag_dir"])
        writer.writeheader()
        for spec in METHODS:
            writer.writerow({
                "setting": spec.setting,
                "slug": spec.slug,
                "display_name": spec.display_name,
                "backend": spec.backend,
                "diag_py": spec.diag_py,
                "run_dir": spec.run_dir,
                "diag_dir": method_dir(spec),
            })


def main() -> None:
    write_manifest()
    status_path = OUTPUT_ROOT / "status.csv"
    if FORCE or not status_path.exists():
        status_path.write_text("timestamp,setting,slug,status,pid_or_returncode\n")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "experiments")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("ZSC_FORWARD_CHUNK_SIZE", "32768")
    env.setdefault("PYTHONUNBUFFERED", "1")
    pending = []
    for spec in METHODS:
        out = method_dir(spec)
        if out.joinpath("report.md").exists() and out.joinpath("coverage_summary.csv").exists() and not FORCE:
            with status_path.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},{spec.setting},{spec.slug},skip_existing,0\n")
            continue
        if not Path(spec.run_dir).exists():
            with status_path.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},{spec.setting},{spec.slug},missing_run_dir,0\n")
            continue
        pending.append(spec)
    running = []
    done = 0
    while pending or running:
        while pending and len(running) < PARALLEL_METHODS:
            spec = pending.pop(0)
            out = method_dir(spec); out.mkdir(parents=True, exist_ok=True)
            log = out / "run.log"
            with log.open("w") as handle:
                proc = subprocess.Popen(command_for(spec), stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            running.append((spec, proc))
            with status_path.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},{spec.setting},{spec.slug},started,{proc.pid}\n")
            print(f"[independent-diag] started {spec.setting}/{spec.slug} pid={proc.pid} ({len(running)}/{PARALLEL_METHODS})", flush=True)
        time.sleep(5)
        next_running = []
        for spec, proc in running:
            rc = proc.poll()
            if rc is None:
                next_running.append((spec, proc)); continue
            done += 1
            status = "ok" if rc == 0 else f"failed_{rc}"
            with status_path.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},{spec.setting},{spec.slug},{status},{rc}\n")
            print(f"[independent-diag] finished {spec.setting}/{spec.slug} status={status} ({done} done)", flush=True)
            aggregate()
        running = next_running
    aggregate()
    print(f"[independent-diag] done output={OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
