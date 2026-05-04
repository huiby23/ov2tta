#!/usr/bin/env python3
"""Audit completed OV2 experiment results.

Only scans result files that already exist under `runs/**/reward_summary*.csv`.
Unfinished runs without reward summaries are intentionally excluded.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
WANDB_DIR = ROOT / "wandb"
OUT_DIR = ROOT / "runs_by_config"
OUT_ALL = OUT_DIR / "all_completed_runs_audit.tsv"
OUT_CROSS = OUT_DIR / "all_completed_cross_runs_audit.tsv"
OUT_UNMATCHED = OUT_DIR / "unmatched_completed_runs.tsv"

RUN_ID_RE = re.compile(r"\d{8}-\d{6}_([a-z0-9]{8})(?:_|$)")
WANDB_RE = re.compile(r"(?:offline-run-|run-)\d{8}_\d{6}-([a-z0-9]+)")

MODEL_KEYS = [
    "TYPE",
    "ARCH",
    "TOTAL_TIMESTEPS",
    "REW_SHAPING_HORIZON",
    "NUM_ENVS",
    "NUM_STEPS",
    "NUM_MINIBATCHES",
    "UPDATE_EPOCHS",
    "LR",
    "ENT_COEF",
    "CONTEXT_UPDATE_EPOCHS",
    "E3T_ENABLE_CE",
    "E3T_CONDITION_ACTOR",
    "E3T_ACTOR_CONDITION",
    "USE_HISTORY_CONTEXT",
    "USE_PARTNER_MIX",
    "PARTNER_MIX_EPS",
    "MOA_COEF",
    "MOA_TO_ACTOR",
]
TOP_KEYS = ["SEED", "NUM_SEEDS", "NUM_CHECKPOINTS", "NUM_ITERATIONS", "OPTIONAL_PREFIX"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.8g}"
    return str(v)


def parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"null", "none"}:
        return None
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    try:
        return float(raw) if any(c in raw for c in ".eE") else int(raw)
    except ValueError:
        return raw


def set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    cur = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        existing = cur.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cur[part] = existing
        cur = existing
    cur[parts[-1]] = value


def merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_debug_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    marker = "config: "
    idx = text.find(marker)
    if idx < 0:
        return {}
    literal = text[idx + len(marker):].splitlines()[0]
    try:
        cfg = ast.literal_eval(literal)
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def parse_metadata_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        meta = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return {}
    cfg: dict[str, Any] = {}
    for arg in meta.get("args", []) or []:
        if not isinstance(arg, str) or "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        set_nested(cfg, key.lstrip("+"), parse_scalar(val))
    if meta.get("program"):
        cfg["program"] = meta["program"]
    return cfg


def parse_simple_config_yaml(path: Path) -> dict[str, Any]:
    # Minimal parser for the flat-ish wandb config.yaml. Debug logs are preferred.
    if not path.exists():
        return {}
    cfg: dict[str, Any] = {}
    current_top = ""
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, val = line.strip().split(":", 1)
        key = key.strip().strip('"').strip("'")
        val = val.strip()
        if indent == 0 and not val:
            current_top = key
            cfg.setdefault(key, {})
        elif current_top and val:
            cfg.setdefault(current_top, {})[key] = parse_scalar(val)
        elif val:
            cfg[key] = parse_scalar(val)
    return cfg


def build_wandb_index() -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    if not WANDB_DIR.exists():
        return idx
    for run_dir in WANDB_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        m = WANDB_RE.search(run_dir.name)
        if not m:
            continue
        run_id = m.group(1)
        cfg: dict[str, Any] = {}
        cfg = merge(cfg, parse_simple_config_yaml(run_dir / "files" / "config.yaml"))
        cfg = merge(cfg, parse_debug_config(run_dir / "logs" / "debug.log"))
        cfg = merge(cfg, parse_metadata_args(run_dir / "files" / "wandb-metadata.json"))
        cfg["wandb_dir"] = rel(run_dir)
        idx[run_id] = cfg
    return idx


def infer_run_id(path: Path) -> str:
    for part in path.parts:
        m = RUN_ID_RE.search(part)
        if m:
            return m.group(1)
    return ""


def get_cfg(cfg: dict[str, Any], *keys: str) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return ""
        cur = cur[key]
    return cur


def infer_method(path: Path, cfg: dict[str, Any]) -> str:
    path_s = rel(path).lower()
    program = str(cfg.get("program", "")).lower()
    prefix = str(cfg.get("OPTIONAL_PREFIX", "")).lower()
    s = " ".join([path_s, program, prefix])

    # Prefer the actual training module when wandb metadata is available. Batch
    # names such as "e3tfirst" should not change the method label.
    if "/ppo_e3t/" in program or "ppo_e3t" in prefix or "ppo_e3t" in path_s:
        return "ppo_e3t"
    if "/ttappo/" in program or "ttappo" in prefix or "ttappo" in path_s:
        return "ttappo"
    if "/mappo/" in program or re.search(r"(^|[_/-])mappo([_/-]|$)", s):
        return "mappo"
    if "/ppo/" in program or "figure4_standard" in path_s or "figure4_state_aug" in path_s or re.search(r"(^|[_/-])ppo(_(cnn|rnn|state|standard)|[_/-]|$)", prefix):
        return "ppo"
    if "/e3t/" in program or re.search(r"(^|[_/-])e3t([_/-]|$)", s):
        return "e3t"
    return "unknown"


def infer_layout(path: Path, cfg: dict[str, Any]) -> str:
    layout = get_cfg(cfg, "env", "ENV_KWARGS", "layout")
    if layout:
        return str(layout)
    s = rel(path).lower()
    for name in ["counter_circuit", "grounded_coord_simple", "cramped_room"]:
        if name in s:
            return name
    return "unknown"


def infer_state_aug(path: Path, cfg: dict[str, Any]) -> str:
    s = rel(path).lower()
    # Check explicit no-state markers first: "no_state_aug" contains "state_aug".
    if "no_state" in s or "standard" in s or "ppo_e3t_official" in s:
        return "no_state"
    num_iter = cfg.get("NUM_ITERATIONS", "")
    if fmt(num_iter) not in {"", "0"}:
        return "state_aug"
    if "state_aug" in s or "sa-" in s:
        return "state_aug"
    return "unknown"


def compute_stats(path: Path) -> tuple[str, str, int, int, str]:
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            if "total_reward" not in fields:
                return "", "", 0, 0, "unsupported_schema"
            sp = xp = 0.0
            nsp = nxp = 0
            for row in reader:
                try:
                    reward = float(row.get("total_reward", ""))
                except ValueError:
                    continue
                is_sp = False
                label = row.get("policy_labels", "")
                m = re.search(r"[-_](\d+)_(\d+)", label)
                if m:
                    is_sp = m.group(1) == m.group(2)
                elif row.get("run") == "sp":
                    is_sp = True
                if is_sp:
                    sp += reward
                    nsp += 1
                else:
                    xp += reward
                    nxp += 1
    except Exception as exc:
        return "", "", 0, 0, f"error:{type(exc).__name__}"
    return (
        f"{sp / nsp:.6g}" if nsp else "",
        f"{xp / nxp:.6g}" if nxp else "",
        nsp,
        nxp,
        "ok",
    )


def row_for(path: Path, wandb_idx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_id = infer_run_id(path.parent)
    cfg = dict(wandb_idx.get(run_id, {}))
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    sp, xp, nsp, nxp, status = compute_stats(path)
    row: dict[str, Any] = {
        "result_file": rel(path),
        "run_dir": rel(path.parent),
        "run_id": run_id,
        "wandb_matched": "yes" if run_id in wandb_idx else "no",
        "wandb_dir": cfg.get("wandb_dir", ""),
        "method": infer_method(path, cfg),
        "layout": infer_layout(path, cfg),
        "state_aug": infer_state_aug(path, cfg),
        "reward_status": status,
        "SP": sp,
        "XP": xp,
        "nSP": nsp,
        "nXP": nxp,
    }
    for k in MODEL_KEYS:
        row[k] = fmt(model.get(k, ""))
    for k in TOP_KEYS:
        row[k] = fmt(cfg.get(k, ""))
    return row


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    wandb_idx = build_wandb_index()
    result_files = sorted(RUNS_DIR.rglob("reward_summary*.csv"))
    rows = [row_for(p, wandb_idx) for p in result_files]
    rows = [r for r in rows if r["reward_status"] == "ok" and (int(r["nSP"]) + int(r["nXP"]) > 0)]
    cross_rows = [r for r in rows if Path(r["result_file"]).name.startswith("reward_summary_cross")]
    unmatched = [r for r in rows if r["wandb_matched"] == "no"]

    fields = [
        "result_file", "run_dir", "run_id", "wandb_matched", "wandb_dir",
        "method", "layout", "state_aug", "reward_status", "SP", "XP", "nSP", "nXP",
        *MODEL_KEYS, *TOP_KEYS,
    ]
    write_tsv(OUT_ALL, rows, fields)
    write_tsv(OUT_CROSS, cross_rows, fields)
    write_tsv(OUT_UNMATCHED, unmatched, fields)

    print(f"completed_reward_files={len(rows)}")
    print(f"completed_cross_reward_files={len(cross_rows)}")
    print(f"wandb_matched={sum(r['wandb_matched'] == 'yes' for r in rows)}")
    print(f"unmatched={len(unmatched)}")
    print(f"out_all={rel(OUT_ALL)}")
    print(f"out_cross={rel(OUT_CROSS)}")
    print(f"out_unmatched={rel(OUT_UNMATCHED)}")


if __name__ == "__main__":
    main()
