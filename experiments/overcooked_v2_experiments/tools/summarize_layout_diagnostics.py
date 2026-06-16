#!/usr/bin/env python3
"""Summarize per-method ZSC diagnostics for a layout mechanism pre-study."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _float(v, default=float("nan")) -> float:
    try:
        if v in (None, ""):
            return default
        return float(v)
    except Exception:
        return default


def _mean(rows: Iterable[dict], key: str) -> float:
    vals = [_float(r.get(key)) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _pearson(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return num / den_x / den_y if den_x and den_y else float("nan")


def _rank(vals: List[float]) -> List[float]:
    order = sorted((v, i) for i, v in enumerate(vals))
    ranks = [float("nan")] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1
        for _, idx in order[i : j + 1]:
            ranks[idx] = rank
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return float("nan")
    rx = _rank([x for x, _ in pairs])
    ry = _rank([y for _, y in pairs])
    return _pearson(rx, ry)


def _svg_scatter(rows: List[dict], x_key: str, y_key: str, out: Path, title: str) -> None:
    vals = [(r["method"], _float(r[x_key]), _float(r[y_key])) for r in rows]
    vals = [(m, x, y) for m, x, y in vals if not math.isnan(x) and not math.isnan(y)]
    width, height = 760, 520
    ml, mr, mt, mb = 72, 32, 58, 78
    x_min = min((x for _, x, _ in vals), default=0.0)
    x_max = max((x for _, x, _ in vals), default=1.0)
    y_min = min((y for _, _, y in vals), default=0.0)
    y_max = max((y for _, _, y in vals), default=1.0)
    if x_min == x_max:
        x_min -= 0.5; x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5; y_max += 0.5
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad
    def sx(x): return ml + (x - x_min) / (x_max - x_min) * (width - ml - mr)
    def sy(y): return height - mb - (y - y_min) / (y_max - y_min) * (height - mt - mb)
    font = "Avenir, Helvetica, Arial, sans-serif"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{ml}" y="34" font-family="{font}" font-size="24" font-weight="700" fill="#111827">{title}</text>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#98A2B3"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#98A2B3"/>',
        f'<text x="{width/2}" y="{height-24}" font-family="{font}" font-size="16" fill="#111827" text-anchor="middle">{x_key}</text>',
        f'<text x="22" y="{height/2}" font-family="{font}" font-size="16" fill="#111827" text-anchor="middle" transform="rotate(-90 22 {height/2})">{y_key}</text>',
    ]
    for method, x, y in vals:
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="6" fill="#2F80ED" opacity="0.88"/>')
        parts.append(f'<text x="{sx(x)+9:.2f}" y="{sy(y)-7:.2f}" font-family="{font}" font-size="12" fill="#344054">{method}</text>')
    parts.append('</svg>')
    out.write_text("\n".join(parts))


def _collect_method(diag_dir: Path) -> dict | None:
    reward_rows = _read_csv(diag_dir / "reward_summary.csv")
    if not reward_rows:
        return None
    reward = reward_rows[0]
    method = reward.get("method") or diag_dir.name
    cov = [r for r in _read_csv(diag_dir / "coverage_summary.csv") if r.get("row_type") == "cross_coverage"]
    mis = _read_csv(diag_dir / "shared_state_mismatch.csv")
    comp = _read_csv(diag_dir / "complementarity_summary.csv")
    oos = _mean(cov, "xp_out_of_sp_support_rate")
    agreement = _mean(mis, "action_agreement")
    tv = _mean(mis, "policy_tv")
    joint_regret = _mean(comp, "joint_regret")
    return {
        "method": method,
        "source_dir": str(diag_dir),
        "sp": _float(reward.get("sp_mean")),
        "xp": _float(reward.get("xp_mean")),
        "oos": oos,
        "coverage_alignment": (1.0 - oos) * agreement if not math.isnan(oos) and not math.isnan(agreement) else float("nan"),
        "agreement": agreement,
        "policy_tv": tv,
        "joint_regret": joint_regret,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    fields = ["method", "sp", "xp", "oos", "coverage_alignment", "agreement", "policy_tv", "joint_regret", "source_dir"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for child in sorted(args.diagnostics_root.iterdir() if args.diagnostics_root.exists() else []):
        if child.is_dir():
            row = _collect_method(child)
            if row:
                rows.append(row)
    rows.sort(key=lambda r: (math.isnan(r["xp"]), r["xp"]))
    _write_csv(args.output_dir / "baseline_results.csv", rows)
    _write_csv(args.output_dir / "diagnostics_results.csv", rows)

    metrics = ["oos", "coverage_alignment", "agreement", "policy_tv", "joint_regret", "sp"]
    lines = ["# Controlled regression / correlation summary", "", f"- diagnostics_root: `{args.diagnostics_root}`", f"- num_methods: `{len(rows)}`", "", "## Correlations with XP", "", "| metric | Pearson r | Spearman rho |", "|---|---:|---:|"]
    xp = [_float(r["xp"]) for r in rows]
    for m in metrics:
        vals = [_float(r[m]) for r in rows]
        lines.append(f"| {m} | {_pearson(vals, xp):.4f} | {_spearman(vals, xp):.4f} |")
    if np is not None and len(rows) >= 3:
        features = ["sp", "oos", "agreement", "policy_tv", "joint_regret"]
        valid = [r for r in rows if all(not math.isnan(_float(r[k])) for k in ["xp"] + features)]
        if len(valid) >= 3:
            x = np.array([[1.0] + [_float(r[k]) for k in features] for r in valid], dtype=float)
            y = np.array([_float(r["xp"]) for r in valid], dtype=float)
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
            lines += ["", "## Linear fit", "", f"- formula: `XP ~ SP + OOS + Agreement + Policy TV + Joint regret`", f"- n: `{len(valid)}`", f"- train_r2: `{r2:.4f}`", "", "| term | coefficient |", "|---|---:|", f"| intercept | {coef[0]:.4f} |"]
            for name, value in zip(features, coef[1:]):
                lines.append(f"| {name} | {value:.4f} |")
    (args.output_dir / "controlled_regression_summary.md").write_text("\n".join(lines))
    for metric in ["coverage_alignment", "agreement", "policy_tv", "joint_regret", "oos"]:
        _svg_scatter(rows, metric, "xp", args.output_dir / f"xp_vs_{metric}.svg", f"XP vs {metric}")
    print(args.output_dir / "controlled_regression_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
