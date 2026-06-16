#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SUMMARY = Path("reports/current_complete_results_with_diagnostics_20260523.csv")
DEFAULT_OUT = Path("reports/paper_analysis_20260523")
DEFAULT_INDEX = Path("runs/current_table_independent_diagnostics_index_20260523_20260523-1053/summary_flat.csv")

LABEL_MAP = {
    "PPO CNN standard": "PPO-CNN std",
    "PPO CNN state-aug": "PPO-CNN aug",
    "MAPPO CNN standard": "MAPPO-CNN std",
    "MAPPO CNN state-aug": "MAPPO-CNN aug",
    "MAPPO RNN standard": "MAPPO-RNN std",
    "MAPPO RNN state-aug": "MAPPO-RNN aug",
    "PPO-E3T CNN no-state predicted CE": "E3T pred std",
    "PPO-E3T CNN no-state no CE": "E3T noCE std",
    "PPO-E3T CNN no-state constant CE": "E3T const std",
    "PPO-E3T CNN state-aug predicted CE": "E3T pred aug",
    "PPO-E3T CNN state-aug no CE": "E3T noCE aug",
    "PPO-E3T CNN state-aug constant CE": "E3T const aug",
    "PPO-E3T RNN standard fixed": "E3T-RNN std",
    "PPO-E3T RNN state-aug fixed": "E3T-RNN aug",
    "FCP old": "FCP old",
    "FCP + MM frozen PPO partner": "FCP+MM",
    "MEP ent=0.1 standard": "MEP .10 std",
    "MEP ent=0.05 standard": "MEP .05 std",
    "MEP ent=0.05 MP=2 standard": "MEP .05 MP2",
    "MEP ent=0.1 state-aug": "MEP .10 aug",
    "MEP ent=0.05 state-aug": "MEP .05 aug",
    "TrajeDi div=0.1 standard": "TrajeDi .10 std",
    "TrajeDi div=0.05 standard": "TrajeDi .05 std",
    "TrajeDi div=0.1 state-aug": "TrajeDi .10 aug",
    "TrajeDi div=0.05 state-aug": "TrajeDi .05 aug",
    "GAMMA mix25 PPO-CNN standard source": "GAMMA PPO src",
    "GAMMA mix25 MAPPO-RNN standard source": "GAMMA MAPPO src",
}

PAIR_MAP = {
    "PPO CNN": ("PPO CNN standard", "PPO CNN state-aug"),
    "MAPPO CNN": ("MAPPO CNN standard", "MAPPO CNN state-aug"),
    "MAPPO RNN": ("MAPPO RNN standard", "MAPPO RNN state-aug"),
    "E3T CNN predicted CE": ("PPO-E3T CNN no-state predicted CE", "PPO-E3T CNN state-aug predicted CE"),
    "E3T CNN no CE": ("PPO-E3T CNN no-state no CE", "PPO-E3T CNN state-aug no CE"),
    "E3T CNN constant CE": ("PPO-E3T CNN no-state constant CE", "PPO-E3T CNN state-aug constant CE"),
    "E3T RNN fixed": ("PPO-E3T RNN standard fixed", "PPO-E3T RNN state-aug fixed"),
    "MEP ent=0.1": ("MEP ent=0.1 standard", "MEP ent=0.1 state-aug"),
    "MEP ent=0.05": ("MEP ent=0.05 standard", "MEP ent=0.05 state-aug"),
    "TrajeDi div=0.1": ("TrajeDi div=0.1 standard", "TrajeDi div=0.1 state-aug"),
    "TrajeDi div=0.05": ("TrajeDi div=0.05 standard", "TrajeDi div=0.05 state-aug"),
}

COLORS = {"no": "#2563eb", "yes": "#dc2626"}
MARKERS = {"no": "o", "yes": "^"}


def safe_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def short_name(method: str) -> str:
    return LABEL_MAP.get(method, method.replace(" standard", " std").replace(" state-aug", " aug"))


def ensure_dirs(out_dir: Path) -> None:
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def bootstrap_corr_ci(data: pd.DataFrame, xcol: str, ycol: str, n_boot: int, seed: int):
    d = data[[xcol, ycol]].dropna()
    if len(d) < 5:
        return (math.nan, math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    pear = []
    spear = []
    vals = d.to_numpy(float)
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), len(vals))
        sample = vals[idx]
        pear.append(pearson(sample[:, 0], sample[:, 1]))
        spear.append(spearman(sample[:, 0], sample[:, 1]))
    pear = np.asarray([v for v in pear if np.isfinite(v)])
    spear = np.asarray([v for v in spear if np.isfinite(v)])
    return (
        float(np.percentile(pear, 2.5)) if len(pear) else math.nan,
        float(np.percentile(pear, 97.5)) if len(pear) else math.nan,
        float(np.percentile(spear, 2.5)) if len(spear) else math.nan,
        float(np.percentile(spear, 97.5)) if len(spear) else math.nan,
    )


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return s * 0.0
    return (s - s.mean()) / std


def ridge_fit_predict(data: pd.DataFrame, features: list[str], ycol: str = "XP", alpha: float = 1.0):
    d = data[features + [ycol]].dropna().copy()
    x = d[features].to_numpy(float)
    y = d[ycol].to_numpy(float)
    mu = x.mean(axis=0)
    sig = x.std(axis=0)
    sig[sig == 0] = 1.0
    xs = (x - mu) / sig
    X = np.c_[np.ones(xs.shape[0]), xs]
    reg = np.eye(X.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    pred = X @ beta
    return d, beta, pred, mu, sig


def loo_predict(data: pd.DataFrame, features: list[str], ycol: str = "XP", alpha: float = 1.0):
    d = data[features + [ycol]].dropna().copy()
    x = d[features].to_numpy(float)
    y = d[ycol].to_numpy(float)
    pred = np.zeros_like(y)
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        xtr = x[train]
        ytr = y[train]
        mu = xtr.mean(axis=0)
        sig = xtr.std(axis=0)
        sig[sig == 0] = 1.0
        Xtr = np.c_[np.ones(xtr.shape[0]), (xtr - mu) / sig]
        Xte = np.c_[np.ones(1), (x[[i]] - mu) / sig]
        reg = np.eye(Xtr.shape[1]) * alpha
        reg[0, 0] = 0.0
        beta = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)
        pred[i] = (Xte @ beta)[0]
    return d, pred


def r2(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum((y - y.mean()) ** 2)
    if denom == 0:
        return math.nan
    return float(1 - np.sum((y - pred) ** 2) / denom)


def obs_view_from_run_dir(run_dir: str) -> str:
    text = str(run_dir or "")
    m = re.search(r"avs-([^/_]+)", text)
    if m:
        return m.group(1)
    if "full_obs" in text or "no_state_aug_full_obs" in text:
        return "full"
    return "unknown"


def load_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = safe_num(df, ["SP", "XP", "OOS", "InBoth", "Agreement", "Policy_TV", "Joint_regret"])
    if "diagnostics" not in df.columns:
        if "status" in df.columns:
            df["diagnostics"] = df["status"].map(lambda x: "available" if x == "available" else str(x))
        else:
            df["diagnostics"] = "unknown"
    if "state_aug" not in df.columns:
        df["state_aug"] = df["setting"].astype(str).str.replace("_", "-").map(lambda x: "yes" if x == "state-aug" else "no")
    if "obs_view" not in df.columns:
        if "run_dir" in df.columns:
            df["obs_view"] = df["run_dir"].map(obs_view_from_run_dir)
        else:
            df["obs_view"] = "unknown"
    if "diag_dir" not in df.columns:
        df["diag_dir"] = ""
    # When loading the central diagnostics index, failed diagnostics rows may not
    # contain reward summaries. Preserve known SP/XP from the current result table.
    fallback_path = DEFAULT_SUMMARY
    if summary_csv.resolve() != fallback_path.resolve() and fallback_path.exists():
        fallback = pd.read_csv(fallback_path)
        fallback = safe_num(fallback, ["SP", "XP"])
        reward_by_method = fallback.set_index("method")[["SP", "XP"]].to_dict("index")
        for idx, row in df.iterrows():
            reward = reward_by_method.get(row["method"])
            if not reward:
                continue
            for col in ["SP", "XP"]:
                if pd.isna(row.get(col)) and pd.notna(reward.get(col)):
                    df.at[idx, col] = reward[col]
    df["Coverage"] = 1.0 - df["OOS"]
    df["CoverageAgreement"] = df["Coverage"] * df["Agreement"]
    df["NegPolicyTV"] = -df["Policy_TV"]
    df["NegJointRegret"] = -df["Joint_regret"]
    df["short"] = df["method"].map(short_name)
    return df


def write_table1(df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["method", "state_aug", "obs_view", "SP", "XP", "OOS", "Agreement", "Policy_TV", "Joint_regret", "diagnostics"]
    table = df[cols].sort_values(["state_aug", "method"]).copy()
    table.to_csv(out_dir / "tables" / "table1_complete_results.csv", index=False)
    lines = [
        "# Table 1: Complete Results With Diagnostics",
        "",
        "`state_aug` is rollout state-sampling via `initial_state_buffer`; `obs_view` is observation scope and is not state augmentation.",
        "",
        "| Method | State aug | Obs view | SP | XP | OOS | Agreement | Policy TV | Joint regret | Diag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in table.iterrows():
        def fmt(v, nd=2):
            return "NA" if pd.isna(v) else f"{float(v):.{nd}f}"
        def fmt4(v):
            return "NA" if pd.isna(v) else f"{float(v):.4f}"
        lines.append(
            f"| {r['method']} | {r['state_aug']} | {r['obs_view']} | {fmt(r['SP'])} | {fmt(r['XP'])} | {fmt4(r['OOS'])} | {fmt4(r['Agreement'])} | {fmt4(r['Policy_TV'])} | {fmt4(r['Joint_regret'])} | {r['diagnostics']} |"
        )
    (out_dir / "tables" / "table1_complete_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scatter_and_correlations(df: pd.DataFrame, out_dir: Path, n_boot: int) -> pd.DataFrame:
    data = df[(df["diagnostics"] == "available") & df["XP"].notna()].copy()
    no_fcp = data[~data["method"].eq("FCP old")].copy()
    metrics = [
        ("OOS", "Out-of-support rate", "lower is better"),
        ("Agreement", "Action agreement", "higher is better"),
        ("Policy_TV", "Policy TV", "lower is better"),
        ("Joint_regret", "Joint regret", "lower is better"),
        ("CoverageAgreement", "(1-OOS) * Agreement", "higher is better"),
    ]
    rows = []
    for subset_name, subset in [("all", data), ("no_fcp_old", no_fcp), ("standard_no_fcp_old", no_fcp[no_fcp["state_aug"] == "no"] )]:
        for col, label, _ in metrics:
            d = subset[[col, "XP"]].dropna()
            if len(d) < 3:
                continue
            pl, ph, sl, sh = bootstrap_corr_ci(d, col, "XP", n_boot, 42)
            rows.append({
                "subset": subset_name,
                "metric": col,
                "label": label,
                "n": len(d),
                "pearson": pearson(d[col], d["XP"]),
                "spearman": spearman(d[col], d["XP"]),
                "pearson_ci_low": pl,
                "pearson_ci_high": ph,
                "spearman_ci_low": sl,
                "spearman_ci_high": sh,
            })
    corr = pd.DataFrame(rows)
    corr.to_csv(out_dir / "tables" / "figure1_correlations_with_bootstrap.csv", index=False)

    # Figure 1: main composite score.
    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=180)
    for state_aug, sub in no_fcp.groupby("state_aug"):
        ax.scatter(sub["CoverageAgreement"], sub["XP"], s=70, c=COLORS.get(state_aug, "#111827"), marker=MARKERS.get(state_aug, "o"), label=f"state_aug={state_aug}", alpha=0.9, edgecolor="white", linewidth=0.8)
    d = no_fcp[["CoverageAgreement", "XP"]].dropna()
    if len(d) >= 4:
        xs = d["CoverageAgreement"].to_numpy(float)
        ys = d["XP"].to_numpy(float)
        coef = np.polyfit(xs, ys, deg=1)
        grid = np.linspace(xs.min(), xs.max(), 160)
        ax.plot(grid, np.polyval(coef, grid), color="#111827", ls="--", lw=2)
    for _, r in no_fcp.iterrows():
        if pd.notna(r["CoverageAgreement"]) and pd.notna(r["XP"]):
            ax.annotate(r["short"], (r["CoverageAgreement"], r["XP"]), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#374151")
    p = corr[(corr["subset"] == "no_fcp_old") & (corr["metric"] == "CoverageAgreement")].iloc[0]
    ax.set_xlabel("Coverage-alignment score: (1 - OOS) * Agreement")
    ax.set_ylabel("XP")
    ax.set_title(f"Figure 1. XP vs coverage-alignment score\nPearson r={p.pearson:+.2f} [{p.pearson_ci_low:+.2f},{p.pearson_ci_high:+.2f}], Spearman={p.spearman:+.2f}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"figure1_xp_vs_coverage_agreement.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Appendix scatter grid.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    for ax, (col, label, hint) in zip(axes.ravel(), metrics[:4]):
        for state_aug, sub in no_fcp.groupby("state_aug"):
            ax.scatter(sub[col], sub["XP"], s=55, c=COLORS.get(state_aug, "#111827"), marker=MARKERS.get(state_aug, "o"), label=f"state_aug={state_aug}", alpha=0.88, edgecolor="white", linewidth=0.7)
        d = no_fcp[[col, "XP"]].dropna()
        if len(d) >= 4 and d[col].nunique() >= 4:
            xs = d[col].to_numpy(float); ys = d["XP"].to_numpy(float)
            coef = np.polyfit(xs, ys, deg=1)
            grid = np.linspace(xs.min(), xs.max(), 160)
            ax.plot(grid, np.polyval(coef, grid), color="#111827", ls="--", lw=1.8)
        ax.set_xlabel(f"{label} ({hint})")
        ax.set_ylabel("XP")
        ax.set_title(f"r={pearson(d[col], d['XP']):+.2f}, rho={spearman(d[col], d['XP']):+.2f}")
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Appendix: XP vs individual diagnostics", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"appendix_xp_vs_individual_diagnostics.{ext}", bbox_inches="tight")
    plt.close(fig)
    return corr


def state_aug_attribution(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    by_method = {r["method"]: r for _, r in df.iterrows()}
    for family, (std_name, aug_name) in PAIR_MAP.items():
        if std_name not in by_method or aug_name not in by_method:
            continue
        s = by_method[std_name]
        a = by_method[aug_name]
        rows.append({
            "family": family,
            "standard_method": std_name,
            "state_aug_method": aug_name,
            "SP_standard": s.get("SP"),
            "SP_state_aug": a.get("SP"),
            "XP_standard": s.get("XP"),
            "XP_state_aug": a.get("XP"),
            "delta_XP": a.get("XP") - s.get("XP") if pd.notna(a.get("XP")) and pd.notna(s.get("XP")) else math.nan,
            "OOS_standard": s.get("OOS"),
            "OOS_state_aug": a.get("OOS"),
            "delta_OOS": a.get("OOS") - s.get("OOS") if pd.notna(a.get("OOS")) and pd.notna(s.get("OOS")) else math.nan,
            "Agreement_standard": s.get("Agreement"),
            "Agreement_state_aug": a.get("Agreement"),
            "delta_Agreement": a.get("Agreement") - s.get("Agreement") if pd.notna(a.get("Agreement")) and pd.notna(s.get("Agreement")) else math.nan,
            "Policy_TV_standard": s.get("Policy_TV"),
            "Policy_TV_state_aug": a.get("Policy_TV"),
            "delta_Policy_TV": a.get("Policy_TV") - s.get("Policy_TV") if pd.notna(a.get("Policy_TV")) and pd.notna(s.get("Policy_TV")) else math.nan,
            "Joint_regret_standard": s.get("Joint_regret"),
            "Joint_regret_state_aug": a.get("Joint_regret"),
            "delta_Joint_regret": a.get("Joint_regret") - s.get("Joint_regret") if pd.notna(a.get("Joint_regret")) and pd.notna(s.get("Joint_regret")) else math.nan,
        })
    delta = pd.DataFrame(rows)
    delta.to_csv(out_dir / "tables" / "figure2_state_aug_attribution.csv", index=False)

    plot_data = delta.dropna(subset=["OOS_standard", "OOS_state_aug", "XP_standard", "XP_state_aug"])
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=180)
    for _, r in plot_data.iterrows():
        ax.annotate("", xy=(r["OOS_state_aug"], r["XP_state_aug"]), xytext=(r["OOS_standard"], r["XP_standard"]), arrowprops=dict(arrowstyle="->", color="#4b5563", lw=1.8, alpha=0.8))
        ax.scatter([r["OOS_standard"]], [r["XP_standard"]], c="#2563eb", s=58, edgecolor="white", linewidth=0.8)
        ax.scatter([r["OOS_state_aug"]], [r["XP_state_aug"]], c="#dc2626", s=58, marker="^", edgecolor="white", linewidth=0.8)
        ax.annotate(r["family"], (r["OOS_state_aug"], r["XP_state_aug"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("OOS (lower is better)")
    ax.set_ylabel("XP")
    ax.set_title("Figure 2. standard -> state-aug attribution arrows")
    ax.grid(True, alpha=0.25)
    ax.scatter([], [], c="#2563eb", label="standard")
    ax.scatter([], [], c="#dc2626", marker="^", label="state-aug")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"figure2_state_aug_arrows_oos_xp.{ext}", bbox_inches="tight")
    plt.close(fig)
    return delta


def regression_analysis(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    data = df[(df["diagnostics"] == "available") & (~df["method"].eq("FCP old"))].copy()
    candidates = [
        ("coverage_agreement", ["CoverageAgreement"], 0.3),
        ("sp_only", ["SP"], 0.3),
        ("sp_coverage_policytv", ["SP", "Coverage", "NegPolicyTV"], 1.0),
        ("sp_coverage_agreement_policytv", ["SP", "Coverage", "Agreement", "NegPolicyTV"], 3.0),
        ("diagnostics_only", ["Coverage", "Agreement", "NegPolicyTV", "NegJointRegret"], 3.0),
    ]
    rows = []
    predictions = []
    for name, feats, alpha in candidates:
        d, beta, pred, _, _ = ridge_fit_predict(data, feats, alpha=alpha)
        loo_d, loo = loo_predict(data, feats, alpha=alpha)
        assert list(d.index) == list(loo_d.index)
        y = d["XP"].to_numpy(float)
        row = {
            "model": name,
            "features": "+".join(feats),
            "alpha": alpha,
            "n": len(d),
            "in_r2": r2(y, pred),
            "in_pearson": pearson(y, pred),
            "in_spearman": spearman(y, pred),
            "loo_r2": r2(y, loo),
            "loo_pearson": pearson(y, loo),
            "loo_spearman": spearman(y, loo),
            "loo_rmse": float(np.sqrt(np.mean((loo - y) ** 2))),
            "intercept": beta[0],
        }
        for i, f in enumerate(feats):
            row[f"coef_z_{f}"] = beta[i + 1]
        rows.append(row)
        tmp = data.loc[d.index, ["method", "state_aug", "SP", "XP", "Coverage", "Agreement", "Policy_TV", "Joint_regret"]].copy()
        tmp["model"] = name
        tmp["pred"] = pred
        tmp["loo_pred"] = loo
        tmp["residual"] = tmp["XP"] - tmp["pred"]
        predictions.append(tmp)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "tables" / "figure3_controlled_regression_models.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out_dir / "tables" / "figure3_controlled_regression_predictions.csv", index=False)

    best_name = "sp_coverage_policytv"
    d, beta, pred, _, _ = ridge_fit_predict(data, ["SP", "Coverage", "NegPolicyTV"], alpha=1.0)
    _, loo = loo_predict(data, ["SP", "Coverage", "NegPolicyTV"], alpha=1.0)
    plot_d = data.loc[d.index].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=180)
    for ax, pcol, title in [(axes[0], pred, "In-sample fit"), (axes[1], loo, "Leave-one-out prediction")]:
        for state_aug, sub in plot_d.groupby("state_aug"):
            idx = [plot_d.index.get_loc(i) for i in sub.index]
            ax.scatter(np.asarray(pcol)[idx], sub["XP"], s=68, c=COLORS.get(state_aug, "#111827"), marker=MARKERS.get(state_aug, "o"), label=f"state_aug={state_aug}", alpha=0.9, edgecolor="white", linewidth=0.8)
        mn = min(np.nanmin(pcol), plot_d["XP"].min())
        mx = max(np.nanmax(pcol), plot_d["XP"].max())
        ax.plot([mn, mx], [mn, mx], color="#111827", ls="--", lw=1.5)
        for idx, r in plot_d.iterrows():
            pos = plot_d.index.get_loc(idx)
            ax.annotate(short_name(r["method"]), (np.asarray(pcol)[pos], r["XP"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Predicted XP")
        ax.set_ylabel("Actual XP")
        ax.set_title(f"{title}: r={pearson(plot_d['XP'], pcol):+.2f}, R2={r2(plot_d['XP'], pcol):+.2f}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 3. Controlled XP regression: SP + Coverage + low Policy TV", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"figure3_controlled_regression_actual_vs_predicted.{ext}", bbox_inches="tight")
    plt.close(fig)
    return res


def load_pair_metrics(row: pd.Series):
    diag_dir = Path(str(row.get("diag_dir", "")))
    if not diag_dir.exists():
        return None
    cov_path = diag_dir / "coverage_summary.csv"
    mis_path = diag_dir / "shared_state_mismatch.csv"
    comp_path = diag_dir / "complementarity_summary.csv"
    if not cov_path.exists() or not mis_path.exists() or not comp_path.exists():
        return None
    slug = row.get("slug") if "slug" in row else None
    method = row.get("method")
    cov = pd.read_csv(cov_path)
    mis = pd.read_csv(mis_path)
    comp = pd.read_csv(comp_path)
    if slug and "method" in cov.columns:
        cov = cov[cov["method"] == slug]
        mis = mis[mis["method"] == slug]
        comp = comp[comp["method"] == slug]
    cov = cov[cov["row_type"] == "cross_coverage"].copy()
    cov = safe_num(cov, ["mean_reward", "xp_out_of_sp_support_rate", "in_both_support_rate", "support_overlap", "topk_bottleneck_state_mass"])
    mis = safe_num(mis, ["action_agreement", "policy_tv", "symmetric_kl", "value_shift", "shared_state_rate"])
    comp = safe_num(comp, ["joint_optimal_rate", "joint_regret", "unilateral_regret"])
    merged = cov.merge(mis.drop(columns=["method"], errors="ignore"), on=["run_i", "run_j"], how="left")
    merged = merged.merge(comp.drop(columns=["method"], errors="ignore"), on=["run_i", "run_j"], how="left")
    merged["display_method"] = method
    merged["state_aug"] = row.get("state_aug", "")
    return merged


def oracle_selector_analysis(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    index_cols = ["method", "state_aug", "SP", "XP", "OOS", "Agreement", "Policy_TV", "Joint_regret", "diagnostics", "diag_dir"]
    if "slug" in df.columns:
        index_cols.append("slug")
    rows = []
    pair_frames = []
    for _, row in df[df["diagnostics"] == "available"].iterrows():
        pairs = load_pair_metrics(row)
        if pairs is None or pairs.empty:
            continue
        pair_frames.append(pairs)
        method = row["method"]
        nonnan = pairs.dropna(subset=["mean_reward"])
        if nonnan.empty:
            continue
        fixed = nonnan[nonnan["run_i"] == "run_0"]
        fixed_mean = fixed["mean_reward"].mean() if not fixed.empty else math.nan
        random_mean = nonnan["mean_reward"].mean()
        best_xp = nonnan.loc[nonnan.groupby("run_j")["mean_reward"].idxmax()]
        low_tv = nonnan.dropna(subset=["policy_tv"])
        low_tv = low_tv.loc[low_tv.groupby("run_j")["policy_tv"].idxmin()] if not low_tv.empty else pd.DataFrame()
        high_agreement = nonnan.dropna(subset=["action_agreement"])
        high_agreement = high_agreement.loc[high_agreement.groupby("run_j")["action_agreement"].idxmax()] if not high_agreement.empty else pd.DataFrame()
        comp = nonnan.copy()
        comp["selector_score"] = (1 - comp["xp_out_of_sp_support_rate"]) * comp["action_agreement"]
        composite = comp.dropna(subset=["selector_score"])
        composite = composite.loc[composite.groupby("run_j")["selector_score"].idxmax()] if not composite.empty else pd.DataFrame()
        rows.extend([
            {"method": method, "state_aug": row["state_aug"], "selector": "fixed_ego_run0", "mean_xp": fixed_mean, "std": fixed["mean_reward"].std(ddof=0) if not fixed.empty else math.nan, "n_partners": fixed["run_j"].nunique() if not fixed.empty else 0},
            {"method": method, "state_aug": row["state_aug"], "selector": "random_ego", "mean_xp": random_mean, "std": nonnan["mean_reward"].std(ddof=0), "n_partners": nonnan["run_j"].nunique()},
            {"method": method, "state_aug": row["state_aug"], "selector": "oracle_best_xp", "mean_xp": best_xp["mean_reward"].mean(), "std": best_xp["mean_reward"].std(ddof=0), "n_partners": best_xp["run_j"].nunique()},
            {"method": method, "state_aug": row["state_aug"], "selector": "oracle_low_policy_tv", "mean_xp": low_tv["mean_reward"].mean() if not low_tv.empty else math.nan, "std": low_tv["mean_reward"].std(ddof=0) if not low_tv.empty else math.nan, "n_partners": low_tv["run_j"].nunique() if not low_tv.empty else 0},
            {"method": method, "state_aug": row["state_aug"], "selector": "oracle_high_agreement", "mean_xp": high_agreement["mean_reward"].mean() if not high_agreement.empty else math.nan, "std": high_agreement["mean_reward"].std(ddof=0) if not high_agreement.empty else math.nan, "n_partners": high_agreement["run_j"].nunique() if not high_agreement.empty else 0},
            {"method": method, "state_aug": row["state_aug"], "selector": "oracle_high_coverage_agreement", "mean_xp": composite["mean_reward"].mean() if not composite.empty else math.nan, "std": composite["mean_reward"].std(ddof=0) if not composite.empty else math.nan, "n_partners": composite["run_j"].nunique() if not composite.empty else 0},
        ])
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "tables" / "figure4_oracle_selector_summary.csv", index=False)
    if pair_frames:
        pd.concat(pair_frames, ignore_index=True).to_csv(out_dir / "tables" / "pair_level_diagnostics_all_available.csv", index=False)

    # Plot all methods excluding FCP old, sorted by random XP.
    plot = summary[(summary["selector"].isin(["random_ego", "oracle_best_xp", "oracle_low_policy_tv", "oracle_high_agreement"])) & (~summary["method"].eq("FCP old"))].copy()
    pivot = plot.pivot_table(index="method", columns="selector", values="mean_xp", aggfunc="first")
    pivot = pivot.dropna(subset=["random_ego"]).sort_values("random_ego")
    fig, ax = plt.subplots(figsize=(11, max(6, len(pivot) * 0.35)), dpi=180)
    y = np.arange(len(pivot))
    h = 0.18
    selectors = ["random_ego", "oracle_low_policy_tv", "oracle_high_agreement", "oracle_best_xp"]
    colors = ["#6b7280", "#0f766e", "#f97316", "#dc2626"]
    for k, sel in enumerate(selectors):
        vals = pivot[sel] if sel in pivot else np.full(len(pivot), np.nan)
        ax.barh(y + (k - 1.5) * h, vals, height=h, color=colors[k], label=sel)
    ax.set_yticks(y)
    ax.set_yticklabels([short_name(m) for m in pivot.index], fontsize=8)
    ax.set_xlabel("XP under selector")
    ax.set_title("Figure 4. Oracle policy selector intervention from existing pair metrics")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"figure4_oracle_selector_intervention.{ext}", bbox_inches="tight")
    plt.close(fig)
    return summary


def failure_decomposition(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    pair_path = out_dir / "tables" / "pair_level_diagnostics_all_available.csv"
    if not pair_path.exists():
        oracle_selector_analysis(df, out_dir)
    pairs = pd.read_csv(pair_path)
    pairs = safe_num(pairs, ["mean_reward", "xp_out_of_sp_support_rate", "policy_tv", "joint_regret", "action_agreement", "topk_bottleneck_state_mass"])
    pairs = pairs[~pairs["display_method"].eq("FCP old")].copy()
    thresholds = {
        "high_oos": pairs["xp_out_of_sp_support_rate"].quantile(0.75),
        "high_tv": pairs["policy_tv"].quantile(0.75),
        "high_regret": pairs["joint_regret"].quantile(0.75),
        "low_agreement": pairs["action_agreement"].quantile(0.25),
    }
    rows = []
    for method, sub in pairs.groupby("display_method"):
        sub = sub.dropna(subset=["mean_reward"])
        if len(sub) < 4:
            continue
        cutoff = sub["mean_reward"].quantile(0.25)
        fail = sub[sub["mean_reward"] <= cutoff].copy()
        if fail.empty:
            continue
        rows.append({
            "method": method,
            "state_aug": fail["state_aug"].iloc[0],
            "failure_cutoff_xp_pair_reward": cutoff,
            "num_failure_pairs": len(fail),
            "failure_mean_reward": fail["mean_reward"].mean(),
            "failure_mean_oos": fail["xp_out_of_sp_support_rate"].mean(),
            "failure_mean_policy_tv": fail["policy_tv"].mean(),
            "failure_mean_joint_regret": fail["joint_regret"].mean(),
            "failure_mean_agreement": fail["action_agreement"].mean(),
            "failure_high_oos_rate": (fail["xp_out_of_sp_support_rate"] >= thresholds["high_oos"]).mean(),
            "failure_high_tv_rate": (fail["policy_tv"] >= thresholds["high_tv"]).mean(),
            "failure_high_regret_rate": (fail["joint_regret"] >= thresholds["high_regret"]).mean(),
            "failure_low_agreement_rate": (fail["action_agreement"] <= thresholds["low_agreement"]).mean(),
            "threshold_high_oos": thresholds["high_oos"],
            "threshold_high_tv": thresholds["high_tv"],
            "threshold_high_regret": thresholds["high_regret"],
            "threshold_low_agreement": thresholds["low_agreement"],
        })
    dec = pd.DataFrame(rows)
    dec.to_csv(out_dir / "tables" / "figure5_failure_decomposition_pair_proxy.csv", index=False)

    plot = dec.sort_values("failure_high_oos_rate")
    fig, ax = plt.subplots(figsize=(11, max(6, len(plot) * 0.35)), dpi=180)
    y = np.arange(len(plot))
    h = 0.2
    fields = [
        ("failure_high_oos_rate", "high OOS", "#2563eb"),
        ("failure_high_tv_rate", "high Policy TV", "#dc2626"),
        ("failure_high_regret_rate", "high joint regret", "#f97316"),
        ("failure_low_agreement_rate", "low Agreement", "#0f766e"),
    ]
    for k, (field, label, color) in enumerate(fields):
        ax.barh(y + (k - 1.5) * h, plot[field], height=h, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([short_name(m) for m in plot["method"]], fontsize=8)
    ax.set_xlabel("Share among bottom-quartile XP pairings")
    ax.set_title("Figure 5. XP failure decomposition proxy from pair-level trajectories")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "figures" / f"figure5_failure_decomposition_proxy.{ext}", bbox_inches="tight")
    plt.close(fig)
    return dec


def mechanism_attribution(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    d = df[df["diagnostics"] == "available"].copy()
    def family(method: str) -> str:
        if method.startswith("PPO CNN"):
            return "PPO"
        if method.startswith("MAPPO"):
            return "MAPPO"
        if method.startswith("PPO-E3T"):
            return "E3T"
        if method.startswith("MEP"):
            return "MEP"
        if method.startswith("TrajeDi"):
            return "TrajeDi"
        if method.startswith("GAMMA"):
            return "GAMMA"
        if method.startswith("FCP"):
            return "FCP"
        return "Other"
    d["family"] = d["method"].map(family)
    agg = d.groupby(["family", "state_aug"], as_index=False).agg(
        n=("method", "count"),
        SP=("SP", "mean"),
        XP=("XP", "mean"),
        OOS=("OOS", "mean"),
        Agreement=("Agreement", "mean"),
        Policy_TV=("Policy_TV", "mean"),
        Joint_regret=("Joint_regret", "mean"),
    )
    agg.to_csv(out_dir / "tables" / "method_mechanism_family_summary.csv", index=False)
    return agg


def write_report(out_dir: Path, df: pd.DataFrame, corr: pd.DataFrame, delta: pd.DataFrame, reg: pd.DataFrame, oracle: pd.DataFrame, fail: pd.DataFrame, mech: pd.DataFrame) -> None:
    def row_for(corr_subset, metric):
        r = corr[(corr["subset"] == corr_subset) & (corr["metric"] == metric)]
        return r.iloc[0] if not r.empty else None
    ca = row_for("no_fcp_old", "CoverageAgreement")
    best_reg = reg.sort_values("loo_rmse").iloc[0] if not reg.empty else None
    oracle_random = oracle[oracle["selector"] == "random_ego"]["mean_xp"].mean() if not oracle.empty else math.nan
    oracle_best = oracle[oracle["selector"] == "oracle_best_xp"]["mean_xp"].mean() if not oracle.empty else math.nan
    lines = [
        "# Paper Analysis From Existing Models (2026-05-23)",
        "",
        "## Scope",
        "- Uses existing trained checkpoints and diagnostics only; no training is performed.",
        "- `state_aug` is rollout state-sampling through `initial_state_buffer`; `obs_view` is tracked separately.",
        "- `FCP old` is retained in tables but excluded from main correlation fits as a collapsed outlier.",
        "",
        "## Main Findings",
    ]
    if ca is not None:
        lines.append(f"- Coverage-alignment score `(1-OOS)*Agreement` correlates with XP after excluding FCP old: Pearson `{ca.pearson:.3f}`, Spearman `{ca.spearman:.3f}`, bootstrap Pearson CI `[{ca.pearson_ci_low:.3f}, {ca.pearson_ci_high:.3f}]`.")
    if best_reg is not None:
        lines.append(f"- Best controlled regression by LOO RMSE is `{best_reg.model}` with LOO Pearson `{best_reg.loo_pearson:.3f}` and LOO RMSE `{best_reg.loo_rmse:.2f}`.")
    if np.isfinite(oracle_random) and np.isfinite(oracle_best):
        lines.append(f"- Oracle best-XP selector mean across diagnosable methods is `{oracle_best:.2f}` vs random-ego mean `{oracle_random:.2f}`, showing convention selection headroom.")
    lines.extend([
        "- State-aug arrows test whether XP gains track OOS reductions within the same method family.",
        "- Failure decomposition is currently a pair-level trajectory proxy, not per-timestep logging; it identifies whether low-XP pairings concentrate in high-OOS, high-TV, high-regret, or low-agreement regimes.",
        "",
        "## Artifacts",
        "- Table 1: `tables/table1_complete_results.md` and `.csv`",
        "- Figure 1: `figures/figure1_xp_vs_coverage_agreement.png/pdf`",
        "- Figure 2: `figures/figure2_state_aug_arrows_oos_xp.png/pdf`",
        "- Figure 3: `figures/figure3_controlled_regression_actual_vs_predicted.png/pdf`",
        "- Figure 4: `figures/figure4_oracle_selector_intervention.png/pdf`",
        "- Figure 5: `figures/figure5_failure_decomposition_proxy.png/pdf`",
        "",
        "## Caveats",
        "- Oracle selector currently uses existing pair-level diagnostics rather than online estimated partner identity. It is an upper-bound/intervention analysis, not a deployable method.",
        "- Failure decomposition uses pair-level trajectory aggregates. A stricter per-timestep version would require extending diagnostics to log state flags at every rollout step.",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    ensure_dirs(args.out_dir)
    df = load_summary(args.summary_csv)
    write_table1(df, args.out_dir)
    corr = plot_scatter_and_correlations(df, args.out_dir, args.bootstrap)
    delta = state_aug_attribution(df, args.out_dir)
    reg = regression_analysis(df, args.out_dir)
    oracle = oracle_selector_analysis(df, args.out_dir)
    fail = failure_decomposition(df, args.out_dir)
    mech = mechanism_attribution(df, args.out_dir)
    write_report(args.out_dir, df, corr, delta, reg, oracle, fail, mech)
    print(f"[paper-analysis] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
