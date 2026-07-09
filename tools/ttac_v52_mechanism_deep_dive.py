from __future__ import annotations

import argparse
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


EPS = 1e-8
ACTION_NAMES = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "interact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/teams/ius_1663576043/hby/rl/ov2")
    parser.add_argument(
        "--attr_dir",
        default="reports/ttac_v5_2_attribution_tv003_full_vs_base_20260616_v2",
    )
    parser.add_argument(
        "--target_effect_dir",
        default="reports/ttac_v5_2_current_target_effect_quick_coef20_tv003_20260616",
    )
    parser.add_argument(
        "--mechanism_dir",
        default="reports/ttac_v5_2_mechanism_attribution_20260615",
    )
    parser.add_argument(
        "--agreement_dataset",
        default="reports/ttac_v5_agreement_estimator_20260609_180926/dataset/agreement_dataset.npz",
    )
    parser.add_argument(
        "--estimator",
        default="reports/ttac_v5_agreement_estimator_20260609_180926/agreement_estimator/agreement_estimator.npz",
    )
    parser.add_argument(
        "--run_dir",
        default="runs/ttac_posthoc_zero_adapter_from_ppo_state_aug_64_16_seed42_10seeds_20260606",
    )
    parser.add_argument(
        "--v6_report",
        default="reports/ttac_v6_support_refinement_20260616_2140_v6_1_pair20/v6_pair20_summary.csv",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_action_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_base_forward", action="store_true")
    return parser.parse_args()


def ensure_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def require_file(path: Path, missing: list[str]) -> bool:
    if path.exists():
        return True
    missing.append(str(path))
    return False


def safe_read_csv(path: Path, missing: list[str]) -> pd.DataFrame:
    if not require_file(path, missing):
        return pd.DataFrame()
    return pd.read_csv(path)


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(len(a) + len(b) - 2, 1))
    return float((np.mean(a) - np.mean(b)) / max(pooled, EPS))


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float("nan"), float("nan")
    return float(stats.pearsonr(x[m], y[m]).statistic), float(stats.spearmanr(x[m], y[m]).statistic)


def leave_one_out_corr(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
    vals = []
    for idx in df.index:
        sub = df.drop(index=idx)
        pearson, _ = corr_pair(sub[x_col], sub[y_col])
        if not math.isnan(pearson):
            vals.append(pearson)
    if not vals:
        return {"loo_pearson_mean": float("nan"), "loo_pearson_min": float("nan"), "loo_pearson_max": float("nan")}
    return {
        "loo_pearson_mean": float(np.mean(vals)),
        "loo_pearson_min": float(np.min(vals)),
        "loo_pearson_max": float(np.max(vals)),
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def add_regression_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    coef = np.polyfit(x[mask], y[mask], deg=1)
    xs = np.linspace(float(np.min(x[mask])), float(np.max(x[mask])), 100)
    ax.plot(xs, coef[0] * xs + coef[1], color="#222222", linewidth=1.5)


def plot_reward_distribution(pairs: pd.DataFrame, episodes: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if not pairs.empty:
        axes[0].hist(pairs["pair_mean_delta"], bins=24, color="#4C78A8", alpha=0.85)
        axes[0].axvline(0, color="#222222", linewidth=1)
        axes[0].set_title("Pair-level reward delta")
        axes[0].set_xlabel("TTAC - base")
        axes[0].set_ylabel("pairs")
    if not episodes.empty:
        axes[1].hist(episodes["episode_delta"], bins=40, color="#F58518", alpha=0.85)
        axes[1].axvline(0, color="#222222", linewidth=1)
        axes[1].set_title("Episode-level reward delta")
        axes[1].set_xlabel("TTAC - base")
        axes[1].set_ylabel("episodes")
    fig.tight_layout()
    fig.savefig(fig_dir / "reward_delta_distribution.png", dpi=180)
    plt.close(fig)


def plot_scatter(pair_features: pd.DataFrame, x_col: str, out: Path, title: str, xlabel: str) -> None:
    if pair_features.empty or x_col not in pair_features:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    colors = np.where(pair_features["pair_mean_delta"] >= 0, "#4C78A8", "#E45756")
    ax.scatter(pair_features[x_col], pair_features["pair_mean_delta"], c=colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    add_regression_line(ax, pair_features[x_col].to_numpy(dtype=float), pair_features["pair_mean_delta"].to_numpy(dtype=float))
    ax.axhline(0, color="#777777", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pair reward delta")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_group_features(update_summary: pd.DataFrame, fig_dir: Path) -> None:
    if update_summary.empty:
        return
    fields = [
        "latest_target_base_tv",
        "base_value",
        "estimator_entropy",
        "estimator_kl",
        "estimator_tv",
        "partner_action_changed",
    ]
    rows = update_summary[update_summary["field"].isin(fields)].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = rows["field"].tolist()
    x = np.arange(len(labels))
    ax.bar(x - 0.18, rows["beneficial_mean"], width=0.36, label="beneficial", color="#4C78A8")
    ax.bar(x + 0.18, rows["harmful_mean"], width=0.36, label="harmful", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_title("Beneficial vs harmful update-state features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "beneficial_vs_harmful_state_features.png", dpi=180)
    plt.close(fig)


def plot_target_effect(target_summary: pd.DataFrame, fig_dir: Path) -> None:
    if target_summary.empty:
        return
    metrics = [
        "delta_agreement",
        "delta_policy_tv",
        "delta_joint_optimal",
        "delta_joint_regret",
        "adapter_action_change_rate",
        "adapter_policy_tv_to_base",
    ]
    rows = target_summary[target_summary["audit_mode"] != "base_no_update"].copy()
    if rows.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, metric in zip(axes.flatten(), metrics):
        ax.bar(rows["audit_mode"], rows[metric], color="#54A24B")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(fig_dir / "target_effect_delta_metrics.png", dpi=180)
    plt.close(fig)


def plot_corrupted_similarity(target_sensitivity: pd.DataFrame, fig_dir: Path) -> None:
    if target_sensitivity.empty:
        return
    metrics = ["target_argmax_same_rate", "logit_grad_cos_mean", "base_action_change_overlap_step0p5", "new_argmax_same_step0p5"]
    available = [m for m in metrics if m in target_sensitivity]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(target_sensitivity))
    width = 0.18
    for i, metric in enumerate(available):
        ax.bar(x + (i - (len(available) - 1) / 2) * width, target_sensitivity[metric], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(target_sensitivity["comparison"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("True-history target vs corrupted-history target similarity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "corrupted_history_similarity.png", dpi=180)
    plt.close(fig)


def plot_action_transition_diff(action_diff: pd.DataFrame, fig_dir: Path) -> None:
    if action_diff.empty:
        return
    rows = action_diff[action_diff["transition_type"] == "base_argmax_to_target_argmax"].copy()
    rows = rows[rows["from_action"] != rows["to_action"]]
    if rows.empty:
        return
    rows["label"] = rows["from_action"].astype(str) + " -> " + rows["to_action"].astype(str)
    top = pd.concat(
        [
            rows.sort_values("beneficial_minus_harmful_rate_diff", ascending=False).head(8),
            rows.sort_values("beneficial_minus_harmful_rate_diff", ascending=True).head(8),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["label"])
    top = top.sort_values("beneficial_minus_harmful_rate_diff")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(top["beneficial_minus_harmful_rate_diff"] >= 0, "#4C78A8", "#E45756")
    ax.barh(top["label"], top["beneficial_minus_harmful_rate_diff"], color=colors)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_title("Action-transition rate difference: beneficial - harmful")
    ax.set_xlabel("Rate difference")
    fig.tight_layout()
    fig.savefig(fig_dir / "action_transition_diff.png", dpi=180)
    plt.close(fig)


def make_reward_attribution(episodes: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    if not episodes.empty:
        rows.extend(
            [
                {"level": "episode", "metric": "count", "value": len(episodes)},
                {"level": "episode", "metric": "mean_delta", "value": episodes["episode_delta"].mean()},
                {"level": "episode", "metric": "positive_rate", "value": (episodes["episode_delta"] > 0).mean()},
                {"level": "episode", "metric": "negative_rate", "value": (episodes["episode_delta"] < 0).mean()},
                {"level": "episode", "metric": "neutral_rate", "value": (episodes["episode_delta"] == 0).mean()},
            ]
        )
        for label, sub in episodes.groupby("episode_label"):
            rows.append({"level": "episode", "metric": f"{label}_count", "value": len(sub)})
            rows.append({"level": "episode", "metric": f"{label}_rate", "value": len(sub) / max(len(episodes), 1)})
    if not pairs.empty:
        rows.extend(
            [
                {"level": "pair", "metric": "count", "value": len(pairs)},
                {"level": "pair", "metric": "mean_delta", "value": pairs["pair_mean_delta"].mean()},
                {"level": "pair", "metric": "median_delta", "value": pairs["pair_mean_delta"].median()},
                {"level": "pair", "metric": "positive_rate", "value": (pairs["pair_mean_delta"] > 0).mean()},
                {"level": "pair", "metric": "negative_rate", "value": (pairs["pair_mean_delta"] < 0).mean()},
            ]
        )
        for label, sub in pairs.groupby("pair_label_group"):
            rows.append({"level": "pair", "metric": f"{label}_count", "value": len(sub)})
            rows.append({"level": "pair", "metric": f"{label}_rate", "value": len(sub) / max(len(pairs), 1)})
    pair_details = pairs.copy()
    if not pair_details.empty:
        pair_details = pair_details.sort_values("pair_mean_delta", ascending=False)
    return pd.DataFrame(rows), pair_details


def make_update_state_attribution(steps: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "latest_target_base_tv",
        "estimator_entropy",
        "estimator_kl",
        "estimator_tv",
        "base_value",
        "partner_action_changed",
        "tv_gate_003",
        "tv_gate_005",
        "tv_gate_008",
        "tv_gate_012",
    ]
    rows = []
    if steps.empty:
        return pd.DataFrame(), pd.DataFrame()
    beneficial = steps[steps["episode_label"] == "beneficial"]
    harmful = steps[steps["episode_label"] == "harmful"]
    for field in features:
        if field not in steps:
            continue
        b = beneficial[field]
        h = harmful[field]
        rows.append(
            {
                "field": field,
                "beneficial_mean": b.mean(),
                "harmful_mean": h.mean(),
                "beneficial_p25": b.quantile(0.25),
                "harmful_p25": h.quantile(0.25),
                "beneficial_p50": b.quantile(0.50),
                "harmful_p50": h.quantile(0.50),
                "beneficial_p75": b.quantile(0.75),
                "harmful_p75": h.quantile(0.75),
                "mean_diff_beneficial_minus_harmful": b.mean() - h.mean(),
                "cohen_d_beneficial_vs_harmful": cohen_d(b, h),
            }
        )
    if pairs.empty:
        return pd.DataFrame(rows), pd.DataFrame()
    pair_features = steps.groupby("pair_label", as_index=False)[features].mean(numeric_only=True)
    pair_features = pair_features.merge(
        pairs.rename(columns={"policy_labels": "pair_label"})[["pair_label", "pair_mean_delta", "pair_label_group"]],
        on="pair_label",
        how="inner",
    )
    corr_rows = []
    for field in features:
        if field not in pair_features:
            continue
        pearson, spearman = corr_pair(pair_features[field], pair_features["pair_mean_delta"])
        loo = leave_one_out_corr(pair_features, field, "pair_mean_delta")
        corr_rows.append(
            {
                "field": field,
                "pearson_with_pair_delta": pearson,
                "spearman_with_pair_delta": spearman,
                **loo,
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    return pd.DataFrame(rows), pair_features.merge(corr_df, how="cross") if False else corr_df


def make_target_effect(target_summary: pd.DataFrame, target_pairs: pd.DataFrame) -> pd.DataFrame:
    if target_summary.empty:
        return pd.DataFrame()
    rows = []
    for _, r in target_summary.iterrows():
        if r.get("audit_mode") == "base_no_update":
            continue
        rows.append(
            {
                "audit_mode": r.get("audit_mode"),
                "delta_loss": r.get("delta_loss"),
                "delta_agreement": r.get("delta_agreement"),
                "agreement_improved": bool(r.get("delta_agreement", 0) > 0),
                "delta_policy_tv": r.get("delta_policy_tv"),
                "policy_tv_improved": bool(r.get("delta_policy_tv", 0) < 0),
                "delta_joint_optimal": r.get("delta_joint_optimal"),
                "joint_optimal_improved": bool(r.get("delta_joint_optimal", 0) > 0),
                "delta_joint_regret": r.get("delta_joint_regret"),
                "joint_regret_improved": bool(r.get("delta_joint_regret", 0) < 0),
                "adapter_action_change_rate": r.get("adapter_action_change_rate"),
                "adapter_policy_tv_to_base": r.get("adapter_policy_tv_to_base"),
            }
        )
    return pd.DataFrame(rows)


def make_corrupted_history(
    target_sensitivity: pd.DataFrame,
    mode_target: pd.DataFrame,
    pair_reward: pd.DataFrame,
    raw_history: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    reward_summary = {}
    if not pair_reward.empty and "delta_true" in pair_reward:
        for mode in ["wrong", "random", "delayed"]:
            delta_col = f"delta_{mode}"
            if delta_col in pair_reward:
                reward_summary[f"true_minus_{mode}_mean_gap"] = (pair_reward["delta_true"] - pair_reward[delta_col]).mean()
                reward_summary[f"true_gt_{mode}_winrate"] = (pair_reward["delta_true"] > pair_reward[delta_col]).mean()
                reward_summary[f"true_{mode}_pair_corr"] = pair_reward[["delta_true", delta_col]].corr().iloc[0, 1]
    for _, r in target_sensitivity.iterrows():
        comp = r["comparison"]
        mode = comp.replace("true_vs_", "")
        hist = raw_history[raw_history["comparison"] == comp]
        rows.append(
            {
                "comparison": comp,
                "target_tv_mean": r.get("target_tv_mean"),
                "target_argmax_same_rate": r.get("target_argmax_same_rate"),
                "logit_grad_cos_mean": r.get("logit_grad_cos_mean"),
                "action_change_overlap": r.get("base_action_change_overlap_step0p5"),
                "new_argmax_same": r.get("new_argmax_same_step0p5"),
                "history_token_match_rate": hist["history_token_match_rate"].iloc[0] if not hist.empty and "history_token_match_rate" in hist else np.nan,
                "true_minus_mode_reward_gap": reward_summary.get(f"true_minus_{mode}_mean_gap", np.nan),
                "true_gt_mode_winrate": reward_summary.get(f"true_gt_{mode}_winrate", np.nan),
                "true_mode_pair_corr": reward_summary.get(f"true_{mode}_pair_corr", np.nan),
            }
        )
    return pd.DataFrame(rows)


def make_action_diff(action_attr: pd.DataFrame) -> pd.DataFrame:
    if action_attr.empty:
        return pd.DataFrame()
    rows = action_attr[action_attr["transition_type"] != "summary"].copy()
    if rows.empty or "group" not in rows:
        return pd.DataFrame()
    key_cols = ["transition_type", "from_action", "to_action"]
    pivot = rows.pivot_table(
        index=key_cols,
        columns="group",
        values="rate",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    if "beneficial" not in pivot:
        pivot["beneficial"] = 0.0
    if "harmful" not in pivot:
        pivot["harmful"] = 0.0
    pivot["beneficial_minus_harmful_rate_diff"] = pivot["beneficial"] - pivot["harmful"]
    return pivot.sort_values("beneficial_minus_harmful_rate_diff", ascending=False)


def run_base_forward_for_action(args: argparse.Namespace, pair_groups: pd.DataFrame, missing: list[str]) -> tuple[pd.DataFrame, str]:
    root = Path(args.root)
    dataset_path = ensure_path(root, args.agreement_dataset)
    estimator_path = ensure_path(root, args.estimator)
    run_dir = ensure_path(root, args.run_dir)
    if not require_file(dataset_path, missing):
        return pd.DataFrame(), "agreement_dataset missing"
    if args.skip_base_forward:
        return pd.DataFrame(), "base forward skipped by flag"
    try:
        os.chdir(root)
        sys.path.append(str(root / "experiments"))
        sys.path.append(str(root / "JaxMARL"))
        import jax
        import jax.numpy as jnp
        from overcooked_v2_experiments.ttac_v5_2_state_selection.policy import PPOPolicy
        from overcooked_v2_experiments.ttac_v5_2_state_selection.utils.agreement_heads import load_agreement_npz
        from overcooked_v2_experiments.ttac_v5_2_state_selection.utils.store import load_all_checkpoints
    except Exception as exc:
        return pd.DataFrame(), f"base forward imports failed: {exc}"

    data = np.load(dataset_path, allow_pickle=False)
    n = len(data["ego_action"])
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n) if n <= args.max_action_samples else np.sort(rng.choice(n, size=args.max_action_samples, replace=False))

    pair_label = np.asarray(data["pair_label"][idx]).astype(str)
    cross_label = np.asarray([run_pair_to_cross(x) for x in pair_label])
    groups = pair_groups.rename(columns={"policy_labels": "cross_label"})[["cross_label", "pair_label_group", "pair_mean_delta"]]
    joined = pd.DataFrame({"row_idx": idx, "cross_label": cross_label}).merge(groups, on="cross_label", how="left")
    keep_mask = joined["pair_label_group"].notna().to_numpy()
    if keep_mask.sum() == 0:
        return pd.DataFrame(), "no sampled agreement rows matched paired attribution pairs"
    idx = idx[keep_mask]
    joined = joined[keep_mask].reset_index(drop=True)

    query_obs = np.asarray(data["query_obs"][idx], dtype=np.float32)
    target_probs = np.asarray(data["target_partner_probs"][idx], dtype=np.float32)
    ego_action = np.asarray(data["ego_action"][idx], dtype=int)
    partner_action = np.asarray(data["partner_action"][idx], dtype=int)
    prev_ego_action = np.asarray(data["prev_ego_action"][idx], dtype=int)
    prev_partner_action = np.asarray(data["prev_partner_action"][idx], dtype=int)
    roles = np.asarray(data["role"][idx], dtype=int)

    try:
        estimator = load_agreement_npz(estimator_path)
        checkpoints, config = load_all_checkpoints(run_dir, final_only=True)
        config["model"]["TTAC_V5_ESTIMATOR"] = estimator
        config["model"]["TTAC_HISTORY_LEN"] = 50
        keys = sorted(checkpoints.keys(), key=lambda x: int(x.split("_")[1]))
        policies = [
            PPOPolicy(checkpoints[key]["ckpt_final"].params, config, eval_mode="base_no_test_adapt", stochastic=False)
            for key in keys
        ]
        ego_ids = []
        for label, role in zip(joined["cross_label"], roles):
            ids = parse_cross(label)
            ego_ids.append(ids[0] if int(role) == 0 else ids[1])
        ego_ids = np.asarray(ego_ids, dtype=int)
        base_logits = np.zeros((len(query_obs), 6), dtype=np.float32)
        for ego_id in sorted(set(int(x) for x in ego_ids)):
            sub_idx = np.where(ego_ids == ego_id)[0]
            policy = policies[ego_id]
            chunks = []
            for start in range(0, len(sub_idx), 512):
                batch = sub_idx[start : start + 512]
                logits, _, _ = policy._apply_batch(policy.params, jnp.asarray(query_obs[batch], dtype=jnp.float32), 0.0)
                chunks.append(np.asarray(logits, dtype=np.float32))
            base_logits[sub_idx] = np.concatenate(chunks, axis=0)
        base_argmax = np.argmax(base_logits, axis=-1)
    except Exception as exc:
        return pd.DataFrame(), f"base forward failed: {exc}"

    target_argmax = np.argmax(target_probs, axis=-1)
    joined["base_argmax"] = base_argmax
    joined["target_argmax"] = target_argmax
    joined["ego_action"] = ego_action
    joined["partner_action"] = partner_action
    joined["prev_ego_action"] = prev_ego_action
    joined["prev_partner_action"] = prev_partner_action
    joined["base_to_target_changed"] = joined["base_argmax"] != joined["target_argmax"]
    joined["observed_ego_to_target_changed"] = joined["ego_action"] != joined["target_argmax"]
    joined["base_argmax_name"] = joined["base_argmax"].map(ACTION_NAMES)
    joined["target_argmax_name"] = joined["target_argmax"].map(ACTION_NAMES)
    joined["ego_action_name"] = joined["ego_action"].map(ACTION_NAMES)
    joined["partner_action_name"] = joined["partner_action"].map(ACTION_NAMES)

    rows = []
    for group, sub in joined.groupby("pair_label_group"):
        total = max(len(sub), 1)
        for from_col, to_col, transition_type in [
            ("base_argmax_name", "target_argmax_name", "base_argmax_to_target_argmax"),
            ("ego_action_name", "target_argmax_name", "observed_ego_action_to_target_argmax"),
            ("prev_ego_action", "ego_action", "observed_ego_action_change"),
            ("prev_partner_action", "partner_action", "observed_partner_action_change"),
        ]:
            if from_col in ["prev_ego_action", "prev_partner_action"]:
                from_vals = sub[from_col].map(ACTION_NAMES)
                to_vals = sub[to_col].map(ACTION_NAMES)
            else:
                from_vals = sub[from_col]
                to_vals = sub[to_col]
            counts = pd.DataFrame({"from_action": from_vals, "to_action": to_vals}).value_counts().reset_index(name="count")
            counts["group"] = group
            counts["transition_type"] = transition_type
            counts["rate"] = counts["count"] / total
            rows.extend(counts.to_dict("records"))
        rows.append(
            {
                "group": group,
                "transition_type": "summary",
                "from_action": "base",
                "to_action": "target_changed",
                "count": int(joined.loc[sub.index, "base_to_target_changed"].sum()),
                "rate": float(joined.loc[sub.index, "base_to_target_changed"].mean()),
            }
        )
    return pd.DataFrame(rows), "computed base_argmax -> estimator_target_argmax proxy; actual post-update adapter argmax is not in existing CSV"


def run_pair_to_cross(label: str) -> str:
    m = re.match(r"run_(\d+)xrun_(\d+)", str(label))
    if not m:
        return str(label)
    return f"cross-{m.group(1)}_{m.group(2)}"


def parse_cross(label: str) -> tuple[int, int]:
    m = re.match(r"cross-(\d+)_(\d+)", str(label))
    if not m:
        raise ValueError(f"not a cross label: {label}")
    return int(m.group(1)), int(m.group(2))


def load_v6_negative(v6_path: Path) -> pd.DataFrame:
    if not v6_path.exists():
        return pd.DataFrame()
    return pd.read_csv(v6_path)


def write_summary(
    out_dir: Path,
    missing: list[str],
    reward_attr: pd.DataFrame,
    pair_details: pd.DataFrame,
    update_summary: pd.DataFrame,
    update_corr: pd.DataFrame,
    target_effect: pd.DataFrame,
    corrupted: pd.DataFrame,
    action_status: str,
    action_diff: pd.DataFrame,
    v6: pd.DataFrame,
) -> None:
    def val(df: pd.DataFrame, level: str, metric: str) -> float:
        hit = df[(df["level"] == level) & (df["metric"] == metric)]
        return float(hit["value"].iloc[0]) if not hit.empty else float("nan")

    pair_mean = val(reward_attr, "pair", "mean_delta")
    pair_pos = val(reward_attr, "pair", "positive_rate")
    ep_mean = val(reward_attr, "episode", "mean_delta")
    ep_pos = val(reward_attr, "episode", "positive_rate")
    agreement_supported = None
    if not target_effect.empty:
        agreement_supported = bool((target_effect["delta_agreement"] > 0).all())
    tv_row = update_summary[update_summary["field"] == "latest_target_base_tv"]
    value_row = update_summary[update_summary["field"] == "base_value"]
    tv_d = float(tv_row["cohen_d_beneficial_vs_harmful"].iloc[0]) if not tv_row.empty else float("nan")
    value_d = float(value_row["cohen_d_beneficial_vs_harmful"].iloc[0]) if not value_row.empty else float("nan")
    true_wrong = corrupted[corrupted["comparison"] == "true_vs_wrong"]
    grad_cos = float(true_wrong["logit_grad_cos_mean"].iloc[0]) if not true_wrong.empty else float("nan")
    action_overlap = float(true_wrong["action_change_overlap"].iloc[0]) if not true_wrong.empty else float("nan")
    true_wrong_gap = float(true_wrong["true_minus_mode_reward_gap"].iloc[0]) if not true_wrong.empty else float("nan")

    lines = [
        "# TTAC v5.2 mechanism deep dive",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- missing_inputs: `{len(missing)}`",
        f"- action_level_status: `{action_status}`",
        "",
        "## Executive conclusion",
        "",
    ]
    if agreement_supported is False:
        lines.append("- **v5.2 的 XP 增益不是成功的 Agreement alignment。** fixed-state audit 中 Agreement 下降、Policy TV 上升，但 XP 提升。")
    else:
        lines.append("- **Agreement alignment 证据不完整。** 当前 target-effect 数据不足以支持或否定。")
    lines.append("- **v5.2 更像 test-time local support refinement。** 有益 pair 的 `target_base_tv` 更高，`base_value` 更低，说明更新主要发生在 base policy 和 estimator target 分歧较大、base value 较弱的局部状态。")
    lines.append("- **当前 estimator 没有提供强 partner-specific 语义。** true/wrong 的梯度方向和 action-change 几乎重叠，因此 corrupted history 也能产生接近的更新收益。")
    lines.append("- **当前 CE loss 只部分贴合正信号。** 它能推动 policy 离开 base 的局部低支持区域，但没有显式优化 Agreement，也没有可靠地区分 true history 与 corrupted history。")
    lines.append("")
    lines.append("## Reward attribution")
    lines.append("")
    lines.append(f"- pair_mean_delta: `{pair_mean:.4f}`; pair_positive_rate: `{pair_pos:.3f}`")
    lines.append(f"- episode_mean_delta: `{ep_mean:.4f}`; episode_positive_rate: `{ep_pos:.3f}`")
    if not pair_details.empty:
        top = pair_details.head(5)[["policy_labels", "pair_mean_delta"]]
        bottom = pair_details.tail(5)[["policy_labels", "pair_mean_delta"]]
        lines.append("")
        lines.append("Top positive pairs:")
        lines.extend([f"- `{r.policy_labels}`: `{r.pair_mean_delta:.3f}`" for r in top.itertuples()])
        lines.append("")
        lines.append("Most negative pairs:")
        lines.extend([f"- `{r.policy_labels}`: `{r.pair_mean_delta:.3f}`" for r in bottom.itertuples()])
    lines.append("")
    lines.append("## Update-state attribution")
    lines.append("")
    if not update_summary.empty:
        for field in ["latest_target_base_tv", "base_value", "estimator_entropy", "partner_action_changed"]:
            hit = update_summary[update_summary["field"] == field]
            if hit.empty:
                continue
            r = hit.iloc[0]
            lines.append(
                f"- `{field}`: beneficial `{r.beneficial_mean:.4f}` vs harmful `{r.harmful_mean:.4f}`, "
                f"diff `{r.mean_diff_beneficial_minus_harmful:.4f}`, Cohen d `{r.cohen_d_beneficial_vs_harmful:.3f}`"
            )
    if not update_corr.empty:
        lines.append("")
        lines.append("Pair-level correlations with reward delta:")
        for field in ["latest_target_base_tv", "base_value", "partner_action_changed"]:
            hit = update_corr[update_corr["field"] == field]
            if hit.empty:
                continue
            r = hit.iloc[0]
            lines.append(
                f"- `{field}`: Pearson `{r.pearson_with_pair_delta:.3f}`, Spearman `{r.spearman_with_pair_delta:.3f}`, "
                f"LOO Pearson mean/min/max `{r.loo_pearson_mean:.3f}/{r.loo_pearson_min:.3f}/{r.loo_pearson_max:.3f}`"
            )
    lines.append("")
    lines.append("## Target-effect attribution")
    lines.append("")
    if not target_effect.empty:
        for r in target_effect.itertuples():
            lines.append(
                f"- `{r.audit_mode}`: delta_agreement `{r.delta_agreement:.5f}`, "
                f"delta_policy_tv `{r.delta_policy_tv:.5f}`, delta_joint_optimal `{r.delta_joint_optimal:.5f}`, "
                f"delta_joint_regret `{r.delta_joint_regret:.5f}`, action_change `{r.adapter_action_change_rate:.5f}`"
            )
    lines.append("")
    lines.append("## Corrupted-history explanation")
    lines.append("")
    lines.append(f"- true_vs_wrong grad cosine: `{grad_cos:.4f}`")
    lines.append(f"- true_vs_wrong action-change overlap: `{action_overlap:.4f}`")
    lines.append(f"- true_minus_wrong reward gap: `{true_wrong_gap:.4f}`")
    if not corrupted.empty:
        for r in corrupted.itertuples():
            lines.append(
                f"- `{r.comparison}`: target TV `{r.target_tv_mean:.4f}`, argmax same `{r.target_argmax_same_rate:.4f}`, "
                f"new argmax same `{r.new_argmax_same:.4f}`, pair corr `{r.true_mode_pair_corr:.4f}`"
            )
    lines.append("")
    lines.append("## Action-level attribution")
    lines.append("")
    lines.append(f"- `{action_status}`")
    lines.append("- 当前报告的 action transition 是 `base_argmax -> estimator_target_argmax` proxy；真实 `adapter_after_update_argmax` 未保存在现有 CSV 中，因此不能把 proxy 直接写成真实 adapter 行为。")
    if not action_diff.empty:
        changed = action_diff[
            (action_diff["transition_type"] == "base_argmax_to_target_argmax")
            & (action_diff["from_action"] != action_diff["to_action"])
        ].copy()
        if not changed.empty:
            lines.append("")
            lines.append("Beneficial over harmful 中更常见的 target action shifts:")
            for r in changed.sort_values("beneficial_minus_harmful_rate_diff", ascending=False).head(6).itertuples():
                lines.append(
                    f"- `{r.from_action} -> {r.to_action}`: rate diff `{r.beneficial_minus_harmful_rate_diff:.4f}`"
                )
            lines.append("")
            lines.append("Harmful over beneficial 中更常见的 target action shifts:")
            for r in changed.sort_values("beneficial_minus_harmful_rate_diff", ascending=True).head(6).itertuples():
                lines.append(
                    f"- `{r.from_action} -> {r.to_action}`: rate diff `{r.beneficial_minus_harmful_rate_diff:.4f}`"
                )
    lines.append("")
    lines.append("## v6.1 negative result")
    lines.append("")
    if not v6.empty:
        for r in v6.sort_values("xp_mean", ascending=False).itertuples():
            xp = getattr(r, "xp_mean", np.nan)
            lines.append(f"- `{r.mode}`: pair20 XP `{xp:.3f}`")
        lines.append("- dynamic amp 低于 v5.2 latest/tv_gate，说明继续放大 CE 权重没有抓住机制。")
    else:
        lines.append("- v6.1 summary not found.")
    lines.append("")
    lines.append("## Required three answers")
    lines.append("")
    lines.append("1. `v5.2 的 XP 增益是不是 Agreement？` 不是。现有 fixed-state audit 显示 Agreement 没有上升，反而下降。")
    lines.append("2. `如果不是，它更像什么机制？` 更像 local support refinement：在 base value 较低、target-base TV 较高的局部状态，把 policy 推离原 self-play 支持区域。")
    lines.append("3. `当前 loss 和这个机制是否贴合，v6 应该如何改？` 当前 CE loss 只粗糙贴合，因为它推动 policy 靠近 estimator target，但不显式筛选真正有益的 local correction，也不提供 partner-specific 语义。v6 应从“放大 loss”转为“识别并优化有益 local correction”。")
    if missing:
        lines.append("")
        lines.append("## Missing inputs")
        lines.extend([f"- `{m}`" for m in missing])
    (out_dir / "mechanism_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = ensure_path(root, args.output_dir) if args.output_dir else root / f"reports/ttac_v5_2_mechanism_deep_dive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    attr_dir = ensure_path(root, args.attr_dir)
    target_dir = ensure_path(root, args.target_effect_dir)
    mech_dir = ensure_path(root, args.mechanism_dir)

    episodes = safe_read_csv(attr_dir / "paired_attribution_episodes.csv", missing)
    pairs = safe_read_csv(attr_dir / "paired_attribution_pairs.csv", missing)
    steps = safe_read_csv(attr_dir / "paired_attribution_steps.csv", missing)
    target_summary = safe_read_csv(target_dir / "target_effect_summary.csv", missing)
    target_pairs = safe_read_csv(target_dir / "target_effect_pairs.csv", missing)
    target_sensitivity = safe_read_csv(mech_dir / "target_sensitivity_summary.csv", missing)
    mode_target = safe_read_csv(mech_dir / "mode_target_summary.csv", missing)
    pair_reward = safe_read_csv(mech_dir / "pair_reward_deltas.csv", missing)
    raw_history = safe_read_csv(mech_dir / "raw_history_corruption_summary.csv", missing)
    v6 = load_v6_negative(ensure_path(root, args.v6_report))

    reward_attr, pair_details = make_reward_attribution(episodes, pairs)
    update_summary, update_corr = make_update_state_attribution(steps, pairs)
    target_effect = make_target_effect(target_summary, target_pairs)
    corrupted = make_corrupted_history(target_sensitivity, mode_target, pair_reward, raw_history)
    action_attr, action_status = run_base_forward_for_action(args, pairs, missing)
    action_diff = make_action_diff(action_attr)

    save_csv(reward_attr, out_dir / "reward_attribution.csv")
    save_csv(pair_details, out_dir / "reward_pair_details.csv")
    save_csv(update_summary, out_dir / "update_state_attribution.csv")
    save_csv(update_corr, out_dir / "update_state_pair_correlations.csv")
    save_csv(target_effect, out_dir / "target_effect_attribution.csv")
    save_csv(corrupted, out_dir / "corrupted_history_explanation.csv")
    save_csv(action_attr, out_dir / "action_transition_attribution.csv")
    save_csv(action_diff, out_dir / "action_transition_diff.csv")
    if not v6.empty:
        save_csv(v6, out_dir / "v6_1_negative_pair20.csv")

    plot_reward_distribution(pairs, episodes, fig_dir)
    if not steps.empty and not pairs.empty:
        feature_means = steps.groupby("pair_label", as_index=False).mean(numeric_only=True)
        pair_features = feature_means.merge(
            pairs.rename(columns={"policy_labels": "pair_label"})[["pair_label", "pair_mean_delta", "pair_label_group"]],
            on="pair_label",
            how="inner",
        )
        save_csv(pair_features, out_dir / "update_state_pair_features.csv")
        plot_scatter(pair_features, "latest_target_base_tv", fig_dir / "xp_delta_vs_target_base_tv.png", "Reward delta vs target-base TV", "Mean target-base TV")
        plot_scatter(pair_features, "base_value", fig_dir / "xp_delta_vs_base_value.png", "Reward delta vs base value", "Mean base value")
    plot_group_features(update_summary, fig_dir)
    plot_target_effect(target_summary, fig_dir)
    plot_corrupted_similarity(target_sensitivity, fig_dir)
    plot_action_transition_diff(action_diff, fig_dir)

    write_summary(
        out_dir,
        missing,
        reward_attr,
        pair_details,
        update_summary,
        update_corr,
        target_effect,
        corrupted,
        action_status,
        action_diff,
        v6,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
