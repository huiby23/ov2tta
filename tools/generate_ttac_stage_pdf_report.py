from __future__ import annotations

import math
import os
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(os.environ.get("OV2_ROOT", "/teams/ius_1663576043/hby/rl/ov2"))
OUT_DIR = ROOT / "reports" / "ttac_stage_report_20260617"
FIG_DIR = OUT_DIR / "page_previews"
PDF_PATH = OUT_DIR / "ttac_stage_report_zh_20260617.pdf"

PAGE_W, PAGE_H = 11.69, 8.27  # A4 landscape
BG = "#FFFFFF"
INK = "#17202A"
MUTED = "#5D6D7E"
BLUE = "#2E86AB"
ORANGE = "#F18F01"
GREEN = "#4CAF50"
RED = "#D64550"
PURPLE = "#7E57C2"
GRAY = "#D5DBDB"
LIGHT_BLUE = "#EAF4F8"
LIGHT_ORANGE = "#FFF3E0"


def pick_font() -> None:
    candidates = [
        "/usr/share/fonts/truetype/SimHei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def wrap(text: str, width: int = 38) -> str:
    lines = []
    for para in str(text).split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width, replace_whitespace=False))
    return "\n".join(lines)


def add_text(ax, x, y, text, size=13, color=INK, weight="normal", ha="left", va="top", width=None, line=1.28):
    if width:
        text = wrap(text, width)
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=line)


def add_title(ax, title: str, subtitle: str | None = None, section: str | None = None):
    if section:
        add_text(ax, 0.055, 0.965, section.upper(), size=9.5, color=BLUE, weight="bold")
    add_text(ax, 0.055, 0.92, title, size=23, weight="bold")
    if subtitle:
        add_text(ax, 0.055, 0.865, subtitle, size=11.5, color=MUTED, width=120)
    ax.plot([0.055, 0.945], [0.835, 0.835], transform=ax.transAxes, color=GRAY, lw=1)


def add_footer(ax, page: int):
    ax.text(0.055, 0.035, "TTAC 阶段性机制报告 - 2026-06-17", transform=ax.transAxes, fontsize=8.5, color="#7B7D7D")
    ax.text(0.945, 0.035, f"{page}", transform=ax.transAxes, fontsize=8.5, color="#7B7D7D", ha="right")


def new_page(pdf: PdfPages, page_num: int, title: str | None = None, subtitle: str | None = None, section: str | None = None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    if title:
        add_title(ax, title, subtitle, section)
    add_footer(ax, page_num)
    return fig, ax


def finish_page(pdf: PdfPages, fig, page_num: int):
    pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
    fig.savefig(FIG_DIR / f"page_{page_num:02d}.png", dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def inset(fig, rect):
    return fig.add_axes(rect)


def load_csv(rel: str) -> pd.DataFrame:
    p = ROOT / rel
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def metric_box(ax, x, y, w, h, label, value, note="", color=BLUE):
    ax.add_patch(plt.Rectangle((x, y - h), w, h, transform=ax.transAxes, facecolor="#F8FBFC", edgecolor="#D6EAF8", lw=1.2))
    add_text(ax, x + 0.025, y - 0.035, label, size=9.5, color=MUTED, weight="bold")
    add_text(ax, x + 0.025, y - 0.095, value, size=22, color=color, weight="bold")
    if note:
        add_text(ax, x + 0.025, y - 0.145, note, size=8.3, color=MUTED, width=24)


def make_table(ax, df: pd.DataFrame, col_widths=None, font_size=9.5, header_color="#ECF3F7", scale_y=1.28):
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, scale_y)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCD1D1")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#FAFAFA")
    return tbl


def safe_float(x, default=np.nan):
    try:
        if x == "" or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def format_num(x, nd=2):
    x = safe_float(x)
    if np.isnan(x):
        return "-"
    return f"{x:.{nd}f}"


def prepare_data():
    table = load_csv("reports/current_complete_results_with_diagnostics_20260523.csv")
    deep_dir = "reports/ttac_v5_2_mechanism_deep_dive_20260616"
    reward_attr = load_csv(f"{deep_dir}/reward_attribution.csv")
    update_attr = load_csv(f"{deep_dir}/update_state_attribution.csv")
    update_corr = load_csv(f"{deep_dir}/update_state_pair_correlations.csv")
    target_effect = load_csv(f"{deep_dir}/target_effect_attribution.csv")
    corrupted = load_csv(f"{deep_dir}/corrupted_history_explanation.csv")
    v6 = load_csv(f"{deep_dir}/v6_1_negative_pair20.csv")
    ttac_full = pd.DataFrame(
        [
            ["base no-adapt", 201.836, 156.764, "PPO CNN state-aug checkpoint, adapter off"],
            ["v5.2 latest coef=1", 201.436, 164.209, "full 500 seeds, all pairings"],
            ["v5.2 latest coef=20", 200.620, 165.768, "best full true-history line"],
            ["v5.2 TV-gate 0.03 coef=20", 200.456, 165.748, "full eval, tied with latest"],
        ],
        columns=["mode", "SP", "XP", "note"],
    )
    return table, reward_attr, update_attr, update_corr, target_effect, corrupted, v6, ttac_full


def add_bullets(ax, x, y, bullets, size=11.5, width=66, gap=0.078, color=INK):
    yy = y
    for b in bullets:
        add_text(ax, x, yy, "-", size=size + 1, color=BLUE, weight="bold")
        add_text(ax, x + 0.027, yy, b, size=size, color=color, width=width)
        yy -= gap + 0.018 * max(0, len(wrap(b, width).splitlines()) - 1)
    return yy


def generate_report():
    pick_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table, reward_attr, update_attr, update_corr, target_effect, corrupted, v6, ttac_full = prepare_data()

    with PdfPages(PDF_PATH) as pdf:
        page = 1
        fig, ax = new_page(pdf, page)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#F7FBFC", edgecolor="none"))
        add_text(ax, 0.07, 0.86, "TTAC 阶段性实验报告", size=31, weight="bold")
        add_text(ax, 0.07, 0.78, "从 Agreement 假设到 local support refinement 的机制归因", size=18, color=BLUE, weight="bold")
        add_text(ax, 0.07, 0.69, "目的：给研究者人工判断下一步方向使用。报告优先定义术语、解释证据、区分已证实结论与仍未证明的假设。", size=12.5, color=MUTED, width=78)
        metric_box(ax, 0.07, 0.50, 0.22, 0.18, "v5.2 full XP gain", "+9.00", "165.768 vs 156.764", BLUE)
        metric_box(ax, 0.32, 0.50, 0.22, 0.18, "positive pair rate", "85%", "100 pairings 中 85 个均值提升", GREEN)
        metric_box(ax, 0.57, 0.50, 0.22, 0.18, "Agreement delta", "-0.0115", "fixed-state audit 下降", RED)
        add_text(ax, 0.07, 0.24, "一页结论：TTAC v5.2 的收益是真实的，但它不是成功的 partner-specific Agreement alignment。当前证据更支持：它在测试时把策略从低 value、高 target-base TV 的局部状态中推离原 self-play 支持区域，形成 local support refinement。", size=14.2, width=92, line=1.45)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "技术摘要：我们已经证明什么，还没有证明什么", section="summary")
        add_bullets(
            ax,
            0.07,
            0.77,
            [
                "已证明：在 PPO CNN state-aug checkpoint 上，posthoc TTAC v5.2 能把 full XP 从 156.764 提到 165.768，增益约 +9.00。",
                "已证明：这个增益不是由 Agreement 直接上升造成的。固定状态审计中 Agreement 下降，Policy TV 上升，Joint regret 也略升。",
                "已证明：true / wrong / random / delayed history 的更新方向高度相似，说明当前 estimator 没有学到强 partner-specific 语义。",
                "较可信的解释：v5.2 更像 local support refinement，即在 base policy 弱、estimator target 与 base 分歧大的局部状态中做临时动作偏好修正。",
                "尚未证明：哪个 test-time surrogate loss 能稳定、因果地选择 return-improving action correction。下一步需要反事实 action-value 或短分叉 rollout 证据。",
            ],
            size=12.5,
            width=92,
            gap=0.09,
        )
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "术语表：本报告所有指标如何定义", section="definitions")
        defs = pd.DataFrame(
            [
                ["SP", "self-play，同一个训练 run 或同种策略与自己合作的平均 reward。"],
                ["XP", "cross-play，不同 policy seed / partner 组合之间合作的平均 reward，是 ZSC 主要评价指标。"],
                ["ZSC", "zero-shot coordination，测试时与未共同训练过的新伙伴合作。"],
                ["state-aug", "rollout state-sampling + initial_state_buffer 的状态采样增强；不是 agent_view_size。"],
                ["OOS", "out-of-support rate，XP 访问到 self-play 支持外状态的比例；越低通常表示 coverage 越好。"],
                ["Agreement", "两个 policy 在同一 shared state / 同一角色设定下动作选择的一致性；越高表示 convention 更接近。"],
                ["Policy TV", "两个策略动作分布的 total variation distance；越低表示策略分布越接近。"],
                ["Joint regret", "当前联合动作相对一步近似更优联合动作的 regret；越低越好。"],
                ["target-base TV", "estimator target 与 base policy 动作分布的 TV 距离；越高表示 target 想把 policy 推离 base 越远。"],
                ["base value", "base critic 对当前局部状态的价值估计；较低表示 base 认为该状态更弱或更困难。"],
            ],
            columns=["术语", "定义"],
        )
        ax_tbl = inset(fig, [0.07, 0.12, 0.86, 0.67])
        make_table(ax_tbl, defs, col_widths=[0.18, 0.72], font_size=9.5, scale_y=1.58)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "研究路线：我们如何走到 TTAC v5.2", section="trajectory")
        stages = [
            ("1. 传统 PPO 诊断", "PPO CNN standard XP 很低；state-aug 把 XP 大幅提高，说明 coverage 是第一瓶颈。"),
            ("2. Partner modeling 迁移", "E3T/TTAPPO/GAMMA/TALENTS 风格尝试没有稳定证明 partner-specific online adaptation。"),
            ("3. TTAC adapter", "冻结 base，用 adapter 在 test-time 更新，避免破坏基础协作能力。"),
            ("4. v5 Agreement estimator", "原目标是估计 partner policy 并提升 Agreement；后来发现 corrupted history 也能工作。"),
            ("5. v5.2 机制转折", "full eval 证明收益真实，但 fixed-state audit 否定 Agreement alignment 解释。"),
        ]
        y = 0.75
        for i, (head, body) in enumerate(stages):
            ax.add_patch(plt.Circle((0.10, y + 0.005), 0.018, transform=ax.transAxes, color=BLUE if i < 4 else ORANGE))
            ax.plot([0.10, 0.10], [y - 0.08, y - 0.025], transform=ax.transAxes, color=GRAY, lw=2)
            add_text(ax, 0.14, y + 0.025, head, size=13, weight="bold")
            add_text(ax, 0.14, y - 0.02, body, size=11.5, color=MUTED, width=82)
            y -= 0.13
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "基础诊断：state-aug 解决了大块 coverage 问题，但不是最终答案", section="baseline")
        base_rows = pd.DataFrame(
            [
                ["PPO CNN standard", "no", 167.860, 61.407, 0.848, 0.615, 0.369],
                ["PPO CNN state-aug", "yes", 201.836, 156.831, 0.682, 0.749, 0.267],
                ["PPO-E3T no-state predicted CE", "no", 163.336, 74.558, 0.761, 0.762, 0.294],
                ["PPO-E3T state-aug predicted CE", "yes", 180.308, 146.947, 0.682, 0.761, 0.273],
                ["MEP ent=0.05 state-aug", "yes", 169.408, 148.108, 0.662, 0.862, 0.222],
            ],
            columns=["method", "state-aug", "SP", "XP", "OOS", "Agreement", "Policy TV"],
        )
        ax_chart = inset(fig, [0.08, 0.43, 0.42, 0.32])
        ax_chart.bar(["PPO std", "PPO state-aug"], [61.407, 156.831], color=[RED, BLUE])
        ax_chart.set_ylabel("XP")
        ax_chart.set_title("PPO state-aug 带来最大基础提升")
        ax_chart.grid(axis="y", alpha=0.25)
        for i, v in enumerate([61.407, 156.831]):
            ax_chart.text(i, v + 3, f"{v:.1f}", ha="center", fontsize=10)
        ax_tbl = inset(fig, [0.54, 0.23, 0.39, 0.52])
        tdf = base_rows.copy()
        for c in ["SP", "XP"]:
            tdf[c] = tdf[c].map(lambda x: f"{x:.1f}")
        for c in ["OOS", "Agreement", "Policy TV"]:
            tdf[c] = tdf[c].map(lambda x: f"{x:.3f}")
        make_table(ax_tbl, tdf, col_widths=[0.34, 0.12, 0.12, 0.12, 0.10, 0.12, 0.12], font_size=7.1, scale_y=1.45)
        add_text(ax, 0.08, 0.23, "解释：state-aug 在 counter_circuit 上显著提高 XP，但它是训练期覆盖增强，不是 test-time adaptation。本研究后续的问题变成：在强 state-aug baseline 上，还能否通过测试时局部更新修复残余失配。", size=11.5, width=58)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "方法全景：standard 与 state-aug 结果不能混为一谈", section="baseline")
        if not table.empty:
            subset_names = [
                "PPO CNN standard",
                "PPO-E3T CNN no-state predicted CE",
                "MEP ent=0.05 standard",
                "TrajeDi div=0.05 standard",
                "GAMMA mix25 MAPPO-RNN standard source",
                "PPO CNN state-aug",
                "PPO-E3T CNN state-aug predicted CE",
                "MEP ent=0.05 state-aug",
                "MEP ent=0.1 state-aug",
                "MAPPO RNN state-aug",
            ]
            sub = table[table["method"].isin(subset_names)].copy()
            sub["XP"] = pd.to_numeric(sub["XP"], errors="coerce")
            sub["SP"] = pd.to_numeric(sub["SP"], errors="coerce")
            sub = sub.sort_values(["state_aug", "XP"], ascending=[True, False])
            colors = np.where(sub["state_aug"].astype(str) == "yes", BLUE, ORANGE)
            ax_bar = inset(fig, [0.08, 0.18, 0.58, 0.58])
            y_pos = np.arange(len(sub))
            ax_bar.barh(y_pos, sub["XP"], color=colors)
            labels = [m.replace("PPO-E3T CNN ", "E3T ").replace("GAMMA mix25 ", "GAMMA ") for m in sub["method"]]
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(labels, fontsize=8.5)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("XP")
            ax_bar.set_title("Representative method XP (mixed historical configurations)")
            ax_bar.grid(axis="x", alpha=0.25)
            for yv, val in zip(y_pos, sub["XP"]):
                ax_bar.text(val + 2, yv, f"{val:.1f}", va="center", fontsize=8)
            add_text(ax, 0.70, 0.70, "阅读限制", size=14, weight="bold", color=RED)
            add_bullets(
                ax,
                0.70,
                0.63,
                [
                    "这张图用于研究地图，不是严格最终排名。",
                    "state-aug 与 standard 分布不同，不能直接说明方法本身强弱。",
                    "TTAC 主比较应固定同一个 PPO state-aug checkpoint，再看 online update 的增量。",
                ],
                size=10.8,
                width=34,
                gap=0.08,
            )
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "TTAC 结构：冻结 base，只在测试时更新 adapter", section="method")
        boxes = [
            (0.08, 0.63, "Observation", "局部观测 s"),
            (0.28, 0.63, "CNN base", "冻结特征和 base actor"),
            (0.50, 0.63, "Adapter", "只更新小参数 phi"),
            (0.72, 0.63, "Adapted policy", "pi_phi(a|s)"),
            (0.28, 0.35, "Partner/history signal", "partner history 或 estimator target"),
            (0.50, 0.35, "Online loss", "CE / AW / v5.2 target"),
        ]
        for x, y, h, b in boxes:
            ax.add_patch(plt.Rectangle((x, y), 0.16, 0.10, transform=ax.transAxes, facecolor="#F8FBFC", edgecolor=BLUE, lw=1.5))
            add_text(ax, x + 0.012, y + 0.078, h, size=11, weight="bold")
            add_text(ax, x + 0.012, y + 0.045, b, size=8.8, color=MUTED, width=20)
        for x1, y1, x2, y2 in [(0.24,0.68,0.28,0.68),(0.44,0.68,0.50,0.68),(0.66,0.68,0.72,0.68),(0.44,0.40,0.50,0.40),(0.58,0.45,0.58,0.63)]:
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
        add_text(ax, 0.08, 0.22, "设计动机：不直接更新 base policy，避免破坏已有 state-aug 协作能力；用 adapter 在测试时做局部修正。问题在于：online loss 必须真正指向有益修正，否则只是在随机扰动或调参。", size=12, width=90)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "TTAC 结果：v5.2 在 full eval 上有真实增益", section="ttac results")
        ax_bar = inset(fig, [0.08, 0.30, 0.50, 0.44])
        ax_bar.bar(ttac_full["mode"], ttac_full["XP"], color=[GRAY, BLUE, GREEN, PURPLE])
        ax_bar.set_ylabel("XP")
        ax_bar.set_title("Full eval: PPO state-aug base vs TTAC variants")
        ax_bar.tick_params(axis="x", rotation=18, labelsize=8.5)
        ax_bar.grid(axis="y", alpha=0.25)
        for i, v in enumerate(ttac_full["XP"]):
            ax_bar.text(i, v + 1.5, f"{v:.2f}", ha="center", fontsize=8.5)
        ax_tbl = inset(fig, [0.63, 0.30, 0.30, 0.44])
        show = ttac_full[["mode", "SP", "XP"]].copy()
        show["SP"] = show["SP"].map(lambda x: f"{x:.2f}")
        show["XP"] = show["XP"].map(lambda x: f"{x:.2f}")
        make_table(ax_tbl, show, col_widths=[0.50, 0.23, 0.23], font_size=8.0, scale_y=1.55)
        add_text(ax, 0.08, 0.18, "解释：v5.2 coef=20 的 full XP 为 165.768，比同 checkpoint 的 base no-adapt 高约 +9.00。TV-gate 0.03 full eval 几乎持平，没有形成比 latest 更稳定的优势。", size=11.5, width=90)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "v5.2 之前：Agreement 路线为什么看起来有希望", section="hypothesis")
        add_text(ax, 0.08, 0.76, "初始假设", size=15, weight="bold", color=BLUE)
        add_text(ax, 0.08, 0.70, "XP 高的方法往往伴随更好的 Agreement、更低 Policy TV、更低 OOS。因此我们希望在测试时估计 partner policy，并让 ego policy 朝当前 partner 的 convention 对齐。", size=12.3, width=86)
        add_text(ax, 0.08, 0.55, "后来发生的转折", size=15, weight="bold", color=RED)
        add_bullets(
            ax,
            0.08,
            0.48,
            [
                "estimator 没有提供强 partner-specific 语义：true / wrong / random / delayed history 的更新方向高度相似。",
                "v5.2 的 fixed-state audit 显示 Agreement 下降，Policy TV 上升。",
                "因此 v5.2 虽然有效，但不能作为 Agreement alignment 成功案例来写。",
            ],
            size=12.1,
            width=88,
            gap=0.085,
        )
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "Reward attribution：v5.2 不是少数 pair 拉高均值", section="mechanism")
        if not reward_attr.empty:
            vals = {r["metric"]: r["value"] for _, r in reward_attr.iterrows() if r["level"] in ["pair", "episode"]}
            metric_box(ax, 0.08, 0.74, 0.19, 0.15, "pair mean delta", f"{vals.get('mean_delta', 7.947):.2f}", "按 pair 平均后的 TTAC-base", BLUE)
            metric_box(ax, 0.31, 0.74, 0.19, 0.15, "pair positive rate", "85%", "100 个 pair 中 85 个均值提升", GREEN)
            metric_box(ax, 0.54, 0.74, 0.19, 0.15, "episode positive", "36.5%", "单局 reward 提升比例", ORANGE)
            metric_box(ax, 0.77, 0.74, 0.16, 0.15, "episode neutral", "47.4%", "离散 reward 导致大量持平", MUTED)
        img = ROOT / "reports/ttac_v5_2_mechanism_deep_dive_20260616/figures/reward_delta_distribution.png"
        if img.exists():
            ax_img = inset(fig, [0.12, 0.18, 0.76, 0.40])
            ax_img.imshow(plt.imread(img))
            ax_img.axis("off")
        add_text(ax, 0.08, 0.11, "定义：delta = reward(TTAC v5.2) - reward(base no-adapt)。positive pair rate 表示 pair 平均 delta > 0 的比例；episode positive rate 表示单个 pair-seed episode 的 delta > 0。", size=10.5, color=MUTED, width=94)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "Update-state attribution：正收益集中在 high-TV / low-value 状态", section="mechanism")
        if not update_attr.empty:
            show = update_attr[update_attr["field"].isin(["latest_target_base_tv", "base_value", "estimator_entropy", "partner_action_changed"])].copy()
            show = show[["field", "beneficial_mean", "harmful_mean", "mean_diff_beneficial_minus_harmful", "cohen_d_beneficial_vs_harmful"]]
            show.columns = ["feature", "beneficial", "harmful", "diff", "effect size"]
            for c in ["beneficial", "harmful", "diff", "effect size"]:
                show[c] = show[c].map(lambda x: f"{x:.3f}")
            ax_tbl = inset(fig, [0.08, 0.44, 0.84, 0.30])
            make_table(ax_tbl, show, col_widths=[0.30, 0.16, 0.16, 0.16, 0.16], font_size=9.2, scale_y=1.55)
        img = ROOT / "reports/ttac_v5_2_mechanism_deep_dive_20260616/figures/beneficial_vs_harmful_state_features.png"
        if img.exists():
            ax_img = inset(fig, [0.11, 0.12, 0.78, 0.27])
            ax_img.imshow(plt.imread(img))
            ax_img.axis("off")
        add_text(ax, 0.08, 0.79, "结果：beneficial pair 的 target-base TV 更高，base value 更低；estimator entropy 基本无差异，partner action change 不是正向触发信号。", size=12, width=88)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "Pair-level 相关性：target-base TV 与 base value 是强解释变量", section="mechanism")
        img1 = ROOT / "reports/ttac_v5_2_mechanism_deep_dive_20260616/figures/xp_delta_vs_target_base_tv.png"
        img2 = ROOT / "reports/ttac_v5_2_mechanism_deep_dive_20260616/figures/xp_delta_vs_base_value.png"
        if img1.exists():
            ax1 = inset(fig, [0.07, 0.30, 0.40, 0.40]); ax1.imshow(plt.imread(img1)); ax1.axis("off")
        if img2.exists():
            ax2 = inset(fig, [0.53, 0.30, 0.40, 0.40]); ax2.imshow(plt.imread(img2)); ax2.axis("off")
        corr_text = "target-base TV: Pearson 0.848, Spearman 0.711；base value: Pearson -0.730, Spearman -0.508。leave-one-pair-out 后仍稳定，说明不是单个 outlier 拉出的趋势。"
        add_text(ax, 0.08, 0.18, corr_text, size=12, width=91)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "Target-effect audit：XP 上升并不伴随 Agreement 上升", section="mechanism")
        if not target_effect.empty:
            img = ROOT / "reports/ttac_v5_2_mechanism_deep_dive_20260616/figures/target_effect_delta_metrics.png"
            if img.exists():
                ax_img = inset(fig, [0.10, 0.22, 0.80, 0.47])
                ax_img.imshow(plt.imread(img)); ax_img.axis("off")
            show = target_effect[["audit_mode", "delta_agreement", "delta_policy_tv", "delta_joint_optimal", "delta_joint_regret", "adapter_action_change_rate"]].copy()
            for c in show.columns[1:]:
                show[c] = show[c].map(lambda x: f"{x:.4f}")
            ax_tbl = inset(fig, [0.08, 0.07, 0.84, 0.13])
            make_table(ax_tbl, show, col_widths=[0.20, 0.16, 0.16, 0.16, 0.16, 0.18], font_size=7.5, scale_y=1.3)
        add_text(ax, 0.08, 0.75, "关键判读：如果 v5.2 真是在优化 Agreement，那么 update 后 Agreement 应该上升、Policy TV 应该下降。但观测结果相反。", size=12.5, width=90)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "Corrupted-history：为什么 wrong/random/delayed 也能涨 XP", section="mechanism")
        if not corrupted.empty:
            show = corrupted[["comparison", "target_tv_mean", "target_argmax_same_rate", "logit_grad_cos_mean", "action_change_overlap", "true_minus_mode_reward_gap", "true_mode_pair_corr"]].copy()
            show.columns = ["comparison", "target TV", "argmax same", "grad cosine", "action overlap", "reward gap", "pair corr"]
            for c in show.columns[1:]:
                show[c] = show[c].map(lambda x: "-" if pd.isna(x) else f"{x:.3f}")
            ax_tbl = inset(fig, [0.08, 0.45, 0.84, 0.24])
            make_table(ax_tbl, show, col_widths=[0.20, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13], font_size=8.2, scale_y=1.45)
        img = ROOT / "reports/ttac_v5_2_mechanism_deep_dive_20260616/figures/corrupted_history_similarity.png"
        if img.exists():
            ax_img = inset(fig, [0.12, 0.14, 0.76, 0.25])
            ax_img.imshow(plt.imread(img)); ax_img.axis("off")
        add_text(ax, 0.08, 0.76, "结论：corrupted history 有效不是因为它包含正确 partner 语义，而是因为它产生的 target 和梯度方向与 true history 高度重叠。", size=12.5, width=90)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "Action-level proxy：有益修正不是简单增加 interact", section="mechanism")
        action_diff = load_csv("reports/ttac_v5_2_mechanism_deep_dive_20260616/action_transition_diff.csv")
        if not action_diff.empty:
            changed = action_diff[(action_diff["transition_type"] == "base_argmax_to_target_argmax") & (action_diff["from_action"] != action_diff["to_action"])].copy()
            top = changed.sort_values("beneficial_minus_harmful_rate_diff", ascending=False).head(6)
            bottom = changed.sort_values("beneficial_minus_harmful_rate_diff", ascending=True).head(6)
            display = pd.concat([top.assign(group="beneficial more"), bottom.assign(group="harmful more")])
            display["shift"] = display["from_action"].astype(str) + " -> " + display["to_action"].astype(str)
            display = display[["group", "shift", "beneficial_minus_harmful_rate_diff"]]
            display.columns = ["where more common", "target shift", "rate diff"]
            display["rate diff"] = display["rate diff"].map(lambda x: f"{x:.4f}")
            ax_tbl = inset(fig, [0.12, 0.28, 0.76, 0.42])
            make_table(ax_tbl, display, col_widths=[0.25, 0.35, 0.18], font_size=9.0, scale_y=1.35)
        add_text(ax, 0.08, 0.75, "注意：这里是 base_argmax -> estimator_target_argmax 的 proxy，不是 adapter 更新后的真实 argmax。它提示有益修正可能包含减少无效 interact，而不是简单让策略更主动。", size=12.2, width=90)
        add_text(ax, 0.08, 0.16, "后续若要写成强机制，需要在 rollout 中保存 adapter update 前后的真实 action distribution / argmax。", size=11.5, color=RED, width=90)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "v6-gate / amp 失败：状态选择不是目标本身", section="negative results")
        if not v6.empty:
            ax_bar = inset(fig, [0.11, 0.33, 0.78, 0.36])
            v6s = v6.sort_values("xp_mean", ascending=True)
            ax_bar.barh(v6s["mode"], v6s["xp_mean"], color=[GRAY if "base" in m else BLUE if "v5" in m else RED for m in v6s["mode"]])
            ax_bar.set_xlabel("pair20 XP")
            ax_bar.set_title("v6.1 dynamic amp 低于 v5.2")
            ax_bar.grid(axis="x", alpha=0.25)
            for y, val in enumerate(v6s["xp_mean"]):
                ax_bar.text(val + 0.8, y, f"{val:.2f}", va="center", fontsize=8.5)
        add_text(ax, 0.08, 0.76, "解释：v6.1 仍然使用同一个 CE target，只是在不同状态上改变更新权重。它没有回答“应该更新到哪个动作方向”。结果变差，说明继续调 coef/gate/amp 没抓住机制。", size=12.2, width=90)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "现在最缺的证据：loss 应该朝哪个动作更新", section="gap")
        add_bullets(
            ax,
            0.08,
            0.76,
            [
                "我们已经有现象归因：v5.2 的收益集中在 high target-base TV / low base value 的局部状态。",
                "但还没有目标归因：没有证明 estimator target 指向的动作在这些状态下真的有更高 return。",
                "因此目前的 CE loss 只是粗糙地靠近 target，而不是严格优化 return-improving local correction。",
                "如果继续做 gate/coef/amp，就会卡在局部最优：围绕偶然有效的 target 打转，却没有获得更本质的 objective。",
            ],
            size=12.5,
            width=88,
            gap=0.095,
        )
        ax.add_patch(plt.Rectangle((0.12, 0.18), 0.76, 0.14, transform=ax.transAxes, facecolor=LIGHT_ORANGE, edgecolor=ORANGE, lw=1.4))
        add_text(ax, 0.15, 0.27, "关键问题", size=13, weight="bold", color=ORANGE)
        add_text(ax, 0.15, 0.23, "在同一个局部状态 s 下，base 偏好的动作 a_base 和 estimator target 偏好的动作 a_target，哪个会带来更高未来 return？", size=12.5, width=80)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "建议下一步：Counterfactual action-value attribution", section="next step")
        cols = pd.DataFrame(
            [
                ["Level 1", "相似状态经验 Q", "找相似 state/history 下不同动作的未来 return", "低成本，先验证方向"],
                ["Level 2", "训练 Qω(s,a,h)", "用已有 trajectories 预测 return-to-go", "可形成 return-aware loss"],
                ["Level 3", "短分叉 rollout", "从 saved state 强制 a_base/a_target 后继续 rollout", "最接近因果证据"],
            ],
            columns=["层级", "实验", "做什么", "用途"],
        )
        ax_tbl = inset(fig, [0.08, 0.42, 0.84, 0.28])
        make_table(ax_tbl, cols, col_widths=[0.12, 0.22, 0.40, 0.22], font_size=9.2, scale_y=1.75)
        add_text(ax, 0.08, 0.31, "成功标准：在 beneficial states 中 Q(a_target) > Q(a_base)，在 harmful states 中不成立；并且该差异能预测 pair/episode delta。只有这样，v6 才能从“调参”变成“return-aware local correction”。", size=12.2, width=90)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "论文叙事建议：不要强写 partner-specific Agreement", section="paper narrative")
        left = [
            "不建议继续写：我们估计 partner policy 并提升 Agreement。",
            "原因：当前 fixed-state audit 和 corrupted-history 对照都不支持。",
            "硬写会被审稿人抓住：Agreement 未提升、wrong history 也有效。",
        ]
        right = [
            "建议改写：ZSC 残余瓶颈包含测试时局部策略支持不足。",
            "TTAC adapter 提供一种不破坏 base 的局部修正通道。",
            "v5.2 是实证起点；最终方法需要 return-aware correction target。",
        ]
        ax.add_patch(plt.Rectangle((0.07, 0.23), 0.40, 0.48, transform=ax.transAxes, facecolor="#FFF5F5", edgecolor=RED, lw=1.2))
        ax.add_patch(plt.Rectangle((0.53, 0.23), 0.40, 0.48, transform=ax.transAxes, facecolor="#F1FAF5", edgecolor=GREEN, lw=1.2))
        add_text(ax, 0.10, 0.67, "避免的叙事", size=14, weight="bold", color=RED)
        add_bullets(ax, 0.10, 0.60, left, size=11.5, width=36, gap=0.09)
        add_text(ax, 0.56, 0.67, "更稳的叙事", size=14, weight="bold", color=GREEN)
        add_bullets(ax, 0.56, 0.60, right, size=11.5, width=36, gap=0.09)
        finish_page(pdf, fig, page)

        page += 1
        fig, ax = new_page(pdf, page, "报告结论：人工决策需要看的分叉", section="decision")
        add_text(ax, 0.08, 0.76, "我建议你现在用这份报告判断两个方向：", size=13, weight="bold")
        add_bullets(
            ax,
            0.08,
            0.68,
            [
                "如果坚持原始 Agreement 叙事：必须重做 partner estimator，让 true history 明显优于 wrong/random/delayed，并证明 Agreement 上升。",
                "如果接受当前正信号叙事：停止调 gate/amp，转向 return-aware local correction，先做反事实 action-value 归因。",
                "无论哪条线，都不要再把 pair20 screening 当成主结论；最终比较必须回到 full 90 XP pairings x 500 eval seeds。",
            ],
            size=12.5,
            width=90,
            gap=0.10,
        )
        add_text(ax, 0.08, 0.29, "当前更务实选择：以 v5.2 的 +9 XP full-eval 信号作为实证支点，但承认它不是 partner-specific Agreement；下一步用反事实 action-value 实验证明真正的 online update target。", size=13.2, color=BLUE, weight="bold", width=86)
        finish_page(pdf, fig, page)

    return PDF_PATH


if __name__ == "__main__":
    path = generate_report()
    print(path)
