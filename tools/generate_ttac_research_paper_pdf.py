from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage


ROOT = Path(os.environ.get("OV2_ROOT", "/teams/ius_1663576043/hby/rl/ov2"))
OUT_DIR = ROOT / "reports" / "ttac_research_paper_zh_20260617"
FIG_DIR = OUT_DIR / "figures"
PREVIEW_DIR = OUT_DIR / "page_previews"
PDF_PATH = OUT_DIR / "ttac_research_paper_zh_20260617.pdf"
MD_PATH = OUT_DIR / "ttac_research_paper_zh_20260617.md"

BLUE = "#256D85"
ORANGE = "#F18F01"
GREEN = "#4CAF50"
RED = "#C44545"
PURPLE = "#7E57C2"
GRAY = "#586069"
LIGHT = "#F5F8FA"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def font_path() -> str:
    candidates = [
        "/usr/share/fonts/truetype/SimHei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError("No CJK font found.")


def setup_fonts() -> str:
    path = font_path()
    fm.fontManager.addfont(path)
    name = fm.FontProperties(fname=path).get_name()
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    pdfmetrics.registerFont(TTFont("CJK", path))
    pdfmetrics.registerFont(TTFont("CJK-Bold", path))
    return name


def load_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(x, digits=2) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="CJK-Bold",
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "authors": ParagraphStyle(
            "authors",
            parent=base["Normal"],
            fontName="CJK",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor(GRAY),
            spaceAfter=14,
            wordWrap="CJK",
        ),
        "abstract_title": ParagraphStyle(
            "abstract_title",
            parent=base["Heading2"],
            fontName="CJK-Bold",
            fontSize=10.5,
            leading=13,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "abstract": ParagraphStyle(
            "abstract",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=9.2,
            leading=13.2,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            leftIndent=0.25 * cm,
            rightIndent=0.25 * cm,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="CJK-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=14,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="CJK-Bold",
            fontSize=11,
            leading=15,
            spaceBefore=10,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=9.4,
            leading=13.2,
            alignment=TA_JUSTIFY,
            firstLineIndent=0.55 * cm,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body_no_indent": ParagraphStyle(
            "body_no_indent",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=9.4,
            leading=13.2,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=8.2,
            leading=10.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor(GRAY),
            spaceBefore=4,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=8.2,
            leading=10.8,
            alignment=TA_LEFT,
            textColor=colors.HexColor(GRAY),
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "equation": ParagraphStyle(
            "equation",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=9.5,
            leading=13,
            alignment=TA_CENTER,
            leftIndent=0.4 * cm,
            rightIndent=0.4 * cm,
            spaceBefore=4,
            spaceAfter=6,
            backColor=colors.HexColor("#F7F9FA"),
            borderPadding=5,
            wordWrap="CJK",
        ),
    }


def para(text: str, styles: dict[str, ParagraphStyle], style: str = "body"):
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Paragraph(safe, styles[style])


def heading(text: str, styles: dict[str, ParagraphStyle], level: int = 1):
    return Paragraph(text, styles["h1" if level == 1 else "h2"])


def table_flowable(
    data: list[list[str]],
    col_widths: list[float] | None = None,
    font_size: float = 7.4,
    header_bg: str = "#EAF2F6",
) -> Table:
    tbl = Table(data, colWidths=col_widths, hAlign="CENTER", repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "CJK", font_size),
                ("FONT", (0, 0), (-1, 0), "CJK-Bold", font_size),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1B2631")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCD1D1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def add_caption(story: list, text: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(text, styles["caption"]))


def add_image(story: list, path: Path, width_cm: float, caption: str, styles: dict[str, ParagraphStyle]) -> None:
    if not path.exists():
        story.append(para(f"图像缺失：{path}", styles, "small"))
        return
    with PILImage.open(path) as pil:
        px_w, px_h = pil.size
    draw_w = width_cm * cm
    draw_h = draw_w * (px_h / max(px_w, 1))
    max_h = 15 * cm
    if draw_h > max_h:
        scale = max_h / draw_h
        draw_w *= scale
        draw_h *= scale
    img = Image(str(path), width=draw_w, height=draw_h)
    story.append(KeepTogether([img, Paragraph(caption, styles["caption"])]))


def select_rows(df: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for name in names:
        m = df[df["method"] == name]
        if not m.empty:
            rows.append(m.iloc[0])
    return pd.DataFrame(rows)


def qlearning_rows() -> pd.DataFrame:
    rows = []
    base = ROOT / "reports" / "qlearning_ov2_1zsc_eval_20260613_qlearning_eval_500"
    for method in ["iql", "vdn", "pqn_vdn"]:
        p = base / method / "summary.csv"
        if p.exists():
            df = pd.read_csv(p)
            if {"metric", "value"}.issubset(df.columns):
                vals = dict(zip(df["metric"], df["value"]))
                rows.append([method.upper(), vals.get("sp_mean", np.nan), vals.get("xp_mean", np.nan)])
            elif {"SP", "XP"}.issubset(df.columns):
                rows.append([method.upper(), df["SP"].mean(), df["XP"].mean()])
    return pd.DataFrame(rows, columns=["method", "SP", "XP"])


def build_figures(
    table: pd.DataFrame,
    update_attr: pd.DataFrame,
    target_effect: pd.DataFrame,
    corrupted: pd.DataFrame,
    v6: pd.DataFrame,
    ttac_full: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    def save(name: str) -> Path:
        p = FIG_DIR / name
        plt.tight_layout()
        plt.savefig(p, dpi=210, bbox_inches="tight", facecolor="white")
        plt.close()
        return p

    if not table.empty:
        names = [
            "PPO CNN standard",
            "PPO CNN state-aug",
            "PPO-E3T CNN no-state predicted CE",
            "PPO-E3T CNN state-aug predicted CE",
            "MEP ent=0.05 standard",
            "MEP ent=0.05 state-aug",
            "TrajeDi div=0.05 standard",
            "TrajeDi div=0.05 state-aug",
            "GAMMA mix25 MAPPO-RNN standard source",
            "MAPPO RNN state-aug",
        ]
        d = select_rows(table, names)
        if not d.empty:
            labels = [
                "PPO\nstd",
                "PPO\nstate",
                "E3T\nstd",
                "E3T\nstate",
                "MEP\nstd",
                "MEP\nstate",
                "TrajeDi\nstd",
                "TrajeDi\nstate",
                "GAMMA\nstd",
                "MAPPO\nstate",
            ][: len(d)]
            colors_list = [GRAY if "state" not in str(s) else BLUE for s in d["setting"]]
            plt.figure(figsize=(9.0, 3.6))
            plt.bar(np.arange(len(d)), d["XP"].astype(float), color=colors_list, alpha=0.88)
            plt.xticks(np.arange(len(d)), labels, fontsize=8)
            plt.ylabel("XP reward")
            plt.title("Representative XP across method families")
            plt.grid(axis="y", alpha=0.25)
            for i, v in enumerate(d["XP"].astype(float)):
                plt.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=7)
            paths["method_xp"] = save("method_xp_overview.png")

            plt.figure(figsize=(6.2, 4.3))
            x = table["OOS"].astype(float)
            y = table["XP"].astype(float)
            plt.scatter(x, y, s=42, alpha=0.75, c=table["Agreement"].astype(float), cmap="viridis")
            plt.colorbar(label="Agreement")
            plt.xlabel("OOS, lower means better coverage")
            plt.ylabel("XP reward")
            plt.title("XP vs OOS, colored by Agreement")
            plt.grid(alpha=0.25)
            paths["xp_oos_agreement"] = save("xp_oos_agreement.png")

    if not ttac_full.empty:
        plt.figure(figsize=(6.2, 3.6))
        colors_list = [GRAY, BLUE, GREEN, PURPLE]
        plt.bar(np.arange(len(ttac_full)), ttac_full["XP"], color=colors_list, alpha=0.88)
        plt.xticks(np.arange(len(ttac_full)), ["base", "v5.2\ncoef=1", "v5.2\ncoef=20", "TV-gate\ncoef=20"], fontsize=8)
        plt.ylabel("Full XP reward")
        plt.title("TTAC v5.2 full evaluation")
        plt.grid(axis="y", alpha=0.25)
        base = float(ttac_full["XP"].iloc[0])
        for i, v in enumerate(ttac_full["XP"]):
            plt.text(i, v + 1.1, f"{v:.2f}\n(+{v-base:.2f})" if i else f"{v:.2f}", ha="center", fontsize=7)
        paths["ttac_full"] = save("ttac_full_eval.png")

    if not update_attr.empty:
        feature_names = ["latest_target_base_tv", "base_value", "estimator_entropy", "partner_action_changed", "tv_gate_003"]
        d = update_attr[update_attr["field"].isin(feature_names)].copy()
        if not d.empty:
            present = [f for f in feature_names if f in set(d["field"])]
            d = d.set_index("field").loc[present]
            label_map = {
                "latest_target_base_tv": "target-base TV",
                "base_value": "base value",
                "estimator_entropy": "estimator entropy",
                "partner_action_changed": "partner action changed",
                "tv_gate_003": "TV gate 0.03",
            }
            labels = [label_map.get(s, s) for s in d.index]
            vals = d["cohen_d_beneficial_vs_harmful"].astype(float)
            colors_list = [GREEN if v > 0 else RED for v in vals]
            y = np.arange(len(vals))
            plt.figure(figsize=(9.2, 4.0))
            plt.barh(y, vals, color=colors_list, alpha=0.85)
            plt.axvline(0, color="black", lw=0.8)
            plt.yticks(y, labels, fontsize=10)
            plt.xlabel("Cohen d, beneficial vs harmful", fontsize=10)
            plt.title("Which update states are associated with reward gain", fontsize=12)
            plt.grid(axis="x", alpha=0.25)
            for i, v in enumerate(vals):
                offset = 0.015 if v >= 0 else -0.015
                ha = "left" if v >= 0 else "right"
                plt.text(v + offset, i, f"{v:.2f}", va="center", ha=ha, fontsize=10)
            paths["update_state"] = save("update_state_effect_size.png")

    if not target_effect.empty:
        fields = ["delta_agreement", "delta_policy_tv", "delta_joint_optimal", "delta_joint_regret"]
        vals = target_effect.set_index("audit_mode")[fields].T
        plt.figure(figsize=(7.0, 3.8))
        x = np.arange(len(fields))
        width = 0.35
        for j, col in enumerate(vals.columns):
            plt.bar(x + (j - 0.5) * width, vals[col], width=width, label=col)
        plt.axhline(0, color="black", lw=0.8)
        plt.xticks(x, ["Agreement", "Policy TV", "Joint optimal", "Joint regret"], fontsize=8)
        plt.ylabel("post update - base")
        plt.title("Fixed-state target-effect audit")
        plt.legend(fontsize=7)
        plt.grid(axis="y", alpha=0.25)
        paths["target_effect"] = save("target_effect_audit.png")

    if not corrupted.empty:
        labels = corrupted["comparison"].str.replace("true_vs_", "", regex=False).tolist()
        x = np.arange(len(labels))
        plt.figure(figsize=(7.0, 3.5))
        plt.plot(x, corrupted["logit_grad_cos_mean"], marker="o", label="gradient cosine")
        plt.plot(x, corrupted["action_change_overlap"], marker="s", label="action-change overlap")
        plt.plot(x, corrupted["true_mode_pair_corr"], marker="^", label="pair reward corr")
        plt.xticks(x, labels, fontsize=8)
        plt.ylim(0.55, 1.03)
        plt.ylabel("similarity / correlation")
        plt.title("Why corrupted history also works")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7)
        paths["corrupted"] = save("corrupted_history_similarity_article.png")

    if not v6.empty:
        plt.figure(figsize=(7.0, 3.4))
        labels = [
            "base",
            "v5.2 latest",
            "v5.2 TV",
            "v6 amp",
            "v6 soft",
            "v6 TV",
        ][: len(v6)]
        plt.bar(np.arange(len(v6)), v6["xp_mean"].astype(float), color=[GRAY, GREEN, BLUE, ORANGE, ORANGE, ORANGE][: len(v6)], alpha=0.9)
        plt.xticks(np.arange(len(v6)), labels, rotation=15, ha="right", fontsize=8)
        plt.ylabel("Pair20 XP")
        plt.title("v6.1 gate/amp variants did not beat v5.2")
        plt.grid(axis="y", alpha=0.25)
        for i, v in enumerate(v6["xp_mean"].astype(float)):
            plt.text(i, v + 0.9, f"{v:.1f}", ha="center", fontsize=7)
        paths["v6"] = save("v6_negative_pair20.png")

    return paths


def build_markdown(
    main_table: pd.DataFrame,
    q_table: pd.DataFrame,
    reward_attr: pd.DataFrame,
    update_attr: pd.DataFrame,
    update_corr: pd.DataFrame,
    target_effect: pd.DataFrame,
    corrupted: pd.DataFrame,
    v6: pd.DataFrame,
) -> str:
    def csv_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No data available._"
        cols = [str(c) for c in df.columns]
        body = []
        body.append("| " + " | ".join(cols) + " |")
        body.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            vals = []
            for c in df.columns:
                v = row[c]
                if isinstance(v, float):
                    vals.append(fmt(v, 4))
                else:
                    vals.append(str(v).replace("\n", " "))
            body.append("| " + " | ".join(vals) + " |")
        return "\n".join(body)

    lines = [
        "# TTAC 阶段性论文草稿",
        "",
        "题目：从诊断到测试时修正：Overcooked V2 零样本协作中的 TTAC 机制分析",
        "",
        "本文是中文研究草稿，不是最终英文投稿稿。它的目的不是包装结果，而是把已验证证据、失败证据和下一步分叉写清楚。",
        "",
        "## 核心结论",
        "",
        "1. state-aug 主要改善 coverage，即让 XP rollout 更少进入 self-play 支持外状态。",
        "2. Agreement 与 XP 在主诊断表中有关，但当前 TTAC v5.2 的 XP 增益并不是通过提高 Agreement 得到的。",
        "3. v5.2 的 full eval 有真实正增益，约为 +9 XP。",
        "4. corrupted history 也有效，原因不是错误历史有语义，而是 estimator 产生的更新方向高度重叠。",
        "5. 当前最可靠的解释是 local support refinement，而不是 partner-specific alignment。",
        "",
        "## 主方法表",
        "",
        csv_table(main_table),
        "",
        "## Q-learning 补充 baseline",
        "",
        csv_table(q_table),
        "",
        "## v5.2 reward attribution",
        "",
        csv_table(reward_attr),
        "",
        "## update-state attribution",
        "",
        csv_table(update_attr),
        "",
        "## pair-level correlation",
        "",
        csv_table(update_corr),
        "",
        "## target-effect audit",
        "",
        csv_table(target_effect),
        "",
        "## corrupted-history explanation",
        "",
        csv_table(corrupted),
        "",
        "## v6.1 negative result",
        "",
        csv_table(v6),
    ]
    return "\n".join(lines)


def method_table_for_pdf(df: pd.DataFrame) -> list[list[str]]:
    cols = ["method", "state_aug", "SP", "XP", "OOS", "Agreement", "Policy_TV", "Joint_regret"]
    out = [["方法", "state-aug", "SP", "XP", "OOS", "Agreement", "Policy TV", "Joint regret"]]
    for _, r in df.iterrows():
        out.append(
            [
                str(r.get("method", ""))[:34],
                str(r.get("state_aug", "")),
                fmt(r.get("SP")),
                fmt(r.get("XP")),
                fmt(r.get("OOS"), 3),
                fmt(r.get("Agreement"), 3),
                fmt(r.get("Policy_TV"), 3),
                fmt(r.get("Joint_regret"), 3),
            ]
        )
    return out


def make_doc(story: list, styles: dict[str, ParagraphStyle], figs: dict[str, Path], data: dict[str, pd.DataFrame]) -> None:
    story.append(Paragraph("从诊断到测试时修正：Overcooked V2 零样本协作中的 TTAC 机制分析", styles["title"]))
    story.append(Paragraph("阶段性中文论文草稿 - 供研究路线决策使用 - 2026-06-17", styles["authors"]))
    story.append(Paragraph("摘要", styles["abstract_title"]))
    story.append(
        Paragraph(
            "零样本协作要求一个智能体在测试时与未共同训练过的新伙伴合作。我们围绕 Overcooked V2 的 counter_circuit 布局，系统比较 PPO、E3T-style partner prediction、population-based 方法、GAMMA/MAPPO 以及一系列 test-time adaptation 方法。早期诊断显示，cross-play 失败可以分解为 coverage deficiency、shared-state convention mismatch 和 policy competence 三类因素。其中 state-aug 明显改善 coverage，但不能直接说明 convention mismatch 已被解决。基于这一观察，我们提出 TTAC 方向：在测试时只更新轻量 adapter，希望用实时交互证据改善协作。当前最强版本 v5.2 在 PPO CNN state-aug checkpoint 上把 full XP 从 156.764 提升到 165.768，约 +9.00。然而机制归因表明，该增益并不是成功的 Agreement alignment：固定状态审计中 Agreement 下降，Policy TV 上升，true/wrong/random/delayed history 的更新方向高度重叠。本文据此给出一个更谨慎的结论：当前 TTAC 正信号更像 local support refinement，即测试时在低 value、高 target-base TV 的局部状态中修正动作偏好。下一步不应继续调 gate/amp，而应通过反事实 action-value 或短分叉 rollout 直接学习哪些局部动作修正会提高回报。",
            styles["abstract"],
        )
    )
    story.append(
        Paragraph(
            "关键词：zero-shot coordination；Overcooked V2；test-time adaptation；policy agreement；support refinement；adapter policy",
            styles["abstract"],
        )
    )

    story.append(heading("1. 引言", styles))
    for text in [
        "多智能体协作中的一个核心问题是：一个策略能否与未见过、未共同训练过的伙伴合作。这个问题通常被称为 zero-shot coordination，简称 ZSC。本文中的 ZSC 评价主要使用 cross-play reward，简称 XP；self-play reward，简称 SP，只衡量策略和自身或同源策略合作的能力。",
        "我们最初的研究假设是：XP 较高的方法往往具有更好的状态覆盖、更高的 shared-state Agreement、更低的 Policy TV 或更低的 Joint regret。因此，如果能在测试时根据当前伙伴提升这些指标，可能就能提升 XP。这个假设是合理的，但后续实验显示它不能被直接等同为“只要优化 Agreement loss 就会提升 XP”。",
        "本文的目标不是给出最终投稿方法，而是把目前的实验证据整理成论文形式，帮助判断下一步路线。核心问题是：当前 TTAC 的正增益究竟来自 partner-specific adaptation，还是来自更通用的测试时局部修正？如果是后者，下一步的 loss 设计必须改变。",
    ]:
        story.append(para(text, styles))

    story.append(heading("2. 问题定义与指标", styles))
    definitions = [
        ("SP", "self-play reward，同一训练来源策略之间合作的平均 reward。SP 高说明基础任务能力强，但不保证能和新伙伴合作。"),
        ("XP", "cross-play reward，不同 seed、不同训练方式或不同 partner 之间合作的平均 reward。本文把 XP 作为 ZSC 的主要结果指标。"),
        ("state-aug", "rollout state-sampling 加 initial_state_buffer 的状态采样增强。这里必须强调：state-aug 不是 agent_view_size，不是视觉范围变大。"),
        ("OOS", "out-of-support rate，XP rollout 中进入 self-play 支持外状态的比例。OOS 越低，通常说明 coverage 越好。"),
        ("Agreement", "两个策略在同一 shared state、同一角色设定下动作选择的一致性。它近似衡量 convention 是否接近。"),
        ("Policy TV", "两个策略动作分布的 total variation distance。数值越低，说明策略分布越接近。"),
        ("Joint regret", "当前联合动作相对一步近似更优联合动作的 regret。数值越低，表示局部联合动作质量越好。"),
        ("target-base TV", "TTAC estimator 给出的 target distribution 与 base policy 的距离。数值越高表示更新目标想更强地推离 base。"),
    ]
    story.append(table_flowable([["术语", "本文定义"]] + definitions, col_widths=[3.25 * cm, 13.55 * cm], font_size=7.8))
    add_caption(story, "表 1：本文使用的主要术语。后文所有图表均使用这些定义。", styles)

    story.append(heading("3. 相关工作定位", styles))
    for text in [
        "Other-Play 类工作强调由于对称性和 convention mismatch，self-play 策略可能在 XP 中失败。Population-based 方法，例如 TrajeDi、MEP 和 FCP，通常试图在训练期产生多样策略或多样协作约定，从而提高与未知伙伴合作的机会。",
        "E3T、TALENTS 和 GAMMA 等 partner modeling 或 latent-strategy 方法试图利用伙伴行为信息。我们的实验曾尝试迁移这些思想，但在当前 OV2 设置下，简单 partner action prediction 或 latent conditioning 并没有稳定带来 state-aug 之上的增益。",
        "TTAC 的定位不同于直接训练一个 policy pool selector。我们希望最终产物是单模型，在测试时根据实时交互证据更新一个轻量 adapter。这个设定保留了 TTA/TTT 的科学问题：哪些信息只有测试时和当前伙伴交互后才可获得，并且这些信息如何转化为有效更新方向。",
    ]:
        story.append(para(text, styles))

    story.append(heading("4. 实验设置", styles))
    for text in [
        "主要环境是 Overcooked V2 的 counter_circuit 布局。这个布局并不是完全开放场景，而是通过柜台、锅、食材和上菜点制造协作约定差异。我们后来尝试过 open_cramped_room_v2，但该布局中 PPO 的 SP 和 XP 差距过小，暂时不适合作为 convention mismatch 主场景。",
        "主训练预算在大多数方法上采用 NUM_ENVS=64、NUM_MINIBATCHES=16、TOTAL_TIMESTEPS=10M、SEED=42、NUM_SEEDS=10。对于已完成的历史实验，本文只纳入当前主表中带完整 diagnostics 的结果。Q-learning/IQL/VDN/PQN 结果作为补充 baseline，因为它们尚未全部接入 OOS/Agreement/Policy TV/Joint regret 诊断表。",
        "TTAC 的主评估使用已有 PPO CNN state-aug checkpoint 的 posthoc zero-adapter 版本。也就是说，base policy 固定，测试时只更新 adapter。full eval 使用完整 pairing 与 500 eval seeds；pair20 只作为筛选实验，不作为最终结论。",
    ]:
        story.append(para(text, styles))

    story.append(heading("5. 主结果：state-aug 与方法家族对 XP 的影响", styles))
    main_rows = data["main_rows"]
    if not main_rows.empty:
        story.append(table_flowable(method_table_for_pdf(main_rows), col_widths=[4.2 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 2.0 * cm, 1.8 * cm, 1.9 * cm], font_size=6.8))
        add_caption(story, "表 2：主方法结果表。state-aug 表示 initial_state_buffer 状态采样增强，不表示视野参数。", styles)
    add_image(story, figs.get("method_xp", Path()), 15.7, "图 1：代表性方法的 XP。state-aug 对 PPO 的提升最明显，而 E3T-style predicted CE 在 state-aug 上没有带来稳定增益。", styles)
    add_image(story, figs.get("xp_oos_agreement", Path()), 12.0, "图 2：XP 与 OOS、Agreement 的关系。颜色越深表示 Agreement 越高。该图支持 coverage 和 convention 指标与 XP 有关，但不构成因果证明。", styles)
    for text in [
        "PPO CNN standard 的 XP 很低，而 PPO CNN state-aug 的 XP 显著提升。这支持第一个机制判断：coverage deficiency 是 counter_circuit 中的重要失败来源。换句话说，部分 XP 失败不是因为策略不知道如何合作，而是测试时进入了 self-play 很少覆盖的状态。",
        "E3T-style 方法在 no-state 设置下能给 XP 一定提升，但在 state-aug 已经改善 coverage 后，predicted CE 版本反而没有超过 PPO CNN state-aug。这说明 partner action prediction 在当前实现和当前环境中更像是补 coverage 或正则化，而不是直接解决剩余 convention mismatch。",
        "Population-based 方法，例如 MEP 和 TrajeDi，在部分设置下能获得不错 XP，但它们主要是训练期多样性方法。它们为我们提供分析参照，却不是本文希望最终采用的单模型 test-time adaptation 方案。",
    ]:
        story.append(para(text, styles))

    story.append(heading("6. TTAC 方法演化", styles))
    for text in [
        "TTAC 的基本结构是：obs 经过 CNN encoder 得到 base feature；base actor 输出 base logits；adapter 输出一个 bounded logit delta；最终 policy logits 等于 base logits 加 adapter delta。测试时冻结 encoder、base actor 和 critic，只更新 adapter。",
        "v1 到 v4 主要尝试 agreement loss、semantic residual、margin、direct-Q 等方向。多数尝试没有稳定证明 true history 明显优于 wrong/random/delayed history。v5 引入 estimator target 与 latest query state 的更新方式，开始出现可复现的正向 XP 信号。v5.1 的 multi-query 平均和 v6.1 的动态 gate/amp 都没有稳定超过 v5.2。",
        "v5.2 的关键不是训练一个新 base，而是在同一个 PPO CNN state-aug checkpoint 上，通过 test-time adapter update 改变局部动作偏好。因此它是一个更干净的 posthoc TTA 证据点。",
    ]:
        story.append(para(text, styles))
    story.append(Paragraph("pi_TTAC(a | o) = softmax(base_logits(o) + adapter_delta_phi(o, h_partner))", styles["equation"]))
    story.append(Paragraph("测试时只更新 phi；base encoder、base actor、critic 均冻结。", styles["small"]))

    story.append(heading("7. TTAC v5.2 的 full-eval 正增益", styles))
    ttac_full = data["ttac_full"]
    if not ttac_full.empty:
        story.append(
            table_flowable(
                [["模式", "SP", "XP", "说明"]]
                + [[str(r["mode"]), fmt(r["SP"]), fmt(r["XP"]), str(r["note"])] for _, r in ttac_full.iterrows()],
                col_widths=[4.2 * cm, 2.0 * cm, 2.0 * cm, 8.0 * cm],
                font_size=7.2,
            )
        )
        add_caption(story, "表 3：TTAC v5.2 full eval 结果。base no-adapt 与 TTAC 使用同一个 base checkpoint。", styles)
    add_image(story, figs.get("ttac_full", Path()), 12.5, "图 3：TTAC v5.2 在 full eval 上有真实增益。coef=20 的 latest 版本 XP 为 165.768，比 base no-adapt 高约 +9.00。", styles)
    story.append(para("这个结果很重要，因为它排除了“pair20 偶然变好”的一部分风险。v5.2 的 full-eval 正信号足以作为下一步研究的支点。但它还没有说明这个正信号来自我们最初设想的 Agreement alignment。", styles))

    story.append(heading("8. 机制归因一：v5.2 不是少数 pair 拉高均值", styles))
    reward_attr = data["reward_attr"]
    if not reward_attr.empty:
        selected = reward_attr[
            reward_attr["metric"].isin(["mean_delta", "positive_rate", "negative_rate", "neutral_rate", "beneficial_count"])
        ]
        story.append(
            table_flowable(
                [["level", "metric", "value"]]
                + [[str(r["level"]), str(r["metric"]), fmt(r["value"], 4)] for _, r in selected.iterrows()],
                col_widths=[3.0 * cm, 5.0 * cm, 3.0 * cm],
                font_size=7.4,
            )
        )
        add_caption(story, "表 4：Reward attribution。positive pair rate 表示按 pair 平均 delta 大于 0 的比例。episode positive rate 表示单个 pair-seed episode 的 delta 大于 0 的比例。", styles)
    deep_fig = ROOT / "reports" / "ttac_v5_2_mechanism_deep_dive_20260616" / "figures" / "reward_delta_distribution.png"
    add_image(story, deep_fig, 13.5, "图 4：pair-level 与 episode-level reward delta 分布。v5.2 的 pair-level 提升不是少数 pair 独自贡献，但 episode-level 仍有大量持平。", styles)
    story.append(para("这里的 delta 定义为 reward(TTAC v5.2) - reward(base no-adapt)。pair positive rate 为 85%，说明多数 pairing 的均值提升为正；episode neutral rate 接近 47.4%，说明许多 episode 中 TTAC 没有改变最终 reward。这提示我们：v5.2 不是强力全局改写策略，而是偶尔在关键局部状态产生修正。", styles))

    story.append(heading("9. 机制归因二：收益集中在高 target-base TV 和低 base value 状态", styles))
    update_attr = data["update_attr"]
    if not update_attr.empty:
        fields = ["latest_target_base_tv", "base_value", "estimator_entropy", "partner_action_changed", "tv_gate_003"]
        small = update_attr[update_attr["field"].isin(fields)]
        story.append(
            table_flowable(
                [["状态特征", "beneficial mean", "harmful mean", "mean diff", "Cohen d"]]
                + [
                    [
                        str(r["field"]),
                        fmt(r["beneficial_mean"], 4),
                        fmt(r["harmful_mean"], 4),
                        fmt(r["mean_diff_beneficial_minus_harmful"], 4),
                        fmt(r["cohen_d_beneficial_vs_harmful"], 3),
                    ]
                    for _, r in small.iterrows()
                ],
                col_widths=[4.5 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 2.2 * cm],
                font_size=7.1,
            )
        )
        add_caption(story, "表 5：Beneficial 与 harmful update 的状态特征差异。beneficial 表示该 episode 中 TTAC reward 高于 base。", styles)
    add_image(story, figs.get("update_state", Path()), 16.2, "图 5：beneficial 与 harmful update 的 effect size。正收益更集中于 target-base TV 更高、base value 更低的状态；partner action change 不是正向触发器。", styles)
    corr = data["update_corr"]
    if not corr.empty:
        corr_small = corr[corr["field"].isin(["latest_target_base_tv", "base_value", "partner_action_changed", "tv_gate_003"])]
        story.append(
            table_flowable(
                [["field", "Pearson with pair delta", "Spearman with pair delta", "LOO min", "LOO max"]]
                + [
                    [
                        str(r["field"]),
                        fmt(r["pearson_with_pair_delta"], 3),
                        fmt(r["spearman_with_pair_delta"], 3),
                        fmt(r["loo_pearson_min"], 3),
                        fmt(r["loo_pearson_max"], 3),
                    ]
                    for _, r in corr_small.iterrows()
                ],
                col_widths=[4.4 * cm, 3.4 * cm, 3.4 * cm, 2.4 * cm, 2.4 * cm],
                font_size=7.1,
            )
        )
        add_caption(story, "表 6：Pair-level delta 与状态特征的相关性。LOO 表示 leave-one-pair-out 稳定性检查。", styles)
    story.append(para("这组结果是当前最有价值的机制线索。target-base TV 与 pair delta 的 Pearson 相关约 0.848，base value 与 pair delta 负相关约 -0.730。相比之下，partner_action_changed 不是一个正向触发器。这说明正信号更像是在 base policy 不自信或局部价值较低时，把动作分布推向一个不同但仍可行的局部支持区域。", styles))

    story.append(heading("10. 机制归因三：Agreement 假设没有被当前 v5.2 实现", styles))
    target_effect = data["target_effect"]
    if not target_effect.empty:
        story.append(
            table_flowable(
                [["mode", "delta Agreement", "delta Policy TV", "delta Joint optimal", "delta Joint regret", "action change"]]
                + [
                    [
                        str(r["audit_mode"]),
                        fmt(r["delta_agreement"], 4),
                        fmt(r["delta_policy_tv"], 4),
                        fmt(r["delta_joint_optimal"], 4),
                        fmt(r["delta_joint_regret"], 4),
                        fmt(r["adapter_action_change_rate"], 4),
                    ]
                    for _, r in target_effect.iterrows()
                ],
                col_widths=[4.0 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm],
                font_size=7.0,
            )
        )
        add_caption(story, "表 7：Target-effect audit。若 v5.2 成功优化 Agreement，则 delta Agreement 应该为正，Policy TV 应该下降；实际相反。", styles)
    add_image(story, figs.get("target_effect", Path()), 13.0, "图 6：固定状态审计显示，v5.2 update 后 Agreement 下降，Policy TV 上升。", styles)
    story.append(para("这并不意味着 Agreement 对 XP 不重要。更准确的说法是：Agreement 可能是跨方法 XP 的解释变量之一，但当前 TTAC v5.2 的具体 loss 并没有真正优化 Agreement。因此，不能把 v5.2 的 +9 XP 写成“我们成功提升 Agreement 从而提升 XP”。这点如果混淆，会让论文主线站不住。", styles))

    story.append(heading("11. 机制归因四：为什么 corrupted history 也有效", styles))
    corrupted = data["corrupted"]
    if not corrupted.empty:
        story.append(
            table_flowable(
                [["comparison", "target TV", "target argmax same", "grad cosine", "action overlap", "reward gap", "pair corr"]]
                + [
                    [
                        str(r["comparison"]),
                        fmt(r["target_tv_mean"], 3),
                        fmt(r["target_argmax_same_rate"], 3),
                        fmt(r["logit_grad_cos_mean"], 3),
                        fmt(r["action_change_overlap"], 3),
                        fmt(r["true_minus_mode_reward_gap"], 3),
                        fmt(r["true_mode_pair_corr"], 3),
                    ]
                    for _, r in corrupted.iterrows()
                ],
                col_widths=[3.2 * cm, 2.3 * cm, 2.8 * cm, 2.3 * cm, 2.5 * cm, 2.3 * cm, 2.2 * cm],
                font_size=6.8,
            )
        )
        add_caption(story, "表 8：Corrupted-history explanation。wrong/random/delayed 与 true 的更新方向高度相似。", styles)
    add_image(story, figs.get("corrupted", Path()), 12.5, "图 7：true 与 corrupted history 的 gradient cosine、action-change overlap、pair reward correlation 均很高。", styles)
    story.append(para("wrong history 有效，并不是因为错误历史也包含正确 partner 语义。更合理的解释是：当前 estimator 输出的 target distribution 大多落在相似方向，导致 true/wrong/random/delayed 的 adapter 更新高度重叠。例如 true_vs_wrong 的 action-change overlap 约 0.989，pair reward correlation 约 0.988。这说明 estimator 没有提供足够强的 partner-specific 信号。", styles))

    story.append(heading("12. v6.1 的负结果：继续调 gate/amp 不是本质进展", styles))
    v6 = data["v6"]
    if not v6.empty:
        story.append(
            table_flowable(
                [["mode", "pair20 XP"]]
                + [[str(r["mode"]), fmt(r["xp_mean"])] for _, r in v6.iterrows()],
                col_widths=[8.0 * cm, 3.0 * cm],
                font_size=7.5,
            )
        )
        add_caption(story, "表 9：v6.1 pair20 结果。动态 amp/gate 没有超过 v5.2。", styles)
    add_image(story, figs.get("v6", Path()), 12.5, "图 8：v6.1 没有把正信号放大，说明继续调 update 强度或 gate 阈值不是当前瓶颈的本质解决方案。", styles)
    story.append(para("这组负结果应该被保留，因为它阻止我们继续在同一个局部最优里打转。v6.1 本质上仍是在选择何时更新、更新多强，而不是回答“应该把 policy 更新到哪个更高回报的动作方向”。", styles))

    story.append(heading("13. 讨论：当前 TTAC 的真实贡献与风险", styles))
    for text in [
        "当前 TTAC 线最强的正面证据是：v5.2 在 full eval 上提供了约 +9 XP，并且 pair-level positive rate 较高。这足以说明测试时 adapter update 可以带来实证增益。",
        "当前 TTAC 线最大的风险是：增益机制与最初的 Agreement narrative 不一致。我们最初希望在线估计 partner policy distribution，并据此提升 Agreement；但现有 estimator 没有做到这一点。",
        "因此，论文如果继续沿用 Agreement 叙事，就必须重新设计 partner estimator，并证明 true history 显著优于 corrupted history，且 update 后 Agreement 上升。否则，论文应转向 local support refinement 叙事，并诚实承认它不是 partner-specific alignment。",
        "local support refinement 并不等于 coverage。coverage 关注训练期或评估状态是否进入 self-play 支持外；support refinement 关注已经在测试轨迹里的局部状态中，base policy 是否停留在一个低回报但高概率的动作偏好上。它可能仍然需要测试时做，因为这些局部失败状态由当前 partner 的实时行为诱导，训练阶段无法枚举所有 partner-induced micro-state。",
    ]:
        story.append(para(text, styles))

    story.append(heading("14. 下一步实验建议", styles))
    story.append(heading("14.1 如果坚持 Agreement 主线", styles, level=2))
    for text in [
        "必须重做 partner estimator。目标不再是只预测单步 partner action，而是估计同一 shared state 下 partner policy distribution。仅从单次动作恢复分布很困难，因此需要 trajectory-level 或 episode-level latent representation，并通过 true/wrong/delayed 区分实验验证语义有效性。",
        "验收标准应很严格：true history 的 target distribution 必须明显不同于 corrupted history；true update 必须提升 Agreement 或降低 Policy TV；full eval 中 true 必须显著优于 wrong/random/delayed。",
    ]:
        story.append(para(text, styles))
    story.append(heading("14.2 如果接受 local support refinement 主线", styles, level=2))
    for text in [
        "不要继续调 gate/amp。下一步应构造 return-aware local correction surrogate。最直接的实验是对关键状态做短分叉 rollout：固定当前 episode prefix，对候选 ego action 做 1 到 N 步分叉，估计哪个 action correction 会提高 future return。",
        "然后训练一个 action-correction scorer，而不是继续用 partner action CE 间接更新 adapter。新的 test-time loss 应直接偏向高 predicted return 的 correction，同时用 KL 防止离 base policy 太远。",
    ]:
        story.append(para(text, styles))
    story.append(Paragraph("L(phi) = - E_a[ pi_phi(a | s) * Q_hat_local(s, a, h_partner) ] + lambda_KL * SKL(pi_phi, pi_base)", styles["equation"]))
    story.append(para("这个公式的含义是：adapter 不再盲目贴近 estimator 的动作分布，而是选择在当前局部状态下预计能提高未来回报的动作，同时通过 KL 约束避免破坏基础协作能力。", styles))

    story.append(heading("15. 结论", styles))
    for text in [
        "本文把目前实验整理为一个谨慎结论：TTAC v5.2 的 positive signal 是真实的，但还不是我们最初设想的 partner-specific Agreement alignment。",
        "最可靠的机制解释是 local support refinement：在低 base value、高 target-base TV 的局部状态中，测试时 adapter 把策略推向另一个可行支持区域，从而提升部分 pair 的 XP。",
        "下一步路线需要人工决策。如果追求最初的 Agreement 故事，就必须重新设计 partner estimator；如果追求当前已被实验证实的正信号，就应转向 return-aware local correction，并通过反事实 action-value 实验确定真正应该优化的目标。",
    ]:
        story.append(para(text, styles))

    story.append(heading("参考文献与写作占位", styles))
    refs = [
        "Carroll et al. On the Utility of Learning about Humans for Human-AI Coordination. NeurIPS, 2019.",
        "Hu et al. Other-Play for Zero-Shot Coordination. ICML, 2020.",
        "Lupu et al. Trajectory Diversity for Zero-Shot Coordination. ICML, 2021.",
        "Strouse et al. Collaborating with Humans without Human Data. NeurIPS, 2021.",
        "Sunehag et al. Value-Decomposition Networks For Cooperative Multi-Agent Learning. AAMAS, 2018.",
        "Tan. Multi-Agent Reinforcement Learning: Independent versus Cooperative Agents. ICML, 1993.",
        "E3T, GAMMA, TALENTS, MEP and Overcooked V2 citations need final BibTeX verification before English submission.",
    ]
    for r in refs:
        story.append(para(r, styles, "body_no_indent"))


def page_decorator(canvas, doc):
    canvas.saveState()
    canvas.setFont("CJK", 7.2)
    canvas.setFillColor(colors.HexColor("#7B7D7D"))
    canvas.drawString(1.7 * cm, 1.05 * cm, "TTAC research paper draft - 2026-06-17")
    canvas.drawRightString(A4[0] - 1.7 * cm, 1.05 * cm, str(doc.page))
    canvas.restoreState()


def render_previews() -> None:
    try:
        import fitz
    except Exception:
        return
    for old in PREVIEW_DIR.glob("page_*.png"):
        old.unlink()
    doc = fitz.open(str(PDF_PATH))
    for idx, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
        pix.save(str(PREVIEW_DIR / f"page_{idx:02d}.png"))
    doc.close()


def generate() -> None:
    ensure_dirs()
    setup_fonts()

    result_table = load_csv("reports/current_complete_results_with_diagnostics_20260523.csv")
    deep = "reports/ttac_v5_2_mechanism_deep_dive_20260616"
    reward_attr = load_csv(f"{deep}/reward_attribution.csv")
    update_attr = load_csv(f"{deep}/update_state_attribution.csv")
    update_corr = load_csv(f"{deep}/update_state_pair_correlations.csv")
    target_effect = load_csv(f"{deep}/target_effect_attribution.csv")
    corrupted = load_csv(f"{deep}/corrupted_history_explanation.csv")
    v6 = load_csv(f"{deep}/v6_1_negative_pair20.csv")

    method_names = [
        "PPO CNN standard",
        "PPO CNN state-aug",
        "PPO-E3T CNN no-state predicted CE",
        "PPO-E3T CNN state-aug predicted CE",
        "PPO-E3T CNN state-aug no CE",
        "MEP ent=0.05 standard",
        "MEP ent=0.05 state-aug",
        "MEP ent=0.1 state-aug",
        "TrajeDi div=0.05 standard",
        "TrajeDi div=0.05 state-aug",
        "GAMMA mix25 MAPPO-RNN standard source",
        "MAPPO RNN state-aug",
    ]
    main_rows = select_rows(result_table, method_names)
    q_rows = qlearning_rows()
    ttac_full = pd.DataFrame(
        [
            ["base no-adapt", 201.836, 156.764, "same PPO CNN state-aug checkpoint, no adapter update"],
            ["v5.2 latest coef=1", 201.436, 164.209, "full eval, all pairings, 500 eval seeds"],
            ["v5.2 latest coef=20", 200.620, 165.768, "best full true-history line"],
            ["v5.2 TV-gate 0.03 coef=20", 200.456, 165.748, "full eval, nearly tied with latest"],
        ],
        columns=["mode", "SP", "XP", "note"],
    )

    figs = build_figures(result_table, update_attr, target_effect, corrupted, v6, ttac_full)
    MD_PATH.write_text(
        build_markdown(main_rows, q_rows, reward_attr, update_attr, update_corr, target_effect, corrupted, v6),
        encoding="utf-8",
    )

    styles = make_styles()
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.65 * cm,
        rightMargin=1.65 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.55 * cm,
        title="TTAC research paper draft",
        author="OV2TTA",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin + 0.25 * cm, doc.width, doc.height - 0.35 * cm, id="normal")
    doc.addPageTemplates([PageTemplate(id="paper", frames=[frame], onPage=page_decorator)])

    story: list = []
    make_doc(
        story,
        styles,
        figs,
        {
            "main_rows": main_rows,
            "q_rows": q_rows,
            "reward_attr": reward_attr,
            "update_attr": update_attr,
            "update_corr": update_corr,
            "target_effect": target_effect,
            "corrupted": corrupted,
            "v6": v6,
            "ttac_full": ttac_full,
        },
    )
    doc.build(story)
    render_previews()
    print(PDF_PATH)
    print(MD_PATH)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    generate()
