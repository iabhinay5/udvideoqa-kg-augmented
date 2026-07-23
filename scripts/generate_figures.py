# -*- coding: utf-8 -*-
"""
generate_figures.py — UDVideoQA KG-Augmented VideoQA
Generates all publication-quality figures for the capstone report.

Figures produced:
  Figure 1  — Per-category accuracy: Ours vs Paper (Qwen2.5-32B), bar chart
  Figure 2  — Ablation ladder: incremental KG component contribution
  Figure 3  — Per-set accuracy heatmap (supplementary)
  Figure 4  — Groq variability note figure (optional)

Usage:
  python scripts/generate_figures.py

Output:
  figures/fig1_per_category.pdf  (+ .png)
  figures/fig2_ablation.pdf      (+ .png)
  figures/fig3_heatmap.pdf       (+ .png)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/sync_for_report")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# Use a clean, publication-grade style
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
})

# ── Exact paper numbers (Qwen2.5-32B, Morning condition, Table 3) ─────────────
# Source: UDVideoQA paper Table 3 — extracted from PDF
PAPER_NUMBERS = {
    "BU":  75.66,
    "Atr": 36.11,
    "ER":  66.67,
    "RR":  25.00,
    "CI":  77.78,
    "Overall": 56.24,
}

# ── Colour palette (light-background, print-safe) ────────────────────────────
COL_PAPER = "#5B7FA6"   # Steel blue  — paper baseline
COL_OURS  = "#2E8B57"   # Sea green   — our method
COL_POS   = "#2E8B57"   # positive delta
COL_NEG   = "#C0392B"   # negative delta (ER regression)
COL_A     = "#9E9E9E"   # condition A (no KG)
COL_B     = "#64B5F6"   # condition B (+ colour)
COL_C     = "#1565C0"   # condition C (+ road marking) — best
COL_D     = "#EF5350"   # condition D (+ YOLO) — regresses


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Per-category accuracy: Ours vs Paper
# ─────────────────────────────────────────────────────────────────────────────
def fig1_per_category():
    print("Generating Figure 1 — Per-category accuracy...")

    df = pd.read_csv(RESULTS_DIR / "overall_judged.csv")
    by_qt = (
        df.groupby("qtype")
          .agg(total=("judge_score", "count"), correct=("judge_score", "sum"))
          .assign(acc=lambda x: x.correct / x.total * 100)
    )

    # Order: BU, Atr, ER, RR, CI + Overall
    order    = ["BU", "Atr", "ER", "RR", "CI"]
    our_vals = [by_qt.loc[q, "acc"] for q in order]
    pap_vals = [PAPER_NUMBERS[q] for q in order]

    # Append overall
    our_overall = df["judge_score"].sum() / len(df) * 100
    pap_overall = PAPER_NUMBERS["Overall"]

    labels    = ["BU", "Attribution", "Event\nReasoning", "Reverse\nReasoning", "Counterfactual\nInference", "Overall"]
    ours_all  = our_vals + [our_overall]
    paper_all = pap_vals + [pap_overall]
    deltas    = [o - p for o, p in zip(ours_all, paper_all)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars_paper = ax.bar(x - width/2, paper_all, width, color=COL_PAPER,
                        label="Qwen2.5-32B (Paper best)", zorder=3,
                        edgecolor="white", linewidth=0.5)
    bars_ours  = ax.bar(x + width/2, ours_all,  width, color=COL_OURS,
                        label="Ours: Qwen2.5-3B + KG (10× smaller)", zorder=3,
                        edgecolor="white", linewidth=0.5)

    # Delta annotations above each pair
    for i, (d, ou) in enumerate(zip(deltas, ours_all)):
        color  = COL_POS if d >= 0 else COL_NEG
        prefix = "+" if d >= 0 else ""
        ax.text(x[i] + width/2, ou + 1.5,
                f"{prefix}{d:.1f}pp",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

    # Vertical separator before "Overall"
    ax.axvline(x=len(order) - 0.5, color="#AAAAAA", linewidth=1.0, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")

    # Annotation box
    ax.text(0.98, 0.96,
            "Morning condition only\n(5 surveillance sets, 4 058 QA pairs)\nJudge: LLaMA-3.3-70B via Groq",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", alpha=0.9))

    ax.set_title("Figure 1  |  Per-Category Accuracy: KG-Augmented Qwen2.5-3B vs. Paper Baseline",
                 pad=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_per_category.pdf")
    fig.savefig(OUT_DIR / "fig1_per_category.png")
    plt.close(fig)
    print("  -> figures/fig1_per_category.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Ablation ladder (controlled FT model)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_ablation():
    print("Generating Figure 2 — Ablation ladder...")

    df = pd.read_csv(RESULTS_DIR / "ablation_FT_judged.csv")
    abl = (
        df.groupby("condition")
          .agg(total=("judge_score", "count"), correct=("judge_score", "sum"))
          .assign(acc=lambda x: x.correct / x.total * 100)
    )

    cond_order  = ["A_FT", "B_ColorOnly_FT", "C_ColorRoad_FT", "D_FullKGv3"]
    cond_labels = [
        "A: No KG\n(Qwen FT baseline)",
        "B: + Building Colour\n(pixel only)",
        "C: + Road Marking\n(pixel only)",
        "D: Full KG v3\n(+ YOLO objects)",
    ]
    acc_vals = [abl.loc[c, "acc"] for c in cond_order]
    colors   = [COL_A, COL_B, COL_C, COL_D]
    hatches  = ["", "", "", "///"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(cond_labels, acc_vals, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8, zorder=3,
                  hatch=hatches)
    # Hatching on D (red) to signal regression
    bars[3].set_edgecolor("#C0392B")
    bars[3].set_linewidth(1.2)

    # Accuracy labels on bars
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222222")

    # Delta arrows between consecutive bars
    arrow_props = dict(arrowstyle="-|>", color="#444444", lw=1.3)
    for i in range(len(acc_vals) - 1):
        x_from = i + 0.30
        x_to   = i + 0.70
        y_from = acc_vals[i] + 2
        y_to   = acc_vals[i+1] + 2
        delta  = acc_vals[i+1] - acc_vals[i]
        col_d  = COL_POS if delta >= 0 else COL_NEG
        prefix = "+" if delta >= 0 else ""
        ax.annotate("",
                    xy=(x_to, y_to), xytext=(x_from, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=col_d,
                                   lw=1.5, mutation_scale=14))
        mid_x = (x_from + x_to) / 2
        mid_y = max(y_from, y_to) + 2.5
        ax.text(mid_x, mid_y, f"{prefix}{delta:.1f}pp",
                ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=col_d)

    # Annotate D bar with explanation
    ax.text(3, acc_vals[3] - 4.5,
            "YOLO adds irrelevant\nobject facts -> noise",
            ha="center", va="top", fontsize=8.5, color="#C0392B",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0EE",
                      edgecolor="#C0392B", alpha=0.9))

    # Paper Qwen2.5-32B baseline line
    ax.axhline(PAPER_NUMBERS["Atr"], color="#5B7FA6", linewidth=1.4,
               linestyle="--", zorder=2, label=f"Paper best Atr: {PAPER_NUMBERS['Atr']:.2f}% (Qwen2.5-32B)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#CCCCCC", fontsize=9)

    ax.set_ylabel("Attribution Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.set_title("Figure 2  |  Ablation Study — Contribution of Each KG Component (Attribution, FT Model)",
                 pad=10, fontweight="bold")

    # Summary box
    summary = ("Controlled: all conditions use the same fine-tuned model\n"
               "403 Attribution questions, 5 morning surveillance sets")
    ax.text(0.98, 0.04, summary, transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_ablation.pdf")
    fig.savefig(OUT_DIR / "fig2_ablation.png")
    plt.close(fig)
    print("  -> figures/fig2_ablation.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Per-set heatmap (supplementary)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_heatmap():
    print("Generating Figure 3 — Per-set heatmap...")

    df = pd.read_csv(RESULTS_DIR / "overall_judged.csv")
    piv = (
        df.groupby(["set", "qtype"])
          .agg(correct=("judge_score", "sum"), total=("judge_score", "count"))
          .assign(acc=lambda x: x.correct / x.total * 100)
          .reset_index()
          .pivot(index="set", columns="qtype", values="acc")
    )
    # Reorder columns
    piv = piv[["BU", "Atr", "ER", "RR", "CI"]]
    piv.index = [s.replace("Set_", "Set ") for s in piv.index]

    import matplotlib.colors as mcolors
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=11)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=10)
    ax.set_xlabel("Question Type")
    ax.set_ylabel("Surveillance Set")

    # Annotate cells
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv.values[i, j]
            text_color = "white" if val < 35 or val > 75 else "black"
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.85)
    ax.set_title("Figure 3  |  Per-Set Accuracy Heatmap — KG-Augmented Qwen2.5-3B",
                 pad=10, fontweight="bold")
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_heatmap.pdf")
    fig.savefig(OUT_DIR / "fig3_heatmap.png")
    plt.close(fig)
    print("  -> figures/fig3_heatmap.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Ablation both runs comparison (ZeroShot vs FT backbone)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_ablation_comparison():
    print("Generating Figure 4 — Ablation: ZeroShot vs FT backbone...")

    df_ft = pd.read_csv(RESULTS_DIR / "ablation_FT_judged.csv")
    df_zs = pd.read_csv(RESULTS_DIR / "ablation_judged.csv")

    def get_accs(df, cond_order):
        abl = (df.groupby("condition")
                 .agg(total=("judge_score", "count"), correct=("judge_score", "sum"))
                 .assign(acc=lambda x: x.correct / x.total * 100))
        return [abl.loc[c, "acc"] for c in cond_order]

    cond_order_ft = ["A_FT",       "B_ColorOnly_FT", "C_ColorRoad_FT", "D_FullKGv3"]
    cond_order_zs = ["A_ZeroShot", "B_ColorOnly",    "C_ColorRoad",    "D_FullKGv3"]
    labels = ["No KG", "+ Bldg Colour", "+ Road Marking", "+ YOLO"]

    acc_ft = get_accs(df_ft, cond_order_ft)
    acc_zs = get_accs(df_zs, cond_order_zs)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(x - width/2, acc_zs, width, color="#78909C", label="ZeroShot backbone",
           zorder=3, edgecolor="white")
    ax.bar(x + width/2, acc_ft, width, color="#1565C0", label="Fine-Tuned backbone",
           zorder=3, edgecolor="white")

    for i, (vz, vf) in enumerate(zip(acc_zs, acc_ft)):
        ax.text(x[i] - width/2, vz + 0.8, f"{vz:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color="#444444")
        ax.text(x[i] + width/2, vf + 0.8, f"{vf:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color="#1565C0",
                fontweight="bold")

    ax.axhline(PAPER_NUMBERS["Atr"], color="#5B7FA6", linewidth=1.3,
               linestyle="--", label=f"Paper best Atr: {PAPER_NUMBERS['Atr']:.2f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Attribution Accuracy (%)")
    ax.set_ylim(0, 95)
    ax.legend(framealpha=0.9, edgecolor="#CCCCCC", fontsize=9)
    ax.set_title("Figure 4  |  Ablation Consistency Check — ZeroShot vs Fine-Tuned Backbone",
                 pad=10, fontweight="bold")

    ax.text(0.98, 0.96,
            "C > D holds in both backbone variants\n-> YOLO noise is backbone-agnostic",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_ablation_comparison.pdf")
    fig.savefig(OUT_DIR / "fig4_ablation_comparison.png")
    plt.close(fig)
    print("  -> figures/fig4_ablation_comparison.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig1_per_category()
    fig2_ablation()
    fig3_heatmap()
    fig4_ablation_comparison()
    print("\nAll figures saved to figures/")
