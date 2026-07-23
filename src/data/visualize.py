"""
UDVideoQA Data Visualizer
--------------------------
Generates charts and visualizations for:
1. QA distribution per reasoning type
2. Performance breakdown by condition (morning/evening/etc)
3. Attribution accuracy heatmap across conditions
4. Word clouds of questions

Usage:
    python src/data/visualize.py --data data/processed/test.jsonl
    python src/data/visualize.py --results results/baseline_results.json
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
REASONING_TYPES = ["BU", "Atr", "ER", "RR", "CI"]
CONDITIONS = ["morning", "midday", "evening", "nighttime"]
WEIGHTS = {"BU": 1.0, "Atr": 1.2, "ER": 1.3, "RR": 1.3, "CI": 1.5}

# ── Chart 1: QA Distribution ───────────────────────────
def plot_qa_distribution(df, save_dir="results/plots"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("UDVideoQA — Dataset Distribution", fontsize=15, fontweight="bold")

    # By reasoning type
    rt_counts = df["reasoning_type"].value_counts()
    rt_counts = rt_counts.reindex(REASONING_TYPES, fill_value=0)
    axes[0].bar(rt_counts.index, rt_counts.values, color=PALETTE)
    axes[0].set_title("QA Pairs by Reasoning Type")
    axes[0].set_ylabel("Count")
    for i, (rtype, count) in enumerate(rt_counts.items()):
        axes[0].text(i, count + 50, f"{count:,}", ha="center", va="bottom", fontsize=9)

    # By condition
    cond_counts = df["condition"].value_counts()
    cond_counts = cond_counts.reindex(CONDITIONS, fill_value=0)
    wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
    axes[1].pie(cond_counts.values, labels=cond_counts.index,
                colors=PALETTE, autopct="%1.1f%%",
                wedgeprops=wedge_props, startangle=90)
    axes[1].set_title("QA Pairs by Time of Day (Condition)")

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = f"{save_dir}/qa_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"✅ Saved: {out}")
    plt.show()

# ── Chart 2: Accuracy Heatmap ─────────────────────────
def plot_accuracy_heatmap(results_data, save_dir="results/plots"):
    """
    results_data: dict like {
      "morning": {"BU": 0.75, "Atr": 0.22, ...},
      "midday":  {...},
      ...
    }
    """
    # Build DataFrame
    rows = []
    for condition in CONDITIONS:
        row = {}
        for rt in REASONING_TYPES:
            row[rt] = results_data.get(condition, {}).get(rt, None)
        rows.append(row)

    df_heat = pd.DataFrame(rows, index=CONDITIONS)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Accuracy"},
    )
    ax.set_title("Accuracy Heatmap: Reasoning Type × Time of Day\n(green=high, red=low)", pad=12)
    ax.set_xlabel("Reasoning Type")
    ax.set_ylabel("Time of Day Condition")

    # Annotate the problem area
    morning_idx = CONDITIONS.index("morning")
    atr_idx = REASONING_TYPES.index("Atr")
    ax.add_patch(plt.Rectangle((atr_idx, morning_idx), 1, 1,
                                fill=False, edgecolor='blue', lw=3, label="Key problem area"))

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = f"{save_dir}/accuracy_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"✅ Saved: {out}")
    plt.show()

# ── Chart 3: Model Comparison Bar Chart ──────────────
def plot_model_comparison(models_data, save_dir="results/plots"):
    """
    models_data: dict like {
      "Qwen2.5-VL 7B (zero-shot)":  {"overall": 0.48, "Atr": 0.22},
      "Qwen2.5-VL 7B (fine-tuned)": {"overall": 0.61, "Atr": 0.45},
      "Our Method":                  {"overall": 0.71, "Atr": 0.60},
    }
    """
    models = list(models_data.keys())
    metrics = ["overall", "BU", "Atr", "ER", "RR", "CI"]

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (model_name, scores) in enumerate(models_data.items()):
        values = [scores.get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance Comparison — UDVideoQA Benchmark")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="50% line")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = f"{save_dir}/model_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"✅ Saved: {out}")
    plt.show()

# ── Chart 4: Paper Results (Hardcoded for reference) ──
def plot_paper_results(save_dir="results/plots"):
    """Reproduces the key result from the paper — morning Atr gap"""
    # From Table 3 of the paper (approximate values)
    paper_results = {
        "morning": {"BU": 0.80, "Atr": 0.22, "ER": 0.55, "RR": 0.60, "CI": 0.92},
        "midday":  {"BU": 0.78, "Atr": 0.35, "ER": 0.52, "RR": 0.58, "CI": 0.88},
        "evening": {"BU": 0.72, "Atr": 0.30, "ER": 0.48, "RR": 0.55, "CI": 0.85},
        "nighttime":{"BU": 0.65, "Atr": 0.28, "ER": 0.44, "RR": 0.50, "CI": 0.80},
    }
    plot_accuracy_heatmap(paper_results, save_dir)

# ── Main ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                        help="Path to JSONL dataset file")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to model results JSON file")
    parser.add_argument("--paper-baseline", action="store_true",
                        help="Plot paper baseline results (hardcoded)")
    parser.add_argument("--save-dir", default="results/plots")
    args = parser.parse_args()

    if args.data:
        records = [json.loads(l) for l in open(args.data)]
        df = pd.DataFrame(records)
        plot_qa_distribution(df, args.save_dir)

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        plot_accuracy_heatmap(results.get("by_condition_and_type", {}), args.save_dir)

    if args.paper_baseline:
        print("📊 Plotting paper baseline results...")
        plot_paper_results(args.save_dir)

    if not any([args.data, args.results, args.paper_baseline]):
        print("Usage: python visualize.py --paper-baseline")
        print("       python visualize.py --data data/processed/test.jsonl")

if __name__ == "__main__":
    main()
