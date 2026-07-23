"""
Analyze evaluation results and save structured report.
Run this after any eval CSV is complete.

Usage:
    python src/eval/analyze_results.py --csv results/eval/Qwen2.5-VL-3B-FT_morning_attribution.csv
    python src/eval/analyze_results.py --csv results/eval/Qwen2.5-VL-3B-ZeroShot_morning_attribution.csv
    python src/eval/analyze_results.py --compare  # Compare all CSVs in results/eval/
"""

import os
import csv
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict


RESULTS_DIR = "/path/to/your/data/results"
ANALYSIS_DIR = f"{RESULTS_DIR}/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def simple_score(pred: str, gt: str) -> float:
    pred, gt = pred.lower().strip(), gt.lower().strip()
    if pred == gt:
        return 1.0
    if gt in pred or pred in gt:
        return 1.0
    stop = {"the","a","an","is","are","it","in","on","of","and","or","there","no","not","was","were"}
    pred_words = set(pred.split()) - stop
    gt_words   = set(gt.split())   - stop
    if not gt_words:
        return 0.0
    overlap = pred_words & gt_words
    return 1.0 if len(overlap) / len(gt_words) >= 0.5 else 0.0


def analyze_csv(csv_path: str) -> dict:
    rows = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    model_name = rows[0].get("model", "unknown")
    results = []

    for r in rows:
        score = simple_score(r["generated_answer"], r["actual_answer"])
        results.append({**r, "score": score})

    total   = len(results)
    correct = sum(1 for r in results if r["score"] >= 0.5)
    wrong   = [r for r in results if r["score"] < 0.5]

    # Breakdown by set
    by_set = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        s = r.get("set", "unknown")
        by_set[s]["total"] += 1
        by_set[s]["correct"] += r["score"]

    # Most common wrong answers
    wrong_qs = Counter(r["question"] for r in wrong)

    analysis = {
        "model":         model_name,
        "csv":           csv_path,
        "total":         total,
        "correct":       correct,
        "accuracy":      round(correct / total, 4) if total else 0,
        "accuracy_pct":  f"{correct/total:.1%}" if total else "0%",
        "by_set": {
            s: {
                "total":    v["total"],
                "correct":  int(v["correct"]),
                "accuracy": f"{v['correct']/v['total']:.1%}" if v["total"] else "0%"
            }
            for s, v in sorted(by_set.items())
        },
        "top_failures": [
            {
                "question":  r["question"],
                "model_ans": r["generated_answer"],
                "actual":    r["actual_answer"],
                "set":       r.get("set",""),
            }
            for r in wrong[:20]
        ],
        "most_common_failing_questions": wrong_qs.most_common(10),
    }

    return analysis, results


def print_report(analysis: dict):
    print(f"\n{'='*65}")
    print(f"  ANALYSIS REPORT — {analysis['model']}")
    print(f"{'='*65}")
    print(f"  Total questions : {analysis['total']}")
    print(f"  Correct         : {analysis['correct']}")
    print(f"  Accuracy        : {analysis['accuracy_pct']}")
    print(f"\n  ── By Set ──────────────────────────────────────")
    for s, v in analysis["by_set"].items():
        bar = "█" * int(float(v["accuracy"].strip("%")) / 5)
        print(f"  {s}: {v['accuracy']:6s}  {bar}  ({v['correct']}/{v['total']})")
    print(f"\n  ── Top 5 Failures ──────────────────────────────")
    for i, f in enumerate(analysis["top_failures"][:5], 1):
        print(f"  {i}. [{f['set']}] Q: {f['question'][:60]}")
        print(f"     Model:  {f['model_ans'][:60]}")
        print(f"     Actual: {f['actual'][:60]}")
    print(f"{'='*65}\n")


def compare_all():
    """Compare all eval CSVs side by side."""
    eval_dir = f"{RESULTS_DIR}/eval"
    csvs = sorted(Path(eval_dir).glob("*.csv"))

    print(f"\n{'='*65}")
    print("  MODEL COMPARISON — Morning Attribution")
    print(f"{'='*65}")
    print(f"  {'Model':<35} {'Acc':>8} {'Correct':>8} {'Total':>7}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*7}")

    all_analyses = []
    for csv_path in csvs:
        result = analyze_csv(str(csv_path))
        if result:
            analysis, _ = result
            all_analyses.append(analysis)
            print(f"  {analysis['model']:<35} {analysis['accuracy_pct']:>8} {analysis['correct']:>8} {analysis['total']:>7}")

    print(f"{'='*65}\n")
    return all_analyses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to eval CSV to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare all CSVs")
    args = parser.parse_args()

    if args.compare:
        all_analyses = compare_all()
        out_path = f"{ANALYSIS_DIR}/comparison.json"
        with open(out_path, "w") as f:
            json.dump(all_analyses, f, indent=2)
        print(f"Saved comparison to {out_path}")
        return

    if args.csv:
        result = analyze_csv(args.csv)
        if result:
            analysis, rows = result
            print_report(analysis)

            # Save analysis JSON
            model_name = analysis["model"].replace("/", "_").replace(" ", "_")
            out_path = f"{ANALYSIS_DIR}/{model_name}_analysis.json"
            with open(out_path, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"Analysis saved to {out_path}")

            # Save failures CSV for inspection
            failures_path = f"{ANALYSIS_DIR}/{model_name}_failures.csv"
            with open(failures_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["question","model_ans","actual","set","score"])
                writer.writeheader()
                for fail in analysis["top_failures"]:
                    writer.writerow({
                        "question":  fail["question"],
                        "model_ans": fail["model_ans"],
                        "actual":    fail["actual"],
                        "set":       fail["set"],
                        "score":     0,
                    })
            print(f"Failures saved to {failures_path}")


if __name__ == "__main__":
    main()
