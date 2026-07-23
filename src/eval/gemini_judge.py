"""
Official Gemini-as-Judge Evaluation Script
Matches the UDVideoQA paper's evaluation methodology.
Usage:
    export GEMINI_API_KEY="your_key_here"
    python src/eval/gemini_judge.py --input results/eval/all_models_comparison.csv \
                                    --output results/eval/official_scores.csv
"""

import csv
import os
import sys
import time
import argparse
import json
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    os.system("pip install -q google-generativeai")
    import google.generativeai as genai

# ── Strict grading prompt matching UDVideoQA paper ──────────────────────────
JUDGE_PROMPT = """You are a strict but fair grader for a traffic video question answering task.

Your job: decide if the MODEL ANSWER is correct given the GROUND TRUTH.

Rules:
1. Semantic equivalence counts as correct. Examples:
   - "olive-gray" = "taupe" = "beige-grey" → CORRECT (all earth-tone/neutral)
   - "sandy-beige" = "light stone" = "warm beige" → CORRECT
   - "white X" = "white cross marking" → CORRECT
   - "keep-clear box" ≠ "white X" → WRONG
2. If the model answer is non-committal ("I cannot determine", "unclear") → WRONG
3. If the model answer contradicts the ground truth → WRONG
4. Partial answers: if the core fact is correct, mark CORRECT
5. Extra irrelevant detail in the model answer is OK if the core is right

Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Respond with exactly one word: CORRECT or WRONG"""


def judge_answer(model, question: str, model_answer: str, ground_truth: str) -> int:
    """Returns 1 if correct, 0 if wrong."""
    prompt = JUDGE_PROMPT.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        model_answer=model_answer.strip()
    )
    try:
        response = model.generate_content(prompt)
        verdict = response.text.strip().upper()
        return 1 if "CORRECT" in verdict else 0
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        time.sleep(5)
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/path/to/your/data/results/eval/all_models_comparison.csv")
    parser.add_argument("--output", default="/path/to/your/data/results/eval/official_scores.csv")
    parser.add_argument("--limit",  type=int, default=None, help="Limit rows for testing (e.g. 20)")
    parser.add_argument("--model_filter", default=None, help="Only judge this model name")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        print("  export GEMINI_API_KEY='your_key_here'")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")  # faster + cheaper than 2.5 Pro
    print(f"Using model: gemini-2.0-flash")

    # Read input
    rows = list(csv.DictReader(open(args.input)))
    if args.model_filter:
        rows = [r for r in rows if args.model_filter in r["model"]]
    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"Judging {total} rows...")

    # Judge each row
    results = []
    stats = {}  # model -> {correct, total}

    for i, row in enumerate(rows):
        score = judge_answer(model, row["question"], row["generated_answer"], row["actual_answer"])
        results.append({**row, "judge_score": score})

        m = row["model"]
        if m not in stats:
            stats[m] = {"correct": 0, "total": 0}
        stats[m]["correct"] += score
        stats[m]["total"] += 1

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] Running totals:")
            for mn, s in stats.items():
                pct = s['correct']/s['total']*100 if s['total'] > 0 else 0
                print(f"    {mn}: {s['correct']}/{s['total']} = {pct:.1f}%")

        # Rate limiting — 15 req/min for free tier
        time.sleep(4)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Final summary
    print("\n" + "="*55)
    print("=== OFFICIAL RESULTS (Gemini-as-Judge) ===")
    print("="*55)
    for mn, s in stats.items():
        pct = s['correct']/s['total']*100 if s['total'] > 0 else 0
        print(f"  {mn:<35} {s['correct']:>4}/{s['total']:<4} = {pct:.1f}%")
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
