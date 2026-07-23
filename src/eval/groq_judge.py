"""
Groq-based LLM Judge for UDVideoQA evaluation.
Uses LLaMA 3.3-70B via Groq free tier — 14,400 req/day, no billing needed.
Usage:
    pip install groq
    export GROQ_API_KEY="gsk_..."
    python src/eval/groq_judge.py \
        --input  results/eval/all_models_comparison.csv \
        --output results/eval/official_scores.csv
"""

import csv, os, sys, time, argparse
from pathlib import Path
from tqdm import tqdm

JUDGE_PROMPT = """You are evaluating a traffic video question answering system.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Answer: {model_answer}

Is the model's answer semantically correct? Apply these rules:
1. Color synonyms count as CORRECT:
   - "olive-gray", "taupe", "beige-grey", "sandy-beige", "light stone", "warm gray" = same earth-tone family
   - "gray", "dark gray", "charcoal", "slate" = same gray family
   - "white" when truth is "taupe/beige/gray/stone" = WRONG (different family)
2. Road markings: "keep-clear box" ≠ "white X marking" = WRONG
3. Non-committal answers ("I cannot determine", "unclear", "I don't know") = WRONG
4. If the core fact is correct but with extra detail = CORRECT
5. Partial answers containing the key fact = CORRECT

Reply with exactly one word: CORRECT or WRONG"""


def load_groq():
    try:
        from groq import Groq
    except ImportError:
        print("Installing groq...")
        os.system(f"{sys.executable} -m pip install -q groq")
        from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable first.")
        print("  Get free key at: https://console.groq.com")
        print("  export GROQ_API_KEY='gsk_...'")
        sys.exit(1)
    return Groq(api_key=api_key)


def judge_one(client, question, model_answer, ground_truth):
    prompt = JUDGE_PROMPT.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        model_answer=model_answer.strip()
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        return 1 if "CORRECT" in verdict else 0
    except Exception as e:
        print(f"\n  API error: {e}", file=sys.stderr)
        time.sleep(10)
        return -1  # -1 = retry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/path/to/your/data/results/eval/all_models_comparison.csv")
    parser.add_argument("--output", default="/path/to/your/data/results/eval/official_scores.csv")
    parser.add_argument("--limit",  type=int, default=None, help="Test with N rows first")
    parser.add_argument("--model_filter", default=None, help="Only judge rows for this model")
    args = parser.parse_args()

    client = load_groq()
    rows = list(csv.DictReader(open(args.input)))

    if args.model_filter:
        rows = [r for r in rows if args.model_filter in r["model"]]
    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"Judging {total} rows using LLaMA-3.3-70B via Groq...")

    results, stats = [], {}

    for i, row in enumerate(tqdm(rows)):
        # Retry up to 3 times
        for attempt in range(3):
            score = judge_one(client, row["question"], row["generated_answer"], row["actual_answer"])
            if score != -1:
                break
            time.sleep(15)
        if score == -1:
            score = 0

        results.append({**row, "judge_score": score, "judge_model": "llama-3.3-70b"})
        m = row["model"]
        stats.setdefault(m, {"correct": 0, "total": 0})
        stats[m]["correct"] += score
        stats[m]["total"] += 1

        # Progress every 50 rows
        if (i + 1) % 50 == 0:
            print(f"\n[{i+1}/{total}] Running totals:")
            for mn, s in sorted(stats.items()):
                pct = s['correct'] / s['total'] * 100
                print(f"  {mn:<40} {s['correct']:>4}/{s['total']:<4} = {pct:.1f}%")

        time.sleep(2)  # Groq free: 30 req/min = 2 sec between calls

    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 60)
    print("=== OFFICIAL RESULTS (LLaMA-3.3-70B Judge via Groq) ===")
    print("=" * 60)
    for mn, s in sorted(stats.items()):
        pct = s['correct'] / s['total'] * 100
        print(f"  {mn:<42} {s['correct']:>4}/{s['total']:<4} = {pct:.1f}%")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
