"""
rejudge_all.py — Re-judge ALL results in a single Groq session for consistency.

This script addresses the ~8pp judge variability issue by running a single
consolidated judging pass on all three result files using the same:
  - Judge model: llama-3.3-70b-versatile
  - Judge prompt template
  - Temperature: 0 (deterministic, but Groq may still vary slightly)
  - Session (one continuous API session)

Usage:
  export GROQ_API_KEY=gsk_...
  python scripts/rejudge_all.py

Output:
  results/sync_for_report/overall_rejudged.csv
  results/sync_for_report/ablation_FT_rejudged.csv
  results/sync_for_report/ablation_rejudged.csv
  results/sync_for_report/rejudge_summary.txt

NOTE: This will consume ~40,000 Groq tokens. Free tier limit = 500K/day
      for llama-3.3-70b-versatile. Should complete in ~30-45 minutes.
"""

import os
import time
import json
import pandas as pd
from pathlib import Path
from groq import Groq

# ── Config ────────────────────────────────────────────────
RESULTS_DIR = Path("results/sync_for_report")
JUDGE_MODEL = "llama-3.3-70b-versatile"
SLEEP_BETWEEN = 0.5   # seconds between calls (rate limit buffer)
BATCH_SIZE    = 10    # Log progress every N rows

JUDGE_SYSTEM = (
    "You are an expert evaluator for video question answering. "
    "Your task is to judge whether a predicted answer is semantically "
    "correct given the ground-truth answer. "
    "Respond with ONLY '1' if the prediction is correct or '0' if incorrect. "
    "Consider paraphrases, synonyms, and partial matches as correct. "
    "Do not explain your reasoning."
)

JUDGE_TEMPLATE = (
    "Question: {question}\n"
    "Ground Truth: {ground_truth}\n"
    "Prediction: {prediction}\n\n"
    "Is the prediction semantically correct? Answer 1 or 0:"
)

# ── Groq client ───────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def judge_one(question: str, ground_truth: str, prediction: str,
              retries: int = 3) -> int:
    """Call the judge and return 0 or 1. Retries on rate limit."""
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                max_tokens=2,
            )
            raw = resp.choices[0].message.content.strip()
            return 1 if raw.startswith("1") else 0
        except Exception as e:
            if "rate" in str(e).lower() and attempt < retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Judge error: {e} — defaulting to 0")
                return 0
    return 0


def rejudge_df(df: pd.DataFrame, out_path: Path, score_col: str = "judge_score"):
    """Re-judge a dataframe in-place and save."""
    print(f"\nRe-judging {len(df)} rows -> {out_path.name}")
    scores = []
    for i, row in enumerate(df.itertuples(index=False)):
        score = judge_one(
            question=str(row.question),
            ground_truth=str(row.actual_answer),
            prediction=str(row.generated_answer),
        )
        scores.append(score)
        time.sleep(SLEEP_BETWEEN)

        if (i + 1) % BATCH_SIZE == 0:
            running_acc = sum(scores) / len(scores) * 100
            print(f"  [{i+1}/{len(df)}]  running accuracy: {running_acc:.1f}%")

    df = df.copy()
    df[score_col] = scores
    df.to_csv(out_path, index=False)
    acc = sum(scores) / len(scores) * 100
    print(f"  Done. Accuracy: {acc:.2f}%  |  Saved to {out_path}")
    return acc


def main():
    summary = {}

    # 1. Overall judged (4,058 rows)
    df1 = pd.read_csv(RESULTS_DIR / "overall_judged.csv")
    acc1 = rejudge_df(df1, RESULTS_DIR / "overall_rejudged.csv")
    summary["overall"] = acc1

    # 2. Ablation FT (1,612 rows)
    df2 = pd.read_csv(RESULTS_DIR / "ablation_FT_judged.csv")
    acc2 = rejudge_df(df2, RESULTS_DIR / "ablation_FT_rejudged.csv")
    # Per-condition breakdown
    df2_out = pd.read_csv(RESULTS_DIR / "ablation_FT_rejudged.csv")
    abl_ft = (df2_out.groupby("condition")
              .agg(total=("judge_score","count"), correct=("judge_score","sum"))
              .assign(acc=lambda x: x.correct/x.total*100))
    summary["ablation_FT"] = abl_ft.to_dict()

    # 3. Ablation ZeroShot (1,612 rows)
    df3 = pd.read_csv(RESULTS_DIR / "ablation_judged.csv")
    acc3 = rejudge_df(df3, RESULTS_DIR / "ablation_rejudged.csv")
    df3_out = pd.read_csv(RESULTS_DIR / "ablation_rejudged.csv")
    abl_zs = (df3_out.groupby("condition")
              .agg(total=("judge_score","count"), correct=("judge_score","sum"))
              .assign(acc=lambda x: x.correct/x.total*100))
    summary["ablation_ZS"] = abl_zs.to_dict()

    # Summary report
    report_path = RESULTS_DIR / "rejudge_summary.txt"
    with open(report_path, "w") as f:
        f.write("=== RE-JUDGE SUMMARY (Single Session) ===\n\n")
        f.write(f"Judge model : {JUDGE_MODEL}\n")
        f.write(f"Temperature : 0\n\n")
        f.write(f"Overall accuracy : {acc1:.2f}%\n\n")
        f.write("Ablation FT:\n")
        f.write(abl_ft.to_string())
        f.write("\n\nAblation ZeroShot:\n")
        f.write(abl_zs.to_string())
    print(f"\nSummary written to {report_path}")
    print("\nRe-judging complete. Use *_rejudged.csv files for final report numbers.")


if __name__ == "__main__":
    main()
