"""
Merge v3 evaluation CSVs and prepare for Groq judge.
Run this AFTER graphrag_eval_v3 completes.

Usage:
    python src/eval/merge_and_judge_v3.py
"""

import csv, os, sys, time
from pathlib import Path
from collections import defaultdict

BASE = Path("/path/to/your/data/results/eval")

V3_FILES = [
    ("E006_GraphRAGv3_Set_33.csv", "GraphRAG-KG-v3"),
    ("E006_GraphRAGv3_Set_34.csv", "GraphRAG-KG-v3"),
    ("E006_GraphRAGv3_Set_35.csv", "GraphRAG-KG-v3"),
]

V2_BASELINE_FILES = [
    ("Qwen2.5-VL-3B-ZeroShot_morning_attribution.csv", "Qwen2.5-VL-3B-ZeroShot"),
    ("Qwen2.5-VL-3B-FT_morning_attribution.csv",       "Qwen2.5-VL-3B-FT"),
]


def load_csv_as_unified(fname, model_name):
    """Load a results CSV into unified format."""
    path = BASE / fname
    if not path.exists():
        print(f"WARNING: {fname} not found, skipping.")
        return []
    
    rows = list(csv.DictReader(open(path)))
    unified = []
    for r in rows:
        # Detect answer column
        ans_col = None
        for col in ["answer_graphrag", "generated_answer"]:
            if col in r:
                ans_col = col
                break
        
        if not ans_col:
            print(f"WARNING: No answer column in {fname}")
            continue
        
        vid_col = next((c for c in r.keys() if "video" in c.lower()), None)
        
        unified.append({
            "question_id":      r.get("question_id", ""),
            "video_id":         r.get(vid_col, "") if vid_col else "",
            "set":              r.get("set", ""),
            "question":         r.get("question", ""),
            "model":            model_name,
            "generated_answer": r.get(ans_col, ""),
            "actual_answer":    r.get("actual_answer", ""),
            "kg_used":          r.get("kg_used", ""),
        })
    return unified


def run_judge(rows, output_path, api_key):
    """Run Groq judge on rows."""
    try:
        from groq import Groq
    except ImportError:
        print("pip install groq first")
        sys.exit(1)

    JUDGE_PROMPT = """You are evaluating a traffic video question answering system.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Answer: {model_answer}

Is the model's answer semantically correct? Apply these rules:
1. Color synonyms count as CORRECT:
   - "olive-gray", "taupe", "beige-grey", "sandy-beige", "light stone", "warm gray" = same earth-tone family
   - "gray", "dark gray", "charcoal", "slate" = same gray family
   - "white" when truth is "taupe/beige/gray/stone" = WRONG
2. "keep-clear box" and "white boxed keep-clear marking" = CORRECT
3. Non-committal answers = WRONG
4. Core fact correct + extra detail = CORRECT

Reply with exactly one word: CORRECT or WRONG"""

    client = Groq(api_key=api_key)
    results, stats = [], defaultdict(lambda: {"correct": 0, "total": 0})

    from tqdm import tqdm
    for row in tqdm(rows, desc="Judging"):
        prompt = JUDGE_PROMPT.format(
            question=row["question"],
            ground_truth=row["actual_answer"],
            model_answer=row["generated_answer"]
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5, temperature=0.0
                )
                score = 1 if "CORRECT" in resp.choices[0].message.content.upper() else 0
                break
            except Exception as e:
                print(f"\nAPI error: {e}")
                time.sleep(15)
                score = 0

        results.append({**row, "judge_score": score})
        m = row["model"]
        stats[m]["correct"] += score
        stats[m]["total"]   += 1
        time.sleep(2)

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    print("\n" + "="*60)
    print("=== OFFICIAL v3 RESULTS (LLaMA-3.1-8b Judge) ===")
    print("="*60)
    for mn, s in sorted(stats.items()):
        pct = s["correct"] / s["total"] * 100
        print(f"  {mn:<42} {s['correct']:>4}/{s['total']:<4} = {pct:.1f}%")
    print(f"\nSaved to: {output_path}")
    return stats


def main():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: export GROQ_API_KEY=gsk_...")
        sys.exit(1)

    # Merge all rows
    all_rows = []
    
    # v2 baselines (ZeroShot + FT on all 403 questions)
    for fname, model in V2_BASELINE_FILES:
        rows = load_csv_as_unified(fname, model)
        all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {fname}")
    
    # v3 KG results
    for fname, model in V3_FILES:
        rows = load_csv_as_unified(fname, model)
        all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {fname}")

    print(f"\nTotal rows to judge: {len(all_rows)}")

    # Save merged CSV
    merged_path = BASE / "all_models_v3.csv"
    with open(merged_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Merged CSV saved: {merged_path}")

    # Run judge
    output_path = BASE / "official_scores_v3.csv"
    run_judge(all_rows, output_path, api_key)


if __name__ == "__main__":
    main()
