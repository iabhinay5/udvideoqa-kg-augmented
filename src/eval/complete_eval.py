"""
COMPLETE OFFICIAL EVALUATION — All models × 403 questions
LLM-as-Judge using LLaMA-3.1-8b-instant via Groq.

Models included:
  - Qwen2.5-VL-3B-ZeroShot        (baseline)
  - Qwen2.5-VL-3B-FT               (fine-tuned)
  - LLaVA-NeXT-Video-7B-ZeroShot   (2nd architecture)
  - GraphRAG-KG-v3                 (our method — Qwen+KG)
  - Phi-3.5-Vision-4.2B-ZeroShot   (3rd architecture)
  - BLIP-2-OPT-2.7B-ZeroShot       (4th architecture)
  - LLaVA-NeXT-Video-7B-KGv3       (LLaVA+KG — proves model-agnostic KG)

Usage:
    export GROQ_API_KEY="gsk_..."

    # Standard 4-model eval (Qwen ZS, Qwen FT, LLaVA, KG-v3):
    python src/eval/complete_eval.py

    # After Phi-3.5 inference finishes (E008):
    python src/eval/complete_eval.py --include_phi35

    # After BLIP-2 inference finishes (E009):
    python src/eval/complete_eval.py --include_blip2

    # All 6 models at once:
    python src/eval/complete_eval.py --include_phi35 --include_blip2

    # Test with first 20 rows only:
    python src/eval/complete_eval.py --limit 20
"""
import csv, os, sys, time, argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

BASE = Path("/path/to/your/data/results/eval")

# ── Input file definitions ──────────────────────────────────────────────────
BASELINE_FILES = [
    ("Qwen2.5-VL-3B-ZeroShot_morning_attribution.csv", "Qwen2.5-VL-3B-ZeroShot", "generated_answer"),
    ("Qwen2.5-VL-3B-FT_morning_attribution.csv",       "Qwen2.5-VL-3B-FT",       "generated_answer"),
]

KG_V3_FILES = [
    ("E006_GraphRAGv3_Set_26.csv", "GraphRAG-KG-v3", "answer_graphrag"),
    ("E006_GraphRAGv3_Set_30.csv", "GraphRAG-KG-v3", "answer_graphrag"),
    ("E006_GraphRAGv3_Set_33.csv", "GraphRAG-KG-v3", "answer_graphrag"),
    ("E006_GraphRAGv3_Set_34.csv", "GraphRAG-KG-v3", "answer_graphrag"),
    ("E006_GraphRAGv3_Set_35.csv", "GraphRAG-KG-v3", "answer_graphrag"),
]

LLAVA_FILES = [
    ("E007_LLaVA7B_ZeroShot_all.csv", "LLaVA-NeXT-Video-7B-ZeroShot", "generated_answer"),
]

PHI35_FILES = [
    ("E008_Phi35Vision_ZeroShot_all.csv", "Phi-3.5-Vision-4.2B-ZeroShot", "generated_answer"),
]

BLIP2_FILES = [
    ("E009_BLIP2_all.csv", "BLIP-2-OPT-2.7B-ZeroShot", "generated_answer"),
]

LLAVA_KG_FILES = [
    ("E010_GraphRAGLLaVA_all.csv", "LLaVA-NeXT-Video-7B-KGv3", "generated_answer"),
]

# ── Judge prompt ─────────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are evaluating a traffic video question answering system.

Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Scoring rules:
1. Color synonyms — CORRECT:
   - "olive-gray" / "taupe" / "beige-grey" / "sandy-beige" / "stone" / "warm gray" = same earth-tone
   - "gray" / "dark gray" / "charcoal" / "slate" = same gray family
2. "white" when ground truth is earth-tone (taupe/beige/sandy/olive) = WRONG
3. Road markings: "keep-clear" / "white boxed keep-clear marking" = CORRECT
4. Non-committal ("I cannot determine", "unclear", "I don't know") = WRONG
5. Extra detail around correct core fact = CORRECT
6. Partial answer that contains the key fact = CORRECT

Reply with exactly one word: CORRECT or WRONG"""


def load_file(fname, model, ans_col):
    path = BASE / fname
    if not path.exists():
        print(f"  WARNING: {fname} not found — skipping")
        return []
    rows = []
    for r in csv.DictReader(open(path)):
        vid_col = next((c for c in r if "video" in c.lower()), "video_id")
        rows.append({
            "question_id":      r.get("question_id", ""),
            "video_id":         r.get(vid_col, ""),
            "set":              r.get("set", ""),
            "question":         r.get("question", ""),
            "model":            model,
            "generated_answer": r.get(ans_col, ""),
            "actual_answer":    r.get("actual_answer", ""),
        })
    print(f"  Loaded {len(rows):>4} rows from {fname}")
    return rows


def judge_batch(rows, client):
    results = []
    for row in tqdm(rows, desc="Judging"):
        prompt = JUDGE_PROMPT.format(
            question=row["question"],
            ground_truth=row["actual_answer"],
            model_answer=row["generated_answer"]
        )
        score = 0
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                score = 1 if "CORRECT" in resp.choices[0].message.content.upper() else 0
                break
            except Exception as e:
                print(f"\n  API error (attempt {attempt+1}): {e}")
                time.sleep(20)
        results.append({**row, "judge_score": score})
        time.sleep(2)  # Groq free tier: 30 req/min
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_phi35", action="store_true",
                        help="Include Phi-3.5-Vision results (E008_Phi35Vision_ZeroShot_all.csv)")
    parser.add_argument("--include_blip2", action="store_true",
                        help="Include BLIP-2 results (E009_BLIP2_all.csv)")
    parser.add_argument("--include_llava_kg", action="store_true",
                        help="Include LLaVA+KG-v3 results (E010_GraphRAGLLaVA_all.csv)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only judge first N rows (for testing)")
    parser.add_argument("--output", default=None,
                        help="Override output CSV path")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: export GROQ_API_KEY=gsk_...")
        sys.exit(1)

    from groq import Groq
    client = Groq(api_key=api_key)

    # ── Build file list ───────────────────────────────────────────────────────
    file_groups = BASELINE_FILES + KG_V3_FILES + LLAVA_FILES
    if args.include_phi35:
        file_groups += PHI35_FILES
    if args.include_blip2:
        file_groups += BLIP2_FILES
    if args.include_llava_kg:
        file_groups += LLAVA_KG_FILES

    # ── Load all rows ─────────────────────────────────────────────────────────
    print("\n=== Loading input files ===")
    all_rows = []
    for fname, model, ans_col in file_groups:
        all_rows.extend(load_file(fname, model, ans_col))

    if args.limit:
        all_rows = all_rows[:args.limit]

    if not all_rows:
        print("ERROR: No rows loaded. Check that CSV files exist in results/eval/")
        sys.exit(1)

    print(f"\nTotal rows to judge: {len(all_rows)}")
    by_model = defaultdict(int)
    for r in all_rows:
        by_model[r["model"]] += 1
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n}")

    # ── Save merged CSV before judging ────────────────────────────────────────
    merged_path = BASE / "all_models_complete.csv"
    with open(merged_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nMerged CSV → {merged_path}")

    # ── Run judge ─────────────────────────────────────────────────────────────
    est_min = len(all_rows) * 2 // 60
    print(f"\n=== Running LLM Judge (~{est_min} min) ===")
    results = judge_batch(all_rows, client)

    # ── Save scored results ───────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = BASE / "official_scores_final.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ── Print final summary table ─────────────────────────────────────────────
    stats = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        m = r["model"]
        stats[m]["c"] += int(r["judge_score"])
        stats[m]["t"] += 1

    print("\n" + "=" * 70)
    print("=== OFFICIAL RESULTS (LLaMA-3.1-8b Judge via Groq) ===")
    print("=" * 70)

    # Print in a consistent order
    MODEL_ORDER = [
        "Qwen2.5-VL-3B-ZeroShot",
        "Qwen2.5-VL-3B-FT",
        "Phi-3.5-Vision-4.2B-ZeroShot",
        "BLIP-2-OPT-2.7B-ZeroShot",
        "LLaVA-NeXT-Video-7B-ZeroShot",
        "LLaVA-NeXT-Video-7B-KGv3",
        "GraphRAG-KG-v3",
    ]
    printed = set()
    for m in MODEL_ORDER:
        if m in stats:
            s = stats[m]
            pct = s["c"] / s["t"] * 100
            bar = "█" * int(pct / 2)
            tag = "  ← OURS" if "GraphRAG" in m else ""
            print(f"  {m:<38} {s['c']:>4}/{s['t']:<4} = {pct:5.1f}%  {bar}{tag}")
            printed.add(m)
    # Any models not in ORDER list
    for m, s in sorted(stats.items()):
        if m not in printed:
            pct = s["c"] / s["t"] * 100
            bar = "█" * int(pct / 2)
            print(f"  {m:<38} {s['c']:>4}/{s['t']:<4} = {pct:5.1f}%  {bar}")

    print("=" * 70)
    print(f"\nDetailed results → {out_path}")


if __name__ == "__main__":
    main()
