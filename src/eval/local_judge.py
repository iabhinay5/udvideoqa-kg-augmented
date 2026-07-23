"""
Local LLM Judge using Qwen2.5-VL-3B already on server.
No API key needed. Runs on GPU.
Usage:
    python src/eval/local_judge.py \
        --input  results/eval/all_models_comparison.csv \
        --output results/eval/official_scores_local.csv
"""

import csv, os, sys, argparse, json
from pathlib import Path
from tqdm import tqdm

BASE_MODEL = "/path/to/your/data/hf_cache/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"

JUDGE_PROMPT = """You are evaluating a traffic video question answering system.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Answer: {model_answer}

Is the model's answer semantically correct?
Rules:
- "olive-gray", "taupe", "beige-grey", "sandy-beige", "light stone" are all the same color family → CORRECT
- "white" when truth is "taupe/beige/gray" → WRONG
- "keep-clear box" when truth is "white X marking" → WRONG
- Partial correct answers that contain the key fact → CORRECT
- Non-committal answers ("I cannot determine") → WRONG

Reply with exactly one word: CORRECT or WRONG"""


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Find snapshot path
    snap_dir = Path(BASE_MODEL)
    if snap_dir.exists():
        snaps = list(snap_dir.iterdir())
        model_path = str(snaps[0]) if snaps else "Qwen/Qwen2.5-3B-Instruct"
    else:
        model_path = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading judge model from: {model_path}")

    # Use text-only tokenizer/model (no vision head needed for judging)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor, "qwen_vl"
    except Exception as e:
        print(f"VL load failed ({e}), trying text-only Qwen...")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",
              torch_dtype="auto", device_map="auto")
        return mdl, tok, "qwen_text"


def judge_one(model, processor, mode, question, model_answer, ground_truth):
    prompt = JUDGE_PROMPT.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        model_answer=model_answer.strip()
    )

    if mode == "qwen_vl":
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
    else:
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor([text], return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    if mode == "qwen_vl":
        resp = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    else:
        resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return 1 if "CORRECT" in resp.upper() else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/path/to/your/data/results/eval/all_models_comparison.csv")
    parser.add_argument("--output", default="/path/to/your/data/results/eval/official_scores_local.csv")
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--model_filter", default=None)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.input)))
    if args.model_filter:
        rows = [r for r in rows if args.model_filter in r["model"]]
    if args.limit:
        rows = rows[:args.limit]

    print(f"Loading judge model...")
    model, processor, mode = load_model()
    print(f"Model loaded. Judging {len(rows)} rows...")

    results = []
    stats = {}

    for row in tqdm(rows):
        score = judge_one(model, processor, mode,
                          row["question"], row["generated_answer"], row["actual_answer"])
        results.append({**row, "judge_score": score})
        m = row["model"]
        stats.setdefault(m, {"correct": 0, "total": 0})
        stats[m]["correct"] += score
        stats[m]["total"] += 1

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "="*55)
    print("=== OFFICIAL RESULTS (Local LLM Judge) ===")
    print("="*55)
    for mn, s in sorted(stats.items()):
        pct = s['correct'] / s['total'] * 100
        print(f"  {mn:<40} {s['correct']:>4}/{s['total']:<4} = {pct:.1f}%")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
