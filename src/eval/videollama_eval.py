"""
VideoLLaMA3-7B Zero-Shot Inference on UDVideoQA Morning Attribution questions.
Produces CSV matching our standard format for the Groq judge.

Usage:
    python src/eval/videollama_eval.py --set Set_34 --max_clips 3   # test
    python src/eval/videollama_eval.py                               # full 403 Qs
"""

import os, csv, sys, argparse, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/path/to/your/data")
MODEL_PATH = str(BASE_DIR / "models" / "videollama3_7b")
DATA_CSV   = str(BASE_DIR / "data" / "morning_attribution.csv")
VIDEO_DIR  = str(BASE_DIR / "data" / "videos")
EVAL_DIR   = BASE_DIR / "results" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def find_video(set_name, clip_name):
    """Find blurred video file for a given set and clip."""
    for p in Path(VIDEO_DIR).rglob(clip_name.replace(".mp4", "_blurred.mp4")):
        if set_name in str(p):
            return str(p)
    for p in Path(VIDEO_DIR).rglob(clip_name):
        if set_name in str(p):
            return str(p)
    return None


def load_model():
    logger.info(f"Loading VideoLLaMA3-7B from {MODEL_PATH}...")
    from transformers import AutoModelForCausalLM, AutoProcessor

    # Try flash_attention_2, fall back to eager
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Using flash_attention_2")
    except ImportError:
        attn_impl = "eager"
        logger.info("flash_attn not found, using eager attention")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    logger.info("Model loaded.")
    return model, processor


def run_inference(model, processor, video_path, question):
    """Run VideoLLaMA3 inference on a single video + question."""
    conversation = [
        {"role": "system", "content": "You are a helpful assistant answering questions about traffic monitoring videos. Be concise and specific."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "max_frames": 8, "fps": 1.0}},
                {"type": "text", "text": question},
            ],
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    answer = processor.decode(generated, skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set",       default=None)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--output",    default=None)
    args = parser.parse_args()

    # Load dataset
    rows = list(csv.DictReader(open(DATA_CSV)))
    if args.set:
        rows = [r for r in rows if r["set"] == args.set]

    if args.max_clips:
        seen, filtered = set(), []
        for r in rows:
            key = (r["set"], r["video_file_path"])
            if key not in seen:
                seen.add(key)
                if len(seen) > args.max_clips:
                    break
            if key in seen:
                filtered.append(r)
        rows = filtered

    logger.info(f"Evaluating {len(rows)} questions...")

    # Output path
    if args.output:
        out_path = Path(args.output)
    elif args.set:
        out_path = EVAL_DIR / f"E008_VideoLLaMA3_ZeroShot_{args.set}.csv"
    else:
        out_path = EVAL_DIR / "E008_VideoLLaMA3_ZeroShot_all.csv"

    # Load model
    model, processor = load_model()

    # Evaluate
    results, errors = [], 0
    fieldnames = ["question_id", "video_id", "set", "question",
                  "model", "generated_answer", "actual_answer"]

    for row in tqdm(rows, desc="VideoLLaMA3 Inference"):
        set_name  = row["set"]
        clip_name = row["video_file_path"]
        question  = row["question"]
        actual    = row["actual_answer"]

        video_path = find_video(set_name, clip_name)
        if not video_path:
            logger.warning(f"Video not found: {set_name}/{clip_name}")
            answer = "Video not found."
            errors += 1
        else:
            try:
                answer = run_inference(model, processor, video_path, question)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                answer = "Error during inference."
                errors += 1

        logger.info(f"Q:    {question[:65]}")
        logger.info(f"VL3:  {answer[:65]}")
        logger.info(f"Act:  {actual[:65]}\n")

        results.append({
            "question_id":      row["question_id"],
            "video_id":         clip_name,
            "set":              set_name,
            "question":         question,
            "model":            "VideoLLaMA3-7B-ZeroShot",
            "generated_answer": answer,
            "actual_answer":    actual,
        })

    # Save
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Done! {len(results)} questions, {errors} errors.")
    logger.info(f"Saved: {out_path}")
    logger.info("Next: add E008_VideoLLaMA3_ZeroShot_all.csv to complete_eval.py and re-judge.")


if __name__ == "__main__":
    main()
