"""
UDVideoQA — Morning Attribution Evaluation Script
===================================================
Runs Qwen2.5-VL-3B + paper's LoRA adapter on morning Attribution questions.
Outputs: CSV with question_id, video_id, question, model, generated_answer, actual_answer

Usage:
    python src/eval/run_eval.py
    python src/eval/run_eval.py --max_questions 50   # quick test
    python src/eval/run_eval.py --no_adapter          # zero-shot (base model only)
"""

import os
import csv
import json
import time
import glob
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from loguru import logger

# ── Paths ────────────────────────────────────────────────
BASE_DIR    = "/path/to/your/data"
CSV_IN      = f"{BASE_DIR}/data/morning_attribution.csv"
VIDEO_DIR   = f"{BASE_DIR}/data/videos"
ADAPTER_DIR = f"{BASE_DIR}/models/baseline_adapter"
BASE_MODEL  = f"{BASE_DIR}/models/base_model"
RESULTS_DIR = f"{BASE_DIR}/results/eval"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Video Path Resolver ──────────────────────────────────
def find_video_path(set_name: str, clip_name: str) -> str | None:
    """
    Maps (set_name, clip_name) → actual absolute path on disk.
    JSONL has 'clip_045.mp4', actual file is 'clip_045_blurred.mp4'.
    Searches recursively under VIDEO_DIR/set_name.
    """
    # Try blurred variant first (what we downloaded)
    clip_stem = clip_name.replace(".mp4", "")
    blurred_name = f"{clip_stem}_blurred.mp4"

    set_path = os.path.join(VIDEO_DIR, set_name)
    if not os.path.exists(set_path):
        return None

    # Recursive search
    for root, dirs, files in os.walk(set_path):
        if blurred_name in files:
            return os.path.join(root, blurred_name)
        if clip_name in files:  # fallback to non-blurred
            return os.path.join(root, clip_name)

    return None


# ── Load Questions ───────────────────────────────────────
def load_questions(csv_path: str, max_questions: int = -1) -> list:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if max_questions > 0:
        rows = rows[:max_questions]
    logger.info(f"Loaded {len(rows)} questions from {csv_path}")
    return rows


# ── Load Model ───────────────────────────────────────────
def load_model(use_adapter: bool = True):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    logger.info(f"Loading base model: {BASE_MODEL}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # No flash-attn
    )

    if use_adapter:
        logger.info(f"Loading LoRA adapter: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model = model.merge_and_unload()  # Merge adapter into base for faster inference
        logger.info("Adapter merged.")

    model.eval()

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        min_pixels=128*28*28,
        max_pixels=360*28*28,
    )

    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Model ready. VRAM used: {vram:.1f} GB")
    return model, processor


# ── Run Inference ────────────────────────────────────────
def run_inference(model, processor, question: str, video_path: str | None) -> str:
    """
    Runs one inference call. Uses video if available, else text-only.
    """
    content = []

    if video_path and os.path.exists(video_path):
        content.append({
            "type": "video",
            "video": video_path,
            "max_pixels": 360 * 28 * 28,
            "fps": 1.0,  # 1 frame/sec from 10s clip = 10 frames
        })
        has_video = True
    else:
        has_video = False

    content.append({
        "type": "text",
        "text": (
            "You are analyzing a traffic video clip. "
            "Answer the following question concisely based on what you observe.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    })

    messages = [{"role": "user", "content": content}]

    try:
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if has_video:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = None, None

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        answer = processor.decode(generated, skip_special_tokens=True).strip()
        return answer

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return f"ERROR: {str(e)[:100]}"


# ── Main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_questions", type=int, default=-1,
                        help="Limit number of questions (default: all 403)")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Run base model without adapter (zero-shot)")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Skip first N questions (for resuming)")
    args = parser.parse_args()

    model_name = "Qwen2.5-VL-3B-ZeroShot" if args.no_adapter else "Qwen2.5-VL-3B-FT"
    out_csv = f"{RESULTS_DIR}/{model_name}_morning_attribution.csv"

    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {out_csv}")

    # Load questions
    questions = load_questions(CSV_IN, args.max_questions)
    if args.start_from > 0:
        questions = questions[args.start_from:]
        logger.info(f"Resuming from question {args.start_from}")

    # Load model
    model, processor = load_model(use_adapter=not args.no_adapter)

    # Track stats
    results = []
    video_found = 0
    video_missing = 0
    errors = 0

    # Run evaluation
    fieldnames = ["question_id", "video_id", "question", "model",
                  "generated_answer", "actual_answer", "has_video", "set"]

    # Open CSV for writing (append mode for resuming)
    write_mode = "a" if args.start_from > 0 else "w"
    with open(out_csv, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if args.start_from == 0:
            writer.writeheader()

        for i, row in enumerate(tqdm(questions, desc=f"Evaluating {model_name}")):
            set_name       = row["set"]
            clip_name      = row["video_file_path"]
            question       = row["question"]
            actual_answer  = row["actual_answer"]
            question_id    = row["question_id"]

            # Find video
            video_path = find_video_path(set_name, clip_name)
            if video_path:
                video_found += 1
            else:
                video_missing += 1
                if video_missing <= 5:
                    logger.warning(f"Video not found: {set_name}/{clip_name}")

            # Run inference
            generated = run_inference(model, processor, question, video_path)

            result = {
                "question_id":    question_id,
                "video_id":       clip_name,
                "question":       question,
                "model":          model_name,
                "generated_answer": generated,
                "actual_answer":  actual_answer,
                "has_video":      video_path is not None,
                "set":            set_name,
            }
            results.append(result)
            writer.writerow(result)
            f.flush()  # Write immediately (safe for long runs)

            # Log progress every 50
            if (i + 1) % 50 == 0:
                logger.info(f"[{i+1}/{len(questions)}] Video found: {video_found}, Missing: {video_missing}")

    # Summary
    logger.info("=" * 60)
    logger.info(f"DONE. Results saved to: {out_csv}")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Videos found:   {video_found}")
    logger.info(f"Videos missing: {video_missing}")
    logger.info(f"\nNext: open {out_csv} to see model answers vs. ground truth")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
