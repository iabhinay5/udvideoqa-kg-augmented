"""
LLaVA-NeXT-Video-7B Inference on UDVideoQA Morning Attribution questions.
Produces CSV matching our standard format for the Groq judge.

Usage:
    python src/eval/llava_eval.py --set Set_34 --max_clips 5   # test
    python src/eval/llava_eval.py                               # full 403 Qs
"""

import os, csv, sys, argparse, torch, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/path/to/your/data")
MODEL_PATH = str(BASE_DIR / "models" / "llava_next_video_7b")
DATA_CSV   = str(BASE_DIR / "data" / "morning_attribution.csv")
VIDEO_DIR  = str(BASE_DIR / "data" / "videos")
EVAL_DIR   = BASE_DIR / "results" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def find_video(set_name, clip_name):
    """Find blurred video file for a given set and clip."""
    base = Path(VIDEO_DIR)
    for p in base.rglob(clip_name.replace(".mp4", "_blurred.mp4")):
        if set_name in str(p):
            return str(p)
    for p in base.rglob(clip_name):
        if set_name in str(p):
            return str(p)
    return None


def extract_frames(video_path, num_frames=8):
    """Extract evenly spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_model():
    logger.info(f"Loading LLaVA-NeXT-Video-7B from {MODEL_PATH}...")
    from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_PATH)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("Model loaded.")
    return model, processor


def run_inference(model, processor, video_path, question):
    """Run LLaVA inference on a single video + question."""
    frames = extract_frames(video_path, num_frames=8)
    if not frames:
        return "Video could not be loaded."

    # LLaVA expects numpy array of shape (T, H, W, C)
    video_array = np.stack(frames, axis=0)  # (8, H, W, 3)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        videos=video_array,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Decode only new tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(generated, skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set",       default=None, help="Evaluate one set only (e.g. Set_34)")
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
        out_path = EVAL_DIR / f"E007_LLaVA7B_ZeroShot_{args.set}.csv"
    else:
        out_path = EVAL_DIR / "E007_LLaVA7B_ZeroShot_all.csv"

    # Load model
    model, processor = load_model()

    # Evaluate
    fieldnames = ["question_id", "video_id", "set", "question",
                  "model", "generated_answer", "actual_answer"]
    results = []
    errors = 0

    for row in tqdm(rows, desc="LLaVA Inference"):
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

        logger.info(f"Q:    {question[:70]}")
        logger.info(f"LLaVA:{answer[:70]}")
        logger.info(f"Act:  {actual[:70]}\n")

        results.append({
            "question_id":      row["question_id"],
            "video_id":         clip_name,
            "set":              set_name,
            "question":         question,
            "model":            "LLaVA-NeXT-Video-7B-ZeroShot",
            "generated_answer": answer,
            "actual_answer":    actual,
        })

    # Save
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\nDone! {len(results)} questions, {errors} errors.")
    logger.info(f"Saved to: {out_path}")
    logger.info("Next: run src/eval/complete_eval.py to judge these results.")


if __name__ == "__main__":
    main()
