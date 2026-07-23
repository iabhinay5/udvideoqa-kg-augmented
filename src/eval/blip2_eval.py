"""
BLIP-2 (Salesforce/blip2-opt-2.7b) Zero-Shot Inference
on UDVideoQA Morning Attribution questions.

BLIP-2 is image-only (not video-native), so we:
  - Extract 8 evenly-spaced frames from each 10s clip
  - Use the MIDDLE frame as the primary image (best scene coverage)
  - Feed all 8 frames individually and take the answer from the middle frame
    (or optionally: query all 8 and vote — controlled by --mode)

Zero custom code needed — fully native in transformers 4.x / 5.x.

Usage:
    python src/eval/blip2_eval.py --set Set_34 --max_clips 3   # quick test
    python src/eval/blip2_eval.py                               # full 403 Qs
    python src/eval/blip2_eval.py --mode vote                   # 8-frame majority vote
"""

import os, csv, argparse, torch, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from PIL import Image
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/path/to/your/data")
MODEL_PATH = str(BASE_DIR / "models" / "blip2_opt_2.7b")   # local cache dir
MODEL_HUB  = "Salesforce/blip2-opt-2.7b"                    # HuggingFace fallback
DATA_CSV   = str(BASE_DIR / "data" / "morning_attribution.csv")
VIDEO_DIR  = str(BASE_DIR / "data" / "videos")
EVAL_DIR   = BASE_DIR / "results" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def find_video(set_name, clip_name):
    """Locate blurred mp4 for a given set/clip (handles double-nested dirs)."""
    base = Path(VIDEO_DIR)
    for p in base.rglob(clip_name.replace(".mp4", "_blurred.mp4")):
        if set_name in str(p):
            return str(p)
    for p in base.rglob(clip_name):
        if set_name in str(p):
            return str(p)
    return None


def extract_frames(video_path, num_frames=8):
    """Extract evenly spaced RGB PIL Images from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def load_model():
    """Load BLIP-2 processor and model. Falls back from local path to HF hub."""
    logger.info("Loading BLIP-2 (Salesforce/blip2-opt-2.7b)...")
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    # Try local path first, fall back to hub
    load_from = MODEL_PATH if Path(MODEL_PATH).exists() else MODEL_HUB
    logger.info(f"  Source: {load_from}")

    processor = Blip2Processor.from_pretrained(load_from)
    model = Blip2ForConditionalGeneration.from_pretrained(
        load_from,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("Model loaded.")
    return model, processor


def ask_blip2(model, processor, image: Image.Image, question: str) -> str:
    """Run BLIP-2 VQA on a single PIL image + question string."""
    # BLIP-2 VQA prompt format
    prompt = f"Question: {question} Answer:"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(model.device, torch.float16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    answer = processor.tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    ).strip()

    # BLIP-2 sometimes echoes the prompt — strip it
    if answer.lower().startswith("question:"):
        # Remove the echoed prompt prefix
        parts = answer.split("Answer:")
        answer = parts[-1].strip() if len(parts) > 1 else answer

    return answer


def run_inference_middle(model, processor, video_path, question):
    """Query only the middle frame (fast, default)."""
    frames = extract_frames(video_path, num_frames=8)
    if not frames:
        return "Video could not be loaded."
    middle = frames[len(frames) // 2]
    return ask_blip2(model, processor, middle, question)


def run_inference_vote(model, processor, video_path, question):
    """Query all 8 frames, return most common answer (slower, more robust)."""
    frames = extract_frames(video_path, num_frames=8)
    if not frames:
        return "Video could not be loaded."

    answers = []
    for frame in frames:
        try:
            ans = ask_blip2(model, processor, frame, question)
            if ans:
                answers.append(ans.lower().strip())
        except Exception:
            pass

    if not answers:
        return "No answer."

    # Majority vote
    vote = Counter(answers).most_common(1)[0][0]
    return vote


def main():
    parser = argparse.ArgumentParser(description="BLIP-2 eval on UDVideoQA morning attribution")
    parser.add_argument("--set",       default=None,      help="Evaluate one set only (e.g. Set_34)")
    parser.add_argument("--max_clips", type=int,          default=None, help="Limit to N clips (for testing)")
    parser.add_argument("--mode",      default="middle",  choices=["middle", "vote"],
                        help="'middle' = use center frame only (fast); 'vote' = majority vote over 8 frames")
    parser.add_argument("--output",    default=None,      help="Override output CSV path")
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
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

    logger.info(f"Evaluating {len(rows)} questions  [mode={args.mode}]")

    # ── Output path ───────────────────────────────────────────────────────────
    suffix = f"_{args.mode}" if args.mode != "middle" else ""
    if args.output:
        out_path = Path(args.output)
    elif args.set:
        out_path = EVAL_DIR / f"E009_BLIP2_{args.set}{suffix}.csv"
    else:
        out_path = EVAL_DIR / f"E009_BLIP2_all{suffix}.csv"

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model()

    infer_fn = run_inference_vote if args.mode == "vote" else run_inference_middle

    # ── Evaluate ──────────────────────────────────────────────────────────────
    fieldnames = ["question_id", "video_id", "set", "question",
                  "model", "generated_answer", "actual_answer"]
    results, errors = [], 0

    for row in tqdm(rows, desc="BLIP-2 Inference"):
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
                answer = infer_fn(model, processor, video_path, question)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                answer = "Error during inference."
                errors += 1

        logger.info(f"Q:     {question[:65]}")
        logger.info(f"BLIP2: {answer[:65]}")
        logger.info(f"Act:   {actual[:65]}\n")

        results.append({
            "question_id":      row["question_id"],
            "video_id":         clip_name,
            "set":              set_name,
            "question":         question,
            "model":            "BLIP-2-OPT-2.7B-ZeroShot",
            "generated_answer": answer,
            "actual_answer":    actual,
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\nDone! {len(results)} questions, {errors} errors.")
    logger.info(f"Saved to: {out_path}")
    logger.info("Next: add E009_BLIP2_all.csv to complete_eval.py and re-judge.")


if __name__ == "__main__":
    main()
