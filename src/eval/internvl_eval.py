"""
InternVL2.5-2B Zero-Shot Inference on UDVideoQA Morning Attribution questions.
Produces CSV matching our standard format for the Groq judge.

Usage:
    python src/eval/internvl_eval.py --set Set_34 --max_clips 3   # test
    python src/eval/internvl_eval.py                               # full 403 Qs
"""

import os, csv, sys, argparse, torch, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/path/to/your/data")
MODEL_PATH = str(BASE_DIR / "models" / "internvl2_5_2b")
DATA_CSV   = str(BASE_DIR / "data" / "morning_attribution.csv")
VIDEO_DIR  = str(BASE_DIR / "data" / "videos")
EVAL_DIR   = BASE_DIR / "results" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def find_video(set_name, clip_name):
    for p in Path(VIDEO_DIR).rglob(clip_name.replace(".mp4", "_blurred.mp4")):
        if set_name in str(p):
            return str(p)
    for p in Path(VIDEO_DIR).rglob(clip_name):
        if set_name in str(p):
            return str(p)
    return None


def extract_frames(video_path, num_frames=8):
    """Extract evenly spaced RGB frames as PIL images."""
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
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def build_transform(input_size=448):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model():
    logger.info(f"Loading InternVL2.5-2B from {MODEL_PATH}...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    logger.info("Model loaded.")
    return model, tokenizer


def run_inference(model, tokenizer, video_path, question):
    """Run InternVL2.5 inference on video frames + question."""
    frames = extract_frames(video_path, num_frames=8)
    if not frames:
        return "Video could not be loaded."

    transform = build_transform(input_size=448)
    pixel_values = torch.stack([transform(f) for f in frames])  # (8, 3, 448, 448)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

    num_patches_list = [1] * len(frames)

    # Build prompt with video frame tokens
    frame_tokens = "\n".join([f"Frame{i+1}: <image>" for i in range(len(frames))])
    prompt = f"You are analyzing frames from a traffic monitoring video.\n{frame_tokens}\n\nQuestion: {question}\nAnswer concisely:"

    generation_config = dict(
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set",       default=None)
    parser.add_argument("--max_clips", type=int, default=None)
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

    out_path = (EVAL_DIR / f"E008_InternVL2_5_2B_ZeroShot_{args.set}.csv") if args.set \
               else (EVAL_DIR / "E008_InternVL2_5_2B_ZeroShot_all.csv")

    model, tokenizer = load_model()

    results, errors = [], 0
    fieldnames = ["question_id", "video_id", "set", "question",
                  "model", "generated_answer", "actual_answer"]

    for row in tqdm(rows, desc="InternVL Inference"):
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
                answer = run_inference(model, tokenizer, video_path, question)
            except Exception as e:
                logger.error(f"Error: {e}")
                answer = "Error during inference."
                errors += 1

        logger.info(f"Q:     {question[:65]}")
        logger.info(f"InVL:  {answer[:65]}")
        logger.info(f"Act:   {actual[:65]}\n")

        results.append({
            "question_id":      row["question_id"],
            "video_id":         clip_name,
            "set":              set_name,
            "question":         question,
            "model":            "InternVL2.5-2B-ZeroShot",
            "generated_answer": answer,
            "actual_answer":    actual,
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Done! {len(results)} questions, {errors} errors.")
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
