"""
Phi-3.5-Vision-Instruct (4.2B) Zero-Shot Inference
on UDVideoQA Morning Attribution questions.

Key requirements for transformers 5.12.1 compatibility:
  - trust_remote_code=True in BOTH AutoProcessor AND AutoModelForCausalLM
  - DynamicCache.seen_tokens patched → get_seq_length() property
  - DynamicCache.from_legacy_cache patched for missing classmethod
  - _attn_implementation="eager" (avoid Flash Attention issues)
  - bfloat16 (more stable than float16 on A6000)

Usage:
    python src/eval/phi35_eval.py --set Set_34 --max_clips 3   # quick test
    python src/eval/phi35_eval.py                               # full 403 Qs
"""

import os, csv, argparse, torch, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/path/to/your/data")
MODEL_PATH = str(BASE_DIR / "models" / "phi35_vision")
DATA_CSV   = str(BASE_DIR / "data" / "morning_attribution.csv")
VIDEO_DIR  = str(BASE_DIR / "data" / "videos")
EVAL_DIR   = BASE_DIR / "results" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Compatibility Patches ──────────────────────────────────────────────────

def patch_dynamic_cache():
    """
    Phi-3.5 was built against transformers ~4.45 where DynamicCache had:
      - .seen_tokens attribute
      - DynamicCache.from_legacy_cache(past_key_values) classmethod
      - instance.to_legacy_cache() method

    In transformers 5.x these were removed/renamed. Patch them back.
    This MUST be called before any model loading.
    """
    try:
        from transformers.cache_utils import DynamicCache

        # 1. .seen_tokens → .get_seq_length()
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
            logger.info("[patch] DynamicCache.seen_tokens → get_seq_length()")

        # 2. DynamicCache.from_legacy_cache(past) classmethod
        if not hasattr(DynamicCache, "from_legacy_cache"):
            @classmethod
            def _from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is not None:
                    for layer_past in past_key_values:
                        cache.update(layer_past[0], layer_past[1], len(cache.key_cache))
                return cache
            DynamicCache.from_legacy_cache = _from_legacy_cache
            logger.info("[patch] DynamicCache.from_legacy_cache added")

        # 3. instance.to_legacy_cache() method
        if not hasattr(DynamicCache, "to_legacy_cache"):
            def _to_legacy_cache(self):
                return tuple(
                    (self.key_cache[i], self.value_cache[i])
                    for i in range(len(self.key_cache))
                )
            DynamicCache.to_legacy_cache = _to_legacy_cache
            logger.info("[patch] DynamicCache.to_legacy_cache added")

        # 4. get_usable_length(new_seq_length, layer_idx) — removed in transformers 5.x
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
            logger.info("[patch] DynamicCache.get_usable_length added")

        # 5. get_max_length() — removed in transformers 5.x (dynamic cache = no limit → None)
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
            logger.info("[patch] DynamicCache.get_max_length added")

    except Exception as e:
        logger.warning(f"Cache patch failed (may be fine): {e}")


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
    """Load Phi-3.5-Vision processor and model with all compatibility patches."""
    # MUST patch before ANY transformers model code runs
    patch_dynamic_cache()

    logger.info(f"Loading Phi-3.5-Vision-Instruct from {MODEL_PATH}...")
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,   # REQUIRED — Phi-3.5 has custom processor code
        num_crops=4,              # Reduce memory: 4 crops vs default 16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,   # REQUIRED — Phi-3.5 has custom modeling code
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        _attn_implementation="eager",   # Avoid Flash Attention which needs newer kernels
    )
    model.eval()
    logger.info("Phi-3.5-Vision loaded successfully.")
    return model, processor


def run_inference(model, processor, video_path, question):
    """Run Phi-3.5-Vision inference on a single video + question."""
    frames = extract_frames(video_path, num_frames=8)
    if not frames:
        return "Video could not be loaded."

    # Phi-3.5 uses <|image_N|>\n tokens for each image
    # We pass up to 8 frames as separate images
    n = len(frames)
    image_tokens = "".join([f"<|image_{i+1}|>\n" for i in range(n)])

    prompt_text = (
        f"{image_tokens}"
        f"These are {n} frames sampled from a 10-second urban traffic monitoring video.\n"
        f"Question: {question}\n"
        f"Give a concise, direct answer (1-5 words if possible):"
    )

    messages = [
        {"role": "user", "content": prompt_text}
    ]

    # Apply chat template to get formatted prompt string
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize with images
    inputs = processor(
        text=prompt,
        images=frames,           # list of PIL Images
        return_tensors="pt",
    ).to(model.device)

    # Cast inputs to model dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=False,          # Bypass KV cache — avoids GQA shape mismatch in transformers 5.x
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode ONLY the newly generated tokens (skip input tokens)
    input_len = inputs["input_ids"].shape[1]
    answer = processor.tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    return answer


def main():
    parser = argparse.ArgumentParser(description="Phi-3.5-Vision eval on UDVideoQA morning attribution")
    parser.add_argument("--set",       default=None,    help="Evaluate one set only (e.g. Set_34)")
    parser.add_argument("--max_clips", type=int,        default=None, help="Limit to N clips (for testing)")
    parser.add_argument("--output",    default=None,    help="Override output CSV path")
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

    logger.info(f"Evaluating {len(rows)} questions...")

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    elif args.set:
        out_path = EVAL_DIR / f"E008_Phi35Vision_ZeroShot_{args.set}.csv"
    else:
        out_path = EVAL_DIR / "E008_Phi35Vision_ZeroShot_all.csv"

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    fieldnames = ["question_id", "video_id", "set", "question",
                  "model", "generated_answer", "actual_answer"]
    results, errors = [], 0

    for row in tqdm(rows, desc="Phi-3.5 Inference"):
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
                logger.error(f"Inference error on {clip_name}: {e}")
                answer = "Error during inference."
                errors += 1

        logger.info(f"Q:    {question[:65]}")
        logger.info(f"Phi:  {answer[:65]}")
        logger.info(f"Act:  {actual[:65]}\n")

        results.append({
            "question_id":      row["question_id"],
            "video_id":         clip_name,
            "set":              set_name,
            "question":         question,
            "model":            "Phi-3.5-Vision-4.2B-ZeroShot",
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
    logger.info("Next: add E008_Phi35Vision_ZeroShot_all.csv to complete_eval.py and re-judge.")


if __name__ == "__main__":
    main()
