"""
UDVideoQA — Zero-Shot Baseline Evaluation
==========================================
Evaluates Qwen2.5-VL 7B on UDVideoQA WITHOUT any fine-tuning.
This gives us the baseline number to compare our improvements against.

Usage:
    export CUDA_VISIBLE_DEVICES=6
    python src/baseline/run_zero_shot.py --num_samples 200
    python src/baseline/run_zero_shot.py --num_samples -1 --condition morning
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm
from loguru import logger

# ── Setup Logging ─────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add("results/logs/zero_shot_{time}.log", level="DEBUG")

# ── Constants ─────────────────────────────────────────
REASONING_TYPES = ["BU", "Atr", "ER", "RR", "CI"]
CONDITIONS = ["morning", "midday", "evening", "nighttime"]
WEIGHTS = {"BU": 1.0, "Atr": 1.2, "ER": 1.3, "RR": 1.3, "CI": 1.5}
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_model(model_name: str):
    """Load Qwen2.5-VL model and processor"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    logger.info(f"Loading model: {model_name}")
    start = time.time()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Falls back to default if not installed
    )
    model.eval()

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    logger.info(f"GPU memory used: {gpu_mem:.1f} GB")

    return model, processor


def load_dataset(data_path: str, condition: str = None, num_samples: int = -1):
    """Load QA pairs from JSONL file"""
    logger.info(f"Loading dataset from {data_path}")

    records = []
    with open(data_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            if condition and record.get("condition") != condition:
                continue
            records.append(record)

    if num_samples > 0:
        records = records[:num_samples]

    logger.info(f"Loaded {len(records)} samples" + (f" (condition={condition})" if condition else ""))
    return records


def prepare_input(processor, record: dict):
    """
    Prepare model input from a QA record.
    Handles both video-based and frame-based inputs.
    """
    question = record["question"]
    video_path = record.get("video_path", "")

    # Build messages in Qwen2.5-VL chat format
    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]

    # Add video if available
    if video_path and os.path.exists(video_path):
        messages[0]["content"].append({
            "type": "video",
            "video": video_path,
            "max_pixels": 360 * 420,  # Keep reasonable for zero-shot
            "fps": 2.0,  # Sample 2 frames per second from 10s clip = 20 frames
        })
    elif "frames" in record and record["frames"]:
        # If frames are provided as image paths
        for frame_path in record["frames"][:8]:  # Limit to 8 frames
            if os.path.exists(frame_path):
                messages[0]["content"].append({
                    "type": "image",
                    "image": frame_path,
                })

    # Add question text
    messages[0]["content"].append({
        "type": "text",
        "text": f"Watch the traffic video carefully and answer the following question concisely.\n\nQuestion: {question}\n\nAnswer:",
    })

    # Process with Qwen2.5-VL processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=None,  # Video is handled via the messages format
        padding=True,
        return_tensors="pt",
    )

    return inputs


def generate_answer(model, processor, inputs, max_new_tokens=128):
    """Generate answer from model"""
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            temperature=1.0,
            top_p=1.0,
        )

    # Decode only the generated tokens (not the input)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    answer = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return answer


def simple_match(predicted: str, ground_truth: str) -> float:
    """
    Simple semantic matching — checks if key content words overlap.
    This is a placeholder. The paper uses Gemini 2.5 Pro as LLM Judge.
    For now, we use a basic matching approach.
    """
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # Exact match
    if pred_lower == gt_lower:
        return 1.0

    # Check if ground truth is contained in prediction (common for short answers)
    if gt_lower in pred_lower or pred_lower in gt_lower:
        return 1.0

    # Word overlap score
    pred_words = set(pred_lower.split())
    gt_words = set(gt_lower.split())

    if not gt_words:
        return 0.0

    overlap = pred_words & gt_words
    # Remove stop words from overlap count
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of", "and", "or"}
    meaningful_overlap = overlap - stop_words
    meaningful_gt = gt_words - stop_words

    if not meaningful_gt:
        return 1.0 if not meaningful_overlap else 0.0

    score = len(meaningful_overlap) / len(meaningful_gt)
    return 1.0 if score >= 0.5 else 0.0


def compute_metrics(results: list) -> dict:
    """Compute accuracy metrics broken down by reasoning type and condition"""
    metrics = {}

    # Overall accuracy
    correct = sum(1 for r in results if r["score"] >= 0.5)
    metrics["overall_accuracy"] = correct / len(results) if results else 0

    # By reasoning type
    metrics["by_type"] = {}
    for rtype in REASONING_TYPES:
        subset = [r for r in results if r.get("reasoning_type") == rtype]
        if subset:
            acc = sum(1 for r in subset if r["score"] >= 0.5) / len(subset)
            metrics["by_type"][rtype] = {"accuracy": acc, "count": len(subset)}

    # By condition
    metrics["by_condition"] = {}
    for cond in CONDITIONS:
        subset = [r for r in results if r.get("condition") == cond]
        if subset:
            acc = sum(1 for r in subset if r["score"] >= 0.5) / len(subset)
            metrics["by_condition"][cond] = {"accuracy": acc, "count": len(subset)}

    # By condition × type (the key analysis)
    metrics["by_condition_and_type"] = {}
    for cond in CONDITIONS:
        metrics["by_condition_and_type"][cond] = {}
        for rtype in REASONING_TYPES:
            subset = [r for r in results if r.get("condition") == cond and r.get("reasoning_type") == rtype]
            if subset:
                acc = sum(1 for r in subset if r["score"] >= 0.5) / len(subset)
                metrics["by_condition_and_type"][cond][rtype] = acc

    # Weighted score (paper's formula)
    total_weighted = 0
    total_weight = 0
    for r in results:
        w = WEIGHTS.get(r.get("reasoning_type", "BU"), 1.0)
        total_weighted += r["score"] * w
        total_weight += w
    metrics["weighted_score"] = total_weighted / total_weight if total_weight > 0 else 0

    return metrics


def print_results(metrics: dict):
    """Pretty print evaluation results"""
    print("\n" + "=" * 70)
    print("📊 ZERO-SHOT EVALUATION RESULTS — Qwen2.5-VL 7B")
    print("=" * 70)
    print(f"\n  Overall Accuracy:  {metrics['overall_accuracy']:.1%}")
    print(f"  Weighted Score:    {metrics['weighted_score']:.1%}")

    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │  By Reasoning Type                              │")
    print("  ├───────────┬──────────┬──────────────────────────┤")
    print("  │ Type      │ Accuracy │ Samples                  │")
    print("  ├───────────┼──────────┼──────────────────────────┤")
    for rtype in REASONING_TYPES:
        data = metrics["by_type"].get(rtype, {})
        acc = data.get("accuracy", 0)
        count = data.get("count", 0)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  │ {rtype:9s} │  {acc:5.1%}  │ {bar} ({count:4d}) │")
    print("  └───────────┴──────────┴──────────────────────────┘")

    print("\n  ┌───────────────────────────────────────────────────────────┐")
    print("  │  By Condition × Reasoning Type (Accuracy %)              │")
    print("  ├────────────┬───────┬───────┬───────┬───────┬─────────────┤")
    print("  │ Condition  │  BU   │  Atr  │  ER   │  RR   │     CI      │")
    print("  ├────────────┼───────┼───────┼───────┼───────┼─────────────┤")
    for cond in CONDITIONS:
        row = metrics["by_condition_and_type"].get(cond, {})
        vals = []
        for rt in REASONING_TYPES:
            v = row.get(rt, 0)
            vals.append(f"{v:5.1%}")
        print(f"  │ {cond:10s} │ {vals[0]} │ {vals[1]} │ {vals[2]} │ {vals[3]} │   {vals[4]}   │")
    print("  └────────────┴───────┴───────┴───────┴───────┴─────────────┘")

    print("\n  Key: BU=Basic Understanding, Atr=Attribution, ER=Event Reasoning")
    print("       RR=Reverse Reasoning, CI=Counterfactual Inference")
    print("  ⚠️  Note: Uses simple word-matching. Paper uses LLM Judge (Gemini 2.5 Pro).")
    print("")


def main():
    parser = argparse.ArgumentParser(description="UDVideoQA Zero-Shot Evaluation")
    parser.add_argument("--data", type=str, default="data/processed/test.jsonl",
                        help="Path to test JSONL file")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="HuggingFace model name")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=CONDITIONS,
                        help="Filter by time-of-day condition")
    parser.add_argument("--output", type=str, default="results/zero_shot_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # Load model
    model, processor = load_model(args.model)

    # Load data
    records = load_dataset(args.data, condition=args.condition, num_samples=args.num_samples)

    if not records:
        logger.error("No data loaded! Check your data path.")
        return

    # Run inference
    results = []
    logger.info(f"Running zero-shot evaluation on {len(records)} samples...")

    for i, record in enumerate(tqdm(records, desc="Evaluating")):
        try:
            inputs = prepare_input(processor, record)
            predicted = generate_answer(model, processor, inputs, args.max_new_tokens)
            ground_truth = record.get("answer", "")

            score = simple_match(predicted, ground_truth)

            result = {
                "idx": i,
                "question": record.get("question", ""),
                "ground_truth": ground_truth,
                "predicted": predicted,
                "score": score,
                "reasoning_type": record.get("reasoning_type", ""),
                "condition": record.get("condition", ""),
            }
            results.append(result)

            # Log every 50
            if (i + 1) % 50 == 0:
                running_acc = sum(r["score"] for r in results) / len(results)
                logger.info(f"[{i+1}/{len(records)}] Running accuracy: {running_acc:.1%}")

        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            results.append({
                "idx": i,
                "question": record.get("question", ""),
                "ground_truth": record.get("answer", ""),
                "predicted": f"ERROR: {e}",
                "score": 0.0,
                "reasoning_type": record.get("reasoning_type", ""),
                "condition": record.get("condition", ""),
            })

    # Compute and display metrics
    metrics = compute_metrics(results)
    print_results(metrics)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "num_samples": len(records),
                "condition_filter": args.condition,
            },
            "metrics": metrics,
            "predictions": results,
        }, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
