"""
UDVideoQA Dataset Downloader and Explorer
-----------------------------------------
Run this script to:
1. Download the dataset from HuggingFace
2. Show basic statistics
3. Sample a few QA pairs for inspection

Usage:
    python src/data/download_dataset.py
    python src/data/download_dataset.py --sample 5 --condition morning
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset

# ── Config ────────────────────────────────────────────
HF_REPO = "UDVideoQA/UDVideoQA"
LOCAL_DATA_DIR = Path("data/processed")
LOCAL_VIDEO_DIR = Path("data/videos")

# QA taxonomy weights (from paper)
WEIGHTS = {"BU": 1.0, "Atr": 1.2, "ER": 1.3, "RR": 1.3, "CI": 1.5}
CONDITIONS = ["morning", "midday", "evening", "nighttime"]

# ── Download ───────────────────────────────────────────
def download_dataset():
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading UDVideoQA from HuggingFace: {HF_REPO}")
    try:
        dataset = load_dataset(HF_REPO, trust_remote_code=True)
        print(f"✅ Dataset loaded: {dataset}")
        return dataset
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("   Try: huggingface-cli login")
        print("   Or visit: https://huggingface.co/UDVideoQA")
        return None

# ── Statistics ─────────────────────────────────────────
def print_statistics(dataset):
    print("\n" + "="*60)
    print("📊 UDVideoQA DATASET STATISTICS")
    print("="*60)

    for split_name, split_data in dataset.items():
        df = split_data.to_pandas()
        print(f"\n[{split_name.upper()}] — {len(df):,} samples")

        # Per reasoning type
        if "reasoning_type" in df.columns:
            print("\n  Reasoning Type Distribution:")
            counts = df["reasoning_type"].value_counts()
            for rtype, count in counts.items():
                weight = WEIGHTS.get(rtype, 1.0)
                print(f"    {rtype:5s}: {count:6,} samples  (weight={weight})")

        # Per condition (time of day)
        if "condition" in df.columns:
            print("\n  Time-of-Day Distribution:")
            for cond in CONDITIONS:
                n = len(df[df["condition"] == cond])
                print(f"    {cond:10s}: {n:6,} samples")

        # Sample questions
        print("\n  Sample QA pairs:")
        sample = df.sample(min(3, len(df))).reset_index(drop=True)
        for i, row in sample.iterrows():
            print(f"\n  [{i+1}] Type: {row.get('reasoning_type','?')} | Cond: {row.get('condition','?')}")
            print(f"       Q: {row.get('question', '?')[:100]}")
            print(f"       A: {row.get('answer', '?')[:80]}")

# ── Filter and Save ────────────────────────────────────
def save_as_jsonl(dataset, split="train"):
    """Save split as JSONL for model training"""
    out_path = LOCAL_DATA_DIR / f"{split}.jsonl"
    data = dataset[split].to_pandas()

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in data.iterrows():
            record = {
                "video_path": row.get("video_path", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "reasoning_type": row.get("reasoning_type", ""),
                "condition": row.get("condition", ""),
                "difficulty": row.get("difficulty", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved {split} split → {out_path} ({len(data):,} rows)")

# ── Main ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UDVideoQA dataset downloader")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download (use if already downloaded)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Print N sample QA pairs")
    parser.add_argument("--condition", default=None,
                        choices=CONDITIONS + [None],
                        help="Filter by time-of-day condition")
    args = parser.parse_args()

    if not args.skip_download:
        dataset = download_dataset()
    else:
        print("⏭️  Skipping download. Loading from cache...")
        dataset = load_dataset(HF_REPO, trust_remote_code=True)

    if dataset is None:
        return

    print_statistics(dataset)

    # Save as JSONL for training
    for split in dataset.keys():
        save_as_jsonl(dataset, split)

    print("\n✅ Setup complete! Next steps:")
    print("   1. Run: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("   2. Run: python src/models/baseline/evaluate.py --config configs/baseline.yaml")

if __name__ == "__main__":
    main()
