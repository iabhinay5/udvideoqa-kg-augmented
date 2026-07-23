"""
UDVideoQA — LoRA Fine-tuning Script
====================================
Fine-tunes Qwen2.5-VL 7B with LoRA on the UDVideoQA training set.
Reproduces the paper's baseline fine-tuning approach.

Usage:
    export CUDA_VISIBLE_DEVICES=6
    python src/baseline/train_lora.py
    python src/baseline/train_lora.py --epochs 3 --lr 1e-4

GPU Memory:
    - A6000 (48GB): bf16 + LoRA → ~28GB used. Comfortable.
    - RTX 3090 (24GB): Use --use_4bit for QLoRA → ~18GB used.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import Dataset
from loguru import logger

# ── Setup Logging ──────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
LOG_FILE = f"results/logs/train_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("results/logs", exist_ok=True)
logger.add(LOG_FILE, level="DEBUG")


# ── Dataset ───────────────────────────────────────
class UDVideoQADataset(Dataset):
    """
    Dataset for UDVideoQA fine-tuning.
    Each item returns a dict with video/image + question + answer.
    """

    def __init__(self, jsonl_path: str, video_dir: str = "data/videos", max_samples: int = -1):
        self.records = []
        self.video_dir = video_dir

        with open(jsonl_path, "r") as f:
            for line in f:
                self.records.append(json.loads(line.strip()))

        if max_samples > 0:
            self.records = self.records[:max_samples]

        logger.info(f"Loaded {len(self.records)} training samples from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            "question": record.get("question", ""),
            "answer": record.get("answer", ""),
            "video_path": record.get("video_path", ""),
            "reasoning_type": record.get("reasoning_type", ""),
            "condition": record.get("condition", ""),
        }


def format_training_example(example: dict, processor) -> dict:
    """
    Format a single example into Qwen2.5-VL chat format for training.
    Returns tokenized inputs with labels.
    """
    question = example["question"]
    answer = example["answer"]
    video_path = example.get("video_path", "")

    # Build messages
    user_content = []

    # Add video if available
    if video_path and os.path.exists(video_path):
        user_content.append({
            "type": "video",
            "video": video_path,
            "max_pixels": 360 * 420,
            "fps": 2.0,
        })

    user_content.append({
        "type": "text",
        "text": f"Watch the traffic video carefully and answer the following question concisely.\n\nQuestion: {question}",
    })

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]

    # Tokenize
    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=[text], padding=False, return_tensors="pt")

    # For causal LM training, labels = input_ids (shifted internally by the model)
    inputs["labels"] = inputs["input_ids"].clone()

    # Mask the user portion so loss is only computed on the assistant's answer
    # Find where the assistant response starts
    full_text = text
    # Get the text up to the assistant's answer
    user_only_messages = [
        {"role": "user", "content": user_content},
    ]
    user_text = processor.apply_chat_template(
        user_only_messages, tokenize=False, add_generation_prompt=True
    )
    user_tokens = processor(text=[user_text], padding=False, return_tensors="pt")
    user_len = user_tokens["input_ids"].shape[1]

    # Set labels to -100 for the user portion (no loss)
    inputs["labels"][0, :user_len] = -100

    return {k: v.squeeze(0) for k, v in inputs.items()}


def collate_fn(batch, processor):
    """Custom collator that pads a batch of examples"""
    from torch.nn.utils.rnn import pad_sequence

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad to max length in batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="UDVideoQA LoRA Fine-tuning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_data", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/processed/validation.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints/baseline_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit QLoRA (for 24GB GPUs)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit training samples (-1=all)")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    # ── GPU Info ──────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    else:
        logger.error("No GPU found! Exiting.")
        return

    # ── Load Model ────────────────────────────────
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info(f"Loading model: {args.model}")

    # Quantization config for 24GB GPUs
    quantization_config = None
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit QLoRA quantization")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # ── Apply LoRA ────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing (saves VRAM)
    model.gradient_checkpointing_enable()

    # ── Load Dataset ──────────────────────────────
    train_dataset = UDVideoQADataset(args.train_data, max_samples=args.max_samples)
    val_dataset = UDVideoQADataset(args.val_data, max_samples=min(500, args.max_samples if args.max_samples > 0 else 500))

    # ── Training Arguments ────────────────────────
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"udvideoqa_lora_r{args.lora_r}_lr{args.lr}" if args.use_wandb else None,
    )

    # ── Trainer ───────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(collate_fn, processor=processor),
    )

    # ── Train ─────────────────────────────────────
    logger.info("🚀 Starting LoRA fine-tuning...")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch: {args.batch_size} × {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    logger.info(f"   LR: {args.lr}")
    logger.info(f"   LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    logger.info(f"   Train samples: {len(train_dataset)}")
    logger.info(f"   Val samples: {len(val_dataset)}")

    trainer.train()

    # ── Save ──────────────────────────────────────
    final_path = f"{args.output_dir}/final"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"✅ Model saved to {final_path}")

    # Save training config
    config_path = f"{args.output_dir}/training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"✅ Config saved to {config_path}")

    logger.info("🎉 Training complete!")
    logger.info(f"   Log file: {LOG_FILE}")
    logger.info(f"   Next: python src/baseline/run_zero_shot.py --model {final_path}")


if __name__ == "__main__":
    main()
