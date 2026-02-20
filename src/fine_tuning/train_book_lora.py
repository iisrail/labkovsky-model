#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Book Vocabulary LoRA Training

Train a small LoRA on raw book text to learn Labkovsky's vocabulary and style.
This adapter targets MLP modules (vocabulary/knowledge) not attention (reasoning).

Usage:
    python src/fine_tuning/train_book_lora.py

The resulting adapter can be merged with the Q&A LoRA for combined style + task.
"""

import torch
from pathlib import Path
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

BOOK_DATA = DATA_DIR / "book_text_only.jsonl"
OUTPUT_DIR = MODELS_DIR / "labkovsky-book-lora-r8-mlp"

# ============================================================
# MODEL CONFIG
# ============================================================

BASE_MODEL = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
MAX_SEQ_LENGTH = 1024

# LoRA config - MLP-only for vocabulary
LORA_R = 8
LORA_ALPHA = 16  # ratio 2
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]  # MLP only

# Training config
BATCH_SIZE = 4
GRAD_ACCUM = 2
EPOCHS = 4
LEARNING_RATE = 5e-4  # higher LR for vocabulary learning
WARMUP_STEPS = 10

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Book Vocabulary LoRA Training")
    print("MLP-only adapter for Labkovsky's vocabulary/style")
    print("=" * 60)

    # Device info
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"BF16: {'supported' if torch.cuda.is_bf16_supported() else 'not supported'}")
    else:
        print("WARNING: CUDA not available!")
    print()

    # Check data file
    if not BOOK_DATA.exists():
        raise FileNotFoundError(f"Book data not found: {BOOK_DATA}")

    # Load dataset
    print(f"Loading data: {BOOK_DATA}")
    dataset = load_dataset("json", data_files=str(BOOK_DATA))["train"]
    print(f"  Loaded {len(dataset)} text chunks")
    print()

    # Load model
    print(f"Loading model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    print()

    # Add LoRA
    print(f"Adding LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Target modules: {TARGET_MODULES}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    print()

    # Training config
    use_bf16 = torch.cuda.is_bf16_supported()

    config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        logging_steps=5,
        save_strategy="epoch",
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_8bit",
        seed=42,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=config,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    trainer.train()

    # Save
    print()
    print(f"Saving adapter to: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print()
    print("Done!")
    print()
    print("Next steps:")
    print("  1. Merge with Q&A LoRA using PEFT add_weighted_adapter()")
    print("  2. Or load both adapters and switch between them")


if __name__ == "__main__":
    main()
