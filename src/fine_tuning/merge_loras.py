#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Two LoRA Adapters

Combines Q&A LoRA (task/style) with Book LoRA (vocabulary).
Uses PEFT's add_weighted_adapter to merge.

Usage:
    python src/fine_tuning/merge_loras.py

Adapters:
    - Q&A LoRA: attention + MLP (task formatting, response style)
    - Book LoRA: MLP only (vocabulary, expressions)

For overlapping modules (MLP), weights are averaged.
For non-overlapping modules (attention), Q&A adapter is used as-is.
"""

import torch
from pathlib import Path
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_DIR / "models"

# Input adapters
QA_LORA = MODELS_DIR / "labkovsky-vikhr-lora-unsloth"
BOOK_LORA = MODELS_DIR / "labkovsky-book-lora-r8-mlp"

# Output merged adapter
OUTPUT_DIR = MODELS_DIR / "labkovsky-merged-qa-book"

# Base model
BASE_MODEL = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"

# Merge weights: how much of each adapter to use
# For overlapping modules (MLP), this determines the blend
QA_WEIGHT = 0.7
BOOK_WEIGHT = 0.3

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("LoRA Merger: Q&A + Book Vocabulary")
    print("=" * 60)
    print()

    # Check adapters exist
    if not QA_LORA.exists():
        raise FileNotFoundError(f"Q&A LoRA not found: {QA_LORA}")
    if not BOOK_LORA.exists():
        raise FileNotFoundError(f"Book LoRA not found: {BOOK_LORA}")

    print(f"Q&A LoRA: {QA_LORA}")
    print(f"Book LoRA: {BOOK_LORA}")
    print(f"Merge weights: Q&A={QA_WEIGHT}, Book={BOOK_WEIGHT}")
    print()

    # Load configs to show info
    qa_config = PeftConfig.from_pretrained(str(QA_LORA))
    book_config = PeftConfig.from_pretrained(str(BOOK_LORA))

    print(f"Q&A config: r={qa_config.r}, alpha={qa_config.lora_alpha}")
    print(f"  modules: {qa_config.target_modules}")
    print(f"Book config: r={book_config.r}, alpha={book_config.lora_alpha}")
    print(f"  modules: {book_config.target_modules}")
    print()

    # Load base model
    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print()

    # Load first adapter (Q&A)
    print("Loading Q&A adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        str(QA_LORA),
        adapter_name="qa",
    )

    # Load second adapter (Book)
    print("Loading Book adapter...")
    model.load_adapter(str(BOOK_LORA), adapter_name="book")

    # List adapters
    print()
    print(f"Loaded adapters: {list(model.peft_config.keys())}")

    # Merge adapters with weighted average
    print()
    print(f"Merging with weights: qa={QA_WEIGHT}, book={BOOK_WEIGHT}")

    # add_weighted_adapter creates a new adapter from weighted combination
    model.add_weighted_adapter(
        adapters=["qa", "book"],
        weights=[QA_WEIGHT, BOOK_WEIGHT],
        adapter_name="merged",
        combination_type="linear",  # linear combination of weights
    )

    # Set merged as active
    model.set_adapter("merged")

    print("Merge complete!")
    print()

    # Save merged adapter
    print(f"Saving merged adapter to: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save only the merged adapter
    model.save_pretrained(
        str(OUTPUT_DIR),
        selected_adapters=["merged"],
    )
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print()
    print("Done!")
    print()
    print(f"Merged adapter saved to: {OUTPUT_DIR}")
    print()
    print("To use in query_rag.py, update LORA_PATH:")
    print(f'  LORA_PATH = MODELS_DIR / "labkovsky-merged-qa-book"')


if __name__ == "__main__":
    main()
