#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Book LoRA into Base Model

Merges the book vocabulary LoRA adapter into Vikhr base weights.
Result: a standalone model with Labkovsky's vocabulary baked in.

Usage:
    python src/fine_tuning/merge_book_lora.py

Note: Requires ~16GB RAM. Uses CPU for merge to avoid VRAM limits.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_MODEL = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
BOOK_LORA = PROJECT_ROOT / "models" / "labkovsky-book-lora-r8-mlp"
OUTPUT_DIR = PROJECT_ROOT / "models" / "vikhr-book-merged"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 50)
    print("Merge Book LoRA into Base Model")
    print("=" * 50)

    if not BOOK_LORA.exists():
        raise FileNotFoundError(f"Book LoRA not found: {BOOK_LORA}")

    # Load base model in full precision on CPU
    print(f"\nLoading base model: {BASE_MODEL}")
    print("  (full precision, CPU - this may take a minute)")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",  # Use CPU to avoid VRAM limits
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print(f"  Model loaded: {model.dtype}")

    # Load LoRA adapter
    print(f"\nLoading LoRA adapter: {BOOK_LORA.name}")
    model = PeftModel.from_pretrained(model, str(BOOK_LORA))
    print(f"  Adapter loaded")

    # Merge adapter into base weights
    print("\nMerging adapter into base weights...")
    model = model.merge_and_unload()
    print("  Merge complete")

    # Free memory before saving
    import gc
    gc.collect()

    # Save merged model in smaller shards to reduce peak RAM
    print(f"\nSaving merged model to: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone!")
    print(f"Merged model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
