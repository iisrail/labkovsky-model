#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge RAG Context LoRA into vikhr-book-merged

Creates the final merged model with both book vocabulary (MLP) and
RAG context adaptation (attention + gate) baked in.

Usage:
    python src/fine_tuning/merge_final_lora.py

Note: Requires ~16GB RAM. Uses CPU for merge to avoid VRAM limits.
"""

import gc
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_MODEL = PROJECT_ROOT / "models" / "vikhr-book-merged"
RAG_LORA = PROJECT_ROOT / "models" / "labkovsky-rag-context-lora-v11"
OUTPUT_DIR = PROJECT_ROOT / "models" / "vikhr-labkovsky-final"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Merge RAG Context LoRA into vikhr-book-merged")
    print("=" * 60)

    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")
    if not RAG_LORA.exists():
        raise FileNotFoundError(f"RAG LoRA not found: {RAG_LORA}")

    # Load base model (vikhr-book-merged) in full precision on CPU
    print(f"\nLoading base model: {BASE_MODEL.name}")
    print("  (full precision, CPU - this may take a minute)")

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        torch_dtype=torch.float16,
        device_map="cpu",  # Use CPU to avoid VRAM limits
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), trust_remote_code=True)
    print(f"  Model loaded: {model.dtype}")

    # Load RAG context LoRA adapter
    print(f"\nLoading LoRA adapter: {RAG_LORA.name}")
    model = PeftModel.from_pretrained(model, str(RAG_LORA))
    print("  Adapter loaded")

    # Merge adapter into base weights
    print("\nMerging adapter into base weights...")
    model = model.merge_and_unload()
    print("  Merge complete")

    # Free memory before saving
    gc.collect()

    # Save merged model
    print(f"\nSaving merged model to: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone!")
    print(f"Final merged model saved to: {OUTPUT_DIR}")
    print("\nThis model includes:")
    print("  - Book LoRA (MLP: r=8, alpha=16) - vocabulary/knowledge")
    print("  - RAG Context LoRA (attn+gate: r=8, alpha=24) - style/reasoning")


if __name__ == "__main__":
    main()
