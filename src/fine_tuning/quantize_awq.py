#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantize merged model to W4A16 (4-bit weights) using llm-compressor.

Uses QA training data for calibration to preserve model quality.
Memory-optimized: clears caches before saving.

Usage (WSL required):
    source venv_wsl/bin/activate
    python src/fine_tuning/quantize_awq.py
"""

import gc
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_MODEL = PROJECT_ROOT / "models" / "vikhr-labkovsky-final"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "vikhr-labkovsky-awq"
QA_DATA = PROJECT_ROOT / "data" / "fine_tuning" / "qa_clean.jsonl"

# Calibration settings
N_SAMPLES = 128  # Number of samples for calibration
MAX_SEQ_LEN = 512  # Max sequence length for calibration


# =============================================================================
# MEMORY UTILS
# =============================================================================

def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("[Memory cleared]")


def print_memory_stats():
    """Print current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved]")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_calibration_dataset(qa_path: Path, n_samples: int = 128) -> Dataset:
    """Load QA data and format as HuggingFace Dataset for calibration."""
    texts = []

    with open(qa_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            try:
                obj = json.loads(line.strip())
                question = obj.get("question", "")
                answer = obj.get("answer", "")
                if question and answer:
                    # Format as conversation
                    text = f"Вопрос: {question}\n\nОтвет: {answer}"
                    texts.append({"text": text})
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(texts)} calibration samples")
    return Dataset.from_list(texts)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("W4A16 Quantization: vikhr-labkovsky-final -> 4-bit")
    print("Using llm-compressor (GPTQ) - Memory Optimized")
    print("=" * 60)
    print()

    if not INPUT_MODEL.exists():
        raise FileNotFoundError(f"Input model not found: {INPUT_MODEL}")
    if not QA_DATA.exists():
        raise FileNotFoundError(f"QA data not found: {QA_DATA}")

    print(f"Input model: {INPUT_MODEL}")
    print(f"Output: {OUTPUT_MODEL}")
    print()

    # Load calibration data
    print("Loading calibration data...")
    calib_dataset = load_calibration_dataset(QA_DATA, n_samples=N_SAMPLES)

    # Define GPTQ quantization config
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",  # 4-bit weights, 16-bit activations
        ignore=["lm_head"],  # Don't quantize output layer
    )

    print("\nStarting quantization...")
    print("  (This takes 20-60 minutes depending on hardware)")
    print_memory_stats()
    print()

    # Clear any existing cache before starting
    clear_memory()

    # Run one-shot quantization
    # oneshot handles model loading, quantization, and saving
    oneshot(
        model=str(INPUT_MODEL),
        dataset=calib_dataset,
        recipe=recipe,
        output_dir=str(OUTPUT_MODEL),
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=N_SAMPLES,
        save_compressed=True,
    )

    print("\nDone!")
    print(f"Quantized model saved to: {OUTPUT_MODEL}")

    # Final cleanup
    clear_memory()

    # Calculate sizes
    if OUTPUT_MODEL.exists():
        input_size = sum(f.stat().st_size for f in INPUT_MODEL.glob("*.safetensors"))
        output_files = list(OUTPUT_MODEL.glob("*.safetensors"))
        if output_files:
            output_size = sum(f.stat().st_size for f in output_files)
            print("\nModel size comparison:")
            print(f"  Original: {input_size / 1e9:.2f} GB")
            print(f"  Quantized: {output_size / 1e9:.2f} GB")
            print(f"  Compression: {input_size / output_size:.1f}x")


if __name__ == "__main__":
    main()
