#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dpo.py

DPO (Direct Preference Optimization) training on top of SFT model.
Reinforces Labkovsky-style responses over generic/copy ones using
chosen/rejected pairs from labeled eval data.

Pipeline:
    1. python src/fine_tuning/build_dpo_pairs.py   # build pairs JSONL
    2. python src/fine_tuning/train_dpo.py          # this script

Base: vikhr-book-merged + SFT LoRA (merged at load time)
DPO trains a new LoRA adapter on top.

Uses standard transformers/PEFT/TRL (no Unsloth) to avoid dtype conflicts
when merging SFT LoRA into 4-bit quantized base.
"""

import json
import random
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"
INPUT_FILE = FINE_TUNING_DIR / "dpo_pairs.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "labkovsky-dpo-lora-v1"

# Base model (book LoRA + SFT LoRA v10 already merged in full precision)
MODEL_NAME = "./models/vikhr-v10-merged"

MAX_SEQ_LENGTH = 3200

# DPO LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# DPO hyperparameters
BETA = 0.1                      # DPO temperature (standard)
BATCH_SIZE = 1                  # limited by VRAM (pairs = 2x sequences)
GRADIENT_ACCUMULATION = 8       # effective batch = 8
LEARNING_RATE = 5e-7            # DPO needs much lower LR than SFT
NUM_EPOCHS = 3                  # DPO converges fast
WARMUP_RATIO = 0.1
RANDOM_SEED = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dpo_data(input_file: Path) -> List[dict]:
    """Load DPO pairs JSONL. Each record has prompt, chosen, rejected as chat messages."""
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("prompt") and obj.get("chosen") and obj.get("rejected"):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(records)} DPO pairs from {input_file.name}")
    return records


def split_data(records: List[dict], val_ratio: float = 0.15, seed: int = 42):
    """Split by qa_id to prevent leakage (all pairs from same question go together)."""
    from collections import defaultdict
    qa_groups = defaultdict(list)
    for r in records:
        qa_groups[r.get("qa_id", "unknown")].append(r)

    group_keys = list(qa_groups.keys())
    random.seed(seed)
    random.shuffle(group_keys)

    n_eval = max(1, int(len(group_keys) * val_ratio))
    eval_keys = set(group_keys[:n_eval])

    train = [r for k in group_keys if k not in eval_keys for r in qa_groups[k]]
    eval_ = [r for k in eval_keys for r in qa_groups[k]]

    random.seed(seed)
    random.shuffle(train)
    random.shuffle(eval_)

    print(f"  Train: {len(train)} pairs ({len(group_keys) - n_eval} questions)")
    print(f"  Eval: {len(eval_)} pairs ({n_eval} questions)")
    return train, eval_


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Labkovsky DPO Training")
    print("Reinforcing style preferences on top of SFT model")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")

    if not INPUT_FILE.exists():
        print(f"\nERROR: {INPUT_FILE} not found!")
        print("Run first: python src/fine_tuning/build_dpo_pairs.py")
        return

    # --- Load data ---
    records = load_dpo_data(INPUT_FILE)
    if not records:
        print("No data found!")
        return

    train_ds = Dataset.from_list(records)  # use all data for training (no eval — OOM)

    # --- Load base model (4-bit quantized) ---
    print(f"\nLoading base model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO needs left padding for paired sequences

    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False

    # --- Prepare for k-bit training ---
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # --- Attach new DPO LoRA ---
    print(f"\nPreparing DPO LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- DPO config ---
    bf16_supported = torch.cuda.is_bf16_supported()
    cfg = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        beta=BETA,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        fp16=not bf16_supported,
        bf16=bf16_supported,
        logging_steps=5,
        eval_strategy="no",              # skip eval — OOM with ref model on 12GB
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH - 512,  # reserve space for responses
        gradient_checkpointing=True,
        precompute_ref_log_probs=True,  # compute ref logprobs once, saves VRAM during train/eval
    )

    # --- Trainer ---
    # With PEFT model and ref_model=None, DPOTrainer creates implicit reference
    # by disabling adapters (no extra VRAM needed)
    try:
        trainer = DPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            args=cfg,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            args=cfg,
        )

    # --- Train ---
    print("\nStarting DPO training...")
    print(f"  Base: {MODEL_NAME}")
    print(f"  DPO LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Modules: {TARGET_MODULES}")
    print(f"  Beta: {BETA}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Output: {OUTPUT_DIR}")

    trainer.train()

    # --- Save ---
    print(f"\nSaving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
