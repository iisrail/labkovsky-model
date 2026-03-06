#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_with_rag_context.py

Stage 2 LoRA training: teach the model to use RAG context in system prompt.
Base model is vikhr-book-merged (Stage 1 book LoRA already merged in).

Training data format matches inference exactly:
  system: [English prompt with retrieved docs as [Book/Article] and [QA Example]]
  user:   [question]
  assistant: [answer]  ← only this part is trained (completion-only)

Pipeline:
    1. python src/fine_tuning/build_rag_training_data.py   # build training JSONL
    2. python src/fine_tuning/train_with_rag_context.py     # this script
"""

# ---- MUST BE FIRST ----
# Unsloth patches transformers for 2x faster training + 60% less VRAM
import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported

# ---- standard imports (after Unsloth) ----
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig


# =============================================================================
# COMPLETION-ONLY COLLATOR
# =============================================================================
# Standard SFT trains on the entire sequence (system + user + assistant).
# We only want to train on the assistant response — the model should learn
# to *generate* answers, not memorize prompts.
#
# This collator finds the "<s>assistant\n" boundary in each sample and sets
# labels to IGNORE_INDEX (-100) for everything before it. The loss is only
# computed on the assistant's tokens.

RESPONSE_TEMPLATE = "<s>assistant\n"  # Vikhr chat template boundary
IGNORE_INDEX = -100  # PyTorch cross-entropy ignores this label


@dataclass
class CompletionOnlyCollator:
    """Masks prompt tokens (system + user), trains only on assistant response."""
    tokenizer: Any
    response_template: str = RESPONSE_TEMPLATE
    ignore_index: int = IGNORE_INDEX

    def __post_init__(self):
        # Pre-tokenize the template to find it in each sample's token IDs
        self.template_ids = self.tokenizer(
            self.response_template,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if not self.template_ids:
            raise ValueError("response_template tokenized to empty list.")
        print(f"[COLLATOR] Response template '{self.response_template!r}' -> {self.template_ids}")

    @staticmethod
    def _find_sublist(haystack: List[int], needle: List[int]) -> int:
        """Find first occurrence of needle in haystack. Returns -1 if not found."""
        n = len(needle)
        if n == 0 or len(haystack) < n:
            return -1
        for i in range(len(haystack) - n + 1):
            if haystack[i:i+n] == needle:
                return i
        return -1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Called by trainer on each batch. Builds labels that mask the prompt portion:
        1. Pad all samples to equal length
        2. Find '<s>assistant\\n' boundary in each sample's token IDs
        3. Set labels=-100 for everything before that boundary (system + user tokens)
        4. Only assistant response tokens contribute to loss
        If boundary not found (sequence truncated), entire sample is skipped.
        """
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        labels = input_ids.clone()
        not_found = 0

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            # Find where assistant response starts
            pos = self._find_sublist(ids, self.template_ids)
            if pos == -1:
                # Template not found — skip entire sample (truncated or malformed)
                labels[i, :] = self.ignore_index
                not_found += 1
                continue

            # Mask everything before and including the template marker
            start_loss = pos + len(self.template_ids)
            labels[i, :start_loss] = self.ignore_index

            # Also mask padding tokens
            if attention_mask is not None:
                labels[i, attention_mask[i] == 0] = self.ignore_index

        batch["labels"] = labels

        if not_found:
            print(f"[COLLATOR] WARNING: template not found in {not_found}/{input_ids.size(0)} samples")

        return batch


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"
OUTPUT_DIR = PROJECT_ROOT / "models" / "labkovsky-rag-context-lora-v10"

# Built by build_rag_training_data.py — each record has question, answer, system_prompt
INPUT_FILE = FINE_TUNING_DIR / "qa_with_rag_context.jsonl"

# Base model: Vikhr + book LoRA merged (Stage 1 already baked in)
MODEL_NAME = "./models/vikhr-book-merged"
MAX_SEQ_LENGTH = 3200  # fits system prompt (~2000 tok docs) + question + answer

# LoRA config — attention only (MLP already trained in book LoRA)
LORA_R = 8
LORA_ALPHA = 24
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters
BATCH_SIZE = 1                  # limited by VRAM with long sequences
GRADIENT_ACCUMULATION = 16      # effective batch = 16
LEARNING_RATE = 2e-5            # balanced LR with alpha=24
NUM_EPOCHS = 20                 # early stopping will cut this short (patience=2)
WARMUP_RATIO = 0.05             # ~1 epoch warmup
VAL_RATIO = 0.15                # 15% held out for eval
RANDOM_SEED = 42

RESUME_FROM_CHECKPOINT = False


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(input_file: Path) -> List[dict]:
    """Load JSONL records, requiring question + answer + system_prompt fields."""
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("question") and obj.get("answer") and obj.get("system_prompt"):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(records)} records from {input_file.name}")
    return records


def split_by_unique_answers(records: List[dict], val_ratio: float, seed: int):
    """
    Split train/eval by unique answers to prevent data leakage.

    Some questions share the same answer (e.g. anti_generic variants).
    Grouping by answer ensures the model can't memorize an answer from
    train and reproduce it verbatim for an eval question with the same answer.
    """
    answer_groups = defaultdict(list)
    for r in records:
        answer_groups[r["answer"]].append(r)

    groups = list(answer_groups.values())
    random.seed(seed)
    random.shuffle(groups)

    n_eval_groups = max(1, int(len(groups) * val_ratio))
    eval_groups = groups[:n_eval_groups]
    train_groups = groups[n_eval_groups:]

    train_records = [r for g in train_groups for r in g]
    eval_records = [r for g in eval_groups for r in g]

    random.shuffle(train_records)
    random.shuffle(eval_records)

    print(f"\nSplit by unique answers:")
    print(f"  Total unique answers: {len(groups)}")
    print(f"  Train: {len(train_groups)} groups -> {len(train_records)} records")
    print(f"  Eval: {len(eval_groups)} groups -> {len(eval_records)} records")

    return train_records, eval_records


# =============================================================================
# FORMATTING
# =============================================================================

def format_for_sft(examples, tokenizer):
    """
    Format each record as a full chat sequence using Vikhr's chat template.

    Result: <s>system\n{sys_prompt}</s><s>user\n{question}</s><s>assistant\n{answer}</s>

    The collator will mask everything up to <s>assistant\n so only the answer is trained.
    """
    texts = []
    for i in range(len(examples["question"])):
        q = examples["question"][i]
        a = examples["answer"][i]
        sys_prompt = examples["system_prompt"][i]

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return {"text": texts}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Labkovsky Fine-tuning WITH RAG CONTEXT")
    print("Training format matches inference format")
    print("System prompt includes retrieved documents")
    print("=" * 60)

    # --- GPU check ---
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")

    # --- Load data ---
    if not INPUT_FILE.exists():
        print(f"\nERROR: {INPUT_FILE} not found!")
        print("Run first: python src/fine_tuning/build_rag_training_data.py")
        return

    records = load_data(INPUT_FILE)
    if not records:
        print("No data found!")
        return

    train_records, eval_records = split_by_unique_answers(
        records, val_ratio=VAL_RATIO, seed=RANDOM_SEED
    )

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)

    # --- Load base model + tokenizer via Unsloth (4-bit quantized) ---
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,      # QLoRA — saves VRAM
        dtype=None,             # auto-detect (bf16 if supported)
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for causal LM training

    # Disable auto BOS/EOS — chat template handles these
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False

    # --- Attach LoRA adapters ---
    print(f"\nPreparing LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    model.print_trainable_parameters()

    # --- Format data as chat sequences ---
    print("\nFormatting dataset (system prompt with RAG docs)...")
    train_ds = train_ds.map(lambda ex: format_for_sft(ex, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda ex: format_for_sft(ex, tokenizer), batched=True)

    # Sanity check: print one sample's token length
    sample_tokens = tokenizer(train_ds[0]["text"], return_tensors="pt")["input_ids"].shape[-1]
    print(f"\nSample token length: {sample_tokens}")
    print(f"Sample (first 500 chars):\n{train_ds[0]['text'][:500]}")

    # --- Collator: masks prompt, trains only on assistant response ---
    collator = CompletionOnlyCollator(tokenizer=tokenizer)

    # --- SFTTrainer config ---
    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",             # smooth decay after warmup
        fp16=not is_bfloat16_supported(),       # use fp16 only if bf16 unavailable
        bf16=is_bfloat16_supported(),           # prefer bf16 for stability
        logging_steps=10,
        eval_strategy="epoch",                  # eval after each epoch
        save_strategy="epoch",                  # checkpoint after each epoch
        save_total_limit=3,                     # keep only 3 best checkpoints
        load_best_model_at_end=True,            # restore best checkpoint after training
        metric_for_best_model="eval_loss",
        greater_is_better=False,                # lower eval_loss = better
        optim="adamw_8bit",                     # 8-bit Adam — saves VRAM
        seed=RANDOM_SEED,
        report_to="none",                       # no wandb/tensorboard
        max_length=MAX_SEQ_LENGTH,              # truncate sequences beyond this
        dataset_text_field="text",              # column name in dataset
        packing=False,                          # no sequence packing (variable lengths)
        dataset_kwargs={"add_special_tokens": False},  # chat template already has them
    )

    # SFTTrainer API changed: newer versions use processing_class, older use tokenizer
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            args=cfg,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            args=cfg,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    # --- Train ---
    print("\nStarting training...")
    print(f"  Base model: {MODEL_NAME}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Modules: {TARGET_MODULES}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Batch effective: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  LR: {LEARNING_RATE}, early stopping patience: 2")
    print(f"  Output: {OUTPUT_DIR}")

    if RESUME_FROM_CHECKPOINT:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # --- Save LoRA adapter + tokenizer ---
    print(f"\nSaving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
