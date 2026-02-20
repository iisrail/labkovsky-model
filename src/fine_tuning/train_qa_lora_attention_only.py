#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qa_lora_attention_only.py

QA LoRA training with ATTENTION MODULES ONLY (no MLP).
This preserves the base model's knowledge/vocabulary while adapting style.

Key differences from full training:
- target_modules: q_proj, k_proj, v_proj, o_proj (attention only)
- r=8, alpha=16
- No book data (clean Q&A only)

Usage (WSL):
    source venv_wsl/bin/activate
    python src/fine_tuning/train_qa_lora_attention_only.py
"""

# ---- MUST BE FIRST ----
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
# COMPLETION-ONLY COLLATOR (train only on assistant response)
# =============================================================================

RESPONSE_TEMPLATE = "<s>assistant\n"
IGNORE_INDEX = -100


@dataclass
class CompletionOnlyCollator:
    """Masks prompt tokens, trains only on assistant response."""
    tokenizer: Any
    response_template: str = RESPONSE_TEMPLATE
    ignore_index: int = IGNORE_INDEX

    def __post_init__(self):
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
        n = len(needle)
        if n == 0 or len(haystack) < n:
            return -1
        for i in range(len(haystack) - n + 1):
            if haystack[i:i+n] == needle:
                return i
        return -1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        labels = input_ids.clone()
        not_found = 0

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            pos = self._find_sublist(ids, self.template_ids)
            if pos == -1:
                labels[i, :] = self.ignore_index
                not_found += 1
                continue

            # Mask everything up to and including the response template
            start_loss = pos + len(self.template_ids)
            labels[i, :start_loss] = self.ignore_index

            # Also mask padding
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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "labkovsky-vikhr-lora-attn5-no-anti"

# Clean Q&A data only - NO anti_generic (testing if it causes short answers)
INPUT_FILES = {
    "qa_final": FINE_TUNING_DIR / "qa_rs_final.jsonl",
    # "anti_generic": FINE_TUNING_DIR / "anti_generic_clean.jsonl",  # removed
    "interviews": PROCESSED_DIR / "interviews.jsonl",
}

MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
MAX_SEQ_LENGTH = 1024

# ATTENTION-ONLY LoRA config
LORA_R = 8
LORA_ALPHA = 16  # ratio 2
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention only, no MLP

# Training
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16  # effective batch = 16
LEARNING_RATE = 2e-4  # Higher LR for attention-only (fewer params)
NUM_EPOCHS = 20
WARMUP_RATIO = 0.05
VAL_RATIO = 0.15
RANDOM_SEED = 42

RESUME_FROM_CHECKPOINT = False

# System prompt for Labkovsky persona
SYSTEM_PROMPT = (
    "Ты — психолог Михаил Лабковский. "
    "Отвечай в его стиле: прямо, уверенно, с конкретными рекомендациями. "
    "Используй простой язык и жизненные примеры."
)


# =============================================================================
# DATA LOADING
# =============================================================================

def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_all_data(input_files: Dict[str, Path]) -> List[dict]:
    """Load data from all sources."""
    records: List[dict] = []
    print("\nLoading data...")

    for src, fp in input_files.items():
        if not fp.exists():
            print(f"  WARNING: {fp} not found — skipping")
            continue

        cnt = 0
        for obj in iter_jsonl(fp):
            # qa_rs_final.jsonl: {"id": ..., "question": ..., "answer_segmented": ...}
            if "question" in obj and "answer_segmented" in obj:
                q = obj.get("question")
                a = obj.get("answer_segmented")
                rec_id = obj.get("id", "")
                if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                    records.append({
                        "question": q.strip(),
                        "answer": a.strip(),
                        "source": src,
                        "id": rec_id
                    })
                    cnt += 1
                continue

            # anti_generic_clean.jsonl and interviews.jsonl: {"question": ..., "answer": ...}
            if "question" in obj and "answer" in obj:
                q = obj.get("question")
                a = obj.get("answer")
                if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                    records.append({
                        "question": q.strip(),
                        "answer": a.strip(),
                        "source": src,
                        "id": obj.get("id", "")
                    })
                    cnt += 1
                continue

        print(f"  {src} ({fp.name}): {cnt} examples")

    print(f"Total: {len(records)} examples")
    return records


def split_by_unique_answers(records: List[dict], val_ratio: float, seed: int):
    """
    Split ALL data by UNIQUE ANSWERS to prevent data leakage.
    """
    # Group ALL records by answer
    answer_groups = defaultdict(list)
    for r in records:
        answer_groups[r["answer"]].append(r)

    # Convert to list of groups
    groups = list(answer_groups.values())

    # Shuffle groups
    random.seed(seed)
    random.shuffle(groups)

    # Calculate split point (15% eval)
    n_eval_groups = max(1, int(len(groups) * val_ratio))

    # Split groups
    eval_groups = groups[:n_eval_groups]
    train_groups = groups[n_eval_groups:]

    # Flatten back to records
    train_records = [r for g in train_groups for r in g]
    eval_records = [r for g in eval_groups for r in g]

    # Shuffle within sets
    random.shuffle(train_records)
    random.shuffle(eval_records)

    print(f"\nSplit by unique answers (85/15):")
    print(f"  Total unique answers: {len(groups)}")
    print(f"  Train: {len(train_groups)} groups -> {len(train_records)} records")
    print(f"  Eval: {len(eval_groups)} groups -> {len(eval_records)} records")

    # Print eval IDs (for testing non-memorized samples)
    eval_ids = [r.get("id", "") for r in eval_records if r.get("id")]
    print(f"\nEval IDs ({len(eval_ids)} samples):")
    print(eval_ids[:50])  # Print first 50
    if len(eval_ids) > 50:
        print(f"  ... and {len(eval_ids) - 50} more")

    return train_records, eval_records


# =============================================================================
# FORMATTING
# =============================================================================

def format_for_sft(examples, tokenizer):
    """Format using Vikhr's native chat template with system prompt."""
    texts = []
    for i in range(len(examples["question"])):
        q = examples["question"][i]
        a = examples["answer"][i]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
    print("Labkovsky Fine-tuning (ATTENTION-ONLY LoRA)")
    print("Modules: q_proj, k_proj, v_proj, o_proj")
    print("Split by UNIQUE ANSWERS (no data leakage)")
    print("COMPLETION-ONLY training")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")
    print(f"BF16: {'supported' if is_bfloat16_supported() else 'not supported'}")

    # Load records
    records = load_all_data(INPUT_FILES)

    if not records:
        print("No data found!")
        return

    # Split by unique answers
    train_records, eval_records = split_by_unique_answers(
        records,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED
    )

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)

    # Model + tokenizer
    print(f"\nLoading model with Unsloth: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False

    # ATTENTION-ONLY LoRA
    print(f"\nPreparing ATTENTION-ONLY LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Target modules: {TARGET_MODULES}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    model.print_trainable_parameters()

    # Format data
    print("\nFormatting dataset with chat template + system prompt...")
    train_ds = train_ds.map(lambda ex: format_for_sft(ex, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda ex: format_for_sft(ex, tokenizer), batched=True)

    print("\nSample (first 500 chars):")
    print(train_ds[0]["text"][:500])

    # Completion-only collator
    collator = CompletionOnlyCollator(tokenizer=tokenizer)

    # Trainer config
    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
    )

    # Build trainer
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            args=cfg,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            args=cfg,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    print("\nStarting training...")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Modules: {TARGET_MODULES} (ATTENTION ONLY)")
    print(f"  Batch effective: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  LR: {LEARNING_RATE}, scheduler: cosine, warmup: {WARMUP_RATIO}")
    print(f"  Epochs: {NUM_EPOCHS}, early stopping patience: 3")
    print(f"  Split: by UNIQUE ANSWERS (no leakage)")
    print(f"  Output: {OUTPUT_DIR}")

    if RESUME_FROM_CHECKPOINT:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    print(f"\nSaving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
