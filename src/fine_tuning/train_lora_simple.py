#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_lora_simple.py

Simplified version WITHOUT RS classifier.
Vikhr-YandexGPT-5-Lite-8B-it fine-tuning with Unsloth + LoRA + TRL SFTTrainer.

Format: simple chat template with question -> answer (no RS prefix).
"""

# ---- MUST BE FIRST ----
import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported

# ---- standard imports (after Unsloth) ----
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict, Counter

import numpy as np
import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback

from trl import SFTTrainer, SFTConfig


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR      = PROJECT_ROOT / "models" / "labkovsky-vikhr-lora-simple"

INPUT_FILES = {
    "qa_corpus":     FINE_TUNING_DIR / "qa_rs_corpus_short.jsonl",
    "interview":     PROCESSED_DIR   / "interviews.jsonl",
    "book":          PROCESSED_DIR   / "Hochu_i_budu_with_questions.jsonl",
    "articles":      PROCESSED_DIR   / "articles_with_questions.jsonl",
    "anti_generic":  FINE_TUNING_DIR / "anti_generic_clean.jsonl",
}

MODEL_NAME     = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
MAX_SEQ_LENGTH = 1024

# LoRA
LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.0

# Training
BATCH_SIZE            = 2
GRADIENT_ACCUMULATION = 4  # total batch = 8
LEARNING_RATE         = 2e-5
NUM_EPOCHS            = 15
WARMUP_STEPS          = 20
VAL_RATIO             = 0.15
RANDOM_SEED           = 42

# Completion-only boundary
RESPONSE_TEMPLATE = "<s>assistant\n"
IGNORE_INDEX = -100


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
    records: List[dict] = []
    print("\nLoading data...")
    skipped_short = 0

    for src, fp in input_files.items():
        if not fp.exists():
            print(f"  WARNING: {fp} not found — skipping")
            continue

        cnt = 0
        for obj in iter_jsonl(fp):
            # Skip short answers
            if obj.get("short_answer", False):
                skipped_short += 1
                continue

            # Format A: {"question":..., "answer":...}
            if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                q = obj.get("question")
                a = obj.get("answer")
                if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                    records.append({"question": q.strip(), "answer": a.strip(), "source": src})
                    cnt += 1
                continue

            # Format B: {"text":..., "potential_questions":[...]}
            if isinstance(obj, dict) and "text" in obj and "potential_questions" in obj:
                text = obj.get("text")
                qs = obj.get("potential_questions") or []
                if isinstance(text, str) and qs and isinstance(qs[0], str):
                    q = qs[0].strip()
                    a = text.strip()
                    if q and a:
                        records.append({"question": q, "answer": a, "source": src})
                        cnt += 1

        print(f"  {src} ({fp.name}): {cnt} examples")

    print(f"Total: {len(records)} examples")
    if skipped_short:
        print(f"  Skipped: {skipped_short} short_answer")
    return records


# =============================================================================
# FORMATTING (NO RS)
# =============================================================================

def format_for_sft(examples, tokenizer):
    """
    Simple chat template: user question -> assistant answer.
    No RS prefix.
    """
    texts = []
    for i in range(len(examples["question"])):
        q = examples["question"][i]
        a = examples["answer"][i]
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


# =============================================================================
# AUDIT
# =============================================================================

def audit_records(records: List[dict], tokenizer, max_seq: int, n_show: int = 2):
    by_src = defaultdict(list)
    issues = []

    for r in records:
        by_src[r.get("source", "unknown")].append(r)
        if isinstance(r.get("answer"), str) and len(r["answer"].strip()) < 20:
            issues.append(("very_short_answer", r.get("source", "unknown")))

    print("\n=== AUDIT: counts per source ===")
    print({k: len(v) for k, v in by_src.items()})

    if issues:
        print("\n=== AUDIT: issues summary ===")
        print(Counter(issues))

    print("\n=== AUDIT: token length stats ===")
    for src, items in by_src.items():
        lens = []
        over = 0
        for r in items:
            messages = [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["answer"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            L = len(ids)
            lens.append(L)
            if L > max_seq:
                over += 1

        lens = np.array(lens) if lens else np.array([0])
        print(f"[{src}] n={len(items)}  p50={np.median(lens):.0f}  p90={np.percentile(lens,90):.0f}  "
              f"max={lens.max():.0f}  >max_seq={over} ({(over/len(items)) if items else 0:.1%})")


# =============================================================================
# CUSTOM COMPLETION-ONLY COLLATOR
# =============================================================================

@dataclass
class CompletionOnlyCollator:
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

        # Pre-compute whitespace token IDs to skip after assistant header
        self.whitespace_ids = set()
        for ws in ["\n", " ", "\t", "\r"]:
            try:
                ws_tokens = self.tokenizer(ws, add_special_tokens=False)["input_ids"]
                if ws_tokens:
                    self.whitespace_ids.add(ws_tokens[0])
            except:
                pass

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
        eos_id = self.tokenizer.eos_token_id

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            pos = self._find_sublist(ids, self.template_ids)
            if pos == -1:
                labels[i, :] = self.ignore_index
                not_found += 1
                continue

            start_loss = pos + len(self.template_ids)

            # Skip leading whitespace after assistant header
            while start_loss < len(ids) and ids[start_loss] in self.whitespace_ids:
                start_loss += 1

            labels[i, :start_loss] = self.ignore_index

            if attention_mask is not None:
                labels[i, attention_mask[i] == 0] = self.ignore_index

            if eos_id is not None:
                labels[i, input_ids[i] == eos_id] = self.ignore_index

        batch["labels"] = labels

        if not_found:
            print(f"[COLLATOR] WARNING: template not found in {not_found}/{input_ids.size(0)} samples")

        return batch


def mask_stats(trainer, batches: int = 20):
    dl = trainer.get_train_dataloader()
    ratios = []
    for _, batch in zip(range(batches), dl):
        labels = batch["labels"][0].tolist()
        kept = sum(x != IGNORE_INDEX for x in labels)
        ratios.append(kept / max(1, len(labels)))
    print(f"[MASK STATS] n={len(ratios)}  min={min(ratios):.2%}  mean={np.mean(ratios):.2%}  max={max(ratios):.2%}")


# =============================================================================
# MAIN
# =============================================================================

def build_trainer(model, tokenizer, train_ds, eval_ds, cfg, collator):
    try:
        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=cfg,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
    except TypeError:
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=cfg,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )


def main():
    print("=" * 60)
    print("Labkovsky Fine-tuning (SIMPLE - no RS)")
    print("Vikhr + Unsloth + LoRA r=32/alpha=64")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"BF16: {'supported' if is_bfloat16_supported() else 'not supported'}")

    # Load records
    records = load_all_data(INPUT_FILES)

    # Split
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"\nSplit: Train={len(train_ds)}, Eval={len(eval_ds)}")

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

    # Prevent double BOS/EOS
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False

    # LoRA
    print(f"\nPreparing LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    model.print_trainable_parameters()

    # Audit
    audit_records(records, tokenizer, max_seq=MAX_SEQ_LENGTH, n_show=2)

    # Format
    print("\nFormatting dataset with chat template...")
    train_ds = train_ds.map(lambda ex: format_for_sft(ex, tokenizer), batched=True)
    eval_ds  = eval_ds.map(lambda ex: format_for_sft(ex, tokenizer), batched=True)

    print("\nSample (first 400 chars):")
    print(train_ds[0]["text"][:400])

    # Collator
    collator = CompletionOnlyCollator(tokenizer=tokenizer, response_template=RESPONSE_TEMPLATE)

    # Trainer config
    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        completion_only_loss=True,
        packing=False,
        group_by_length=False,
        dataset_kwargs={"add_special_tokens": False},
    )

    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, cfg, collator)

    # Mask stats
    mask_stats(trainer, batches=20)

    print("\nStarting training...")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Completion-only: YES (response_template={RESPONSE_TEMPLATE!r})")
    print(f"  Train batch effective: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    trainer.train()

    print(f"\nSaving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
