#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined QA + Book Training

- QA (~70%): system → user → assistant
- Book (~30%): system → assistant (vocabulary only)
- Eval only on QA
"""

import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported

import json
import random
from collections import defaultdict
from pathlib import Path
import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "models" / "labkovsky-qa-book-combined"

QA_FILES = [
    PROJECT_ROOT / "data" / "fine_tuning" / "qa_rs_final.jsonl",
    PROJECT_ROOT / "data" / "fine_tuning" / "anti_generic_clean.jsonl",
    PROJECT_ROOT / "data" / "processed" / "interviews.jsonl",
]
BOOK_FILE = PROJECT_ROOT / "data" / "processed" / "book_text_only.jsonl"

MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
MAX_SEQ_LENGTH = 1024
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4
EPOCHS = 20
VAL_RATIO = 0.15
SEED = 42

QA_SYSTEM = "Ты — психолог Михаил Лабковский. Отвечай прямо, уверенно, с конкретными рекомендациями."
BOOK_SYSTEM = "Продолжи текст Лабковского."

# =============================================================================
# DATA
# =============================================================================

def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_data():
    # Load QA
    qa = []
    for fp in QA_FILES:
        for obj in load_jsonl(fp):
            q = obj.get("question", "")
            a = obj.get("answer_segmented") or obj.get("answer", "")
            if q.strip() and a.strip():
                qa.append({"q": q.strip(), "a": a.strip(), "type": "qa"})
    print(f"QA: {len(qa)}")

    # Load Book
    book = []
    for obj in load_jsonl(BOOK_FILE):
        text = obj.get("text", "").strip()
        if text:
            book.append({"text": text, "type": "book"})
    print(f"Book: {len(book)}")

    return qa, book


def split_qa(qa, val_ratio, seed):
    """Split by unique answers."""
    groups = defaultdict(list)
    for r in qa:
        groups[r["a"]].append(r)

    groups = list(groups.values())
    random.seed(seed)
    random.shuffle(groups)

    n_eval = max(1, int(len(groups) * val_ratio))
    train = [r for g in groups[n_eval:] for r in g]
    eval_ = [r for g in groups[:n_eval] for r in g]

    random.shuffle(train)
    random.shuffle(eval_)
    return train, eval_


def format_example(r, tokenizer):
    if r["type"] == "qa":
        msgs = [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": r["q"]},
            {"role": "assistant", "content": r["a"]},
        ]
    else:
        msgs = [
            {"role": "system", "content": BOOK_SYSTEM},
            {"role": "assistant", "content": r["text"]},
        ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

# =============================================================================
# COLLATOR (completion-only)
# =============================================================================

class CompletionOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.template_ids = tokenizer("<s>assistant\n", add_special_tokens=False)["input_ids"]

    def __call__(self, features):
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()

        for i, ids in enumerate(batch["input_ids"]):
            ids_list = ids.tolist()
            pos = self._find(ids_list, self.template_ids)
            if pos >= 0:
                labels[i, :pos + len(self.template_ids)] = -100
            if batch.get("attention_mask") is not None:
                labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch

    def _find(self, haystack, needle):
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 50)
    print("Combined QA + Book Training")
    print("=" * 50)

    # Load data
    qa, book = load_all_data()
    qa_train, qa_eval = split_qa(qa, VAL_RATIO, SEED)

    # Combine for training
    train_data = qa_train + book
    random.seed(SEED)
    random.shuffle(train_data)

    print(f"\nTrain: {len(qa_train)} QA + {len(book)} book = {len(train_data)}")
    print(f"Eval: {len(qa_eval)} QA only")

    # Model
    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0,
        target_modules=TARGET_MODULES, bias="none",
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )
    model.print_trainable_parameters()

    # Format
    train_ds = Dataset.from_dict({"text": [format_example(r, tokenizer) for r in train_data]})
    eval_ds = Dataset.from_dict({"text": [format_example(r, tokenizer) for r in qa_eval]})

    # Print full examples for verification
    print("\n" + "=" * 50)
    print("QA EXAMPLE (full):")
    print("=" * 50)
    qa_sample = next(r for r in train_data if r["type"] == "qa")
    print(format_example(qa_sample, tokenizer))

    print("\n" + "=" * 50)
    print("BOOK EXAMPLE (full):")
    print("=" * 50)
    book_sample = next(r for r in train_data if r["type"] == "book")
    print(format_example(book_sample, tokenizer))

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CompletionOnlyCollator(tokenizer),
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",
            seed=SEED,
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            packing=False,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\nTraining: r={LORA_R}, alpha={LORA_ALPHA}, lr={LR}")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
