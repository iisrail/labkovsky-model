#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lora_original.py

Replica of checkpoint-1130 run (eval_loss 0.7376).
Settings extracted from training_args.bin and adapter_config.json.

Uses CUSTOM COLLATOR for completion-only loss since TRL's built-in
completion_only_loss doesn't work correctly with Vikhr chat template.

Key settings:
- Custom CompletionOnlyCollator with response_template="<s>assistant\n"
- gradient_accumulation_steps=16
- max_length=512
- optim=paged_adamw_8bit
- LoRA r=32, alpha=64, dropout=0.02
"""

import json
import random
import torch
import gc
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ==============================================================
# CUSTOM COMPLETION-ONLY COLLATOR
# ==============================================================

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


class MemoryCleanupCallback(TrainerCallback):
    """Periodically clear CUDA cache."""

    def __init__(self, cleanup_every_n_steps: int = 50):
        self.cleanup_every_n_steps = cleanup_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.cleanup_every_n_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================
# SETTINGS (exact copy from checkpoint-1130)
# ==============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "labkovsky-vikhr-lora-original"

INPUT_FILES = {
    "qa_corpus": FINE_TUNING_DIR / "qa_rs_corpus_short.jsonl",
    "interview": PROCESSED_DIR / "interviews.jsonl",
    "book": PROCESSED_DIR / "Hochu_i_budu_with_questions.jsonl",
    "articles": PROCESSED_DIR / "articles_with_questions.jsonl",
    "anti_generic": FINE_TUNING_DIR / "anti_generic_clean.jsonl",
}

MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"

# LoRA (from adapter_config.json)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.02

# Training (from training_args.bin)
MAX_SEQ_LENGTH = 512  # Original used 512, not 1024!
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16  # Original used 16, not 8!
LEARNING_RATE = 2e-5
NUM_EPOCHS = 15
WARMUP_STEPS = 20
VAL_RATIO = 0.15
RANDOM_SEED = 42


# ==============================================================
# DATA LOADING
# ==============================================================

def load_all_data(input_files: dict) -> list:
    records = []
    skipped = 0

    for source_name, filepath in input_files.items():
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found — skipping")
            continue

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip short answers
                if data.get("short_answer", False):
                    skipped += 1
                    continue

                # Format A: question/answer
                if 'question' in data and 'answer' in data:
                    q = data['question']
                    a = data['answer']
                    if q and a:
                        records.append({'question': q, 'answer': a, 'source': source_name})
                        count += 1
                # Format B: text + potential_questions (use ALL questions)
                elif 'text' in data and 'potential_questions' in data:
                    questions = data.get('potential_questions') or []
                    for q in questions:
                        if q and q.strip():
                            records.append({
                                'question': q.strip(),
                                'answer': data['text'],
                                'source': source_name,
                            })
                            count += 1

        print(f"  {source_name}: {count} examples")

    print(f"Total: {len(records)} examples")
    if skipped:
        print(f"  Skipped: {skipped} short_answer")
    return records


def format_for_sft(examples, tokenizer):
    """Format using chat template (same as original run)."""
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


# ==============================================================
# MAIN
# ==============================================================

def main():
    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 60)
    print("Labkovsky Fine-tuning (ORIGINAL checkpoint-1130 settings)")
    print("TRL completion_only_loss=True (built-in)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"CUDA: {device_name} ({vram_gb:.1f} GB)")

    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"BF16: {'supported' if bf16_supported else 'NOT supported'}")

    # Load data
    print(f"\nLoading data...")
    records = load_all_data(INPUT_FILES)

    if not records:
        print("No data found!")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Split: Train={len(train_dataset)}, Eval={len(eval_dataset)}")

    # Quantization (4-bit nf4, same as original)
    print(f"\nSetting up 4-bit quantization...")
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA (exact settings from adapter_config.json)
    print(f"Preparing LoRA (r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # Format data
    print(f"\nFormatting data with chat template...")
    format_fn = partial(format_for_sft, tokenizer=tokenizer)
    train_dataset = train_dataset.map(format_fn, batched=True)
    eval_dataset = eval_dataset.map(format_fn, batched=True)

    print(f"\nSample (first 500 chars):\n{train_dataset[0]['text'][:500]}\n")

    # Custom collator for completion-only loss
    collator = CompletionOnlyCollator(tokenizer=tokenizer)

    # Trainer config (settings from training_args.bin)
    # Using CUSTOM COLLATOR instead of TRL's completion_only_loss
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",  # Original used this
        seed=RANDOM_SEED,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,  # 512, not 1024!
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=True,
        dataset_kwargs={"add_special_tokens": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=sft_config,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            MemoryCleanupCallback(cleanup_every_n_steps=50),
        ],
    )

    print(f"\nStarting training (ORIGINAL settings + custom collator)...")
    print(f"  max_length: {MAX_SEQ_LENGTH}")
    print(f"  Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Optimizer: paged_adamw_8bit")
    print(f"  Completion-only: Custom collator (response_template='{RESPONSE_TEMPLATE!r}')")
    print(f"  Epochs: {NUM_EPOCHS}")

    trainer.train()

    print(f"\nSaving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
