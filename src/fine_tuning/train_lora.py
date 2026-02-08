#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Vikhr-YandexGPT on Labkovsky Q&A data.

Reconstructed from checkpoint-1130 settings (eval_loss 0.7376).

Uses the model's NATIVE chat template (via tokenizer.apply_chat_template)
instead of custom Вопрос/Ответ format, so training and inference use
the same format the model was originally trained with.

IMPORTANT: Full sequence training is used intentionally (no completion-only loss).
The model learns from both questions AND answers, which is critical for
Labkovsky's reactive style — his answers directly respond to the emotional
tone and specific wording of the question.

Usage:
    python train_lora.py
"""

import json
import random
import sys
import torch
import gc
from functools import partial
from pathlib import Path
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


class MemoryCleanupCallback(TrainerCallback):
    """Periodically clear CUDA cache to prevent memory buildup."""

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
# SETTINGS (from checkpoint-1130)
# ==============================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "labkovsky-vikhr-yandex-lora"

# Data sources (4 datasets — same as checkpoint-1130)
INPUT_FILES = {
    "qa_corpus": FINE_TUNING_DIR / "qa_rs_corpus_short.jsonl",
    "interview": PROCESSED_DIR / "interviews.jsonl",
    "book": PROCESSED_DIR / "Hochu_i_budu_with_questions.jsonl",
    "articles": PROCESSED_DIR / "articles_with_questions.jsonl",
}

# Model
MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"

# LoRA settings (from adapter_config.json)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.02

# Training settings (from trainer_state.json)
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 15
WARMUP_STEPS = 20
VAL_RATIO = 0.15
RANDOM_SEED = 42


# ==============================================================
# DATA LOADING
# ==============================================================

def load_all_data(input_files: dict) -> list:
    """
    Load training data from all specified JSONL files.

    Handles two formats:
      - qa_corpus: has 'question' and 'answer' fields directly
      - interviews/book/articles: has 'text' (answer) and 'potential_questions' (list),
        we take only the first question from the list
    """
    records = []

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

                # Format A: direct question/answer (qa_corpus)
                if 'question' in data and 'answer' in data:
                    question = data['question']
                    answer = data['answer']
                # Format B: text + potential_questions (interviews, book, articles)
                elif 'text' in data and 'potential_questions' in data:
                    questions_list = data['potential_questions']
                    if not questions_list:
                        continue
                    question = questions_list[0]  # Take only the first question
                    answer = data['text']
                else:
                    continue

                if question and answer:
                    records.append({
                        'question': question,
                        'answer': answer,
                        'source': source_name
                    })
                    count += 1

        print(f"  {source_name} ({filepath.name}): {count} examples")

    print(f"Total: {len(records)} examples")
    return records


def format_for_sft(examples, tokenizer):
    """
    Format using the model's native chat template.

    Uses tokenizer.apply_chat_template() so the training data matches
    exactly what the model expects at inference time. No custom format needed.

    FULL SEQUENCE TRAINING: We return the complete formatted text (user + assistant).
    No DataCollatorForCompletionOnlyLM is used — the model calculates loss on ALL
    tokens, learning both to understand questions and generate Labkovsky-style answers.
    """
    texts = []

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]

        # Build messages in the standard chat format
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        # Let the tokenizer apply the model's native template
        # tokenize=False returns a string (not token IDs) — SFTTrainer handles tokenization
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # We include the full assistant response
        )
        texts.append(text)

    return {"text": texts}


# ==============================================================
# MAIN
# ==============================================================

def main():
    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 60)
    print("Labkovsky Fine-tuning (Vikhr + LoRA r=32)")
    print("Reconstructed from checkpoint-1130")
    print("Using native chat template + full sequence training")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    print(f"BF16: {'supported' if bf16_supported else 'not supported'}")

    # Load data
    print(f"\nLoading data from 4 sources...")
    records = load_all_data(INPUT_FILES)

    if not records:
        print("No data found!")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    dataset = Dataset.from_list(records)

    # Split
    print(f"\nSplitting: {int((1-VAL_RATIO)*100)}% train / {int(VAL_RATIO*100)}% val")
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Quantization
    print(f"\nSetting up 4-bit quantization...")
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

    # Show what the chat template looks like
    print(f"\n--- Chat template preview ---")
    sample_messages = [
        {"role": "user", "content": "Тестовый вопрос?"},
        {"role": "assistant", "content": "Тестовый ответ."},
    ]
    preview = tokenizer.apply_chat_template(sample_messages, tokenize=False, add_generation_prompt=False)
    print(repr(preview))
    print("--- End preview ---\n")

    # LoRA
    print(f"Preparing LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
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

    # Format data using native chat template
    print(f"\nFormatting data with native chat template...")
    format_fn = partial(format_for_sft, tokenizer=tokenizer)
    train_dataset = train_dataset.map(format_fn, batched=True)
    eval_dataset = eval_dataset.map(format_fn, batched=True)

    print(f"\nSample (first 400 chars):\n{train_dataset[0]['text'][:400]}\n")

    # Trainer
    # IMPORTANT: No DataCollatorForCompletionOnlyLM — this is intentional.
    # Full sequence training: loss is computed on ALL tokens (question + answer).
    # This ensures the model learns Labkovsky's reactive conversational dynamics,
    # not just pattern-matching on answer tokens.
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            MemoryCleanupCallback(cleanup_every_n_steps=50),
        ],
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Full sequence training: YES (no completion-only loss)")

    trainer.train()

    # Save
    print(f"\nSaving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()