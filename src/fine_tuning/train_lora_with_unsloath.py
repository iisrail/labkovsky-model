#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Vikhr-YandexGPT on Labkovsky Q&A data using Unsloth.

Same training config as checkpoint-1130 (eval_loss 0.7376), but uses
Unsloth for ~2x faster training and ~60% less VRAM.

Uses RS classifier to predict response signal, then formats with
native chat template (RS prefix in user message).

Uses completion-only loss with DataCollatorForCompletionOnlyLM (response_template="Ответ:").

Requirements (install in this order):
    # Option 1: pip (install PyTorch FIRST, then Unsloth)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install unsloth
    pip install trl datasets

    # Option 2: WSL with existing venv
    # (if you already have venv_wsl with Unsloth working, just activate it)

Usage:
    python train_lora_with_unsloth.py
"""

import json
import random
import pickle
import torch
from functools import partial
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from sentence_transformers import SentenceTransformer


# ==============================================================
# SETTINGS (matching checkpoint-1130)
# ==============================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "labkovsky-vikhr-lora-v2"

# Data sources (5 datasets)
INPUT_FILES = {
    "qa_corpus": FINE_TUNING_DIR / "qa_rs_corpus_short.jsonl",
    "interview": PROCESSED_DIR / "interviews.jsonl",
    "book": PROCESSED_DIR / "Hochu_i_budu_with_questions.jsonl",
    "articles": PROCESSED_DIR / "articles_with_questions.jsonl",
    "anti_generic": FINE_TUNING_DIR / "anti_generic_clean.jsonl",
}

# RS Classifier
RS_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "rs_classifier" / "rs_classifier.pkl"

# Model
MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
MAX_SEQ_LENGTH = 1024

# LoRA settings (from adapter_config.json of checkpoint-1130)
LORA_R = 16
LORA_ALPHA = 32
# NOTE: Unsloth recommends lora_dropout=0 (optimized kernels).
# Your best run used 0.02. Try 0 first for speed; if results are worse, switch to 0.02.
LORA_DROPOUT = 0.02

# Training settings (from trainer_state.json of checkpoint-1130)
BATCH_SIZE = 4          # Unsloth is more memory efficient — can use larger batch
GRADIENT_ACCUMULATION = 4  # Effective batch = 8 (same as original: 1 * 8)
LEARNING_RATE = 2e-5  # Same as 0.74 checkpoint
NUM_EPOCHS = 15
WARMUP_STEPS = 20
VAL_RATIO = 0.15
RANDOM_SEED = 42


# ==============================================================
# RS CLASSIFIER
# ==============================================================

class RSClassifier:
    """Predict Response Signal for questions."""

    def __init__(self, model_path: Path):
        print(f"Loading RS classifier from {model_path}...")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.classifier = model_data["classifier"]
        self.id2rs = model_data["id2rs"]
        self.embedding_model_name = model_data["embedding_model_name"]

        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embed_model = SentenceTransformer(self.embedding_model_name)
        print("RS classifier ready")

    def predict(self, question: str) -> str:
        """Predict RS for a question."""
        embedding = self.embed_model.encode([f"query: {question}"])
        pred_id = self.classifier.predict(embedding)[0]
        return self.id2rs[pred_id]

    def predict_batch(self, questions: list) -> list:
        """Predict RS for multiple questions."""
        prefixed = [f"query: {q}" for q in questions]
        embeddings = self.embed_model.encode(prefixed)
        pred_ids = self.classifier.predict(embeddings)
        return [self.id2rs[pid] for pid in pred_ids]


# ==============================================================
# DATA LOADING
# ==============================================================

def load_all_data(input_files: dict, rs_classifier: RSClassifier) -> list:
    """
    Load training data and predict RS for each question.
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
                    question = questions_list[0]
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

    # Predict RS for all questions
    print("Predicting RS for all questions...")
    questions = [r['question'] for r in records]
    rs_predictions = rs_classifier.predict_batch(questions)

    for record, rs in zip(records, rs_predictions):
        record['rs'] = rs

    # Count RS distribution
    rs_counts = {}
    for r in records:
        rs_counts[r['rs']] = rs_counts.get(r['rs'], 0) + 1
    print(f"RS distribution: {rs_counts}")

    return records


def format_for_sft(examples, tokenizer):
    """
    Format as full text with RS prefix.
    Format: <RS={rs}>
    Вопрос: {question}
    Ответ: {answer}
    """
    texts = []
    eos = tokenizer.eos_token

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]
        rs = examples["rs"][i]

        text = f"<RS={rs}>\nВопрос: {question}\nОтвет: {answer}{eos}"
        texts.append(text)

    return {"text": texts}


def formatting_func_for_unsloth(example):
    """Unsloth requires this function to extract text from examples."""
    return example["text"]


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("Labkovsky Fine-tuning (Vikhr + Unsloth + LoRA r=32)")
    print("With RS classifier + Unsloth acceleration")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"BF16: {'supported' if is_bfloat16_supported() else 'not supported'}")

    # ----------------------------------------------------------
    # Load RS classifier and prepare data FIRST (before LLM)
    # ----------------------------------------------------------
    rs_classifier = RSClassifier(RS_CLASSIFIER_PATH)

    print(f"\nLoading data from 4 sources...")
    records = load_all_data(INPUT_FILES, rs_classifier)

    if not records:
        print("No data found!")
        return

    # Free RS classifier memory before loading LLM
    del rs_classifier
    torch.cuda.empty_cache()

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    dataset = Dataset.from_list(records)

    print(f"\nSplitting: {int((1-VAL_RATIO)*100)}% train / {int(VAL_RATIO*100)}% val")
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ----------------------------------------------------------
    # Load model with Unsloth (replaces manual quantization setup)
    # ----------------------------------------------------------
    print(f"\nLoading model with Unsloth: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,       # 4-bit quantization (same as BitsAndBytesConfig nf4)
        dtype=None,              # Auto-detect (bf16 if supported, else fp16)
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----------------------------------------------------------
    # LoRA with Unsloth (replaces manual PEFT setup)
    # ----------------------------------------------------------
    print(f"Preparing LoRA with Unsloth (r={LORA_R}, alpha={LORA_ALPHA})...")
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
        # "unsloth" gradient checkpointing uses 30% less VRAM than standard
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    model.print_trainable_parameters()

    # ----------------------------------------------------------
    # Format data with RS prefix
    # ----------------------------------------------------------
    print(f"\nFormatting data with RS prefix...")
    format_fn = partial(format_for_sft, tokenizer=tokenizer)
    train_dataset = train_dataset.map(format_fn, batched=True)
    eval_dataset = eval_dataset.map(format_fn, batched=True)

    print(f"\nSample (first 400 chars):\n{train_dataset[0]['text'][:400]}\n")

    # ----------------------------------------------------------
    # Trainer (same SFTTrainer, Unsloth optimizes it under the hood)
    # ----------------------------------------------------------
    # TRL auto-detects prompt-completion format and masks prompt tokens
    sft_config = SFTConfig(
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
        save_total_limit=2,  # Keep only 2 checkpoints to reduce disk wear
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func_for_unsloth,
        args=sft_config,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    # Apply completion-only loss (train only on answer, mask prompt)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="Вопрос:",
        response_part="Ответ:",
    )
    print("Applied train_on_responses_only with response_part='Ответ:'")

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    print(f"\nStarting training (Unsloth accelerated)...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Completion-only loss: YES (response_template='Ответ:')")
    print(f"  Gradient checkpointing: Unsloth optimized")

    trainer.train()

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    print(f"\nSaving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()