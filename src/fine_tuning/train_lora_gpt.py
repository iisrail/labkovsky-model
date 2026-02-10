#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repro-focused Unsloth LoRA SFT for Vikhr-YandexGPT-5-Lite-8B-it
Targets matching the ~0.74 run characteristics:
- LoRA: r=32, alpha=64, dropout=0 (fast Unsloth path)
- Completion-only loss (mask prompt tokens)
- Mask boundary taken from the *actual chat_template.jinja*:
    <s>user\n ... </s>\n
    <s>assistant\n ... </s>\n
So we train only on tokens after "<s>assistant\n".
"""

import json
import random
import pickle
import torch
from functools import partial
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import EarlyStoppingCallback

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig


# ==============================================================
# SETTINGS
# ==============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR      = PROJECT_ROOT / "models" / "labkovsky-vikhr-lora-repro074"

INPUT_FILES = {
    "qa_corpus":     FINE_TUNING_DIR / "qa_rs_corpus_short.jsonl",
    "interview":     PROCESSED_DIR   / "interviews.jsonl",
    "book":          PROCESSED_DIR   / "Hochu_i_budu_with_questions.jsonl",
    "articles":      PROCESSED_DIR   / "articles_with_questions.jsonl",
    "anti_generic":  FINE_TUNING_DIR / "anti_generic_clean.jsonl",
}

RS_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "rs_classifier" / "rs_classifier.pkl"

MODEL_NAME     = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
MAX_SEQ_LENGTH = 1024

# === Match the 0.74-era adapter_config ===
LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.0  # IMPORTANT: enables Unsloth fast patching for QKV/O/MLP

# Training
BATCH_SIZE            = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE         = 2e-5
NUM_EPOCHS            = 15
WARMUP_STEPS          = 20
VAL_RATIO             = 0.15
RANDOM_SEED           = 42

# Chat-template boundaries (from your chat_template.jinja)
INSTRUCTION_PART = "<s>user\n"
RESPONSE_PART    = "<s>assistant\n"


# ==============================================================
# RS CLASSIFIER
# ==============================================================

class RSClassifier:
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

    def predict_batch(self, questions: list[str]) -> list[str]:
        prefixed = [f"query: {q}" for q in questions]
        embeddings = self.embed_model.encode(prefixed)
        pred_ids = self.classifier.predict(embeddings)
        return [self.id2rs[pid] for pid in pred_ids]


# ==============================================================
# DATA
# ==============================================================

def load_all_data(input_files: dict, rs_classifier: RSClassifier) -> list[dict]:
    records = []

    for source_name, filepath in input_files.items():
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found — skipping")
            continue

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Format A
                if "question" in data and "answer" in data:
                    question = data["question"]
                    answer = data["answer"]
                # Format B
                elif "text" in data and "potential_questions" in data:
                    qs = data.get("potential_questions") or []
                    if not qs:
                        continue
                    question = qs[0]
                    answer = data["text"]
                else:
                    continue

                if question and answer:
                    records.append(
                        {"question": question, "answer": answer, "source": source_name}
                    )
                    count += 1

        print(f"  {source_name} ({filepath.name}): {count} examples")

    print(f"Total: {len(records)} examples")

    print("Predicting RS for all questions...")
    questions = [r["question"] for r in records]
    rs_predictions = rs_classifier.predict_batch(questions)
    for r, rs in zip(records, rs_predictions):
        r["rs"] = rs

    rs_counts = {}
    for r in records:
        rs_counts[r["rs"]] = rs_counts.get(r["rs"], 0) + 1
    print(f"RS distribution: {rs_counts}")

    return records


def format_for_sft(examples, tokenizer):
    """
    Put RS inside user content so the chat template stays canonical.
    """
    texts = []
    for i in range(len(examples["question"])):
        rs = examples["rs"][i]
        q  = examples["question"][i]
        a  = examples["answer"][i]

        messages = [
            {"role": "user", "content": f"<RS={rs}>\n{q}"},
            {"role": "assistant", "content": a},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("Repro 0.74-style training (Unsloth + completion-only)")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"BF16: {'supported' if is_bfloat16_supported() else 'not supported'}")

    # --- Data first (CPU) ---
    rs_classifier = RSClassifier(RS_CLASSIFIER_PATH)
    print("\nLoading data...")
    records = load_all_data(INPUT_FILES, rs_classifier)

    del rs_classifier
    torch.cuda.empty_cache()

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    print(f"\nSplit: Train={len(train_dataset)}, Eval={len(eval_dataset)}")

    # --- Model ---
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

    # --- LoRA (match capacity + enable fast patching) ---
    print(f"\nPreparing LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,   # MUST be 0 for best Unsloth patching
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

    # --- Format dataset using chat template ---
    print("\nFormatting dataset with chat template...")
    format_fn = partial(format_for_sft, tokenizer=tokenizer)
    train_dataset = train_dataset.map(format_fn, batched=True)
    eval_dataset  = eval_dataset.map(format_fn, batched=True)

    print("\nSample (first 400 chars):")
    print(train_dataset[0]["text"][:400])

    # --- Trainer config ---
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        completion_only_loss=True,   # explicit
        packing=False,               # keep simple/repro-friendly
        group_by_length=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # --- Force masking by true chat-template boundary ---
    # This makes the loss apply only to tokens after "<s>assistant\n".
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    print("\nStarting training...")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Completion-only: YES (response_part={RESPONSE_PART!r})")
    print(f"  Train batch effective: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    trainer.train()

    print(f"\nSaving to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
