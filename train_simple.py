#!/usr/bin/env python3
"""
Simple Single-Stage LoRA Fine-tuning for Labkovsky Model

No RS tags, no multi-stage, no instruction prefix.
Just teach the model to sound like Labkovsky.

Data: QA + articles + book + interviews (~1050 samples)
Format: "Вопрос: X\nОтвет: Y"
"""

import json
import random
import sys
import torch
import gc
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import FINE_TUNING_DATA_DIR, MODEL_NAME, MODELS_DIR

# === PATHS ===
DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

ADAPTER_DIR = MODELS_DIR / "labkovsky-simple-adapter"
FINAL_MERGED = MODELS_DIR / "labkovsky-simple-merged"

# === SETTINGS ===
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 768
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
WARMUP_STEPS = 20
LEARNING_RATE = 2e-4
NUM_EPOCHS = 8
VAL_RATIO = 0.10
RANDOM_SEED = 42


def load_all_data() -> list:
    """Load all data sources into one dataset."""
    records = []

    # 1. QA pairs (use original answer, no RS tags)
    qa_file = FINE_TUNING_DATA_DIR / "qa_rs_corpus_short.jsonl"
    if qa_file.exists():
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                records.append({
                    'question': data['question'],
                    'answer': data['answer']
                })
        print(f"  QA pairs: {len(records)}")

    # 2. Articles (1 question per item)
    count_before = len(records)
    articles_file = PROCESSED_DIR / "articles_with_questions.jsonl"
    if articles_file.exists():
        with open(articles_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get('text', '')
                questions = data.get('potential_questions', [])
                if questions and text:
                    records.append({'question': questions[0], 'answer': text})
        print(f"  Articles: {len(records) - count_before}")

    # 3. Interviews (1 question per item)
    count_before = len(records)
    interviews_file = PROCESSED_DIR / "interviews.jsonl"
    if interviews_file.exists():
        with open(interviews_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get('text', '')
                questions = data.get('potential_questions', [])
                if questions and text:
                    records.append({'question': questions[0], 'answer': text})
        print(f"  Interviews: {len(records) - count_before}")

    # 4. Book (1 question per item)
    count_before = len(records)
    book_file = PROCESSED_DIR / "Hochu_i_budu_with_questions.jsonl"
    if book_file.exists():
        with open(book_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get('text', '')
                questions = data.get('potential_questions', [])
                if questions and text:
                    records.append({'question': questions[0], 'answer': text})
        print(f"  Book: {len(records) - count_before}")

    # 5. Anti-generic examples (cleaned, no RS tags)
    count_before = len(records)
    anti_generic_file = FINE_TUNING_DATA_DIR / "anti_generic_clean.jsonl"
    if anti_generic_file.exists():
        with open(anti_generic_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                answer = data.get('answer', '')
                if answer and data.get('question'):
                    records.append({'question': data['question'], 'answer': answer})
        print(f"  Anti-generic: {len(records) - count_before}")

    print(f"\nTotal: {len(records)} samples")
    return records


def format_prompts(examples):
    """Simple Q&A format, no tags, no instruction."""
    texts = []
    for i in range(len(examples["question"])):
        text = f"Вопрос: {examples['question'][i]}\nОтвет: {examples['answer'][i]}"
        texts.append(text)
    return {"text": texts}


def main():
    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 60)
    print("Labkovsky Simple Training")
    print("No RS, no multi-stage, just style")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading data...")
    records = load_all_data()

    # Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    # Split
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Format
    train_dataset = train_dataset.map(format_prompts, batched=True)
    eval_dataset = eval_dataset.map(format_prompts, batched=True)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA
    print(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
    )

    # Train
    print(f"\nTraining: {NUM_EPOCHS} epochs, LR={LEARNING_RATE}")

    sft_config = SFTConfig(
        output_dir=str(ADAPTER_DIR / "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    trainer.train()

    # Save adapter
    print(f"\nSaving adapter: {ADAPTER_DIR}")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    # Merge
    print(f"\nSaving merged 16-bit: {FINAL_MERGED}")
    FINAL_MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(FINAL_MERGED),
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Adapter:  {ADAPTER_DIR}")
    print(f"Merged:   {FINAL_MERGED}")


if __name__ == "__main__":
    main()