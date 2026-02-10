#!/usr/bin/env python3
"""
3-Stage LoRA Fine-tuning for Labkovsky Model (Unsloth)

Stage 1: Style soak (8 epochs, LR=2e-4)  - vocabulary, rhythm, worldview
Stage 2: Schema lock (2 epochs, LR=5e-5) - force [ОБЪЯСНЕНИЕ]/[ВМЕШАТЕЛЬСТВО]/[ЭСКАЛАЦИЯ]
Stage 3: Anti-generic (0.5 epoch, LR=2e-5) - suppress generic advice

Single LoRA adapter trained across all stages with decreasing LR.
Merge to 16-bit only once at the end.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FINE_TUNING_DATA_DIR, MODEL_NAME, MODELS_DIR

# === PATHS ===
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Adapter checkpoints (small, ~50MB each)
STAGE1_ADAPTER = MODELS_DIR / "labkovsky-stage1-adapter"
STAGE2_ADAPTER = MODELS_DIR / "labkovsky-stage2-adapter"
STAGE3_ADAPTER = MODELS_DIR / "labkovsky-stage3-adapter"

# Final merged model (one 16-bit merge at the end)
FINAL_MERGED = MODELS_DIR / "labkovsky-final-merged"

# === SETTINGS ===
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
WARMUP_STEPS = 20
VAL_RATIO = 0.10
RANDOM_SEED = 42

# Stage-specific settings
STAGES = {
    "Stage 1: Style Soak": {
        "epochs": 8,
        "lr": 2e-4,
    },
    "Stage 2: Schema Lock": {
        "epochs": 2,
        "lr": 5e-5,
    },
    "Stage 3: Anti-Generic": {
        "epochs": 0.5,
        "lr": 2e-5,
    },
}

# === INSTRUCTION PREFIX (Stage 2 & 3) ===
INSTRUCTION_PREFIX = """Инструкция: Ответь строго в формате: [ОБЪЯСНЕНИЕ]...[ВМЕШАТЕЛЬСТВО]...[ЭСКАЛАЦИЯ]...

"""


# ============================================================
# DATA LOADING
# ============================================================

def load_stage1_data() -> list:
    """Load and merge all Stage 1 data sources."""
    records = []

    # 1. qa_rs_corpus_short.jsonl
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
        print(f"  qa_rs_corpus_short: {len(records)} samples")

    # 2. articles_with_questions.jsonl
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
        print(f"  articles_with_questions: {len(records) - count_before} samples")

    # 3. interviews.jsonl
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
        print(f"  interviews: {len(records) - count_before} samples")

    # 4. Hochu_i_budu_with_questions.jsonl
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
        print(f"  Hochu_i_budu: {len(records) - count_before} samples")

    print(f"Stage 1 total: {len(records)} samples")
    return records


def load_stage2_data() -> list:
    """Load Stage 2 data (RS schema)."""
    records = []
    qa_file = FINE_TUNING_DATA_DIR / "qa_rs_final.jsonl"

    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            answer = data.get('answer_segmented', data.get('answer', ''))
            if answer:
                records.append({
                    'question': data['question'],
                    'answer': answer
                })

    print(f"Stage 2 total: {len(records)} samples")
    return records


def load_stage3_data() -> list:
    """Load Stage 3 data (anti-generic)."""
    records = []
    qa_file = FINE_TUNING_DATA_DIR / "anti_generic_stage3.jsonl"

    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            answer = data.get('answer_segmented', '')
            if answer:
                records.append({
                    'question': data['question'],
                    'answer': answer
                })

    print(f"Stage 3 total: {len(records)} samples")
    return records


# ============================================================
# FORMATTING
# ============================================================

def format_stage1(examples):
    """Format for Stage 1: simple Q&A without instruction."""
    texts = []
    for i in range(len(examples["question"])):
        text = f"Вопрос: {examples['question'][i]}\nОтвет: {examples['answer'][i]}"
        texts.append(text)
    return {"text": texts}


def format_stage2_3(examples):
    """Format for Stage 2 & 3: with instruction prefix."""
    texts = []
    for i in range(len(examples["question"])):
        text = f"{INSTRUCTION_PREFIX}Вопрос: {examples['question'][i]}\nОтвет: {examples['answer'][i]}"
        texts.append(text)
    return {"text": texts}


# ============================================================
# MODEL
# ============================================================

def load_base_model():
    """Load base model and apply LoRA once."""
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("Applying LoRA adapter (once for all stages)")
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

    return model, tokenizer


# ============================================================
# TRAINING
# ============================================================

def train_stage(
    stage_name: str,
    model,
    tokenizer,
    records: list,
    format_fn,
    num_epochs: float,
    learning_rate: float,
    checkpoint_dir: Path,
):
    """Run training for a single stage. Model is modified in-place."""
    print(f"\n{'='*60}")
    print(f"{stage_name}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}")
    print(f"{'='*60}")

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    train_dataset = train_dataset.map(format_fn, batched=True)
    eval_dataset = eval_dataset.map(format_fn, batched=True)

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir / "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
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

    # Save adapter checkpoint (small, ~50MB)
    print(f"  Saving adapter checkpoint: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 60)
    print("Labkovsky 3-Stage Fine-tuning (Unsloth)")
    print("=" * 60)
    print("Single LoRA, decreasing LR, merge once at end")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Load data for all stages upfront
    print("\nLoading data...")
    stage1_records = load_stage1_data()
    stage2_records = load_stage2_data()
    stage3_records = load_stage3_data()

    # Load model + LoRA once
    model, tokenizer = load_base_model()

    # Stage 1: Style Soak
    train_stage(
        stage_name="Stage 1: Style Soak",
        model=model,
        tokenizer=tokenizer,
        records=stage1_records,
        format_fn=format_stage1,
        num_epochs=STAGES["Stage 1: Style Soak"]["epochs"],
        learning_rate=STAGES["Stage 1: Style Soak"]["lr"],
        checkpoint_dir=STAGE1_ADAPTER,
    )

    # Stage 2: Schema Lock
    train_stage(
        stage_name="Stage 2: Schema Lock",
        model=model,
        tokenizer=tokenizer,
        records=stage2_records,
        format_fn=format_stage2_3,
        num_epochs=STAGES["Stage 2: Schema Lock"]["epochs"],
        learning_rate=STAGES["Stage 2: Schema Lock"]["lr"],
        checkpoint_dir=STAGE2_ADAPTER,
    )

    # Stage 3: Anti-Generic
    train_stage(
        stage_name="Stage 3: Anti-Generic",
        model=model,
        tokenizer=tokenizer,
        records=stage3_records,
        format_fn=format_stage2_3,
        num_epochs=STAGES["Stage 3: Anti-Generic"]["epochs"],
        learning_rate=STAGES["Stage 3: Anti-Generic"]["lr"],
        checkpoint_dir=STAGE3_ADAPTER,
    )

    # Final merge: one 16-bit merge
    print(f"\n{'='*60}")
    print("Saving final merged 16-bit model")
    print(f"{'='*60}")
    FINAL_MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(FINAL_MERGED),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Saved to: {FINAL_MERGED}")

    print(f"\n{'='*60}")
    print("3-Stage Training Complete!")
    print(f"{'='*60}")
    print(f"\nAdapter checkpoints:")
    print(f"  Stage 1 (style):       {STAGE1_ADAPTER}")
    print(f"  Stage 2 (schema):      {STAGE2_ADAPTER}")
    print(f"  Stage 3 (anti-generic):{STAGE3_ADAPTER}")
    print(f"\nFinal merged model:      {FINAL_MERGED}")


if __name__ == "__main__":
    main()