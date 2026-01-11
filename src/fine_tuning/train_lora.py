#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Qwen2.5-7B on Labkovsky Q&A data using transformers + peft.

Features:
- Combined data: QA + multi-turn dialogues (same messages format)
- 85/15 train/validation split
- Monitors validation loss to detect overfitting

Requirements:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers peft trl accelerate bitsandbytes datasets

Usage:
    python train_lora.py
"""
import random
import json
import torch
from functools import partial
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ==============================================================
# SETTINGS
# ==============================================================

# Data paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
QA_PATH = DATA_DIR / "train_data_clean.jsonl"
DIALOGUES_PATH = DATA_DIR / "Hochu_i_budu_dialogues.jsonl"

# Train/validation split
VAL_RATIO = 0.15 # 15% for validation
RANDOM_SEED = 42

# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# LoRA settings
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Training settings
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1           # Small for 12GB VRAM
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 10

# Output
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "models" / "labkovsky-qwen7b-lora"

# ==============================================================
# LOAD DATA
# ==============================================================

def load_all_data(qa_path: Path, dialogues_path: Path) -> list:
    """
    Load all data into unified messages format.
    """
    all_records = []

    # Load QA (input/output format → messages)
    qa_count = 0
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                all_records.append({
                    "messages": [
                        {"role": "user", "content": r["input"]},
                        {"role": "assistant", "content": r["output"]}
                    ],
                    "source": "qa"
                })
                qa_count += 1

    # Load dialogues (turns → messages)
    dialogue_count = 0
    with open(dialogues_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                all_records.append({
                    "messages": r["turns"],
                    "source": "dialogue"
                })
                dialogue_count += 1

    print(f"📄 Loaded: {qa_count} QA + {dialogue_count} dialogues = {len(all_records)} total")

    return all_records


# ==============================================================
# FORMAT FOR CHAT
# ==============================================================

def format_prompts(examples, tokenizer):
    """Format examples using Qwen chat template"""
    texts = []

    for i in range(len(examples["messages"])):
        messages = examples["messages"][i]

        # Add Labkovsky system prompt
        messages_with_system = [
            {"role": "system", "content": "Ты — Михаил Лабковский, российский психолог."}
        ] + messages

        text = tokenizer.apply_chat_template(
            messages_with_system,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return {"text": texts}


# ==============================================================
# MAIN TRAINING
# ==============================================================

def main():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    print("=" * 60)
    print("Labkovsky Fine-tuning with transformers + peft")
    print("With train/validation split")
    print("=" * 60)
 
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training will be very slow.")
        return
    
    print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 1. Setup quantization (4-bit to fit in 12GB VRAM)
    print(f"\n🔧 Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 2. Load model
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    print("   (This may take a few minutes on first run)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 3. Prepare for training
    print(f"\n🔧 Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # 4. Add LoRA adapters
    print(f"\n🔧 Adding LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA})")
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
    
    # 5. Load all data
    print(f"\n📚 Loading training data...")
    all_records = load_all_data(QA_PATH, DIALOGUES_PATH)
    
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)

    # Create dataset
    dataset = Dataset.from_list(all_records)
    
    # 6. Split into train/validation
    print(f"\n📊 Splitting data: {int((1-VAL_RATIO)*100)}% train / {int(VAL_RATIO*100)}% validation")
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    # Show split stats
    train_qa = sum(1 for x in train_dataset if x["source"] == "qa")
    train_dialogue = sum(1 for x in train_dataset if x["source"] == "dialogue")
    eval_qa = sum(1 for x in eval_dataset if x["source"] == "qa")
    eval_dialogue = sum(1 for x in eval_dataset if x["source"] == "dialogue")

    print(f"   Train: {len(train_dataset)} total ({train_qa} QA + {train_dialogue} dialogues)")
    print(f"   Eval:  {len(eval_dataset)} total ({eval_qa} QA + {eval_dialogue} dialogues)")
    
    # 7. Format for chat template
    print(f"\n📝 Formatting data...")
    format_fn = partial(format_prompts, tokenizer=tokenizer)
    train_dataset = train_dataset.map(format_fn, batched=True)
    eval_dataset = eval_dataset.map(format_fn, batched=True)
    
    print(f"\n📝 Sample formatted text:")
    print(f"{train_dataset[0]['text'][:300]}...")
    
    # 8. Setup trainer
    print(f"\n🏋️ Setting up trainer")
    
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=False,
        logging_steps=20,
        eval_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,  # Load best model based on eval loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower eval_loss is better
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # 9. Train!
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Effective batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Train examples: {len(train_dataset)}")
    print(f"   Eval examples: {len(eval_dataset)}")
    
    trainer.train()
    
    # 10. Show final metrics
    print(f"\n📊 Final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"   Final eval_loss: {final_metrics['eval_loss']:.4f}")
    
    # 11. Save best model
    print(f"\n💾 Saving LoRA adapter to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print(f"   Best model selected based on lowest eval_loss")


if __name__ == "__main__":
    main()