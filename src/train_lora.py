#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Qwen2.5-7B on Labkovsky Q&A data using transformers + peft.
No unsloth required - more stable on Windows.

Requirements:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers peft trl accelerate bitsandbytes datasets

Usage:
    python train_lora.py
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig

)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ==============================================================
# SETTINGS
# ==============================================================

# Data
SCRIPT_DIR = Path(__file__).parent
TRAIN_DATA_PATH = SCRIPT_DIR.parent / "data" / "fine_tuning" / "train_data.jsonl"

# Model - use standard HuggingFace model (not unsloth)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# LoRA settings
LORA_R = 16              # Rank
LORA_ALPHA = 32          # Scaling factor (typically 2x rank)
LORA_DROPOUT = 0.05

# Training settings
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1           # Small for 12GB VRAM
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 5

# Output
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "labkovsky-qwen7b-lora"

# ==============================================================
# LOAD DATA
# ==============================================================

def load_data(path: Path) -> Dataset:
    """Load training data from JSONL"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"📄 Loaded {len(records)} training examples")
    return Dataset.from_list(records)


# ==============================================================
# FORMAT FOR CHAT
# ==============================================================

def format_prompts(examples):
    """Format examples for Qwen chat template"""
    
    texts = []
    for input_text, output in zip(examples["input"], examples["output"]):
        # Qwen chat format
        text = f"""<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
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
        bnb_4bit_compute_dtype=torch.float16,
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
    model.print_trainable_parameters()
    
    # 5. Load and format data
    print(f"\n📚 Loading training data: {TRAIN_DATA_PATH}")
    dataset = load_data(TRAIN_DATA_PATH)
    dataset = dataset.map(format_prompts, batched=True)
    
    print(f"   Sample text:\n{dataset[0]['text'][:200]}...")
    
    # 6. Setup trainer
    print(f"\n🏋️ Setting up trainer")
    
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        bf16=False,
        logging_steps=1,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        seed=42,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    
    # 7. Train!
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Effective batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    trainer.train()
    
    # 8. Save
    print(f"\n💾 Saving LoRA adapter to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()