#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Qwen2.5-7B on Labkovsky Q&A data with RS conditioning.

Architecture:
- No system prompt
- RS prefix as behavioral signal
- Pure Q&A data (no RAG context in training)

Format:
<RS=INTERVENTION>
Вопрос: ...
Ответ: ...

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
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FINE_TUNING_DATA_DIR,
    MODEL_NAME,
    LORA_PATH,
    RS_TOKENS,
)

# ==============================================================
# SETTINGS
# ==============================================================

# Paths (using config)
INPUT_DATA = FINE_TUNING_DATA_DIR / "qa_rs_corpus_clean.jsonl"
FORMATTED_DATA = FINE_TUNING_DATA_DIR / "train_rs_formatted.jsonl"
OUTPUT_DIR = LORA_PATH

# LoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05  # Lower dropout to preserve rare patterns (stories, idioms)

# Training settings
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-5  # Higher LR for small dataset with LoRA
NUM_EPOCHS = 10
WARMUP_STEPS = 10
VAL_RATIO = 0.15
RANDOM_SEED = 42

# ==============================================================
# STEP 1: CREATE FORMATTED TRAINING DATA
# ==============================================================

def create_formatted_data(input_path: Path, output_path: Path) -> int:
    """
    Convert qa_rs_corpus_clean.jsonl to prompt/completion format.
    
    Input format:
    {"question": "...", "answer": "...", "response_signal": "INTERVENTION"}
    
    Output format:
    {"prompt": "<RS=INTERVENTION>\nВопрос: ...\nОтвет:", "completion": "..."}
    """
    print(f"\n📄 Creating formatted training data...")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    
    records = []
    rs_counts = {}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                question = data["question"]
                answer = data["answer"]
                rs = data["response_signal"]
                
                # Count RS distribution
                rs_counts[rs] = rs_counts.get(rs, 0) + 1
                
                # Format prompt
                prompt = f"<RS={rs}>\nВопрос: {question}\nОтвет:"
                completion = answer
                
                records.append({
                    "prompt": prompt,
                    "completion": completion
                })
    
    # Save formatted data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Created {len(records)} formatted examples")
    print(f"   RS distribution:")
    for rs, count in sorted(rs_counts.items()):
        print(f"      {rs}: {count}")
    
    return len(records)


# ==============================================================
# STEP 2: LOAD DATA FOR TRAINING
# ==============================================================

def load_training_data(data_path: Path) -> list:
    """Load formatted prompt/completion data."""
    records = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    return records


def format_for_sft(examples, tokenizer):
    """
    Combine prompt + completion into single text for SFT.
    
    Loss is computed only on tokens AFTER "Ответ:" delimiter
    (handled by DataCollatorForCompletionOnlyLM).
    """
    texts = []
    eos = tokenizer.eos_token
    
    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i]
        completion = examples["completion"][i]
        
        # Full text: prompt + space + completion + EOS
        # Collator will mask everything up to and including "Ответ:"
        text = f"{prompt} {completion}{eos}"
        texts.append(text)
    
    return {"text": texts}


# ==============================================================
# MAIN TRAINING
# ==============================================================

def main():
    torch.cuda.empty_cache()
    gc.collect()
    
    print("=" * 60)
    print("Labkovsky Fine-tuning with RS Conditioning")
    print("No system prompt, pure behavioral learning")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check bf16 support (Ampere+ GPUs)
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"   BF16: {'supported' if bf16_supported else 'NOT supported (will use fp32)'}")
    
    # Set compute dtype based on GPU capability
    if bf16_supported:
        compute_dtype = torch.bfloat16
        use_bf16 = True
    else:
        compute_dtype = torch.float16
        use_bf16 = False
    
    # Step 1: Create formatted data
    create_formatted_data(INPUT_DATA, FORMATTED_DATA)
    
    # Step 2: Load and prepare data
    print(f"\n📚 Loading training data...")
    records = load_training_data(FORMATTED_DATA)
    
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    
    dataset = Dataset.from_list(records)
    
    # Split train/validation
    print(f"\n📊 Splitting data: {int((1-VAL_RATIO)*100)}% train / {int(VAL_RATIO*100)}% validation")
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    print(f"   Train: {len(train_dataset)}")
    print(f"   Eval:  {len(eval_dataset)}")
    
    # Step 3: Setup quantization
    print(f"\n🔧 Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    # Step 4: Load model
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    
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
    
    # Add RS tags as special tokens (atomic, not split into subwords)
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": RS_TOKENS
    })
    print(f"   Added {num_added} special RS tokens: {RS_TOKENS}")
    
    # Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"   Resized embeddings to {len(tokenizer)} tokens")
    
    # Step 5: Prepare for k-bit training
    print(f"\n🔧 Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Step 6: Add LoRA adapters
    # Note: embed_tokens in modules_to_save (not LoRA) so new RS tokens get real embeddings
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
        modules_to_save=["embed_tokens"],  # Save full embeddings (for new RS tokens)
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    
    # Step 7: Format data for SFT
    print(f"\n📝 Formatting data for training...")
    print(f"   EOS token: {repr(tokenizer.eos_token)}")
    
    format_fn = partial(format_for_sft, tokenizer=tokenizer)
    train_dataset = train_dataset.map(format_fn, batched=True)
    eval_dataset = eval_dataset.map(format_fn, batched=True)
    
    print(f"\n📝 Sample formatted text:")
    print("-" * 40)
    print(train_dataset[0]['text'][:500])
    print("-" * 40)
    
    # Step 8: Setup trainer with completion-only loss
    print(f"\n🏋️ Setting up trainer...")
    print(f"   Using completion-only loss (masking prompt tokens)")
    
    # Response template - loss computed only AFTER this delimiter
    response_template = "Ответ:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not use_bf16 and torch.cuda.is_available(),  # Use fp16 if bf16 not supported
        bf16=use_bf16,  # Match bnb_4bit_compute_dtype
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
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Step 9: Train
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Effective batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   LoRA r: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"   Train examples: {len(train_dataset)}")
    print(f"   Eval examples: {len(eval_dataset)}")
    
    trainer.train()
    
    # Step 10: Final evaluation
    print(f"\n📊 Final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"   Final eval_loss: {final_metrics['eval_loss']:.4f}")
    
    # Step 11: Save
    print(f"\n💾 Saving LoRA adapter to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print(f"   Format: <RS=...>\\nВопрос: ...\\nОтвет: ...")
    print("=" * 60)


if __name__ == "__main__":
    main()