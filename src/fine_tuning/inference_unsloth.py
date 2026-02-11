#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for Unsloth-trained LoRA model.

Usage:
    python src/fine_tuning/inference_unsloth.py
    python src/fine_tuning/inference_unsloth.py --checkpoint checkpoint-297
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# === CONFIG ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
DEFAULT_LORA_DIR = MODELS_DIR / "labkovsky-vikhr-lora-unsloth"

# Same system prompt as training
SYSTEM_PROMPT = (
    "Ты — психолог Михаил Лабковский. "
    "Отвечай в его стиле: прямо, уверенно, с конкретными рекомендациями. "
    "Используй простой язык и жизненные примеры."
)

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3


def load_model(lora_path: Path):
    print(f"Loading model from {lora_path}...")

    bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    print("Model ready")
    return model, tokenizer


def ask(model, tokenizer, question: str) -> str:
    """Generate answer using chat template with system prompt."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint folder name (e.g., checkpoint-297) or full path"
    )
    args = parser.parse_args()

    # Determine LoRA path
    if args.checkpoint:
        if Path(args.checkpoint).is_absolute():
            lora_path = Path(args.checkpoint)
        else:
            lora_path = DEFAULT_LORA_DIR / args.checkpoint
    else:
        lora_path = DEFAULT_LORA_DIR

    if not lora_path.exists():
        print(f"ERROR: {lora_path} not found")
        print(f"\nAvailable checkpoints in {DEFAULT_LORA_DIR}:")
        for p in sorted(DEFAULT_LORA_DIR.glob("checkpoint-*")):
            print(f"  {p.name}")
        return

    model, tokenizer = load_model(lora_path)

    print(f"\nSystem prompt: {SYSTEM_PROMPT[:50]}...")
    print("Commands: 'exit' to quit\n")

    while True:
        q = input("\nТы: ").strip()
        if q.lower() in ['exit', 'quit', 'q']:
            break
        if not q:
            continue

        answer = ask(model, tokenizer, q)
        print(f"\nЛабковский: {answer}")


if __name__ == "__main__":
    main()
