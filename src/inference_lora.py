#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the fine-tuned Labkovsky LoRA model.

Usage:
    python inference_lora.py
"""

from pathlib import Path

# ==============================================================
# SETTINGS
# ==============================================================

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
LORA_PATH = Path("models/labkovsky-qwen7b-lora")

SYSTEM_PROMPT = "Ты — Михаил Лабковский, российский психолог. Отвечай в своём стиле: прямо, честно, без воды."

MAX_NEW_TOKENS = 512

# ==============================================================
# LOAD MODEL
# ==============================================================

def load_model():
    """Load base model with LoRA adapter"""
    from unsloth import FastLanguageModel
    
    print(f"🤖 Loading model: {MODEL_NAME}")
    print(f"🔧 Loading LoRA: {LORA_PATH}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(LORA_PATH),  # Load from LoRA path
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


# ==============================================================
# GENERATE
# ==============================================================

def generate_answer(model, tokenizer, question: str) -> str:
    """Generate Labkovsky-style answer"""
    
    # Format as chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    
    return response.strip()


# ==============================================================
# INTERACTIVE
# ==============================================================

def main():
    print("=" * 60)
    print("🎤 Лабковский (Fine-tuned)")
    print("=" * 60)
    
    model, tokenizer = load_model()
    print("\n✅ Model loaded! Type 'exit' to quit.\n")
    
    while True:
        try:
            question = input("Ты: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Пока!")
            break
        
        if not question:
            continue
        
        if question.lower() in ['exit', 'quit', 'выход']:
            print("👋 Пока!")
            break
        
        print("\n🤔 Думаю...\n")
        answer = generate_answer(model, tokenizer, question)
        print(f"Лабковский: {answer}\n")


if __name__ == "__main__":
    main()
