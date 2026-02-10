#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test if Vikhr-it model WITH LoRA can segment Labkovsky answers into RS categories.

RS Categories:
- EXPLANATION: Psychology explanation (why this happens)
- INTERVENTION: Practical advice (what to do)
- ESCALATION: Professional help recommendation (when to see a specialist)

Usage:
    python test_rs_segmentation_with_lora.py
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_DIR / "models"

LLM_MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
LORA_PATH = MODELS_DIR / "labkovsky-vikhr-yandex-lora"

# QA corpus path
QA_CORPUS_PATH = PROJECT_DIR / "data" / "fine_tuning" / "qa_rs_corpus_short.jsonl"
NUM_TEST_SAMPLES = 3


def load_test_data():
    """Load first N samples from QA corpus."""
    import json

    samples = []
    with open(QA_CORPUS_PATH, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i >= NUM_TEST_SAMPLES:
                break
            if line.strip():
                try:
                    data = json.loads(line)
                    samples.append({
                        "question": data.get("question", ""),
                        "answer": data.get("answer", "")
                    })
                except json.JSONDecodeError:
                    continue
    return samples

# Prompt for segmentation
SEGMENT_PROMPT = """Раздели этот ответ психолога на три части и поставь метку перед каждой частью.

Метки:
[EXPLANATION] — объяснение психологии, почему так происходит
[INTERVENTION] — практический совет, что делать
[ESCALATION] — рекомендация обратиться к специалисту (если есть в тексте)

Если какой-то части нет в тексте — не добавляй её.

Ответ психолога:
{answer}

Размеченный текст:"""

# ============================================================
# MODEL
# ============================================================

def load_model():
    print(f"[+] Loading model: {LLM_MODEL_NAME}")
    print(f"    With LoRA: {LORA_PATH}")

    # Load tokenizer from LoRA
    tokenizer = AutoTokenizer.from_pretrained(str(LORA_PATH), trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, str(LORA_PATH))

    model.eval()
    print("[OK] Model + LoRA loaded\n")

    return model, tokenizer


def segment_answer(model, tokenizer, answer: str) -> str:
    prompt = SEGMENT_PROMPT.format(answer=answer)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Test RS Segmentation - WITH LoRA")
    print("=" * 60)

    # Load test data from corpus
    print(f"\n📂 Loading {NUM_TEST_SAMPLES} samples from: {QA_CORPUS_PATH.name}")
    samples = load_test_data()
    print(f"   Loaded {len(samples)} samples\n")

    model, tokenizer = load_model()

    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}")
        print("="*60)

        print("\n❓ QUESTION:")
        print("-" * 40)
        print(sample["question"][:200] + "..." if len(sample["question"]) > 200 else sample["question"])

        print("\n📝 ORIGINAL ANSWER:")
        print("-" * 40)
        print(sample["answer"])

        print("\n🔍 SEGMENTED:")
        print("-" * 40)
        result = segment_answer(model, tokenizer, sample["answer"])
        print(result)

        print("\n" + "="*60)
        if i < len(samples):
            input("Press Enter to continue to next test...")

    print("\n✅ Test complete!")
    print("\nEvaluate the results:")
    print("- Are segments correctly identified?")
    print("- Are labels (EXPLANATION/INTERVENTION/ESCALATION) appropriate?")
    print("- Is the structure consistent?")


if __name__ == "__main__":
    main()
