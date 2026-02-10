#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test if Vikhr-it BASE model can segment Labkovsky answers into RS categories.

RS Categories:
- EXPLANATION: Psychology explanation (why this happens)
- INTERVENTION: Practical advice (what to do)
- ESCALATION: Professional help recommendation (when to see a specialist)

Usage:
    python test_rs_segmentation.py
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_DIR / "models"

LLM_MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"

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
SEGMENT_PROMPT = """Расставь метки в тексте. Сохрани ВЕСЬ текст, только добавь метки.

Метки:
[EXPLANATION] — объяснение психологии, причины, анализ проблемы (почему так происходит)
[INTERVENTION] — конкретный совет, что ДЕЛАТЬ (действие, рекомендация)
[ESCALATION] — только при угрозе жизни (суицид, острый психоз)

Пример:
Текст: У вас низкая самооценка из-за детских травм. Родители критиковали, и вы усвоили это отношение к себе. Попробуйте каждый день записывать три вещи, за которые себя хвалите.
Результат:
[EXPLANATION] У вас низкая самооценка из-за детских травм. Родители критиковали, и вы усвоили это отношение к себе.
[INTERVENTION] Попробуйте каждый день записывать три вещи, за которые себя хвалите.

Теперь размети этот текст:
{answer}

Результат:"""

# ============================================================
# MODEL
# ============================================================

def load_model():
    print(f"[+] Loading BASE model: {LLM_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)

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

    model.eval()
    print("[OK] Base model loaded (no LoRA)\n")

    return model, tokenizer


import re

def consolidate_labels(text: str) -> str:
    """Merge consecutive sentences with the same label into one block."""
    # Pattern to find [LABEL] followed by text until next label or end
    pattern = r'\[(EXPLANATION|INTERVENTION|ESCALATION)\]\s*'

    # Split by labels, keeping the labels
    parts = re.split(pattern, text)

    if len(parts) < 2:
        return text  # No labels found

    # parts[0] is text before first label (usually empty)
    # parts[1] is first label, parts[2] is text after it, etc.

    segments = []
    current_label = None
    current_text = []

    i = 1  # Start from first label
    while i < len(parts):
        label = parts[i]
        text_part = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if label == current_label:
            # Same label - append text
            current_text.append(text_part)
        else:
            # New label - save previous and start new
            if current_label and current_text:
                segments.append(f"[{current_label}] {' '.join(current_text)}")
            current_label = label
            current_text = [text_part] if text_part else []

        i += 2

    # Don't forget last segment
    if current_label and current_text:
        segments.append(f"[{current_label}] {' '.join(current_text)}")

    return "\n\n".join(segments)


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

    # Post-process: merge consecutive same labels
    consolidated = consolidate_labels(response)

    return consolidated


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Test RS Segmentation - BASE MODEL (no LoRA)")
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
