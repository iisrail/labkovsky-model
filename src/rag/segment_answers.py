#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment Labkovsky answers into RS categories using -it model.

Categories:
- EXPLANATION: Psychology behind the issue
- INTERVENTION: Practical advice
- ESCALATION: When to seek professional help

Usage:
    python segment_answers.py
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_DIR / "models"

LLM_MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
LORA_PATH = MODELS_DIR / "labkovsky-vikhr-yandex-lora"

# ============================================================
# MODEL
# ============================================================

_llm = None
_tokenizer = None


def init_llm(use_lora: bool = True):
    global _llm, _tokenizer
    if _llm is None:
        print(f"[+] Loading LLM: {LLM_MODEL_NAME}")

        if use_lora and LORA_PATH.exists():
            _tokenizer = AutoTokenizer.from_pretrained(str(LORA_PATH), trust_remote_code=True)
        else:
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if use_lora and LORA_PATH.exists():
            base_model.resize_token_embeddings(len(_tokenizer))
            _llm = PeftModel.from_pretrained(base_model, str(LORA_PATH))
        else:
            _llm = base_model

        _llm.eval()
        print("[OK] LLM loaded")

    return _llm, _tokenizer


# ============================================================
# SEGMENTATION
# ============================================================

SEGMENT_PROMPT = """Раздели ответ психолога на части, поставив метку перед каждой частью текста.

Метки:
[EXPLANATION] — объяснение психологии
[INTERVENTION] — практический совет
[ESCALATION] — рекомендация к специалисту

Пример:
Ответ: У вас низкая самооценка. Попробуйте хвалить себя каждый день. Если не поможет, идите к психологу.
Результат:
[EXPLANATION] У вас низкая самооценка.
[INTERVENTION] Попробуйте хвалить себя каждый день.
[ESCALATION] Если не поможет, идите к психологу.

Теперь раздели этот ответ:
{answer}

Результат:
"""


def segment_answer(answer: str) -> str:
    """Segment an answer into RS categories."""
    llm, tokenizer = init_llm()

    prompt = SEGMENT_PROMPT.format(answer=answer)

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = llm.generate(
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
# CLI
# ============================================================

def main():
    print("=" * 60)
    print("Labkovsky Answer Segmentation")
    print("=" * 60)
    print("\nPaste an answer, then press Enter twice to segment.")
    print("Type 'exit' to quit.\n")

    init_llm()

    while True:
        print("Answer to segment:")
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break
                if line.lower() == "exit":
                    print("Poka!")
                    return
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            print("\nPoka!")
            break

        if not lines:
            continue

        answer = "\n".join(lines)
        print("\n[Segmenting...]\n")

        result = segment_answer(answer)
        print("=" * 40)
        print(result)
        print("=" * 40)
        print()


if __name__ == "__main__":
    main()
