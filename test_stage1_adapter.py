#!/usr/bin/env python3
"""
Test Stage 1 adapter vs Final merged model.
Run from project root: python test_stage1_adapter.py
"""

import torch
from pathlib import Path
from unsloth import FastLanguageModel

# === PATHS (adjust if needed) ===
BASE_MODEL = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
STAGE1_ADAPTER = Path("models/labkovsky-stage1-adapter")
STAGE2_ADAPTER = Path("models/labkovsky-stage2-adapter")
STAGE3_ADAPTER = Path("models/labkovsky-stage3-adapter")
FINAL_MERGED = Path("models/labkovsky-final-merged")

MAX_SEQ_LENGTH = 1024

TEST_QUESTIONS = [
    "Михаил, мне 30 лет, и у меня четыре зависимости: трачу больше чем зарабатываю, алкоголь, зависимость от женатого мужчины и телефон. Как бороться?",
    "Мой парень сказал, что если я его брошу, он спрыгнет с крыши. Что делать?",
    "Как перестать бояться одиночества?",
]


def load_base_with_adapter(adapter_path: Path):
    """Load base model + LoRA adapter."""
    print(f"\nLoading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Loading adapter: {adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))

    return model, tokenizer


def load_merged(merged_path: Path):
    """Load merged model directly."""
    print(f"\nLoading merged model: {merged_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens=300):
    """Generate response."""
    FastLanguageModel.for_inference(model)

    prompt = f"Вопрос: {question}\nОтвет:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    # Decode only new tokens
    response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def test_model(model, tokenizer, label: str):
    """Test a model on all questions."""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {q[:80]}...")
        response = generate(model, tokenizer, q)
        # Show first 500 chars to compare
        print(f"A: {response[:500]}")
        if len(response) > 500:
            print(f"   ... ({len(response)} total chars)")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Test each available model
    models_to_test = []

    if STAGE1_ADAPTER.exists():
        models_to_test.append(("Stage 1 adapter", "adapter", STAGE1_ADAPTER))
    if STAGE2_ADAPTER.exists():
        models_to_test.append(("Stage 2 adapter", "adapter", STAGE2_ADAPTER))
    if STAGE3_ADAPTER.exists():
        models_to_test.append(("Stage 3 adapter", "adapter", STAGE3_ADAPTER))
    if FINAL_MERGED.exists():
        models_to_test.append(("Final merged", "merged", FINAL_MERGED))

    if not models_to_test:
        print("No models found! Check paths.")
        return

    print(f"\nFound {len(models_to_test)} models to test:")
    for name, _, path in models_to_test:
        print(f"  - {name}: {path}")

    for name, model_type, path in models_to_test:
        try:
            if model_type == "adapter":
                model, tokenizer = load_base_with_adapter(path)
            else:
                model, tokenizer = load_merged(path)

            test_model(model, tokenizer, name)

        except Exception as e:
            print(f"\nERROR loading {name}: {e}")

        finally:
            # Free memory before loading next model
            del model, tokenizer
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    print(f"\n{'='*60}")
    print("Compare outputs above. If Stage 1 is good but Final is bad,")
    print("then Stage 2/3 damaged the adapter.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()