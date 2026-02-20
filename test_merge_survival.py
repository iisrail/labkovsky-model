"""Compare original Vikhr vs merged model outputs to check vocabulary changes."""

import gc
import torch
from unsloth import FastLanguageModel

PROMPTS = [
    "Невроз — это когда",
    "Чувство вины — это не",
    "Тревога заставляет человека",
]

GEN_KWARGS = dict(
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)


def generate_outputs(model, tokenizer, prompts: list[str], seed: int = 42) -> list[str]:
    results = []
    for prompt in prompts:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, **GEN_KWARGS)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(text)
    return results


def main():
    # --- Original model ---
    print("=" * 80)
    print("Loading ORIGINAL Vikhr model...")
    print("=" * 80)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    original_outputs = generate_outputs(model, tokenizer, PROMPTS)

    # Free GPU
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("\nOriginal model unloaded, GPU freed.\n")

    # --- Merged model ---
    print("=" * 80)
    print("Loading MERGED model (vikhr-book-merged)...")
    print("=" * 80)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./models/vikhr-book-merged",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    merged_outputs = generate_outputs(model, tokenizer, PROMPTS)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- Print side by side ---
    print("\n" + "=" * 80)
    print("COMPARISON: Original vs Merged")
    print("=" * 80)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'─' * 80}")
        print(f"PROMPT: {prompt}")
        print(f"{'─' * 80}")
        print(f"\n🔵 ORIGINAL:\n{original_outputs[i]}")
        print(f"\n🟢 MERGED:\n{merged_outputs[i]}")


if __name__ == "__main__":
    main()
