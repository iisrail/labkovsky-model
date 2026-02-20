"""Compare original Vikhr vs merged base model: generation quality + perplexity on QA data."""

import gc
import json
import math
import random
from pathlib import Path

import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent
QA_FILE = PROJECT_ROOT / "data" / "fine_tuning" / "qa_rs_final.jsonl"

MODELS = {
    "original": "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it",
    "merged": "./models/vikhr-book-merged",
}

SYSTEM_PROMPT = "Ты — психолог. Ответь кратко."
USER_PROMPT = "Мой муж мне изменяет, что делать?"

N_PPL_SAMPLES = 10
RANDOM_SEED = 42


def load_qa_samples(path: Path, n: int, seed: int) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            a = obj.get("answer_segmented", obj.get("answer", ""))
            if q and a:
                records.append({"question": q, "answer": a})
    random.seed(seed)
    random.shuffle(records)
    return records[:n]


def generate_response(model, tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    # Decode only generated tokens
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def compute_perplexity(model, tokenizer, samples: list[dict], system_prompt: str) -> tuple[float, list[float]]:
    """Compute perplexity on assistant completions only."""
    per_sample = []

    for s in samples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": s["question"]},
            {"role": "assistant", "content": s["answer"]},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # Get prompt-only (without assistant answer) to find where completion starts
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": s["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        full_ids = tokenizer(full_text, return_tensors="pt").to(model.device)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**full_ids)
            logits = outputs.logits  # (1, seq_len, vocab)

        # Shift: predict token i+1 from position i
        # We only care about positions [prompt_len-1 : seq_len-1] predicting [prompt_len : seq_len]
        shift_logits = logits[0, prompt_len - 1:-1, :]  # (completion_len, vocab)
        shift_labels = full_ids["input_ids"][0, prompt_len:]  # (completion_len,)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits, shift_labels).item()
        ppl = math.exp(loss)
        per_sample.append(ppl)

    avg_ppl = sum(per_sample) / len(per_sample)
    return avg_ppl, per_sample


def run_model(name: str, model_path: str, qa_samples: list[dict]) -> dict:
    print(f"\n{'=' * 70}")
    print(f"Loading {name}: {model_path}")
    print(f"{'=' * 70}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Generation test
    print(f"\nGenerating response...")
    response = generate_response(model, tokenizer, SYSTEM_PROMPT, USER_PROMPT)

    # Perplexity test
    print(f"Computing perplexity on {len(qa_samples)} QA samples...")
    avg_ppl, per_sample = compute_perplexity(
        model, tokenizer, qa_samples,
        "Ты — психолог Михаил Лабковский. "
        "Отвечай в его стиле: прямо, уверенно, с конкретными рекомендациями. "
        "Используй простой язык и жизненные примеры."
    )

    result = {
        "response": response,
        "avg_ppl": avg_ppl,
        "per_sample_ppl": per_sample,
    }

    # Free GPU
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{name} unloaded.")

    return result


def main():
    qa_samples = load_qa_samples(QA_FILE, N_PPL_SAMPLES, RANDOM_SEED)
    print(f"Loaded {len(qa_samples)} QA samples for perplexity test")

    results = {}
    for name, path in MODELS.items():
        results[name] = run_model(name, path, qa_samples)

    # Print comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")

    print(f"\nPrompt: [{SYSTEM_PROMPT}] {USER_PROMPT}")
    for name in MODELS:
        print(f"\n{'─' * 70}")
        print(f"{'ORIGINAL' if name == 'original' else 'MERGED'} response:")
        print(f"{'─' * 70}")
        print(results[name]["response"])

    print(f"\n{'─' * 70}")
    print("PERPLEXITY (on assistant completions, 10 QA samples)")
    print(f"{'─' * 70}")
    print(f"  Original avg PPL: {results['original']['avg_ppl']:.2f}")
    print(f"  Merged   avg PPL: {results['merged']['avg_ppl']:.2f}")
    diff_pct = (results['merged']['avg_ppl'] - results['original']['avg_ppl']) / results['original']['avg_ppl'] * 100
    print(f"  Difference: {diff_pct:+.1f}%")

    print(f"\nPer-sample perplexity:")
    print(f"  {'#':<4} {'Original':>12} {'Merged':>12} {'Diff%':>8}")
    for i in range(len(qa_samples)):
        o = results['original']['per_sample_ppl'][i]
        m = results['merged']['per_sample_ppl'][i]
        d = (m - o) / o * 100
        print(f"  {i:<4} {o:>12.2f} {m:>12.2f} {d:>+7.1f}%")


if __name__ == "__main__":
    main()
