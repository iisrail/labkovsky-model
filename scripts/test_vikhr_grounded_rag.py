#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypothesis Test: Does Vikhr's grounded generation fail on ideational/psychological content,
or does it only work for factual extraction?

Test Matrix:
  Test 1: Original Vikhr demo (climate docs + topic query) — baseline, should work
  Test 2: Labkovsky psychological principles + real user question — the critical test
  Test 3: Labkovsky principles + simple topic query (like their demo style) — isolates content vs query style

Uses transformers directly with 4-bit quantization (fits on 12GB GPU).

Usage:
    source venv_wsl/bin/activate
    python scripts/test_vikhr_grounded_rag.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"

# ============================================================
# GROUNDED SYSTEM PROMPT — exact copy from Vikhr documentation
# ============================================================
GROUNDED_SYSTEM_PROMPT = (
    "Your task is to answer the user's questions using only the information "
    "from the provided documents. Give two answers to each question: one with "
    "a list of relevant document identifiers and the second with the answer to "
    "the question itself, using documents with these identifiers."
)

# ============================================================
# LABKOVSKY-STYLE PROMPT — teaches HOW to apply principles
# ============================================================
LABKOVSKY_SYSTEM_PROMPT = (
    "You are psychologist Mikhail Labkovsky. Below are reference materials "
    "containing psychological principles.\n\n"
    "Use these principles as the foundation for your reasoning. "
    "Apply them to the user's specific situation — do not copy verbatim.\n\n"
    "Answer in Labkovsky's style: blunt, confident, with tough love if needed. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
)

# ============================================================
# TEST 1: Exact Vikhr demo — factual climate documents
# ============================================================
FACTUAL_DOCUMENTS = [
    {
        "doc_id": 0,
        "title": "Глобальное потепление: ледники",
        "content": "За последние 50 лет объем ледников в мире уменьшился на 30%"
    },
    {
        "doc_id": 1,
        "title": "Глобальное потепление: Уровень моря",
        "content": "Уровень мирового океана повысился на 20 см с 1880 года и продолжает расти на 3,3 мм в год"
    }
]
FACTUAL_QUERY = "Глобальное потепление"

# ============================================================
# TEST 2: Labkovsky principles + real psychological question
# ============================================================
LABKOVSKY_DOCUMENTS = [
    {
        "doc_id": 0,
        "title": "Лабковский: Границы с родителями",
        "content": (
            "Взрослый человек не обязан терпеть то, что ему не нравится, даже если это исходит от родителей. "
            "Если мать звонит и критикует — можно сказать: 'Мама, если ты будешь продолжать, я повешу трубку.' "
            "И повесить. Это не жестокость, это граница. Родители учатся уважать ваши границы только тогда, "
            "когда вы их последовательно защищаете."
        )
    },
    {
        "doc_id": 1,
        "title": "Лабковский: Почему люди остаются в плохих отношениях",
        "content": (
            "Люди остаются в отношениях, которые причиняют боль, не потому что любят партнёра, "
            "а потому что привыкли к страданию. Это знакомое состояние — оно ощущается как 'нормальное'. "
            "Здоровый человек уходит оттуда, где ему плохо. Если вы не можете уйти — проблема не в партнёре, "
            "а в вашем отношении к себе."
        )
    },
    {
        "doc_id": 2,
        "title": "Лабковский: Что такое любовь к себе",
        "content": (
            "Любовь к себе — это не пузырьковые ванны и шоппинг. Это способность не соглашаться на то, "
            "что вам не подходит. Не терпеть. Не ждать, что станет лучше. Делать только то, что хочется. "
            "Это звучит эгоистично, но на практике именно такие люди строят здоровые отношения."
        )
    }
]
LABKOVSKY_REAL_QUERY = "Почему я не могу уйти от мужа, который меня не уважает?"
LABKOVSKY_TOPIC_QUERY = "Отношения и любовь к себе"  # topic-style query like their demo


# ============================================================
# MODEL LOADING
# ============================================================
_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print(f"[+] Loading model: {MODEL_NAME} (4-bit quantized)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()

    print("[OK] Model loaded")
    return _model, _tokenizer


# ============================================================
# GROUNDED GENERATION
# ============================================================

def run_single_step_with_docs_role(documents: list, query: str, test_name: str, system_prompt: str):
    """
    Single-step generation using documents role but with custom system prompt.
    Tests whether the documents role itself works when given a better prompt.
    """
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Documents: {len(documents)} provided")
    print(f"System prompt: {system_prompt[:80]}...")
    print(f"{'-'*70}")

    model, tokenizer = load_model()

    # Format docs as text (not JSON) for the documents role
    docs_text = "\n\n".join([
        f"[Doc {doc['doc_id']}] {doc['title']}: {doc['content']}"
        for doc in documents
    ])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "documents", "content": docs_text},
        {"role": "user", "content": query},
    ]

    try:
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
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        print(f"\nAnswer:\n{answer}")
        return {"test": test_name, "answer": answer}
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"test": test_name, "answer": f"ERROR: {e}"}


def run_grounded_rag(documents: list, query: str, test_name: str):
    """
    Runs Vikhr's two-step grounded generation exactly as documented.
    Step 1: model returns relevant doc IDs
    Step 2: model generates answer using those docs

    Uses the special 'documents' role in chat template.
    """
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Documents: {len(documents)} provided")
    for doc in documents:
        print(f"  doc_id={doc['doc_id']}: {doc['title']}")
    print(f"{'-'*70}")

    model, tokenizer = load_model()

    # Build messages with 'documents' role (Vikhr's special RAG format)
    messages = [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
        {"role": "documents", "content": json.dumps(documents, ensure_ascii=False)},
        {"role": "user", "content": query},
    ]

    # Step 1: Get relevant document IDs
    try:
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
                max_new_tokens=256,
                do_sample=False,  # greedy for doc selection
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        relevant_indexes = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        print(f"\nStep 1 (doc selection): {relevant_indexes}")
    except Exception as e:
        print(f"\nStep 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"test": test_name, "step1": f"ERROR: {e}", "step2": None}

    # Step 2: Generate answer using selected docs
    try:
        messages_step2 = messages + [{"role": "assistant", "content": relevant_indexes}]
        prompt2 = tokenizer.apply_chat_template(
            messages_step2,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
        input_len2 = inputs2["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs2 = model.generate(
                **inputs2,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        final_answer = tokenizer.decode(outputs2[0][input_len2:], skip_special_tokens=True).strip()
        print(f"\nStep 2 (final answer):\n{final_answer}")
    except Exception as e:
        print(f"\nStep 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"test": test_name, "step1": relevant_indexes, "step2": f"ERROR: {e}"}

    return {
        "test": test_name,
        "step1_doc_ids": relevant_indexes,
        "step2_answer": final_answer,
    }


def main():
    print("=" * 70)
    print("Vikhr Grounded RAG Hypothesis Test")
    print("Testing 'documents' role with factual vs psychological content")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = []

    # Test 1: Their exact demo — must work as baseline
    results.append(run_grounded_rag(
        FACTUAL_DOCUMENTS, FACTUAL_QUERY,
        "TEST 1: Vikhr demo (factual docs + topic query) — BASELINE"
    ))

    # Test 2: Labkovsky + real user question — the critical test
    results.append(run_grounded_rag(
        LABKOVSKY_DOCUMENTS, LABKOVSKY_REAL_QUERY,
        "TEST 2: Labkovsky docs + real psychological question — CRITICAL"
    ))

    # Test 3: Labkovsky + topic-style query — isolates content type from query style
    results.append(run_grounded_rag(
        LABKOVSKY_DOCUMENTS, LABKOVSKY_TOPIC_QUERY,
        "TEST 3: Labkovsky docs + topic query — ISOLATION"
    ))

    # Test 4: Labkovsky prompt + documents role + real question
    # Tests: is the problem the two-step prompt, or the documents role itself?
    results.append(run_single_step_with_docs_role(
        LABKOVSKY_DOCUMENTS, LABKOVSKY_REAL_QUERY,
        "TEST 4: Labkovsky prompt + documents role + real question — PROMPT TEST",
        LABKOVSKY_SYSTEM_PROMPT
    ))

    # ============================================================
    # INTERPRETATION GUIDE
    # ============================================================
    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("""
    ORIGINAL TESTS (1-3):

    Test 1 WORKS + Test 2 FAILS + Test 3 WORKS:
      -> The two-step "extractive" prompt is the problem.
        It can match topics but can't reason from principles.

    Test 1 FAILS:
      -> The documents role mechanism doesn't work at all.

    NEW TEST (4) — ISOLATES PROMPT VS DOCUMENTS ROLE:

    Test 4 WORKS:
      -> The documents role CAN work with a better prompt!
        The problem was Vikhr's two-step extractive prompt,
        not the documents role itself.
        BUT: you still need LoRA training to get Labkovsky's style.

    Test 4 FAILS (generic/flat answer):
      -> The documents role itself has limitations.
        Even with a good prompt, the model doesn't fully utilize
        the documents for reasoning. Your system-prompt approach
        (docs in system message, not documents role) works better.

    Test 4 REFUSES ("can't find info"):
      -> The documents role enforces strict grounding that
        prevents reasoning from principles. The mechanism itself
        is designed for extraction, not application.
    """)


if __name__ == "__main__":
    main()
