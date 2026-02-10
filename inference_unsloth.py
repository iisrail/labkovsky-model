#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for Labkovsky model trained with Unsloth.
Handles RS tokens properly.

Usage:
    python inference_unsloth.py
"""

import json
import pickle
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# ==============================================================
# SETTINGS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
LORA_PATH = PROJECT_ROOT / "models" / "labkovsky-qwen7b-lora-unsloth"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
RS_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "rs_classifier" / "rs_classifier.pkl"

RS_TOKENS = ["<RS=INTERVENTION>", "<RS=EXPLANATION>", "<RS=ESCALATION>"]
NUM_FEW_SHOT = 3  # Number of examples to retrieve for content

# Context instruction - forces model to USE the retrieved content
CONTEXT_INSTRUCTION = """Ты — Михаил Лабковский. Ниже твои РЕАЛЬНЫЕ ответы на похожие вопросы.

ВАЖНО: Ответь ТОЛЬКО на основе информации ниже. Перефразируй и адаптируй, но НЕ выдумывай новое.

---
ТВОИ ПРЕДЫДУЩИЕ ОТВЕТЫ:
---"""

# Global state
_embed_model = None
_collection = None
_rs_classifier = None
_rs_id2label = None

# ==============================================================
# RAG: CHROMADB RETRIEVAL
# ==============================================================

def load_rag_corpus():
    """Load ChromaDB collection for RAG retrieval."""
    global _embed_model, _collection

    if _collection is not None:
        return

    print("📚 Loading RAG from ChromaDB...")

    # Load embedding model
    _embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = client.get_collection("labkovsky")

    print(f"   Loaded {_collection.count()} documents")
    print("✅ RAG ready")


def retrieve_similar(question: str, top_k: int = NUM_FEW_SHOT):
    """Retrieve similar documents from ChromaDB."""
    query_embedding = _embed_model.encode(f"query: {question}")

    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    retrieved = []
    for i in range(len(results['ids'][0])):
        doc = results['documents'][0][i]
        metadata = results['metadatas'][0][i]

        # Extract RS from metadata if available
        rs = metadata.get('response_signal', 'INTERVENTION')

        retrieved.append({
            'text': doc,
            'rs': rs,
            'source': metadata.get('source_file', 'unknown')
        })

    return retrieved


# ==============================================================
# RS CLASSIFIER
# ==============================================================

def load_rs_classifier():
    """Load RS classifier model."""
    global _rs_classifier, _rs_id2label

    if _rs_classifier is not None:
        return

    print("🎯 Loading RS classifier...")

    with open(RS_CLASSIFIER_PATH, "rb") as f:
        model_data = pickle.load(f)

    _rs_classifier = model_data["classifier"]
    _rs_id2label = model_data["id2rs"]

    print(f"   Labels: {list(_rs_id2label.values())}")
    print("✅ RS classifier ready")


def predict_rs(question: str) -> str:
    """Predict response signal for a question."""
    question_prefixed = f"query: {question}"
    embedding = _embed_model.encode([question_prefixed])
    pred_id = _rs_classifier.predict(embedding)[0]
    return _rs_id2label[pred_id]


# ==============================================================
# LOAD MODEL
# ==============================================================

def load_model():
    print("🤖 Loading model...")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer from LoRA path (has RS tokens)
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    # Resize base model embeddings to match tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    
    print("✅ Model ready")
    return model, tokenizer


def ask(model, tokenizer, question, rs="INTERVENTION", use_rag=True, verbose=False):
    """Generate response for a question with optional RAG few-shot examples."""

    if use_rag and _collection is not None:
        # Retrieve similar examples from ChromaDB
        examples = retrieve_similar(question, top_k=NUM_FEW_SHOT)

        # Build context prompt - emphasize using retrieved content
        context_parts = []
        for i, ex in enumerate(examples, 1):
            context_parts.append(f"[{i}] {ex['text']}")

        context = "\n\n".join(context_parts)

        prompt = f"""{CONTEXT_INSTRUCTION}

{context}

---

<RS={rs}>
Вопрос: {question}
Используй информацию выше для полного ответа.
Ответ:"""

        if verbose:
            print(f"\n[RAG] Using {len(examples)} examples from ChromaDB")
            for i, ex in enumerate(examples, 1):
                preview = ex['text'][:80].replace('\n', ' ')
                print(f"  [{i}] ({ex['source']}) {preview}...")
            print(f"[RAG] Total prompt tokens: ~{len(prompt.split())}")
    else:
        prompt = f"<RS={rs}>\nВопрос: {question}\nОтвет:"
        if verbose:
            print("[RAG] Disabled or not loaded")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,
            do_sample=True,
            top_p=0.85,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )

    # Decode only the generated part
    input_len = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Clean up - stop at various markers
    stop_markers = ["#", "Источник", "http", "Что можно", "<RS=", "---", "Вопрос:", "\n\n\n"]
    for stop in stop_markers:
        if stop in response:
            response = response.split(stop)[0]

    # Remove trailing gibberish (non-Cyrillic text at the end)
    # Find last complete Russian sentence
    sentences = re.split(r'(?<=[.!?])\s+', response)
    cleaned = []
    for sent in sentences:
        # Skip if sentence has too much non-Cyrillic
        cyrillic_ratio = len(re.findall(r'[а-яА-ЯёЁ]', sent)) / max(len(sent), 1)
        if cyrillic_ratio < 0.5:
            break
        cleaned.append(sent)

    return ' '.join(cleaned).strip() if cleaned else response.strip()


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("🎤 Labkovsky Inference (Unsloth-trained + RAG + RS)")
    print("=" * 60)

    # Load RAG corpus first (also loads embedding model)
    load_rag_corpus()

    # Load RS classifier (uses same embedding model)
    load_rs_classifier()

    model, tokenizer = load_model()

    # Test questions
    test_questions = [
        "Муж не помогает с ребенком, что делать?",
        "Как полюбить себя?",
        "Парень не звонит уже неделю, ждать или написать первой?",
    ]
    
    print("\n" + "=" * 60)
    print("📝 Test responses (with RAG + RS classifier):")
    print("=" * 60)

    for q in test_questions:
        rs = predict_rs(q)
        print(f"\n❓ {q}")
        print(f"🎯 RS: {rs}")
        answer = ask(model, tokenizer, q, rs=rs, use_rag=True, verbose=True)
        print(f"💬 {answer}")
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive mode (type 'exit' to quit, 'verbose' to toggle)")
    print("=" * 60)

    verbose = True

    while True:
        try:
            q = input("\nТы: ").strip()
            if q.lower() in ['exit', 'quit', 'q', 'выход']:
                print("👋 Пока!")
                break
            if q.lower() == 'verbose':
                verbose = not verbose
                print(f"Verbose: {'ON' if verbose else 'OFF'}")
                continue
            if not q:
                continue

            rs = predict_rs(q)
            if verbose:
                print(f"🎯 RS: {rs}")
            answer = ask(model, tokenizer, q, rs=rs, use_rag=True, verbose=verbose)
            print(f"\nЛабковский [{rs}]: {answer}")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Пока!")
            break


if __name__ == "__main__":
    main()