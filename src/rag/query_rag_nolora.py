#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline using merged Labkovsky model.

Model: models/labkovsky-final-merged (16-bit merged, no LoRA needed)

Usage:
    python query_rag_nolora.py
"""

import argparse
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel
import chromadb

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
MODELS_DIR = PROJECT_DIR / "models"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
MERGED_MODEL_PATH = MODELS_DIR / "labkovsky-final-merged"

# ============================================================
# PROMPT FORMAT (matches training)
# ============================================================

INSTRUCTION = "Инструкция: Ответь строго в формате: [ОБЪЯСНЕНИЕ]...[ВМЕШАТЕЛЬСТВО]...[ЭСКАЛАЦИЯ]..."

def format_prompt(query: str) -> str:
    return f"{INSTRUCTION}\n\nВопрос: {query}\nОтвет:"

# ============================================================
# GLOBALS
# ============================================================

_embed_model = None
_collection = None
_llm = None
_tokenizer = None

def init_embedding():
    global _embed_model
    if _embed_model is None:
        print("[+] Loading embedding model...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model

def init_retrieval():
    global _collection
    if _collection is None:
        print("[+] Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection("labkovsky")
        print(f"    {_collection.count()} documents")
    return _collection

def init_llm():
    global _llm, _tokenizer
    if _llm is None:
        print(f"[+] Loading model: {MERGED_MODEL_PATH.name}")
        _llm, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(MERGED_MODEL_PATH),
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_llm)
        print("[OK] Model loaded")
    return _llm, _tokenizer

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query: str, top_k: int = 3) -> list:
    embed_model = init_embedding()
    collection = init_retrieval()

    query_embedding = embed_model.encode(f"query: {query}")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    if not results['ids'][0]:
        return []

    docs = []
    for doc_id, doc, meta, dist in zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        docs.append({"id": doc_id, "text": doc, "metadata": meta, "distance": dist})

    return docs

# ============================================================
# GENERATION
# ============================================================

def generate(query: str, docs: list) -> str:
    llm, tokenizer = init_llm()

    # Build prompt with retrieved context
    if docs:
        context = "\n\n".join([doc["text"] for doc in docs[:3]])
        prompt = f"""{INSTRUCTION}

Контекст:
{context}

Вопрос: {query}
Ответ:"""
    else:
        prompt = format_prompt(query)

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return response

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Labkovsky RAG")
    print("=" * 60)

    init_embedding()
    init_retrieval()
    init_llm()

    print("\n[OK] Ready! ('exit' to quit, 'verbose' to toggle docs)\n")

    verbose = False

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nPoka!")
            break

        if not query:
            continue
        if query.lower() in ['exit', 'quit', 'q']:
            print("Poka!")
            break
        if query.lower() == 'verbose':
            verbose = not verbose
            print(f"Verbose: {'ON' if verbose else 'OFF'}")
            continue

        print("\n...")

        docs = retrieve(query)
        if verbose and docs:
            print(f"\n--- Docs ({len(docs)}) ---")
            for i, doc in enumerate(docs, 1):
                print(f"[{i}] dist={doc['distance']:.3f} | {doc['text'][:100]}...")

        answer = generate(query, docs)
        print(f"\nLabkovsky: {answer}\n")

if __name__ == "__main__":
    main()
