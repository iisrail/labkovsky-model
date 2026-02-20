#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Combined LoRA with RAG

Interactive testing of merged Q&A + Book LoRA adapters with ChromaDB retrieval.
Supports dynamic weight adjustment to compare different blend ratios.

Usage:
    python src/fine_tuning/test_combined_lora_rag.py

Commands:
    - Type a question to get a response
    - "weight 0.3" - change book weight (qa weight stays 1.0)
    - "verbose" - toggle showing retrieved chunks
    - "exit" - quit
"""

import re
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_DIR / "models"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

# Adapters
QA_LORA = MODELS_DIR / "labkovsky-vikhr-lora-attn4"
BOOK_LORA = MODELS_DIR / "labkovsky-book-lora-r8-mlp"

# Models
BASE_MODEL = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# ChromaDB
COLLECTION_NAME = "labkovsky"

# Generation
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_K_RETRIEVAL = 3

# Default weights
DEFAULT_QA_WEIGHT = 1.0
DEFAULT_BOOK_WEIGHT = 0.9

# System prompt template - docs will be inserted (NOT documents role - that's for ranking)
SYSTEM_PROMPT_TEMPLATE = (
    "Ты психолог Михаил Лабковский. Используй следующие документы для ответа:\n\n"
    "{docs}\n\n"
    "Отвечай в стиле Лабковского: прямо, уверенно, с конкретными рекомендациями."
)

# ============================================================
# GLOBAL STATE
# ============================================================

_model = None
_tokenizer = None
_embed_model = None
_collection = None
_current_book_weight = DEFAULT_BOOK_WEIGHT
_verbose = True  # Show chunks by default


def init_embedding():
    """Load embedding model for retrieval."""
    global _embed_model
    if _embed_model is None:
        print("[+] Loading embedding model...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def init_chromadb():
    """Connect to ChromaDB collection."""
    global _collection
    if _collection is None:
        print("[+] Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
        print(f"   Loaded {_collection.count()} documents")
    return _collection


def init_model():
    """Load base model with both LoRA adapters."""
    global _model, _tokenizer

    print(f"[+] Loading base model: {BASE_MODEL}")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    print(f"[+] Loading Q&A adapter: {QA_LORA.name}")
    _model = PeftModel.from_pretrained(
        _model,
        str(QA_LORA),
        adapter_name="qa",
    )

    print(f"[+] Loading Book adapter: {BOOK_LORA.name}")
    _model.load_adapter(str(BOOK_LORA), adapter_name="book")

    print(f"   Adapters loaded: {list(_model.peft_config.keys())}")

    # Create initial combined adapter
    rebuild_combined_adapter(DEFAULT_BOOK_WEIGHT)

    # Enable inference mode
    FastLanguageModel.for_inference(_model)

    return _model, _tokenizer


def rebuild_combined_adapter(book_weight: float):
    """Rebuild combined adapter with new weights."""
    global _model, _current_book_weight

    qa_weight = DEFAULT_QA_WEIGHT
    _current_book_weight = book_weight

    print(f"[*] Combining adapters: qa={qa_weight}, book={book_weight}")

    # Remove old combined adapter if exists
    if "combined" in _model.peft_config:
        _model.delete_adapter("combined")

    # Create new combined adapter
    _model.add_weighted_adapter(
        adapters=["qa", "book"],
        weights=[qa_weight, book_weight],
        adapter_name="combined",
        combination_type="linear",
    )

    _model.set_adapter("combined")
    print(f"   Active adapter: combined (qa={qa_weight}, book={book_weight})")


def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
    """Retrieve relevant chunks from ChromaDB."""
    embed_model = init_embedding()
    collection = init_chromadb()

    # Encode query with prefix (same as index creation)
    query_embedding = embed_model.encode(f"query: {query}")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    if not results['ids'][0]:
        return []

    chunks = []
    for doc_id, doc, meta, dist in zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        chunks.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
        })

    return chunks


def build_prompt(query: str, chunks: list) -> str:
    """Build prompt with docs in system prompt (NOT documents role - that's for ranking)."""

    # Format documents as text for system prompt
    docs_text = "\n\n".join([
        f"Документ {i+1}: {chunk['text']}"
        for i, chunk in enumerate(chunks)
    ])

    # Put docs in system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def generate(prompt: str) -> str:
    """Generate response from the model."""
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,  # Prevent copying 4+ word sequences from RAG
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only new tokens
    response = _tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # Strip RS markers (including garbled/merged ones)
    response = re.sub(r'\[[А-ЯЁа-яё]{3,30}\]\s*', '', response)  # Any Cyrillic bracketed marker
    response = re.sub(r'(?:НЕ\s+)?[Нн]е требу[ею]тся\.?\s*', '', response)
    response = re.sub(r'НЕ ТРЕБУЕТСЯ\s*', '', response)
    # Clean up trailing escalation phrases
    response = re.sub(r',?\s*(?:но\s+)?рекомендую обратиться к (?:психологу|специалисту)\.?\s*$', '', response)
    response = re.sub(r',?\s*так как ситуация требует психологической помощи\.?\s*$', '', response)
    response = re.sub(r'\n\s*\n', '\n\n', response)
    response = re.sub(r'  +', ' ', response)

    return response.strip()


def print_chunks(chunks: list):
    """Print retrieved chunks for visibility."""
    print("\n" + "─" * 50)
    print(f"Retrieved {len(chunks)} chunks:")
    print("─" * 50)
    for i, chunk in enumerate(chunks, 1):
        source = chunk['metadata'].get('source', '?')
        dist = chunk['distance']
        text_preview = chunk['text'][:200].replace('\n', ' ')
        print(f"\n[{i}] ({source}, dist={dist:.3f})")
        print(f"    {text_preview}...")
    print("─" * 50 + "\n")


def main():
    global _verbose

    print("=" * 60)
    print("Combined LoRA + RAG Test")
    print("=" * 60)
    print()

    # Initialize everything
    init_model()
    init_embedding()
    init_chromadb()

    print()
    print("[OK] Ready!")
    print()
    print("Commands:")
    print("  'weight X.X' - change book weight (e.g., 'weight 0.3')")
    print("  'verbose'    - toggle chunk display")
    print("  'exit'       - quit")
    print()

    while True:
        try:
            user_input = input(f"You [book={_current_book_weight}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        # Check for commands
        if user_input.lower() == "exit":
            print("Пока!")
            break

        if user_input.lower() == "verbose":
            _verbose = not _verbose
            print(f"Verbose mode: {'ON' if _verbose else 'OFF'}")
            continue

        if user_input.lower().startswith("weight "):
            try:
                new_weight = float(user_input.split()[1])
                if 0 <= new_weight <= 2:
                    rebuild_combined_adapter(new_weight)
                else:
                    print("Weight should be between 0 and 2")
            except (IndexError, ValueError):
                print("Usage: weight 0.5")
            continue

        # Regular query - do RAG
        print()

        # Retrieve
        chunks = retrieve(user_input)

        if _verbose and chunks:
            print_chunks(chunks)

        if not chunks:
            print("(No relevant chunks found)")

        # Build prompt and generate
        prompt = build_prompt(user_input, chunks)
        response = generate(prompt)

        print(f"Лабковский: {response}")
        print()


if __name__ == "__main__":
    main()
