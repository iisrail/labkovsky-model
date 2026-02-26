#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline using Vikhr-YandexGPT

Flow:
1. Retrieve top-k relevant documents from ChromaDB
2. Generate response using Vikhr's RAG format (documents role)

Uses the same data sources as fine-tuning for grounded responses.

Usage:
    python query_rag.py
"""

import argparse
import json
import re
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import chromadb

# ============================================================
# PATHS (adjust to your setup)
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up to project root
CHROMA_DIR = PROJECT_DIR / "chroma_db"
MODELS_DIR = PROJECT_DIR / "models"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "./models/vikhr-book-merged"
LORA_PATH = MODELS_DIR / "labkovsky-rag-context-lora"

# System prompt template - docs will be inserted
SYSTEM_PROMPT_TEMPLATE = (
    "Ты психолог Михаил Лабковский. Используй следующие документы для ответа:\n\n"
    "{docs}\n\n"
    "Отвечай в стиле Лабковского: прямо, уверенно, с конкретными рекомендациями. "
    "Сначала объясни причину проблемы, затем дай конкретные шаги для решения. "
    "Если видишь, что кому-то в ситуации нужна профессиональная помощь — скажи об этом прямо."
)

# ============================================================
# GLOBAL STATE
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

def init_retrieval(chroma_dir: Path):
    global _collection
    if _collection is None:
        print("[+] Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _collection = client.get_collection("labkovsky")
        print(f"   Loaded {_collection.count()} documents")
    return _collection

def init_llm(use_lora: bool = True, lora_path: Path = None):
    global _llm, _tokenizer
    if _llm is None:
        print(f"[+] Loading LLM: {LLM_MODEL_NAME}")

        # Always load tokenizer from base model (Unsloth LoRA doesn't add tokens)
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        _tokenizer.pad_token = _tokenizer.eos_token

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

        if use_lora and lora_path and lora_path.exists():
            print(f"   Loading LoRA: {lora_path}")
            _llm = PeftModel.from_pretrained(base_model, str(lora_path))
        else:
            print("   (base model, no LoRA)")
            _llm = base_model

        _llm.eval()
        print(f"[OK] LLM loaded ({'with LoRA' if use_lora and lora_path and lora_path.exists() else 'base only'})")

    return _llm, _tokenizer

# ============================================================
# RETRIEVAL
# ============================================================

TOP_K = 2
INCLUDE_SIBLINGS = True


def _get_sibling_ids(doc_id: str, metadata: dict) -> list:
    """
    Generate IDs for neighboring chunks (N-1, N+1).

    Handles formats like:
    - "article_id_chunk2" -> ["article_id_chunk1", "article_id_chunk3"]
    - "video_01_expl" -> [] (Q&A, no siblings)
    """
    chunk_id = metadata.get("chunk_id")
    if chunk_id is None:
        return []

    try:
        chunk_num = int(chunk_id)
    except (ValueError, TypeError):
        return []

    # Build sibling IDs based on document ID pattern
    siblings = []
    base_id = doc_id.rsplit("_chunk", 1)[0] if "_chunk" in doc_id else None

    if base_id:
        # Article/book format: "article_id_chunk2"
        if chunk_num > 0:
            siblings.append(f"{base_id}_chunk{chunk_num - 1}")
        siblings.append(f"{base_id}_chunk{chunk_num + 1}")

    return siblings


def _fetch_siblings(collection, doc_ids: list) -> dict:
    """Fetch documents by IDs from collection."""
    if not doc_ids:
        return {}

    try:
        results = collection.get(
            ids=doc_ids,
            include=["documents", "metadatas"]
        )

        siblings = {}
        for doc_id, doc, meta in zip(
            results['ids'],
            results['documents'],
            results['metadatas']
        ):
            siblings[doc_id] = {
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "distance": None,  # Not from similarity search
                "is_sibling": True
            }
        return siblings
    except Exception:
        return {}


def retrieve(query: str, chroma_dir: Path = CHROMA_DIR, top_k: int = TOP_K, include_siblings: bool = INCLUDE_SIBLINGS) -> list:
    """
    Retrieve top-k relevant documents from ChromaDB.

    If include_siblings=True, also fetches neighboring chunks (N-1, N+1)
    for articles/book to provide context for connected chunks.

    Returns list of dicts with 'text', 'metadata', 'distance' keys.
    """
    embed_model = init_embedding()
    collection = init_retrieval(chroma_dir)

    query_embedding = embed_model.encode(f"query: {query}")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    if not results['ids'][0]:
        return []

    documents = []
    seen_ids = set()
    sibling_ids_to_fetch = []

    for doc_id, doc, meta, dist in zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        documents.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "is_sibling": False
        })
        seen_ids.add(doc_id)

        # Collect sibling IDs
        if include_siblings:
            for sib_id in _get_sibling_ids(doc_id, meta):
                if sib_id not in seen_ids:
                    sibling_ids_to_fetch.append(sib_id)
                    seen_ids.add(sib_id)

    # Fetch siblings
    if sibling_ids_to_fetch:
        siblings = _fetch_siblings(collection, sibling_ids_to_fetch)

        # Insert siblings in correct order (before/after their parent)
        expanded_docs = []
        for doc in documents:
            doc_id = doc["id"]
            chunk_id = doc["metadata"].get("chunk_id")

            if chunk_id is not None:
                try:
                    chunk_num = int(chunk_id)
                    base_id = doc_id.rsplit("_chunk", 1)[0] if "_chunk" in doc_id else None

                    if base_id:
                        # Add previous sibling first
                        prev_id = f"{base_id}_chunk{chunk_num - 1}"
                        if prev_id in siblings:
                            expanded_docs.append(siblings[prev_id])

                        # Add main doc
                        expanded_docs.append(doc)

                        # Add next sibling after
                        next_id = f"{base_id}_chunk{chunk_num + 1}"
                        if next_id in siblings:
                            expanded_docs.append(siblings[next_id])
                    else:
                        expanded_docs.append(doc)
                except (ValueError, TypeError):
                    expanded_docs.append(doc)
            else:
                expanded_docs.append(doc)

        documents = expanded_docs

    return documents

# ============================================================
# GENERATION
# ============================================================



def generate(query: str, context_docs: list, use_lora: bool = True, lora_path: Path = LORA_PATH) -> str:
    """
    Generate response with docs in system prompt (not documents role).

    Args:
        query: User question
        context_docs: List of dicts with 'text' and 'metadata' keys
        use_lora: Whether to use LoRA adapter
        lora_path: Path to LoRA adapter

    Returns:
        Generated response string
    """
    llm, tokenizer = init_llm(use_lora=use_lora, lora_path=lora_path)

    # Format documents as text for system prompt
    docs_text = "\n\n".join([
        f"Документ {i+1}: {doc['text']}"
        for i, doc in enumerate(context_docs)
    ])

    # Put docs in system prompt (NOT documents role - that's for ranking)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

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
            no_repeat_ngram_size=4,  # Prevent copying 4+ word sequences from RAG
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # No stripping — show raw output with RS markers for debugging

    return response

# ============================================================
# MAIN FUNCTION
# ============================================================

def ask_labkovsky(query: str) -> dict:
    """Query the RAG pipeline and generate a response."""
    # Step 1: Retrieve top-k relevant documents
    docs = retrieve(query)

    if not docs:
        return {
            "answer": "Не нашёл подходящих документов.",
            "docs": [],
            "best_distance": None,
            "docs_used": 0,
        }

    best_distance = next((d["distance"] for d in docs if d["distance"] is not None), None)

    # Step 2: Use all retrieved docs for richer context
    docs_for_generation = docs

    # Step 3: Generate response grounded on documents
    answer = generate(query, docs_for_generation)

    return {
        "answer": answer,
        "docs": docs,  # Return all retrieved for verbose display
        "best_distance": best_distance,
        "docs_used": len(docs_for_generation),
    }
# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-dir", type=str, default=str(CHROMA_DIR), help="ChromaDB directory")
    parser.add_argument("--lora-path", type=str, default=str(LORA_PATH), help="LoRA adapter path")
    parser.add_argument("--no-lora", action="store_true", help="Use base model without LoRA")
    args = parser.parse_args()

    print("=" * 60)
    print("Labkovsky RAG")
    print("=" * 60)
    print("\nCommands:")
    print("  'exit' — quit")
    print("  'verbose' — toggle source display")
    print()

    # Initialize all components
    print("Loading models...")
    init_embedding()
    init_retrieval(Path(args.chroma_dir))

    lora_path = None if args.no_lora else Path(args.lora_path)
    init_llm(use_lora=not args.no_lora, lora_path=lora_path)

    print("\n[OK] Ready!\n")

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

        try:
            result = ask_labkovsky(query)

            if verbose:
                dist = result['best_distance']
                used = result['docs_used']
                dist_str = f"{dist:.3f}" if dist is not None else "N/A"
                print(f"\n--- Best Distance: {dist_str} | Docs used: {used}/{len(result['docs'])} ---")
                print(f"\n--- Retrieved Documents ({len(result['docs'])}) ---")
                for i, doc in enumerate(result['docs'], 1):
                    meta = doc['metadata']
                    source = meta.get('source_type', 'unknown')
                    doc_id = doc.get('id', 'unknown')
                    is_sib = doc.get('is_sibling', False)
                    sib_tag = " [sibling]" if is_sib else ""
                    dist_str = f"dist: {doc['distance']:.3f}" if doc['distance'] else "context"
                    print(f"\n[{i}] {source} | {doc_id} | {dist_str}{sib_tag}")
                    print(f"    {doc['text'][:300]}...")
                print(f"\n--- Response ---")
            print(f"\nLabkovsky: {result['answer']}\n")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()