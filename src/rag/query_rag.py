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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
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
EXPAND_CHUNKS = True  # concatenate next chunks from same chapter/article


def _expand_with_next_chunks(collection, doc_id: str, doc_text: str, metadata: dict, max_extra: int = 2) -> str:
    """
    Concatenate up to max_extra next chunks within same chapter/article.
    Matches build_rag_training_data.py format exactly.
    """
    chunk_id = metadata.get("chunk_id")
    if chunk_id is None:
        return doc_text

    try:
        chunk_num = int(chunk_id)
    except (ValueError, TypeError):
        return doc_text

    group_id = metadata.get("chapter_id") or metadata.get("article_id")
    base_id = doc_id.rsplit("_chunk", 1)[0] if "_chunk" in doc_id else None

    if not base_id:
        return doc_text

    for offset in range(1, max_extra + 1):
        next_id = f"{base_id}_chunk{chunk_num + offset}"
        try:
            next_doc = collection.get(ids=[next_id], include=["documents", "metadatas"])
            if not next_doc["ids"]:
                break
            next_group = (next_doc["metadatas"][0].get("chapter_id")
                          or next_doc["metadatas"][0].get("article_id"))
            if next_group != group_id:
                break
            doc_text = doc_text + "\n" + next_doc["documents"][0]
        except Exception:
            break

    return doc_text


def retrieve(query: str, chroma_dir: Path = CHROMA_DIR, top_k: int = TOP_K, expand_chunks: bool = EXPAND_CHUNKS) -> list:
    """
    Retrieve documents matching training format:
    1. Top-k docs from book/articles/interviews (with chunk expansion)
    2. One closest QA doc (separate query)

    Returns list of dicts with 'text', 'metadata', 'distance' keys.
    """
    embed_model = init_embedding()
    collection = init_retrieval(chroma_dir)

    query_embedding = embed_model.encode(f"query: {query}")

    # Step 1: Retrieve from book/articles/interviews
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"source_type": {"$in": ["articles", "book", "interviews"]}},
        include=["documents", "metadatas", "distances"]
    )

    documents = []

    if results['ids'][0]:
        for doc_id, doc_text, meta, dist in zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Expand with next chunks (same chapter/article)
            if expand_chunks:
                doc_text = _expand_with_next_chunks(collection, doc_id, doc_text, meta)

            documents.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "distance": dist,
            })

    # Step 2: Add one closest QA doc
    qa_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1,
        where={"source_type": {"$eq": "qa_corpus"}},
        include=["documents", "metadatas", "distances"]
    )

    if qa_results['ids'][0]:
        qa_id = qa_results['ids'][0][0]
        qa_text = qa_results['documents'][0][0]
        qa_meta = qa_results['metadatas'][0][0]
        qa_dist = qa_results['distances'][0][0]

        documents.append({
            "id": qa_id,
            "text": qa_text,
            "metadata": qa_meta,
            "distance": qa_dist,
        })

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

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,  # Prevent copying 4+ word sequences from RAG
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

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

    best_distance = docs[0]["distance"] if docs else None

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
    print("Type 'exit' to quit\n")

    # Initialize all components
    print("Loading models...")
    init_embedding()
    init_retrieval(Path(args.chroma_dir))

    lora_path = None if args.no_lora else Path(args.lora_path)
    init_llm(use_lora=not args.no_lora, lora_path=lora_path)

    print("\n[OK] Ready!\n")

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

        print("\n...")

        try:
            result = ask_labkovsky(query)

            dist = result['best_distance']
            used = result['docs_used']
            dist_str = f"{dist:.3f}" if dist is not None else "N/A"
            print(f"\n--- Best Distance: {dist_str} | Docs used: {used}/{len(result['docs'])} ---")
            print(f"\n--- Retrieved Documents ({len(result['docs'])}) ---")
            for i, doc in enumerate(result['docs'], 1):
                meta = doc['metadata']
                source = meta.get('source_type', 'unknown')
                doc_id = doc.get('id', 'unknown')
                dist_str = f"dist: {doc['distance']:.3f}"
                text_len = len(doc['text'])
                print(f"\n[{i}] {source} | {doc_id} | {dist_str} | {text_len} chars")
                print(f"    {doc['text'][:300]}...")
            print(f"\n--- Response (streaming) ---\n")
            # Response already printed by TextStreamer during generation
            print()  # newline after stream
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()