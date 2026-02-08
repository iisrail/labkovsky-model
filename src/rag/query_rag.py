#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline using Vikhr-YandexGPT

Flow:
1. Retrieve relevant chunks from ChromaDB (grouped by qa_id)
2. Derive RS (Response Signal) from best matching chunk
3. Generate response using Vikhr's RAG format (documents role)

RS guides response composition:
- EXPLANATION: explain why this happens
- INTERVENTION: give direct action advice
- ESCALATION: recommend professional help

Usage:
    python query_rag.py
"""

import argparse
import json
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
LLM_MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"  # Russian-native, RAG-trained
LORA_PATH = MODELS_DIR / "labkovsky-vikhr-yandex-lora"

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

        # Load tokenizer first - from LoRA dir if using LoRA (has added tokens)
        if use_lora and lora_path and lora_path.exists():
            print(f"   Loading tokenizer from LoRA: {lora_path}")
            _tokenizer = AutoTokenizer.from_pretrained(str(lora_path), trust_remote_code=True)
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

        if use_lora and lora_path and lora_path.exists():
            # Resize embeddings to match LoRA tokenizer vocab
            print(f"   Resizing embeddings: {base_model.get_input_embeddings().weight.shape[0]} -> {len(_tokenizer)}")
            base_model.resize_token_embeddings(len(_tokenizer))
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

def retrieve_grouped(query: str) -> list:
    """
    Retrieve all chunks from the same Q&A pair as best match.
    Orders chunks: EXPLANATION → INTERVENTION → ESCALATION (each optional)
    """
    embed_model = init_embedding()
    collection = _collection

    query_embedding = embed_model.encode(f"query: {query}")

    # Step 1: Find best matching chunk
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1,
        include=["metadatas", "distances"]
    )

    if not results['ids'][0]:
        return []

    best_id = results['ids'][0][0]  # e.g., "srJvn19GKNA_01_expl"
    best_distance = results['distances'][0][0]

    # Step 2: Extract qa_id and get all chunks with same qa_id
    qa_id = best_id.rsplit("_", 1)[0]  # "srJvn19GKNA_01"

    group = collection.get(
        where={"qa_id": qa_id},
        include=["documents", "metadatas"]
    )

    # Step 3: Organize by RS type
    chunks_by_rs = {}
    for doc_id, doc, meta in zip(group["ids"], group["documents"], group["metadatas"]):
        suffix = doc_id.rsplit("_", 1)[-1]  # "expl", "interv", "escal"
        chunks_by_rs[suffix] = {
            "text": doc,
            "metadata": meta,
            "distance": best_distance if doc_id == best_id else None
        }

    # Step 4: Return in order: EXPLANATION → INTERVENTION → ESCALATION
    documents = []
    for suffix in ["expl", "interv", "escal"]:
        if suffix in chunks_by_rs:
            documents.append(chunks_by_rs[suffix])

    return documents

# ============================================================
# GENERATION
# ============================================================

# Base system prompt (Vikhr recommendation for grounded RAG)
# Style comes from LoRA fine-tuning
BASE_SYSTEM_PROMPT = """Answer the user's question using only the information from the provided documents. Answer in Russian. Do not add information not present in documents."""

# RS-specific composition instructions
RS_INSTRUCTIONS = {
    "EXPLANATION": "Focus on explaining WHY this happens. Help the user understand the cause. Don't push or command.",
    "INTERVENTION": "Give a direct instruction: what to do or stop doing. Be concrete and actionable.",
    "ESCALATION": "Recommend professional help. Don't try to solve this yourself, explain why a specialist is needed.",
}


def generate(query: str, context_docs: list) -> str:
    """
    Generate response using Vikhr's RAG format with documents role.
    RS instruction is derived from the retrieved chunks.

    Args:
        query: User question
        context_docs: List of dicts with 'text' and 'metadata' keys
    """
    llm, tokenizer = _llm, _tokenizer

    # Format documents for Vikhr's documents role
    documents = []
    for i, doc in enumerate(context_docs):
        rs_type = doc['metadata'].get('rs', f'Document {i}')
        documents.append({
            "doc_id": i,
            "title": rs_type,
            "content": doc['text']
        })

    # Determine RS from the best matching chunk (one with distance set)
    rs = None
    for doc in context_docs:
        if doc.get("distance") is not None:
            rs = doc['metadata'].get('rs')
            break

    # Build system prompt with RS instruction if available
    if rs and rs in RS_INSTRUCTIONS:
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nComposition: {RS_INSTRUCTIONS[rs]}"
    else:
        system_prompt = BASE_SYSTEM_PROMPT

    # Vikhr RAG format: system + documents role + user
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "documents", "content": json.dumps(documents, ensure_ascii=False)},
        {"role": "user", "content": query},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(llm.device)

    input_len = inputs.shape[-1]

    with torch.inference_mode():
        outputs = llm.generate(
            inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.3,  # Vikhr recommends 0.1-0.5
            top_k=40,         # Vikhr recommends 30-50
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response

# ============================================================
# MAIN FUNCTION
# ============================================================

def ask_labkovsky(query: str) -> dict:
    # Step 1: Retrieve all chunks for best matching Q&A pair
    docs = retrieve_grouped(query)

    # Step 2: Get RS and distance from best matching chunk
    best_distance = None
    rs = None
    for doc in docs:
        if doc.get("distance") is not None:
            best_distance = doc["distance"]
            rs = doc['metadata'].get('rs')
            break

    # Step 3: Generate (RS instruction derived from retrieved chunks)
    answer = generate(query, docs)

    return {
        "answer": answer,
        "docs": docs,
        "rs": rs,
        "rag_distance": best_distance,
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
                dist = result['rag_distance']
                rs = result['rs'] or "unknown"
                print(f"\n--- RAG Distance: {dist:.3f} | RS: {rs} ---")
                print(f"\n--- Retrieved Chunks ({len(result['docs'])}) ---")
                for i, doc in enumerate(result['docs'], 1):
                    meta = doc['metadata']
                    qa_id = meta.get('qa_id', 'unknown')
                    doc_rs = meta.get('rs', 'unknown')
                    print(f"\n[{i}] QA: {qa_id} | RS: {doc_rs}")
                    print(f"    {doc['text'][:300]}...")
                print(f"\n--- Response ---")
            print(f"\nLabkovsky: {result['answer']}\n")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()