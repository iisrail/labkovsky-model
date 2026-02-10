#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline for RAFT-trained pretrain model.

Uses the same format as RAFT training:
Документы:
{retrieved_document}

Вопрос: {question}
Ответ:

Usage:
    python query_rag_for_pretrain.py
    python query_rag_for_pretrain.py --lora-path models/labkovsky-raft-lora
"""

import argparse
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import chromadb

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
MODELS_DIR = PROJECT_DIR / "models"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-pretrain"
LORA_PATH = MODELS_DIR / "labkovsky-raft-lora"

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
        print(f"    Loaded {_collection.count()} documents")
    return _collection


def init_llm(use_lora: bool = True, lora_path: Path = None):
    global _llm, _tokenizer
    if _llm is None:
        print(f"[+] Loading LLM: {LLM_MODEL_NAME}")

        if use_lora and lora_path and lora_path.exists():
            print(f"    Loading tokenizer from LoRA: {lora_path}")
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
            print(f"    Resizing embeddings: {base_model.get_input_embeddings().weight.shape[0]} -> {len(_tokenizer)}")
            base_model.resize_token_embeddings(len(_tokenizer))
            print(f"    Loading LoRA: {lora_path}")
            _llm = PeftModel.from_pretrained(base_model, str(lora_path))
        else:
            print("    (base model, no LoRA)")
            _llm = base_model

        _llm.eval()
        print(f"[OK] LLM loaded ({'with RAFT LoRA' if use_lora and lora_path and lora_path.exists() else 'base only'})")

    return _llm, _tokenizer


# ============================================================
# RETRIEVAL
# ============================================================

TOP_K = 3
GOOD_MATCH_THRESHOLD = 0.5  # Below this = good match, use document
NO_DOC_THRESHOLD = 1.2      # Above this = poor match, don't use document


def retrieve(query: str, chroma_dir: Path = CHROMA_DIR, top_k: int = TOP_K) -> list:
    """Retrieve top-k relevant documents from ChromaDB."""
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
            "distance": dist
        })

    return documents


# ============================================================
# GENERATION - RAFT FORMAT
# ============================================================

def generate(query: str, context_docs: list, use_doc: bool = True, use_lora: bool = True, lora_path: Path = LORA_PATH) -> str:
    """
    Generate response using RAFT format with RS (Response Signal) structure.

    Uses all retrieved documents grouped by chunk_role:
    - EXPLANATION: Psychology behind the issue
    - INTERVENTION: Practical advice
    - ESCALATION: When to seek professional help

    Format:
    Документы:

    [Объяснение]
    {explanation_doc}

    [Рекомендация]
    {intervention_doc}

    [Важно]
    {escalation_doc}

    Вопрос: {question}
    Ответ: {composed_answer}
    """
    llm, tokenizer = init_llm(use_lora=use_lora, lora_path=lora_path)

    if context_docs and use_doc:
        # Order: EXPLANATION → INTERVENTION → ESCALATION (Labkovsky's style)
        rs_order = ["EXPLANATION", "INTERVENTION", "ESCALATION"]
        rs_labels = {
            "EXPLANATION": "Объяснение",
            "INTERVENTION": "Рекомендация",
            "ESCALATION": "Важно"
        }

        # Sort documents by chunk_role order
        def get_order(doc):
            role = doc.get('metadata', {}).get('chunk_role', '')
            return rs_order.index(role) if role in rs_order else 99

        sorted_docs = sorted(context_docs, key=get_order)

        # Build structured document section in correct order
        doc_sections = []
        for doc in sorted_docs:
            chunk_role = doc.get('metadata', {}).get('chunk_role', '')
            label = rs_labels.get(chunk_role, '')

            if label:
                doc_sections.append(f"[{label}]\n{doc['text']}")
            else:
                doc_sections.append(doc['text'])

        docs_text = "\n\n".join(doc_sections)
        prompt = f"Документы:\n\n{docs_text}\n\nВопрос: {query}\nОтвет:"
    else:
        prompt = f"Вопрос: {query}\nОтвет:"

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response


# ============================================================
# MAIN FUNCTION
# ============================================================

def ask_labkovsky(query: str, structured: bool = True) -> dict:
    """Query the RAFT RAG pipeline and generate a response.

    Args:
        query: User question
        structured: If True, generate separate responses per chunk_role and combine
    """
    docs = retrieve(query)

    if not docs:
        answer = generate(query, [], use_doc=False)
        return {
            "answer": answer,
            "docs": [],
            "best_distance": None,
            "docs_used": 0,
        }

    best_distance = docs[0]["distance"]
    use_doc = best_distance < NO_DOC_THRESHOLD

    if not use_doc:
        answer = generate(query, [], use_doc=False)
        return {
            "answer": answer,
            "docs": docs,
            "best_distance": best_distance,
            "docs_used": 0,
        }

    if structured:
        # Generate structured response: one answer per chunk_role
        rs_order = ["EXPLANATION", "INTERVENTION", "ESCALATION"]
        rs_labels = {
            "EXPLANATION": "Объяснение",
            "INTERVENTION": "Рекомендация",
            "ESCALATION": "Важно"
        }

        # Group docs by chunk_role
        docs_by_role = {}
        for doc in docs:
            role = doc.get('metadata', {}).get('chunk_role', 'OTHER')
            if role not in docs_by_role:
                docs_by_role[role] = doc

        # Generate answer for each role in order
        answer_parts = []
        for role in rs_order:
            if role in docs_by_role:
                doc = docs_by_role[role]
                part_answer = generate(query, [doc], use_doc=True)
                label = rs_labels.get(role, role)
                answer_parts.append(f"**{label}:** {part_answer}")

        answer = "\n\n".join(answer_parts) if answer_parts else generate(query, docs, use_doc=True)
    else:
        # Single combined response
        answer = generate(query, docs, use_doc=True)

    return {
        "answer": answer,
        "docs": docs,
        "best_distance": best_distance,
        "docs_used": len(docs),
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-dir", type=str, default=str(CHROMA_DIR))
    parser.add_argument("--lora-path", type=str, default=str(LORA_PATH))
    parser.add_argument("--no-lora", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Labkovsky RAFT RAG (Pretrain Model)")
    print("=" * 60)
    print("\nCommands:")
    print("  'exit' — quit")
    print("  'verbose' — toggle source display")
    print()

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
                print(f"\n--- Best Distance: {dist:.3f} | Docs used: {used}/{len(result['docs'])} ---")
                print(f"\n--- Retrieved Documents ({len(result['docs'])}) ---")
                for i, doc in enumerate(result['docs'], 1):
                    meta = doc['metadata']
                    source = meta.get('source_type', 'unknown')
                    doc_id = doc.get('id', 'unknown')
                    print(f"\n[{i}] {source} | {doc_id} | dist: {doc['distance']:.3f}")
                    print(f"    {doc['text'][:300]}...")
                print(f"\n--- Response ---")
            print(f"\nLabkovsky: {result['answer']}\n")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
