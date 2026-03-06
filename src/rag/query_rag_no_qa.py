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
from datetime import datetime
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
EVAL_LOG_PATH = PROJECT_DIR / "data" / "eval_log.jsonl"

# System prompt template - docs will be inserted
SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reference materials.\n\n"
    "Book and article fragments contain core principles — use them as the foundation "
    "for your reasoning, but do not copy verbatim.\n\n"
    "{docs}\n\n"
    "Answer in Labkovsky's style: direct, confident, with specific recommendations. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
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

def _resolve_best_checkpoint(lora_path: Path) -> Path:
    """Find best checkpoint by reading trainer_state.json, fall back to lora_path itself."""
    # If already pointing to a checkpoint dir, use as-is
    if lora_path.name.startswith("checkpoint-"):
        return lora_path

    # Look for trainer_state.json in checkpoint dirs
    checkpoints = sorted(lora_path.glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        return lora_path

    # Read the latest trainer_state (has full history and best_model_checkpoint)
    try:
        state = json.loads(checkpoints[-1].read_text())
        best_path = state.get("best_model_checkpoint")
        if best_path:
            best_path = Path(best_path)
            # Handle absolute paths that may differ between machines
            if not best_path.exists():
                best_path = lora_path / best_path.name
            if best_path.exists():
                best_metric = state.get("best_metric", "?")
                print(f"   Best checkpoint: {best_path.name} (eval_loss={best_metric})")
                return best_path
    except (json.JSONDecodeError, KeyError):
        pass

    return lora_path


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
            lora_path = _resolve_best_checkpoint(lora_path)
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

TOP_K = 2  # 2 book/article docs (no QA doc in context — prevents verbatim copy)
EXPAND_CHUNKS = True  # concatenate next chunks from same chapter/article
MAX_DOCS_CHARS = 3000  # budget for all docs (~2000 tokens Russian)

# decision_type → short topic label (must match build_rag_training_data.py)
DT_TOPIC = {
    "SELF_ESTEEM_CORRECTIVE": "self-esteem",
    "DEPENDENCY_BOUNDARIES": "boundaries",
    "ANXIETY_MANAGEMENT": "anxiety",
    "ADDICTION_PATTERN": "addiction",
    "CLINICAL_ESCALATION": "escalation",
    "AFFECTIVE_ADDICTION": "affective addiction",
    "PARENTING_MODEL": "parenting",
    "FEAR_SCENARIO_COPING": "fear coping",
}


def _make_chunk_id(doc_id: str, num: int) -> str:
    """Build chunk ID from base doc_id and chunk number."""
    if "_chunk" in doc_id:
        base = doc_id.rsplit("_chunk", 1)[0]
        return f"{base}_chunk{num}"
    else:
        base = doc_id.rsplit("_", 1)[0] if "_" in doc_id else doc_id
        return f"{base}_{num}"


def _find_chapter_end(collection, doc_id: str, chunk_num: int, group_id: str) -> int:
    """Scan forward to find the last chunk number in this chapter."""
    last_num = chunk_num
    for offset in range(1, 30):
        nid = _make_chunk_id(doc_id, chunk_num + offset)
        next_doc = collection.get(ids=[nid], include=["metadatas"])
        if not next_doc["ids"]:
            break
        next_group = (next_doc["metadatas"][0].get("chapter_id")
                      or next_doc["metadatas"][0].get("article_id"))
        if next_group != group_id:
            break
        last_num = chunk_num + offset
    return last_num


def _expand_with_next_chunks(collection, doc_id: str, doc_text: str, metadata: dict) -> str:
    """
    Expand doc with sibling chunks from same chapter/article.
    - Book: jump to pre-last + last chunk of chapter (has principle/solution)
    - Articles: concatenate up to 2 next siblings
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
    is_book = metadata.get("source_type") == "book"

    if is_book:
        # Book: jump to pre-last + last chunk of chapter (principle/solution)
        last_num = _find_chapter_end(collection, doc_id, chunk_num, group_id)

        if last_num > chunk_num + 1:
            pre_last = last_num - 1
            texts = []
            for num in [pre_last, last_num]:
                nid = _make_chunk_id(doc_id, num)
                chunk_doc = collection.get(ids=[nid], include=["documents"])
                if chunk_doc["ids"]:
                    texts.append(chunk_doc["documents"][0])
            if texts:
                doc_text = "\n".join(texts)
        elif last_num == chunk_num + 1:
            nid = _make_chunk_id(doc_id, last_num)
            chunk_doc = collection.get(ids=[nid], include=["documents"])
            if chunk_doc["ids"]:
                doc_text = doc_text + "\n" + chunk_doc["documents"][0]
    else:
        # Articles: concatenate up to 2 next siblings
        for offset in range(1, 3):
            nid = _make_chunk_id(doc_id, chunk_num + offset)
            try:
                next_doc = collection.get(ids=[nid], include=["documents", "metadatas"])
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


def retrieve(query: str, chroma_dir: Path = CHROMA_DIR, expand_chunks: bool = EXPAND_CHUNKS) -> list:
    """
    Retrieve documents matching training format:
    1. One closest QA doc (to determine decision_type)
    2. One best book/article doc (dt-filtered, then cosine fallback)

    Returns list of dicts with 'text', 'metadata', 'distance' keys.
    """
    embed_model = init_embedding()
    collection = init_retrieval(chroma_dir)

    query_embedding = embed_model.encode(f"query: {query}")

    documents = []

    # Step 1: Retrieve QA for decision_type lookup only (not included in context)
    qa_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1,
        where={"source_type": {"$eq": "qa_corpus"}},
        include=["metadatas", "distances"]
    )

    qa_dt = ""
    if qa_results['ids'][0]:
        qa_meta = qa_results['metadatas'][0][0]
        qa_dt = qa_meta.get("decision_type", "")

    # Step 2: Retrieve 2 book/article docs — dt filter first, cosine fallback
    # No QA in context — forces model to synthesize from principles
    seen_ids = set()
    seen_groups = set()

    def _add_book_article_docs(results):
        for doc_id, doc_text, meta, dist in zip(
            results['ids'][0], results['documents'][0],
            results['metadatas'][0], results['distances'][0]
        ):
            if doc_id in seen_ids:
                continue
            # Skip docs from same chapter/article group (avoid redundancy)
            group = meta.get("chapter_id") or meta.get("article_id") or doc_id
            if group in seen_groups:
                continue
            if expand_chunks:
                doc_text = _expand_with_next_chunks(collection, doc_id, doc_text, meta)
            seen_ids.add(doc_id)
            seen_groups.add(group)
            documents.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "distance": dist,
            })
            if len(documents) >= TOP_K:
                break

    if qa_dt:
        # Try with dt filter
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5,
                where={"$and": [
                    {"source_type": {"$in": ["articles", "book"]}},
                    {"decision_type": {"$eq": qa_dt}},
                ]},
                include=["documents", "metadatas", "distances"]
            )
            _add_book_article_docs(results)
        except Exception:
            pass

    if len(documents) < TOP_K:
        # Cosine fallback (no dt filter)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,
            where={"source_type": {"$in": ["articles", "book"]}},
            include=["documents", "metadatas", "distances"]
        )
        _add_book_article_docs(results)

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

    # Format docs: 2 book/article docs, no QA (prevents verbatim copy)
    # When too long, truncate from BEGINNING (keep later chunks with principles)
    doc_parts = []
    budget_per_doc = MAX_DOCS_CHARS // max(len(context_docs), 1)
    for i, d in enumerate(context_docs):
        label = f"[Book/Article] Doc {i+1}: "
        text = d['text']
        available = budget_per_doc - len(label)
        if len(text) > available:
            cut = text[-(available - 3):]
            # Snap to next word boundary (don't cut mid-word)
            space = cut.find(" ")
            if space > 0 and space < 50:
                cut = cut[space + 1:]
            text = "..." + cut
        doc_parts.append(label + text)

    docs_text = "\n\n".join(doc_parts)

    # Put docs in system prompt
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

    # Stop generation at EOS or start of a new user turn
    stop_token_ids = tokenizer.encode("<s>user\n", add_special_tokens=False)
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnUserTurn(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if len(input_ids[0]) >= len(stop_token_ids):
                return input_ids[0][-len(stop_token_ids):].tolist() == stop_token_ids
            return False

    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=380,  # ~250 words in Russian
            do_sample=True,
            temperature=0.5,
            top_p=0.85,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnUserTurn()]),
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

    docs.sort(key=lambda d: d["distance"])
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

def log_eval(model_path: str, query: str, answer: str, docs: list, notes: str):
    """Append evaluation entry to eval log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "question": query,
        "answer": answer,
        "docs_used": len(docs),
        "best_distance": docs[0]["distance"] if docs else None,
        "doc_ids": [d["id"] for d in docs],
        "notes": notes,
    }
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[saved to {EVAL_LOG_PATH.name}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-dir", type=str, default=str(CHROMA_DIR), help="ChromaDB directory")
    parser.add_argument("--lora-path", type=str, default=str(LORA_PATH), help="LoRA adapter path")
    parser.add_argument("--no-lora", action="store_true", help="Use base model without LoRA")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved RAG documents")
    parser.add_argument("--no-log", action="store_true", help="Disable eval logging")
    args = parser.parse_args()

    print("=" * 60)
    print("Labkovsky RAG")
    print("=" * 60)
    print("Type 'exit' to quit, 'verbose' to toggle RAG docs display")
    if not args.no_log:
        print("After each answer, enter notes (or press Enter to skip)")
    print()

    verbose = args.verbose

    # Initialize all components
    print("Loading models...")
    init_embedding()
    init_retrieval(Path(args.chroma_dir))

    lora_path = None if args.no_lora else Path(args.lora_path)
    resolved_lora = _resolve_best_checkpoint(lora_path) if lora_path else None
    init_llm(use_lora=not args.no_lora, lora_path=lora_path)

    if args.no_lora:
        model_label = "base-no-lora"
    else:
        try:
            model_label = str(resolved_lora.relative_to(PROJECT_DIR))
        except ValueError:
            model_label = str(resolved_lora)
    print(f"\nModel: {model_label}")
    print("[OK] Ready!\n")

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

            dist = result['best_distance']
            used = result['docs_used']
            dist_str = f"{dist:.3f}" if dist is not None else "N/A"
            print(f"\n--- Best Distance: {dist_str} | Docs used: {used}/{len(result['docs'])} ---")
            if verbose:
                print(f"\n--- Retrieved Documents ({len(result['docs'])}) ---")
                for i, doc in enumerate(result['docs'], 1):
                    meta = doc['metadata']
                    source = meta.get('source_type', 'unknown')
                    doc_id = doc.get('id', 'unknown')
                    dt = meta.get('decision_type', '')
                    dist_str = f"dist: {doc['distance']:.3f}"
                    # Clean up text: collapse multiple newlines, trim
                    import re as _re
                    clean_text = _re.sub(r'\n{2,}', '\n', doc['text']).strip()
                    text_len = len(doc['text'])
                    print(f"\n[{i}] {source} | {doc_id} | {dist_str} | {text_len} chars | dt: {dt}")
                    print(f"    {clean_text[:400]}")
            print(f"\nЛабковский: {result['answer']}\n")

            # Eval logging
            if not args.no_log:
                try:
                    notes = input("Notes (Enter to skip): ").strip()
                except (KeyboardInterrupt, EOFError):
                    notes = ""
                if notes:
                    log_eval(model_label, query, result["answer"], result["docs"], notes)

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()