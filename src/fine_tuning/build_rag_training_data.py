#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build training data with RAG context.

For each QA pair, retrieve top-K docs from ChromaDB (excluding self by ID),
then format with docs in system prompt — matching inference format.

Retrieval prefers docs with matching decision_type, falls back to cosine-only.
QA docs include topic hint from decision_type instead of original question.

Usage:
    source venv_wsl/bin/activate
    python src/fine_tuning/build_rag_training_data.py
"""

import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"

INPUT_FILES = [
    FINE_TUNING_DIR / "qa_clean.jsonl",
    FINE_TUNING_DIR / "anti_generic_finance.jsonl",
]
OUTPUT_FILE = FINE_TUNING_DIR / "qa_with_rag_context.jsonl"

# decision_type lookup from segmented QA
DT_LOOKUP_FILE = FINE_TUNING_DIR / "qa_rs_segmented.jsonl"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TOP_K = 2
RETRIEVE_K = 5  # retrieve extra to have enough after filtering
MAX_DISTANCE = 0.35  # skip docs further than this
ADD_QA_DOC = True  # add closest non-self QA as extra doc
QA_MAX_DISTANCE = 0.30  # stricter for QA to avoid noise
MAX_DOCS_TOKENS = 1800  # token budget for docs only (~1.5 chars/token for Russian)
MAX_DOCS_CHARS = int(MAX_DOCS_TOKENS * 1.5)  # ~2700 chars for docs portion

# decision_type → short topic label
DT_TOPIC = {
    "SELF_ESTEEM_CORRECTIVE": "self-esteem",
    "EXPLANATION": "explanation",
    "DEPENDENCY_BOUNDARIES": "boundaries",
    "ANXIETY_MANAGEMENT": "anxiety",
    "ADDICTION_PATTERN": "addiction",
    "CLINICAL_ESCALATION": "escalation",
    "AFFECTIVE_ADDICTION": "affective addiction",
    "PARENTING_MODEL": "parenting",
    "FEAR_SCENARIO_COPING": "fear coping",
    "PARENTING_LIMITS": "parenting limits",
}

# Must match query_rag.py inference format
SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reference materials.\n\n"
    "Book and article fragments contain core principles — use them as the foundation "
    "for your reasoning, but do not copy verbatim.\n"
    "QA examples show how to structure a response — use them as a guide for tone and format.\n\n"
    "{docs}\n\n"
    "Answer in Labkovsky's style: direct, confident, with specific recommendations. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_dt_lookup(path: Path) -> dict:
    """Load decision_type lookup from qa_rs_segmented.jsonl."""
    lookup = {}
    if not path.exists():
        return lookup
    for rec in iter_jsonl(path):
        rec_id = rec.get("id", "")
        dt = rec.get("decision_type", "")
        if rec_id and dt:
            lookup[rec_id] = dt
    return lookup


def expand_with_siblings(collection, doc_id: str, doc_text: str, meta: dict) -> str:
    """Expand doc with up to 2 next chunks within same chapter/article."""
    chunk_id = meta.get("chunk_id")
    if chunk_id is None:
        return doc_text

    try:
        chunk_num = int(chunk_id)
        group_id = meta.get("chapter_id") or meta.get("article_id")

        if "_chunk" in doc_id:
            base = doc_id.rsplit("_chunk", 1)[0]
        else:
            base = doc_id.rsplit("_", 1)[0] if "_" in doc_id else doc_id

        for offset in range(1, 3):
            if "_chunk" in doc_id:
                nid = f"{base}_chunk{chunk_num + offset}"
            else:
                nid = f"{base}_{chunk_num + offset}"
            next_doc = collection.get(ids=[nid], include=["documents", "metadatas"])
            if not next_doc["ids"]:
                break
            next_group = (next_doc["metadatas"][0].get("chapter_id")
                          or next_doc["metadatas"][0].get("article_id"))
            if next_group != group_id:
                break
            doc_text = doc_text + "\n" + next_doc["documents"][0]
    except (ValueError, Exception):
        pass

    return doc_text


def retrieve_book_article_docs(collection, query_embedding, rec_dt: str, top_k: int = TOP_K) -> list:
    """
    Retrieve book/article docs, preferring same decision_type.
    1. Try with decision_type filter first
    2. Fill remaining slots with cosine-only fallback
    """
    docs = []

    # Pass 1: same decision_type
    if rec_dt:
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=RETRIEVE_K,
                where={"$and": [
                    {"source_type": {"$in": ["articles", "book"]}},
                    {"decision_type": {"$eq": rec_dt}},
                ]},
                include=["documents", "metadatas", "distances"]
            )
            for doc_id, doc_text, meta, dist in zip(
                results['ids'][0], results['documents'][0],
                results['metadatas'][0], results['distances'][0]
            ):
                if dist > MAX_DISTANCE:
                    continue
                doc_text = expand_with_siblings(collection, doc_id, doc_text, meta)
                docs.append({
                    "id": doc_id, "text": doc_text,
                    "distance": dist, "type": "book_article",
                })
                if len(docs) >= top_k:
                    break
        except Exception:
            pass  # fall through to cosine-only

    # Pass 2: cosine-only fallback for remaining slots
    if len(docs) < top_k:
        seen_ids = {d["id"] for d in docs}
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=RETRIEVE_K,
            where={"source_type": {"$in": ["articles", "book"]}},
            include=["documents", "metadatas", "distances"]
        )
        for doc_id, doc_text, meta, dist in zip(
            results['ids'][0], results['documents'][0],
            results['metadatas'][0], results['distances'][0]
        ):
            if doc_id in seen_ids:
                continue
            if dist > MAX_DISTANCE:
                continue
            doc_text = expand_with_siblings(collection, doc_id, doc_text, meta)
            docs.append({
                "id": doc_id, "text": doc_text,
                "distance": dist, "type": "book_article",
            })
            if len(docs) >= top_k:
                break

    return docs


def retrieve_qa_doc(collection, query_embedding, rec_id: str, rec_dt: str) -> list:
    """
    Retrieve closest non-self QA doc, preferring same decision_type.
    Adds topic hint from decision_type metadata.
    """
    rec_qa_id = rec_id.rsplit("_", 1)[0] if "_" in rec_id else rec_id

    def _find_non_self_qa(results) -> dict | None:
        for qa_id, qa_text, qa_meta, qa_dist in zip(
            results['ids'][0], results['documents'][0],
            results['metadatas'][0], results['distances'][0]
        ):
            qa_qa_id = qa_meta.get("qa_id", qa_id)
            if qa_qa_id == rec_qa_id or qa_id == rec_id:
                continue
            if qa_dist > QA_MAX_DISTANCE:
                return None
            # Add topic hint from decision_type
            qa_dt = qa_meta.get("decision_type", "")
            topic = DT_TOPIC.get(qa_dt, "")
            if topic:
                qa_text = f"Topic: {topic}\n{qa_text}"
            return {
                "id": qa_id, "text": qa_text,
                "distance": qa_dist, "type": "qa",
            }
        return None

    # Pass 1: same decision_type
    if rec_dt:
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5,
                where={"$and": [
                    {"source_type": {"$eq": "qa_corpus"}},
                    {"decision_type": {"$eq": rec_dt}},
                ]},
                include=["documents", "metadatas", "distances"]
            )
            doc = _find_non_self_qa(results)
            if doc:
                return [doc]
        except Exception:
            pass

    # Pass 2: cosine-only fallback
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        where={"source_type": {"$eq": "qa_corpus"}},
        include=["documents", "metadatas", "distances"]
    )
    doc = _find_non_self_qa(results)
    if doc:
        return [doc]

    return []


def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("labkovsky")
    print(f"  {collection.count()} documents in index")

    # Load decision_type lookup
    dt_lookup = load_dt_lookup(DT_LOOKUP_FILE)
    print(f"  decision_type lookup: {len(dt_lookup)} records")

    # Load QA data from all input files
    records = []
    for input_file in INPUT_FILES:
        if not input_file.exists():
            print(f"  WARNING: {input_file.name} not found, skipping")
            continue
        file_records = list(iter_jsonl(input_file))
        print(f"  {input_file.name}: {len(file_records)} records")
        records.extend(file_records)
    print(f"Loaded {len(records)} total QA records")

    output_records = []
    no_docs_count = 0
    dt_match_count = 0

    for i, rec in enumerate(records):
        rec_id = rec.get("id", "")
        question = rec.get("question", "")
        answer = rec.get("answer", "")

        if not question or not answer:
            continue

        rec_dt = dt_lookup.get(rec_id, "")
        query_embedding = embed_model.encode(f"query: {question}")

        # Retrieve book/article docs (prefer same decision_type)
        docs = retrieve_book_article_docs(collection, query_embedding, rec_dt)

        # Retrieve QA doc (prefer same decision_type, topic hint instead of question)
        if ADD_QA_DOC:
            qa_docs = retrieve_qa_doc(collection, query_embedding, rec_id, rec_dt)
            docs.extend(qa_docs)

        if not docs:
            no_docs_count += 1

        # Track dt matches
        if rec_dt and any(d.get("type") == "book_article" for d in docs):
            dt_match_count += 1

        # Truncate docs to fit within token budget
        doc_parts = []
        total_chars = 0
        for j, d in enumerate(docs):
            label = "[Book/Article]" if d["type"] == "book_article" else "[QA Example]"
            part = f"{label} Doc {j+1}: {d['text']}"
            if total_chars + len(part) > MAX_DOCS_CHARS:
                remaining = MAX_DOCS_CHARS - total_chars
                if remaining > 200:  # only add if meaningful amount left
                    part = part[:remaining] + "..."
                    doc_parts.append(part)
                break
            doc_parts.append(part)
            total_chars += len(part)
        docs_text = "\n\n".join(doc_parts)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)

        output_records.append({
            "id": rec_id,
            "question": question,
            "answer": answer,
            "system_prompt": system_prompt,
            "num_docs": len(docs),
            "doc_ids": [d["id"] for d in docs],
            "decision_type": rec_dt,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(output_records)} records -> {OUTPUT_FILE.name}")
    print(f"  Records with no docs: {no_docs_count}")
    print(f"  Records with decision_type match: {dt_match_count}")

    # Show example
    if output_records:
        ex = output_records[0]
        print(f"\n--- Example ---")
        print(f"ID: {ex['id']}")
        print(f"DT: {ex['decision_type']}")
        print(f"Q: {ex['question'][:100]}...")
        print(f"Docs: {ex['num_docs']} ({ex['doc_ids']})")
        print(f"System prompt (first 500): {ex['system_prompt'][:500]}...")
        print(f"A: {ex['answer'][:200]}...")


if __name__ == "__main__":
    main()
