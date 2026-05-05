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

def trim_at_section_break(text: str) -> str:
    """Trim text at first section break (3+ consecutive newlines)."""
    match = re.search(r'\n{3,}', text)
    if match:
        return text[:match.start()].strip()
    return text


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
TOP_K = 1              # 1 book/article doc (best by distance, no source preference)
RETRIEVE_K = 5         # retrieve extra to have enough after filtering
MAX_DISTANCE = 0.35    # skip docs further than this
ADD_QA_DOC = True      # add closest non-self QA as extra doc
QA_MAX_DISTANCE = 0.35 # same as book/article threshold
MAX_DOCS_TOKENS = 2000 # token budget for docs only (~1.5 chars/token for Russian)
MAX_DOCS_CHARS = int(MAX_DOCS_TOKENS * 1.5)  # ~3000 chars for docs portion

# decision_type → short topic label
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

# Must match query_rag.py inference format
SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reference materials.\n\n"
    "Book and article fragments contain core principles — use them as the foundation "
    "for your reasoning, but do not copy verbatim.\n"
    "QA examples show how to structure a response — use them as a guide for tone and format.\n\n"
    "{docs}\n\n"
    "Answer in Labkovsky's style: blunt, confident, with tough love if needed. "
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
    for offset in range(1, 30):  # scan up to 30 chunks forward
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


def expand_with_siblings(collection, doc_id: str, doc_text: str, meta: dict, source_type: str = "") -> str:
    """
    Expand doc with sibling chunks from same chapter/article.
    - Book: jump to pre-last + last chunk of chapter (has principle/solution)
    - Articles: concatenate up to 2 next siblings
    Truncation (prefer later chunks) happens in formatting stage.
    """
    chunk_id = meta.get("chunk_id")
    if chunk_id is None:
        return doc_text

    try:
        chunk_num = int(chunk_id)
        group_id = meta.get("chapter_id") or meta.get("article_id")
        is_book = source_type == "book"

        if is_book:
            # Book: jump to last chunk of chapter (has principle/conclusion)
            last_num = _find_chapter_end(collection, doc_id, chunk_num, group_id)

            if last_num > chunk_num:
                # Replace with last chunk of chapter
                nid = _make_chunk_id(doc_id, last_num)
                chunk_doc = collection.get(ids=[nid], include=["documents"])
                if chunk_doc["ids"]:
                    doc_text = chunk_doc["documents"][0]
            # else: matched chunk IS the last — keep as is
        else:
            # Articles: concatenate up to 2 next siblings
            for offset in range(1, 3):
                nid = _make_chunk_id(doc_id, chunk_num + offset)
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

    # Trim at section breaks (3+ newlines)
    doc_text = trim_at_section_break(doc_text)

    return doc_text


EXPANSION_RANGE = 2  # how many siblings expansion can touch


def _chunk_key(meta: dict) -> tuple:
    """Return (group_id, chunk_num) for proximity checks, or None if not a chunk."""
    group = meta.get("article_id") or meta.get("chapter_id") or ""
    chunk_id = meta.get("chunk_id")
    if not group or chunk_id is None:
        return None
    try:
        return (group, int(chunk_id))
    except (ValueError, TypeError):
        return None


def _overlaps_existing(meta: dict, seen_chunks: list) -> bool:
    """Check if this chunk overlaps with any already-selected chunk after expansion.
    Two chunks from the same group overlap if they're within EXPANSION_RANGE of each other."""
    key = _chunk_key(meta)
    if key is None:
        return False
    group, chunk_num = key
    for s_group, s_num in seen_chunks:
        if s_group == group and abs(chunk_num - s_num) <= EXPANSION_RANGE:
            return True
    return False


def _query_and_collect(collection, query_embedding, where_filter, docs, seen_ids, seen_chunks, top_k, expand=True) -> list:
    """Helper: query ChromaDB, expand chunks, collect into docs list.
    Skips chunks that would overlap after expansion (within EXPANSION_RANGE of already-selected)."""
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=RETRIEVE_K,
        where=where_filter,
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
        if _overlaps_existing(meta, seen_chunks):
            continue
        if expand:
            source_type = meta.get("source_type", "")
            doc_text = expand_with_siblings(collection, doc_id, doc_text, meta, source_type)
        seen_ids.add(doc_id)
        key = _chunk_key(meta)
        if key:
            seen_chunks.append(key)
        docs.append({
            "id": doc_id, "text": doc_text,
            "distance": dist, "type": "book_article",
        })
        if len(docs) >= top_k:
            break
    return docs


def retrieve_book_article_doc(collection, query_embedding, rec_dt: str) -> list:
    """
    Retrieve 1 best book/article doc (no source preference, best by distance).
    DT filter first, cosine fallback for records without dt.
    """
    docs = []
    seen_ids = set()
    seen_chunks = []

    if rec_dt:
        # Same dt, both sources together — best by distance wins
        try:
            _query_and_collect(collection, query_embedding,
                {"$and": [
                    {"source_type": {"$in": ["articles", "book"]}},
                    {"decision_type": {"$eq": rec_dt}},
                ]},
                docs, seen_ids, seen_chunks, top_k=1)
        except Exception:
            pass
    else:
        # No dt available (e.g. anti_generic records) — cosine fallback
        _query_and_collect(collection, query_embedding,
            {"source_type": {"$in": ["articles", "book"]}},
            docs, seen_ids, seen_chunks, top_k=1)

    return docs


def retrieve_qa_doc(collection, query_embedding, rec_id: str, rec_dt: str) -> list:
    """
    Retrieve closest non-self QA doc, preferring same decision_type.
    Topic hint uses the record's dt (describes what the question is about).
    """
    rec_qa_id = rec_id.rsplit("_", 1)[0] if "_" in rec_id else rec_id
    # Topic hint from record's dt, not the retrieved QA's dt
    topic = DT_TOPIC.get(rec_dt, "")

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
            if topic:
                qa_text = f"Topic: {topic}\n{qa_text}"
            return {
                "id": qa_id, "text": qa_text,
                "distance": qa_dist, "type": "qa",
            }
        return None

    # Only retrieve QA with same decision_type — no dt mismatch allowed
    if not rec_dt:
        return []

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

        # Retrieve 1 book/article doc (best by distance, no source preference)
        docs = retrieve_book_article_doc(collection, query_embedding, rec_dt)

        # Retrieve 1 QA doc (same decision_type, topic hint)
        if ADD_QA_DOC:
            qa_docs = retrieve_qa_doc(collection, query_embedding, rec_id, rec_dt)
            docs.extend(qa_docs)

        if not docs:
            no_docs_count += 1

        # Track dt matches
        if rec_dt and any(d.get("type") == "book_article" for d in docs):
            dt_match_count += 1

        # Format docs with budget: 1 book/article (Doc 1) + 1 QA (Doc 2)
        # Reserve space for QA doc first
        book_docs = [d for d in docs if d["type"] == "book_article"]
        qa_docs_list = [d for d in docs if d["type"] == "qa"]

        qa_parts = []
        qa_chars = 0
        for d in qa_docs_list:
            part = f"[QA Example] Doc 2: {d['text']}"
            qa_chars += len(part)
            qa_parts.append(part)

        # Book/article doc gets remaining budget
        # When too long, truncate from BEGINNING (keep later chunks with principles)
        book_budget = MAX_DOCS_CHARS - qa_chars
        doc_parts = []
        for d in book_docs:
            label = "[Book/Article] Doc 1: "
            text = d['text']
            available = book_budget - len(label)
            if len(text) > available:
                cut = text[-(available - 3):]
                # Snap to next word boundary (don't cut mid-word)
                space = cut.find(" ")
                if space > 0 and space < 50:
                    cut = cut[space + 1:]
                text = "..." + cut
            doc_parts.append(label + text)

        # Append QA doc
        doc_parts.extend(qa_parts)

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
