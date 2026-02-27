#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build training data with RAG context.

For each QA pair, retrieve top-K docs from ChromaDB (excluding self by ID),
then format with docs in system prompt — matching inference format.

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

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TOP_K = 2
RETRIEVE_K = 5  # retrieve extra to have enough after filtering
MAX_DISTANCE = 0.35  # skip docs further than this
ADD_QA_DOC = True  # add closest non-self QA as extra doc
QA_MAX_DISTANCE = 0.30  # stricter for QA to avoid noise

# Must match query_rag.py inference format
SYSTEM_PROMPT_TEMPLATE = (
    "Ты психолог Михаил Лабковский. Используй следующие документы для ответа:\n\n"
    "{docs}\n\n"
    "Отвечай в стиле Лабковского: прямо, уверенно, с конкретными рекомендациями. "
    "Сначала объясни причину проблемы, затем дай конкретные шаги для решения. "
    "Если видишь, что кому-то в ситуации нужна профессиональная помощь — скажи об этом прямо."
)

NO_DOCS_PROMPT = (
    "Ты психолог Михаил Лабковский. "
    "Отвечай в стиле Лабковского: прямо, уверенно, с конкретными рекомендациями. "
    "Сначала объясни причину проблемы, затем дай конкретные шаги для решения. "
    "Если видишь, что кому-то в ситуации нужна профессиональная помощь — скажи об этом прямо."
)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("labkovsky")
    print(f"  {collection.count()} documents in index")

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

    for i, rec in enumerate(records):
        rec_id = rec.get("id", "")
        question = rec.get("question", "")
        answer = rec.get("answer", "")

        if not question or not answer:
            continue

        # Retrieve docs — only from articles and book, not QA
        query_embedding = embed_model.encode(f"query: {question}")
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=RETRIEVE_K,
            where={"source_type": {"$in": ["articles", "book", "interviews"]}},
            include=["documents", "metadatas", "distances"]
        )

        # Filter by distance, then expand with siblings for book/article chunks
        docs = []
        for doc_id, doc_text, meta, dist in zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            if dist > MAX_DISTANCE:
                continue

            # Expand with next chunk if same chapter/article
            chunk_id = meta.get("chunk_id")
            if chunk_id is not None:
                try:
                    chunk_num = int(chunk_id)
                    group_id = meta.get("chapter_id") or meta.get("article_id")

                    if "_chunk" in doc_id:
                        base = doc_id.rsplit("_chunk", 1)[0]
                        next_id = f"{base}_chunk{chunk_num + 1}"
                    else:
                        prefix = doc_id.rsplit("_", 1)[0] if "_" in doc_id else doc_id
                        next_id = f"{prefix}_{chunk_num + 1}"

                    # Fetch up to 2 next chunks within same chapter/article
                    for offset in range(1, 3):
                        if "_chunk" in doc_id:
                            nid = f"{base}_chunk{chunk_num + offset}"
                        else:
                            nid = f"{prefix}_{chunk_num + offset}"
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

            docs.append({
                "id": doc_id,
                "text": doc_text,
                "distance": dist,
            })
            if len(docs) >= TOP_K:
                break

        # Separate QA query: find closest non-self QA
        if ADD_QA_DOC:
            qa_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5,
                where={"source_type": {"$eq": "qa_corpus"}},
                include=["documents", "metadatas", "distances"]
            )
            # Find first QA that isn't the same record
            rec_qa_id = rec_id.rsplit("_", 1)[0] if "_" in rec_id else rec_id
            for qa_id, qa_text, qa_meta, qa_dist in zip(
                qa_results['ids'][0],
                qa_results['documents'][0],
                qa_results['metadatas'][0],
                qa_results['distances'][0]
            ):
                # Skip self (same qa_id group)
                qa_qa_id = qa_meta.get("qa_id", qa_id)
                if qa_qa_id == rec_qa_id or qa_id == rec_id:
                    continue
                if qa_dist > QA_MAX_DISTANCE:
                    break
                docs.append({
                    "id": qa_id,
                    "text": qa_text,
                    "distance": qa_dist,
                })
                break

        if not docs:
            no_docs_count += 1
            system_prompt = NO_DOCS_PROMPT
        else:
            docs_text = "\n\n".join([
                f"Документ {j+1}: {d['text']}"
                for j, d in enumerate(docs)
            ])
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)

        output_records.append({
            "id": rec_id,
            "question": question,
            "answer": answer,
            "system_prompt": system_prompt,
            "num_docs": len(docs),
            "doc_ids": [d["id"] for d in docs],
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(output_records)} records -> {OUTPUT_FILE.name}")
    print(f"  Records with no docs (self was only match): {no_docs_count}")

    # Show example
    if output_records:
        ex = output_records[0]
        print(f"\n--- Example ---")
        print(f"ID: {ex['id']}")
        print(f"Q: {ex['question'][:100]}...")
        print(f"Docs: {ex['num_docs']} ({ex['doc_ids']})")
        print(f"System prompt (first 300): {ex['system_prompt'][:300]}...")
        print(f"A: {ex['answer'][:200]}...")


if __name__ == "__main__":
    main()
