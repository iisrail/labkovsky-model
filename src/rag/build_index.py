#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline - Build Index

Builds ChromaDB index from training data sources.

Sources:
    - qa_rs_final.jsonl: Q&A with RS segmentation (in-context examples)
    - articles_with_questions.jsonl: Article excerpts

Usage:
    python build_index.py

Output:
    ./chroma_db/ (vector store)
"""

import json
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DATA_DIR = PROJECT_ROOT / "data" / "fine_tuning"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Files to index
TARGET_FILES = {
    "qa_corpus": DATA_DIR / "qa_clean.jsonl",
    "articles": PROCESSED_DIR / "articles_with_questions.jsonl",
    "interviews": PROCESSED_DIR / "interviews.jsonl",
    "book": PROCESSED_DIR / "Hochu_i_budu_with_questions.jsonl",
}

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Embed questions only — matches user queries better at inference
QUERY_FIELDS = ["question", "potential_questions"]

# ============================================================
# DATA LOADING
# ============================================================

def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Line {line_num}: JSON error - {e}")
    return records


def extract_text_for_embedding(record: dict) -> str:
    """Extract questions only for embedding — matches user queries at inference."""
    parts = []

    for field in QUERY_FIELDS:
        if field in record:
            value = record[field]
            if isinstance(value, list):
                parts.extend(value)
            elif isinstance(value, str) and value.strip():
                parts.append(value)

    # Fallback: if no question fields, use text (for records without questions)
    if not parts:
        text = record.get("text", record.get("answer", ""))
        if text:
            parts.append(text)

    return " ".join(parts)


def extract_answer_for_document(record: dict) -> str:
    """Extract answer for document content (what model sees for generation)."""
    return record.get("answer", record.get("text", ""))


def extract_metadata(record: dict, source_file: str) -> dict:
    """Extract metadata from record."""
    exclude_fields = {"text", "answer", "potential_questions", "char_count"}
    metadata = {"source_file": source_file}

    for key, value in record.items():
        if key in exclude_fields:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            metadata[key] = " | ".join(value[:3])

    # Add qa_id from id field
    # "srJvn19GKNA_01_expl" → "srJvn19GKNA_01"
    if "id" in record:
        parts = record["id"].rsplit("_", 1)
        if len(parts) == 2:
            metadata["qa_id"] = parts[0]

    return metadata


def generate_doc_id(record: dict, source_file: str, index: int) -> str:
    """Generate unique document ID."""
    if "id" in record:
        return str(record["id"])

    source = record.get("source", source_file.replace(".jsonl", ""))

    if "article_id" in record:
        chunk = record.get("chunk_id", index)
        return f"{record['article_id']}_chunk{chunk}"
    elif "video_id" in record:
        return f"{record['video_id']}_{index}"
    else:
        return f"{source}_{index}"


# ============================================================
# BUILD INDEX
# ============================================================

def prepare_documents() -> tuple[list, list, list, list]:
    """Load all target files and prepare for indexing.

    Returns:
        texts: Text for embedding (question + answer for matching)
        metadatas: Metadata dicts
        ids: Document IDs
        answers: Answer-only text (what model sees for generation)
    """
    texts, metadatas, ids, answers = [], [], [], []
    seen_ids = set()

    # Check which files exist
    existing = {name: path for name, path in TARGET_FILES.items() if path.exists()}
    missing = {name: path for name, path in TARGET_FILES.items() if not path.exists()}

    for name, path in missing.items():
        print(f"⚠ {name} not found: {path}")

    if not existing:
        print("❌ No files to index!")
        return [], [], [], []

    print(f"📂 Indexing {len(existing)} files:")
    for name, path in existing.items():
        print(f"   - {name}: {path.name}")
    print()

    for name, filepath in existing.items():
        records = load_jsonl(filepath)

        print(f"📄 {filepath.name}: {len(records)} records")

        for i, record in enumerate(records):
            text = extract_text_for_embedding(record)
            if not text.strip():
                continue

            text_for_embedding = f"passage: {text}"
            answer = extract_answer_for_document(record)
            doc_id = generate_doc_id(record, filepath.name, i)

            if doc_id in seen_ids:
                doc_id = f"{doc_id}_{len(ids)}"
            seen_ids.add(doc_id)

            metadata = extract_metadata(record, filepath.name)
            metadata["source_type"] = name  # Add source type

            texts.append(text_for_embedding)
            answers.append(answer)
            metadatas.append(metadata)
            ids.append(doc_id)

    return texts, metadatas, ids, answers


def build_index():
    """Build the ChromaDB index."""
    print("=" * 60)
    print("RAG Pipeline - Build Index")
    print("=" * 60)
    print(f"💾 Output: {CHROMA_DIR}\n")

    # Load data
    print("📚 Loading data...")
    texts, metadatas, ids, answers = prepare_documents()

    if not texts:
        return

    print(f"\n✅ Total documents: {len(texts)}")

    # Load embedding model
    print(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Create embeddings
    print(f"\n🔢 Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # Save to ChromaDB
    print(f"\n💾 Saving to ChromaDB...")

    chroma_dir = CHROMA_DIR
    if chroma_dir.exists():
        try:
            shutil.rmtree(chroma_dir)
            print("   (Removed old index)")
        except PermissionError:
            print("   ⚠ Could not remove old index (files locked). Using new directory.")
            chroma_dir = PROJECT_ROOT / "chroma_db_new"
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.create_collection(
        name="labkovsky",
        metadata={"description": "Labkovsky RAG knowledge base"}
    )

    # Store ANSWERS as documents (what model sees for generation)
    # Embeddings are from question+answer (for semantic matching)
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=answers
    )

    print(f"✅ Saved {collection.count()} documents")

    print("\n" + "=" * 60)
    print("✅ INDEX CREATED!")
    print("=" * 60)
    print(f"   Documents: {len(texts)}")
    print(f"   Storage: {chroma_dir}")


if __name__ == "__main__":
    build_index()
