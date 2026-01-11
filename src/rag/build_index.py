#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline - Build Index (v2)
Flexible version - auto-discovers JSONL files and handles any source type.

Usage:
    python build_index.py                    # Use default data folder
    python build_index.py /path/to/data      # Use custom folder

Expected JSONL format (each file):
{
    "text": "Main content for RAG",
    "source": "book|article|qa|interview|...",
    "id": "unique_id",                        # Optional but recommended
    "potential_questions": ["q1", "q2"],      # Optional - improves retrieval
    ... any other metadata fields ...
}

Output:
    ./chroma_db/ (vector store)
"""

import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
CHROMA_DIR = SCRIPT_DIR.parent / "chroma_db"

# Embedding model (excellent for Russian)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Fields that go into the embedding text (if present)
TEXT_FIELDS = ["text", "answer"]  # Main content
QUERY_FIELDS = ["potential_questions", "question"]  # Questions to prepend

# ============================================================
# DATA LOADING
# ============================================================

def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Line {line_num}: JSON error - {e}")
    return records


def discover_jsonl_files(data_dir: Path) -> list[Path]:
    """Find all .jsonl files in directory"""
    files = list(data_dir.glob("*.jsonl"))
    return sorted(files)


def extract_text_for_embedding(record: dict) -> str:
    """
    Extract text for embedding from a record.
    Combines questions (for better retrieval) + main content.
    """
    parts = []
    
    # 1. Add questions/queries first (helps with retrieval)
    for field in QUERY_FIELDS:
        if field in record:
            value = record[field]
            if isinstance(value, list):
                parts.extend(value)
            elif isinstance(value, str) and value.strip():
                parts.append(value)
    
    # 2. Add main text content
    for field in TEXT_FIELDS:
        if field in record:
            value = record[field]
            if isinstance(value, str) and value.strip():
                parts.append(value)
    
    return " ".join(parts)


def extract_metadata(record: dict, source_file: str) -> dict:
    """
    Extract metadata from record.
    Keeps all fields except large text fields.
    """
    # Fields to exclude from metadata (too large or redundant)
    exclude_fields = {"text", "answer", "potential_questions", "char_count"}
    
    metadata = {"source_file": source_file}
    
    for key, value in record.items():
        if key in exclude_fields:
            continue
        
        # ChromaDB metadata must be str, int, float, or bool
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, list) and value:
            # Convert list to string (e.g., questions)
            if all(isinstance(v, str) for v in value):
                metadata[key] = " | ".join(value[:3])  # Keep first 3
    
    return metadata


def generate_doc_id(record: dict, source_file: str, index: int) -> str:
    """Generate unique document ID"""
    # Try to use existing ID from record
    if "id" in record:
        return str(record["id"])
    
    # Build ID from available fields
    source = record.get("source", source_file.replace(".jsonl", ""))
    
    # Try different ID patterns
    if "article_id" in record:
        chunk = record.get("chunk_id", index)
        return f"{record['article_id']}_chunk{chunk}"
    elif "interview_id" in record:
        chunk = record.get("chunk_id", index)
        return f"{record['interview_id']}_chunk{chunk}"
    elif "video_id" in record:
        return f"{record['video_id']}_{index}"
    elif "book_id" in record:
        chunk = record.get("chunk_id", index)
        chapter = record.get("chapter_id", "")
        if chapter:
            return f"{record['book_id']}_{chapter}_{chunk}"
        return f"{record['book_id']}_chunk{chunk}"
    else:
        return f"{source}_{index}"


# ============================================================
# MAIN PROCESSING
# ============================================================

def prepare_documents(data_dir: Path) -> tuple[list, list, list]:
    """
    Load all JSONL files and prepare for indexing.
    
    Returns:
        texts: List of texts for embedding
        metadatas: List of metadata dicts
        ids: List of document IDs
    """
    texts = []
    metadatas = []
    ids = []
    seen_ids = set()
    
    # Discover all JSONL files
    jsonl_files = discover_jsonl_files(data_dir)
    
    if not jsonl_files:
        print(f"❌ No .jsonl files found in {data_dir}")
        return [], [], []
    
    print(f"📂 Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"   - {f.name}")
    print()
    
    # Process each file
    for filepath in jsonl_files:
        records = load_jsonl(filepath)
        print(f"📄 {filepath.name}: {len(records)} records")
        
        for i, record in enumerate(records):
            # Extract text for embedding
            text = extract_text_for_embedding(record)
            
            if not text.strip():
                print(f"  ⚠ Record {i}: empty text, skipping")
                continue
            
            # For e5 models - add "passage:" prefix
            text_for_embedding = f"passage: {text}"
            
            # Generate unique ID
            doc_id = generate_doc_id(record, filepath.name, i)
            
            # Handle duplicate IDs
            if doc_id in seen_ids:
                doc_id = f"{doc_id}_{len(ids)}"
            seen_ids.add(doc_id)
            
            # Extract metadata
            metadata = extract_metadata(record, filepath.name)
            
            texts.append(text_for_embedding)
            metadatas.append(metadata)
            ids.append(doc_id)
    
    return texts, metadatas, ids


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(data_dir: Path = None):
    """Main function to build the index"""
    
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    print("=" * 60)
    print("RAG Pipeline - Build Index (v2)")
    print("=" * 60)
    print(f"📂 Data directory: {data_dir}")
    print(f"💾 Output: {CHROMA_DIR}")
    print()
    
    # Check data directory
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    # 1. Load all data
    print("📚 Loading data...")
    texts, metadatas, ids = prepare_documents(data_dir)
    
    if not texts:
        print("❌ No documents to index!")
        return
    
    print(f"\n✅ Total documents: {len(texts)}")
    
    # Show source distribution
    sources = {}
    for meta in metadatas:
        src = meta.get("source", meta.get("source_file", "unknown"))
        sources[src] = sources.get(src, 0) + 1
    print("   By source:")
    for src, count in sorted(sources.items()):
        print(f"      {src}: {count}")
    
    # 2. Load embedding model
    print(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}")
    print("   (First run downloads ~2GB)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded")
    
    # 3. Create embeddings
    print(f"\n🔢 Creating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✅ Embeddings shape: {embeddings.shape}")
    
    # 4. Save to ChromaDB
    print(f"\n💾 Saving to ChromaDB...")
    
    # Remove old database
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("   (Removed old index)")
    
    # Create client and collection
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(
        name="labkovsky",
        metadata={"description": "Labkovsky RAG knowledge base"}
    )
    
    # Add documents (remove "passage: " prefix for storage)
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=[t.replace("passage: ", "") for t in texts]
    )
    
    print(f"✅ Saved {collection.count()} documents")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("✅ INDEX CREATED!")
    print("=" * 60)
    print(f"   Documents: {len(texts)}")
    print(f"   Embedding dim: {embeddings.shape[1]}")
    print(f"   Storage: {CHROMA_DIR}")
    print("\n   Run: python query_rag.py")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    # Allow custom data directory from command line
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = DEFAULT_DATA_DIR
    
    build_index(data_dir)