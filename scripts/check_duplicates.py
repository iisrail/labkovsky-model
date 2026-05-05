#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check for remaining duplicates in deduped file.
"""

import json
import sys
import io
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

INPUT_FILE = Path("data/fine_tuning/card_principals_deduped_v2.jsonl")
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Check multiple thresholds
THRESHOLDS = [0.92, 0.90, 0.88, 0.85]


def load_jsonl(filepath: Path):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    print(f"Loading {INPUT_FILE}...")
    records = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(records)} records")

    print(f"\nLoading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing embeddings...")
    texts = [f"passage: {r.get('core_principle', '')}" for r in records]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print("\nChecking for duplicates...")
    n = len(records)

    # Find all high-similarity pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= 0.85:  # Lowest threshold we care about
                pairs.append((i, j, sim))

    # Sort by similarity descending
    pairs.sort(key=lambda x: -x[2])

    # Report by threshold
    print("\n=== DUPLICATE CHECK ===")
    for thresh in THRESHOLDS:
        count = sum(1 for p in pairs if p[2] >= thresh)
        print(f"Pairs with similarity >= {thresh}: {count}")

    # Show top pairs
    if pairs:
        print(f"\n=== TOP 20 SIMILAR PAIRS ===")
        for i, j, sim in pairs[:20]:
            rec_i = records[i]
            rec_j = records[j]
            print(f"\nSim={sim:.3f}")
            print(f"  [{i}] {rec_i.get('source', '')}: {rec_i.get('core_principle', '')[:100]}...")
            print(f"  [{j}] {rec_j.get('source', '')}: {rec_j.get('core_principle', '')[:100]}...")


if __name__ == "__main__":
    main()
