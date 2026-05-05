#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-dedupe card_principals.jsonl.

Removes very similar duplicates (>= 0.92 similarity).
Keeps original structure.

Output: card_principals_deduped.jsonl
"""

import json
import sys
import io
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

from sentence_transformers import SentenceTransformer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

INPUT_FILE = Path("data/fine_tuning/card_principals.jsonl")
OUTPUT_FILE = Path("data/fine_tuning/card_principals_deduped.jsonl")

DUPLICATE_THRESHOLD = 0.92  # Very high - only true duplicates
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def load_records(filepath: Path) -> List[Dict[str, Any]]:
    """Load records."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    rec['_idx'] = idx
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def find_duplicates(records: List[Dict], embeddings: np.ndarray) -> set:
    """Find indices of duplicate records to remove."""
    n = len(records)
    to_remove = set()

    print(f"Comparing {n} records...")

    for i in range(n):
        if i in to_remove:
            continue

        for j in range(i + 1, n):
            if j in to_remove:
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])

            if sim >= DUPLICATE_THRESHOLD:
                # Keep the one with more detail (longer) or from book source
                rec_i = records[i]
                rec_j = records[j]

                # Prefer book over QA
                source_i = rec_i.get('source', '')
                source_j = rec_j.get('source', '')

                if source_i == 'reasoning_card' and source_j == 'qa_principle':
                    to_remove.add(j)
                elif source_j == 'reasoning_card' and source_i == 'qa_principle':
                    to_remove.add(i)
                else:
                    # Keep longer one
                    len_i = len(rec_i.get('core_principle', ''))
                    len_j = len(rec_j.get('core_principle', ''))
                    if len_i >= len_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

                print(f"  Duplicate found (sim={sim:.3f}):")
                print(f"    [{i}]: {rec_i.get('core_principle', '')[:80]}...")
                print(f"    [{j}]: {rec_j.get('core_principle', '')[:80]}...")
                print(f"    -> Removing [{j if j in to_remove else i}]")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n}")

    return to_remove


def main():
    print(f"Loading records from {INPUT_FILE}...")
    records = load_records(INPUT_FILE)
    print(f"Loaded {len(records)} records")

    # Stats by source
    sources = defaultdict(int)
    for rec in records:
        sources[rec.get('source', 'unknown')] += 1
    print("\nBy source:")
    for src, count in sources.items():
        print(f"  {src}: {count}")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute embeddings
    print("Computing embeddings...")
    texts = [f"passage: {r.get('core_principle', '')}" for r in records]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Computed {len(embeddings)} embeddings")

    # Find duplicates
    print(f"\nFinding duplicates (threshold >= {DUPLICATE_THRESHOLD})...")
    to_remove = find_duplicates(records, embeddings)
    print(f"\nFound {len(to_remove)} duplicates to remove")

    # Filter records
    kept_records = []
    for i, rec in enumerate(records):
        if i not in to_remove:
            # Remove internal index
            rec.pop('_idx', None)
            kept_records.append(rec)

    print(f"\nKept {len(kept_records)} records")

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Final stats
    print(f"\n=== SUMMARY ===")
    print(f"Input: {len(records)} records")
    print(f"Removed: {len(to_remove)} duplicates")
    print(f"Output: {len(kept_records)} records")
    print(f"\nOutput file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
