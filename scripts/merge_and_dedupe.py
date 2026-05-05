#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge card_principals_deduped_v1.jsonl + card_qa_principles_strict_v2.jsonl
Dedupe and output card_principals_deduped_v2.jsonl
"""

import json
import sys
import io
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Input files
FILE_BOOK = Path("data/fine_tuning/card_principals_deduped_v1.jsonl")
FILE_QA = Path("data/fine_tuning/card_qa_principles_strict_v2.jsonl")
OUTPUT_FILE = Path("data/fine_tuning/card_principals_deduped_v2.jsonl")

DUPLICATE_THRESHOLD = 0.88
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    # Load both files
    print(f"Loading {FILE_BOOK}...")
    book_records = load_jsonl(FILE_BOOK)
    print(f"  Loaded {len(book_records)} book principles")

    print(f"Loading {FILE_QA}...")
    qa_records = load_jsonl(FILE_QA)
    print(f"  Loaded {len(qa_records)} QA principles")

    # Mark source for priority
    for rec in book_records:
        rec['_source_priority'] = 1  # Higher priority
        rec['_original_source'] = 'book'

    for rec in qa_records:
        rec['_source_priority'] = 2  # Lower priority
        rec['_original_source'] = 'qa'
        # Preserve source_id if exists
        if 'source_id' not in rec and 'qa_id' in rec:
            rec['source_id'] = rec['qa_id']

    # Combine
    all_records = book_records + qa_records
    print(f"\nTotal combined: {len(all_records)}")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute embeddings
    print("Computing embeddings...")
    texts = [f"passage: {r.get('core_principle', '')}" for r in all_records]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Find duplicates
    print(f"\nFinding duplicates (threshold >= {DUPLICATE_THRESHOLD})...")
    n = len(all_records)
    to_remove = set()
    duplicate_pairs = []

    for i in range(n):
        if i in to_remove:
            continue

        for j in range(i + 1, n):
            if j in to_remove:
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])

            if sim >= DUPLICATE_THRESHOLD:
                rec_i = all_records[i]
                rec_j = all_records[j]

                # Priority: book > qa, then longer
                prio_i = rec_i['_source_priority']
                prio_j = rec_j['_source_priority']

                if prio_i < prio_j:
                    to_remove.add(j)
                    removed_idx = j
                elif prio_j < prio_i:
                    to_remove.add(i)
                    removed_idx = i
                else:
                    # Same priority - keep longer
                    len_i = len(rec_i.get('core_principle', ''))
                    len_j = len(rec_j.get('core_principle', ''))
                    if len_i >= len_j:
                        to_remove.add(j)
                        removed_idx = j
                    else:
                        to_remove.add(i)
                        removed_idx = i

                duplicate_pairs.append({
                    'sim': sim,
                    'kept': i if removed_idx == j else j,
                    'removed': removed_idx,
                })

    print(f"Found {len(duplicate_pairs)} duplicate pairs")
    print(f"Removing {len(to_remove)} records")

    # Show some duplicates
    if duplicate_pairs:
        print("\nSample duplicates:")
        for pair in duplicate_pairs[:5]:
            kept = all_records[pair['kept']]
            removed = all_records[pair['removed']]
            print(f"\n  Sim={pair['sim']:.3f}")
            print(f"  KEPT ({kept['_original_source']}): {kept.get('core_principle', '')[:80]}...")
            print(f"  REMOVED ({removed['_original_source']}): {removed.get('core_principle', '')[:80]}...")

    # Filter and clean
    kept_records = []
    for i, rec in enumerate(all_records):
        if i not in to_remove:
            # Remove internal fields only
            rec.pop('_source_priority', None)
            rec.pop('_original_source', None)
            rec.pop('qa_id', None)  # Use source_id instead
            # Keep source_id if exists
            kept_records.append(rec)

    # Stats
    book_kept = sum(1 for r in kept_records if r.get('source') == 'reasoning_card')
    qa_kept = sum(1 for r in kept_records if r.get('source') == 'qa_principle')

    print(f"\n=== SUMMARY ===")
    print(f"Input: {len(book_records)} book + {len(qa_records)} QA = {len(all_records)} total")
    print(f"Removed: {len(to_remove)} duplicates")
    print(f"Output: {len(kept_records)} records")
    print(f"  - Book principles: {book_kept}")
    print(f"  - QA principles: {qa_kept}")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("Done!")


if __name__ == "__main__":
    main()
