#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-dedupe card_principals_deduped_v2.jsonl with threshold 0.88
"""

import json
import sys
import io
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

INPUT_FILE = Path("data/fine_tuning/card_principals_deduped_v2.jsonl")
OUTPUT_FILE = Path("data/fine_tuning/card_principals_deduped_v2_090.jsonl")

DUPLICATE_THRESHOLD = 0.90
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


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

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing embeddings...")
    texts = [f"passage: {r.get('core_principle', '')}" for r in records]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print(f"\nFinding duplicates (threshold >= {DUPLICATE_THRESHOLD})...")
    n = len(records)
    to_remove = set()

    for i in range(n):
        if i in to_remove:
            continue

        for j in range(i + 1, n):
            if j in to_remove:
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])

            if sim >= DUPLICATE_THRESHOLD:
                rec_i = records[i]
                rec_j = records[j]

                # Prefer reasoning_card over qa_principle
                src_i = rec_i.get('source', '')
                src_j = rec_j.get('source', '')

                if src_i == 'reasoning_card' and src_j == 'qa_principle':
                    to_remove.add(j)
                    removed = j
                elif src_j == 'reasoning_card' and src_i == 'qa_principle':
                    to_remove.add(i)
                    removed = i
                else:
                    # Keep longer
                    len_i = len(rec_i.get('core_principle', ''))
                    len_j = len(rec_j.get('core_principle', ''))
                    if len_i >= len_j:
                        to_remove.add(j)
                        removed = j
                    else:
                        to_remove.add(i)
                        removed = i

                print(f"Sim={sim:.3f} | Remove [{removed}]")
                print(f"  KEEP: {records[i if removed == j else j].get('core_principle', '')[:80]}...")
                print(f"  DROP: {records[removed].get('core_principle', '')[:80]}...")

    print(f"\nRemoving {len(to_remove)} duplicates")

    # Filter
    kept = [rec for i, rec in enumerate(records) if i not in to_remove]

    print(f"\n=== SUMMARY ===")
    print(f"Input: {len(records)}")
    print(f"Removed: {len(to_remove)}")
    print(f"Output: {len(kept)}")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("Done!")


if __name__ == "__main__":
    main()
