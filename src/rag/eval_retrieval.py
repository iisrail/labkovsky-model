#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieval Evaluation Script
Tests if the correct chunk is retrieved for questions from training data.

Metrics:
- Recall@1: Is the correct chunk the #1 result?
- Recall@3: Is the correct chunk in top 3?
- Recall@5: Is the correct chunk in top 5?
- MRR: Mean Reciprocal Rank

Usage:
    python eval_retrieval.py
"""

import json
from pathlib import Path
from query_rag import retrieve, init

# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = SCRIPT_DIR.parent / "data" / "evaluation" / "test_set_retrieval.jsonl"

TOP_K = 5  # How many chunks to retrieve

# ============================================================
# LOAD TEST SET
# ============================================================

def load_test_set(path: Path):
    """Load test set from JSONL file"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ============================================================
# EVALUATION
# ============================================================

def evaluate_retrieval(test_rows, top_k=TOP_K):
    """
    Evaluate retrieval performance.
    
    For each test question, check if the expected chunk ID
    appears in the retrieved results.
    """
    results = []
    
    for i, row in enumerate(test_rows, 1):
        question = row["question"]
        expected_id = row["id"]
        
        print(f"\n[{i}/{len(test_rows)}] {question[:60]}...")
        
        # Retrieve chunks
        docs = retrieve(question, top_k=top_k)
        
        # Get IDs of retrieved chunks
        retrieved_ids = []
        for doc in docs:
            meta = doc["metadata"]
            # Build ID based on source type
            if meta.get("source") == "qa":
                doc_id = meta.get("id", "")
            elif meta.get("source") == "article":
                doc_id = f"{meta.get('article_id', '')}_{meta.get('chunk_id', '')}"
            elif meta.get("source") == "interview":
                doc_id = f"{meta.get('interview_id', '')}_{meta.get('chunk_id', '')}"
            else:
                doc_id = ""
            retrieved_ids.append(doc_id)
        
        # Find position of expected chunk (1-indexed, 0 if not found)
        try:
            position = retrieved_ids.index(expected_id) + 1
        except ValueError:
            position = 0
        
        # Calculate metrics for this query
        recall_1 = 1 if position == 1 else 0
        recall_3 = 1 if 0 < position <= 3 else 0
        recall_5 = 1 if 0 < position <= 5 else 0
        mrr = 1 / position if position > 0 else 0
        
        result = {
            "question": question,
            "expected_id": expected_id,
            "retrieved_ids": retrieved_ids,
            "position": position,
            "recall@1": recall_1,
            "recall@3": recall_3,
            "recall@5": recall_5,
            "mrr": mrr
        }
        results.append(result)
        
        # Show result
        status = f"✅ #{position}" if position > 0 else "❌ NOT FOUND"
        print(f"   Expected: {expected_id} → {status}")
        if position == 0:
            print(f"   Retrieved: {retrieved_ids}")
    
    return results


def print_summary(results):
    """Print evaluation summary"""
    n = len(results)
    
    recall_1 = sum(r["recall@1"] for r in results) / n
    recall_3 = sum(r["recall@3"] for r in results) / n
    recall_5 = sum(r["recall@5"] for r in results) / n
    mrr = sum(r["mrr"] for r in results) / n
    
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test questions: {n}")
    print(f"Top-K retrieved: {TOP_K}")
    print()
    print(f"Recall@1: {recall_1:.2%} ({sum(r['recall@1'] for r in results)}/{n})")
    print(f"Recall@3: {recall_3:.2%} ({sum(r['recall@3'] for r in results)}/{n})")
    print(f"Recall@5: {recall_5:.2%} ({sum(r['recall@5'] for r in results)}/{n})")
    print(f"MRR:      {mrr:.3f}")
    print("=" * 60)
    
    # Show failures
    failures = [r for r in results if r["position"] == 0]
    if failures:
        print(f"\n❌ FAILED ({len(failures)}):")
        for r in failures:
            print(f"   - {r['expected_id']}: {r['question'][:50]}...")
    
    # Show non-#1 results
    not_first = [r for r in results if r["position"] > 1]
    if not_first:
        print(f"\n⚠️ NOT FIRST ({len(not_first)}):")
        for r in not_first:
            print(f"   - {r['expected_id']} at #{r['position']}: {r['question'][:50]}...")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Retrieval Evaluation")
    print("=" * 60)
    
    # Check test set exists
    if not TEST_SET_PATH.exists():
        print(f"❌ Test set not found: {TEST_SET_PATH}")
        return
    
    # Load test set
    test_rows = load_test_set(TEST_SET_PATH)
    print(f"📄 Loaded {len(test_rows)} test questions")
    
    # Initialize retrieval
    print("\n🔧 Initializing retrieval...")
    init()
    
    # Run evaluation
    print("\n🔍 Running retrieval evaluation...")
    results = evaluate_retrieval(test_rows, top_k=TOP_K)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()