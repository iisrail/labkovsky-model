#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response Signal Classifier using Embeddings + Sklearn.

Simple approach:
1. Embed questions with sentence-transformers
2. Train LogisticRegression classifier

Often works better than fine-tuned BERT for small datasets (<500 examples).

Usage:
    python train_dt_sklearn.py --data path/to/qa_rs_corpus.jsonl --output path/to/model
"""

import json
import argparse
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Same as your RAG
VAL_RATIO = 0.15
SEED = 42
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "fine_tuning" / "qa_rs_corpus.jsonl"

# ============================================================
# DATA LOADING
# ============================================================

def load_data(data_path: Path):
    """Load data from JSONL file."""
    questions = []
    response_signals = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                questions.append(record["question"])
                response_signals.append(record["response_signal"])

    # Build label mappings
    rs_labels = sorted(list(set(response_signals)))
    rs2id = {rs: i for i, rs in enumerate(rs_labels)}
    id2rs = {i: rs for i, rs in enumerate(rs_labels)}

    # Convert to numeric labels
    labels = [rs2id[rs] for rs in response_signals]

    print(f"[*] Loaded {len(questions)} examples")
    print(f"[*] {len(rs_labels)} response signals:")
    for rs in rs_labels:
        count = response_signals.count(rs)
        print(f"    {rs}: {count}")

    return questions, labels, rs_labels, rs2id, id2rs

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Response Signal classifier with sklearn")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to qa_rs_corpus.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output directory for model")
    parser.add_argument("--classifier", type=str, default="logreg",
                        choices=["logreg", "svm", "rf"], help="Classifier type")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Response Signal Classifier - Embeddings + Sklearn")
    print("=" * 60)

    # Load data
    print(f"\n[+] Loading data from {data_path}")
    questions, labels, rs_labels, rs2id, id2rs = load_data(data_path)

    # Load embedding model
    print(f"\n[+] Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed all questions
    print(f"\n[+] Embedding {len(questions)} questions...")
    # Add "query: " prefix for e5 models
    questions_prefixed = [f"query: {q}" for q in questions]
    embeddings = embed_model.encode(questions_prefixed, show_progress_bar=True)
    print(f"   Embedding shape: {embeddings.shape}")

    # Split data
    print(f"\n[*] Splitting: {int((1-VAL_RATIO)*100)}% train / {int(VAL_RATIO*100)}% val")
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=labels
    )
    print(f"   Train: {len(X_train)}")
    print(f"   Val: {len(X_val)}")

    # Cross-validation on full data first
    print(f"\n[*] 5-Fold Cross-validation...")

    classifiers = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "svm": SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED)
    }

    for name, clf in classifiers.items():
        scores = cross_val_score(clf, embeddings, labels, cv=5, scoring="accuracy")
        print(f"   {name}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

    # Train selected classifier
    print(f"\n[>] Training {args.classifier}...")
    clf = classifiers[args.classifier]
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n[*] Validation Accuracy: {accuracy:.3f}")

    print(f"\n[*] Classification Report:")
    # Get unique labels in validation set
    unique_labels = sorted(list(set(y_val)))
    target_names = [id2rs[i] for i in unique_labels]
    print(classification_report(
        y_val, y_pred,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    ))

    # Train final model on ALL data
    print(f"\n[>] Training final model on ALL data...")
    clf_final = classifiers[args.classifier]
    clf_final.fit(embeddings, labels)

    # Save model
    print(f"\n[+] Saving model to {output_dir}")

    model_data = {
        "classifier": clf_final,
        "embedding_model_name": EMBEDDING_MODEL,
        "rs_labels": rs_labels,
        "rs2id": rs2id,
        "id2rs": id2rs
    }

    with open(output_dir / "rs_classifier.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("\n[OK] Training complete!")
    print(f"   Model saved to: {output_dir / 'rs_classifier.pkl'}")


if __name__ == "__main__":
    main()
