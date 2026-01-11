#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict Response Signal using trained sklearn classifier.

Usage:
    python predict_rs_sklearn.py --model path/to/rs_classifier.pkl --question "Your question here"
    python predict_rs_sklearn.py --model path/to/rs_classifier.pkl --interactive
"""

import argparse
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# PREDICTOR CLASS
# ============================================================

class ResponseSignalPredictor:
    def __init__(self, model_path: Path):
        """Load model from pickle file."""
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.classifier = model_data["classifier"]
        self.rs_labels = model_data["rs_labels"]
        self.rs2id = model_data["rs2id"]
        self.id2rs = model_data["id2rs"]
        self.embedding_model_name = model_data["embedding_model_name"]

        print(f"🤖 Loading embedding model: {self.embedding_model_name}")
        self.embed_model = SentenceTransformer(self.embedding_model_name)
        print("✅ Model loaded")

    def predict(self, question: str) -> str:
        """Predict response signal for a question."""
        # Add prefix for e5 models
        question_prefixed = f"query: {question}"
        embedding = self.embed_model.encode([question_prefixed])
        pred_id = self.classifier.predict(embedding)[0]
        return self.id2rs[pred_id]

    def predict_proba(self, question: str) -> dict:
        """Predict response signal with probabilities (if classifier supports it)."""
        question_prefixed = f"query: {question}"
        embedding = self.embed_model.encode([question_prefixed])

        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(embedding)[0]
            return {self.id2rs[i]: float(p) for i, p in enumerate(probs)}
        else:
            # SVM doesn't have predict_proba by default
            pred_id = self.classifier.predict(embedding)[0]
            return {self.id2rs[pred_id]: 1.0}

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Predict Response Signal")
    parser.add_argument("--model", type=str, required=True, help="Path to rs_classifier.pkl")
    parser.add_argument("--question", type=str, help="Question to classify")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    model_path = Path(args.model)
    predictor = ResponseSignalPredictor(model_path)

    if args.interactive:
        print("\n" + "=" * 60)
        print("Response Signal Classifier - Interactive Mode")
        print("Type 'quit' to exit")
        print("=" * 60)

        while True:
            question = input("\n❓ Question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue

            rs = predictor.predict(question)
            probs = predictor.predict_proba(question)

            print(f"\n🎯 Response Signal: {rs}")
            print("📊 Probabilities:")
            for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
                print(f"   {label}: {prob:.3f}")

    elif args.question:
        rs = predictor.predict(args.question)
        probs = predictor.predict_proba(args.question)

        print(f"\n🎯 Response Signal: {rs}")
        print("📊 Probabilities:")
        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"   {label}: {prob:.3f}")
    else:
        print("Please provide --question or --interactive flag")


if __name__ == "__main__":
    main()
