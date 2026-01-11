#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Decision Type (DT) classifier using rubert-tiny2.

Input: qa_dt_corpus.jsonl with {"question": "...", "decision_type": "..."}
Output: Fine-tuned classifier model

Usage:
    python train_dt_classifier.py --data path/to/qa_dt_corpus.jsonl --output path/to/model
"""

import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "DeepPavlov/rubert-base-cased"  # Best Russian BERT
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-5  # Lower LR to reduce overfitting
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.1  # Stronger regularization
VAL_RATIO = 0.15
SEED = 42
WARMUP_RATIO = 0.1
VAL_RATIO = 0.15
SEED = 42

# Decision types (will be populated from data)
DT_LABELS = []
DT2ID = {}
ID2DT = {}

# ============================================================
# DATASET
# ============================================================

class DTDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_length=MAX_LENGTH):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.questions[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================================
# DATA LOADING
# ============================================================

def load_data(data_path: Path):
    """Load and prepare data from JSONL file."""
    global DT_LABELS, DT2ID, ID2DT
    
    questions = []
    decision_types = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                questions.append(record["question"])
                decision_types.append(record["decision_type"])
    
    # Build label mappings
    DT_LABELS = sorted(list(set(decision_types)))
    DT2ID = {dt: i for i, dt in enumerate(DT_LABELS)}
    ID2DT = {i: dt for i, dt in enumerate(DT_LABELS)}
    
    # Convert to numeric labels
    labels = [DT2ID[dt] for dt in decision_types]
    
    print(f"📊 Loaded {len(questions)} examples")
    print(f"📊 {len(DT_LABELS)} decision types:")
    for dt in DT_LABELS:
        count = decision_types.count(dt)
        print(f"   {dt}: {count}")
    
    return questions, labels

# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    """Compute accuracy and per-class metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train DT classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to qa_dt_corpus.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()
    
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DT Classifier Training")
    print("=" * 60)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\n📂 Loading data from {data_path}")
    questions, labels = load_data(data_path)
    
    # Split data
    print(f"\n📊 Splitting: {int((1-VAL_RATIO)*100)}% train / {int(VAL_RATIO*100)}% val")
    train_q, val_q, train_l, val_l = train_test_split(
        questions, labels, 
        test_size=VAL_RATIO, 
        random_state=SEED,
        stratify=labels
    )
    print(f"   Train: {len(train_q)}")
    print(f"   Val: {len(val_q)}")
    
    # Load tokenizer and model
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(DT_LABELS),
        id2label=ID2DT,
        label2id=DT2ID,
        hidden_dropout_prob=0.2,  # Add dropout for regularization
        attention_probs_dropout_prob=0.2
    )
    
    # Freeze BERT layers, train only classifier head
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    model.to(device)
    
    # Create datasets
    train_dataset = DTDataset(train_q, train_l, tokenizer)
    val_dataset = DTDataset(val_q, val_l, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,  # Stronger regularization
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print(f"\n🚀 Training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    
    trainer.train()
    
    # Evaluate
    print(f"\n📊 Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Detailed classification report
    print(f"\n📊 Classification Report:")
    val_predictions = trainer.predict(val_dataset)
    preds = np.argmax(val_predictions.predictions, axis=-1)
    
    # Get unique labels in validation set
    unique_labels = sorted(list(set(val_l)))
    target_names = [ID2DT[i] for i in unique_labels]
    
    print(classification_report(
        val_l, preds, 
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    ))
    
    # Save model
    print(f"\n💾 Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"dt2id": DT2ID, "id2dt": ID2DT}, f, ensure_ascii=False, indent=2)
    print(f"   Label map saved to {label_map_path}")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()