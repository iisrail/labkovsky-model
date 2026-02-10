#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAFT-style Fine-tuning: Train with RAG context in prompts.

Based on RAFT (Retrieval-Augmented Fine-Tuning) approach:
- https://arxiv.org/abs/2403.10131
- https://gorilla.cs.berkeley.edu/blogs/9_raft.html

The model learns to use retrieved documents to generate answers,
matching the inference format exactly.

Training format:
Документы:
{retrieved_document}

Вопрос: {question}
Ответ: {answer}

Usage:
    python train_lora_with_rag.py
"""

import json
import random
import sys
import torch
import gc
from functools import partial
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from sentence_transformers import SentenceTransformer
import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FINE_TUNING_DATA_DIR,
    MODEL_NAME,
    MODELS_DIR,
    CHROMA_DIR,
)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# ==============================================================
# MEMORY CLEANUP CALLBACK
# ==============================================================

class MemoryCleanupCallback(TrainerCallback):
    """Periodically clear CUDA cache to prevent memory buildup."""

    def __init__(self, cleanup_every_n_steps: int = 50):
        self.cleanup_every_n_steps = cleanup_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.cleanup_every_n_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================
# SETTINGS
# ==============================================================

# Paths
DATA_DIR = FINE_TUNING_DATA_DIR
PROCESSED_DIR = DATA_DIR.parent / "processed"

# Data sources
INPUT_FILES = {
    "qa_corpus": DATA_DIR / "qa_rs_corpus_short.jsonl",
    "interview": PROCESSED_DIR / "intervie_ft.jsonl",
    "book": PROCESSED_DIR / "Hochu_i_budu_with_questions.jsonl",
    "articles": PROCESSED_DIR / "articles_with_questions.jsonl",
}

FORMATTED_DATA = DATA_DIR / "train_raft_formatted.jsonl"
OUTPUT_DIR = MODELS_DIR / "labkovsky-yandex-it-lora"  # For -it model

# RAG settings
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TOP_K_DOCS = 1  # Number of documents to include (1 = just oracle, more = with distractors)
ALT_SOURCE_RATIO = 0.2  # 20% of examples use book/articles instead of qa_corpus

# LoRA settings (based on RAFT/torchtune recommendations)
LORA_R = 8            # Low rank as recommended
LORA_ALPHA = 16       # 2x rank
LORA_DROPOUT = 0.0    # No dropout per torchtune config

# Training settings (RAFT recommends lower LR, fewer epochs)
MAX_SEQ_LENGTH = 1024  # Longer to fit documents + question + answer
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 3e-4  # Higher LR per torchtune recommendations
NUM_EPOCHS = 10       # Allow more epochs, early stopping will find optimal
WARMUP_STEPS = 20
VAL_RATIO = 0.10  # 90/10 split - more training data
RANDOM_SEED = 42

# ==============================================================
# RAG RETRIEVAL
# ==============================================================

_embed_model = None
_collection = None

def init_rag():
    """Initialize RAG components for training data preparation."""
    global _embed_model, _collection

    if _embed_model is None:
        print(f"[+] Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)

    if _collection is None:
        print(f"[+] Connecting to ChromaDB: {CHROMA_DIR}")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection("labkovsky")
        print(f"    Loaded {_collection.count()} documents")

    return _embed_model, _collection


def retrieve_document(question: str, exclude_id: str = None, source_filter: list = None) -> str:
    """
    Retrieve relevant document for a question.

    Args:
        question: The question to find relevant context for
        exclude_id: Document ID to exclude (to avoid retrieving the exact answer)
        source_filter: List of source_type values to filter by (e.g., ["book", "articles"])

    Returns:
        Retrieved document text
    """
    embed_model, collection = init_rag()

    query_embedding = embed_model.encode(f"query: {question}")

    # Retrieve more to have options after filtering
    n_results = 10

    # Build where clause for source filtering
    where_clause = None
    if source_filter:
        where_clause = {"source_type": {"$in": source_filter}}

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas"],
            where=where_clause
        )
    except Exception:
        # Fallback without filter if metadata doesn't exist
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas"]
        )

    if not results['ids'][0]:
        return ""

    # Filter out excluded ID and take top result
    for doc_id, doc in zip(results['ids'][0], results['documents'][0]):
        if exclude_id and exclude_id in doc_id:
            continue
        return doc

    return results['documents'][0][0] if results['documents'][0] else ""


# ==============================================================
# DATA LOADING WITH RAG
# ==============================================================

def format_with_rag(question: str, answer: str, doc_id: str = None) -> dict:
    """
    Format a Q&A pair with RAG context.

    RAFT approach:
    - 80% of examples: retrieve from any source (best match)
    - 20% of examples: retrieve only from book/articles (alternative sources)

    This teaches model to use related content even when exact QA isn't in RAG.

    Format:
    Документы:
    {retrieved_doc}

    Вопрос: {question}
    Ответ: {answer}
    """
    # 20% of time: use alternative sources (book/articles) instead of qa_corpus
    use_alt_source = random.random() < ALT_SOURCE_RATIO

    if use_alt_source:
        # Retrieve from book/articles/interview only (not qa_corpus)
        # This teaches model to use related content when exact QA isn't available
        rag_doc = retrieve_document(
            question,
            exclude_id=doc_id,
            source_filter=["book", "articles", "interview"]
        )
    else:
        # Normal retrieval from any source (exclude exact match)
        rag_doc = retrieve_document(question, exclude_id=doc_id)

    if rag_doc:
        prompt = f"Документы:\n{rag_doc}\n\nВопрос: {question}\nОтвет:"
    else:
        # Fallback if no document found
        prompt = f"Вопрос: {question}\nОтвет:"

    return {
        "prompt": prompt,
        "completion": answer
    }


def load_qa_corpus(filepath: Path) -> list:
    """Load Q&A corpus with RAG context."""
    records = []
    skipped = 0

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"   Line {line_num}: JSON error - {e}")
                    continue

                if data.get("short_answer"):
                    skipped += 1
                    continue

                # Get document ID if available (to exclude from RAG)
                doc_id = data.get("id", "")

                record = format_with_rag(
                    question=data['question'],
                    answer=data['answer'],
                    doc_id=doc_id
                )
                records.append(record)

    print(f"   {filepath.name}: {len(records)} examples (skipped {skipped})")
    return records


def load_interview(filepath: Path) -> list:
    """Load interview format with RAG context."""
    records = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        for obj in content.split('\n{'):
            if not obj.strip():
                continue
            if not obj.startswith('{'):
                obj = '{' + obj
            try:
                data = json.loads(obj.rstrip(',\n'))
                messages = data.get("messages", [])
                for i in range(0, len(messages) - 1, 2):
                    if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                        record = format_with_rag(
                            question=messages[i]['content'],
                            answer=messages[i+1]['content']
                        )
                        records.append(record)
            except json.JSONDecodeError:
                continue

    print(f"   {filepath.name}: {len(records)} examples")
    return records


def load_text_with_questions(filepath: Path) -> list:
    """Load text with questions format with RAG context."""
    records = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = data.get("text", "")
                questions = data.get("potential_questions", [])

                for q in questions:
                    record = format_with_rag(
                        question=q,
                        answer=text
                    )
                    records.append(record)

    print(f"   {filepath.name}: {len(records)} examples")
    return records


def create_formatted_data(input_files: dict, output_path: Path) -> int:
    """Combine all data sources with RAG context."""
    print(f"\n Creating RAFT training data...")

    # Initialize RAG first
    init_rag()

    all_records = []

    for name, filepath in input_files.items():
        if not filepath.exists():
            print(f"   {filepath.name} not found, skipping")
            continue

        if name == "qa_corpus":
            all_records.extend(load_qa_corpus(filepath))
        elif name == "interview":
            all_records.extend(load_interview(filepath))
        elif name in ["book", "articles"]:
            all_records.extend(load_text_with_questions(filepath))

    # Save combined data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n   Total: {len(all_records)} RAFT-formatted examples")
    print(f"   Output: {output_path}")

    return len(all_records)


# ==============================================================
# TRAINING DATA PREPARATION
# ==============================================================

def load_training_data(data_path: Path) -> list:
    """Load formatted prompt/completion data."""
    records = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def format_for_sft(examples, tokenizer):
    """Combine prompt + completion for SFT."""
    texts = []
    eos = tokenizer.eos_token

    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i]
        completion = examples["completion"][i]
        text = f"{prompt} {completion}{eos}"
        texts.append(text)

    return {"text": texts}


# ==============================================================
# MAIN TRAINING
# ==============================================================

def main():
    torch.cuda.empty_cache()
    gc.collect()

    print("=" * 60)
    print("RAFT: Retrieval-Augmented Fine-Tuning")
    print("Training with RAG context in prompts")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"   BF16: {'supported' if bf16_supported else 'NOT supported'}")

    if bf16_supported:
        compute_dtype = torch.bfloat16
        use_bf16 = True
    else:
        compute_dtype = torch.float16
        use_bf16 = False

    # Step 1: Create RAFT-formatted data
    create_formatted_data(INPUT_FILES, FORMATTED_DATA)

    # Clear RAG models from memory before loading LLM
    global _embed_model, _collection
    del _embed_model, _collection
    _embed_model = None
    _collection = None
    torch.cuda.empty_cache()
    gc.collect()

    # Step 2: Load and prepare data
    print(f"\n Loading training data...")
    records = load_training_data(FORMATTED_DATA)

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    dataset = Dataset.from_list(records)

    # Split train/validation
    print(f"\n Splitting data: {int((1-VAL_RATIO)*100)}% train / {int(VAL_RATIO*100)}% val")
    split = dataset.train_test_split(test_size=VAL_RATIO, seed=RANDOM_SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"   Train: {len(train_dataset)}")
    print(f"   Eval:  {len(eval_dataset)}")

    # Step 3: Setup quantization
    print(f"\n Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Step 4: Load model
    print(f"\n Loading model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Step 5: Prepare for k-bit training
    print(f"\n Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Step 6: Add LoRA adapters
    print(f"\n Adding LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA})")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # Step 7: Format data for SFT
    print(f"\n Formatting data for training...")

    format_fn = partial(format_for_sft, tokenizer=tokenizer)
    train_dataset = train_dataset.map(format_fn, batched=True)
    eval_dataset = eval_dataset.map(format_fn, batched=True)

    print(f"\n Sample formatted text:")
    print("-" * 40)
    print(train_dataset[0]['text'][:800])
    print("-" * 40)

    # Step 8: Setup trainer
    print(f"\n Setting up trainer...")

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        completion_only_loss=True,  # Mask prompt tokens, train only on completion
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            MemoryCleanupCallback(cleanup_every_n_steps=50),
        ],
    )

    # Step 9: Train
    print(f"\n Starting RAFT training...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Effective batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   LoRA r: {LORA_R}, alpha: {LORA_ALPHA}")

    trainer.train()

    # Step 10: Final evaluation
    print(f"\n Final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"   Final eval_loss: {final_metrics['eval_loss']:.4f}")

    # Step 11: Save
    print(f"\n Saving RAFT LoRA adapter to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("RAFT Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print(f"   Format: Документы:\\n{{doc}}\\n\\nВопрос: ...\\nОтвет: ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
