#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_rag_v2.py

RAG inference with reasoning cards (no ChromaDB).

Pipeline:
1. User question → cosine find closest QA → get its DT
2. Filter cards by DT → rank by cosine → top 2 cards
3. Build system prompt with [Reasoning Card] Doc 1/2
4. Generate response with LoRA

Usage:
    python src/rag/query_rag_v2.py
    python src/rag/query_rag_v2.py --no-lora
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"

# Data files
CARDS_FILE = FINE_TUNING_DIR / "card_principals_deduped_v2_092.jsonl"
QA_FILE = FINE_TUNING_DIR / "qa_rs_segmented.jsonl"

# Models
BASE_MODEL = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
DEFAULT_LORA_PATH = PROJECT_ROOT / "models" / "labkovsky-reasoning-cards-v1"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

MAX_CARDS = 10

# DT mapping (QA DT -> Card DT vocabulary)
DT_MAPPING = {
    "ADDICTION_PATTERN": "AFFECTIVE_ADDICTION",
}

# System prompt (must match training format)
SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reasoning principles.\n\n"
    "Use these principles as the foundation for your response — "
    "apply them to the user's situation, but do not copy verbatim.\n\n"
    "{docs}\n\n"
    "Answer in Labkovsky's style: blunt, confident, with tough love if needed. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
)


# =============================================================================
# DATA LOADING
# =============================================================================

def iter_jsonl(path: Path):
    """Iterate over JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_cards(path: Path) -> List[Dict[str, Any]]:
    """Load reasoning cards with index."""
    cards = []
    for i, card in enumerate(iter_jsonl(path)):
        card["_idx"] = i
        cards.append(card)
    print(f"Loaded {len(cards)} reasoning cards")
    return cards


def load_qa(path: Path) -> List[Dict[str, Any]]:
    """Load QA records for DT lookup."""
    records = []
    for rec in iter_jsonl(path):
        if rec.get("question") and rec.get("decision_type"):
            records.append(rec)
    print(f"Loaded {len(records)} QA records for DT lookup")
    return records


# =============================================================================
# EMBEDDING & SIMILARITY
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_card_dts(card: Dict[str, Any]) -> set:
    """Get all DTs from a card (primary and first secondary only)."""
    dts = set()
    if card.get("dt_primary"):
        dts.add(card["dt_primary"])
    dt_sec = card.get("dt_secondary")
    if dt_sec:
        if isinstance(dt_sec, list) and len(dt_sec) > 0:
            dts.add(dt_sec[0])
        elif isinstance(dt_sec, str):
            dts.add(dt_sec)
    return dts


class ReasoningCardRetriever:
    """In-memory retriever for reasoning cards using cosine similarity."""

    def __init__(self, cards: List[Dict], qa_records: List[Dict], embed_model: SentenceTransformer):
        self.cards = cards
        self.qa_records = qa_records
        self.embed_model = embed_model

        # Pre-compute embeddings
        print("Computing card embeddings...")
        card_texts = [f"passage: {c.get('core_principle', '')}" for c in cards]
        self.card_embeddings = embed_model.encode(card_texts, normalize_embeddings=True)

        print("Computing QA embeddings...")
        qa_texts = [f"query: {r.get('question', '')}" for r in qa_records]
        self.qa_embeddings = embed_model.encode(qa_texts, normalize_embeddings=True)

        print(f"Ready: {len(cards)} cards, {len(qa_records)} QA records")

    def find_dt_from_closest_qa(self, question: str) -> set:
        """Find closest QA and return its DT(s)."""
        q_emb = self.embed_model.encode(f"query: {question}", normalize_embeddings=True)

        # Find closest QA
        similarities = [cosine_similarity(q_emb, qa_emb) for qa_emb in self.qa_embeddings]
        best_idx = np.argmax(similarities)
        best_qa = self.qa_records[best_idx]
        best_sim = similarities[best_idx]

        # Get DTs from closest QA
        dts = set()
        if best_qa.get("decision_type"):
            dts.add(best_qa["decision_type"])
        if best_qa.get("dt_primary"):
            dts.add(best_qa["dt_primary"])
        if best_qa.get("dt_secondary"):
            dts.add(best_qa["dt_secondary"])

        # Apply DT mapping
        mapped_dts = set()
        for dt in dts:
            mapped_dts.add(dt)
            if dt in DT_MAPPING:
                mapped_dts.add(DT_MAPPING[dt])

        print(f"  Closest QA (sim={best_sim:.3f}): {best_qa.get('question', '')[:60]}...")
        print(f"  DTs: {mapped_dts}")

        return mapped_dts

    def select_cards(self, question: str, max_cards: int = MAX_CARDS) -> List[Dict]:
        """Select top cards by DT filter + cosine ranking."""
        # Step 1: Get DT from closest QA
        qa_dts = self.find_dt_from_closest_qa(question)

        # Step 2: Embed question for card ranking
        q_emb = self.embed_model.encode(f"query: {question}", normalize_embeddings=True)

        # Step 3: Filter cards by DT match
        dt_matched = []
        for i, card in enumerate(self.cards):
            card_dts = get_card_dts(card)
            if card_dts & qa_dts:
                sim = cosine_similarity(q_emb, self.card_embeddings[i])
                dt_matched.append((i, sim, "dt_match"))

        # Sort by similarity
        dt_matched.sort(key=lambda x: x[1], reverse=True)

        selected = []
        for idx, sim, match_type in dt_matched[:max_cards]:
            selected.append({
                "card": self.cards[idx],
                "similarity": sim,
                "match_type": match_type
            })

        # Step 4: Global fallback if needed
        if len(selected) < max_cards:
            selected_indices = {s["card"].get("_idx", i) for i, s in enumerate(selected)}

            remaining = []
            for i, card in enumerate(self.cards):
                if i not in selected_indices:
                    sim = cosine_similarity(q_emb, self.card_embeddings[i])
                    remaining.append((i, sim))

            remaining.sort(key=lambda x: x[1], reverse=True)

            for idx, sim in remaining[:max_cards - len(selected)]:
                selected.append({
                    "card": self.cards[idx],
                    "similarity": sim,
                    "match_type": "global_fallback"
                })

        return selected


def format_docs(selected_cards: List[Dict]) -> str:
    """Format selected cards as doc context."""
    parts = []
    for i, item in enumerate(selected_cards, 1):
        card = item["card"]
        principle = card.get("core_principle", "")
        chapter = card.get("chapter_title", "")

        # Include chapter title for context
        if chapter:
            parts.append(f"[Reasoning Card] Doc {i} ({chapter}): {principle}")
        else:
            parts.append(f"[Reasoning Card] Doc {i}: {principle}")

    return "\n\n".join(parts)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(lora_path: Optional[Path] = None):
    """Load base model with optional LoRA adapter (4-bit quantized)."""
    print(f"\nLoading base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # 4-bit quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Offload dir for when model doesn't fit in VRAM
    offload_dir = PROJECT_ROOT / "offload_cache"
    offload_dir.mkdir(exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder=str(offload_dir),
        trust_remote_code=True,
    )

    if lora_path and lora_path.exists():
        print(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(
            model,
            str(lora_path),
            offload_folder=str(offload_dir),
            is_trainable=False,
        )
        # Ensure adapter is active
        model.set_adapter("default")
        print(f"LoRA loaded, active adapters: {model.active_adapters}")
    else:
        print("No LoRA adapter (baseline mode)")

    model.eval()
    return model, tokenizer


# =============================================================================
# GENERATION
# =============================================================================

def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate response using chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG query with reasoning cards (v2)")
    parser.add_argument("--lora-path", type=str, default=str(DEFAULT_LORA_PATH),
                        help="Path to LoRA adapter")
    parser.add_argument("--no-lora", action="store_true",
                        help="Run without LoRA (baseline)")
    parser.add_argument("--question", "-q", type=str,
                        help="Single question to answer (non-interactive)")
    args = parser.parse_args()

    # Load embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load data
    cards = load_cards(CARDS_FILE)
    qa_records = load_qa(QA_FILE)

    # Create retriever
    retriever = ReasoningCardRetriever(cards, qa_records, embed_model)

    # Load LLM
    lora_path = None if args.no_lora else Path(args.lora_path)
    model, tokenizer = load_model(lora_path)

    print("\n" + "=" * 60)
    print("RAG v2: Reasoning Cards + In-Memory Cosine Search")
    print("=" * 60)

    if args.question:
        # Single question mode
        questions = [args.question]
    else:
        # Interactive mode
        questions = None

    while True:
        if questions:
            question = questions.pop(0)
            print(f"\nQuestion: {question}")
        else:
            print("\n" + "-" * 40)
            question = input("Question (or 'quit'): ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

        # Retrieve cards
        print("\nRetrieving reasoning cards...")
        selected = retriever.select_cards(question)

        print(f"\nSelected {len(selected)} cards:")
        for i, item in enumerate(selected, 1):
            card = item["card"]
            print(f"  {i}. card_{card.get('_idx')} [{item['match_type']}] (sim={item['similarity']:.3f})")
            print(f"     Chapter: {card.get('chapter_title', 'N/A')}")
            print(f"     DT: {card.get('dt_primary', 'N/A')}")
            print(f"     Principle: {card.get('core_principle', '')[:100]}...")

        # Build prompt
        docs_text = format_docs(selected)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)

        # Generate
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, question, system_prompt)

        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response)

        if questions is not None and not questions:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
