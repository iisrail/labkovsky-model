#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build training data with reasoning cards as RAG context (v2).

Instead of book/article chunks + QA examples, uses condensed reasoning cards
from card_merged.jsonl, matched by decision_type and ranked by cosine similarity.

Selection order:
1. Strict DT filter (card.dt_primary or dt_secondary matches QA dt)
2. Cosine ranking inside filtered cards
3. Global fallback only when fewer than 2 DT-matching cards

Usage:
    python src/fine_tuning/build_rag_training_data_v2.py
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"

# Input files
INPUT_FILES = [
    FINE_TUNING_DIR / "qa_clean.jsonl",
    FINE_TUNING_DIR / "anti_generic_finance.jsonl",
]
CARDS_FILE = FINE_TUNING_DIR / "card_merged.jsonl"

# DT lookup from segmented QA (for records without dt_primary)
DT_LOOKUP_FILE = FINE_TUNING_DIR / "qa_rs_segmented.jsonl"

# Output
OUTPUT_FILE = FINE_TUNING_DIR / "qa_with_rag_context_v2.jsonl"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
MAX_CARDS = 2  # Up to 2 reasoning cards per QA

# DT mapping: QA DT -> Card DT (handle naming mismatches)
DT_MAPPING = {
    "ADDICTION_PATTERN": "AFFECTIVE_ADDICTION",  # Same concept, different name
}

# System prompt template (no book/article or QA example references)
SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reasoning principles.\n\n"
    "Use these principles as the foundation for your response — "
    "apply them to the user's situation, but do not copy verbatim.\n\n"
    "{docs}\n\n"
    "Answer in Labkovsky's style: blunt, confident, with tough love if needed. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
)


def iter_jsonl(path: Path):
    """Iterate over JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_cards(path: Path) -> List[Dict[str, Any]]:
    """Load reasoning cards from JSONL."""
    cards = []
    for i, card in enumerate(iter_jsonl(path)):
        card["_idx"] = i  # Add index for ID
        cards.append(card)
    return cards


def load_dt_lookup(path: Path) -> dict:
    """Load decision_type lookup from qa_rs_segmented.jsonl."""
    lookup = {}
    if not path.exists():
        return lookup
    for rec in iter_jsonl(path):
        rec_id = rec.get("id", "")
        dt = rec.get("decision_type", "")
        dt_secondary = rec.get("dt_secondary")
        if rec_id:
            lookup[rec_id] = {"dt_primary": dt, "dt_secondary": dt_secondary}
    return lookup


def get_card_dts(card: Dict[str, Any]) -> set:
    """Get all DTs from a card (primary and secondary)."""
    dts = set()
    if card.get("dt_primary"):
        dts.add(card["dt_primary"])
    if card.get("dt_secondary"):
        dts.add(card["dt_secondary"])
    return dts


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def select_cards(
    qa_dts: set,
    qa_embedding: np.ndarray,
    cards: List[Dict[str, Any]],
    card_embeddings: np.ndarray,
    max_cards: int = 2
) -> List[Dict[str, Any]]:
    """
    Select up to max_cards reasoning cards for a QA record.

    Selection order:
    1. Strict DT filter (card.dt_primary or dt_secondary matches any QA dt)
    2. Cosine ranking inside filtered cards
    3. Global fallback only when fewer than max_cards DT-matching cards
    """
    selected = []

    # Step 1: Filter cards by DT match
    dt_matched_indices = []
    for i, card in enumerate(cards):
        card_dts = get_card_dts(card)
        if card_dts & qa_dts:  # Intersection - any DT matches
            dt_matched_indices.append(i)

    # Step 2: Rank DT-matched cards by cosine similarity
    if dt_matched_indices:
        similarities = [
            (i, cosine_similarity(qa_embedding, card_embeddings[i]))
            for i in dt_matched_indices
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        for idx, sim in similarities[:max_cards]:
            selected.append({
                "card": cards[idx],
                "similarity": sim,
                "match_type": "dt_match"
            })

    # Step 3: Global fallback if fewer than max_cards
    if len(selected) < max_cards:
        # Get indices already selected
        selected_indices = {s["card"]["_idx"] for s in selected}

        # Rank all remaining cards by cosine similarity
        remaining = [
            (i, cosine_similarity(qa_embedding, card_embeddings[i]))
            for i in range(len(cards))
            if i not in selected_indices
        ]
        remaining.sort(key=lambda x: x[1], reverse=True)

        needed = max_cards - len(selected)
        for idx, sim in remaining[:needed]:
            selected.append({
                "card": cards[idx],
                "similarity": sim,
                "match_type": "global_fallback"
            })

    return selected


def format_docs(selected_cards: List[Dict[str, Any]]) -> str:
    """Format selected cards as doc context."""
    parts = []
    for i, item in enumerate(selected_cards, 1):
        card = item["card"]
        principle = card.get("core_principle", "")
        parts.append(f"[Reasoning Card] Doc {i}: {principle}")
    return "\n\n".join(parts)


def validate_output(records: List[Dict[str, Any]]) -> bool:
    """Validate no forbidden patterns in output."""
    errors = []

    for i, rec in enumerate(records):
        prompt = rec.get("system_prompt", "")

        if "[Book/Article]" in prompt:
            errors.append(f"Record {i}: contains '[Book/Article]'")

        if "[QA Example]" in prompt:
            errors.append(f"Record {i}: contains '[QA Example]'")

    if errors:
        print("\n=== VALIDATION ERRORS ===")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False

    return True


def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load reasoning cards
    print(f"Loading reasoning cards from {CARDS_FILE.name}...")
    cards = load_cards(CARDS_FILE)
    print(f"  Loaded {len(cards)} cards")

    # Pre-compute card embeddings (embed core_principle)
    print("Computing card embeddings...")
    card_texts = [f"passage: {c.get('core_principle', '')}" for c in cards]
    card_embeddings = embed_model.encode(card_texts, normalize_embeddings=True)
    print(f"  Embedded {len(card_embeddings)} cards")

    # Load DT lookup
    dt_lookup = load_dt_lookup(DT_LOOKUP_FILE)
    print(f"  DT lookup: {len(dt_lookup)} records")

    # Load QA records
    records = []
    for input_file in INPUT_FILES:
        if not input_file.exists():
            print(f"  WARNING: {input_file.name} not found, skipping")
            continue
        file_records = list(iter_jsonl(input_file))
        print(f"  {input_file.name}: {len(file_records)} records")
        records.extend(file_records)
    print(f"Loaded {len(records)} total QA records")

    output_records = []
    dt_match_count = 0
    fallback_count = 0

    for i, rec in enumerate(records):
        rec_id = rec.get("id", "")
        question = rec.get("question", "")
        answer = rec.get("answer", "")

        if not question or not answer:
            continue

        # Get QA decision types
        qa_dts = set()

        # From record itself
        if rec.get("dt_primary"):
            qa_dts.add(rec["dt_primary"])
        if rec.get("dt_secondary"):
            qa_dts.add(rec["dt_secondary"])
        if rec.get("decision_type"):
            qa_dts.add(rec["decision_type"])

        # From lookup
        if rec_id in dt_lookup:
            lookup_data = dt_lookup[rec_id]
            if lookup_data.get("dt_primary"):
                qa_dts.add(lookup_data["dt_primary"])
            if lookup_data.get("dt_secondary"):
                qa_dts.add(lookup_data["dt_secondary"])

        # Apply DT mapping (QA DT -> Card DT)
        mapped_dts = set()
        for dt in qa_dts:
            mapped_dts.add(dt)
            if dt in DT_MAPPING:
                mapped_dts.add(DT_MAPPING[dt])
        qa_dts = mapped_dts

        # Embed question
        qa_embedding = embed_model.encode(
            f"query: {question}",
            normalize_embeddings=True
        )

        # Select cards
        selected = select_cards(
            qa_dts=qa_dts,
            qa_embedding=qa_embedding,
            cards=cards,
            card_embeddings=card_embeddings,
            max_cards=MAX_CARDS
        )

        # Track match types
        has_dt_match = any(s["match_type"] == "dt_match" for s in selected)
        has_fallback = any(s["match_type"] == "global_fallback" for s in selected)
        if has_dt_match:
            dt_match_count += 1
        if has_fallback:
            fallback_count += 1

        # Format docs
        docs_text = format_docs(selected)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)

        # Build output record
        output_records.append({
            "id": rec_id,
            "question": question,
            "answer": answer,
            "system_prompt": system_prompt,
            "num_docs": len(selected),
            "doc_ids": [f"card_{s['card']['_idx']}" for s in selected],
            "decision_type": list(qa_dts)[0] if qa_dts else "",
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    # Validate
    print("\nValidating output...")
    valid = validate_output(output_records)

    if not valid:
        print("WARNING: Validation failed!")
    else:
        print("All records pass validation")

    # Write output
    print(f"\nWriting to {OUTPUT_FILE.name}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(output_records)} records -> {OUTPUT_FILE.name}")
    print(f"  Records with DT-matched cards: {dt_match_count}")
    print(f"  Records with fallback cards: {fallback_count}")

    # Show example
    if output_records:
        ex = output_records[0]
        print(f"\n--- Example ---")
        print(f"ID: {ex['id']}")
        print(f"DT: {ex['decision_type']}")
        print(f"Q: {ex['question'][:100]}...")
        print(f"Docs: {ex['num_docs']} ({ex['doc_ids']})")
        print(f"System prompt (first 600):\n{ex['system_prompt'][:600]}...")
        print(f"A: {ex['answer'][:200]}...")


if __name__ == "__main__":
    main()
