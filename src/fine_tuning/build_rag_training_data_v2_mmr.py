#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build training data with reasoning cards as RAG context (v4 fixed).

Uses condensed reasoning cards from card_principals_deduped_v2_092.jsonl.

Selection order:
1. Read card DT from dt / dt_primary / decision_type / dt_secondary.
2. Prefer exact QA DT matches.
3. Add alias DT matches only if exact matches are insufficient.
4. Use question + answer embedding and MMR diversity.
5. Do not pull global fallback cards by default, because noisy context is worse than fewer cards.

Usage:
    python src/fine_tuning/build_rag_training_data_v4_fixed.py
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
CARDS_FILE = FINE_TUNING_DIR / "card_principals_deduped_v2_092.jsonl"

# DT lookup from segmented QA (for records without dt_primary)
DT_LOOKUP_FILE = FINE_TUNING_DIR / "qa_rs_segmented.jsonl"

# Output
OUTPUT_FILE = FINE_TUNING_DIR / "qa_with_rag_context_v4_fixed.jsonl"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
MAX_CARDS = 3  # Keep context sharp: top 3 reasoning cards per QA

# Optional DT aliases for matching only.
# Important: aliases are NOT written back as the QA decision_type.
# Exact DT matches are always preferred over aliases.
DT_ALIASES = {
    "ADDICTION_PATTERN": ["AFFECTIVE_ADDICTION"],
}

# Global fallback often injects weak unrelated cards. Keep it off for training quality.
ALLOW_GLOBAL_FALLBACK = False

# System prompt template (no book/article or QA example references)
SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reasoning principles.\n\n"
    "Use these principles as the foundation for your response.\n\n"
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


def _add_dt_value(dts: set, value: Any) -> None:
    """Add DT value(s) to a set, handling strings and lists."""
    if not value:
        return
    if isinstance(value, list):
        for item in value:
            if item:
                dts.add(str(item))
    else:
        dts.add(str(value))


def get_card_dts(card: Dict[str, Any]) -> set:
    """Get all DTs from a card.

    Your card files may use different schemas:
    - dt
    - dt_primary
    - decision_type
    - dt_secondary (string or list)

    The previous version ignored `dt`, which caused QA-derived cards like
    {"dt": "ADDICTION_PATTERN", ...} to be skipped during DT filtering.
    """
    dts = set()
    for key in ("dt", "dt_primary", "decision_type", "dt_secondary"):
        _add_dt_value(dts, card.get(key))
    return dts


def get_record_dts(rec: Dict[str, Any], lookup_data: Optional[Dict[str, Any]] = None) -> set:
    """Get all DTs from a QA record + optional lookup record."""
    dts = set()
    for key in ("dt", "dt_primary", "decision_type", "dt_secondary"):
        _add_dt_value(dts, rec.get(key))
    if lookup_data:
        for key in ("dt", "dt_primary", "decision_type", "dt_secondary"):
            _add_dt_value(dts, lookup_data.get(key))
    return dts


def get_original_decision_type(rec: Dict[str, Any], lookup_data: Optional[Dict[str, Any]] = None) -> str:
    """Preserve the QA's original DT for output/debug. Do not output aliases."""
    for source in (rec, lookup_data or {}):
        for key in ("decision_type", "dt", "dt_primary"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def expand_dt_aliases(dts: set) -> set:
    """Return alias DTs for matching only."""
    aliases = set()
    for dt in dts:
        for alias in DT_ALIASES.get(dt, []):
            aliases.add(alias)
    return aliases


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def select_cards(
    qa_dts: set,
    qa_alias_dts: set,
    qa_embedding: np.ndarray,
    cards: List[Dict[str, Any]],
    card_embeddings: np.ndarray,
    max_cards: int = 3,
    relevance_weight: float = 0.75,
    redundancy_weight: float = 0.25,
    allow_global_fallback: bool = ALLOW_GLOBAL_FALLBACK,
) -> List[Dict[str, Any]]:
    """Select reasoning cards with exact-DT priority + optional alias + MMR diversity.

    Important behavior:
    - Exact QA DT cards are preferred.
    - Alias DT cards are added only if fewer than max_cards exact matches exist.
    - Global fallback is disabled by default; fewer relevant cards are better than noisy cards.
    """
    if max_cards <= 0:
        return []

    exact_indices = []
    alias_indices = []

    for i, card in enumerate(cards):
        card_dts = get_card_dts(card)
        if card_dts & qa_dts:
            exact_indices.append(i)
        elif card_dts & qa_alias_dts:
            alias_indices.append(i)

    if exact_indices:
        candidate_indices = exact_indices + alias_indices
    elif alias_indices:
        candidate_indices = alias_indices
    elif allow_global_fallback:
        candidate_indices = list(range(len(cards)))
    else:
        candidate_indices = []

    if not candidate_indices:
        return []

    relevance = {i: cosine_similarity(qa_embedding, card_embeddings[i]) for i in candidate_indices}
    selected_indices = []
    selected_scores = {}

    # MMR selection
    while len(selected_indices) < max_cards and len(selected_indices) < len(candidate_indices):
        best_idx = None
        best_score = -1e9

        for idx in candidate_indices:
            if idx in selected_indices:
                continue

            if selected_indices:
                redundancy = max(cosine_similarity(card_embeddings[idx], card_embeddings[j]) for j in selected_indices)
            else:
                redundancy = 0.0

            score = relevance_weight * relevance[idx] - redundancy_weight * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        selected_indices.append(best_idx)
        selected_scores[best_idx] = best_score

    selected = []
    for idx in selected_indices:
        if idx in exact_indices:
            match_type = "dt_exact"
        elif idx in alias_indices:
            match_type = "dt_alias"
        else:
            match_type = "global_fallback"

        selected.append({
            "card": cards[idx],
            "similarity": relevance[idx],
            "mmr_score": selected_scores[idx],
            "match_type": match_type,
        })

    return selected


def format_docs(selected_cards: List[Dict[str, Any]]) -> str:
    """Format selected cards as doc context."""
    parts = []
    for i, item in enumerate(selected_cards, 1):
        card = item["card"]
        principle = card.get("core_principle", "")
        parts.append(f"Doc {i}: {principle}")
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

    # Pre-compute card embeddings. Include DT lightly for better clustering/debug.
    print("Computing card embeddings...")
    card_texts = [
        f"passage: DT: {', '.join(sorted(get_card_dts(c)))}\nCORE: {c.get('core_principle', '')}"
        for c in cards
    ]
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

        # Get QA decision types. Preserve original DT separately from aliases.
        lookup_data = dt_lookup.get(rec_id, {})
        qa_dts = get_record_dts(rec, lookup_data)
        qa_alias_dts = expand_dt_aliases(qa_dts)
        original_decision_type = get_original_decision_type(rec, lookup_data)

        # Embed question + answer, because the mechanism is often explicit in the answer
        qa_text_for_retrieval = f"{question}\n{answer}"
        qa_embedding = embed_model.encode(
            f"query: {qa_text_for_retrieval}",
            normalize_embeddings=True
        )

        # Select cards
        selected = select_cards(
            qa_dts=qa_dts,
            qa_alias_dts=qa_alias_dts,
            qa_embedding=qa_embedding,
            cards=cards,
            card_embeddings=card_embeddings,
            max_cards=MAX_CARDS,
            allow_global_fallback=ALLOW_GLOBAL_FALLBACK,
        )

        # Track match types
        has_dt_match = any(s["match_type"] in {"dt_exact", "dt_alias"} for s in selected)
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
            "doc_match_types": [s["match_type"] for s in selected],
            "doc_similarities": [round(float(s["similarity"]), 4) for s in selected],
            "decision_type": original_decision_type,
            "matching_dts": sorted(qa_dts),
            "alias_dts": sorted(qa_alias_dts),
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
