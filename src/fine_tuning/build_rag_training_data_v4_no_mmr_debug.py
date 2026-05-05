#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build training data with reasoning cards as RAG context (v4 no-MMR debug).

Uses condensed reasoning cards from card_principals_deduped_v2_092.jsonl.

Selection order:
1. Read card DT from dt / dt_primary / decision_type / dt_secondary.
2. Build candidates from exact QA DT matches, plus alias DT matches.
3. Score candidates by cosine similarity, with +0.02 for exact QA DT.
4. Add a small bounded lexical boost for shared case-specific terms.
5. Do not pull global fallback cards by default, because noisy context is worse than fewer cards.

Usage:
    python src/fine_tuning/build_rag_training_data_v4_no_mmr_debug.py
"""

import json
import os
import re
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
EXACT_DT_BOOST = 0.02  # Prefer exact QA DT over alias DT when similarities are close
LEXICAL_MATCH_BOOST = 0.012  # Per case-specific token shared by QA and card
MAX_LEXICAL_BOOST = 0.06  # Keep lexical matching as a tie-breaker, not a hard rule

STOP_TERMS = {
    "без", "был", "была", "были", "быть", "вам", "вас", "ваш", "все",
    "для", "его", "если", "еще", "или", "как", "мне", "мой", "над",
    "нас", "нее", "нет", "они", "она", "оно", "при", "про", "сам",
    "себ", "так", "там", "тем", "тех", "тог", "тут", "уже", "что",
    "это", "этот", "your", "with", "that", "this", "from", "have",
    "должен", "есть", "которы", "люди", "челове", "пробле",
}

# Optional DT aliases for matching only.
# Important: aliases are NOT written back as the QA decision_type.
# Exact DT matches receive EXACT_DT_BOOST during ranking.
DT_ALIASES = {
    "ADDICTION_PATTERN": ["AFFECTIVE_ADDICTION"],
}

# Global fallback often injects weak unrelated cards. Keep it off for training quality.
ALLOW_GLOBAL_FALLBACK = False

# Debug controls: override DEBUG_QA_ID / DEBUG_GOLD_TEXT env vars to inspect one case.
DEBUG_QA_ID = os.getenv("DEBUG_QA_ID", "srJvn19GKNA_01")
DEBUG_GOLD_TEXT = os.getenv("DEBUG_GOLD_TEXT", "В основе многих зависимостей лежит тревога")
DEBUG_TOP_N = 15

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


def tokenize_for_overlap(text: str) -> set:
    """Return coarse Russian/English term keys for lexical tie-breaking."""
    terms = set()
    for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower().replace("ё", "е")):
        if len(token) < 4 or token in STOP_TERMS:
            continue
        key = token[:6] if len(token) > 6 else token
        if key not in STOP_TERMS:
            terms.add(key)
    return terms


def lexical_overlap_boost(qa_terms: set, card_text: str) -> tuple:
    """Score shared case-specific terms between QA text and a card principle."""
    card_terms = tokenize_for_overlap(card_text)
    overlap = qa_terms & card_terms
    boost = min(MAX_LEXICAL_BOOST, LEXICAL_MATCH_BOOST * len(overlap))
    return boost, sorted(overlap)


def select_cards(
    qa_dts: set,
    qa_alias_dts: set,
    qa_text: str,
    qa_embedding: np.ndarray,
    cards: List[Dict[str, Any]],
    card_embeddings: np.ndarray,
    max_cards: int = 3,
    exact_dt_boost: float = EXACT_DT_BOOST,
    allow_global_fallback: bool = ALLOW_GLOBAL_FALLBACK,
) -> List[Dict[str, Any]]:
    """Select top cards by cosine ranking plus bounded tie-break boosts.

    Behavior:
    - For training context, keep the strongest mechanism cards by score, even
      when selected cards are semantically close.
    - Exact QA DT cards and alias cards can both enter the candidate pool.
    - Exact DT cards receive EXACT_DT_BOOST, so close similarities prefer
      the original QA DT without rewriting the output decision_type.
    - Cards sharing case-specific words with the QA receive a small bounded
      lexical boost, which helps avoid generic same-DT relationship cards.
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

    qa_terms = tokenize_for_overlap(qa_text)
    scored = []
    for idx in candidate_indices:
        card_text = cards[idx].get("core_principle", "")
        sim = cosine_similarity(qa_embedding, card_embeddings[idx])
        lexical_boost, overlap_terms = lexical_overlap_boost(qa_terms, card_text)
        if idx in exact_indices:
            match_type = "dt_exact"
            final_score = sim + exact_dt_boost + lexical_boost
        elif idx in alias_indices:
            match_type = "dt_alias"
            final_score = sim + lexical_boost
        else:
            match_type = "global_fallback"
            final_score = sim + lexical_boost

        scored.append((idx, sim, final_score, match_type, lexical_boost, overlap_terms))

    scored.sort(key=lambda x: x[2], reverse=True)

    selected = []
    for idx, sim, final_score, match_type, lexical_boost, overlap_terms in scored[:max_cards]:
        selected.append({
            "card": cards[idx],
            "similarity": sim,
            "selection_score": final_score,
            "lexical_boost": lexical_boost,
            "overlap_terms": overlap_terms,
            "match_type": match_type,
        })

    return selected

def debug_find_gold_card(cards: List[Dict[str, Any]]) -> List[int]:
    """Print cards containing DEBUG_GOLD_TEXT and return their zero-based indices."""
    if not DEBUG_GOLD_TEXT:
        return []

    matches = []
    print("\n=== DEBUG: GOLD CARD SEARCH ===")
    for card in cards:
        text = card.get("core_principle", "")
        if DEBUG_GOLD_TEXT in text:
            idx = card.get("_idx")
            matches.append(idx)
            print(f"GOLD FOUND: card_{idx}")
            print(f"  DTs: {sorted(get_card_dts(card))}")
            print(f"  Text: {text}")

    if not matches:
        print(f"GOLD NOT FOUND by text: {DEBUG_GOLD_TEXT!r}")
    print("=== END DEBUG: GOLD CARD SEARCH ===\n")
    return matches


def debug_rank_cards_for_qa(
    rec_id: str,
    question: str,
    qa_text: str,
    qa_dts: set,
    qa_alias_dts: set,
    qa_embedding: np.ndarray,
    cards: List[Dict[str, Any]],
    card_embeddings: np.ndarray,
    gold_indices: List[int],
) -> None:
    """Print exact/alias candidate counts and raw cosine ranks before final selection."""
    if rec_id != DEBUG_QA_ID:
        return

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
        pool_type = "exact + alias"
    elif alias_indices:
        candidate_indices = alias_indices
        pool_type = "alias only"
    elif ALLOW_GLOBAL_FALLBACK:
        candidate_indices = list(range(len(cards)))
        pool_type = "global fallback"
    else:
        candidate_indices = []
        pool_type = "empty"

    qa_terms = tokenize_for_overlap(qa_text)
    scored = []
    for idx in candidate_indices:
        sim = cosine_similarity(qa_embedding, card_embeddings[idx])
        lexical_boost, overlap_terms = lexical_overlap_boost(
            qa_terms,
            cards[idx].get("core_principle", ""),
        )
        dt_boost = EXACT_DT_BOOST if idx in exact_indices else 0
        final_score = sim + dt_boost + lexical_boost
        scored.append((idx, sim, final_score, lexical_boost, overlap_terms, cards[idx]))
    scored.sort(key=lambda x: x[2], reverse=True)

    print("\n=== DEBUG: QA CARD RANKING ===")
    print(f"QA ID: {rec_id}")
    print(f"QA DTs: {sorted(qa_dts)}")
    print(f"Alias DTs: {sorted(qa_alias_dts)}")
    print(f"Question: {question[:220]}")
    print(f"Pool: {pool_type}")
    print(f"Exact candidates: {len(exact_indices)}")
    print(f"Alias candidates: {len(alias_indices)}")
    print(f"Total candidates used: {len(candidate_indices)}")

    for gold_idx in gold_indices:
        gold_card = cards[gold_idx]
        in_exact = gold_idx in exact_indices
        in_alias = gold_idx in alias_indices
        in_candidates = gold_idx in candidate_indices
        rank = next((pos + 1 for pos, (idx, _, _, _, _, _) in enumerate(scored) if idx == gold_idx), None)
        sim = next((score for idx, score, _, _, _, _ in scored if idx == gold_idx), None)
        print("\nGold card status:")
        print(f"  card_{gold_idx}")
        print(f"  DTs: {sorted(get_card_dts(gold_card))}")
        print(f"  in_exact={in_exact}, in_alias={in_alias}, in_candidates={in_candidates}")
        print(f"  final_rank={rank}, raw_similarity={None if sim is None else round(float(sim), 4)}")

    print(f"\nTOP {DEBUG_TOP_N} BEFORE FINAL SELECTION:")
    for rank, (idx, sim, final_score, lexical_boost, overlap_terms, card) in enumerate(scored[:DEBUG_TOP_N], 1):
        text = card.get("core_principle", "")
        marker = "  <-- GOLD" if idx in gold_indices else ""
        print(
            f"{rank:02d}. card_{idx} sim={sim:.4f} "
            f"lex={lexical_boost:.4f} score={final_score:.4f} "
            f"dts={sorted(get_card_dts(card))}{marker}"
        )
        if overlap_terms:
            print(f"    overlap={overlap_terms}")
        print(f"    {text[:180]}")
    print("=== END DEBUG: QA CARD RANKING ===\n")


def debug_selected_cards(rec_id: str, selected_cards: List[Dict[str, Any]], gold_indices: List[int]) -> None:
    """Print final cards after raw cosine + exact-DT boost for one debug QA."""
    if rec_id != DEBUG_QA_ID:
        return

    print("\n=== DEBUG: FINAL SELECTED CARDS ===")
    for rank, item in enumerate(selected_cards, 1):
        card = item["card"]
        idx = card["_idx"]
        marker = "  <-- GOLD" if idx in gold_indices else ""
        print(
            f"{rank}. card_{idx} "
            f"match={item.get('match_type')} "
            f"sim={float(item.get('similarity', 0)):.4f} "
            f"lex={float(item.get('lexical_boost', 0)):.4f} "
            f"score={float(item.get('selection_score', 0)):.4f}" + marker
        )
        if item.get("overlap_terms"):
            print(f"   overlap={item.get('overlap_terms')}")
        print(f"   {card.get('core_principle', '')[:220]}")
    print("=== END DEBUG: FINAL SELECTED CARDS ===\n")

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

    gold_indices = debug_find_gold_card(cards)

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

        # Debug raw ranking before final selection
        debug_rank_cards_for_qa(
            rec_id=rec_id,
            question=question,
            qa_text=qa_text_for_retrieval,
            qa_dts=qa_dts,
            qa_alias_dts=qa_alias_dts,
            qa_embedding=qa_embedding,
            cards=cards,
            card_embeddings=card_embeddings,
            gold_indices=gold_indices,
        )

        # Select cards
        selected = select_cards(
            qa_dts=qa_dts,
            qa_alias_dts=qa_alias_dts,
            qa_text=qa_text_for_retrieval,
            qa_embedding=qa_embedding,
            cards=cards,
            card_embeddings=card_embeddings,
            max_cards=MAX_CARDS,
            allow_global_fallback=ALLOW_GLOBAL_FALLBACK,
        )

        debug_selected_cards(rec_id, selected, gold_indices)

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
            "doc_selection_scores": [round(float(s["selection_score"]), 4) for s in selected],
            "doc_lexical_boosts": [round(float(s["lexical_boost"]), 4) for s in selected],
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
