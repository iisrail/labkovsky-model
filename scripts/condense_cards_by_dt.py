#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Condense reasoning cards by grouping by decision_type.

Input: card_principals.jsonl (book + QA principles)
Output: card_merged_by_dt.jsonl

Groups principles by DT, deduplicates, and merges into 300-450 token cards.
"""

import json
import re
import sys
import io
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
INPUT_FILE = Path("data/fine_tuning/card_principals.jsonl")
OUTPUT_FILE = Path("data/fine_tuning/card_merged_by_dt.jsonl")

# Token limits
MIN_TOKENS = 200
MAX_TOKENS = 450


def estimate_tokens(text: str) -> int:
    """Rough token estimation (Russian: ~1.5 chars per token)."""
    return len(text) // 4 + len(text.split())


def load_cards(filepath: Path) -> List[Dict[str, Any]]:
    """Load cards from JSONL."""
    cards = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cards.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cards


def normalize_principle(text: str) -> str:
    """Normalize text for deduplication."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[«»""„‟]', '"', text)
    return text


def get_signature(text: str) -> str:
    """Get first N chars as dedup signature."""
    norm = normalize_principle(text)
    return norm[:60]


def deduplicate_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove near-duplicate cards."""
    seen = set()
    unique = []

    for card in cards:
        sig = get_signature(card.get('core_principle', ''))
        if sig not in seen:
            seen.add(sig)
            unique.append(card)

    return unique


def group_by_dt(cards: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group cards by decision_type."""
    groups = defaultdict(list)
    for card in cards:
        dt = card.get('dt', 'EXPLANATION')
        groups[dt].append(card)
    return groups


def merge_principles(cards: List[Dict[str, Any]], max_tokens: int = MAX_TOKENS) -> List[str]:
    """Merge principles into chunks under max_tokens."""
    principles = [c.get('core_principle', '').strip() for c in cards]
    principles = [p for p in principles if p]

    if not principles:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for p in principles:
        p_tokens = estimate_tokens(p)

        if current_tokens + p_tokens <= max_tokens:
            current_chunk.append(p)
            current_tokens += p_tokens
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [p]
            current_tokens = p_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def build_merged_card(dt: str, principle: str, source_count: int) -> Dict[str, Any]:
    """Build a merged card."""
    return {
        "source": "reasoning_card_merged",
        "dt_primary": dt,
        "dt_secondary": None,
        "core_principle": principle,
        "token_estimate": estimate_tokens(principle),
        "merged_from_count": source_count,
    }


def main():
    print(f"Loading cards from {INPUT_FILE}...")
    all_cards = load_cards(INPUT_FILE)
    print(f"Loaded {len(all_cards)} cards")

    # Deduplicate
    print("\nDeduplicating...")
    unique_cards = deduplicate_cards(all_cards)
    print(f"After dedup: {len(unique_cards)} cards (removed {len(all_cards) - len(unique_cards)})")

    # Group by DT
    print("\nGrouping by DT...")
    dt_groups = group_by_dt(unique_cards)
    print(f"Found {len(dt_groups)} decision types")

    for dt, cards in sorted(dt_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {dt}: {len(cards)} cards")

    # Merge within each DT
    print(f"\nMerging (target: {MIN_TOKENS}-{MAX_TOKENS} tokens)...")
    merged_cards = []

    for dt, cards in dt_groups.items():
        chunks = merge_principles(cards, MAX_TOKENS)

        for chunk in chunks:
            # Count how many source cards went into this chunk
            chunk_lower = chunk.lower()
            source_count = sum(1 for c in cards if c.get('core_principle', '').lower()[:50] in chunk_lower)
            source_count = max(1, source_count)  # At least 1

            merged_card = build_merged_card(dt, chunk, source_count)
            merged_cards.append(merged_card)

    print(f"Created {len(merged_cards)} merged cards")

    # Stats by DT
    print("\nMerged cards by DT:")
    merged_by_dt = defaultdict(list)
    for card in merged_cards:
        merged_by_dt[card['dt_primary']].append(card)

    for dt, cards in sorted(merged_by_dt.items(), key=lambda x: -len(x[1])):
        avg_tokens = sum(c['token_estimate'] for c in cards) / len(cards)
        print(f"  {dt}: {len(cards)} cards (avg {avg_tokens:.0f} tokens)")

    # Validate
    print("\nValidating...")
    over_limit = [c for c in merged_cards if c['token_estimate'] > MAX_TOKENS]
    if over_limit:
        print(f"  WARNING: {len(over_limit)} cards over {MAX_TOKENS} tokens")
    else:
        print(f"  All cards within {MAX_TOKENS} token limit")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for card in merged_cards:
            f.write(json.dumps(card, ensure_ascii=False) + '\n')

    # Final stats
    token_counts = [c['token_estimate'] for c in merged_cards]
    print(f"\n=== FINAL STATS ===")
    print(f"Total cards: {len(merged_cards)}")
    print(f"Token range: {min(token_counts)} - {max(token_counts)}")
    print(f"Avg tokens: {sum(token_counts) / len(token_counts):.0f}")

    # Sample
    print("\n=== SAMPLE ===")
    for card in merged_cards[:2]:
        print(f"\n[{card['dt_primary']}] ({card['token_estimate']} tokens)")
        print(f"  {card['core_principle'][:200]}...")


if __name__ == "__main__":
    main()
