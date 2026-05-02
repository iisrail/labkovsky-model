#!/usr/bin/env python3
"""
Condense reasoning cards by merging related cards within chapters.

Input: card_principals.jsonl (~610 cards)
Output: card_merged.jsonl (1-4 cards per chapter, <=200 tokens each)

Rules:
- Merge cards within same chapter that share semantic theme
- Choose dt_primary (dominant) and dt_secondary (if real secondary theme exists)
- core_principle must be <= 200 tokens; if longer, split into two cards
- Don't force 4 cards per chapter; only create if distinct principle exists
"""

import json
import sys
import io
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
INPUT_FILE = Path("data/fine_tuning/card_principals.jsonl")
OUTPUT_FILE = Path("data/fine_tuning/card_merged.jsonl")

# Token limit
MAX_TOKENS = 500


def estimate_tokens(text: str) -> int:
    """Rough token estimation (Russian: ~1.5 chars per token average)."""
    return len(text) // 4 + len(text.split())


def load_cards(filepath: Path) -> List[Dict[str, Any]]:
    """Load cards from JSONL file, skip invalid lines."""
    cards = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                card = json.loads(line)
                cards.append(card)
            except json.JSONDecodeError:
                print(f"Skipping invalid line {line_num}: {line[:50]}...")
    return cards


def group_by_chapter(cards: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """Group cards by (book, chapter) tuple."""
    groups = defaultdict(list)
    for card in cards:
        key = (card.get('book', ''), card.get('chapter', ''))
        groups[key].append(card)
    return groups


def get_dt_primary_secondary(cards: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """
    Get dt_primary and dt_secondary from a list of cards.

    - dt_primary: most frequent/central dt
    - dt_secondary: second most frequent, only if genuinely different theme
    """
    dt_counts = Counter(card.get('dt', 'EXPLANATION') for card in cards)

    if len(dt_counts) == 0:
        return 'EXPLANATION', None

    sorted_dts = dt_counts.most_common()
    dt_primary = sorted_dts[0][0]

    # Only set secondary if there's a real second theme with meaningful count
    dt_secondary = None
    if len(sorted_dts) > 1:
        second_dt, second_count = sorted_dts[1]
        primary_count = sorted_dts[0][1]
        # Only include secondary if it represents at least 20% of cards
        # or has at least 2 cards
        if second_count >= 2 or (second_count / sum(dt_counts.values())) >= 0.2:
            dt_secondary = second_dt

    return dt_primary, dt_secondary


def split_text_by_sentences(text: str, max_tokens: int) -> List[str]:
    """
    Split text into chunks that fit within max_tokens.
    Tries to split at sentence boundaries.
    """
    # Split by common sentence endings
    import re
    sentences = re.split(r'(?<=[.;!?])\s+', text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            # Save current chunk if non-empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            # Start new chunk
            if sentence_tokens <= max_tokens:
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                # Sentence itself is too long, force split by words
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                for word in words:
                    word_tokens = estimate_tokens(word)
                    if temp_tokens + word_tokens <= max_tokens:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_tokens = word_tokens
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens

    # Don't forget last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks if chunks else [text[:500]]  # Fallback


def merge_principles(cards: List[Dict[str, Any]]) -> str:
    """
    Merge core_principles from cards into a single text.
    Removes redundant statements and combines unique ideas.
    """
    principles = [c.get('core_principle', '') for c in cards]

    # If only 1 card, return as-is
    if len(principles) == 1:
        return principles[0]

    # Deduplicate by key phrase
    seen_ideas = set()
    unique_parts = []

    for p in principles:
        # Use first 40 chars lowercase as signature
        key_phrase = p[:40].lower().strip()
        if key_phrase not in seen_ideas:
            seen_ideas.add(key_phrase)
            unique_parts.append(p)

    # Join with space (sentences already end with periods)
    merged = ' '.join(unique_parts)

    return merged


def cluster_cards_by_theme(cards: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Cluster cards into groups based on dt similarity.
    Don't force a minimum number of clusters.
    """
    # Group by dt
    dt_groups = defaultdict(list)
    for card in cards:
        dt = card.get('dt', 'EXPLANATION')
        dt_groups[dt].append(card)

    # If too many groups, merge smallest ones to max 4
    while len(dt_groups) > 4:
        sorted_dts = sorted(dt_groups.keys(), key=lambda x: len(dt_groups[x]))
        smallest, second_smallest = sorted_dts[0], sorted_dts[1]
        dt_groups[second_smallest].extend(dt_groups[smallest])
        del dt_groups[smallest]

    return list(dt_groups.values())


def condense_chapter(
    chapter_cards: List[Dict[str, Any]],
    book: str,
    chapter: str
) -> List[Dict[str, Any]]:
    """
    Condense all cards in a chapter into 1-4 merged cards.
    Each card must have core_principle <= 200 tokens.
    """
    # Get chapter metadata from first card
    sample = chapter_cards[0]
    chapter_title = sample.get('chapter_title', '')

    # Cluster cards into thematic groups
    clusters = cluster_cards_by_theme(chapter_cards)

    condensed = []
    for cluster in clusters:
        if not cluster:
            continue

        # Get dt_primary and dt_secondary
        dt_primary, dt_secondary = get_dt_primary_secondary(cluster)

        # Get rule if any card has it
        rules = [c.get('rule') for c in cluster if c.get('rule')]
        rule = rules[0] if rules else None

        # Merge principles
        merged_principle = merge_principles(cluster)
        merged_tokens = estimate_tokens(merged_principle)

        # Check token limit (use 480 threshold to account for estimation error)
        if merged_tokens <= 480:
            # Single card
            card = build_card(
                book=book,
                chapter=chapter,
                chapter_title=chapter_title,
                rule=rule,
                dt_primary=dt_primary,
                dt_secondary=dt_secondary,
                core_principle=merged_principle
            )
            condensed.append(card)
        else:
            # Split into multiple cards (use 480 threshold for safety margin)
            chunks = split_text_by_sentences(merged_principle, 480)
            for i, chunk in enumerate(chunks):
                card = build_card(
                    book=book,
                    chapter=chapter,
                    chapter_title=chapter_title,
                    rule=rule,
                    dt_primary=dt_primary,
                    dt_secondary=dt_secondary if i == 0 else None,  # Secondary only on first
                    core_principle=chunk
                )
                condensed.append(card)

    return condensed


def build_card(
    book: str,
    chapter: str,
    chapter_title: str,
    rule: Optional[str],
    dt_primary: str,
    dt_secondary: Optional[str],
    core_principle: str
) -> Dict[str, Any]:
    """Build a condensed card record."""
    card = {
        "source": "reasoning_card",
        "book": book,
        "chapter": chapter,
    }

    if chapter_title:
        card["chapter_title"] = chapter_title
    if rule:
        card["rule"] = rule

    card["dt_primary"] = dt_primary
    card["dt_secondary"] = dt_secondary
    card["core_principle"] = core_principle

    return card


def validate_output(cards: List[Dict[str, Any]]) -> bool:
    """Validate all cards meet requirements."""
    errors = []

    for i, card in enumerate(cards):
        # Check token limit
        tokens = estimate_tokens(card.get('core_principle', ''))
        if tokens > MAX_TOKENS:
            errors.append(f"Card {i}: {tokens} tokens > {MAX_TOKENS}")

        # Check no section_position field
        if 'section_position' in card:
            errors.append(f"Card {i}: has forbidden 'section_position' field")

        # Check no old 'dt' field
        if 'dt' in card:
            errors.append(f"Card {i}: has old 'dt' field (should be dt_primary/dt_secondary)")

        # Check dt_primary exists
        if not card.get('dt_primary'):
            errors.append(f"Card {i}: missing dt_primary")

        # Check dt_secondary is null or different from dt_primary
        dt_primary = card.get('dt_primary')
        dt_secondary = card.get('dt_secondary')
        if dt_secondary is not None and dt_secondary == dt_primary:
            errors.append(f"Card {i}: dt_secondary '{dt_secondary}' equals dt_primary")

    if errors:
        print("\n=== VALIDATION ERRORS ===")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False

    return True


def main():
    print(f"Loading cards from {INPUT_FILE}...")
    cards = load_cards(INPUT_FILE)
    print(f"Loaded {len(cards)} valid cards")

    print("\nGrouping by chapter...")
    chapter_groups = group_by_chapter(cards)
    print(f"Found {len(chapter_groups)} unique chapters")

    print("\nCondensing chapters...")
    all_condensed = []

    for (book, chapter), chapter_cards in sorted(chapter_groups.items()):
        condensed = condense_chapter(chapter_cards, book, chapter)
        all_condensed.extend(condensed)
        print(f"  {book} | Ch {chapter}: {len(chapter_cards)} -> {len(condensed)} cards")

    print(f"\nTotal: {len(cards)} -> {len(all_condensed)} cards")

    # Validate
    print("\nValidating output...")
    valid = validate_output(all_condensed)

    if not valid:
        print("\nWARNING: Validation failed, check errors above")
    else:
        print("All cards pass validation")

    # Save output
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for card in all_condensed:
            f.write(json.dumps(card, ensure_ascii=False) + '\n')

    # Stats
    print("\n=== OUTPUT STATS ===")
    token_counts = [estimate_tokens(c.get('core_principle', '')) for c in all_condensed]

    print(f"Cards: {len(all_condensed)}")
    print(f"Avg tokens: {sum(token_counts) / len(token_counts):.0f}")
    print(f"Min/Max tokens: {min(token_counts)} / {max(token_counts)}")

    # Token distribution
    ranges = [(0, 100), (100, 150), (150, 200), (200, 250)]
    print("\nToken distribution:")
    for lo, hi in ranges:
        count = sum(1 for t in token_counts if lo <= t < hi)
        pct = count / len(token_counts) * 100
        print(f"  {lo}-{hi}: {count} cards ({pct:.0f}%)")

    # Cards per chapter
    output_groups = group_by_chapter(all_condensed)
    dist = Counter(len(v) for v in output_groups.values())
    print("\nCards per chapter:")
    for n, count in sorted(dist.items()):
        print(f"  {n} cards: {count} chapters")

    # dt_secondary stats
    with_secondary = sum(1 for c in all_condensed if c.get('dt_secondary'))
    print(f"\nCards with dt_secondary: {with_secondary} ({with_secondary/len(all_condensed)*100:.0f}%)")


if __name__ == "__main__":
    main()
