#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract reasoning principles from QA answers.

Input: qa_rs_segmented.jsonl
Output: card_qa_principles.jsonl

Extracts key statements from Labkovsky's answers that express
psychological principles, rules, and insights.
"""

import json
import re
import sys
import io
from pathlib import Path
from typing import List, Dict, Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
INPUT_FILE = Path("data/fine_tuning/qa_rs_segmented.jsonl")
OUTPUT_FILE = Path("data/fine_tuning/card_qa_principles.jsonl")

# Patterns that indicate a principle/rule statement
PRINCIPLE_PATTERNS = [
    # Definitive statements
    r'^[А-ЯЁ][^.!?]*(?:всегда|никогда|любой|каждый|все|всё)[^.!?]*[.!]',
    r'^[А-ЯЁ][^.!?]*(?:— это|это —|– это)[^.!?]*[.!]',
    r'^[А-ЯЁ][^.!?]*(?:потому что|из-за того)[^.!?]*[.!]',

    # Conditional rules
    r'^Если[^.!?]+(?:то|—|–)[^.!?]*[.!]',
    r'^Когда[^.!?]+(?:тогда|—|–)[^.!?]*[.!]',
    r'^Пока[^.!?]+[.!]',

    # Problem statements
    r'^Проблема (?:в том|не в)[^.!?]*[.!]',
    r'^Дело (?:в том|не в)[^.!?]*[.!]',
    r'^Причина[^.!?]*[.!]',

    # Key insights
    r'^[А-ЯЁ][^.!?]*(?:на самом деле|в основе|корень)[^.!?]*[.!]',
    r'^[А-ЯЁ][^.!?]*(?:невроз|невротик|тревога|тревожн)[^.!?]*[.!]',
    r'^[А-ЯЁ][^.!?]*(?:самооценк|любовь к себе)[^.!?]*[.!]',
    r'^[А-ЯЁ][^.!?]*(?:зависимост)[^.!?]*[.!]',

    # Labkovsky rules
    r'^[А-ЯЁ][^.!?]*(?:правило|закон)[^.!?]*[.!]',
    r'^Надо[^.!?]*[.!]',
    r'^Нужно[^.!?]*[.!]',
    r'^Нельзя[^.!?]*[.!]',
    r'^Важно[^.!?]*[.!]',
]

# Minimum length for a principle (chars)
MIN_PRINCIPLE_LEN = 50
MAX_PRINCIPLE_LEN = 500

# Skip patterns (questions, filler, etc.)
SKIP_PATTERNS = [
    r'^\?',
    r'^[А-ЯЁа-яё]{1,3}[.!?]$',  # Very short
    r'^(Да|Нет|Ну|Так|Вот)[,.]',  # Filler starts
    r'^(Спасибо|Пожалуйста|Здравствуйте)',  # Greetings
]


def load_qa(filepath: Path) -> List[Dict[str, Any]]:
    """Load QA records from JSONL."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def clean_text(text: str) -> str:
    """Clean text for processing."""
    # Remove RS markers
    text = re.sub(r'\[(ОБЪЯСНЕНИЕ|ВМЕШАТЕЛЬСТВО|ЭСКАЛАЦИЯ|EXPLANATION|INTERVENTION|ESCALATION)\]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on sentence endings, keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def is_principle(sentence: str) -> bool:
    """Check if sentence looks like a principle/rule."""
    # Length check
    if len(sentence) < MIN_PRINCIPLE_LEN or len(sentence) > MAX_PRINCIPLE_LEN:
        return False

    # Skip patterns
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, sentence, re.IGNORECASE):
            return False

    # Must match at least one principle pattern
    for pattern in PRINCIPLE_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True

    return False


def extract_principles_simple(answer: str) -> List[str]:
    """Extract principle sentences from answer using patterns."""
    answer = clean_text(answer)
    sentences = split_into_sentences(answer)

    principles = []
    for sent in sentences:
        if is_principle(sent):
            principles.append(sent)

    return principles


def extract_principles_contextual(answer: str) -> List[str]:
    """Extract principles with surrounding context for better understanding."""
    answer = clean_text(answer)
    sentences = split_into_sentences(answer)

    principles = []

    for i, sent in enumerate(sentences):
        if is_principle(sent):
            # Get context: previous sentence if short principle
            principle = sent

            # If principle is short and there's a previous sentence, add context
            if len(sent) < 100 and i > 0:
                prev = sentences[i-1]
                if len(prev) < 150 and not is_principle(prev):
                    principle = prev + " " + sent

            principles.append(principle)

    return principles


def deduplicate_principles(principles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove near-duplicate principles."""
    seen = set()
    unique = []

    for p in principles:
        # Use first 50 chars as key
        key = p['core_principle'][:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def main():
    print(f"Loading QA from {INPUT_FILE}...")
    qa_records = load_qa(INPUT_FILE)
    print(f"Loaded {len(qa_records)} QA records")

    all_principles = []

    for qa in qa_records:
        answer = qa.get('answer', '')
        if not answer:
            continue

        qa_id = qa.get('id', '')
        dt = qa.get('decision_type', 'EXPLANATION')
        dt_secondary = qa.get('dt_secondary')

        # Extract principles
        principles = extract_principles_contextual(answer)

        for principle in principles:
            card = {
                "source": "qa_principle",
                "qa_id": qa_id,
                "dt_primary": dt,
                "dt_secondary": dt_secondary,
                "core_principle": principle,
            }
            all_principles.append(card)

    print(f"Extracted {len(all_principles)} raw principles")

    # Deduplicate
    unique_principles = deduplicate_principles(all_principles)
    print(f"After deduplication: {len(unique_principles)} principles")

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for p in unique_principles:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # Stats
    print("\n=== STATS ===")
    print(f"Total principles: {len(unique_principles)}")

    # By DT
    dt_counts = {}
    for p in unique_principles:
        dt = p.get('dt_primary', 'UNKNOWN')
        dt_counts[dt] = dt_counts.get(dt, 0) + 1

    print("\nBy decision_type:")
    for dt, count in sorted(dt_counts.items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}")

    # Sample
    print("\n=== SAMPLE PRINCIPLES ===")
    for p in unique_principles[:5]:
        print(f"\n[{p['dt_primary']}]")
        print(f"  {p['core_principle'][:150]}...")

    print(f"\nDone. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
