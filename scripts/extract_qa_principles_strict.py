#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ONLY abstract principles from QA answers.

Filters out:
- Specific advice ("вы должны", "вам надо")
- Context-dependent statements
- Incomplete thoughts

Keeps only:
- General truths about psychology
- Universal rules/patterns
- Abstract insights
"""

import json
import re
import sys
import io
from pathlib import Path
from typing import List, Dict, Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

INPUT_FILE = Path("data/fine_tuning/qa_rs_segmented.jsonl")
OUTPUT_FILE = Path("data/fine_tuning/card_qa_principles_strict.jsonl")

# Minimum/maximum length
MIN_LEN = 40
MAX_LEN = 400

# Patterns that indicate GOOD abstract principles
GOOD_STARTS = [
    r'^Все\s',           # Все зависимости...
    r'^Любая\s',         # Любая зависимость...
    r'^Любой\s',         # Любой человек...
    r'^Каждый\s',        # Каждый человек...
    r'^Человек\s',       # Человек не может...
    r'^Люди\s',          # Люди, которые...
    r'^Невротик',        # Невротик всегда...
    r'^Невроз',          # Невроз — это...
    r'^Тревога\s',       # Тревога — это...
    r'^Тревожность\s',   # Тревожность проявляется...
    r'^Зависимость\s',   # Зависимость — это...
    r'^Самооценка\s',    # Самооценка — это...
    r'^Любовь\s',        # Любовь — это...
    r'^Страх\s',         # Страх — это...
    r'^Проблема\s(?!в том, что вы)',  # Проблема в том... (general)
    r'^Причина\s',       # Причина всегда...
    r'^Ребенок\s',       # Ребенок, которого...
    r'^Дети\s',          # Дети, которых...
    r'^Родители\s',      # Родители, которые...
    r'^Отношения\s',     # Отношения — это...
    r'^В основе\s',      # В основе лежит...
    r'^Корень\s',        # Корень проблемы...
]

# Patterns that indicate it's a UNIVERSAL statement (not specific advice)
GOOD_PATTERNS = [
    r'— это\s',          # X — это Y (definition)
    r'всегда\s',         # Always true
    r'никогда\s',        # Never true
    r'не может\s',       # Cannot (universal)
    r'не способен',      # Incapable (universal)
    r'означает',         # Means (definition)
    r'связан[оа]?\sс',   # Connected to (cause)
    r'ведет к',          # Leads to
    r'приводит к',       # Results in
    r'рождается из',     # Born from
    r'растет из',        # Grows from
    r'корнями уходит',   # Rooted in
]

# Patterns that indicate BAD (specific, contextual)
BAD_PATTERNS = [
    r'^Вы\s',            # You (specific person)
    r'^Вам\s',           # To you
    r'^Ваш[аие]?\s',     # Your
    r'^У вас\s',         # You have
    r'^Вот вы\s',        # Here you
    r'надо\sвам',        # You need to
    r'вам\sнадо',        # You need to
    r'должны\sвы',       # You must
    r'вы\sдолжны',       # You must
    r'^Если вы\s',       # If you (specific)
    r'^Когда вы\s',      # When you (specific)
    r'^Попробуйте',      # Try to
    r'^Сделайте',        # Do this
    r'^Начните',         # Start with
    r'^Идите',           # Go to
    r'^Обратитесь',      # Contact
    r'^Возьмите',        # Take
    r'^Скажите',         # Say
    r'^Спросите',        # Ask
    r'^\d+[.)]',         # Numbered list
    r'^Первое[,:]',      # First (list)
    r'^Второе[,:]',      # Second (list)
    r'^—\s',             # Dash start (continuation)
    r'^\?\s',            # Question
    r'^Ну\s',            # Filler
    r'^Так\s',           # Filler
    r'^Вот\s',           # Filler
    r'^И\s',             # And (continuation)
    r'^А\s',             # And/But (continuation)
    r'^Но\s',            # But (continuation)
    r'^То есть\s',       # That is (continuation)
    r'^Потому что\s',    # Because (continuation)
    r'конкретно\s',      # Specifically
    r'в вашем случае',   # In your case
    r'у вас',            # You have (specific)
]


def load_qa(filepath: Path) -> List[Dict[str, Any]]:
    """Load QA records."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    continue
    return records


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove RS markers
    text = re.sub(r'\[(ОБЪЯСНЕНИЕ|ВМЕШАТЕЛЬСТВО|ЭСКАЛАЦИЯ|EXPLANATION|INTERVENTION|ESCALATION)\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    """Split into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def is_abstract_principle(sentence: str) -> bool:
    """Check if sentence is an abstract principle."""
    # Length check
    if len(sentence) < MIN_LEN or len(sentence) > MAX_LEN:
        return False

    # Must NOT match bad patterns (specific advice)
    for pattern in BAD_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return False

    # Must match either good start OR contain good pattern
    has_good_start = False
    for pattern in GOOD_STARTS:
        if re.match(pattern, sentence, re.IGNORECASE):
            has_good_start = True
            break

    has_good_pattern = False
    for pattern in GOOD_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            has_good_pattern = True
            break

    return has_good_start or has_good_pattern


def extract_principles(answer: str) -> List[str]:
    """Extract abstract principles from answer."""
    answer = clean_text(answer)
    sentences = split_sentences(answer)

    principles = []
    for sent in sentences:
        if is_abstract_principle(sent):
            principles.append(sent)

    return principles


def deduplicate(principles: List[Dict]) -> List[Dict]:
    """Remove duplicates by first 50 chars."""
    seen = set()
    unique = []
    for p in principles:
        key = p['core_principle'][:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    print(f"Loading QA from {INPUT_FILE}...")
    qa_records = load_qa(INPUT_FILE)
    print(f"Loaded {len(qa_records)} records")

    all_principles = []

    for qa in qa_records:
        answer = qa.get('answer', '')
        if not answer:
            continue

        dt = qa.get('decision_type', 'EXPLANATION')
        principles = extract_principles(answer)

        for p in principles:
            all_principles.append({
                "source": "qa_principle",
                "dt_primary": dt,
                "core_principle": p,
            })

    print(f"Extracted {len(all_principles)} raw principles")

    # Dedupe
    unique = deduplicate(all_principles)
    print(f"After dedup: {len(unique)} principles")

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for p in unique:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # Stats
    print(f"\n=== BY DT ===")
    dt_counts = {}
    for p in unique:
        dt = p['dt_primary']
        dt_counts[dt] = dt_counts.get(dt, 0) + 1
    for dt, count in sorted(dt_counts.items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}")

    # Samples
    print(f"\n=== SAMPLES ===")
    for p in unique[:10]:
        print(f"\n[{p['dt_primary']}]")
        print(f"  {p['core_principle']}")

    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
