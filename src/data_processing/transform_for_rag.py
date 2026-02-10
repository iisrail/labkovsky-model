#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform Labkovsky Q&A Corpus for RAG

Splits answers into EXPLANATION and INTERVENTION chunks based on content analysis.

Usage:
    python src/data_processing/transform_for_rag.py
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "fine_tuning" / "qa_rs_corpus_short.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "fine_tuning" / "qa_corpus_rag_optimized.jsonl"

# ============================================================
# MARKERS AND PATTERNS
# ============================================================

# Explanation markers (causation, diagnosis)
EXPLANATION_MARKERS = [
    "потому что", "поэтому", "дело в том", "это связано",
    "это называется", "это признак", "это невроз", "это страх",
    "причина в том", "это значит", "когда человек", "если человек",
    "это про", "это о том", "это зависимость", "это созависимость"
]

# Directive patterns (imperatives, boundaries, decisive questions, closures)
DIRECTIVE_PATTERNS = [
    # Imperatives (common forms)
    r'\b(прекратите|уходите|решите|начните|перестаньте|скажите|делайте|'
    r'бросьте|оставьте|примите|выбирайте|идите|живите|любите|'
    r'прекращайте|уходи|реши|начни|перестань|скажи|делай|'
    r'брось|оставь|прими|выбирай|иди|живи|люби)\b',
    # Boundaries
    r'либо\s+.{5,50}\s+либо',
    # Decisive rhetorical questions
    r'А вам это нужно\?',
    r'Зачем вам это\?',
    r'А какого хрена\?',
    r'Вам это надо\?',
    r'Оно вам надо\?',
    # Closures
    r'Всё\.$',
    r'Точка\.$',
]

# Patterns to delete
DELETE_PATTERNS = [
    r'Я понимаю[^.]*\.\s*',
    r'Вам тяжело[^.]*\.\s*',
    r'Я вас понимаю[^.]*\.\s*',
    r'Мне очень жаль[^.]*\.\s*',
    # Therapy referral only as ending
    r'\s*Обратитесь к психотерапевту\.?\s*$',
    r'\s*Вам нужен психотерапевт\.?\s*$',
    # Exploratory questions
    r'Как вы думаете[^?]*\?\s*',
    r'Что вы чувствуете[^?]*\?\s*',
    r'Как вы себя чувствуете[^?]*\?\s*',
]

# Patterns to KEEP (harsh diagnoses, etc.)
KEEP_PATTERNS = [
    r'У вас невроз',
    r'Идите к психиатру',
    r'6 правил',
    r'шесть правил',
]

# ============================================================
# CONTENT DETECTION
# ============================================================

def contains_explanation(text: str) -> bool:
    """Check if text contains explanation markers."""
    text_lower = text.lower()
    for marker in EXPLANATION_MARKERS:
        if marker in text_lower:
            return True
    return False


def contains_directive(text: str) -> bool:
    """Check if text contains directive patterns."""
    for pattern in DIRECTIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def has_directive_element(text: str) -> bool:
    """Check if INTERVENTION chunk has required directive element."""
    return contains_directive(text)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_answer(text: str) -> str:
    """Remove unwanted patterns from answer."""
    result = text
    for pattern in DELETE_PATTERNS:
        result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
    # Clean up whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def split_to_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence endings, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def is_directive_sentence(sentence: str) -> bool:
    """Check if a sentence is directive."""
    for pattern in DIRECTIVE_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    return False


# ============================================================
# ANSWER SPLITTING
# ============================================================

def split_answer(answer: str) -> tuple[str, str]:
    """
    Split answer at transition point between explanation and directive.
    Returns (explanation_part, intervention_part).
    """
    sentences = split_to_sentences(answer)

    if len(sentences) <= 1:
        # Can't split single sentence
        return answer, ""

    # Find first directive sentence
    split_idx = None
    for i, sent in enumerate(sentences):
        if is_directive_sentence(sent):
            split_idx = i
            break

    # If no directive found or it's the first sentence, use fallback
    if split_idx is None or split_idx == 0:
        # Fallback: split at ~60% for explanation
        split_idx = max(1, int(len(sentences) * 0.6))

    expl_sentences = sentences[:split_idx]
    interv_sentences = sentences[split_idx:]

    expl_text = ' '.join(expl_sentences)
    interv_text = ' '.join(interv_sentences)

    return expl_text, interv_text


def adjust_split_for_char_counts(expl: str, interv: str, sentences: list[str], split_idx: int) -> tuple[str, str]:
    """
    Adjust split point to meet character count requirements.
    EXPLANATION: 200-1500, INTERVENTION: 80-1000 (relaxed limits)
    """
    # If counts are already good, return as-is
    if 200 <= len(expl) <= 1500 and 80 <= len(interv) <= 1000:
        return expl, interv

    # Try adjusting split point
    best_split = split_idx
    best_score = float('inf')

    for try_idx in range(1, len(sentences)):
        try_expl = ' '.join(sentences[:try_idx])
        try_interv = ' '.join(sentences[try_idx:])

        expl_ok = 200 <= len(try_expl) <= 1500
        interv_ok = 80 <= len(try_interv) <= 1000

        if expl_ok and interv_ok:
            return try_expl, try_interv

        # Score by how close we are to targets
        expl_penalty = 0 if expl_ok else min(abs(len(try_expl) - 200), abs(len(try_expl) - 1500))
        interv_penalty = 0 if interv_ok else min(abs(len(try_interv) - 80), abs(len(try_interv) - 1000))
        score = expl_penalty + interv_penalty

        if score < best_score:
            best_score = score
            best_split = try_idx

    return ' '.join(sentences[:best_split]), ' '.join(sentences[best_split:])


# ============================================================
# CHUNK CREATION
# ============================================================

@dataclass
class ChunkResult:
    chunk: dict
    warnings: list[str]


def make_chunk(record: dict, text: str, role: str, suffix: str, counter: int) -> ChunkResult:
    """Create output chunk with validation."""
    # Generate ID
    base_id = record.get('id')
    if base_id:
        chunk_id = f"{base_id}{suffix}"
    else:
        chunk_id = f"qa_{counter:03d}{suffix}"

    chunk = {
        "type": "qa",
        "chunk_role": role,
        "question": record['question'],
        "answer": text,
        "id": chunk_id,
        "char_count": len(text)
    }

    # Validate
    warnings = validate_chunk(chunk)

    return ChunkResult(chunk=chunk, warnings=warnings)


def validate_chunk(chunk: dict) -> list[str]:
    """Validate chunk and return warnings."""
    warnings = []
    role = chunk['chunk_role']
    chars = chunk['char_count']
    text = chunk['answer']
    chunk_id = chunk['id']

    # Character count validation (relaxed limits)
    if role == "EXPLANATION":
        if chars < 200:
            warnings.append(f"{chunk_id}: EXPLANATION too short ({chars} < 200)")
        elif chars > 1500:
            warnings.append(f"{chunk_id}: EXPLANATION too long ({chars} > 1500)")
    elif role == "INTERVENTION":
        if chars < 80:
            warnings.append(f"{chunk_id}: INTERVENTION too short ({chars} < 80)")
        elif chars > 1000:
            warnings.append(f"{chunk_id}: INTERVENTION too long ({chars} > 1000)")

        # Must have directive element
        if not has_directive_element(text):
            warnings.append(f"{chunk_id}: INTERVENTION missing imperative/boundary/decisive question")

    return warnings


# ============================================================
# MAIN TRANSFORMATION
# ============================================================

def classify_and_split(record: dict, counter: int) -> list[ChunkResult]:
    """
    Transform a record into 1-2 output chunks.
    """
    answer = clean_answer(record['answer'])

    has_expl = contains_explanation(answer)
    has_dir = contains_directive(answer)

    results = []

    if has_expl and has_dir:
        # Split into 2 chunks
        expl_part, interv_part = split_answer(answer)

        # Try to optimize split for char counts
        sentences = split_to_sentences(answer)
        if len(sentences) > 1:
            # Find current split index
            expl_sentences = split_to_sentences(expl_part)
            split_idx = len(expl_sentences)
            expl_part, interv_part = adjust_split_for_char_counts(
                expl_part, interv_part, sentences, split_idx
            )

        if expl_part:
            results.append(make_chunk(record, expl_part, "EXPLANATION", "_expl", counter))
        if interv_part:
            results.append(make_chunk(record, interv_part, "INTERVENTION", "_interv", counter))

    elif has_dir:
        # Directive only
        results.append(make_chunk(record, answer, "INTERVENTION", "_interv", counter))

    else:
        # Explanation only (or neither - default to explanation)
        results.append(make_chunk(record, answer, "EXPLANATION", "_expl", counter))

    return results


def process_file():
    """Main processing function."""
    print("=" * 60)
    print("Transform Labkovsky Q&A for RAG")
    print("=" * 60)
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}\n")

    # Read input
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  JSON error: {e}")

    print(f"Loaded {len(records)} input records\n")

    # Process
    all_chunks = []
    all_warnings = []
    stats = {
        "split_both": 0,
        "expl_only": 0,
        "interv_only": 0,
        "expl_chars": [],
        "interv_chars": [],
    }

    for i, record in enumerate(records):
        results = classify_and_split(record, i)

        # Track stats
        roles = [r.chunk['chunk_role'] for r in results]
        if len(roles) == 2:
            stats["split_both"] += 1
        elif roles[0] == "EXPLANATION":
            stats["expl_only"] += 1
        else:
            stats["interv_only"] += 1

        for result in results:
            all_chunks.append(result.chunk)
            all_warnings.extend(result.warnings)

            if result.chunk['chunk_role'] == "EXPLANATION":
                stats["expl_chars"].append(result.chunk['char_count'])
            else:
                stats["interv_chars"].append(result.chunk['char_count'])

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Report
    print("=" * 60)
    print("Transformation Complete")
    print("=" * 60)
    print(f"\nInput:  {len(records)} records")
    print(f"Output: {len(all_chunks)} chunks")

    expl_count = len(stats["expl_chars"])
    interv_count = len(stats["interv_chars"])
    print(f"        ({expl_count} EXPLANATION + {interv_count} INTERVENTION)")

    print(f"\nSplit distribution:")
    print(f"  Split into 2:     {stats['split_both']}")
    print(f"  EXPLANATION only: {stats['expl_only']}")
    print(f"  INTERVENTION only: {stats['interv_only']}")

    if stats["expl_chars"]:
        print(f"\nEXPLANATION char counts:")
        print(f"  avg: {sum(stats['expl_chars']) // len(stats['expl_chars'])}")
        print(f"  min: {min(stats['expl_chars'])}")
        print(f"  max: {max(stats['expl_chars'])}")

    if stats["interv_chars"]:
        print(f"\nINTERVENTION char counts:")
        print(f"  avg: {sum(stats['interv_chars']) // len(stats['interv_chars'])}")
        print(f"  min: {min(stats['interv_chars'])}")
        print(f"  max: {max(stats['interv_chars'])}")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings[:20]:  # Show first 20
            print(f"  - {w}")
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings) - 20} more")
    else:
        print("\nNo warnings!")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_file()
