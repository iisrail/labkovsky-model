#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment Labkovsky answers using Claude API.

Creates training data for Vikhr to learn RS segmentation.

Usage:
    export ANTHROPIC_API_KEY=your_key
    python segment_with_claude.py

Output: data/fine_tuning/qa_rs_segmented.jsonl
"""

import json
import time
from pathlib import Path
from anthropic import Anthropic

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "data" / "fine_tuning" / "qa_rs_corpus_short.jsonl"
OUTPUT_PATH = PROJECT_DIR / "data" / "fine_tuning" / "qa_rs_segmented.jsonl"

NUM_SAMPLES = 50
MODEL = "claude-sonnet-4-20250514"

# ============================================================
# PROMPT
# ============================================================

SEGMENT_PROMPT = """Расставь метки в тексте ответа психолога. Сохрани ВЕСЬ текст полностью.

Метки:
[EXPLANATION] — объяснение психологии, причины, анализ (почему так происходит)
[INTERVENTION] — конкретный совет, что ДЕЛАТЬ (действие, рекомендация, включая "идите к психологу")
[ESCALATION] — ТОЛЬКО при угрозе жизни (суицид, срочная госпитализация в психиатрию)

Правила:
1. Сохрани весь оригинальный текст
2. Ставь метку в начале каждой смысловой части
3. ESCALATION используй очень редко, только при явной угрозе жизни
4. Обычно ответ содержит EXPLANATION + INTERVENTION

Пример:
Текст: У вас низкая самооценка из-за детских травм. Родители критиковали, вы это усвоили. Попробуйте каждый день записывать три вещи, за которые себя хвалите. Идите к психологу.
Результат:
[EXPLANATION] У вас низкая самооценка из-за детских травм. Родители критиковали, вы это усвоили.
[INTERVENTION] Попробуйте каждый день записывать три вещи, за которые себя хвалите. Идите к психологу.

Теперь размети этот текст:
{answer}

Результат:"""


# ============================================================
# MAIN
# ============================================================

def segment_with_claude(client: Anthropic, answer: str) -> str:
    """Send answer to Claude for segmentation."""
    prompt = SEGMENT_PROMPT.format(answer=answer)

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text.strip()


def load_samples(path: Path, num: int) -> list:
    """Load N samples from JSONL file."""
    samples = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            if line.strip():
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError:
                    continue
    return samples


def main():
    print("=" * 60)
    print("Segment Labkovsky Answers with Claude API")
    print("=" * 60)

    # Initialize client
    client = Anthropic()
    print(f"[+] Using model: {MODEL}")

    # Load samples
    print(f"[+] Loading {NUM_SAMPLES} samples from: {INPUT_PATH.name}")
    samples = load_samples(INPUT_PATH, NUM_SAMPLES)
    print(f"    Loaded {len(samples)} samples")

    # Process each sample
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Processing: {sample.get('id', 'unknown')}")

        answer = sample.get("answer", "")
        if not answer:
            print("    Skipping - no answer")
            continue

        try:
            segmented = segment_with_claude(client, answer)

            result = {
                "id": sample.get("id"),
                "question": sample.get("question"),
                "answer_original": answer,
                "answer_segmented": segmented,
                "video_id": sample.get("video_id"),
                "decision_type": sample.get("decision_type"),
            }
            results.append(result)

            # Show preview
            preview = segmented[:150].replace('\n', ' ')
            print(f"    OK: {preview}...")

            # Rate limit - be nice to API
            time.sleep(0.5)

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Save results
    print(f"\n[+] Saving {len(results)} results to: {OUTPUT_PATH.name}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print(f"Done! Segmented {len(results)} answers.")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
