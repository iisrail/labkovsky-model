#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean qa_rs_final.jsonl: strip RS markers and 'Не требуется' stubs.

Input:  data/fine_tuning/qa_rs_final.jsonl
Output: data/fine_tuning/qa_clean.jsonl

Usage:
    python src/fine_tuning/clean_qa_data.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "fine_tuning" / "qa_rs_final.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "fine_tuning" / "qa_clean.jsonl"

_RS_TAG = re.compile(r'\[[^\]]+\]\s*')
_NOT_NEEDED = re.compile(r'\s*Не требуется\.?\s*')


def clean_text(text: str) -> str:
    text = _RS_TAG.sub('', text)
    text = _NOT_NEEDED.sub(' ', text)
    return text.strip()


def main():
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Clean answer_segmented -> answer
            answer = obj.get("answer_segmented", obj.get("answer", ""))
            obj["answer"] = clean_text(answer)

            # Remove answer_segmented — no longer needed
            obj.pop("answer_segmented", None)

            records.append(obj)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Cleaned {len(records)} records")
    print(f"  Input:  {INPUT_FILE.name}")
    print(f"  Output: {OUTPUT_FILE.name}")

    # Show example
    if records:
        ex = records[0]
        print(f"\n--- Example ---")
        print(f"Q: {ex.get('question', '')[:100]}")
        print(f"A: {ex['answer'][:300]}")


if __name__ == "__main__":
    main()
