#!/usr/bin/env python3
"""
Strip RS tags from anti_generic_stage3.jsonl → anti_generic_clean.jsonl
Run before training.
"""

import json
import re
from pathlib import Path

INPUT = Path("data/fine_tuning/anti_generic_stage3.jsonl")
OUTPUT = Path("data/fine_tuning/anti_generic_clean.jsonl")

TAG_PATTERN = r'\[(ОБЪЯСНЕНИЕ|ВМЕШАТЕЛЬСТВО|ЭСКАЛАЦИЯ|EXPLANATION|INTERVENTION|ESCALATION)\]\s*'

records = []
with open(INPUT, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        answer = data.get('answer_segmented', data.get('answer', ''))
        answer = re.sub(TAG_PATTERN, '', answer).strip()
        answer = re.sub(r'\n?Не требуется\.?', '', answer).strip()

        records.append({
            'question': data['question'],
            'answer': answer
        })

with open(OUTPUT, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"Done: {len(records)} records")
print(f"Saved: {OUTPUT}")