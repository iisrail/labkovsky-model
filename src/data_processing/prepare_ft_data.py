#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Labkovsky Q&A data to fine-tuning format.
Output: JSONL with instruction/input/output format for unsloth.
"""

import json
from pathlib import Path

# ==============================================================
# SETTINGS
# ==============================================================

INPUT_FILE = Path("data/fine_tuning/test_video_full.jsonl")
OUTPUT_FILE = Path("data/fine_tuning/train_data.jsonl")

# ==============================================================
# CONVERSION
# ==============================================================

def convert_to_ft_format(input_path: Path, output_path: Path):
    """Convert Q&A pairs to fine-tuning format"""
    
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"📄 Loaded {len(records)} Q&A pairs")
    
    # Convert to FT format
    ft_data = []
    for r in records:
        ft_record = {            
            "input": r["question"],
            "output": r["answer"]
        }
        ft_data.append(ft_record)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in ft_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(ft_data)} records to {output_path}")
    
    # Show sample
    print("\n📝 Sample record:")
    print(json.dumps(ft_data[0], ensure_ascii=False, indent=2)[:500] + "...")
    
    # Stats
    total_chars = sum(len(r["input"]) + len(r["output"]) for r in ft_data)
    avg_output_len = sum(len(r["output"]) for r in ft_data) / len(ft_data)
    print(f"\n📊 Stats:")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Avg answer length: {avg_output_len:.0f} chars")


if __name__ == "__main__":
    convert_to_ft_format(INPUT_FILE, OUTPUT_FILE)
