#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert book dialogues to qa_rs_corpus format.

Input:  book_dialogues.jsonl (your format)
Output: Appends to qa_rs_corpus_short.jsonl
"""

import json
from pathlib import Path

# Paths - adjust if needed
INPUT_FILE = Path("data/fine_tuning/book_dialogues.jsonl")  # Your book dialogues
OUTPUT_FILE = Path("data/fine_tuning/qa_rs_corpus_short.jsonl")  # Append here

def classify_rs(question: str, answer: str) -> str:
    """
    Simple RS classification based on answer content.
    You may want to review and adjust manually.
    """
    answer_lower = answer.lower()
    
    # ESCALATION: refers to specialist
    escalation_markers = ["психиатр", "психотерапевт", "специалист", "врач", "обратитесь"]
    if any(m in answer_lower for m in escalation_markers):
        return "ESCALATION"
    
    # INTERVENTION: direct advice, commands
    intervention_markers = ["уходите", "бросайте", "перестаньте", "прекратите", 
                           "не надо", "хватит", "поменять", "подать в суд",
                           "надо", "нужно", "попробуйте", "сделайте"]
    if any(m in answer_lower for m in intervention_markers):
        return "INTERVENTION"
    
    # Default: EXPLANATION
    return "EXPLANATION"


def convert_dialogues():
    if not INPUT_FILE.exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("   Create this file with your book dialogues first.")
        return
    
    converted = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {line_num}: JSON error - {e}")
                continue
            
            # Extract Q&A from turns
            turns = data.get("turns", [])
            if len(turns) < 2:
                print(f"⚠️ Line {line_num}: Not enough turns")
                continue
            
            question = turns[0].get("content", "")
            answer = turns[1].get("content", "")
            
            if not question or not answer:
                print(f"⚠️ Line {line_num}: Empty question or answer")
                continue
            
            # Classify RS
            rs = classify_rs(question, answer)
            
            # Check if answer is short (for short_answer flag)
            word_count = len(answer.split())
            short_answer = word_count < 20
            
            record = {
                "question": question,
                "answer": answer,
                "response_signal": rs,
                "source": "book",
                "book_id": data.get("book_id", "unknown"),
                "chapter_id": data.get("chapter_id", "unknown"),
            }
            
            if short_answer:
                record["short_answer"] = True
            
            converted.append(record)
            print(f"✓ [{rs}] {question[:50]}...")
    
    if not converted:
        print("❌ No dialogues converted!")
        return
    
    # Append to output file
    print(f"\n📝 Appending {len(converted)} dialogues to {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Done! Added {len(converted)} Q&A pairs.")
    
    # Summary
    rs_counts = {}
    for r in converted:
        rs = r["response_signal"]
        rs_counts[rs] = rs_counts.get(rs, 0) + 1
    
    print(f"\nRS distribution of new data:")
    for rs, count in sorted(rs_counts.items()):
        print(f"   {rs}: {count}")


if __name__ == "__main__":
    convert_dialogues()