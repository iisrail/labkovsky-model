#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate questions for book chunks using Qwen via Ollama.

Usage:
    ollama serve  # In one terminal
    python generate_questions_qwen.py <input_chunks.jsonl>

Example:
    python generate_questions_qwen.py pro_pervoe_pravilo_chunks.jsonl
    python generate_questions_qwen.py pro_smysl_zhizni_chunks.jsonl
"""

import json
import sys
import time
from pathlib import Path
import ollama

# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).parent

MODEL_NAME = "qwen2.5:14b"  # or "qwen2.5:7b" for faster processing

# Prompt template for question generation (English instructions, Russian output)
PROMPT_TEMPLATE = """You are analyzing text from Russian psychologist Mikhail Labkovsky.

TEXT:
{text}

Generate 2-3 questions (in Russian) that a person might ask a psychologist if they were looking for this exact information.

Requirements:
- Questions must be in Russian using ONLY Cyrillic alphabet (no Latin letters!)
- Natural phrasing (as a real person would ask)
- Different formulations (not just rephrasing the same question)
- Relevant to the content of the text
- DO NOT mention "Labkovsky" or "психолог" in questions - user doesn't know the source yet
- Questions should be like real user queries: "Как справиться с...", "Почему я...", "Что делать если..."

Response - JSON array ONLY, no explanation, no markdown:
["вопрос 1", "вопрос 2", "вопрос 3"]"""


# ============================================================
# QUESTION GENERATION
# ============================================================

def generate_questions(text: str, retries: int = 3) -> list:
    """Generate questions using Qwen via Ollama"""
    
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
                ],
                options={
                    "temperature": 0.7,
                    "num_predict": 300,  # Limit output tokens
                }
            )
            
            content = response['message']['content'].strip()
            
            # Clean up markdown wrappers if present
            if content.startswith("```"):
                # Remove ```json or ``` at start
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            
            content = content.strip()
            
            # Parse JSON
            questions = json.loads(content)
            
            # Validate it's a list of strings
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
            else:
                print(f"  ⚠ Invalid format (attempt {attempt+1}): not a list of strings")
                
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse error (attempt {attempt+1}): {e}")
            print(f"    Response was: {content[:100]}...")
            if attempt < retries - 1:
                time.sleep(1)
                
        except Exception as e:
            print(f"  ⚠ Ollama error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    return []


# ============================================================
# MAIN
# ============================================================

def main():
    # Check command line argument
    if len(sys.argv) < 2:
        print("Usage: python generate_questions_qwen.py <input_chunks.jsonl>")
        print("Example: python generate_questions_qwen.py pro_pervoe_pravilo_chunks.jsonl")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    # Output file based on input name
    output_file = SCRIPT_DIR / f"{input_file.stem.replace('_chunks', '')}_with_questions.jsonl"
    
    print("=" * 60)
    print("Generate Questions for Book Chunks (Qwen)")
    print("=" * 60)
    
    # Check input file
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Load chunks
    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    print(f"📄 Input: {input_file.name}")
    print(f"📝 Loaded {len(records)} chunks")
    print(f"🤖 Model: {MODEL_NAME}")
    print("=" * 60)
    
    # Process each chunk
    results = []
    total_questions = 0
    
    for i, record in enumerate(records):
        chunk_id = record.get('chunk_id', i)
        chapter = record.get('chapter', '')[:30]
        
        print(f"\n[{i+1}/{len(records)}] Chapter: {chapter}, Chunk: {chunk_id}")
        print(f"   Text: \"{record['text'][:60]}...\"")
        
        # Generate questions
        questions = generate_questions(record['text'])
        
        # Add to record
        record['potential_questions'] = questions
        results.append(record)
        
        if questions:
            total_questions += len(questions)
            print(f"   ✅ {len(questions)} questions:")
            for q in questions:
                print(f"      - {q[:60]}...")
        else:
            print(f"   ❌ Failed to generate questions")
        
        # Save intermediate results every 10 chunks
        if (i + 1) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                for rec in results:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f"\n   💾 Saved intermediate ({i+1} chunks)")
    
    # Final save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Statistics
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"   Chunks processed: {len(results)}")
    print(f"   Questions generated: {total_questions}")
    print(f"   Average per chunk: {total_questions/len(results):.1f}")
    print(f"   Output: {output_file}")


if __name__ == "__main__":
    main()