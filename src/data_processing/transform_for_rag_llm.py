#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform Labkovsky Q&A Corpus for RAG using LLM.

Uses Qwen2.5-7B-Instruct to intelligently split and condense answers
into EXPLANATION and INTERVENTION chunks.

Usage (in WSL with GPU):
    python src/data_processing/transform_for_rag_llm.py
"""

import json
import re
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "fine_tuning" / "qa_rs_corpus_short.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "fine_tuning" / "qa_corpus_rag_optimized.jsonl"
OUTPUT_FILE_TEST = PROJECT_ROOT / "data" / "fine_tuning" / "qa_corpus_rag_test.jsonl"

# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048

# ============================================================
# TRANSFORMATION PROMPT
# ============================================================

SYSTEM_PROMPT = """Ты — редактор корпуса Q&A Михаила Лабковского для RAG-системы.

## Твоя задача
Преобразовать один Q&A в 1-2 чанка: EXPLANATION и/или INTERVENTION.

## Правила

### 1. Аутентичность (КРИТИЧНО)
- НЕ перефразируй, НЕ смягчай
- МОЖНО сокращать и переставлять
- НУЖНО сохранять резкость и стиль Лабковского

### 2. Логика разделения

**Есть и объяснение, и директива** → Раздели на 2:
- **EXPLANATION**: объясняет "почему", БЕЗ императивов
- **INTERVENTION**: содержит императивы ИЛИ решительные риторические вопросы

**Только директива** → INTERVENTION
**Только объяснение** → EXPLANATION

### 3. Что УДАЛИТЬ:
- "Я понимаю", "Вам тяжело"
- "Обратитесь к психотерапевту" (как концовка)
- Исследовательские вопросы: "Как вы думаете...?", "Что вы чувствуете?"

### 4. Что СОХРАНИТЬ:
- Жёсткие диагнозы: "У вас невроз", "Это созависимость"
- "Идите к психиатру" (как директива)
- Решительные вопросы: "А вам это нужно?", "Зачем вам это?"
- Конкретные инструкции

### 5. Формат ответа
Верни ТОЛЬКО валидный JSON (без markdown, без ```):
[
  {"chunk_role": "EXPLANATION", "answer": "текст"},
  {"chunk_role": "INTERVENTION", "answer": "текст"}
]

Или один элемент, если разделение не нужно:
[
  {"chunk_role": "INTERVENTION", "answer": "текст"}
]

## Пример

Вход:
{"question": "Муж не помогает по дому, как его заставить?", "answer": "Вы не можете заставить взрослого человека что-то делать. Он живёт как хочет, потому что вы это терпите. У вас два варианта: либо принять ситуацию и не беситься, либо уйти. Всё."}

Выход:
[
  {"chunk_role": "EXPLANATION", "answer": "Вы не можете заставить взрослого человека что-то делать. Он живёт как хочет, потому что вы это терпите."},
  {"chunk_role": "INTERVENTION", "answer": "У вас два варианта: либо принять ситуацию и не беситься, либо уйти. Всё."}
]"""

USER_PROMPT_TEMPLATE = """Преобразуй этот Q&A:

{{"question": "{question}", "answer": "{answer}"}}

Верни JSON массив с 1-2 чанками."""

# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """Load Qwen model with unsloth."""
    print(f"🤖 Loading model: {MODEL_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Enable faster inference
    FastLanguageModel.for_inference(model)

    print("✅ Model loaded")
    return model, tokenizer


# ============================================================
# TRANSFORMATION
# ============================================================

def transform_record(model, tokenizer, record: dict) -> list[dict]:
    """Transform a single record using LLM."""

    question = record['question']
    answer = record['answer']

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question.replace('"', '\\"'),
        answer=answer.replace('"', '\\"')
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.shape[-1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # Parse JSON from response
    chunks = parse_llm_response(response, record)

    return chunks


def parse_llm_response(response: str, record: dict) -> list[dict]:
    """Parse LLM response and create output chunks."""

    # Try to extract JSON array
    try:
        # Find JSON array in response
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            json_str = match.group()
            parsed = json.loads(json_str)
        else:
            raise ValueError("No JSON array found")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠ JSON parse error: {e}")
        print(f"    Response: {response[:200]}...")
        # Fallback: return original as EXPLANATION
        return [{
            "type": "qa",
            "chunk_role": "EXPLANATION",
            "question": record['question'],
            "answer": record['answer'],
            "id": f"{record.get('id', 'unknown')}_expl",
            "video_id": record.get('video_id', ''),
            "char_count": len(record['answer']),
            "parse_error": True
        }]

    # Build output chunks
    chunks = []
    base_id = record.get('id', 'unknown')
    video_id = record.get('video_id', '')

    for item in parsed:
        role = item.get('chunk_role', 'EXPLANATION')
        answer_text = item.get('answer', '')

        if not answer_text:
            continue

        suffix = "_expl" if role == "EXPLANATION" else "_interv"

        chunk = {
            "type": "qa",
            "chunk_role": role,
            "question": record['question'],
            "answer": answer_text,
            "id": f"{base_id}{suffix}",
            "char_count": len(answer_text)
        }

        if video_id:
            chunk["video_id"] = video_id

        chunks.append(chunk)

    return chunks if chunks else [{
        "type": "qa",
        "chunk_role": "EXPLANATION",
        "question": record['question'],
        "answer": record['answer'],
        "id": f"{base_id}_expl",
        "video_id": video_id,
        "char_count": len(record['answer']),
        "fallback": True
    }]


# ============================================================
# MAIN
# ============================================================

def process_file():
    """Main processing function."""
    print("=" * 60)
    print("Transform Labkovsky Q&A for RAG (LLM-based)")
    print("=" * 60)
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}\n")

    # Load model
    model, tokenizer = load_model()

    # Read input
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ⚠ JSON error: {e}")

    # Limit for testing (set to None for full processing)
    TEST_LIMIT = 20
    if TEST_LIMIT:
        records = records[:TEST_LIMIT]
        output_file = OUTPUT_FILE_TEST
        print(f"\n📄 Testing on first {len(records)} records")
        print(f"Output: {output_file}\n")
    else:
        output_file = OUTPUT_FILE
        print(f"\n📄 Loaded {len(records)} input records")
        print(f"⏱ Estimated time: {len(records) * 15 // 60} minutes\n")

    # Process with progress bar
    all_chunks = []
    stats = {
        "split_both": 0,
        "single": 0,
        "parse_errors": 0,
        "expl_chars": [],
        "interv_chars": [],
    }

    # Open output file for incremental writing
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in tqdm(records, desc="Transforming"):
            try:
                chunks = transform_record(model, tokenizer, record)

                # Track stats
                if len(chunks) == 2:
                    stats["split_both"] += 1
                else:
                    stats["single"] += 1

                for chunk in chunks:
                    if chunk.get('parse_error'):
                        stats["parse_errors"] += 1

                    if chunk['chunk_role'] == "EXPLANATION":
                        stats["expl_chars"].append(chunk['char_count'])
                    else:
                        stats["interv_chars"].append(chunk['char_count'])

                    # Write immediately
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    all_chunks.append(chunk)

            except Exception as e:
                print(f"\n❌ Error processing {record.get('id', 'unknown')}: {e}")
                # Write original as fallback
                fallback = {
                    "type": "qa",
                    "chunk_role": "EXPLANATION",
                    "question": record['question'],
                    "answer": record['answer'],
                    "id": f"{record.get('id', 'unknown')}_expl",
                    "char_count": len(record['answer']),
                    "error": str(e)
                }
                f_out.write(json.dumps(fallback, ensure_ascii=False) + '\n')
                all_chunks.append(fallback)

    # Report
    print("\n" + "=" * 60)
    print("Transformation Complete")
    print("=" * 60)
    print(f"\nInput:  {len(records)} records")
    print(f"Output: {len(all_chunks)} chunks")

    expl_count = len(stats["expl_chars"])
    interv_count = len(stats["interv_chars"])
    print(f"        ({expl_count} EXPLANATION + {interv_count} INTERVENTION)")

    print(f"\nSplit distribution:")
    print(f"  Split into 2: {stats['split_both']}")
    print(f"  Single chunk: {stats['single']}")
    print(f"  Parse errors: {stats['parse_errors']}")

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

    print(f"\n✅ Output saved to: {output_file}")


if __name__ == "__main__":
    process_file()
