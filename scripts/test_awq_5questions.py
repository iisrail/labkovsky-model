#!/usr/bin/env python3
"""Test AWQ model with 5 diverse questions."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "src/rag")

from query_rag import init_embedding, init_retrieval, init_llm, ask_labkovsky
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
AWQ_MODEL = "models/vikhr-labkovsky-awq"

QUESTIONS = [
    "Почему я не могу уйти от мужа который меня не уважает?",
    "Мой ребёнок не слушается, что делать?",
    "Я боюсь остаться одна, как перестать бояться?",
    "Муж пьёт, но я его люблю. Как быть?",
    "Я не могу найти работу по душе, всё раздражает.",
]

TOPICS = ["Relationships", "Parenting", "Fear", "Addiction", "Career"]

print("Loading models...")
init_embedding()
init_retrieval(CHROMA_DIR)
init_llm(use_lora=False, lora_path=None, base_model_path=AWQ_MODEL, skip_quantization=True)
print("Ready!\n")
print("=" * 70)

for i, (question, topic) in enumerate(zip(QUESTIONS, TOPICS), 1):
    print(f"\n[{i}/5] {topic.upper()}")
    print(f"Q: {question}\n")

    result = ask_labkovsky(
        question,
        temperature=0.5,
        repetition_penalty=1.15,
        top_p=0.85,
        max_tokens=400
    )

    print(f"A: {result['answer']}")
    print("\n" + "=" * 70)

print("\nDone!")
