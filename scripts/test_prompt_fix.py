#!/usr/bin/env python3
"""Test if updated prompt helps with user detail attention and overcopying."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "src/rag")

from query_rag import init_embedding, init_retrieval, init_llm, ask_labkovsky
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
AWQ_MODEL = "models/vikhr-labkovsky-awq"

# Questions with specific details the model should acknowledge
QUESTIONS = [
    "Увидел, что моя дочь курит. Ей 15. Сорвался, наорал на неё. Я сам бросил курить 10 лет назад. Что делать?",
    "Муж изменил, я его уже простила, но не могу забыть. Как жить дальше?",
    "Я уже хожу к психологу полгода, но всё равно не могу справиться с тревогой.",
]

print("Loading models...")
init_embedding()
init_retrieval(CHROMA_DIR)
init_llm(use_lora=False, lora_path=None, base_model_path=AWQ_MODEL, skip_quantization=True)
print("Ready!\n")
print("=" * 70)

for i, question in enumerate(QUESTIONS, 1):
    print(f"\n[{i}/3] QUESTION:")
    print(f"{question}\n")

    result = ask_labkovsky(
        question,
        temperature=0.5,
        repetition_penalty=1.15,
        top_p=0.85,
        max_tokens=400
    )

    print(f"ANSWER:\n{result['answer']}")
    print("\n" + "=" * 70)

print("\nDone!")
