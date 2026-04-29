#!/usr/bin/env python3
"""Quick test of merged model."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "src/rag")

from query_rag import init_embedding, init_retrieval, init_llm, ask_labkovsky
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
MERGED_MODEL = "models/vikhr-labkovsky-awq"

print("Loading models...")
init_embedding()
init_retrieval(CHROMA_DIR)
init_llm(use_lora=False, lora_path=None, base_model_path=MERGED_MODEL, skip_quantization=True)
print("Ready!\n")

# Test question
question = "Почему я не могу уйти от мужа который меня не уважает?"
print(f"Q: {question}\n")

result = ask_labkovsky(
    question,
    temperature=0.5,
    repetition_penalty=1.15,
    top_p=0.85,
    max_tokens=400
)

print(f"\nA: {result['answer']}")
print(f"\nTokens: {result.get('tokens_generated', '?')}")
