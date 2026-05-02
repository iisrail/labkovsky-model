#!/usr/bin/env python3
"""Test single question locally."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "src/rag")

from query_rag import init_embedding, init_retrieval, init_llm, ask_labkovsky

print("Loading models...")
init_embedding()
init_retrieval("chroma_db")
init_llm(use_lora=False, base_model_path="models/vikhr-labkovsky-awq", skip_quantization=True)
print("Ready!\n")

question = "Муж бьёт меня, но только когда пьяный. Трезвый он хороший. Как его изменить?"
print(f"Q: {question}\n")

result = ask_labkovsky(question, temperature=0.5, repetition_penalty=1.15, top_p=0.85, max_tokens=400)
print(f"A: {result['answer']}")
