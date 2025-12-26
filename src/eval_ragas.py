# eval_ragas.py
# pip install ragas datasets openai
# + ollama должен быть запущен: ollama serve

import json
from pathlib import Path
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    context_recall
)

# RAGAS LLM через Ollama (OpenAI-compatible)
from openai import OpenAI
from ragas.llms import llm_factory

# Твои функции (из твоего проекта)
from query_rag import retrieve  # использует Chroma + E5 + Ollama(Qwen)

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = SCRIPT_DIR.parent / "data" / "evaluation" / "test_set.jsonl"

TOP_K = 5

def load_testset(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    # 1) Подключаем evaluator LLM для RAGAS (через Ollama)
    # Ollama даёт OpenAI-compatible API на /v1 если включено (как в доке ragas quickstart)
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    evaluator_llm = llm_factory(
        "qwen2.5:7b",
        provider="openai",
        client=client,
        temperature=0,
        max_tokens=2048
    )

    # 2) Собираем eval dataset
    test_rows = load_testset(TEST_SET_PATH)
    data = {"question": [], "contexts": [],  "ground_truth": []}

    for r in test_rows:
        q = r["question"]
        gt = r["answer"]

        # retrieval
        docs = retrieve(q, top_k=TOP_K)
        contexts = [d["text"] [:600] for d in docs]

        # generation (твоя RAG-генерация)
        #ans = ask_labkovsky(q, top_k=TOP_K, verbose=False)

        data["question"].append(q)
        data["contexts"].append(contexts)
        #data["answer"].append(ans)
        data["ground_truth"].append(gt)

    dataset = Dataset.from_dict(data)

    # 3) Считаем метрики
    result = evaluate(
        dataset,
        metrics=[context_recall],
        llm=evaluator_llm,
    )

    print("\n=== RAGAS RESULTS (mean) ===")
    print(result)

    # 4) Если хочешь увидеть построчно:
    df = result.to_pandas()
    print("\n=== Per-sample ===")
    #print(df[["question", "context_recall"]])
    for q, score in zip(data["question"], df["context_recall"]):
        print(f"{score:.2f} | {q}")


if __name__ == "__main__":
    main()
