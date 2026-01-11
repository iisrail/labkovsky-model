#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline - Query (with Fine-tuned Model)
Поиск релевантных чанков + генерация ответа через fine-tuned Qwen2.5 + LoRA

Использование:
1. Сначала запусти build_index.py
2. python query_rag.py

Или импортируй как модуль:
    from query_rag import ask_labkovsky
    answer = ask_labkovsky("Как полюбить себя?")
"""

import argparse
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import chromadb
from config import CHROMA_DIR, MODELS_DIR

# ============================================================
# НАСТРОЙКИ
# ============================================================

# Embedding модель (та же что при индексации!)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Fine-tuned LLM
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = MODELS_DIR / "labkovsky-qwen7b-lora"

# Сколько чанков искать
TOP_K = 5

# Prompt modes: "full", "minimal", "none"
PROMPT_MODE = "full"

SYSTEM_PROMPTS = {
    "full": """Ты — Михаил Лабковский, известный российский психолог.

Твой стиль:
- Прямой, без воды
- С юмором и иронией
- Иногда провокационный
- Говоришь простым языком, без научных терминов
- Часто используешь примеры из жизни
- Твоя главная идея: психические проблемы формируются из-за закреплённых поведенческих реакций, и избавиться от них можно, системно применяя 6 правил.

Отвечай на вопросы используя контекст ниже.""",

    "minimal": """Ты — Михаил Лабковский, психолог. Отвечай используя контекст ниже.""",

    "none": None
}

# ============================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================

# Глобальные переменные для кэширования
_embed_model = None
_collection = None
_llm = None
_tokenizer = None


def init_retrieval():
    """Инициализация embedding модели и ChromaDB"""
    global _embed_model, _collection
    
    if _embed_model is None:
        print("🤖 Загрузка embedding модели...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    if _collection is None:
        print("💾 Подключение к ChromaDB...")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection("labkovsky")
        print(f"✅ Загружено {_collection.count()} документов")
    
    return _embed_model, _collection


def init_llm(use_lora: bool = True):
    """Инициализация fine-tuned LLM с LoRA"""
    global _llm, _tokenizer
    
    if _llm is None:
        print(f"🤖 Загрузка LLM: {MODEL_NAME}")

        
        # 4-bit quantization for 12GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        if use_lora:
            print(f"   LoRA: {LORA_PATH}")
            _llm = PeftModel.from_pretrained(base_model, str(LORA_PATH))
        else:
            print("   (base model, no LoRA)")
            _llm = base_model
        # Load LoRA adapter
        
        _llm.eval()
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        print("✅ LLM загружен")
    
    return _llm, _tokenizer


def init(use_lora: bool = True):
    """Инициализация всех компонентов"""
    init_retrieval()
    init_llm(use_lora)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query: str, top_k: int = TOP_K) -> list:
    """
    Поиск релевантных документов
    
    Args:
        query: Вопрос пользователя
        top_k: Сколько документов вернуть
    
    Returns:
        Список документов с метаданными
    """
    embed_model, collection = init_retrieval()
    
    # Для e5 моделей нужен префикс "query:"
    query_embedding = embed_model.encode(f"query: {query}")
    
    # Поиск в ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Форматируем результаты
    documents = []
    for i in range(len(results['ids'][0])):
        documents.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })
    
    return documents


# ============================================================
# GENERATION
# ============================================================

def generate(query: str, context_docs: list, prompt_mode: str = None) -> str:
    if prompt_mode is None:
        prompt_mode = PROMPT_MODE

    llm, tokenizer = init_llm()

    # ---- Формируем контекст ----
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc['metadata'].get('source', 'unknown')
        if source == 'qa':
            context_parts.append(f"[{i}] {doc['text']}")
        else:
            identifier = (
                doc['metadata'].get('article_id')
                or doc['metadata'].get('interview_id')
                or doc['metadata'].get('book_id')
                or doc['metadata'].get('chapter_id')
                or ''
            )
            context_parts.append(f"[{i}] ({identifier}) {doc['text']}")

    context = "\n\n".join(context_parts)

    user_message = f"""Контекст из моих материалов:

{context}

---

Вопрос: {query}"""

    system_prompt = SYSTEM_PROMPTS.get(prompt_mode)

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    else:
        messages = [{"role": "user", "content": user_message}]

    # ---- ВАЖНО: tokenize=True ----
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(llm.device)

    input_len = inputs.shape[-1]

    with torch.inference_mode():
        outputs = llm.generate(
            inputs,
            max_new_tokens=1536,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # ---- КОРРЕКТНОЕ извлечение ответа ----
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response



# ============================================================
# MAIN FUNCTION
# ============================================================

def ask_labkovsky(query: str, top_k: int = TOP_K, verbose: bool = False, prompt_mode: str = None) -> str:
    """
    Главная функция: задай вопрос — получи ответ от Лабковского
    
    Args:
        query: Вопрос
        top_k: Сколько документов использовать для контекста
        verbose: Показывать найденные документы
        prompt_mode: "full", "minimal", or "none"
    
    Returns:
        Ответ в стиле Лабковского
    """
    # Retrieval
    docs = retrieve(query, top_k)
    
    if verbose:
        print("\n📚 Найденные документы:")
        for i, doc in enumerate(docs, 1):
            meta = doc['metadata']
            source = meta.get('source', 'unknown')
            dist = doc['distance']

            # Build display ID based on source
            if source == 'qa':
                display_id = meta.get('id', '?')
            elif source == 'article':
                display_id = f"{meta.get('article_id', '?')[:20]}..._chunk{meta.get('chunk_id')}"
            elif source == 'interview':
                display_id = f"int_chunk{meta.get('chunk_id')}"
            elif source == 'book':
                display_id = f"{meta.get('book_id', '?')}_{meta.get('chapter_id', '?')}"
            else:
                display_id = '?'
            
            preview = doc['text'][:80] + "..."
            print(f"  [{i}] ({source}, {display_id}, dist={dist:.3f}) {preview}")
        print()
    
    # Generation
    answer = generate(query, docs, prompt_mode)
    
    return answer


# ============================================================
# CLI
# ============================================================

def main():
    """Интерактивный режим"""
    global PROMPT_MODE

        # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-lora', action='store_true', help='Use base model without LoRA')
    args = parser.parse_args()
    use_lora = not args.no_lora
    
    print("="*60)
    print("🎤 Спроси Лабковского! (Fine-tuned)")
    print("="*60)
    print("Команды:")
    print("  'выход' или 'exit' - выход")
    print("  'verbose' - показать источники")
    print("  'full' / 'minimal' / 'none' - режим промпта")
    print()
    
    verbose = False
    
    # Инициализация
    print("Загрузка моделей...")
    init(use_lora)
    print(f"\nРежим промпта: {PROMPT_MODE}")
    print()
    
    while True:
        try:
            query = input("Ты: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Пока!")
            break
        
        if not query:
            continue
        
        if query.lower() in ['выход', 'exit', 'quit', 'q']:
            print("👋 Пока!")
            break
        
        if query.lower() == 'verbose':
            verbose = not verbose
            print(f"Verbose режим: {'включен' if verbose else 'выключен'}")
            continue
        
        if query.lower() in ['full', 'minimal', 'none']:
            PROMPT_MODE = query.lower()
            print(f"Режим промпта: {PROMPT_MODE}")
            continue
        
        print("\n🤔 Думаю...\n")
        
        try:
            answer = ask_labkovsky(query, verbose=verbose)
            print(f"Лабковский: {answer}\n")
        except Exception as e:
            print(f"❌ Ошибка: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()