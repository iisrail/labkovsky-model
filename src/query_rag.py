#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline - Query
Поиск релевантных чанков + генерация ответа через Qwen2.5

Использование:
1. Сначала запусти build_index.py
2. Убедись что Ollama запущен: ollama serve
3. python query_rag.py

Или импортируй как модуль:
    from query_rag import ask_labkovsky
    answer = ask_labkovsky("Как полюбить себя?")
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# ============================================================
# НАСТРОЙКИ
# ============================================================

SCRIPT_DIR = Path(__file__).parent
CHROMA_DIR = SCRIPT_DIR / "chroma_db"

# Embedding модель (та же что при индексации!)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# LLM для генерации
LLM_MODEL = "qwen2.5:14b"

# Сколько чанков искать
TOP_K = 5

# System prompt - личность Лабковского
SYSTEM_PROMPT = """Ты — Михаил Лабковский, известный российский психолог.

Твой стиль:
- Прямой, без воды
- С юмором и иронией
- Иногда провокационный
- Говоришь простым языком, без научных терминов
- Часто используешь примеры из жизни
- Твоя главная идея: делай только то, что хочешь

Твои 6 правил:
1. Делать только то, что хочется
2. Не делать того, чего не хочется
3. Сразу говорить о том, что не нравится
4. Не отвечать, когда не спрашивают
5. Отвечать только на вопрос
6. Выясняя отношения, говорить только о себе

Отвечай на вопросы используя контекст ниже. Если в контексте нет информации — отвечай исходя из своей философии и принципов.
"""

# ============================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================

# Глобальные переменные для кэширования
_model = None
_collection = None


def init():
    """Инициализация модели и ChromaDB"""
    global _model, _collection
    
    if _model is None:
        print("🤖 Загрузка embedding модели...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    
    if _collection is None:
        print("💾 Подключение к ChromaDB...")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection("labkovsky")
        print(f"✅ Загружено {_collection.count()} документов")
    
    return _model, _collection


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
    model, collection = init()
    
    # Для e5 моделей нужен префикс "query:"
    query_embedding = model.encode(f"query: {query}")
    
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

def generate(query: str, context_docs: list) -> str:
    """
    Генерация ответа через Qwen2.5
    
    Args:
        query: Вопрос пользователя
        context_docs: Релевантные документы из retrieval
    
    Returns:
        Ответ в стиле Лабковского
    """
    # Форматируем контекст
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc['metadata'].get('source', 'unknown')
        if source == 'qa':
            context_parts.append(f"[{i}] {doc['text']}")
        else:
            title = doc['metadata'].get('article_id') or doc['metadata'].get('interview_id') or ''
            context_parts.append(f"[{i}] ({title}) {doc['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Формируем промпт
    user_message = f"""Контекст из моих материалов:

{context}

---

Вопрос: {query}

Ответь как Михаил Лабковский:"""

    # Генерация через Ollama
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    
    return response['message']['content']


# ============================================================
# MAIN FUNCTION
# ============================================================

def ask_labkovsky(query: str, top_k: int = TOP_K, verbose: bool = False) -> str:
    """
    Главная функция: задай вопрос — получи ответ от Лабковского
    
    Args:
        query: Вопрос
        top_k: Сколько документов использовать для контекста
        verbose: Показывать найденные документы
    
    Returns:
        Ответ в стиле Лабковского
    """
    # Retrieval
    docs = retrieve(query, top_k)
    
    if verbose:
        print("\n📚 Найденные документы:")
        for i, doc in enumerate(docs, 1):
            source = doc['metadata'].get('source', '?')
            dist = doc['distance']
            preview = doc['text'][:100] + "..."
            print(f"  [{i}] ({source}, dist={dist:.3f}) {preview}")
        print()
    
    # Generation
    answer = generate(query, docs)
    
    return answer


# ============================================================
# CLI
# ============================================================

def main():
    """Интерактивный режим"""
    print("="*60)
    print("🎤 Спроси Лабковского!")
    print("="*60)
    print("Команды: 'выход' или 'exit' для выхода")
    print("         'verbose' для показа источников")
    print()
    
    verbose = False
    
    # Инициализация
    init()
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
        
        print("\n🤔 Думаю...\n")
        
        try:
            answer = ask_labkovsky(query, verbose=verbose)
            print(f"Лабковский: {answer}\n")
        except Exception as e:
            print(f"❌ Ошибка: {e}\n")
            print("Убедись что Ollama запущен: ollama serve")


if __name__ == "__main__":
    main()
