#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline - Build Index
Создаёт embeddings и сохраняет в ChromaDB

Использование:
1. pip install chromadb sentence-transformers
2. python build_index.py

Входные файлы (папка data/):
- articles_with_questions.jsonl
- interviews.jsonl  
- qa_pairs.jsonl

Выход:
- ./chroma_db/ (vector store)
"""

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ============================================================
# НАСТРОЙКИ
# ============================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
CHROMA_DIR = SCRIPT_DIR.parent / "chroma_db"

# Embedding модель (отлично для русского)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Файлы для RAG
RAG_FILES = [
    "articles_with_questions.jsonl",
    "interviews.jsonl",
    "qa_pairs.jsonl"
]

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_jsonl(filepath):
    """Загрузка JSONL файла"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def prepare_documents():
    """
    Подготовка документов из всех источников
    Возвращает: texts, metadatas, ids
    """
    texts = []
    metadatas = []
    ids = []
    
    doc_id = 0
    
    for filename in RAG_FILES:
        filepath = DATA_DIR / filename
        
        if not filepath.exists():
            print(f"⚠ Файл не найден: {filepath}")
            continue
        
        records = load_jsonl(filepath)
        print(f"📄 {filename}: {len(records)} записей")
        
        for record in records:
            source = record.get("source", "unknown")
            
            # Формируем текст для embedding
            if source == "article":
                # Статьи: text + potential_questions
                text = record["text"]
                questions = record.get("potential_questions", [])
                if questions:
                    text = " ".join(questions) + " " + text
                
                metadata = {
                    "source": "article",
                    "article_id": record.get("article_id", ""),
                    "chunk_id": record.get("chunk_id", 0)
                }
                
            elif source == "interview":
                # Интервью: text + potential_questions
                text = record["text"]
                questions = record.get("potential_questions", [])
                if questions:
                    text = " ".join(questions) + " " + text
                
                metadata = {
                    "source": "interview",
                    "interview_id": record.get("interview_id", ""),
                    "topic": record.get("topic", ""),
                    "chunk_id": record.get("chunk_id", 0)
                }
                
            else:
                # Q&A pairs: question + answer
                question = record.get("question", "")
                answer = record.get("answer", "")
                text = f"Вопрос: {question} Ответ: {answer}"
                
                metadata = {
                    "source": "qa",
                    "video_id": record.get("video_id", ""),
                    "video_title": record.get("video_title", ""),
                    "question": question
                }
            
            # Для e5 моделей нужен префикс "passage:"
            text_for_embedding = f"passage: {text}"
            
            texts.append(text_for_embedding)
            metadatas.append(metadata)
            ids.append(f"doc_{doc_id}")
            doc_id += 1
    
    return texts, metadatas, ids


# ============================================================
# EMBEDDING + VECTOR STORE
# ============================================================

def build_index():
    """Основная функция построения индекса"""
    
    print("="*60)
    print("RAG Pipeline - Build Index")
    print("="*60)
    
    # 1. Загрузка данных
    print("\n📚 Загрузка данных...")
    texts, metadatas, ids = prepare_documents()
    print(f"✅ Всего документов: {len(texts)}")
    
    if not texts:
        print("❌ Нет данных для индексации!")
        return
    
    # 2. Загрузка embedding модели
    print(f"\n🤖 Загрузка модели {EMBEDDING_MODEL}...")
    print("   (первый раз скачает ~2GB)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Модель загружена")
    
    # 3. Создание embeddings
    print(f"\n🔢 Создание embeddings для {len(texts)} документов...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✅ Embeddings созданы: shape {embeddings.shape}")
    
    # 4. Сохранение в ChromaDB
    print(f"\n💾 Сохранение в ChromaDB ({CHROMA_DIR})...")
    
    # Удаляем старую базу если есть
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    
    # Создаём клиент
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Создаём коллекцию
    collection = client.create_collection(
        name="labkovsky",
        metadata={"description": "Labkovsky RAG knowledge base"}
    )
    
    # Добавляем документы
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=[t.replace("passage: ", "") for t in texts]  # сохраняем без префикса
    )
    
    print(f"✅ Сохранено {collection.count()} документов")
    
    # 5. Статистика
    print("\n" + "="*60)
    print("✅ ИНДЕКС СОЗДАН!")
    print("="*60)
    print(f"   Документов: {len(texts)}")
    print(f"   Embedding размер: {embeddings.shape[1]}")
    print(f"   Хранилище: {CHROMA_DIR}")
    print("\n   Теперь запусти: python query_rag.py")


if __name__ == "__main__":
    build_index()
