#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from openai import OpenAI
import time
import os
from pathlib import Path

# Файлы
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR.parent / "data" / "articles_semantic.jsonl"
OUTPUT_FILE = SCRIPT_DIR.parent / "data" / "articles_with_questions.jsonl"

# Промпт для генерации вопросов
PROMPT_TEMPLATE = """Ты анализируешь текст психолога Михаила Лабковского.

Прочитай этот фрагмент и сгенерируй 2-3 вопроса, которые человек мог бы задать, если бы искал именно эту информацию.

Вопросы должны быть:
- На русском языке
- Естественными (как реальный человек спросил бы)
- Разными по формулировке
- Релевантными содержанию фрагмента

ТЕКСТ:
{text}

Ответь ТОЛЬКО JSON-массивом вопросов, без пояснений:
["вопрос 1", "вопрос 2", "вопрос 3"]"""


def generate_questions(client, text, retries=3):
    """Генерация вопросов через Claude API"""
    for attempt in range(retries):
        try:
            # response = client.messages.create(
            #     model="claude-sonnet-4-20250514",
            #     max_tokens=300,
            #     messages=[
            #         {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
            #     ]
            # )

            response = client.responses.create(
                model="gpt-4.1-mini",
                max_output_tokens=300,
                input=PROMPT_TEMPLATE.format(text=text)
            )            
            
            #content = response.content[0].text.strip()
            content = response.output_text.strip()
            
            # Убираем markdown-обёртки если есть
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            
            questions = json.loads(content)
            return questions
            
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  ⚠ API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    return []


def main():
    
    # Проверяем входной файл
    if not INPUT_FILE.exists():
        print(f"❌ Ошибка: файл {INPUT_FILE} не найден!")
        return
    
    # Инициализация клиента
    #client = anthropic.Anthropic()
    client = OpenAI()
    
    # Читаем входной файл
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    
    print(f"📄 Загружено {len(records)} чанков из {INPUT_FILE.name}")
    print("="*60)
    
    # Обрабатываем каждый чанк
    results = []
    total_questions = 0
    
    for i, record in enumerate(records):
        title_short = record['article_id'][:35] + "..." if len(record['article_id']) > 35 else record['article_id']
        print(f"[{i+1}/{len(records)}] {title_short} (chunk {record['chunk_id']})")
        
        # Генерируем вопросы
        questions = generate_questions(client, record['text'])
        
        # Добавляем в запись
        record['potential_questions'] = questions
        results.append(record)
        
        if questions:
            total_questions += len(questions)
            print(f"  ✓ {len(questions)} вопросов")
        else:
            print(f"  ✗ Не удалось")
        
        # Пауза между запросами
        time.sleep(0.3)
        
        # Сохраняем промежуточный результат каждые 20 чанков
        if (i + 1) % 20 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for rec in results:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f"  💾 Промежуточное сохранение ({i+1} чанков)")
    
    # Финальное сохранение
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Статистика
    print("\n" + "="*60)
    print("✅ ГОТОВО!")
    print(f"   Обработано чанков: {len(results)}")
    print(f"   Сгенерировано вопросов: {total_questions}")
    print(f"   Среднее вопросов на чанк: {total_questions/len(results):.1f}")
    print(f"   Результат: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()