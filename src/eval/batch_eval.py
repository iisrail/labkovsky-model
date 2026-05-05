#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_eval.py — Automated batch evaluation for DPO data collection.

Runs a list of test questions N times each through the RAG pipeline,
auto-detects copies (high overlap with retrieved QA doc), and saves
results for manual/LLM labeling.

Usage:
    python src/eval/batch_eval.py                          # run all 36 questions x3
    python src/eval/batch_eval.py --runs 5                 # 5 runs per question
    python src/eval/batch_eval.py --lora-path models/labkovsky-rag-context-lora-v10
    python src/eval/batch_eval.py --label-only             # skip inference, just re-label existing results
"""

import argparse
import json
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Add project root so we can import from src.rag
sys.path.insert(0, str(PROJECT_ROOT))

QA_CLEAN_PATH = PROJECT_ROOT / "data" / "fine_tuning" / "qa_clean.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_batch.jsonl"

# ============================================================
# TEST QUESTIONS (36 from corpus, grouped by decision_type)
# ============================================================

TEST_QUESTIONS = [
    # ADDICTION_PATTERN (3)
    {"qa_id": "srJvn19GKNA_05", "dt": "ADDICTION_PATTERN",
     "question": "Как справиться с едой по ночам? Как от этого избавиться? И зависимость ли это? Как контролировать себя, не заедать стресс? Начинаю есть, а скорее жрать всё подряд. И зависимость от сладкого и фастфуда. Пока всё не съем, буду есть. Куда копать, где искать причину?"},
    {"qa_id": "C1M2DIq0DsM_02", "dt": "ADDICTION_PATTERN",
     "question": "Скупаю вещи и трачу все деньги. Это шопоголизм?"},
    {"qa_id": "hochu_i_budu_podrostkov_04", "dt": "ADDICTION_PATTERN",
     "question": "У нас благополучная семья, а дочь начала баловаться наркотиками. Как это объяснить?"},
    # AFFECTIVE_ADDICTION (5)
    {"qa_id": "MLe9XIXlZYA_01", "dt": "AFFECTIVE_ADDICTION",
     "question": "Как понять, какой партнёр мне нужен и что такое любовь? Как отличить любовь от привязанности или влюблённости?"},
    {"qa_id": "8yFloIeq7kg_04", "dt": "AFFECTIVE_ADDICTION",
     "question": "Влюбляюсь только в недоступных мужчин. Как разорвать круг?"},
    {"qa_id": "tg_2025_01_cold_men_01", "dt": "AFFECTIVE_ADDICTION",
     "question": "Меня тянет к холодным, жёстким мужчинам. А заботливые становятся неинтересны. Почему так?"},
    {"qa_id": "tg_2025_01_husband_left_after_15years_01", "dt": "AFFECTIVE_ADDICTION",
     "question": "Была замужем 15 лет, он полюбил другую и ушёл. Как теперь жить? Как перестать его любить?"},
    {"qa_id": "VVm2iO_XmPE_01", "dt": "AFFECTIVE_ADDICTION",
     "question": "У меня нет отношений полтора года. Я хожу на свидания, три-четыре, и всё заканчивается. Каждый раз одно и то же — я как будто делаю всё для того, чтобы перестать общаться с человеком. Всё начинается хорошо, романтика, прогулки. Потом мне начинает казаться, что я ему недостаточно сильно нравлюсь — не дарит цветы, меньше пишет, не соответствует каким-то моментам."},
    # ANXIETY_MANAGEMENT (5)
    {"qa_id": "GJ4Kip53J60_01", "dt": "ANXIETY_MANAGEMENT",
     "question": "Михаил, я человек очень тревожный и очень часто не могу остановить вот это состояние прямо до тошноты. Не знаю, как с этим справляться. Сейчас дошло до того, что я придумываю себе, что может произойти со мной на следующий день. Это всё начинается с вечера. Засыпаю к 5:00 утра, весь день потом разбитый. И то, что я себе надумал, часто происходит. Что это такое и как это остановить?"},
    {"qa_id": "Tx3-1QGpp_c_02", "dt": "ANXIETY_MANAGEMENT",
     "question": "Как помочь себе при панической атаке?"},
    {"qa_id": "tg_2025_01_sleep_problems_01", "dt": "ANXIETY_MANAGEMENT",
     "question": "Плохо сплю, по 3-4 часа. Иногда тревожат мысли, но чаще просто не могу спать. Что не так?"},
    {"qa_id": "tg_2025_01_trevoga_beremennost_01", "dt": "ANXIETY_MANAGEMENT",
     "question": "Планируем беременность, но я тревожусь, что родится нездоровый малыш. Тревога поглотила и не даёт радоваться. По медицинским показаниям мы здоровы."},
    {"qa_id": "1PXu55Yi2Gc_01", "dt": "ANXIETY_MANAGEMENT",
     "question": "Михаил, мне 45. Я работаю в IT и давно не хочу идти на работу. Думаю о смене профессии, но держат деньги и комфорт. Как понять, это реальное желание или попытка убежать от трудностей?"},
    # CLINICAL_ESCALATION (4)
    {"qa_id": "SDioKXY5U6k_02", "dt": "CLINICAL_ESCALATION",
     "question": "Мама хотела сжечь меня в детстве. Как простить? «Мама в детстве хотела меня сжечь, облила бензином, доказывая кому-то, что таким образом вернёт свои деньги. Папа тоже был рядом. Мне было 4–5 лет, сейчас мне 25, и этот груз я до сих пор тащу. Как простить?»"},
    {"qa_id": "tg_2025_01_grief_01", "dt": "CLINICAL_ESCALATION",
     "question": "Как пережить смерть мужа? Вместе с ним я словно потеряла и саму себя. Больно."},
    {"qa_id": "hochu_i_budu_podrostkov_03", "dt": "CLINICAL_ESCALATION",
     "question": "Подруга 12-летней дочери покончила с собой. Дочь в очень подавленном состоянии. Как нам ей помочь?"},
    {"qa_id": "16pNCx8XDZU_01_child", "dt": "CLINICAL_ESCALATION",
     "question": "Мне 31 год, у нас тёплые отношения в семье. Но сын, когда сильно рассердится, начинает угрожать — обещает перерезать горло ножом, сбросить бомбу, сбить с ног, побить, предлагает выброситься из окна. Иногда бросается с кулаками и кусается. Дома боевики не смотрим, мы с женой спокойные, не скандалим, голос не повышаем. С другими детьми сын не конфликтует, воспитатели хвалят. Как понимать такое поведение, с чем связано и как реагировать на взрывы ярости?"},
    # DEPENDENCY_BOUNDARIES (5)
    {"qa_id": "jz6VYBZvaeU_01", "dt": "DEPENDENCY_BOUNDARIES",
     "question": "Михаил, если мама, а мне уже 34 года, всё время говорит, что страшнее меня на свете нет, что просто надо же, как не повезло вообще девочке — родилась и такая убогая. Она говорит это даже моему мужу: «Господи, ну слава богу, что ты её взял такую страшную-то. Ой, а не дай бог это детям передастся». Заткнуть маму я не могу, потому что встречаем мы её даже иногда на улице — живём в соседних домах. Это какой-то кошмар. Что ей сказать, как сделать? Маму я люблю, но время от времени просто ненавижу."},
    {"qa_id": "tg_2025_01_people_pleasing_01", "dt": "DEPENDENCY_BOUNDARIES",
     "question": "Постоянно стараюсь быть удобной для всех и боюсь отказать. Потом на себя злюсь и на них обижаюсь. Как научиться отказывать?"},
    {"qa_id": "tg_2025_01_coerced_sex_01", "dt": "DEPENDENCY_BOUNDARIES",
     "question": "Парень почти принудил меня к сексу, он остановился, но было неприятно. Это значит, что он абьюзер?"},
    {"qa_id": "tg_2025_01_inlaws_boundaries_01", "dt": "DEPENDENCY_BOUNDARIES",
     "question": "Родители моего мужчины лезут в наши отношения. Кто должен ставить границы — я или он?"},
    {"qa_id": "tg_2024_002", "dt": "DEPENDENCY_BOUNDARIES",
     "question": "Я много работаю. Не могу сказать, что работа мне не нравится, но я реально МНОГО работаю. Проблема в том, что начальник принимает это за должное и не повышает зарплату. Как повлиять на ситуацию?"},
    # FEAR_SCENARIO_COPING (4)
    {"qa_id": "GJ4Kip53J60_02", "dt": "FEAR_SCENARIO_COPING",
     "question": "Михаил, как можно побороть страх начать что-то новое? Есть возможность, желание, но страх, что не получится, просто вводит в какое-то оцепенение. Я просто всё откладываю, ничего не делаю, просто не могу себя заставить. Возможно, это из-за детства. Никогда не забуду слова бабушки: «Ты победнее, поскромнее веди себя». И это было очень часто. Почему-то родные тоже никогда меня не поддерживали в молодости. Если я что-то хотела начать, то натыкалась на стену непонимания и отрицания. В итоге я просто сдавалась и отказывалась от идей и возвращалась туда, где привычней."},
    {"qa_id": "q1eArxEvCfU_01", "dt": "FEAR_SCENARIO_COPING",
     "question": "Михаил, как адаптироваться к изменениям? Резкий переезд в другую страну к парню, с которым вместе почти полгода. Не знаю язык, здесь нет друзей, и я очень скучаю по дому. Вернуться пока не могу."},
    {"qa_id": "MhBzb1l-9Hs_01", "dt": "FEAR_SCENARIO_COPING",
     "question": "Михаил, на работе отдел расформировали. Дали 2 месяца на поиск другой должности с сохранением зарплаты. Как решить — соглашаться на меньшую зарплату или уехать в город к молодому человеку и искать другую работу?"},
    {"qa_id": "tg_2025_01_utrata_pes_01", "dt": "FEAR_SCENARIO_COPING",
     "question": "Недавно скончался мой пёс, с которым я прожила 12 лет. Как справиться с утратой?"},
    # PARENTING_MODEL (5)
    {"qa_id": "fSiOrfDlQfo_01", "dt": "PARENTING_MODEL",
     "question": "Как ограничить ребёнка, который хочет только играть в компьютер?"},
    {"qa_id": "4YYKqz17dok_03", "dt": "PARENTING_MODEL",
     "question": "Поймала дочь на воровстве. Что делать?"},
    {"qa_id": "tg_2025_01_parenting_conflict_01", "dt": "PARENTING_MODEL",
     "question": "У нас с мужем разные подходы к воспитанию. Если я говорю «драться нельзя», муж говорит «драться можно». Сыну 3 года. Как найти компромисс?"},
    {"qa_id": "hochu_i_budu_podrostkov_02", "dt": "PARENTING_MODEL",
     "question": "Случайно прочла переписку сына во «ВКонтакте», из которой понятно, что у него сексуальная связь с мужчиной. Виду не подаю, но что делать, не знаю."},
    {"qa_id": "tg_2024_018", "dt": "PARENTING_MODEL",
     "question": "Разругались с дочерью. Она сдала ЕГЭ в этом году, результаты неплохие. Только теперь она не хочет поступать туда, куда мы договорились идти. В её голове резко всё поменялось! Не знаю, как быть и что делать."},
    # SELF_ESTEEM_CORRECTIVE (5)
    {"qa_id": "tg_2025_01_scar_01", "dt": "SELF_ESTEEM_CORRECTIVE",
     "question": "У меня остался большой шрам на животе после операции. Из-за этого чувствую себя не сексуальной. Как перестроиться?"},
    {"qa_id": "-aCsNAR0w38_03", "dt": "SELF_ESTEEM_CORRECTIVE",
     "question": "Купил диплом и устроился в банк. Теперь хочу пойти работать по первому образованию, но боюсь чувствовать себя самозванцем."},
    {"qa_id": "tg_2025_01_meaning_60_01", "dt": "SELF_ESTEEM_CORRECTIVE",
     "question": "За что зацепиться в жизни, если тебе за 60, пенсия мизерная и все родственники ушли?"},
    {"qa_id": "tg_2024_021", "dt": "SELF_ESTEEM_CORRECTIVE",
     "question": "Не могу смириться с несправедливостью. Почему у моей подруги богатый отец, который подарил ей квартиру, а я пашу 24/7? Почему у какой-то девушки точёная фигура и при этом она может есть всё, что захочет, а я толстею от воздуха? Почему одним всё, а другим — ничего?"},
    {"qa_id": "8BhceyPaUe8_11", "dt": "SELF_ESTEEM_CORRECTIVE",
     "question": "В моей жизни есть конфликт уверенности в себе и религиозности. Есть амбиции, но при этом боюсь податься греху гордыни. Как быть?"},
]

# ============================================================
# COPY DETECTION
# ============================================================

def text_overlap(response: str, doc_text: str) -> float:
    """
    Compute text overlap ratio between response and a document.
    Uses SequenceMatcher for fuzzy matching (handles paraphrases).
    Returns ratio 0.0-1.0.
    """
    # Normalize whitespace
    r = " ".join(response.lower().split())
    d = " ".join(doc_text.lower().split())
    if not r or not d:
        return 0.0
    return SequenceMatcher(None, r, d).ratio()


def detect_copy(response: str, docs: list, threshold: float = 0.60) -> dict:
    """
    Check if response copies from any retrieved document.

    Returns dict with:
      - is_copy: bool
      - max_overlap: float (0-1)
      - copied_from: doc id or None
      - copied_source_type: str or None
    """
    max_overlap = 0.0
    copied_from = None
    copied_source = None

    for doc in docs:
        overlap = text_overlap(response, doc["text"])
        if overlap > max_overlap:
            max_overlap = overlap
            copied_from = doc.get("id")
            copied_source = doc.get("metadata", {}).get("source_type")

    return {
        "is_copy": max_overlap >= threshold,
        "max_overlap": round(max_overlap, 3),
        "copied_from": copied_from if max_overlap >= threshold else None,
        "copied_source_type": copied_source if max_overlap >= threshold else None,
    }


def load_reference_answers(qa_path: Path) -> dict:
    """Load qa_id -> answer mapping for reference comparison."""
    refs = {}
    if not qa_path.exists():
        return refs
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qa_id = obj.get("id", "")
            answer = obj.get("answer", "")
            if qa_id and answer:
                refs[qa_id] = answer
    return refs


# ============================================================
# BATCH INFERENCE
# ============================================================

def run_batch(args):
    """Run all test questions through the RAG pipeline."""
    # Import RAG pipeline (needs GPU)
    from src.rag.query_rag import (
        init_embedding, init_retrieval, init_llm,
        retrieve, generate, CHROMA_DIR, LORA_PATH,
        _resolve_best_checkpoint,
    )

    lora_path = None if args.no_lora else Path(args.lora_path)

    # Initialize once
    print("Loading models...")
    init_embedding()
    init_retrieval(Path(args.chroma_dir))
    init_llm(use_lora=not args.no_lora, lora_path=lora_path)

    resolved_lora = _resolve_best_checkpoint(lora_path) if lora_path else None
    if args.no_lora:
        model_label = "base-no-lora"
    else:
        try:
            model_label = str(resolved_lora.relative_to(PROJECT_ROOT))
        except (ValueError, TypeError):
            model_label = str(resolved_lora) if resolved_lora else str(lora_path)

    print(f"Model: {model_label}")

    # Load reference answers
    refs = load_reference_answers(QA_CLEAN_PATH)

    # Load existing results to support resume
    existing = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                key = (obj.get("qa_id"), obj.get("run"))
                existing.add(key)
        print(f"Resuming: {len(existing)} results already done")

    total = len(TEST_QUESTIONS) * args.runs
    done = 0

    for q_idx, tq in enumerate(TEST_QUESTIONS):
        qa_id = tq["qa_id"]
        question = tq["question"]
        dt = tq["dt"]
        ref_answer = refs.get(qa_id, "")

        for run in range(1, args.runs + 1):
            # Skip if already done (resume mode)
            if (qa_id, run) in existing:
                done += 1
                continue

            done += 1
            print(f"\n{'='*60}")
            print(f"[{done}/{total}] Q{q_idx+1} run {run}/{args.runs} | {dt} | {qa_id}")
            print(f"Q: {question[:100]}...")

            try:
                # Retrieve
                docs = retrieve(question)
                docs.sort(key=lambda d: d["distance"])

                # Generate
                response = generate(question, docs, use_lora=not args.no_lora, lora_path=lora_path)

                # Copy detection
                copy_info = detect_copy(response, docs)

                # Reference overlap (with original answer from corpus)
                ref_overlap = text_overlap(response, ref_answer) if ref_answer else 0.0

                # Build result entry
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model_label,
                    "qa_id": qa_id,
                    "dt": dt,
                    "run": run,
                    "question": question,
                    "response": response,
                    "response_len": len(response),
                    "docs_used": len(docs),
                    "best_distance": docs[0]["distance"] if docs else None,
                    "doc_ids": [d["id"] for d in docs],
                    "doc_texts": [d["text"][:500] for d in docs],  # truncated for review
                    "doc_source_types": [d.get("metadata", {}).get("source_type", "") for d in docs],
                    # Copy detection
                    "is_copy": copy_info["is_copy"],
                    "max_doc_overlap": copy_info["max_overlap"],
                    "copied_from": copy_info["copied_from"],
                    "copied_source_type": copy_info["copied_source_type"],
                    # Reference overlap
                    "ref_overlap": round(ref_overlap, 3),
                    "ref_answer": ref_answer[:300] if ref_answer else "",
                    # Labels (to be filled by labeling step)
                    "auto_label": "copy" if copy_info["is_copy"] else "",
                    "label": "",  # final label: good / bad / copy
                    "label_reason": "",
                }

                # Save immediately (append)
                OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                status = "COPY" if copy_info["is_copy"] else "ok"
                print(f"R: {response[:150]}...")
                print(f"[{status}] overlap={copy_info['max_overlap']:.2f} ref_overlap={ref_overlap:.2f} len={len(response)}")

            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                # Save error entry
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model_label,
                    "qa_id": qa_id,
                    "dt": dt,
                    "run": run,
                    "question": question,
                    "response": f"ERROR: {e}",
                    "auto_label": "error",
                    "label": "",
                    "label_reason": "",
                }
                with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! {done} results saved to {OUTPUT_PATH}")
    print(f"Run --label-only to see auto-labels and stats")


# ============================================================
# LABELING & STATS
# ============================================================

def label_and_stats(args):
    """Print stats from existing batch results."""
    if not OUTPUT_PATH.exists():
        print(f"No results found at {OUTPUT_PATH}")
        return

    results = []
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Total results: {len(results)}")

    # Stats
    copies = [r for r in results if r.get("is_copy")]
    errors = [r for r in results if r.get("auto_label") == "error"]

    print(f"Auto-detected copies: {len(copies)} ({len(copies)/len(results)*100:.0f}%)")
    print(f"Errors: {len(errors)}")

    # Per-question stats
    from collections import defaultdict
    by_q = defaultdict(list)
    for r in results:
        by_q[r["qa_id"]].append(r)

    print(f"\nPer-question breakdown ({len(by_q)} questions):")
    print(f"{'qa_id':<40} {'runs':>4} {'copies':>6} {'avg_overlap':>11} {'avg_ref':>8} {'avg_len':>7}")
    print("-" * 80)

    for qa_id, runs in sorted(by_q.items()):
        n = len(runs)
        n_copy = sum(1 for r in runs if r.get("is_copy"))
        avg_overlap = sum(r.get("max_doc_overlap", 0) for r in runs) / n
        avg_ref = sum(r.get("ref_overlap", 0) for r in runs) / n
        avg_len = sum(r.get("response_len", 0) for r in runs) / n
        flag = " <<<" if n_copy > 0 else ""
        print(f"{qa_id:<40} {n:>4} {n_copy:>6} {avg_overlap:>11.3f} {avg_ref:>8.3f} {avg_len:>7.0f}{flag}")

    # Consistency analysis: how different are runs of the same question?
    print(f"\nConsistency (response similarity between runs of same question):")
    for qa_id, runs in sorted(by_q.items()):
        if len(runs) < 2:
            continue
        # Compare all pairs
        sims = []
        for i in range(len(runs)):
            for j in range(i+1, len(runs)):
                r1 = runs[i].get("response", "")
                r2 = runs[j].get("response", "")
                if r1 and r2:
                    sims.append(text_overlap(r1, r2))
        if sims:
            avg_sim = sum(sims) / len(sims)
            dt = runs[0].get("dt", "")
            flag = " (unstable)" if avg_sim < 0.3 else " (stable)" if avg_sim > 0.7 else ""
            print(f"  {qa_id:<40} avg_sim={avg_sim:.3f}{flag}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Batch eval for DPO data collection")
    parser.add_argument("--runs", type=int, default=8, help="Number of runs per question (default: 8)")
    parser.add_argument("--lora-path", type=str,
                        default=str(PROJECT_ROOT / "models" / "labkovsky-rag-context-lora-v10"),
                        help="LoRA adapter path")
    parser.add_argument("--no-lora", action="store_true", help="Use base model without LoRA")
    parser.add_argument("--chroma-dir", type=str,
                        default=str(PROJECT_ROOT / "chroma_db"),
                        help="ChromaDB directory")
    parser.add_argument("--label-only", action="store_true", help="Skip inference, show stats only")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing results (default: True)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (delete existing results)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output JSONL path")
    args = parser.parse_args()

    output_path = Path(args.output)
    # Update module-level for use in run_batch/label_and_stats
    globals()["OUTPUT_PATH"] = output_path

    if args.fresh and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        print(f"Deleted {OUTPUT_PATH}")

    if args.label_only:
        label_and_stats(args)
    else:
        run_batch(args)
        print()
        label_and_stats(args)


if __name__ == "__main__":
    main()
