#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check training data for Labkovsky's core ideas using SEMANTIC SIMILARITY.
FIXED: Higher threshold, better counting logic.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
_possible_paths = [
    SCRIPT_DIR / "data" / "fine_tuning" / "qa_rs_corpus_short.jsonl",
    Path(r"C:\Projects\projects_py\labkovsky-model\data\fine_tuning\qa_rs_corpus_short.jsonl"),
]

DATA_PATH = None
for p in _possible_paths:
    if p.exists():
        DATA_PATH = p
        break

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Higher threshold - 0.7+ means strong semantic match
HIGH_THRESHOLD = 0.75  # Strong match - clearly about this idea
MED_THRESHOLD = 0.65   # Moderate match - related to this idea

# ============================================================
# LABKOVSKY'S CORE IDEAS
# ============================================================

LABKOVSKY_IDEAS = {
    "self_priority": [
        "Вы самый важный человек в своей жизни. Ставьте себя на первое место.",
        "Любите себя, заботьтесь о себе в первую очередь.",
        "Ваши желания и потребности важнее чужих ожиданий.",
    ],
    
    "leave_bad_relationships": [
        "Уходите от человека, который вам не подходит.",
        "Не бегайте за теми, кто вас не хочет.",
        "Расставайтесь с теми, кто вас не ценит.",
        "Если отношения причиняют боль - уходите.",
    ],
    
    "no_compromise": [
        "Компромиссы вредят, потом будет больно.",
        "Не терпите то, что вам не нравится.",
        "Терпение в отношениях разрушает вас.",
    ],
    
    "ignore_critics": [
        "Не обращайте внимания на тех, кому вы не нравитесь.",
        "Плевать на то, что думают другие о вас.",
        "Мнение окружающих не должно влиять на вашу жизнь.",
    ],
    
    "six_rules": [
        "Делайте только то, что хочется.",
        "Не делайте того, что делать не хочется.",
        "Сразу говорите о том, что не нравится.",
        "Не отвечайте, когда не спрашивают.",
        "Отвечайте только на вопрос.",
        "Выясняя отношения, говорите только о себе.",
    ],
    
    "say_once_then_leave": [
        "Скажите один раз что не нравится. Если не изменится - уходите.",
        "Не повторяйте просьбы. Сказали раз - и всё.",
    ],
    
    "behavior_change": [
        "Меняйте поведение, а не копайтесь в причинах.",
        "Невроз лечится изменением поведения, а не анализом прошлого.",
    ],
}

# ============================================================
# FUNCTIONS
# ============================================================

def load_data(path):
    if path is None or not path.exists():
        print(f"❌ File not found: {path}")
        return []
    
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if not data.get("short_answer"):
                    records.append(data)
    return records


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def analyze_similarities(records, model):
    """
    Compute similarity scores for all answers against all ideas.
    Returns detailed statistics.
    """
    print(f"📊 Embedding {len(records)} answers...")
    
    # Embed all answers
    answers = [f"passage: {r['answer']}" for r in records]
    answer_embeddings = model.encode(answers, show_progress_bar=True)
    
    # Embed all ideas
    all_ideas = []
    idea_to_category = {}
    for category, texts in LABKOVSKY_IDEAS.items():
        for text in texts:
            all_ideas.append(f"query: {text}")
            idea_to_category[len(all_ideas)-1] = category
    
    print(f"   Embedding {len(all_ideas)} reference ideas...")
    idea_embeddings = model.encode(all_ideas)
    
    # Compute all similarities
    print(f"   Computing similarities...")
    
    # For each answer, find best matching category and its similarity
    answer_best_matches = []
    
    for ans_idx, ans_emb in enumerate(answer_embeddings):
        category_max_sims = {}
        
        for idea_idx, idea_emb in enumerate(idea_embeddings):
            cat = idea_to_category[idea_idx]
            sim = cosine_similarity(ans_emb, idea_emb)
            
            if cat not in category_max_sims or sim > category_max_sims[cat]["sim"]:
                category_max_sims[cat] = {
                    "sim": sim,
                    "idea_idx": idea_idx,
                }
        
        # Find best category for this answer
        best_cat = max(category_max_sims.keys(), key=lambda c: category_max_sims[c]["sim"])
        best_sim = category_max_sims[best_cat]["sim"]
        
        answer_best_matches.append({
            "answer_idx": ans_idx,
            "best_category": best_cat,
            "best_sim": best_sim,
            "all_categories": {c: v["sim"] for c, v in category_max_sims.items()},
            "question": records[ans_idx]["question"][:80],
            "answer": records[ans_idx]["answer"][:150],
        })
    
    return answer_best_matches


def main():
    print("=" * 70)
    print("🎯 LABKOVSKY SEMANTIC CONTENT CHECK (FIXED)")
    print(f"   Using: {EMBEDDING_MODEL}")
    print(f"   High threshold: {HIGH_THRESHOLD}")
    print(f"   Medium threshold: {MED_THRESHOLD}")
    print("=" * 70)
    
    # Load data
    records = load_data(DATA_PATH)
    if not records:
        return
    
    print(f"\n📂 Loaded {len(records)} answers\n")
    
    # Load model
    print(f"🤖 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("   Done.\n")
    
    # Analyze
    matches = analyze_similarities(records, model)
    
    # Statistics
    print("\n" + "=" * 70)
    print("📊 SIMILARITY DISTRIBUTION")
    print("=" * 70)
    
    sims = [m["best_sim"] for m in matches]
    print(f"\n   Best similarity per answer:")
    print(f"   Min:  {min(sims):.3f}")
    print(f"   Max:  {max(sims):.3f}")
    print(f"   Mean: {np.mean(sims):.3f}")
    print(f"   Median: {np.median(sims):.3f}")
    
    # Distribution buckets
    buckets = {
        "0.80+": len([s for s in sims if s >= 0.80]),
        "0.75-0.80": len([s for s in sims if 0.75 <= s < 0.80]),
        "0.70-0.75": len([s for s in sims if 0.70 <= s < 0.75]),
        "0.65-0.70": len([s for s in sims if 0.65 <= s < 0.70]),
        "0.60-0.65": len([s for s in sims if 0.60 <= s < 0.65]),
        "<0.60": len([s for s in sims if s < 0.60]),
    }
    
    print(f"\n   Distribution:")
    for bucket, count in buckets.items():
        pct = 100 * count / len(sims)
        bar = "█" * int(pct / 2)
        print(f"   {bucket:10} {bar:25} {count:3} ({pct:.1f}%)")
    
    # By category (high threshold)
    print("\n" + "=" * 70)
    print(f"📋 COVERAGE BY CATEGORY (threshold >= {HIGH_THRESHOLD})")
    print("=" * 70)
    
    category_counts = {cat: {"high": 0, "med": 0} for cat in LABKOVSKY_IDEAS.keys()}
    
    for m in matches:
        for cat, sim in m["all_categories"].items():
            if sim >= HIGH_THRESHOLD:
                category_counts[cat]["high"] += 1
            elif sim >= MED_THRESHOLD:
                category_counts[cat]["med"] += 1
    
    for cat in LABKOVSKY_IDEAS.keys():
        high = category_counts[cat]["high"]
        med = category_counts[cat]["med"]
        high_pct = 100 * high / len(records)
        med_pct = 100 * med / len(records)
        print(f"\n   {cat}:")
        print(f"      Strong (>={HIGH_THRESHOLD}): {high:3} ({high_pct:.1f}%)")
        print(f"      Moderate (>={MED_THRESHOLD}): {med:3} ({med_pct:.1f}%)")
    
    # Top examples
    print("\n" + "=" * 70)
    print("📝 TOP MATCHING ANSWERS (highest similarity)")
    print("=" * 70)
    
    top_matches = sorted(matches, key=lambda x: -x["best_sim"])[:5]
    
    for i, m in enumerate(top_matches, 1):
        print(f"\n{i}. Similarity: {m['best_sim']:.3f} | Category: {m['best_category']}")
        print(f"   Q: {m['question']}...")
        print(f"   A: {m['answer']}...")
    
    # Lowest examples
    print("\n" + "=" * 70)
    print("📝 LOWEST MATCHING ANSWERS (generic content?)")
    print("=" * 70)
    
    low_matches = sorted(matches, key=lambda x: x["best_sim"])[:5]
    
    for i, m in enumerate(low_matches, 1):
        print(f"\n{i}. Similarity: {m['best_sim']:.3f} | Category: {m['best_category']}")
        print(f"   Q: {m['question']}...")
        print(f"   A: {m['answer']}...")
    
    # Final assessment
    print("\n" + "=" * 70)
    print("📋 ASSESSMENT")
    print("=" * 70)
    
    high_count = len([s for s in sims if s >= HIGH_THRESHOLD])
    med_count = len([s for s in sims if s >= MED_THRESHOLD])
    
    high_pct = 100 * high_count / len(records)
    med_pct = 100 * med_count / len(records)
    
    print(f"\n   Strong matches (>={HIGH_THRESHOLD}): {high_count}/{len(records)} ({high_pct:.1f}%)")
    print(f"   Moderate+ matches (>={MED_THRESHOLD}): {med_count}/{len(records)} ({med_pct:.1f}%)")
    
    if high_pct < 20:
        print(f"\n   ⚠️ Low strong coverage. Most answers may be too generic.")
    elif high_pct < 40:
        print(f"\n   ℹ️ Moderate coverage. Some distinctive content present.")
    else:
        print(f"\n   ✅ Good coverage. Data contains distinctive Labkovsky ideas.")


if __name__ == "__main__":
    main()