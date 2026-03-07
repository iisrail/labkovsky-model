#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build DPO (Direct Preference Optimization) pairs from labeled eval data.

Reads eval_batch_labeled.jsonl (288 entries, 36 questions x 8 runs),
groups by qa_id, and creates chosen/rejected pairs based on labels:
  good > ok > bad/copy

Output: data/fine_tuning/dpo_pairs.jsonl

Usage:
    source venv_wsl/bin/activate
    python src/fine_tuning/build_dpo_pairs.py
"""

import json
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "eval_batch_labeled.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "fine_tuning" / "dpo_pairs.jsonl"

MAX_PAIRS_PER_QUESTION = 4

# Label ranking: higher = better
LABEL_RANK = {
    "good": 3,
    "ok": 2,
    "bad": 0,
    "copy": 0,
}

# Pair strength for prioritization (chosen_label, rejected_label) -> priority
# Lower priority number = stronger contrast = picked first
PAIR_PRIORITY = {
    ("good", "bad"): 0,
    ("good", "copy"): 0,
    ("ok", "bad"): 1,
    ("ok", "copy"): 1,
    ("good", "ok"): 2,
}

# Must match query_rag.py inference format
SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reference materials.\n\n"
    "Book and article fragments contain core principles — use them as the foundation "
    "for your reasoning, but do not copy verbatim.\n"
    "QA examples show how to structure a response — use them as a guide for tone and format.\n\n"
    "{docs}\n\n"
    "Answer in Labkovsky's style: direct, confident, with specific recommendations. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def reconstruct_system_prompt(rec: dict) -> str:
    """Reconstruct system prompt from doc_texts and doc_source_types.

    Matches query_rag.py inference format:
      [Book/Article] Документ N: {text}
      [QA Example] Документ N: {text}
    """
    doc_texts = rec.get("doc_texts", [])
    doc_source_types = rec.get("doc_source_types", [])

    doc_parts = []
    for i, (text, stype) in enumerate(zip(doc_texts, doc_source_types)):
        label = "[QA Example]" if stype == "qa_corpus" else "[Book/Article]"
        doc_parts.append(f"{label} Документ {i+1}: {text}")

    docs_text = "\n\n".join(doc_parts)
    return SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)


def build_pairs(groups: dict) -> list:
    """Build chosen/rejected pairs from grouped runs."""
    all_pairs = []

    for qa_id, runs in groups.items():
        # Bucket runs by label
        by_label = defaultdict(list)
        for run in runs:
            by_label[run["label"]].append(run)

        # Generate candidate pairs with priority
        candidates = []
        for chosen_label, rejected_label in PAIR_PRIORITY:
            priority = PAIR_PRIORITY[(chosen_label, rejected_label)]
            for chosen in by_label.get(chosen_label, []):
                for rejected in by_label.get(rejected_label, []):
                    candidates.append((priority, chosen, rejected))

        # Sort by priority (strongest contrasts first)
        candidates.sort(key=lambda x: x[0])

        # Take up to MAX_PAIRS_PER_QUESTION, avoiding duplicate responses
        seen_pairs = set()
        question_pairs = []
        for priority, chosen, rejected in candidates:
            pair_key = (chosen["run"], rejected["run"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Reconstruct system prompt (same for all runs of same question)
            system_prompt = reconstruct_system_prompt(chosen)

            question_pairs.append({
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chosen["question"]},
                ],
                "chosen": [
                    {"role": "assistant", "content": chosen["response"]},
                ],
                "rejected": [
                    {"role": "assistant", "content": rejected["response"]},
                ],
                "qa_id": qa_id,
                "chosen_label": chosen["label"],
                "rejected_label": rejected["label"],
                "chosen_run": chosen["run"],
                "rejected_run": rejected["run"],
            })

            if len(question_pairs) >= MAX_PAIRS_PER_QUESTION:
                break

        all_pairs.extend(question_pairs)

    return all_pairs


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found")
        return

    # Load and group by qa_id
    records = list(iter_jsonl(INPUT_FILE))
    print(f"Loaded {len(records)} eval records")

    groups = defaultdict(list)
    for rec in records:
        groups[rec["qa_id"]].append(rec)
    print(f"  {len(groups)} unique questions")

    # Count labels
    label_counts = defaultdict(int)
    for rec in records:
        label_counts[rec["label"]] += 1
    print(f"  Labels: {dict(label_counts)}")

    # Questions with mixed labels (can produce pairs)
    mixed = 0
    for qa_id, runs in groups.items():
        labels = set(r["label"] for r in runs)
        if len(labels) > 1:
            mixed += 1
    print(f"  Questions with mixed labels: {mixed}")

    # Build pairs
    pairs = build_pairs(groups)
    print(f"\nBuilt {len(pairs)} DPO pairs")

    # Stats
    pair_types = defaultdict(int)
    for p in pairs:
        pair_types[f"{p['chosen_label']} vs {p['rejected_label']}"] += 1
    for k, v in sorted(pair_types.items()):
        print(f"  {k}: {v}")

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nWritten to {OUTPUT_FILE}")

    # Show example
    if pairs:
        ex = pairs[0]
        print(f"\n--- Example pair ---")
        print(f"QA ID: {ex['qa_id']}")
        print(f"Type: {ex['chosen_label']} vs {ex['rejected_label']}")
        print(f"Question: {ex['prompt'][1]['content'][:100]}...")
        print(f"Chosen (first 200): {ex['chosen'][0]['content'][:200]}...")
        print(f"Rejected (first 200): {ex['rejected'][0]['content'][:200]}...")


if __name__ == "__main__":
    main()
