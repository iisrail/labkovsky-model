#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Script: Analyze Training Data Quality
Run this to understand your dataset before fine-tuning decisions.
"""

import json
import re
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION - Update these paths to match your setup
# ============================================================

# Auto-detect path relative to script location (like train_lora.py does)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # src/fine_tuning -> src -> project root
# Try common locations
_possible_paths = [
    PROJECT_ROOT / "data" / "fine_tuning" / "qa_rs_corpus_short.jsonl",         # relative to project root
    Path("data/fine_tuning/qa_rs_corpus_short.jsonl"),                          # relative to cwd
]

DATA_PATH = None
for p in _possible_paths:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    DATA_PATH = _possible_paths[0]  # Default, will show error message

# Labkovsky's signature patterns (Russian)
LABKOVSKY_MARKERS = [
    # Signature phrases
    r"вот смотрите",
    r"понимаете",
    r"на самом деле",
    r"дело в том",
    r"проблема в том",
    r"это про то",
    r"это история про",
    r"невротик",
    r"невроз",
    r"тревожность",
    r"самооценка",    
    r"созависим",
    r"шесть правил",
    r"6 правил",
    # Rhetorical patterns
    r"а зачем\?",
    r"а почему\?", 
    r"вы спрашиваете",
    r"когда человек",
    r"если человек",
    # Direct style markers
    r"бросайте",
    r"уходите",
    r"прекратите",
    r"перестаньте",
    r"не надо",
    r"хватит",
]

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def load_data(path: Path) -> list:
    """Load JSONL data, handling missing file gracefully."""
    if not path.exists():
        print(f"❌ File not found: {path}")
        print("   Please update DATA_PATH in the script")
        return []
    
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"   ⚠ Line {line_num}: JSON error - {e}")
    return records


def analyze_basic_stats(records: list) -> dict:
    """Basic dataset statistics."""
    if not records:
        return {}
    
    # Filter out short answers (matching train_lora.py logic)
    full_records = [r for r in records if not r.get("short_answer")]
    short_records = [r for r in records if r.get("short_answer")]
    
    # RS distribution
    rs_counts = Counter(r.get("response_signal", "UNKNOWN") for r in full_records)
    
    # Length statistics
    q_lengths = [len(r["question"].split()) for r in full_records]
    a_lengths = [len(r["answer"].split()) for r in full_records]
    
    return {
        "total_records": len(records),
        "full_records": len(full_records),
        "short_records": len(short_records),
        "rs_distribution": dict(rs_counts),
        "question_length": {
            "mean": sum(q_lengths) / len(q_lengths) if q_lengths else 0,
            "min": min(q_lengths) if q_lengths else 0,
            "max": max(q_lengths) if q_lengths else 0,
        },
        "answer_length": {
            "mean": sum(a_lengths) / len(a_lengths) if a_lengths else 0,
            "min": min(a_lengths) if a_lengths else 0,
            "max": max(a_lengths) if a_lengths else 0,
        },
    }


def analyze_labkovsky_style(records: list) -> dict:
    """Check for Labkovsky's distinctive markers in answers."""
    full_records = [r for r in records if not r.get("short_answer")]
    
    marker_counts = {}
    records_with_markers = 0
    
    for record in full_records:
        answer = record.get("answer", "").lower()
        found_any = False
        
        for marker in LABKOVSKY_MARKERS:
            if re.search(marker, answer, re.IGNORECASE):
                marker_counts[marker] = marker_counts.get(marker, 0) + 1
                found_any = True
        
        if found_any:
            records_with_markers += 1
    
    return {
        "records_with_markers": records_with_markers,
        "percentage_with_markers": 100 * records_with_markers / len(full_records) if full_records else 0,
        "top_markers": sorted(marker_counts.items(), key=lambda x: -x[1])[:10],
    }


def analyze_answer_quality(records: list) -> dict:
    """Analyze answer quality indicators."""
    full_records = [r for r in records if not r.get("short_answer")]
    
    # Check for potential issues
    issues = {
        "very_short": 0,      # < 20 words
        "very_long": 0,       # > 300 words
        "has_urls": 0,        # Contains http/www
        "has_hashtags": 0,    # Contains #
        "repetitive": 0,      # Same phrases repeated
        "generic": 0,         # Lacks Labkovsky markers
    }
    
    problematic_examples = []
    
    for record in full_records:
        answer = record.get("answer", "")
        word_count = len(answer.split())
        
        has_issue = False
        issue_types = []
        
        if word_count < 20:
            issues["very_short"] += 1
            issue_types.append("very_short")
            has_issue = True
        
        if word_count > 300:
            issues["very_long"] += 1
            issue_types.append("very_long")
            has_issue = True
        
        if "http" in answer or "www" in answer:
            issues["has_urls"] += 1
            issue_types.append("has_urls")
            has_issue = True
        
        if "#" in answer:
            issues["has_hashtags"] += 1
            issue_types.append("has_hashtags")
            has_issue = True
        
        # Check for any Labkovsky marker
        has_marker = any(re.search(m, answer, re.IGNORECASE) for m in LABKOVSKY_MARKERS)
        if not has_marker:
            issues["generic"] += 1
        
        if has_issue and len(problematic_examples) < 5:
            problematic_examples.append({
                "issues": issue_types,
                "question": record["question"][:80] + "...",
                "answer_preview": answer[:100] + "...",
                "word_count": word_count,
            })
    
    return {
        "issue_counts": issues,
        "problematic_examples": problematic_examples,
    }


def estimate_training_steps(num_records: int, batch_size: int = 1, 
                           grad_accum: int = 16, epochs: int = 10,
                           val_ratio: float = 0.15) -> dict:
    """Estimate training steps given dataset size."""
    train_size = int(num_records * (1 - val_ratio))
    steps_per_epoch = train_size // (batch_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    
    return {
        "train_size": train_size,
        "val_size": num_records - train_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "recommendation": get_training_recommendation(total_steps, train_size),
    }


def get_training_recommendation(total_steps: int, train_size: int) -> str:
    """Get recommendation based on training steps."""
    if train_size < 200:
        return "⚠️ CRITICAL: Dataset too small (<200). Need more data or data augmentation."
    elif train_size < 350:
        return "⚠️ WARNING: Dataset small (200-350). Consider: more epochs, higher LR, data augmentation."
    elif total_steps < 300:
        return "⚠️ WARNING: Few training steps (<300). Increase epochs or reduce gradient accumulation."
    elif total_steps < 500:
        return "ℹ️ OK: Moderate steps (300-500). Monitor for underfitting."
    else:
        return "✅ GOOD: Sufficient training steps (500+)."


def show_sample_examples(records: list, n: int = 3):
    """Show sample training examples."""
    full_records = [r for r in records if not r.get("short_answer")]
    
    print("\n" + "=" * 70)
    print("📝 SAMPLE TRAINING EXAMPLES")
    print("=" * 70)
    
    # Show one from each RS category if possible
    by_rs = {}
    for r in full_records:
        rs = r.get("response_signal", "UNKNOWN")
        if rs not in by_rs:
            by_rs[rs] = r
    
    for rs, record in list(by_rs.items())[:n]:
        print(f"\n--- [{rs}] ---")
        print(f"Q: {record['question']}")
        print(f"A: {record['answer'][:500]}{'...' if len(record['answer']) > 500 else ''}")
        print(f"   (Word count: {len(record['answer'].split())})")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🔍 LABKOVSKY TRAINING DATA DIAGNOSTIC")
    print("=" * 70)
    print(f"\nData path: {DATA_PATH}")
    
    # Load data
    records = load_data(DATA_PATH)
    if not records:
        return
    
    # Basic stats
    print("\n" + "-" * 50)
    print("📊 BASIC STATISTICS")
    print("-" * 50)
    
    stats = analyze_basic_stats(records)
    print(f"\nTotal records in file:     {stats['total_records']}")
    print(f"Full answers (for training): {stats['full_records']}")
    print(f"Short answers (skipped):     {stats['short_records']}")
    
    print(f"\nRS Distribution:")
    for rs, count in sorted(stats['rs_distribution'].items()):
        pct = 100 * count / stats['full_records']
        print(f"  {rs}: {count} ({pct:.1f}%)")
    
    print(f"\nQuestion length (words): mean={stats['question_length']['mean']:.1f}, "
          f"min={stats['question_length']['min']}, max={stats['question_length']['max']}")
    print(f"Answer length (words):   mean={stats['answer_length']['mean']:.1f}, "
          f"min={stats['answer_length']['min']}, max={stats['answer_length']['max']}")
    
    # Style analysis
    print("\n" + "-" * 50)
    print("🎭 LABKOVSKY STYLE MARKERS")
    print("-" * 50)
    
    style = analyze_labkovsky_style(records)
    print(f"\nRecords with style markers: {style['records_with_markers']} "
          f"({style['percentage_with_markers']:.1f}%)")
    
    print(f"\nTop markers found:")
    for marker, count in style['top_markers']:
        print(f"  '{marker}': {count} occurrences")
    
    # Quality issues
    print("\n" + "-" * 50)
    print("⚠️ POTENTIAL QUALITY ISSUES")
    print("-" * 50)
    
    quality = analyze_answer_quality(records)
    print(f"\nIssue counts:")
    for issue, count in quality['issue_counts'].items():
        if count > 0:
            pct = 100 * count / stats['full_records']
            print(f"  {issue}: {count} ({pct:.1f}%)")
    
    if quality['problematic_examples']:
        print(f"\nProblematic examples:")
        for ex in quality['problematic_examples'][:3]:
            print(f"  - Issues: {ex['issues']}")
            print(f"    Q: {ex['question']}")
            print(f"    A: {ex['answer_preview']}")
            print()
    
    # Training estimation
    print("\n" + "-" * 50)
    print("🏋️ TRAINING ESTIMATION")
    print("-" * 50)
    
    training = estimate_training_steps(stats['full_records'])
    print(f"\nWith current settings (batch=1, grad_accum=16, epochs=10):")
    print(f"  Train samples: {training['train_size']}")
    print(f"  Val samples:   {training['val_size']}")
    print(f"  Steps/epoch:   {training['steps_per_epoch']}")
    print(f"  Total steps:   {training['total_steps']}")
    print(f"\n  {training['recommendation']}")
    
    # Show samples
    show_sample_examples(records)
    
    # Final summary
    print("\n" + "=" * 70)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    issues_found = []
    
    if stats['full_records'] < 300:
        issues_found.append(f"❌ Small dataset: {stats['full_records']} examples (need 300+)")
    
    if training['total_steps'] < 300:
        issues_found.append(f"❌ Few training steps: {training['total_steps']} (need 300+)")
    
    if style['percentage_with_markers'] < 50:
        issues_found.append(f"⚠️ Low style marker coverage: {style['percentage_with_markers']:.1f}%")
    
    generic_pct = 100 * quality['issue_counts']['generic'] / stats['full_records']
    if generic_pct > 30:
        issues_found.append(f"⚠️ Many generic answers: {generic_pct:.1f}% lack Labkovsky markers")
    
    if issues_found:
        print("\nIssues found:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("\n✅ No major issues detected!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()