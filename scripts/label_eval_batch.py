"""Label all entries in eval_batch.jsonl using rule-based heuristics."""
import json
import re
from pathlib import Path

INPUT = Path("/mnt/c/Projects/projects_py/labkovsky-model/data/eval_batch.jsonl")
OUTPUT = Path("/mnt/c/Projects/projects_py/labkovsky-model/data/eval_batch_labeled.jsonl")

# Generic/hedging phrases (lower quality)
GENERIC_PHRASES = [
    "попробуйте",
    "важно понимать",
    "работайте над",
    "обратитесь к специалисту",
    "обратитесь к психологу",
    "возможно,",
    "постарайтесь",
    "рекомендую вам",
    "стоит задуматься",
    "не стесняйтесь",
    "удачи!",
    "удачи вам",
    "берегите себя",
    "всё будет хорошо",
    "помните, что вы",
    "вы заслуживаете",
    "начните с малого",
    "шаг за шагом",
]

# Root cause keywords (Labkovsky style)
ROOT_CAUSE_KEYWORDS = [
    "детств",
    "родител",
    "страх",
    "тревог",
    "самооценк",
    "невроз",
    "травм",
    "воспитан",
    "мам",
    "пап",
    "ребёнк",
    "ребенк",
]

# Strong hedging patterns
HEDGING_PATTERNS = [
    r"может быть[,.]",
    r"скорее всего",
    r"как правило",
    r"в целом",
    r"в принципе",
]


def count_numbered_items(text: str) -> int:
    """Count numbered list items like '1.', '2.', etc."""
    return len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', text))


def label_entry(entry: dict) -> dict:
    response = entry.get("response", "")
    response_len = entry.get("response_len", len(response))
    is_copy = entry.get("is_copy", False)
    max_doc_overlap = entry.get("max_doc_overlap", 0.0)
    ref_overlap = entry.get("ref_overlap", 0.0)
    response_lower = response.lower()

    # --- Copy detection ---
    if is_copy or max_doc_overlap > 0.40:
        entry["label"] = "copy"
        entry["label_reason"] = "auto-detected doc copy"
        entry["auto_label"] = ""
        return entry

    # --- Too short ---
    if response_len < 200:
        entry["label"] = "bad"
        entry["label_reason"] = "response too short"
        entry["auto_label"] = ""
        return entry

    # --- Scoring ---
    score = 0

    # Root cause keywords (max +6)
    root_cause_count = 0
    for kw in ROOT_CAUSE_KEYWORDS:
        if kw in response_lower:
            root_cause_count += 1
    score += min(root_cause_count * 2, 6)

    # Generic phrases penalty
    generic_count = 0
    for phrase in GENERIC_PHRASES:
        if phrase.lower() in response_lower:
            generic_count += 1
    score -= generic_count

    # Hedging patterns penalty
    for pat in HEDGING_PATTERNS:
        if re.search(pat, response_lower):
            score -= 1

    # Length scoring
    if 300 <= response_len <= 800:
        score += 1
    elif response_len > 1200:
        score -= 2
    elif response_len > 1000:
        score -= 1

    # Numbered lists penalty
    num_items = count_numbered_items(response)
    if num_items >= 5:
        score -= 2
    elif num_items >= 8:
        score -= 3

    # Direct/confident tone bonus
    # Check for direct assertions without excessive hedging
    if generic_count == 0 and response_len >= 250:
        score += 2

    # Very low ref_overlap combined with high doc_overlap may indicate
    # the response follows docs but contradicts ref answer
    if ref_overlap < 0.01 and max_doc_overlap > 0.15 and response_len > 300:
        score -= 1

    # --- Determine label ---
    if score >= 4:
        label = "good"
    elif score <= -2:
        label = "bad"
    else:
        label = "ok"

    # --- Build reason ---
    reasons = []
    if root_cause_count >= 2:
        reasons.append("root cause keywords present")
    if generic_count >= 3:
        reasons.append("too many generic phrases")
    elif generic_count >= 1:
        reasons.append("some generic phrases")
    if response_len > 1200:
        reasons.append("too long")
    if num_items >= 5:
        reasons.append("long numbered list")
    if generic_count == 0 and root_cause_count >= 1:
        reasons.append("direct confident tone")
    if response_len < 300:
        reasons.append("somewhat short")

    if not reasons:
        if label == "good":
            reasons.append("strong style and content")
        elif label == "bad":
            reasons.append("weak content or style")
        else:
            reasons.append("adequate but unremarkable")

    entry["label"] = label
    entry["label_reason"] = "; ".join(reasons[:3])
    entry["auto_label"] = ""
    return entry


def main():
    entries = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Read {len(entries)} entries from {INPUT}")

    labeled = [label_entry(e) for e in entries]

    # Stats
    from collections import Counter
    counts = Counter(e["label"] for e in labeled)
    print(f"Label distribution: {dict(counts)}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for e in labeled:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {len(labeled)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
