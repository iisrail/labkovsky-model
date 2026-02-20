"""Replace RS markers with short digit tokens [0], [1], [2] in training data."""

import json
import re
from pathlib import Path

MARKER_MAP = {
    "[ОБЪЯСНЕНИЕ]": "[0]",
    "[ВМЕШАТЕЛЬСТВО]": "[1]",
    "[ЭСКАЛАЦИЯ]": "[2]",
}

# Also catch case variations from the data
PATTERN_MAP = [
    (re.compile(r'\[ОБЪЯСНЕНИЕ\]'), '[0]'),
    (re.compile(r'\[ВМЕШАТЕЛЬСТВО\]'), '[1]'),
    (re.compile(r'\[ЭСКАЛАЦИЯ\]'), '[2]'),
]


def replace_markers(text: str) -> str:
    for pattern, replacement in PATTERN_MAP:
        text = pattern.sub(replacement, text)
    return text


def process_file(input_path: Path, output_path: Path, answer_field: str = "answer"):
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    converted = 0
    for rec in records:
        old = rec.get(answer_field, "")
        new = replace_markers(old)
        if old != new:
            converted += 1
        rec[answer_field] = new

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{input_path.name} -> {output_path.name}: {len(records)} records, {converted} had markers replaced")
    return records


def main():
    ft_dir = Path("data/fine_tuning")

    # qa_rs_final.jsonl -> qa_rs_short.jsonl (answer_segmented field)
    qa_records = []
    with open(ft_dir / "qa_rs_final.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_records.append(json.loads(line))

    converted = 0
    for rec in qa_records:
        old = rec.get("answer_segmented", "")
        new = replace_markers(old)
        if old != new:
            converted += 1
        rec["answer_segmented"] = new

    with open(ft_dir / "qa_rs_short.jsonl", "w", encoding="utf-8") as f:
        for rec in qa_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"qa_rs_final.jsonl -> qa_rs_short.jsonl: {len(qa_records)} records, {converted} had markers replaced")

    # anti_generic_tagged.jsonl -> anti_generic_short.jsonl (answer field)
    process_file(
        ft_dir / "anti_generic_tagged.jsonl",
        ft_dir / "anti_generic_short.jsonl",
        answer_field="answer",
    )

    # Print examples
    print("\n" + "=" * 60)
    print("QA example:")
    print("=" * 60)
    for rec in qa_records[:1]:
        print(f"Q: {rec['question'][:80]}...")
        print(f"A: {rec['answer_segmented'][:300]}...")

    ag_records = []
    with open(ft_dir / "anti_generic_short.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ag_records.append(json.loads(line))

    print("\n" + "=" * 60)
    print("Anti-generic example:")
    print("=" * 60)
    print(f"Q: {ag_records[0]['question'][:80]}...")
    print(f"A: {ag_records[0]['answer'][:300]}...")

    print(f"\n" + "=" * 60)
    print("Anti-generic escalation example:")
    print("=" * 60)
    for rec in ag_records:
        if "[2]" in rec["answer"] and "Не требуется" not in rec["answer"]:
            print(f"Q: {rec['question'][:80]}...")
            print(f"A: {rec['answer'][:300]}...")
            break


if __name__ == "__main__":
    main()
