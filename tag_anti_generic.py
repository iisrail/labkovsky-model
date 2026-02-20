"""Add RS tags to anti_generic_clean.jsonl answers."""

import json
import re
from pathlib import Path

INPUT = Path("data/fine_tuning/anti_generic_clean.jsonl")
OUTPUT = Path("data/fine_tuning/anti_generic_tagged.jsonl")

# Keywords to detect escalation-worthy topics
SUICIDE_KEYWORDS = [
    r"суицид", r"покончить с собой", r"повесит", r"спрыгн", r"прыгн",
    r"хочет умереть", r"не хочет жить", r"не видит смысла жить",
    r"порез\w* вен", r"режет себ", r"самоповрежден",
    r"сопьётся", r"отравится",
]

VIOLENCE_KEYWORDS = [
    r"бьёт", r"ударит", r"ударить", r"домашне\w* насили",
    r"физическ\w* сил", r"агрессив",
    r"бьёт посуду",
]

SELF_HARM_KEYWORDS = [
    r"режет себе руки", r"самоповрежден", r"порез",
    r"причиняет себе вред",
]

EATING_DISORDER_KEYWORDS = [
    r"отказывается есть", r"анорекси", r"вижу рёбра",
    r"расстройств\w* пищевого поведения",
]


def detect_escalation(question: str, answer: str) -> str | None:
    """Detect if Q&A involves suicide, violence, self-harm, or eating disorders.
    Returns appropriate escalation text or None."""
    combined = (question + " " + answer).lower()

    # Check suicide/self-harm threats
    for kw in SUICIDE_KEYWORDS:
        if re.search(kw, combined):
            # Distinguish between someone threatening vs someone at risk
            if any(re.search(k, combined) for k in [r"подросток", r"сын.*лет", r"дочь.*лет", r"ребён"]):
                return "Требуется срочная консультация подросткового психиатра."
            if any(re.search(k, combined) for k in [r"угрож", r"шантаж", r"манипуляц"]):
                return "При реальной угрозе суицида — вызвать скорую психиатрическую помощь."
            return "Рекомендуется консультация психиатра."

    # Check self-harm
    for kw in SELF_HARM_KEYWORDS:
        if re.search(kw, combined):
            if any(re.search(k, combined) for k in [r"подросток", r"сын", r"дочь"]):
                return "Требуется срочная консультация подросткового психиатра."
            return "Рекомендуется консультация психиатра."

    # Check violence
    for kw in VIOLENCE_KEYWORDS:
        if re.search(kw, combined):
            return "При угрозе физического насилия — обратиться в кризисный центр или полицию."

    # Check eating disorders
    for kw in EATING_DISORDER_KEYWORDS:
        if re.search(kw, combined):
            return "Требуется консультация психиатра, специализирующегося на расстройствах пищевого поведения."

    return None


def tag_answer(question: str, answer: str) -> str:
    """Add [ОБЪЯСНЕНИЕ], [ВМЕШАТЕЛЬСТВО], [ЭСКАЛАЦИЯ] tags to answer."""
    # Split on newlines to find paragraphs
    paragraphs = [p.strip() for p in answer.split("\n") if p.strip()]

    if len(paragraphs) >= 2:
        explanation = "\n".join(paragraphs[:-1])
        intervention = paragraphs[-1]
    else:
        text = paragraphs[0] if paragraphs else answer
        last_split = -1
        for i in range(len(text) - 2, 0, -1):
            if text[i] == "." and i + 2 < len(text) and text[i + 1] == " " and text[i + 2].isupper():
                last_split = i
                break

        if last_split > 0:
            explanation = text[: last_split + 1]
            intervention = text[last_split + 2 :]
        else:
            explanation = text
            intervention = None

    # Detect escalation
    escalation_text = detect_escalation(question, answer)
    escalation = escalation_text if escalation_text else "Не требуется."

    result = f"[ОБЪЯСНЕНИЕ] {explanation}"
    if intervention:
        result += f"\n[ВМЕШАТЕЛЬСТВО] {intervention}"
    result += f"\n[ЭСКАЛАЦИЯ] {escalation}"

    return result


def main():
    records = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    tagged = []
    escalation_count = 0
    escalation_indices = []
    for i, rec in enumerate(records):
        tagged_answer = tag_answer(rec["question"], rec["answer"])
        tagged.append({
            "question": rec["question"],
            "answer": tagged_answer,
        })
        if "[ЭСКАЛАЦИЯ] Не требуется." not in tagged_answer:
            escalation_count += 1
            escalation_indices.append(i)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in tagged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Tagged {len(tagged)} records -> {OUTPUT}")
    print(f"Escalation cases: {escalation_count}\n")

    # Print all escalation examples
    print("=" * 70)
    print("ESCALATION CASES:")
    print("=" * 70)
    for i in escalation_indices:
        # Extract just the escalation line
        lines = tagged[i]["answer"].split("\n")
        esc_line = [l for l in lines if l.startswith("[ЭСКАЛАЦИЯ]")][0]
        print(f"  #{i}: {esc_line}")
        print(f"       Q: {tagged[i]['question'][:80]}...")
        print()

    # Print 3 examples: one normal, one suicide, one violence
    examples = [0]  # normal
    # Find a suicide and violence example
    for i in escalation_indices:
        combined = (records[i]["question"] + " " + records[i]["answer"]).lower()
        if "суицид" in combined or "покончить" in combined or "умереть" in combined:
            examples.append(i)
            break
    for i in escalation_indices:
        combined = (records[i]["question"] + " " + records[i]["answer"]).lower()
        if "бьёт" in combined or "насили" in combined:
            examples.append(i)
            break

    print("\n" + "=" * 70)
    print("FULL EXAMPLES:")
    print("=" * 70)
    for i in examples:
        print(f"\n--- Record #{i} ---")
        print(f"Q: {tagged[i]['question']}")
        print(f"\nA:\n{tagged[i]['answer']}")
        print()


if __name__ == "__main__":
    main()
