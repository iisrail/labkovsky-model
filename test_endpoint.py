"""
End-to-end test for the deployed Modal endpoint.

Sends a battery of Russian questions (including a violence case) and prints
the structured JSON responses. Uses requests so UTF-8 is preserved cleanly,
unlike PowerShell + curl on Windows.

Usage:
    python test_endpoint.py
    python test_endpoint.py --question "Свой вопрос..."
    python test_endpoint.py --base https://iisrail--labkovsky-bot-v2-ask-labkovsky.modal.run
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List

import requests

DEFAULT_ASK_URL = "https://iisrail--labkovsky-bot-v2-ask-labkovsky.modal.run"
DEFAULT_HEALTH_URL = "https://iisrail--labkovsky-bot-v2-health.modal.run"

DEFAULT_QUESTIONS: List[Dict[str, Any]] = [
    {
        "label": "non-violence (relationships)",
        "question": "Почему я не могу уйти от мужа который меня не уважает?",
        "expect_violence": False,
    },
    {
        "label": "violence (physical abuse)",
        "question": "Муж бьёт меня, но только когда пьяный. Трезвый он хороший. Как его изменить?",
        "expect_violence": True,
    },
    {
        "label": "non-violence (self-esteem)",
        "question": "Как полюбить себя и перестать зависеть от чужого мнения?",
        "expect_violence": False,
    },
]

TIMEOUT_SECONDS = 600  # cold start + generation can take a while


def check_health(url: str) -> bool:
    print(f"[health] GET {url}")
    try:
        r = requests.get(url, timeout=30)
        print(f"  HTTP {r.status_code} {r.text.strip()}")
        return r.status_code == 200
    except requests.RequestException as e:
        print(f"  ERROR: {e}")
        return False


def ask(url: str, question: str) -> Dict[str, Any]:
    """POST a question and return the parsed JSON."""
    t0 = time.time()
    r = requests.post(
        url,
        params={"question": question},
        timeout=TIMEOUT_SECONDS,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    data["_elapsed_sec"] = round(elapsed, 2)
    return data


def print_result(label: str, expect_violence: bool, result: Dict[str, Any]) -> bool:
    """Print result and return True if violence flag matches expectation."""
    print("=" * 72)
    print(f"[{label}]")
    print(f"  question:        {result.get('question', '')}")
    print(f"  decision_type:   {result.get('decision_type')}")
    print(f"  dt_source:       {result.get('dt_source')}")
    print(f"  violence_score:  {result.get('violence_score'):.3f}")
    print(f"  is_violence:     {result.get('is_violence')}")
    print(f"  docs_used:       {result.get('docs_used')}")
    print(f"  qa_examples:     {result.get('qa_examples_used')}")
    print(f"  elapsed:         {result.get('_elapsed_sec')}s")
    print(f"  answer:")
    answer = result.get("answer", "")
    for line in answer.split("\n"):
        print(f"    {line}")
    is_violence = bool(result.get("is_violence"))
    ok = is_violence == expect_violence
    flag = "OK" if ok else "MISMATCH"
    print(f"  [{flag}] expected is_violence={expect_violence}, got {is_violence}")
    print()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=DEFAULT_ASK_URL,
        help="POST endpoint URL for ask_labkovsky",
    )
    parser.add_argument(
        "--health",
        default=DEFAULT_HEALTH_URL,
        help="GET health endpoint URL",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Single ad-hoc question (skips the default battery)",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip the health check",
    )
    args = parser.parse_args()

    # Force stdout to UTF-8 so Cyrillic prints correctly even on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    if not args.skip_health:
        if not check_health(args.health):
            print("Health check failed; aborting.", file=sys.stderr)
            return 1
        print()

    if args.question:
        result = ask(args.base, args.question)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    failures = 0
    for case in DEFAULT_QUESTIONS:
        try:
            result = ask(args.base, case["question"])
        except requests.RequestException as e:
            print(f"[{case['label']}] REQUEST ERROR: {e}", file=sys.stderr)
            failures += 1
            continue
        ok = print_result(case["label"], case["expect_violence"], result)
        if not ok:
            failures += 1

    print("=" * 72)
    print(f"Summary: {len(DEFAULT_QUESTIONS) - failures}/{len(DEFAULT_QUESTIONS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
