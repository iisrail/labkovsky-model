#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared tester for RAG-context LoRA adapters."""

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FINE_TUNING_DIR = PROJECT_ROOT / "data" / "fine_tuning"
DATA_FILE = FINE_TUNING_DIR / "qa_with_rag_context_v4_fixed.jsonl"
CARDS_FILE = FINE_TUNING_DIR / "card_principals_deduped_v2_092.jsonl"
QA_RS_FILE = FINE_TUNING_DIR / "qa_rs_segmented.jsonl"
DT_CLASSIFIER_FILE = MODELS_DIR / "dt-classifier-sklearn" / "dt_classifier.pkl"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_rag_training_data_v4_no_mmr_debug as rag_builder


TEST_QUESTIONS = [
    "Месяц назад я сменила профессию. Новое дело пока не приносит прибыли, хорошо, что есть сбережения. Я теряю мотивацию и думаю, что, возможно, сделала неправильный выбор. Как быть?",
    "Михаил, мне 30 лет, и у меня четыре зависимости. Первое, я трачу больше, чем зарабатываю из-за этого по горлу, залезла в долги и кредиты. Второе, я зависима от алкоголя. Третье, я зависима от человека, он женат. И четвёртое, я зависима от телефона, я постоянно в соцсетях, я всё осознаю, но продолжаю так жить. Как бороться с этими зависимостями и с чего начать?",
    "Михаил, как выгнать бывшего мужа алкоголика из дома насовсем, чтобы не мучила совесть? Ему есть где жить, мне с тремя детьми нет, мы мучаемся от его зависимости.",
    "Мой парень сказал, что если я его брошу, то он спрыгнет с крыши. Не могу грех на душу взять, а парня тоже уже не выдерживаю. Что мне делать?",
    "Михаил, на работе отказали в повышении зарплаты. Уходить не хочу, но чувствую себя виноватой и не такой. Как перестать обесценивать себя из-за отказов?",
    "Я хочу выступать на сцене, рассказывать кейсы по своей сфере интернет-маркетинга, но мне страшно. Я не люблю публичность. Как выйти из зоны комфорта и побороть этот страх?",
    "видел, что моя дочь курит. Ей 15. Сорвался, наорал на неё. И успокоиться не могу, злюсь, как только думаю об этом. Что делать?",
    "Мне не перезвонили после двух собеседований, а в одном проекте коллега просто проигнорировала моё предложение. После каждого такого отказа или игнора неосознанно начинаю в себе сомневаться. Думаю, хороший ли я специалист? Это нормально?",
    "Мне 45, понял что не люблю свою жену, но у нас ипотека и двое детей. Развод разрушит всё. Как жить?",
    "Муж бьёт меня, но только когда пьяный. Трезвый он хороший. Как его изменить?",
]

MAX_SEQ_LENGTH = 3200
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.1
QA_RAG_TOP_K = 1

SAFETY_INSTRUCTION = (
    "\n\nSafety constraints for the current answer:\n"
    "- If the question includes suicide threats, self-harm, or threats to jump/die, do not joke, dismiss, or encourage it. "
    "Say the user is not responsible for another adult's choice, but the threatened person needs urgent psychiatric/crisis help; advise contacting emergency services or a crisis specialist if risk is immediate.\n"
    "- If the question includes physical violence, hitting, beating, coercion, or danger at home, call it violence directly. "
    "Do not blame the victim, do not excuse it by alcohol, and advise prioritizing safety, support, and legal/emergency help where needed.\n"
    "- If the question involves children, do not suggest aggression or intimidation; focus on the adult regulating themselves and preserving trust.\n"
)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records_by_question() -> Dict[str, Dict[str, Any]]:
    records = {}
    for rec in iter_jsonl(DATA_FILE):
        question = rec.get("question")
        if question:
            records[question] = rec
    return records


def normalize_question(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_saved_record(
    question: str,
    records_by_question: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    rec = records_by_question.get(question)
    if rec:
        return rec

    query_norm = normalize_question(question)
    if not query_norm:
        return None

    for saved_question, saved_rec in records_by_question.items():
        saved_norm = normalize_question(saved_question)
        if query_norm == saved_norm:
            return saved_rec
        if len(query_norm) >= 40 and (query_norm in saved_norm or saved_norm in query_norm):
            return saved_rec
    return None


class DynamicContextBuilder:
    def __init__(self):
        print("Loading DT classifier...")
        with open(DT_CLASSIFIER_FILE, "rb") as f:
            model_data = pickle.load(f)
        self.classifier = model_data["classifier"]
        self.id2dt = model_data["id2dt"]
        self.embedding_model_name = model_data["embedding_model_name"]

        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embed_model = SentenceTransformer(self.embedding_model_name)

        print(f"Loading reasoning cards: {CARDS_FILE.name}")
        self.cards = rag_builder.load_cards(CARDS_FILE)
        card_texts = [
            "passage: DT: "
            + ", ".join(sorted(rag_builder.get_card_dts(card)))
            + "\nCORE: "
            + card.get("core_principle", "")
            for card in self.cards
        ]
        self.card_embeddings = self.embed_model.encode(card_texts, normalize_embeddings=True)

        print(f"Loading inference QA RAG: {QA_RS_FILE.name}")
        self.qa_records = list(iter_jsonl(QA_RS_FILE))
        qa_texts = [
            "passage: DT: "
            + str(rec.get("decision_type", ""))
            + "\nQUESTION: "
            + rec.get("question", "")
            + "\nANSWER: "
            + rec.get("answer", "")
            for rec in self.qa_records
        ]
        self.qa_embeddings = self.embed_model.encode(qa_texts, normalize_embeddings=True)

    def predict_dt(self, question: str) -> str:
        embedding = self.embed_model.encode([f"query: {question}"])
        pred_id = self.classifier.predict(embedding)[0]
        return self.id2dt[pred_id]

    def build_record(self, question: str) -> Dict[str, Any]:
        dt = self.predict_dt(question)
        qa_embedding = self.embed_model.encode(f"query: {question}", normalize_embeddings=True)
        selected = rag_builder.select_cards(
            qa_dts={dt},
            qa_text=question,
            qa_embedding=qa_embedding,
            cards=self.cards,
            card_embeddings=self.card_embeddings,
            max_cards=rag_builder.MAX_CARDS,
            allow_global_fallback=rag_builder.ALLOW_GLOBAL_FALLBACK,
        )
        docs_text = rag_builder.format_docs(selected)
        system_prompt = rag_builder.SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)
        return {
            "id": "dynamic",
            "question": question,
            "answer": "",
            "system_prompt": system_prompt,
            "num_docs": len(selected),
            "doc_ids": [f"card_{item['card']['_idx']}" for item in selected],
            "doc_selection_scores": [round(float(item["selection_score"]), 4) for item in selected],
            "doc_lexical_boosts": [round(float(item["lexical_boost"]), 4) for item in selected],
            "decision_type": dt,
            "matching_dts": [dt],
            "context_source": "dynamic",
        }

    def select_qa_examples(
        self,
        question: str,
        dt: str,
        exclude_id: str = "",
        exclude_question: str = "",
        top_k: int = QA_RAG_TOP_K,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embed_model.encode(f"query: {question}", normalize_embeddings=True)
        exclude_question_norm = normalize_question(exclude_question or "")

        scored = []
        for idx, rec in enumerate(self.qa_records):
            if exclude_id and rec.get("id", "") == exclude_id:
                continue
            if exclude_question_norm and normalize_question(rec.get("question", "")) == exclude_question_norm:
                continue
            if dt and rec.get("decision_type") != dt:
                continue
            sim = rag_builder.cosine_similarity(query_embedding, self.qa_embeddings[idx])
            scored.append((sim, rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        examples = []
        for sim, rec in scored[:top_k]:
            examples.append({
                "id": rec.get("id", ""),
                "decision_type": rec.get("decision_type", ""),
                "similarity": round(float(sim), 4),
                "question": rec.get("question", ""),
                "answer": rec.get("answer", ""),
            })
        return examples


def format_qa_examples(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return ""
    parts = [
        "\n\nRelevant previous Q&A examples:",
        "Use these examples only to understand the reasoning pattern and answer structure. "
        "Do not copy sentences or paragraphs verbatim; write a fresh answer for the current question. "
        "Do not transfer case facts from examples: keep only facts stated by the current user, including ages, "
        "number of children, family roles, job details, money, dates, diagnoses, substances, threats, and actions.",
        "Response tags inside examples, such as [EXPLANATION], [ESCALATION], and [INTERVENTION], are signals: "
        "use [ESCALATION] as a cue to include urgent professional/crisis help when relevant, but do not print these tags in the final answer.",
    ]
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"\nQA {i} (id={ex['id']}, sim={ex['similarity']}):\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}"
        )
    return "\n".join(parts)


def strip_response_signal_tags(text: str) -> str:
    text = re.sub(r"\s*\[[A-Za-z_]{2,32}\]\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def load_model(base_model: str, lora_dir_name: str):
    lora_path = MODELS_DIR / lora_dir_name
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA directory not found: {lora_path}")

    print(f"Loading base: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path))
    FastLanguageModel.for_inference(model)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prompt(tokenizer, rec: Dict[str, Any]) -> str:
    system_prompt = rec["system_prompt"] + SAFETY_INSTRUCTION
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rec["question"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean_response(text: str) -> str:
    text = re.sub(r"\[[A-Za-z_]{2,32}\]\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate(model, tokenizer, rec: Dict[str, Any]) -> str:
    prompt = build_prompt(tokenizer, rec)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    return clean_response(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True))


def resolve_record(
    question: str,
    records_by_question: Dict[str, Dict[str, Any]],
    dynamic_builder: Optional[DynamicContextBuilder],
) -> Dict[str, Any]:
    rec = find_saved_record(question, records_by_question)
    if rec:
        rec = dict(rec)
        rec["context_source"] = "saved"
        if dynamic_builder is not None:
            examples = dynamic_builder.select_qa_examples(
                question=rec["question"],
                dt=rec.get("decision_type", ""),
                exclude_id=rec.get("id", ""),
                exclude_question=rec.get("question", ""),
            )
            rec["qa_rag_examples"] = examples
            rec["system_prompt"] = rec["system_prompt"] + format_qa_examples(examples)
        return rec
    if dynamic_builder is None:
        raise ValueError("Question is not in saved RAG data and dynamic context is disabled.")
    rec = dynamic_builder.build_record(question)
    examples = dynamic_builder.select_qa_examples(
        question=question,
        dt=rec.get("decision_type", ""),
    )
    rec["qa_rag_examples"] = examples
    rec["system_prompt"] = rec["system_prompt"] + format_qa_examples(examples)
    return rec


def print_context(rec: Dict[str, Any]) -> None:
    print(f"Context: {rec.get('context_source')}")
    print(f"ID: {rec.get('id')}")
    print(f"DT: {rec.get('decision_type')}")
    print(f"Docs: {rec.get('doc_ids')} scores={rec.get('doc_selection_scores')} lex={rec.get('doc_lexical_boosts')}")
    examples = rec.get("qa_rag_examples") or []
    if examples:
        print("QA RAG:", [(ex["id"], ex["similarity"]) for ex in examples])
    print(f"Question: {rec.get('question')}")


def run_question(model, tokenizer, rec: Dict[str, Any], show_gold: bool) -> None:
    print("\n" + "=" * 100)
    print_context(rec)
    answer = generate(model, tokenizer, rec)
    print("\nMODEL ANSWER:")
    print(answer)
    if show_gold and rec.get("answer"):
        print("\nGOLD ANSWER:")
        print(rec["answer"])


def parse_args(label: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Test {label}")
    parser.add_argument("--question", help="Run one custom question")
    parser.add_argument("--suite", action="store_true", help="Run the built-in 10-question suite")
    parser.add_argument("--interactive", action="store_true", help="Ask questions in a loop")
    parser.add_argument("--saved-only", action="store_true", help="Do not build dynamic context for new questions")
    parser.add_argument("--no-gold", action="store_true", help="Do not print gold answers for saved records")
    return parser.parse_args()


def main(base_model: str, lora_dir_name: str, label: str) -> None:
    args = parse_args(label)
    records_by_question = load_records_by_question()

    needs_dynamic = not args.saved_only
    dynamic_builder = DynamicContextBuilder() if needs_dynamic else None

    print(f"\nTesting: {label}")
    print(f"Data: {DATA_FILE}")
    model, tokenizer = load_model(base_model, lora_dir_name)

    if args.interactive:
        print("\nType a question, or 'exit'.")
        while True:
            question = input("\nquestion> ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                break
            if not question:
                continue
            rec = resolve_record(question, records_by_question, dynamic_builder)
            run_question(model, tokenizer, rec, show_gold=not args.no_gold)
        return

    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        questions = TEST_QUESTIONS

    for question in questions:
        rec = resolve_record(question, records_by_question, dynamic_builder)
        run_question(model, tokenizer, rec, show_gold=not args.no_gold)
