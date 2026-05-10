"""
Modal deployment matching current test_rag_cards_common.py inference flow.

New flow (matches local testing):
1. Violence vector check (7 anchors, threshold 0.84)
2. DT resolution: violence → force DEPENDENCY_BOUNDARIES + search only boundary QAs
                   otherwise → nearest QA → extract DT
3. Card selection by DT (top 3)
4. QA example selection by DT (top 1)
5. Violence-specific prompt if triggered

Usage:
    modal deploy modal_deploy_v2.py
"""

import modal
import json
import re
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional

# =============================================================================
# MODAL CONFIG
# =============================================================================

APP_NAME = "labkovsky-bot-v2"
GPU_TYPE = "A10G"  # 24GB VRAM

HF_MODEL_ID = "iisrail/vikhr-labkovsky-final"
DATA_VOLUME = "labkovsky-data"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# RAG config (matches test_rag_cards_common.py)
MAX_CARDS = 3
QA_RAG_TOP_K = 1
MAX_SEQ_LENGTH = 3200
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Violence detection
VIOLENCE_THRESHOLD = 0.84
VIOLENCE_DT = "DEPENDENCY_BOUNDARIES"
VIOLENCE_ANCHORS = [
    "муж бьёт жену избивает",
    "партнёр поднимает руку ударяет",
    "распускает руки дерётся",
    "физическое насилие в семье",
    "душит и угрожает убить",
    "избивает когда пьяный",
    "бьёт меня и детей",
]

# Card selection
ALLOW_GLOBAL_FALLBACK = False
LEXICAL_MATCH_BOOST = 0.012
MAX_LEXICAL_BOOST = 0.06

# Topic gate: if the best retrieved card is below this cosine similarity,
# the question is treated as out-of-scope and the model is bypassed.
# Violence-routed questions always pass (safety override).
TOPIC_FLOOR = 0.80
OUT_OF_SCOPE_ANSWER = "Это не ко мне. Я только психолог."

STOP_TERMS = {
    "без", "был", "была", "были", "быть", "вам", "вас", "ваш", "все",
    "для", "его", "если", "еще", "или", "как", "мне", "мой", "над",
    "нас", "нее", "нет", "они", "она", "оно", "при", "про", "сам",
    "себ", "так", "там", "тем", "тех", "тог", "тут", "уже", "что",
    "это", "этот", "your", "with", "that", "this", "from", "have",
    "должен", "есть", "которы", "люди", "челове", "пробле",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reasoning principles.\n\n"
    "Use these principles as the foundation for your response.\n\n"
    "{docs}\n\n"
    "Answer in Labkovsky's style: blunt, confident, with tough love if needed. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
)

SAFETY_INSTRUCTION = (
    "\n\nSafety constraints for the current answer:\n"
    "- If the question includes suicide threats, self-harm, or threats to jump/die, do not joke, dismiss, or encourage it. "
    "Say the user is not responsible for another adult's choice, but the threatened person needs urgent psychiatric/crisis help; advise contacting emergency services or a crisis specialist if risk is immediate.\n"
    "- If the question includes physical violence, hitting, beating, coercion, or danger at home, call it violence directly. "
    "Do not blame the victim, do not excuse it by alcohol, and advise prioritizing safety, support, and legal/emergency help where needed.\n"
    "- If the question involves children, do not suggest aggression or intimidation; focus on the adult regulating themselves and preserving trust.\n"
)

VIOLENCE_RESPONSE_INSTRUCTION = (
    "\n\nAdditional instruction for physical violence cases:\n"
    "- Do not present staying with a violent partner as an acceptable lifestyle choice.\n"
    "- Say directly that a person cannot live safely with someone who hits them.\n"
    "- Treat alcohol as secondary: drunkenness does not cause or excuse violence.\n"
    "- Reframe the user's side as self-worth, fear, dependency, and boundaries: ask why they allow someone to beat them, not how to change the aggressor.\n"
    "- Give directive steps: protect yourself, get support from trusted people, leave or separate safely, and contact legal/emergency help if there is danger.\n"
)

# Create Modal app
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

# Container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.45.0",
        "torch>=2.4.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.44.0",
        "sentence-transformers>=3.0.0",
        "pydantic>=2.0.0",
        "huggingface_hub>=0.25.0",
        "numpy>=1.24.0",
        "fastapi>=0.115.0",
    )
    .run_commands(
        # Download models during image build using hf CLI
        f"hf download {HF_MODEL_ID}",
        f"hf download {EMBEDDING_MODEL}",
    )
    .env({"HF_HUB_OFFLINE": "1"})  # Force offline mode at runtime - use cached models only
)

# =============================================================================
# RAG HELPER FUNCTIONS
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def get_card_dts(card: Dict[str, Any]) -> Set[str]:
    """Get all DTs from a card."""
    dts = set()
    for key in ("dt", "dt_primary", "decision_type", "dt_secondary"):
        value = card.get(key)
        if not value:
            continue
        if isinstance(value, list):
            dts.update(str(v) for v in value if v)
        else:
            dts.add(str(value))
    return dts

def tokenize_for_overlap(text: str) -> Set[str]:
    """Tokenize for lexical matching."""
    terms = set()
    for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower().replace("ё", "е")):
        if len(token) < 4 or token in STOP_TERMS:
            continue
        key = token[:6] if len(token) > 6 else token
        if key not in STOP_TERMS:
            terms.add(key)
    return terms

def lexical_overlap_boost(qa_terms: Set[str], card_text: str) -> tuple:
    """Compute lexical boost."""
    card_terms = tokenize_for_overlap(card_text)
    overlap = qa_terms & card_terms
    boost = min(MAX_LEXICAL_BOOST, LEXICAL_MATCH_BOOST * len(overlap))
    return boost, sorted(overlap)

def select_cards(
    qa_dts: Set[str],
    qa_text: str,
    qa_embedding: np.ndarray,
    cards: List[Dict],
    card_embeddings: np.ndarray,
    max_cards: int = MAX_CARDS,
    allow_global_fallback: bool = ALLOW_GLOBAL_FALLBACK,
) -> List[Dict]:
    """Select top cards by DT + cosine + lexical."""
    if max_cards <= 0:
        return []

    # Filter by DT
    candidate_indices = [
        i for i, card in enumerate(cards)
        if get_card_dts(card) & qa_dts
    ]

    if not candidate_indices and allow_global_fallback:
        candidate_indices = list(range(len(cards)))

    if not candidate_indices:
        return []

    # Score candidates
    qa_terms = tokenize_for_overlap(qa_text)
    scored = []
    for idx in candidate_indices:
        card_text = cards[idx].get("core_principle", "")
        sim = cosine_similarity(qa_embedding, card_embeddings[idx])
        lexical_boost, overlap_terms = lexical_overlap_boost(qa_terms, card_text)
        match_type = "dt_exact" if get_card_dts(cards[idx]) & qa_dts else "global_fallback"
        final_score = sim + lexical_boost
        scored.append((idx, sim, final_score, match_type, lexical_boost, overlap_terms))

    scored.sort(key=lambda x: x[2], reverse=True)

    selected = []
    for idx, sim, final_score, match_type, lexical_boost, overlap_terms in scored[:max_cards]:
        selected.append({
            "card": cards[idx],
            "similarity": sim,
            "selection_score": final_score,
            "lexical_boost": lexical_boost,
            "match_type": match_type,
        })

    return selected

def format_docs(selected_cards: List[Dict]) -> str:
    """Format cards as context."""
    parts = []
    for i, item in enumerate(selected_cards, 1):
        principle = item["card"].get("core_principle", "")
        parts.append(f"Doc {i}: {principle}")
    return "\n\n".join(parts)

def normalize_question(text: str) -> str:
    """Normalize for matching."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def format_qa_examples(examples: List[Dict]) -> str:
    """Format QA examples."""
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
            f"\nQA {i} (id={ex['id']}, sim={ex['similarity']:.3f}):\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}"
        )
    return "\n".join(parts)

# =============================================================================
# VIOLENCE DETECTION
# =============================================================================

def build_violence_anchor_embeddings(embed_model, anchors: List[str] = VIOLENCE_ANCHORS) -> np.ndarray:
    """Embed violence anchors once."""
    return embed_model.encode(
        [f"passage: {anchor}" for anchor in anchors],
        normalize_embeddings=True,
    )

def check_violence(
    question_embedding: np.ndarray,
    anchor_embeddings: np.ndarray,
    threshold: float = VIOLENCE_THRESHOLD,
) -> Tuple[bool, float]:
    """Check if query is violence-related."""
    similarities = anchor_embeddings @ question_embedding
    score = float(np.max(similarities)) if len(similarities) else 0.0
    return score > threshold, score

# =============================================================================
# INFERENCE CLASS
# =============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume},
    scaledown_window=600,  # Keep container alive 10 min after last request
    timeout=900,
    startup_timeout=1200,  # Allow 20 min for cold start model loading
)
class LabkovskyInference:
    """Inference matching test_rag_cards_common.py."""

    @modal.enter()
    def load_models(self):
        """Load all models."""
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import json

        print("[1/4] Loading embedding model...")
        embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL, device=embed_device)
        print(f"  Loaded on {embed_device}!")

        print("[2/4] Loading reasoning cards...")
        self.cards = []
        with open("/data/card_principals_deduped_v2_092.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                card = json.loads(line.strip())
                card["_idx"] = i
                self.cards.append(card)
        print(f"  Loaded {len(self.cards)} cards")

        # Pre-compute card embeddings
        card_texts = [
            "passage: DT: " + ", ".join(sorted(get_card_dts(card))) +
            "\nCORE: " + card.get("core_principle", "")
            for card in self.cards
        ]
        self.card_embeddings = self.embed_model.encode(card_texts, normalize_embeddings=True, batch_size=64)
        print("  Card embeddings computed!")

        print("[3/4] Loading QA records...")
        self.qa_records = []
        with open("/data/qa_rs_segmented.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.qa_records.append(json.loads(line.strip()))
        print(f"  Loaded {len(self.qa_records)} QA records")

        # Index boundary QAs for violence routing
        self.boundary_qa_indices = [
            idx for idx, rec in enumerate(self.qa_records)
            if rec.get("decision_type") == VIOLENCE_DT
        ]
        print(f"  Found {len(self.boundary_qa_indices)} DEPENDENCY_BOUNDARIES QAs")

        # Pre-compute QA embeddings (question-only for DT lookup)
        qa_question_texts = [
            "passage: QUESTION: " + rec.get("question", "")
            for rec in self.qa_records
        ]
        self.qa_question_embeddings = self.embed_model.encode(qa_question_texts, normalize_embeddings=True, batch_size=64)

        # Full QA embeddings for RAG examples
        qa_texts = [
            "passage: DT: " + str(rec.get("decision_type", "")) +
            "\nQUESTION: " + rec.get("question", "") +
            "\nANSWER: " + rec.get("answer", "")
            for rec in self.qa_records
        ]
        self.qa_embeddings = self.embed_model.encode(qa_texts, normalize_embeddings=True, batch_size=64)
        print("  QA embeddings computed!")

        # Violence anchors
        self.violence_anchor_embeddings = build_violence_anchor_embeddings(self.embed_model)
        print("  Violence anchors embedded!")

        print(f"[4/4] Loading model from HuggingFace: {HF_MODEL_ID}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID,
            trust_remote_code=True,
            local_files_only=True,  # Use cached files from image build
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("  Tokenizer loaded!")

        self.model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            local_files_only=True,  # Use cached files from image build
        )
        self.model.eval()
        print("  Model loaded (fp16, no quantization)!")

        print("\n[OK] All models loaded and ready!")

    def resolve_dt_from_nearest_qa(self, question: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        """Resolve DT with violence detection (matches test_rag_cards_common.py)."""
        # Violence check
        is_violence, violence_score = check_violence(query_embedding, self.violence_anchor_embeddings)
        candidate_indices = self.boundary_qa_indices if is_violence else range(len(self.qa_records))

        # Find nearest QA
        best_sim = -1.0
        best_rec = None
        for idx in candidate_indices:
            rec = self.qa_records[idx]
            sim = cosine_similarity(query_embedding, self.qa_question_embeddings[idx])
            if sim > best_sim:
                best_sim = sim
                best_rec = rec

        if best_rec is None:
            return {
                "decision_type": "SELF_ESTEEM_CORRECTIVE",
                "dt_source": "fallback",
                "violence_score": violence_score,
                "is_violence": is_violence,
            }

        # DT assignment
        if is_violence:
            decision_type = VIOLENCE_DT
            source = "violence_vector"
        else:
            decision_type = best_rec.get("decision_type", "SELF_ESTEEM_CORRECTIVE")
            source = "nearest_qa"

        return {
            "decision_type": decision_type,
            "dt_source": source,
            "nearest_qa_id": best_rec.get("id", ""),
            "nearest_qa_similarity": float(best_sim),
            "violence_score": float(violence_score),
            "is_violence": is_violence,
        }

    def select_qa_examples(
        self,
        question: str,
        dt: str,
        exclude_id: str = "",
        exclude_question: str = "",
        top_k: int = QA_RAG_TOP_K,
    ) -> List[Dict]:
        """Select QA examples by DT."""
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
            sim = cosine_similarity(query_embedding, self.qa_embeddings[idx])
            scored.append((sim, rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        examples = []
        for sim, rec in scored[:top_k]:
            examples.append({
                "id": rec.get("id", ""),
                "decision_type": rec.get("decision_type", ""),
                "similarity": float(sim),
                "question": rec.get("question", ""),
                "answer": rec.get("answer", ""),
            })
        return examples

    @modal.method()
    def ask(
        self,
        question: str,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_NEW_TOKENS,
        top_p: float = TOP_P,
        repetition_penalty: float = REPETITION_PENALTY,
    ) -> dict:
        """Answer question with new inference flow."""

        # Step 1: Embed question
        query_embedding = self.embed_model.encode(f"query: {question}", normalize_embeddings=True)

        # Step 2: Resolve DT (with violence detection)
        dt_info = self.resolve_dt_from_nearest_qa(question, query_embedding)
        dt = dt_info["decision_type"]
        is_violence = dt_info["is_violence"]

        # Step 3: Select reasoning cards
        selected_cards = select_cards(
            qa_dts={dt},
            qa_text=question,
            qa_embedding=query_embedding,
            cards=self.cards,
            card_embeddings=self.card_embeddings,
            max_cards=MAX_CARDS,
            allow_global_fallback=ALLOW_GLOBAL_FALLBACK,
        )

        # Step 4: Select QA examples
        qa_examples = self.select_qa_examples(question, dt)

        # Step 4b: Topic gate — bypass model if retrieval is too weak
        top_card_sim = selected_cards[0]["similarity"] if selected_cards else 0.0
        topic_passed = is_violence or (
            len(selected_cards) > 0 and top_card_sim >= TOPIC_FLOOR
        )
        if not topic_passed:
            cards_trace = [
                {
                    "id": f"card_{c['card'].get('_idx')}",
                    "dt": sorted(get_card_dts(c["card"])),
                    "sim": round(c["similarity"], 4),
                    "score": round(c["selection_score"], 4),
                    "lex_boost": round(c["lexical_boost"], 4),
                    "match_type": c["match_type"],
                    "principle_preview": (c["card"].get("core_principle", "") or "")[:160],
                }
                for c in selected_cards
            ]
            qa_example_trace = (
                {
                    "id": qa_examples[0]["id"],
                    "dt": qa_examples[0]["decision_type"],
                    "sim": round(qa_examples[0]["similarity"], 4),
                    "question_preview": (qa_examples[0]["question"] or "")[:160],
                }
                if qa_examples else None
            )
            return {
                "question": question,
                "answer": OUT_OF_SCOPE_ANSWER,
                "decision_type": "OUT_OF_SCOPE",
                "dt_source": "topic_gate",
                "nearest_qa_id": dt_info.get("nearest_qa_id", ""),
                "nearest_qa_similarity": dt_info.get("nearest_qa_similarity"),
                "violence_score": dt_info["violence_score"],
                "is_violence": is_violence,
                "docs_used": len(selected_cards),
                "qa_examples_used": len(qa_examples),
                "cards": cards_trace,
                "qa_example": qa_example_trace,
                "topic_gate": {
                    "passed": False,
                    "top_sim": round(top_card_sim, 4),
                    "threshold": TOPIC_FLOOR,
                },
            }

        # Step 5: Build system prompt
        docs_text = format_docs(selected_cards)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)
        system_prompt += SAFETY_INSTRUCTION

        # Add violence-specific instruction if triggered
        if is_violence:
            system_prompt += VIOLENCE_RESPONSE_INSTRUCTION

        system_prompt += format_qa_examples(qa_examples)

        # Step 6: Format messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Step 7: Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean response tags (both [BRACKETED] and {BRACED})
        answer = re.sub(r"(\[[A-Za-z_]{2,32}\]|\{[A-Za-z_]{2,32}\})\s*", "", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

        # Build per-card trace for RAG inspection
        cards_trace = [
            {
                "id": f"card_{c['card'].get('_idx')}",
                "dt": sorted(get_card_dts(c["card"])),
                "sim": round(c["similarity"], 4),
                "score": round(c["selection_score"], 4),
                "lex_boost": round(c["lexical_boost"], 4),
                "match_type": c["match_type"],
                "principle_preview": (c["card"].get("core_principle", "") or "")[:160],
            }
            for c in selected_cards
        ]
        qa_example_trace = (
            {
                "id": qa_examples[0]["id"],
                "dt": qa_examples[0]["decision_type"],
                "sim": round(qa_examples[0]["similarity"], 4),
                "question_preview": (qa_examples[0]["question"] or "")[:160],
            }
            if qa_examples else None
        )

        return {
            "question": question,
            "answer": answer,
            "decision_type": dt,
            "dt_source": dt_info["dt_source"],
            "nearest_qa_id": dt_info.get("nearest_qa_id", ""),
            "nearest_qa_similarity": dt_info.get("nearest_qa_similarity"),
            "violence_score": dt_info["violence_score"],
            "is_violence": is_violence,
            "docs_used": len(selected_cards),
            "qa_examples_used": len(qa_examples),
            "cards": cards_trace,
            "qa_example": qa_example_trace,
            "topic_gate": {
                "passed": True,
                "top_sim": round(top_card_sim, 4),
                "threshold": TOPIC_FLOOR,
            },
        }

# =============================================================================
# WEB ENDPOINT
# =============================================================================

@app.function(image=image, timeout=900)
@modal.fastapi_endpoint(method="POST")
def ask_labkovsky(
    question: str,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_NEW_TOKENS,
):
    """POST endpoint: returns JSON with the model's answer."""
    if not question:
        return {"error": "No question provided"}

    inference = LabkovskyInference()
    return inference.ask.remote(
        question=question,
        temperature=temperature,
        max_tokens=max_tokens,
    )

@app.function(image=image, timeout=30)
@modal.fastapi_endpoint(method="GET")
def health():
    """GET endpoint: liveness check."""
    return {"status": "ok"}

# =============================================================================
# TEST FUNCTION
# =============================================================================

@app.function(image=image, timeout=900)
def test_query():
    """Test the inference pipeline."""
    inference = LabkovskyInference()

    questions = [
        "Почему я не могу уйти от мужа который меня не уважает?",
        "Муж бьёт меня, но только когда пьяный. Трезвый он хороший. Как его изменить?",  # Violence case
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = inference.ask.remote(q)
        print(f"\nDT: {result['decision_type']} (source: {result['dt_source']})")
        if result['is_violence']:
            print(f"⚠️  VIOLENCE DETECTED (score: {result['violence_score']:.3f})")
        print(f"Docs: {result['docs_used']}, QA examples: {result['qa_examples_used']}")
        print(f"\nA: {result['answer']}")

# Commented out for production deployment - uncomment to run test queries
# @app.local_entrypoint()
# def main():
#     """Run test queries."""
#     test_query.remote()
